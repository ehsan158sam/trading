import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    action: int
    confidence: float
    model_predictions: Dict[str, float]
    feature_importance: Dict[str, float]
    uncertainty: float
    timestamp: datetime

class DeepEnsembleModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # Output layer for mean and variance
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, 1)
        self.var_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        var = torch.exp(self.var_head(features))  # Ensure positive variance
        return mean, var

class EnsemblePredictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'deep_ensemble': [
                DeepEnsembleModel(
                    input_dim=config['input_dim'],
                    hidden_dims=config['hidden_dims']
                ) for _ in range(5)  # 5 models in deep ensemble
            ]
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        features = []
        
        # Technical Indicators
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', pd.Series(1, index=close.index))
        
        # Trend Features
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        features.extend([
            (close - sma_20) / sma_20,
            (close - sma_50) / sma_50,
            (close - sma_200) / sma_200,
            (sma_20 - sma_50) / sma_50
        ])
        
        # Volatility Features
        atr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1).rolling(14).mean()
        
        returns = np.log(close / close.shift(1))
        volatility = returns.rolling(20).std()
        
        features.extend([
            atr / close,
            volatility,
            returns.rolling(5).mean(),
            returns.rolling(20).mean()
        ])
        
        # Volume Features
        volume_sma = volume.rolling(20).mean()
        features.extend([
            volume / volume_sma,
            (volume * returns).rolling(20).mean()
        ])
        
        # Momentum Features
        rsi = self._calculate_rsi(close)
        macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        
        features.extend([
            rsi,
            macd,
            macd.rolling(9).mean()
        ])
        
        feature_matrix = np.column_stack([f.fillna(0) for f in features])
        self.feature_names = [
            'trend_20', 'trend_50', 'trend_200', 'ma_cross',
            'atr_ratio', 'volatility', 'ret_5', 'ret_20',
            'volume_ratio', 'volume_price_corr',
            'rsi', 'macd', 'macd_signal'
        ]
        
        return feature_matrix

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train all models in the ensemble"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            
            # Train classical models
            self.models['rf'].fit(X_scaled, y)
            self.models['gb'].fit(X_scaled, y)
            
            # Train deep ensemble
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
            
            for model in self.models['deep_ensemble']:
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                for epoch in range(100):  # Adjust epochs as needed
                    optimizer.zero_grad()
                    mean, var = model(X_tensor)
                    loss = self._negative_log_likelihood(y_tensor, mean, var)
                    loss.backward()
                    optimizer.step()
            
            self.is_trained = True
            logger.info("Ensemble training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during ensemble training: {e}")
            raise

    def _negative_log_likelihood(self, y: torch.Tensor, mean: torch.Tensor, 
                               var: torch.Tensor) -> torch.Tensor:
        """Calculate negative log likelihood loss for deep ensemble"""
        return 0.5 * torch.mean(
            torch.log(var) + (y - mean)**2 / var
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
            
        try:
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from classical models
            rf_pred = self.models['rf'].predict_proba(X_scaled)[-1]
            gb_pred = self.models['gb'].predict_proba(X_scaled)[-1]
            
            # Get predictions from deep ensemble
            X_tensor = torch.FloatTensor(X_scaled)
            deep_preds = []
            deep_vars = []
            
            for model in self.models['deep_ensemble']:
                model.eval()
                with torch.no_grad():
                    mean, var = model(X_tensor)
                    deep_preds.append(mean.numpy())
                    deep_vars.append(var.numpy())
            
            # Combine predictions
            deep_mean = np.mean(deep_preds, axis=0)
            deep_uncertainty = np.mean(deep_vars, axis=0) + np.var(deep_preds, axis=0)
            
            # Weight predictions based on uncertainty
            ensemble_weights = {
                'rf': 0.3,
                'gb': 0.3,
                'deep': 0.4
            }
            
            final_prediction = (
                ensemble_weights['rf'] * rf_pred +
                ensemble_weights['gb'] * gb_pred +
                ensemble_weights['deep'] * deep_mean
            )
            
            # Calculate action and confidence
            action = np.argmax(final_prediction) - 1  # -1 for sell, 0 for hold, 1 for buy
            confidence = np.max(final_prediction)
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_names,
                self.models['rf'].feature_importances_
            ))
            
            return PredictionResult(
                action=int(action),
                confidence=float(confidence),
                model_predictions={
                    'random_forest': float(rf_pred[action + 1]),
                    'gradient_boosting': float(gb_pred[action + 1]),
                    'deep_ensemble': float(deep_mean[action + 1])
                },
                feature_importance=feature_importance,
                uncertainty=float(deep_uncertainty),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error during ensemble prediction: {e}")
            raise

    def save_models(self, path: str):
        """Save all models to disk"""
        try:
            # Save classical models
            joblib.dump(self.models['rf'], f"{path}/rf_model.joblib")
            joblib.dump(self.models['gb'], f"{path}/gb_model.joblib")
            
            # Save deep ensemble models
            for i, model in enumerate(self.models['deep_ensemble']):
                torch.save(model.state_dict(), f"{path}/deep_ensemble_{i}.pt")
            
            # Save scaler
            joblib.dump(self.scaler, f"{path}/scaler.joblib")
            
            logger.info(f"Models saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def load_models(self, path: str):
        """Load all models from disk"""
        try:
            # Load classical models
            self.models['rf'] = joblib.load(f"{path}/rf_model.joblib")
            self.models['gb'] = joblib.load(f"{path}/gb_model.joblib")
            
            # Load deep ensemble models
            for i, model in enumerate(self.models['deep_ensemble']):
                state_dict = torch.load(f"{path}/deep_ensemble_{i}.pt")
                model.load_state_dict(state_dict)
                
            # Load scaler
            self.scaler = joblib.load(f"{path}/scaler.joblib")
            
            self.is_trained = True
            logger.info(f"Models loaded successfully from {path}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise 