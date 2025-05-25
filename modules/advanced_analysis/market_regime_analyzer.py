import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    regime_type: str
    volatility_level: str
    trend_strength: float
    correlation_state: str
    liquidity_score: float
    risk_score: float

class MarketRegimeAnalyzer:
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        self.scaler = StandardScaler()
        self.regime_model = GaussianMixture(n_components=4, random_state=42)
        self.regime_history = []
        
    def analyze_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Analyze current market regime using multiple factors"""
        try:
            # 1. Volatility Analysis
            volatility = self._analyze_volatility(data)
            
            # 2. Trend Analysis
            trend_strength = self._analyze_trend_strength(data)
            
            # 3. Market Microstructure
            liquidity = self._analyze_liquidity(data)
            
            # 4. Correlation Analysis
            correlation_state = self._analyze_correlations(data)
            
            # 5. Risk Analysis
            risk_score = self._calculate_risk_score(volatility, trend_strength, liquidity)
            
            # Determine Overall Regime
            regime_type = self._determine_regime_type(
                volatility['current_level'],
                trend_strength,
                liquidity,
                risk_score
            )
            
            regime = MarketRegime(
                regime_type=regime_type,
                volatility_level=volatility['current_level'],
                trend_strength=trend_strength,
                correlation_state=correlation_state,
                liquidity_score=liquidity,
                risk_score=risk_score
            )
            
            self.regime_history.append(regime)
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
                
            return regime
            
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return MarketRegime(
                regime_type="unknown",
                volatility_level="medium",
                trend_strength=0.0,
                correlation_state="neutral",
                liquidity_score=0.5,
                risk_score=0.5
            )

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, any]:
        """Advanced volatility analysis"""
        returns = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate multiple volatility measures
        current_vol = returns.rolling(self.lookback_periods['short']).std() * np.sqrt(252)
        historical_vol = returns.rolling(self.lookback_periods['long']).std() * np.sqrt(252)
        
        # Volatility regime using Gaussian Mixture Model
        vol_features = np.column_stack([
            current_vol.fillna(0),
            historical_vol.fillna(0)
        ])
        
        if len(vol_features) > self.lookback_periods['short']:
            self.regime_model.fit(vol_features)
            current_regime = self.regime_model.predict(vol_features[-1].reshape(1, -1))[0]
        else:
            current_regime = 0
            
        # Determine volatility level
        vol_percentile = stats.percentileofscore(historical_vol.dropna(), current_vol.iloc[-1])
        if vol_percentile > 80:
            vol_level = "high"
        elif vol_percentile < 20:
            vol_level = "low"
        else:
            vol_level = "medium"
            
        return {
            'current_level': vol_level,
            'current_value': current_vol.iloc[-1],
            'regime': current_regime,
            'percentile': vol_percentile
        }

    def _analyze_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        # ADX for trend strength
        high = data['high']
        low = data['low']
        close = data['close']
        
        # +DM and -DM
        hdiff = high.diff()
        ldiff = low.diff()
        
        pos_dm = (hdiff > 0) & (hdiff > -ldiff) * hdiff
        neg_dm = (ldiff < 0) & (-ldiff > hdiff) * -ldiff
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        
        # Smooth with Wilder's smoothing
        period = 14
        tr_smooth = tr.rolling(period).sum()
        pos_dm_smooth = pos_dm.rolling(period).sum()
        neg_dm_smooth = neg_dm.rolling(period).sum()
        
        # Calculate +DI and -DI
        pos_di = 100 * pos_dm_smooth / tr_smooth
        neg_di = 100 * neg_dm_smooth / tr_smooth
        
        # Calculate ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        # Normalize ADX to 0-1 range
        trend_strength = min(adx.iloc[-1] / 100, 1.0) if not pd.isna(adx.iloc[-1]) else 0.5
        return trend_strength

    def _analyze_liquidity(self, data: pd.DataFrame) -> float:
        """Analyze market liquidity"""
        if 'volume' not in data.columns:
            return 0.5
            
        # Volume analysis
        volume = data['volume']
        avg_volume = volume.rolling(self.lookback_periods['medium']).mean()
        relative_volume = volume / avg_volume
        
        # Spread approximation using high-low range
        spread_proxy = (data['high'] - data['low']) / data['close']
        avg_spread = spread_proxy.rolling(self.lookback_periods['short']).mean()
        
        # Combine metrics
        liquidity_score = (
            0.7 * (relative_volume.iloc[-1] / relative_volume.max()) +
            0.3 * (1 - (avg_spread.iloc[-1] / avg_spread.max()))
        )
        
        return min(max(liquidity_score, 0), 1)

    def _analyze_correlations(self, data: pd.DataFrame) -> str:
        """Analyze market correlations"""
        returns = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate rolling correlations with different timeframes
        short_corr = returns.rolling(self.lookback_periods['short']).corr(
            returns.shift(1)
        )
        long_corr = returns.rolling(self.lookback_periods['long']).corr(
            returns.shift(1)
        )
        
        # Determine correlation regime
        if abs(short_corr.iloc[-1]) > 0.7:
            if short_corr.iloc[-1] > 0:
                return "strong_positive"
            else:
                return "strong_negative"
        elif abs(short_corr.iloc[-1]) < 0.3:
            return "weak"
        else:
            return "moderate"

    def _calculate_risk_score(self, volatility: Dict[str, any], 
                            trend_strength: float, liquidity: float) -> float:
        """Calculate overall risk score"""
        vol_score = volatility['percentile'] / 100
        
        # Higher risk when:
        # - High volatility
        # - Weak trend
        # - Low liquidity
        risk_score = (
            0.4 * vol_score +
            0.3 * (1 - trend_strength) +
            0.3 * (1 - liquidity)
        )
        
        return min(max(risk_score, 0), 1)

    def _determine_regime_type(self, volatility_level: str, trend_strength: float,
                             liquidity: float, risk_score: float) -> str:
        """Determine overall market regime type"""
        if risk_score > 0.7:
            if volatility_level == "high":
                return "crisis"
            else:
                return "high_risk"
        elif risk_score < 0.3:
            if trend_strength > 0.7:
                return "strong_trend"
            else:
                return "low_risk"
        else:
            if trend_strength > 0.6:
                return "trending"
            elif liquidity < 0.4:
                return "choppy"
            else:
                return "ranging"

    def get_regime_transition_probabilities(self) -> pd.DataFrame:
        """Calculate regime transition probabilities from history"""
        if len(self.regime_history) < 2:
            return pd.DataFrame()
            
        transitions = pd.DataFrame([
            (r1.regime_type, r2.regime_type)
            for r1, r2 in zip(self.regime_history[:-1], self.regime_history[1:])
        ], columns=['from', 'to'])
        
        return pd.crosstab(
            transitions['from'], 
            transitions['to'], 
            normalize='index'
        ) 