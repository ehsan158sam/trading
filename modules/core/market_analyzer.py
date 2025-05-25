import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import tensorflow as tf
from datetime import datetime
import logging
import json
from .trading_activity_monitor import TradingActivityMonitor, ActivityStatus

class MarketAnalyzer:
    def __init__(self,
                 lookback_period: int = 100,
                 prediction_horizon: int = 10,
                 confidence_level: float = 0.95,
                 volatility_window: int = 20):
        """
        سیستم تحلیل و پیش‌بینی پیشرفته بازار
        
        Parameters:
        -----------
        lookback_period : int
            دوره زمانی گذشته برای تحلیل
        prediction_horizon : int
            افق پیش‌بینی
        confidence_level : float
            سطح اطمینان برای فاصله‌های پیش‌بینی
        volatility_window : int
            پنجره زمانی برای محاسبه نوسانات
        """
        self.logger = self._setup_logger()
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.confidence_level = confidence_level
        self.volatility_window = volatility_window
        
        # مدل‌ها و مقیاس‌کننده‌ها
        self.models = self._initialize_models()
        self.scalers = {}
        
        # وضعیت بازار
        self.market_state = {}
        
        # وزن استراتژی‌ها
        self.strategy_weights = self._initialize_strategy_weights()
        
        # اضافه کردن مانیتور فعالیت
        self.activity_monitor = TradingActivityMonitor()
        
    def _setup_logger(self):
        logger = logging.getLogger('MarketAnalyzer')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('logs/market_analysis.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        تحلیل جامع بازار
        """
        analysis_results = {}
        
        for symbol, data in market_data.items():
            self.logger.info(f"Analyzing {symbol}")
            
            # تحلیل تکنیکال
            technical_indicators = self._calculate_technical_indicators(data)
            
            # تحلیل آماری
            statistical_analysis = self._perform_statistical_analysis(data)
            
            # تشخیص الگوها
            patterns = self._detect_patterns(data)
            
            # پیش‌بینی
            predictions = self._generate_predictions(data)
            
            # تحلیل نوسانات
            volatility_analysis = self._analyze_volatility(data)
            
            # تشخیص ناهنجاری
            anomalies = self._detect_anomalies(data)
            
            # بررسی وضعیت فعالیت معاملاتی
            activity_status = self._check_trading_activity(data)
            
            analysis_results[symbol] = {
                'technical_analysis': technical_indicators,
                'statistical_analysis': statistical_analysis,
                'patterns': patterns,
                'predictions': predictions,
                'volatility_analysis': volatility_analysis,
                'anomalies': anomalies,
                'market_state': self._determine_market_state(data),
                'activity_status': activity_status
            }
            
            # اعمال تنظیمات پیشنهادی در صورت نیاز
            if activity_status['status'] != 'NORMAL':
                self._apply_activity_adjustments(activity_status['adjustments'])
            
        return analysis_results
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """
        محاسبه شاخص‌های تکنیکال
        """
        close = data['close']
        volume = data['volume']
        
        # میانگین‌های متحرک
        ma_periods = [5, 10, 20, 50, 100]
        mas = {f'MA_{period}': close.rolling(period).mean() 
               for period in ma_periods}
        
        # شاخص قدرت نسبی
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # میانگین متحرک همگرا-واگرا
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # باندهای بولینگر
        bb_period = 20
        bb_std = 2
        bb_middle = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        bb_upper = bb_middle + bb_std * bb_std_dev
        bb_lower = bb_middle - bb_std * bb_std_dev
        
        # شاخص جریان پول
        typical_price = (data['high'] + data['low'] + close) / 3
        money_flow = typical_price * volume
        mfi = (money_flow.rolling(14).sum() / volume.rolling(14).sum()) * 100
        
        return {
            'moving_averages': {k: v.tolist() for k, v in mas.items()},
            'rsi': rsi.tolist(),
            'macd': {
                'macd': macd.tolist(),
                'signal': signal.tolist(),
                'histogram': (macd - signal).tolist()
            },
            'bollinger_bands': {
                'upper': bb_upper.tolist(),
                'middle': bb_middle.tolist(),
                'lower': bb_lower.tolist()
            },
            'mfi': mfi.tolist()
        }
        
    def _perform_statistical_analysis(self, data: pd.DataFrame) -> Dict:
        """
        تحلیل آماری داده‌ها
        """
        returns = data['close'].pct_change().dropna()
        
        # آزمون نرمال بودن
        normality_test = stats.normaltest(returns)
        
        # آزمون مانایی
        adf_test = adfuller(returns)
        
        # آزمون خودهمبستگی
        ljung_box_test = acorr_ljungbox(returns)
        
        # آمار توصیفی
        descriptive_stats = returns.describe()
        
        # چولگی و کشیدگی
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'normality_test': {
                'statistic': float(normality_test[0]),
                'p_value': float(normality_test[1])
            },
            'stationarity_test': {
                'adf_statistic': float(adf_test[0]),
                'p_value': float(adf_test[1]),
                'critical_values': {str(k): v for k, v in adf_test[4].items()}
            },
            'autocorrelation_test': {
                'lb_statistic': float(ljung_box_test[0][0]),
                'p_value': float(ljung_box_test[1][0])
            },
            'descriptive_statistics': {
                'mean': float(descriptive_stats['mean']),
                'std': float(descriptive_stats['std']),
                'min': float(descriptive_stats['min']),
                'max': float(descriptive_stats['max']),
                'quartiles': {
                    '25%': float(descriptive_stats['25%']),
                    '50%': float(descriptive_stats['50%']),
                    '75%': float(descriptive_stats['75%'])
                }
            },
            'higher_moments': {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis)
            }
        }
        
    def _detect_patterns(self, data: pd.DataFrame) -> Dict:
        """
        تشخیص الگوهای قیمتی
        """
        patterns = {}
        
        # الگوی دو قله
        patterns['double_top'] = self._detect_double_top(data)
        
        # الگوی دو دره
        patterns['double_bottom'] = self._detect_double_bottom(data)
        
        # الگوی سر و شانه
        patterns['head_and_shoulders'] = self._detect_head_and_shoulders(data)
        
        # الگوی مثلث
        patterns['triangle'] = self._detect_triangle(data)
        
        return patterns
        
    def _generate_predictions(self, data: pd.DataFrame) -> Dict:
        """
        تولید پیش‌بینی‌ها با استفاده از مدل‌های مختلف
        """
        # آماده‌سازی داده
        X, y = self._prepare_data_for_prediction(data)
        
        # تقسیم داده
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # مقیاس‌بندی داده
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # مدل جنگل تصادفی
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_predictions = rf_model.predict(X_test_scaled)
        
        # مدل شبکه عصبی
        nn_model = self._build_neural_network(X_train.shape[1])
        nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
        nn_predictions = nn_model.predict(X_test_scaled)
        
        return {
            'random_forest': {
                'predictions': rf_predictions.tolist(),
                'confidence_intervals': self._calculate_prediction_intervals(rf_model, X_test_scaled)
            },
            'neural_network': {
                'predictions': nn_predictions.flatten().tolist(),
                'uncertainty': self._estimate_prediction_uncertainty(nn_model, X_test_scaled)
            },
            'ensemble': {
                'predictions': ((rf_predictions + nn_predictions.flatten()) / 2).tolist()
            }
        }
        
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """
        تحلیل نوسانات
        """
        returns = data['close'].pct_change().dropna()
        
        # نوسانات تاریخی
        historical_volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # تجزیه نوسانات
        volatility_components = self._decompose_volatility(returns)
        
        # پیش‌بینی نوسانات
        volatility_forecast = self._forecast_volatility(returns)
        
        return {
            'historical': historical_volatility.tolist(),
            'components': volatility_components,
            'forecast': volatility_forecast
        }
        
    def _detect_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        تشخیص ناهنجاری‌ها
        """
        # آماده‌سازی ویژگی‌ها
        features = self._prepare_anomaly_features(data)
        
        # مدل جنگل ایزوله
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(features)
        
        # محاسبه نمرات ناهنجاری
        statistical_scores = self._calculate_statistical_anomalies(data)
        
        return {
            'isolation_forest': {
                'scores': anomaly_scores.tolist(),
                'anomaly_indices': np.where(anomaly_scores == -1)[0].tolist()
            },
            'statistical': statistical_scores
        }
        
    def _determine_market_state(self, data: pd.DataFrame) -> Dict:
        """
        تعیین وضعیت بازار
        """
        returns = data['close'].pct_change().dropna()
        volume = data['volume']
        
        # روند
        trend = self._calculate_trend(data)
        
        # نوسانات
        volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # نقدشوندگی
        liquidity = self._calculate_liquidity(volume)
        
        # فاز بازار
        market_phase = self._identify_market_phase(data)
        
        return {
            'trend': trend,
            'volatility_regime': {
                'current': float(volatility.iloc[-1]),
                'regime': 'high' if volatility.iloc[-1] > volatility.mean() + volatility.std() else 'normal'
            },
            'liquidity_state': liquidity,
            'market_phase': market_phase
        }
        
    def _prepare_data_for_prediction(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        آماده‌سازی داده برای پیش‌بینی
        """
        # ساخت ویژگی‌ها
        features = []
        target = []
        
        close = data['close'].values
        for i in range(self.lookback_period, len(close) - self.prediction_horizon):
            features.append(close[i-self.lookback_period:i])
            target.append(close[i:i+self.prediction_horizon])
            
        return np.array(features), np.array(target)
        
    def _build_neural_network(self, input_dim: int) -> tf.keras.Model:
        """
        ساخت مدل شبکه عصبی
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def _calculate_prediction_intervals(self, model: RandomForestRegressor,
                                     X: np.ndarray) -> Dict:
        """
        محاسبه فاصله‌های اطمینان پیش‌بینی
        """
        predictions = []
        for estimator in model.estimators_:
            predictions.append(estimator.predict(X))
            
        predictions = np.array(predictions)
        
        lower = np.percentile(predictions, ((1 - self.confidence_level) / 2) * 100, axis=0)
        upper = np.percentile(predictions, (1 - (1 - self.confidence_level) / 2) * 100, axis=0)
        
        return {
            'lower': lower.tolist(),
            'upper': upper.tolist()
        }
        
    def _estimate_prediction_uncertainty(self, model: tf.keras.Model,
                                      X: np.ndarray) -> Dict:
        """
        تخمین عدم قطعیت پیش‌بینی‌های شبکه عصبی
        """
        # استفاده از Monte Carlo Dropout
        predictions = []
        for _ in range(100):
            predictions.append(model(X, training=True))
            
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return {
            'mean': mean_pred.tolist(),
            'std': std_pred.tolist(),
            'confidence_intervals': {
                'lower': (mean_pred - 1.96 * std_pred).tolist(),
                'upper': (mean_pred + 1.96 * std_pred).tolist()
            }
        }
        
    def _decompose_volatility(self, returns: pd.Series) -> Dict:
        """
        تجزیه نوسانات به اجزای مختلف
        """
        # نوسانات پایه
        baseline_vol = returns.std() * np.sqrt(252)
        
        # نوسانات شرطی
        conditional_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # نوسانات جهت‌دار
        directional_vol = (returns[returns > 0].std() - returns[returns < 0].std()) * np.sqrt(252)
        
        return {
            'baseline': float(baseline_vol),
            'conditional': conditional_vol.tolist(),
            'directional': float(directional_vol)
        }
        
    def _forecast_volatility(self, returns: pd.Series) -> Dict:
        """
        پیش‌بینی نوسانات
        """
        # محاسبه نوسانات تاریخی
        hist_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        
        # پیش‌بینی ساده با میانگین متحرک نمایی
        forecast = hist_vol.ewm(span=20).mean()
        
        return {
            'point_forecast': forecast.iloc[-1],
            'trend': 'increasing' if forecast.iloc[-1] > forecast.iloc[-2] else 'decreasing',
            'confidence_intervals': {
                'lower': float(forecast.iloc[-1] - forecast.std()),
                'upper': float(forecast.iloc[-1] + forecast.std())
            }
        }
        
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        آماده‌سازی ویژگی‌ها برای تشخیص ناهنجاری
        """
        features = []
        
        # تغییرات قیمت
        price_changes = data['close'].pct_change()
        
        # تغییرات حجم
        volume_changes = data['volume'].pct_change()
        
        # نوسانات
        volatility = price_changes.rolling(self.volatility_window).std()
        
        features = np.column_stack([
            price_changes,
            volume_changes,
            volatility
        ])
        
        return features
        
    def _calculate_statistical_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        محاسبه ناهنجاری‌های آماری
        """
        returns = data['close'].pct_change().dropna()
        volume = data['volume']
        
        # ناهنجاری‌های قیمتی
        price_mean = returns.mean()
        price_std = returns.std()
        price_anomalies = returns[abs(returns - price_mean) > 3 * price_std]
        
        # ناهنجاری‌های حجمی
        volume_mean = volume.mean()
        volume_std = volume.std()
        volume_anomalies = volume[abs(volume - volume_mean) > 3 * volume_std]
        
        return {
            'price': {
                'indices': price_anomalies.index.tolist(),
                'values': price_anomalies.tolist()
            },
            'volume': {
                'indices': volume_anomalies.index.tolist(),
                'values': volume_anomalies.tolist()
            }
        }
        
    def _calculate_trend(self, data: pd.DataFrame) -> Dict:
        """
        محاسبه روند
        """
        close = data['close']
        
        # روند خطی
        x = np.arange(len(close))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
        
        # قدرت روند
        trend_strength = abs(r_value)
        
        # جهت روند
        if slope > 0:
            direction = 'upward'
        elif slope < 0:
            direction = 'downward'
        else:
            direction = 'sideways'
            
        return {
            'direction': direction,
            'strength': float(trend_strength),
            'slope': float(slope),
            'r_squared': float(r_value ** 2)
        }
        
    def _calculate_liquidity(self, volume: pd.Series) -> Dict:
        """
        محاسبه وضعیت نقدشوندگی
        """
        current_volume = volume.iloc[-1]
        avg_volume = volume.mean()
        vol_ratio = current_volume / avg_volume
        
        if vol_ratio > 1.5:
            state = 'high'
        elif vol_ratio < 0.5:
            state = 'low'
        else:
            state = 'normal'
            
        return {
            'state': state,
            'volume_ratio': float(vol_ratio),
            'current_volume': float(current_volume),
            'average_volume': float(avg_volume)
        }
        
    def _identify_market_phase(self, data: pd.DataFrame) -> str:
        """
        شناسایی فاز بازار
        """
        close = data['close']
        returns = close.pct_change().dropna()
        volatility = returns.rolling(self.volatility_window).std()
        
        # محاسبه روند
        trend = self._calculate_trend(data)
        
        # محاسبه نوسانات نسبی
        relative_volatility = volatility.iloc[-1] / volatility.mean()
        
        if trend['direction'] == 'upward' and relative_volatility < 1.2:
            return 'accumulation'
        elif trend['direction'] == 'upward' and relative_volatility >= 1.2:
            return 'markup'
        elif trend['direction'] == 'downward' and relative_volatility < 1.2:
            return 'distribution'
        else:
            return 'markdown'
        
    def _check_trading_activity(self, market_data: pd.DataFrame) -> Dict:
        """
        بررسی وضعیت فعالیت معاملاتی
        """
        # تبدیل داده‌های بازار به فرمت مورد نیاز مانیتور
        market_data_dict = {
            'volume': market_data['volume'].values
        }
        
        # ساخت تاریخچه معاملات مصنوعی برای تست
        recent_trades = self._generate_sample_trades(market_data)
        
        # دریافت استراتژی‌های فعال
        active_strategies = self._get_active_strategies()
        
        return self.activity_monitor.monitor_activity(
            recent_trades,
            market_data_dict,
            active_strategies
        )
        
    def _generate_sample_trades(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        تولید نمونه معاملات برای تست مانیتور
        """
        trades = []
        for i in range(min(100, len(market_data))):
            trades.append({
                'timestamp': market_data.index[i],
                'profit': market_data['close'].pct_change().iloc[i] * 100,
                'volume': market_data['volume'].iloc[i]
            })
        return trades
        
    def _get_active_strategies(self) -> List[str]:
        """
        دریافت لیست استراتژی‌های فعال
        """
        return list(self.models.keys())
        
    def _apply_activity_adjustments(self, adjustments: Dict):
        """
        اعمال تنظیمات پیشنهادی مانیتور فعالیت
        """
        if 'risk_params' in adjustments:
            self._adjust_risk_parameters(adjustments['risk_params'])
            
        if 'trading_params' in adjustments:
            self._adjust_trading_parameters(adjustments['trading_params'])
            
        if 'strategy_params' in adjustments:
            self._adjust_strategy_parameters(adjustments['strategy_params'])
            
        self.logger.info(f"Applied activity adjustments: {adjustments}")
        
    def _adjust_risk_parameters(self, params: Dict):
        """
        تنظیم پارامترهای ریسک
        """
        for param, change in params.items():
            if isinstance(change, str) and (change.endswith('%')):
                value = float(change.rstrip('%')) / 100
                if param == 'confidence_level':
                    self.confidence_level = min(0.99, self.confidence_level * (1 + value))
                elif param == 'volatility_window':
                    self.volatility_window = int(self.volatility_window * (1 + value))
                    
    def _adjust_trading_parameters(self, params: Dict):
        """
        تنظیم پارامترهای معاملاتی
        """
        for param, change in params.items():
            if isinstance(change, str) and (change.endswith('%')):
                value = float(change.rstrip('%')) / 100
                if param == 'lookback_period':
                    self.lookback_period = int(self.lookback_period * (1 + value))
                elif param == 'prediction_horizon':
                    self.prediction_horizon = int(self.prediction_horizon * (1 + value))
                    
    def _adjust_strategy_parameters(self, params: Dict):
        """
        تنظیم پارامترهای استراتژی
        """
        if 'active_strategies' in params:
            change = params['active_strategies']
            if isinstance(change, str) and change.startswith('+'):
                num_new = int(change[1:])
                self._activate_new_strategies(num_new)
                
        if params.get('strategy_weights') == 'rebalance':
            self._rebalance_strategy_weights()
            
    def _activate_new_strategies(self, num_new: int):
        """
        فعال‌سازی استراتژی‌های جدید
        """
        available_strategies = [
            'momentum', 'mean_reversion', 'breakout', 
            'trend_following', 'statistical_arbitrage',
            'volatility_trading', 'pattern_recognition'
        ]
        
        current_strategies = set(self.models.keys())
        new_strategies = set(available_strategies) - current_strategies
        
        for strategy in list(new_strategies)[:num_new]:
            self.models[strategy] = self._initialize_strategy(strategy)
            
    def _initialize_strategy(self, strategy_name: str) -> object:
        """
        مقداردهی اولیه استراتژی جدید
        """
        if strategy_name == 'momentum':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif strategy_name == 'mean_reversion':
            return IsolationForest(contamination=0.1, random_state=42)
        elif strategy_name == 'breakout':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif strategy_name == 'trend_following':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif strategy_name == 'statistical_arbitrage':
            return IsolationForest(contamination=0.1, random_state=42)
        elif strategy_name == 'volatility_trading':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif strategy_name == 'pattern_recognition':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}")
            return None
        
    def _rebalance_strategy_weights(self):
        """
        تنظیم مجدد وزن استراتژی‌ها
        """
        num_strategies = len(self.models)
        if num_strategies > 0:
            base_weight = 1.0 / num_strategies
            self.strategy_weights = {
                strategy: base_weight for strategy in self.models.keys()
            }

    def _initialize_models(self) -> Dict:
        """
        مقداردهی اولیه مدل‌ها
        """
        models = {}
        available_strategies = [
            'momentum', 'mean_reversion', 'breakout', 
            'trend_following', 'statistical_arbitrage',
            'volatility_trading', 'pattern_recognition'
        ]
        
        for strategy in available_strategies[:3]:  # شروع با 3 استراتژی اولیه
            models[strategy] = self._initialize_strategy(strategy)
            
        return models
        
    def _initialize_strategy_weights(self) -> Dict:
        """
        مقداردهی اولیه وزن استراتژی‌ها
        """
        weights = {}
        active_strategies = list(self.models.keys())
        base_weight = 1.0 / len(active_strategies)
        
        for strategy in active_strategies:
            weights[strategy] = base_weight
            
        return weights

if __name__ == "__main__":
    # نمونه استفاده
    analyzer = MarketAnalyzer()
    
    # داده تست
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'close': np.random.normal(100, 2, 1000),
            'high': np.random.normal(101, 2, 1000),
            'low': np.random.normal(99, 2, 1000),
            'volume': np.random.normal(1000000, 200000, 1000)
        })
    }
    
    analysis = analyzer.analyze_market(test_data) 