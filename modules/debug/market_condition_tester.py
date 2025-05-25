import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import json

class MarketConditionTester:
    def __init__(self):
        self.logger = self._setup_logger()
        self.test_results = {}
        self.market_conditions = {
            'trending_up': self._generate_trending_up_data,
            'trending_down': self._generate_trending_down_data,
            'sideways': self._generate_sideways_data,
            'volatile': self._generate_volatile_data,
            'low_volatility': self._generate_low_volatility_data,
            'crisis': self._generate_crisis_data
        }
        
    def _setup_logger(self):
        logger = logging.getLogger('MarketConditionTester')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('debug_logs/market_conditions.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def run_condition_tests(self):
        """
        اجرای تست در شرایط مختلف بازار
        """
        try:
            self.logger.info("Starting market condition tests...")
            
            for condition, data_generator in self.market_conditions.items():
                self.logger.info(f"Testing {condition} market condition...")
                
                # تولید داده تست برای شرایط خاص بازار
                test_data = data_generator()
                
                # اجرای تست استراتژی
                results = self._test_strategy_in_condition(test_data, condition)
                
                self.test_results[condition] = results
                
            # تولید گزارش
            self._generate_condition_report()
            
        except Exception as e:
            self.logger.error(f"Error in market condition testing: {str(e)}")
            
    def _generate_trending_up_data(self) -> pd.DataFrame:
        """
        تولید داده برای بازار صعودی
        """
        periods = 1000
        trend = np.linspace(0, 100, periods)
        noise = np.random.normal(0, 2, periods)
        price = trend + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(1000000, 200000, periods)
        })
        
    def _generate_trending_down_data(self) -> pd.DataFrame:
        """
        تولید داده برای بازار نزولی
        """
        periods = 1000
        trend = np.linspace(100, 0, periods)
        noise = np.random.normal(0, 2, periods)
        price = trend + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(1000000, 200000, periods)
        })
        
    def _generate_sideways_data(self) -> pd.DataFrame:
        """
        تولید داده برای بازار رنج
        """
        periods = 1000
        base_price = 100
        noise = np.random.normal(0, 1, periods)
        price = base_price + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(1000000, 100000, periods)
        })
        
    def _generate_volatile_data(self) -> pd.DataFrame:
        """
        تولید داده برای بازار پر نوسان
        """
        periods = 1000
        base_price = 100
        noise = np.random.normal(0, 5, periods)
        price = base_price + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(2000000, 500000, periods)
        })
        
    def _generate_low_volatility_data(self) -> pd.DataFrame:
        """
        تولید داده برای بازار کم نوسان
        """
        periods = 1000
        base_price = 100
        noise = np.random.normal(0, 0.5, periods)
        price = base_price + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(500000, 50000, periods)
        })
        
    def _generate_crisis_data(self) -> pd.DataFrame:
        """
        تولید داده برای شرایط بحرانی بازار
        """
        periods = 1000
        price = np.zeros(periods)
        
        # شبیه‌سازی سقوط شدید
        normal_period = price[:700]
        crash_period = np.linspace(100, 20, 100)  # سقوط 80 درصدی
        recovery_period = np.linspace(20, 40, 200)  # بازیابی نسبی
        
        price = np.concatenate([normal_period, crash_period, recovery_period])
        noise = np.random.normal(0, 2, periods)
        price = price + noise
        
        return pd.DataFrame({
            'close': price,
            'volume': np.random.normal(3000000, 1000000, periods)
        })
        
    def _test_strategy_in_condition(self, data: pd.DataFrame, condition: str) -> Dict:
        """
        تست استراتژی در شرایط خاص بازار
        """
        results = {
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'sharpe_ratio': 0
        }
        
        # محاسبه شاخص‌های عملکرد
        returns = data['close'].pct_change().dropna()
        
        results['volatility'] = returns.std() * np.sqrt(252)
        results['max_drawdown'] = self._calculate_max_drawdown(data['close'])
        results['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        
        # شبیه‌سازی معاملات
        signals = self._generate_trading_signals(data)
        trade_returns = signals * returns
        
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns < 0]
        
        if len(winning_trades) + len(losing_trades) > 0:
            results['win_rate'] = len(winning_trades) / (len(winning_trades) + len(losing_trades))
        
        if len(losing_trades) > 0 and losing_trades.sum() != 0:
            results['profit_factor'] = abs(winning_trades.sum() / losing_trades.sum())
            
        return results
        
    def _generate_trading_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        تولید سیگنال‌های معاملاتی برای تست
        """
        # استراتژی ساده میانگین متحرک
        short_ma = data['close'].rolling(window=20).mean()
        long_ma = data['close'].rolling(window=50).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        
        return signals
        
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        محاسبه حداکثر افت سرمایه
        """
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        محاسبه نسبت شارپ
        """
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _generate_condition_report(self):
        """
        تولید گزارش جامع از عملکرد سیستم در شرایط مختلف
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'condition_results': self.test_results,
            'analysis': {},
            'recommendations': []
        }
        
        # تحلیل نتایج
        for condition, results in self.test_results.items():
            report['analysis'][condition] = {
                'performance_rating': self._rate_performance(results),
                'risk_rating': self._rate_risk(results)
            }
            
            # توصیه‌های بهبود
            if results['win_rate'] < 0.5:
                report['recommendations'].append(
                    f"Low win rate in {condition} condition - consider adjusting strategy parameters"
                )
            if results['max_drawdown'] > 0.2:
                report['recommendations'].append(
                    f"High drawdown in {condition} condition - review risk management"
                )
                
        # ذخیره گزارش
        with open('debug_logs/market_conditions_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info("Market conditions report generated successfully")
        return report
        
    def _rate_performance(self, results: Dict) -> str:
        """
        رتبه‌بندی عملکرد بر اساس معیارهای مختلف
        """
        score = 0
        
        if results['win_rate'] > 0.6: score += 2
        elif results['win_rate'] > 0.5: score += 1
        
        if results['profit_factor'] > 2: score += 2
        elif results['profit_factor'] > 1.5: score += 1
        
        if results['sharpe_ratio'] > 2: score += 2
        elif results['sharpe_ratio'] > 1: score += 1
        
        if score >= 5: return "Excellent"
        elif score >= 3: return "Good"
        elif score >= 1: return "Fair"
        else: return "Poor"
        
    def _rate_risk(self, results: Dict) -> str:
        """
        رتبه‌بندی ریسک بر اساس معیارهای مختلف
        """
        risk_score = 0
        
        if results['max_drawdown'] < 0.1: risk_score += 2
        elif results['max_drawdown'] < 0.2: risk_score += 1
        
        if results['volatility'] < 0.15: risk_score += 2
        elif results['volatility'] < 0.25: risk_score += 1
        
        if risk_score >= 3: return "Low"
        elif risk_score >= 1: return "Medium"
        else: return "High"

if __name__ == "__main__":
    tester = MarketConditionTester()
    tester.run_condition_tests() 