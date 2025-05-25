import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import json

class BacktestValidator:
    def __init__(self):
        self.logger = self._setup_logger()
        self.validation_results = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('BacktestValidator')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('debug_logs/backtest_validation.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def validate_system(self, historical_data: Dict[str, pd.DataFrame]):
        """
        اجرای تست‌های جامع روی داده‌های تاریخی
        """
        try:
            self.logger.info("Starting backtest validation...")
            
            # تست استراتژی‌های مختلف
            self._validate_strategies(historical_data)
            
            # تست مدیریت ریسک
            self._validate_risk_management(historical_data)
            
            # تست عملکرد سیستم
            self._validate_performance(historical_data)
            
            # تولید گزارش
            self._generate_validation_report()
            
        except Exception as e:
            self.logger.error(f"Error in backtest validation: {str(e)}")
            
    def _validate_strategies(self, data: Dict[str, pd.DataFrame]):
        """
        تست استراتژی‌های معاملاتی
        """
        strategy_results = {
            'trend_following': self._test_trend_strategy(data),
            'mean_reversion': self._test_mean_reversion(data),
            'breakout': self._test_breakout_strategy(data)
        }
        
        self.validation_results['strategies'] = strategy_results
        
    def _test_trend_strategy(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        تست استراتژی روند
        """
        results = {
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        for symbol, df in data.items():
            # محاسبه میانگین متحرک
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            # شناسایی سیگنال‌ها
            signals = []
            for i in range(len(df)):
                if i < 50:
                    continue
                if df['SMA20'].iloc[i] > df['SMA50'].iloc[i] and \
                   df['SMA20'].iloc[i-1] <= df['SMA50'].iloc[i-1]:
                    signals.append(1)  # خرید
                elif df['SMA20'].iloc[i] < df['SMA50'].iloc[i] and \
                     df['SMA20'].iloc[i-1] >= df['SMA50'].iloc[i-1]:
                    signals.append(-1)  # فروش
                else:
                    signals.append(0)
            
            # محاسبه معیارهای عملکرد
            returns = pd.Series(signals) * df['close'].pct_change()
            
            results['win_rate'] = len(returns[returns > 0]) / len(returns[returns != 0])
            results['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
            results['max_drawdown'] = self._calculate_max_drawdown(returns.cumsum())
            results['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            
        return results
        
    def _test_mean_reversion(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        تست استراتژی بازگشت به میانگین
        """
        results = {
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        for symbol, df in data.items():
            # محاسبه باندهای بولینگر
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['MA20'] + (df['std20'] * 2)
            df['lower_band'] = df['MA20'] - (df['std20'] * 2)
            
            # شناسایی سیگنال‌ها
            signals = []
            for i in range(len(df)):
                if i < 20:
                    continue
                if df['close'].iloc[i] < df['lower_band'].iloc[i]:
                    signals.append(1)  # خرید
                elif df['close'].iloc[i] > df['upper_band'].iloc[i]:
                    signals.append(-1)  # فروش
                else:
                    signals.append(0)
            
            # محاسبه معیارهای عملکرد
            returns = pd.Series(signals) * df['close'].pct_change()
            
            results['win_rate'] = len(returns[returns > 0]) / len(returns[returns != 0])
            results['profit_factor'] = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
            results['max_drawdown'] = self._calculate_max_drawdown(returns.cumsum())
            results['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            
        return results
        
    def _validate_risk_management(self, data: Dict[str, pd.DataFrame]):
        """
        تست سیستم مدیریت ریسک
        """
        risk_results = {
            'position_sizing': self._test_position_sizing(data),
            'stop_loss': self._test_stop_loss(data),
            'portfolio_risk': self._test_portfolio_risk(data)
        }
        
        self.validation_results['risk_management'] = risk_results
        
    def _generate_validation_report(self):
        """
        تولید گزارش جامع از نتایج تست
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'recommendations': []
        }
        
        # تحلیل نتایج و ارائه توصیه‌ها
        strategy_results = self.validation_results.get('strategies', {})
        for strategy, results in strategy_results.items():
            if results['win_rate'] < 0.5:
                report['recommendations'].append(
                    f"Strategy {strategy} shows low win rate ({results['win_rate']:.2f}) - consider optimization"
                )
            if results['max_drawdown'] > 0.2:
                report['recommendations'].append(
                    f"High drawdown in {strategy} strategy ({results['max_drawdown']:.2f}) - review risk management"
                )
        
        # ذخیره گزارش
        with open('debug_logs/validation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info("Validation report generated successfully")
        return report
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        محاسبه حداکثر افت سرمایه
        """
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(drawdowns.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        محاسبه نسبت شارپ
        """
        if len(returns) == 0:
            return 0
        return np.sqrt(252) * returns.mean() / returns.std()

if __name__ == "__main__":
    # نمونه استفاده
    validator = BacktestValidator()
    
    # ساخت داده تست
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'close': np.random.normal(100, 2, 1000),
            'volume': np.random.normal(1000000, 200000, 1000)
        })
    }
    
    validator.validate_system(test_data) 