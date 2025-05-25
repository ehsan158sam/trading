import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import json

class RiskManager:
    def __init__(self, max_position_size: float = 0.1,
                 max_drawdown_limit: float = 0.2,
                 var_confidence: float = 0.95):
        """
        پیاده‌سازی سیستم مدیریت ریسک پیشرفته
        
        Parameters:
        -----------
        max_position_size : float
            حداکثر اندازه پوزیشن به عنوان درصدی از کل سرمایه
        max_drawdown_limit : float
            حداکثر افت سرمایه قابل قبول
        var_confidence : float
            سطح اطمینان برای محاسبه Value at Risk
        """
        self.logger = self._setup_logger()
        self.max_position_size = max_position_size
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        self.risk_metrics = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('RiskManager')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('debug_logs/risk_management.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def calculate_position_size(self, capital: float, 
                              volatility: float,
                              risk_per_trade: float = 0.02) -> float:
        """
        محاسبه اندازه بهینه پوزیشن با استفاده از معیارهای ریسک
        """
        # محاسبه اندازه پوزیشن بر اساس نوسانات و ریسک هر معامله
        position_size = (capital * risk_per_trade) / volatility
        
        # محدود کردن به حداکثر اندازه مجاز
        max_size = capital * self.max_position_size
        return min(position_size, max_size)
        
    def calculate_value_at_risk(self, returns: pd.Series) -> Dict:
        """
        محاسبه Value at Risk با استفاده از روش‌های پارامتریک و تاریخی
        """
        # VaR پارامتریک
        mean = returns.mean()
        std = returns.std()
        z_score = abs(np.percentile(np.random.normal(0, 1, 10000), 
                                  (1 - self.var_confidence) * 100))
        parametric_var = -(mean + z_score * std)
        
        # VaR تاریخی
        historical_var = -np.percentile(returns, (1 - self.var_confidence) * 100)
        
        return {
            'parametric_var': parametric_var,
            'historical_var': historical_var
        }
        
    def calculate_stress_metrics(self, returns: pd.Series) -> Dict:
        """
        محاسبه معیارهای استرس تست
        """
        worst_loss = returns.min()
        worst_drawdown = self._calculate_max_drawdown(returns.cumsum())
        volatility = returns.std() * np.sqrt(252)  # نوسانات سالانه
        
        # محاسبه Expected Shortfall (CVaR)
        var = self.calculate_value_at_risk(returns)
        es = returns[returns < -var['parametric_var']].mean()
        
        return {
            'worst_loss': worst_loss,
            'worst_drawdown': worst_drawdown,
            'volatility': volatility,
            'expected_shortfall': es
        }
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        محاسبه حداکثر افت سرمایه
        """
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(drawdowns.min())
        
    def check_risk_limits(self, current_metrics: Dict) -> List[str]:
        """
        بررسی محدودیت‌های ریسک و صدور هشدار
        """
        warnings = []
        
        if current_metrics['worst_drawdown'] > self.max_drawdown_limit:
            warnings.append(f"Maximum drawdown limit exceeded: {current_metrics['worst_drawdown']:.2%}")
            
        if current_metrics['volatility'] > 0.4:  # 40% نوسانات سالانه
            warnings.append(f"High volatility detected: {current_metrics['volatility']:.2%}")
            
        return warnings
        
    def generate_risk_report(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        تولید گزارش جامع ریسک
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_metrics': {},
            'risk_warnings': [],
            'recommendations': []
        }
        
        for symbol, data in portfolio_data.items():
            returns = data['close'].pct_change().dropna()
            
            # محاسبه معیارهای ریسک
            var = self.calculate_value_at_risk(returns)
            stress_metrics = self.calculate_stress_metrics(returns)
            
            report['portfolio_metrics'][symbol] = {
                'var': var,
                'stress_metrics': stress_metrics
            }
            
            # بررسی هشدارها
            warnings = self.check_risk_limits(stress_metrics)
            if warnings:
                report['risk_warnings'].extend([f"{symbol}: {w}" for w in warnings])
                
            # توصیه‌های مدیریت ریسک
            if stress_metrics['volatility'] > 0.3:
                report['recommendations'].append(
                    f"Consider reducing position size for {symbol} due to high volatility"
                )
            if stress_metrics['worst_drawdown'] > self.max_drawdown_limit * 0.8:
                report['recommendations'].append(
                    f"Review stop-loss levels for {symbol} - approaching maximum drawdown limit"
                )
                
        # ذخیره گزارش
        with open('debug_logs/risk_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info("Risk report generated successfully")
        return report

if __name__ == "__main__":
    # نمونه استفاده
    risk_manager = RiskManager()
    
    # داده تست
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'close': np.random.normal(100, 2, 1000),
            'volume': np.random.normal(1000000, 200000, 1000)
        })
    }
    
    report = risk_manager.generate_risk_report(test_data) 