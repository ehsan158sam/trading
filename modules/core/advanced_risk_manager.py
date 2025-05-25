import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import logging
from datetime import datetime
import json

class AdvancedRiskManager:
    def __init__(self, 
                 max_portfolio_var: float = 0.2,
                 max_asset_correlation: float = 0.7,
                 risk_free_rate: float = 0.02,
                 confidence_level: float = 0.99,
                 lookback_period: int = 252):
        """
        مدیریت ریسک پیشرفته با تحلیل چند فاکتوری
        
        Parameters:
        -----------
        max_portfolio_var : float
            حداکثر واریانس قابل قبول پورتفولیو
        max_asset_correlation : float
            حداکثر همبستگی قابل قبول بین دارایی‌ها
        risk_free_rate : float
            نرخ بدون ریسک سالانه
        confidence_level : float
            سطح اطمینان برای محاسبات ریسک
        lookback_period : int
            دوره زمانی برای محاسبات تاریخی (روز)
        """
        self.logger = self._setup_logger()
        self.max_portfolio_var = max_portfolio_var
        self.max_asset_correlation = max_asset_correlation
        self.risk_free_rate = risk_free_rate
        self.confidence_level = confidence_level
        self.lookback_period = lookback_period
        self.risk_metrics = {}
        
    def _setup_logger(self):
        logger = logging.getLogger('AdvancedRiskManager')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('debug_logs/advanced_risk.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def calculate_portfolio_risk(self, 
                               portfolio_data: Dict[str, pd.DataFrame],
                               weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        محاسبه ریسک پورتفولیو با در نظر گرفتن همبستگی‌ها
        """
        # تبدیل داده‌ها به بازده
        returns_dict = {
            symbol: data['close'].pct_change().dropna()
            for symbol, data in portfolio_data.items()
        }
        
        # ساخت ماتریس بازده
        returns_df = pd.DataFrame(returns_dict)
        
        # اگر وزن‌ها مشخص نشده باشند، وزن یکسان در نظر می‌گیریم
        if weights is None:
            weights = {symbol: 1.0/len(portfolio_data) for symbol in portfolio_data.keys()}
            
        # تبدیل وزن‌ها به آرایه
        weight_array = np.array([weights[symbol] for symbol in returns_df.columns])
        
        # محاسبه ماتریس کوواریانس
        cov_matrix = self._calculate_robust_covariance(returns_df)
        
        # محاسبه ریسک پورتفولیو
        portfolio_var = self._calculate_portfolio_variance(weight_array, cov_matrix)
        portfolio_std = np.sqrt(portfolio_var)
        
        # محاسبه همبستگی‌ها
        correlation_matrix = returns_df.corr()
        
        # محاسبه VaR و CVaR
        var_metrics = self._calculate_var_cvar(returns_df, weight_array)
        
        # محاسبه نسبت‌های ریسک
        risk_ratios = self._calculate_risk_ratios(returns_df, weight_array)
        
        return {
            'portfolio_variance': portfolio_var,
            'portfolio_volatility': portfolio_std * np.sqrt(252),  # سالانه شده
            'correlation_matrix': correlation_matrix.to_dict(),
            'var_metrics': var_metrics,
            'risk_ratios': risk_ratios,
            'risk_decomposition': self._calculate_risk_contribution(weight_array, cov_matrix)
        }
        
    def _calculate_robust_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        محاسبه ماتریس کوواریانس با روش مقاوم
        """
        emp_cov = EmpiricalCovariance(assume_centered=True)
        emp_cov.fit(returns_df)
        return emp_cov.covariance_
        
    def _calculate_portfolio_variance(self, weights: np.ndarray, 
                                    covariance_matrix: np.ndarray) -> float:
        """
        محاسبه واریانس پورتفولیو
        """
        return weights.dot(covariance_matrix).dot(weights)
        
    def _calculate_var_cvar(self, returns_df: pd.DataFrame, 
                           weights: np.ndarray) -> Dict:
        """
        محاسبه Value at Risk و Conditional Value at Risk
        """
        portfolio_returns = returns_df.dot(weights)
        
        # محاسبه VaR پارامتریک
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        z_score = stats.norm.ppf(1 - self.confidence_level)
        parametric_var = -(mean + z_score * std)
        
        # محاسبه VaR تاریخی
        historical_var = -np.percentile(portfolio_returns, 
                                      (1 - self.confidence_level) * 100)
        
        # محاسبه CVaR (Expected Shortfall)
        cvar = -portfolio_returns[portfolio_returns <= -historical_var].mean()
        
        return {
            'parametric_var': parametric_var,
            'historical_var': historical_var,
            'cvar': cvar
        }
        
    def _calculate_risk_ratios(self, returns_df: pd.DataFrame, 
                              weights: np.ndarray) -> Dict:
        """
        محاسبه نسبت‌های مختلف ریسک
        """
        portfolio_returns = returns_df.dot(weights)
        
        # نسبت شارپ
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
        
        # نسبت سورتینو
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        
        # نسبت اطلاعات
        benchmark_returns = returns_df.mean(axis=1)  # میانگین ساده به عنوان بنچمارک
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (portfolio_returns - benchmark_returns).mean() * 252 / tracking_error
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
        
    def _calculate_risk_contribution(self, weights: np.ndarray, 
                                   covariance_matrix: np.ndarray) -> Dict:
        """
        محاسبه سهم هر دارایی در ریسک کل
        """
        portfolio_var = self._calculate_portfolio_variance(weights, covariance_matrix)
        marginal_risk = covariance_matrix.dot(weights)
        component_risk = weights * marginal_risk
        
        return {
            'marginal_risk': marginal_risk.tolist(),
            'component_risk': component_risk.tolist(),
            'risk_contribution': (component_risk / portfolio_var).tolist()
        }
        
    def generate_risk_report(self, portfolio_data: Dict[str, pd.DataFrame],
                            weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        تولید گزارش جامع ریسک
        """
        risk_metrics = self.calculate_portfolio_risk(portfolio_data, weights)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': risk_metrics,
            'warnings': [],
            'recommendations': []
        }
        
        # بررسی هشدارها
        if risk_metrics['portfolio_volatility'] > self.max_portfolio_var:
            report['warnings'].append(
                f"Portfolio volatility ({risk_metrics['portfolio_volatility']:.2%}) "
                f"exceeds maximum threshold ({self.max_portfolio_var:.2%})"
            )
            
        correlation_matrix = pd.DataFrame(risk_metrics['correlation_matrix'])
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > self.max_asset_correlation:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr
                    ))
                    
        if high_corr_pairs:
            report['warnings'].append("High correlation detected between assets:")
            for asset1, asset2, corr in high_corr_pairs:
                report['warnings'].append(
                    f"- {asset1} and {asset2}: {corr:.2f}"
                )
                
        # توصیه‌ها
        if risk_metrics['portfolio_volatility'] > self.max_portfolio_var:
            report['recommendations'].append(
                "Consider reducing position sizes or hedging to lower portfolio volatility"
            )
            
        if high_corr_pairs:
            report['recommendations'].append(
                "Consider diversifying into less correlated assets"
            )
            
        if risk_metrics['risk_ratios']['sharpe_ratio'] < 1:
            report['recommendations'].append(
                "Review strategy performance, risk-adjusted returns are low"
            )
            
        # ذخیره گزارش
        with open('debug_logs/advanced_risk_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info("Advanced risk report generated successfully")
        return report

if __name__ == "__main__":
    # نمونه استفاده
    risk_manager = AdvancedRiskManager()
    
    # داده تست
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'close': np.random.normal(100, 2, 1000),
            'volume': np.random.normal(1000000, 200000, 1000)
        }),
        'ETH/USDT': pd.DataFrame({
            'close': np.random.normal(2000, 40, 1000),
            'volume': np.random.normal(500000, 100000, 1000)
        })
    }
    
    # وزن‌های پورتفولیو
    weights = {
        'BTC/USDT': 0.6,
        'ETH/USDT': 0.4
    }
    
    report = risk_manager.generate_risk_report(test_data, weights) 