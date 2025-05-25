import unittest
import pandas as pd
import numpy as np
from backtest_validator import BacktestValidator
from market_condition_tester import MarketConditionTester

class TestTradingSystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.backtest_validator = BacktestValidator()
        self.market_tester = MarketConditionTester()
        
        # Create sample test data
        self.test_data = {
            'BTC/USDT': pd.DataFrame({
                'close': np.array([100, 101, 102, 101, 99, 98, 99, 100, 102, 103]),
                'volume': np.array([1000] * 10)
            })
        }

    def test_trend_strategy(self):
        """Test trend following strategy performance"""
        results = self.backtest_validator._test_trend_strategy(self.test_data)
        self.assertIsInstance(results, dict)
        self.assertTrue(0 <= results['win_rate'] <= 1)
        self.assertTrue(results['profit_factor'] >= 0)
        
    def test_mean_reversion(self):
        """Test mean reversion strategy performance"""
        results = self.backtest_validator._test_mean_reversion(self.test_data)
        self.assertIsInstance(results, dict)
        self.assertTrue(0 <= results['win_rate'] <= 1)
        
    def test_market_conditions(self):
        """Test market condition generation"""
        conditions = [
            'trending_up',
            'trending_down',
            'sideways',
            'volatile',
            'low_volatility',
            'crisis'
        ]
        
        for condition in conditions:
            data = getattr(self.market_tester, f'_generate_{condition}_data')()
            self.assertIsInstance(data, pd.DataFrame)
            self.assertTrue('close' in data.columns)
            self.assertTrue('volume' in data.columns)
            
    def test_risk_metrics(self):
        """Test risk management metrics calculation"""
        prices = pd.Series([100, 95, 90, 85, 80, 85, 90, 95, 100])
        max_drawdown = self.backtest_validator._calculate_max_drawdown(prices)
        self.assertTrue(0 <= max_drawdown <= 1)
        
        returns = prices.pct_change().dropna()
        sharpe = self.backtest_validator._calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

if __name__ == '__main__':
    unittest.main() 