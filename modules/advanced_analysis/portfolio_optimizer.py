import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from scipy.optimize import minimize
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PortfolioAllocation:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: pd.DataFrame
    timestamp: datetime

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.historical_allocations = []
        self.market_regimes = {}
        
    def optimize_portfolio(self, 
                         returns: pd.DataFrame,
                         constraints: Dict[str, Any],
                         market_regime: str = "normal") -> PortfolioAllocation:
        """Optimize portfolio weights using advanced techniques"""
        try:
            # Calculate key metrics
            mean_returns = returns.mean() * 252  # Annualized returns
            cov_matrix = returns.cov() * 252     # Annualized covariance
            correlation = returns.corr()
            
            # Initial weights
            n_assets = len(returns.columns)
            init_weights = np.array([1/n_assets] * n_assets)
            
            # Define optimization constraints
            bounds = [(0, constraints.get('max_weight', 1.0)) for _ in range(n_assets)]
            
            # Adjust optimization based on market regime
            if market_regime == "crisis":
                # More conservative during crisis
                risk_weight = 0.8
                min_weight = 0.05
            elif market_regime == "trending":
                # More aggressive during trends
                risk_weight = 0.4
                min_weight = 0.0
            else:
                # Balanced approach
                risk_weight = 0.6
                min_weight = 0.02
                
            # Define objective function
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
                
                # Combine return and risk objectives
                return -(risk_weight * sharpe + (1 - risk_weight) * portfolio_return)
                
            # Additional constraints
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'ineq', 'fun': lambda x: x - min_weight}  # Minimum weight
            ]
            
            # Add correlation constraints
            max_correlation = constraints.get('max_correlation', 0.7)
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    if correlation.iloc[i,j] > max_correlation:
                        # Limit combined weight of highly correlated assets
                        constraints_list.append({
                            'type': 'ineq',
                            'fun': lambda x, i=i, j=j: constraints.get('max_combined_weight', 0.4) - (x[i] + x[j])
                        })
            
            # Optimize
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list
            )
            
            if not result.success:
                logger.warning(f"Portfolio optimization did not converge: {result.message}")
            
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            port_return = np.sum(mean_returns * optimal_weights)
            port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol
            
            # Calculate maximum drawdown
            portfolio_values = (1 + (returns * optimal_weights).sum(axis=1)).cumprod()
            rolling_max = portfolio_values.expanding().max()
            drawdowns = (portfolio_values - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Create allocation object
            allocation = PortfolioAllocation(
                weights=dict(zip(returns.columns, optimal_weights)),
                expected_return=float(port_return),
                volatility=float(port_vol),
                sharpe_ratio=float(sharpe),
                max_drawdown=float(max_drawdown),
                correlation_matrix=correlation,
                timestamp=datetime.now()
            )
            
            self.historical_allocations.append(allocation)
            if len(self.historical_allocations) > 100:
                self.historical_allocations.pop(0)
                
            return allocation
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            raise

    def get_rebalancing_trades(self, 
                             current_positions: Dict[str, float],
                             target_allocation: PortfolioAllocation,
                             total_value: float) -> Dict[str, float]:
        """Calculate required trades to achieve target allocation"""
        try:
            trades = {}
            current_total = sum(current_positions.values())
            
            for symbol, target_weight in target_allocation.weights.items():
                current_position = current_positions.get(symbol, 0.0)
                target_position = total_value * target_weight
                
                # Calculate trade size
                trade_size = target_position - current_position
                
                # Apply minimum trade size filter
                if abs(trade_size) > (total_value * 0.01):  # 1% minimum trade size
                    trades[symbol] = trade_size
                    
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {e}")
            raise

    def analyze_allocation_history(self) -> Dict[str, Any]:
        """Analyze historical allocation changes"""
        if len(self.historical_allocations) < 2:
            return {}
            
        try:
            # Calculate allocation stability
            weight_changes = []
            for prev, curr in zip(self.historical_allocations[:-1], self.historical_allocations[1:]):
                changes = {
                    symbol: abs(curr.weights.get(symbol, 0) - prev.weights.get(symbol, 0))
                    for symbol in set(curr.weights) | set(prev.weights)
                }
                weight_changes.append(changes)
                
            avg_changes = pd.DataFrame(weight_changes).mean()
            
            # Calculate performance metrics
            returns = [alloc.expected_return for alloc in self.historical_allocations]
            volatilities = [alloc.volatility for alloc in self.historical_allocations]
            sharpe_ratios = [alloc.sharpe_ratio for alloc in self.historical_allocations]
            
            return {
                'allocation_stability': 1 - avg_changes.mean(),
                'return_trend': np.polyfit(range(len(returns)), returns, 1)[0],
                'volatility_trend': np.polyfit(range(len(volatilities)), volatilities, 1)[0],
                'sharpe_ratio_trend': np.polyfit(range(len(sharpe_ratios)), sharpe_ratios, 1)[0],
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_turnover': avg_changes.mean()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing allocation history: {e}")
            raise

    def adjust_for_market_regime(self, 
                               base_allocation: PortfolioAllocation,
                               regime: str,
                               regime_data: Dict[str, Any]) -> PortfolioAllocation:
        """Adjust portfolio allocation based on market regime"""
        try:
            adjusted_weights = base_allocation.weights.copy()
            
            if regime == "crisis":
                # Reduce risk during crisis
                for symbol, weight in adjusted_weights.items():
                    if regime_data.get('volatility', {}).get(symbol, 0) > 0.3:  # High volatility
                        adjusted_weights[symbol] = weight * 0.5  # Reduce exposure
                        
            elif regime == "trending":
                # Increase exposure to trending assets
                for symbol, weight in adjusted_weights.items():
                    if regime_data.get('trend_strength', {}).get(symbol, 0) > 0.7:  # Strong trend
                        adjusted_weights[symbol] = min(weight * 1.5, 0.4)  # Increase exposure
                        
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            
            return PortfolioAllocation(
                weights=adjusted_weights,
                expected_return=base_allocation.expected_return,
                volatility=base_allocation.volatility,
                sharpe_ratio=base_allocation.sharpe_ratio,
                max_drawdown=base_allocation.max_drawdown,
                correlation_matrix=base_allocation.correlation_matrix,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error adjusting for market regime: {e}")
            raise 