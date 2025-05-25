import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json
from .trading_activity_monitor import TradingActivityMonitor
from .safety_manager import SafetyManager

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"
    
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    
@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_window: Optional[int] = None  # برای سفارشات TWAP/VWAP
    num_slices: Optional[int] = None   # برای سفارشات TWAP/VWAP
    visible_size: Optional[float] = None  # برای سفارشات Iceberg
    
class SmartOrderRouter:
    def __init__(self, 
                 max_slippage: float = 0.001,
                 min_execution_size: float = 0.01,
                 market_impact_threshold: float = 0.05):
        """
        سیستم اجرای سفارش هوشمند
        """
        self.logger = self._setup_logger()
        self.max_slippage = max_slippage
        self.min_execution_size = min_execution_size
        self.market_impact_threshold = market_impact_threshold
        
        # پارامترهای ریسک
        self.max_position_size = 1.0
        self.stop_loss_multiplier = 1.0
        self.take_profit_multiplier = 1.0
        
        # پارامترهای معاملاتی
        self.entry_threshold = 0.01
        self.min_profit_target = 0.02
        self.signal_sensitivity = 1.0
        
        # تاریخچه اجرا
        self.execution_history = []
        
        # مانیتورها
        self.activity_monitor = TradingActivityMonitor()
        self.safety_manager = SafetyManager()
        
        # استراتژی‌ها
        self.active_strategies = set(['trend_following', 'mean_reversion', 'breakout'])
        self.strategy_weights = {
            'trend_following': 0.4,
            'mean_reversion': 0.3,
            'breakout': 0.3
        }
        
        # وضعیت پورتفولیو
        self.portfolio_state = {
            'equity_curve': [],
            'positions': {},
            'strategy_allocations': self.strategy_weights.copy(),
            'strategy_returns': {strat: [] for strat in self.active_strategies},
            'total_equity': 1000000  # مقدار اولیه
        }
        
    def _setup_logger(self):
        logger = logging.getLogger('SmartOrderRouter')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('logs/order_execution.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def route_order(self, order: Order, market_data: Dict) -> Dict:
        """
        مسیریابی و اجرای هوشمند سفارش
        """
        self.logger.info(f"Processing order: {order}")
        
        # بررسی وضعیت ایمنی
        safety_status = self._check_safety(market_data)
        if safety_status['status'] != 'SAFE':
            self._apply_safety_actions(safety_status['actions'])
            if safety_status['status'] == 'DANGER':
                return {'status': 'REJECTED', 'reason': 'Safety checks failed'}
                
        # بررسی وضعیت فعالیت معاملاتی
        activity_status = self._check_trading_activity(market_data)
        if activity_status['status'] != 'NORMAL':
            self._apply_activity_adjustments(activity_status['adjustments'])
            
        execution_plan = self._create_execution_plan(order, market_data)
        
        if execution_plan['estimated_impact'] > self.market_impact_threshold:
            self.logger.warning("High market impact detected, switching to algorithmic execution")
            result = self._execute_algorithmic_order(order, market_data)
        else:
            result = self._execute_order(order, execution_plan)
            
        # به‌روزرسانی وضعیت پورتفولیو
        self._update_portfolio_state(result, market_data)
        
        # به‌روزرسانی تاریخچه معاملات
        self._update_trading_history(result, market_data)
        
        return result
        
    def _check_safety(self, market_data: Dict) -> Dict:
        """
        بررسی وضعیت ایمنی
        """
        return self.safety_manager.check_safety(
            self.portfolio_state,
            market_data,
            list(self.active_strategies)
        )
        
    def _apply_safety_actions(self, actions: Dict):
        """
        اعمال اقدامات ایمنی
        """
        if 'emergency_actions' in actions:
            self._handle_emergency_actions(actions['emergency_actions'])
            
        if 'risk_reduction' in actions:
            self._adjust_risk_parameters(actions['risk_reduction'])
            
        if 'volatility_control' in actions:
            self._adjust_volatility_controls(actions['volatility_control'])
            
        if 'diversification' in actions:
            self._handle_diversification(actions['diversification'])
            
    def _handle_emergency_actions(self, actions: Dict):
        """
        اجرای اقدامات اضطراری
        """
        if actions.get('reduce_exposure'):
            self._reduce_exposure(float(actions['reduce_exposure'].rstrip('%')) / 100)
            
        if actions.get('close_risky_positions'):
            self._close_risky_positions()
            
        if actions.get('activate_hedging'):
            self._activate_hedging()
            
    def _reduce_exposure(self, reduction_ratio: float):
        """
        کاهش میزان مواجهه با ریسک
        """
        self.max_position_size *= (1 - reduction_ratio)
        for position in self.portfolio_state['positions'].values():
            position['size'] *= (1 - reduction_ratio)
            
    def _close_risky_positions(self):
        """
        بستن موقعیت‌های پرریسک
        """
        risky_positions = self._identify_risky_positions()
        for position in risky_positions:
            self._close_position(position)
            
    def _activate_hedging(self):
        """
        فعال‌سازی استراتژی‌های پوششی
        """
        # پیاده‌سازی استراتژی‌های پوششی
        pass
        
    def _update_portfolio_state(self, execution_result: Dict, market_data: Dict):
        """
        به‌روزرسانی وضعیت پورتفولیو
        """
        # به‌روزرسانی منحنی سرمایه
        current_equity = self._calculate_current_equity()
        self.portfolio_state['equity_curve'].append(current_equity)
        
        # به‌روزرسانی موقعیت‌ها
        if execution_result['status'] == 'FILLED':
            symbol = execution_result['symbol']
            if symbol not in self.portfolio_state['positions']:
                self.portfolio_state['positions'][symbol] = {
                    'size': 0,
                    'average_price': 0
                }
                
            position = self.portfolio_state['positions'][symbol]
            if execution_result['side'] == 'BUY':
                position['size'] += execution_result['executed_quantity']
            else:
                position['size'] -= execution_result['executed_quantity']
                
            position['average_price'] = execution_result['average_price']
            
        # به‌روزرسانی بازده استراتژی‌ها
        for strategy in self.active_strategies:
            if strategy in execution_result.get('strategy_returns', {}):
                self.portfolio_state['strategy_returns'][strategy].append(
                    execution_result['strategy_returns'][strategy]
                )
                
    def _calculate_current_equity(self) -> float:
        """
        محاسبه ارزش فعلی پورتفولیو
        """
        total = self.portfolio_state['total_equity']
        for position in self.portfolio_state['positions'].values():
            total += position['size'] * position['average_price']
        return total
        
    def _identify_risky_positions(self) -> List[Dict]:
        """
        شناسایی موقعیت‌های پرریسک
        """
        risky_positions = []
        for symbol, position in self.portfolio_state['positions'].items():
            if abs(position['size']) * position['average_price'] > self.max_position_size:
                risky_positions.append({**position, 'symbol': symbol})
        return risky_positions
        
    def _close_position(self, position: Dict):
        """
        بستن یک موقعیت
        """
        order = Order(
            symbol=position['symbol'],
            side=OrderSide.SELL if position['size'] > 0 else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=abs(position['size'])
        )
        self.route_order(order, {})  # نیاز به داده‌های بازار دارد
        
    def _check_trading_activity(self, market_data: Dict) -> Dict:
        """
        بررسی وضعیت فعالیت معاملاتی
        """
        recent_trades = self.execution_history[-100:] if len(self.execution_history) > 0 else []
        active_strategies = list(self.active_strategies)
        
        return self.activity_monitor.monitor_activity(
            recent_trades,
            market_data,
            active_strategies
        )
        
    def _apply_activity_adjustments(self, adjustments: Dict):
        """
        اعمال تنظیمات پیشنهادی مانیتور فعالیت
        """
        if 'risk_params' in adjustments:
            self._adjust_risk_parameters(adjustments['risk_params'])
            
        if 'trading_params' in adjustments:
            self._adjust_trading_parameters(adjustments['trading_params'])
            
        if 'execution_params' in adjustments:
            self._adjust_execution_parameters(adjustments['execution_params'])
            
        self.logger.info(f"Applied activity adjustments: {adjustments}")
        
    def _adjust_risk_parameters(self, params: Dict):
        """
        تنظیم پارامترهای ریسک
        """
        for param, change in params.items():
            if isinstance(change, str) and (change.endswith('%')):
                value = float(change.rstrip('%')) / 100
                if param == 'max_position_size':
                    self.max_position_size *= (1 + value)
                elif param == 'stop_loss':
                    self.stop_loss_multiplier *= (1 + value)
                elif param == 'take_profit':
                    self.take_profit_multiplier *= (1 + value)
                    
    def _adjust_trading_parameters(self, params: Dict):
        """
        تنظیم پارامترهای معاملاتی
        """
        for param, change in params.items():
            if isinstance(change, str) and (change.endswith('%')):
                value = float(change.rstrip('%')) / 100
                if param == 'entry_threshold':
                    self.entry_threshold *= (1 + value)
                elif param == 'min_profit_target':
                    self.min_profit_target *= (1 + value)
                elif param == 'signal_sensitivity':
                    self.signal_sensitivity *= (1 + value)
                    
    def _adjust_execution_parameters(self, params: Dict):
        """
        تنظیم پارامترهای اجرایی
        """
        for param, change in params.items():
            if isinstance(change, str) and (change.endswith('%')):
                value = float(change.rstrip('%')) / 100
                if param == 'min_trade_size':
                    self.min_execution_size *= (1 + value)
                elif param == 'max_slippage':
                    self.max_slippage *= (1 + value)
                    
    def _adjust_volatility_controls(self, controls: Dict):
        """
        تنظیم کنترل‌های نوسان
        """
        if 'position_scaling' in controls:
            value = float(controls['position_scaling'].rstrip('%')) / 100
            self.max_position_size *= (1 + value)
            
        if 'trade_frequency' in controls:
            value = float(controls['trade_frequency'].rstrip('%')) / 100
            self.min_execution_size *= (1 + value)
            
    def _handle_diversification(self, diversification: Dict):
        """
        مدیریت تنوع استراتژی‌ها
        """
        if diversification.get('rebalance_strategies'):
            self._rebalance_strategies()
            
        if 'add_uncorrelated_strategies' in diversification:
            self._add_new_strategies(diversification['add_uncorrelated_strategies'])
            
    def _rebalance_strategies(self):
        """
        تنظیم مجدد وزن استراتژی‌ها
        """
        num_strategies = len(self.active_strategies)
        if num_strategies > 0:
            base_weight = 1.0 / num_strategies
            self.strategy_weights = {
                strategy: base_weight for strategy in self.active_strategies
            }
            self.portfolio_state['strategy_allocations'] = self.strategy_weights.copy()
            
    def _add_new_strategies(self, num_new: int):
        """
        اضافه کردن استراتژی‌های جدید
        """
        available_strategies = [
            'momentum', 'mean_reversion', 'breakout', 
            'trend_following', 'statistical_arbitrage',
            'volatility_trading', 'pattern_recognition'
        ]
        
        current_strategies = self.active_strategies
        new_strategies = set(available_strategies) - current_strategies
        
        for strategy in list(new_strategies)[:num_new]:
            self.active_strategies.add(strategy)
            self.portfolio_state['strategy_returns'][strategy] = []
            
        self._rebalance_strategies()
        
    def _create_execution_plan(self, order: Order, market_data: Dict) -> Dict:
        """
        ایجاد برنامه اجرای سفارش
        """
        volume_profile = self._analyze_volume_profile(market_data)
        price_impact = self._estimate_price_impact(order, market_data)
        liquidity_score = self._calculate_liquidity_score(market_data)
        
        plan = {
            'timestamp': datetime.now().isoformat(),
            'order_details': {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity
            },
            'market_analysis': {
                'volume_profile': volume_profile,
                'liquidity_score': liquidity_score,
                'estimated_impact': price_impact
            },
            'execution_strategy': self._determine_execution_strategy(order, price_impact),
            'risk_checks': self._perform_risk_checks(order, market_data)
        }
        
        return plan
        
    def _analyze_volume_profile(self, market_data: Dict) -> Dict:
        """
        تحلیل پروفایل حجم معاملات
        """
        volume = pd.Series(market_data['volume'])
        
        return {
            'average_volume': volume.mean(),
            'volume_std': volume.std(),
            'volume_percentiles': {
                '25': volume.quantile(0.25),
                '50': volume.quantile(0.50),
                '75': volume.quantile(0.75)
            }
        }
        
    def _estimate_price_impact(self, order: Order, market_data: Dict) -> float:
        """
        تخمین تاثیر قیمتی سفارش
        """
        avg_volume = pd.Series(market_data['volume']).mean()
        order_volume_ratio = order.quantity / avg_volume
        
        # مدل ساده تاثیر قیمت
        impact = 0.1 * np.sqrt(order_volume_ratio)
        return min(impact, 0.1)  # محدود کردن به حداکثر 10%
        
    def _calculate_liquidity_score(self, market_data: Dict) -> float:
        """
        محاسبه امتیاز نقدشوندگی
        """
        volume = pd.Series(market_data['volume'])
        spread = pd.Series(market_data['ask']) - pd.Series(market_data['bid'])
        
        volume_score = volume.mean() / volume.std()
        spread_score = 1 / (spread.mean() * 100)
        
        return (volume_score + spread_score) / 2
        
    def _determine_execution_strategy(self, order: Order, price_impact: float) -> Dict:
        """
        تعیین استراتژی اجرای سفارش
        """
        if price_impact > self.market_impact_threshold:
            if order.quantity > self.min_execution_size * 100:
                return {
                    'type': 'ALGORITHMIC',
                    'algorithm': 'VWAP',
                    'parameters': {
                        'num_slices': 10,
                        'time_window': 3600  # یک ساعت
                    }
                }
            else:
                return {
                    'type': 'ALGORITHMIC',
                    'algorithm': 'TWAP',
                    'parameters': {
                        'num_slices': 5,
                        'time_window': 1800  # 30 دقیقه
                    }
                }
        else:
            return {
                'type': 'DIRECT',
                'method': order.order_type.value
            }
            
    def _perform_risk_checks(self, order: Order, market_data: Dict) -> Dict:
        """
        بررسی‌های ریسک قبل از اجرا
        """
        avg_volume = pd.Series(market_data['volume']).mean()
        current_price = (market_data['ask'][-1] + market_data['bid'][-1]) / 2
        
        checks = {
            'volume_check': order.quantity < avg_volume * 0.1,
            'price_check': True,
            'risk_limits': True
        }
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            price_deviation = abs(order.price - current_price) / current_price
            checks['price_check'] = price_deviation < 0.1
            
        return checks
        
    def _execute_algorithmic_order(self, order: Order, market_data: Dict) -> Dict:
        """
        اجرای الگوریتمی سفارش
        """
        if order.order_type == OrderType.VWAP:
            return self._execute_vwap(order, market_data)
        elif order.order_type == OrderType.TWAP:
            return self._execute_twap(order, market_data)
        elif order.order_type == OrderType.ICEBERG:
            return self._execute_iceberg(order, market_data)
        else:
            return self._execute_order(order, self._create_execution_plan(order, market_data))
            
    def _execute_vwap(self, order: Order, market_data: Dict) -> Dict:
        """
        اجرای سفارش با الگوریتم VWAP
        """
        volume_profile = self._analyze_volume_profile(market_data)
        slice_sizes = self._calculate_vwap_slices(order.quantity, volume_profile)
        
        execution_result = {
            'algorithm': 'VWAP',
            'slices': len(slice_sizes),
            'executed_quantity': 0,
            'average_price': 0,
            'details': []
        }
        
        for size in slice_sizes:
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=size
            )
            result = self._execute_order(slice_order, self._create_execution_plan(slice_order, market_data))
            execution_result['details'].append(result)
            execution_result['executed_quantity'] += result['executed_quantity']
            execution_result['average_price'] = (
                (execution_result['average_price'] * (len(execution_result['details']) - 1) +
                 result['average_price']) / len(execution_result['details'])
            )
            
        return execution_result
        
    def _execute_twap(self, order: Order, market_data: Dict) -> Dict:
        """
        اجرای سفارش با الگوریتم TWAP
        """
        slice_size = order.quantity / order.num_slices
        time_interval = order.time_window / order.num_slices
        
        execution_result = {
            'algorithm': 'TWAP',
            'slices': order.num_slices,
            'executed_quantity': 0,
            'average_price': 0,
            'details': []
        }
        
        for i in range(order.num_slices):
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_size
            )
            result = self._execute_order(slice_order, self._create_execution_plan(slice_order, market_data))
            execution_result['details'].append(result)
            execution_result['executed_quantity'] += result['executed_quantity']
            execution_result['average_price'] = (
                (execution_result['average_price'] * i + result['average_price']) / (i + 1)
            )
            
        return execution_result
        
    def _execute_iceberg(self, order: Order, market_data: Dict) -> Dict:
        """
        اجرای سفارش با الگوریتم Iceberg
        """
        visible_size = order.visible_size or order.quantity * 0.1
        remaining_quantity = order.quantity
        
        execution_result = {
            'algorithm': 'ICEBERG',
            'visible_size': visible_size,
            'executed_quantity': 0,
            'average_price': 0,
            'details': []
        }
        
        while remaining_quantity > 0:
            current_size = min(visible_size, remaining_quantity)
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=current_size,
                price=order.price
            )
            result = self._execute_order(slice_order, self._create_execution_plan(slice_order, market_data))
            execution_result['details'].append(result)
            execution_result['executed_quantity'] += result['executed_quantity']
            remaining_quantity -= result['executed_quantity']
            execution_result['average_price'] = (
                (execution_result['average_price'] * 
                 (execution_result['executed_quantity'] - result['executed_quantity']) +
                 result['average_price'] * result['executed_quantity']) /
                execution_result['executed_quantity']
            )
            
        return execution_result
        
    def _execute_order(self, order: Order, execution_plan: Dict) -> Dict:
        """
        اجرای مستقیم سفارش
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'order_id': f"ORD-{len(self.execution_history) + 1}",
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'requested_quantity': order.quantity,
            'executed_quantity': order.quantity,  # در اینجا فرض می‌کنیم کل سفارش اجرا می‌شود
            'average_price': order.price or execution_plan['market_analysis']['estimated_impact'],
            'execution_time': datetime.now().isoformat(),
            'status': 'FILLED'
        }
        
        self.execution_history.append(result)
        self.logger.info(f"Order executed: {result}")
        
        return result
        
    def _calculate_vwap_slices(self, total_quantity: float, 
                              volume_profile: Dict) -> List[float]:
        """
        محاسبه اندازه برش‌های VWAP
        """
        percentiles = volume_profile['volume_percentiles']
        weights = [0.2, 0.3, 0.5]  # وزن‌های مختلف برای هر سطح حجم
        
        slice_sizes = []
        remaining_quantity = total_quantity
        
        while remaining_quantity > 0:
            for weight, percentile in zip(weights, percentiles.values()):
                size = min(remaining_quantity, percentile * weight)
                if size >= self.min_execution_size:
                    slice_sizes.append(size)
                    remaining_quantity -= size
                    
            if remaining_quantity < self.min_execution_size:
                slice_sizes[-1] += remaining_quantity
                break
                
        return slice_sizes

if __name__ == "__main__":
    # نمونه استفاده
    router = SmartOrderRouter()
    
    # داده تست
    market_data = {
        'volume': np.random.normal(1000000, 200000, 100),
        'ask': np.random.normal(100, 0.1, 100),
        'bid': np.random.normal(99.9, 0.1, 100)
    }
    
    # سفارش نمونه
    test_order = Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.VWAP,
        quantity=10.0,
        time_window=3600,
        num_slices=10
    )
    
    result = router.route_order(test_order, market_data) 