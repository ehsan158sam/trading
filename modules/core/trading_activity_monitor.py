import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from enum import Enum

class ActivityStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

@dataclass
class ActivityMetrics:
    trades_count: int
    win_rate: float
    profit_factor: float
    avg_trade_frequency: float
    market_participation: float
    strategy_diversity: float

class TradingActivityMonitor:
    def __init__(self,
                 min_daily_trades: int = 5,
                 min_win_rate: float = 0.45,
                 min_profit_factor: float = 1.2,
                 monitoring_window: int = 7,  # days
                 alert_threshold: float = 0.7,
                 critical_threshold: float = 0.5,
                 adjustment_cooldown: int = 4):  # hours
        """
        نظارت و تنظیم خودکار فعالیت‌های معاملاتی
        
        Parameters:
        -----------
        min_daily_trades : int
            حداقل تعداد معاملات روزانه مورد انتظار
        min_win_rate : float
            حداقل نرخ موفقیت قابل قبول
        min_profit_factor : float
            حداقل فاکتور سود قابل قبول
        monitoring_window : int
            پنجره زمانی نظارت (روز)
        alert_threshold : float
            آستانه هشدار برای کاهش فعالیت
        critical_threshold : float
            آستانه بحرانی برای کاهش فعالیت
        adjustment_cooldown : int
            زمان انتظار بین تنظیمات (ساعت)
        """
        self.logger = self._setup_logger()
        self.min_daily_trades = min_daily_trades
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.monitoring_window = monitoring_window
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.adjustment_cooldown = adjustment_cooldown
        
        self.trading_history = []
        self.active_strategies = set()
        self.parameter_history = []
        self.last_adjustment_time = datetime.now()
        self.consecutive_warnings = 0
        self.adjustment_limits = {
            'risk_params': {'max_adjustments': 3, 'current': 0},
            'trading_params': {'max_adjustments': 5, 'current': 0},
            'strategy_params': {'max_adjustments': 2, 'current': 0},
            'execution_params': {'max_adjustments': 4, 'current': 0}
        }
        
    def _setup_logger(self):
        logger = logging.getLogger('TradingActivityMonitor')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('logs/activity_monitor.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def monitor_activity(self, 
                        recent_trades: List[Dict],
                        market_data: Dict,
                        active_strategies: List[str]) -> Dict:
        """
        نظارت بر فعالیت معاملاتی و تنظیم خودکار پارامترها
        """
        # بررسی زمان انتظار بین تنظیمات
        if not self._can_make_adjustments():
            return self._create_monitoring_report(None, ActivityStatus.NORMAL, {})
            
        # محاسبه متریک‌های فعالیت
        metrics = self._calculate_activity_metrics(recent_trades, market_data, active_strategies)
        
        # ارزیابی وضعیت
        status = self._evaluate_activity_status(metrics)
        
        # به‌روزرسانی شمارنده هشدارها
        self._update_warning_counter(status)
        
        # اقدامات اصلاحی در صورت نیاز
        adjustments = {}
        if status != ActivityStatus.NORMAL and self._should_make_adjustments():
            adjustments = self._make_adjustments(metrics, status)
            if adjustments:
                self.last_adjustment_time = datetime.now()
                self._update_adjustment_counters(adjustments)
            
        report = self._create_monitoring_report(metrics, status, adjustments)
        
        # ذخیره تاریخچه
        self._update_history(metrics, status, adjustments)
        
        return report
        
    def _can_make_adjustments(self) -> bool:
        """
        بررسی امکان انجام تنظیمات جدید
        """
        time_since_last = (datetime.now() - self.last_adjustment_time).total_seconds() / 3600
        return time_since_last >= self.adjustment_cooldown
        
    def _should_make_adjustments(self) -> bool:
        """
        تصمیم‌گیری برای انجام تنظیمات
        """
        # بررسی محدودیت‌های تنظیمات
        for param_type, limits in self.adjustment_limits.items():
            if limits['current'] >= limits['max_adjustments']:
                self.logger.warning(f"Adjustment limit reached for {param_type}")
                return False
                
        return True
        
    def _update_warning_counter(self, status: ActivityStatus):
        """
        به‌روزرسانی شمارنده هشدارها
        """
        if status == ActivityStatus.WARNING:
            self.consecutive_warnings += 1
        elif status == ActivityStatus.CRITICAL:
            self.consecutive_warnings += 2
        else:
            self.consecutive_warnings = max(0, self.consecutive_warnings - 1)
            
    def _update_adjustment_counters(self, adjustments: Dict):
        """
        به‌روزرسانی شمارنده‌های تنظیمات
        """
        for param_type in adjustments:
            if param_type in self.adjustment_limits:
                self.adjustment_limits[param_type]['current'] += 1
                
    def _calculate_activity_metrics(self,
                                  recent_trades: List[Dict],
                                  market_data: Dict,
                                  active_strategies: List[str]) -> ActivityMetrics:
        """
        محاسبه متریک‌های فعالیت معاملاتی
        """
        # تعداد معاملات
        trades_count = len(recent_trades)
        
        # نرخ موفقیت
        winning_trades = sum(1 for trade in recent_trades if trade['profit'] > 0)
        win_rate = winning_trades / trades_count if trades_count > 0 else 0
        
        # فاکتور سود
        total_profit = sum(trade['profit'] for trade in recent_trades if trade['profit'] > 0)
        total_loss = abs(sum(trade['profit'] for trade in recent_trades if trade['profit'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # فرکانس معاملات
        if trades_count > 1:
            time_span = (recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']).total_seconds() / 3600
            avg_trade_frequency = trades_count / time_span if time_span > 0 else 0
        else:
            avg_trade_frequency = 0
        
        # مشارکت در بازار
        total_volume = sum(trade['volume'] for trade in recent_trades)
        market_volume = sum(market_data['volume'])
        market_participation = total_volume / market_volume if market_volume > 0 else 0
        
        # تنوع استراتژی
        strategy_diversity = len(set(active_strategies)) / len(active_strategies) if active_strategies else 0
        
        return ActivityMetrics(
            trades_count=trades_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_frequency=avg_trade_frequency,
            market_participation=market_participation,
            strategy_diversity=strategy_diversity
        )
        
    def _evaluate_activity_status(self, metrics: ActivityMetrics) -> ActivityStatus:
        """
        ارزیابی وضعیت فعالیت معاملاتی
        """
        # محاسبه امتیاز کلی فعالیت
        activity_score = self._calculate_activity_score(metrics)
        
        if activity_score < self.critical_threshold:
            return ActivityStatus.CRITICAL
        elif activity_score < self.alert_threshold:
            return ActivityStatus.WARNING
        else:
            return ActivityStatus.NORMAL
            
    def _calculate_activity_score(self, metrics: ActivityMetrics) -> float:
        """
        محاسبه امتیاز کلی فعالیت معاملاتی
        """
        weights = {
            'trades_count': 0.3,
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'avg_trade_frequency': 0.1,
            'market_participation': 0.1,
            'strategy_diversity': 0.1
        }
        
        normalized_metrics = {
            'trades_count': min(1.0, metrics.trades_count / (self.min_daily_trades * self.monitoring_window)),
            'win_rate': min(1.0, metrics.win_rate / self.min_win_rate),
            'profit_factor': min(1.0, metrics.profit_factor / self.min_profit_factor),
            'avg_trade_frequency': min(1.0, metrics.avg_trade_frequency / (self.min_daily_trades / 24)),
            'market_participation': metrics.market_participation,
            'strategy_diversity': metrics.strategy_diversity
        }
        
        return sum(normalized_metrics[metric] * weight 
                  for metric, weight in weights.items())
                  
    def _make_adjustments(self, 
                         metrics: ActivityMetrics, 
                         status: ActivityStatus) -> Dict:
        """
        انجام تنظیمات خودکار برای بهبود فعالیت معاملاتی
        """
        adjustments = {}
        
        # تنظیم پارامترهای ریسک در وضعیت بحرانی
        if (status == ActivityStatus.CRITICAL and 
            self.adjustment_limits['risk_params']['current'] < 
            self.adjustment_limits['risk_params']['max_adjustments']):
            adjustments['risk_params'] = {
                'max_position_size': '+20%',
                'stop_loss': '-10%',
                'take_profit': '-5%'
            }
            
        # تنظیم پارامترهای معاملاتی
        if (metrics.trades_count < self.min_daily_trades and
            self.adjustment_limits['trading_params']['current'] <
            self.adjustment_limits['trading_params']['max_adjustments']):
            adjustments['trading_params'] = {
                'entry_threshold': '-15%',
                'min_profit_target': '-10%',
                'signal_sensitivity': '+25%'
            }
            
        # تنظیم تنوع استراتژی
        if (metrics.strategy_diversity < 0.5 and
            self.adjustment_limits['strategy_params']['current'] <
            self.adjustment_limits['strategy_params']['max_adjustments']):
            adjustments['strategy_params'] = {
                'active_strategies': '+2',
                'strategy_weights': 'rebalance'
            }
            
        # تنظیم پارامترهای اجرایی
        if (metrics.market_participation < 0.01 and
            self.adjustment_limits['execution_params']['current'] <
            self.adjustment_limits['execution_params']['max_adjustments']):
            adjustments['execution_params'] = {
                'min_trade_size': '-20%',
                'max_slippage': '+10%',
                'order_types': ['market', 'limit', 'stop']
            }
            
        return adjustments
        
    def _create_monitoring_report(self,
                                metrics: Optional[ActivityMetrics],
                                status: ActivityStatus,
                                adjustments: Dict) -> Dict:
        """
        ایجاد گزارش نظارت
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': status.value,
            'consecutive_warnings': self.consecutive_warnings,
            'adjustment_limits': self.adjustment_limits,
            'time_since_last_adjustment': 
                (datetime.now() - self.last_adjustment_time).total_seconds() / 3600
        }
        
        if metrics:
            report['metrics'] = self._metrics_to_dict(metrics)
            report['adjustments'] = adjustments
            report['recommendations'] = self._generate_recommendations(metrics, status)
            
        return report
        
    def _metrics_to_dict(self, metrics: ActivityMetrics) -> Dict:
        """
        تبدیل متریک‌ها به دیکشنری
        """
        return {
            'trades_count': metrics.trades_count,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'avg_trade_frequency': metrics.avg_trade_frequency,
            'market_participation': metrics.market_participation,
            'strategy_diversity': metrics.strategy_diversity
        }
        
    def _generate_recommendations(self,
                                metrics: ActivityMetrics,
                                status: ActivityStatus) -> List[str]:
        """
        تولید توصیه‌های بهبود فعالیت
        """
        recommendations = []
        
        if status == ActivityStatus.CRITICAL:
            recommendations.extend([
                "افزایش حساسیت سیگنال‌های معاملاتی",
                "کاهش حد سود برای افزایش تعداد معاملات",
                "فعال‌سازی استراتژی‌های جدید"
            ])
            
        if metrics.win_rate < self.min_win_rate:
            recommendations.extend([
                "بهینه‌سازی فیلترهای ورود به معامله",
                "افزایش دوره تحلیل قبل از معامله"
            ])
            
        if metrics.strategy_diversity < 0.5:
            recommendations.extend([
                "اضافه کردن استراتژی‌های جدید",
                "متنوع‌سازی ابزارهای معاملاتی"
            ])
            
        return recommendations
        
    def _update_history(self,
                       metrics: ActivityMetrics,
                       status: ActivityStatus,
                       adjustments: Dict):
        """
        به‌روزرسانی تاریخچه نظارت
        """
        self.trading_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': self._metrics_to_dict(metrics),
            'status': status.value,
            'adjustments': adjustments
        })
        
        # حفظ تاریخچه برای دوره مانیتورینگ
        cutoff_date = datetime.now() - timedelta(days=self.monitoring_window)
        self.trading_history = [
            record for record in self.trading_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]

if __name__ == "__main__":
    # نمونه استفاده
    monitor = TradingActivityMonitor()
    
    # داده تست
    test_trades = [
        {'timestamp': datetime.now() - timedelta(hours=i),
         'profit': np.random.normal(0, 100),
         'volume': np.random.normal(1, 0.2)}
        for i in range(24)
    ]
    
    test_market_data = {
        'volume': np.random.normal(1000000, 200000, 24)
    }
    
    test_strategies = ['trend_following', 'mean_reversion', 'breakout']
    
    report = monitor.monitor_activity(test_trades, test_market_data, test_strategies) 