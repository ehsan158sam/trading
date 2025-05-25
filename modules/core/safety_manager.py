import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class SafetyStatus(Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    DANGER = "DANGER"

@dataclass
class SafetyMetrics:
    drawdown: float
    volatility: float
    exposure: float
    risk_concentration: float
    strategy_correlation: float

class SafetyManager:
    def __init__(self,
                 max_drawdown: float = 0.1,
                 max_volatility: float = 0.2,
                 max_exposure: float = 0.8,
                 max_risk_concentration: float = 0.3,
                 max_strategy_correlation: float = 0.7):
        """
        مدیریت ایمنی سیستم معاملاتی
        
        Parameters:
        -----------
        max_drawdown : float
            حداکثر افت سرمایه مجاز
        max_volatility : float
            حداکثر نوسان مجاز
        max_exposure : float
            حداکثر میزان مواجهه با ریسک
        max_risk_concentration : float
            حداکثر تمرکز ریسک در یک استراتژی
        max_strategy_correlation : float
            حداکثر همبستگی مجاز بین استراتژی‌ها
        """
        self.logger = self._setup_logger()
        self.max_drawdown = max_drawdown
        self.max_volatility = max_volatility
        self.max_exposure = max_exposure
        self.max_risk_concentration = max_risk_concentration
        self.max_strategy_correlation = max_strategy_correlation
        
        self.safety_history = []
        self.emergency_stops = 0
        self.last_check_time = datetime.now()
        
    def _setup_logger(self):
        logger = logging.getLogger('SafetyManager')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('logs/safety_manager.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def check_safety(self,
                    portfolio_state: Dict,
                    market_data: Dict,
                    active_strategies: List[str]) -> Dict:
        """
        بررسی وضعیت ایمنی سیستم
        """
        # محاسبه متریک‌های ایمنی
        metrics = self._calculate_safety_metrics(
            portfolio_state, market_data, active_strategies
        )
        
        # ارزیابی وضعیت
        status = self._evaluate_safety_status(metrics)
        
        # اقدامات ایمنی در صورت نیاز
        safety_actions = {}
        if status != SafetyStatus.SAFE:
            safety_actions = self._determine_safety_actions(metrics, status)
            
        report = self._create_safety_report(metrics, status, safety_actions)
        
        # ذخیره تاریخچه
        self._update_history(metrics, status, safety_actions)
        
        return report
        
    def _calculate_safety_metrics(self,
                                portfolio_state: Dict,
                                market_data: Dict,
                                active_strategies: List[str]) -> SafetyMetrics:
        """
        محاسبه متریک‌های ایمنی
        """
        # محاسبه افت سرمایه
        equity_curve = portfolio_state.get('equity_curve', [])
        peak = max(equity_curve) if equity_curve else 0
        current = equity_curve[-1] if equity_curve else 0
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        # محاسبه نوسانات
        returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else [0]
        volatility = np.std(returns) * np.sqrt(252)  # سالانه شده
        
        # محاسبه میزان مواجهه با ریسک
        total_position = sum(abs(pos['size']) for pos in portfolio_state.get('positions', {}).values())
        exposure = total_position / portfolio_state.get('total_equity', 1)
        
        # محاسبه تمرکز ریسک
        strategy_allocations = portfolio_state.get('strategy_allocations', {})
        risk_concentration = max(strategy_allocations.values()) if strategy_allocations else 0
        
        # محاسبه همبستگی استراتژی‌ها
        strategy_returns = portfolio_state.get('strategy_returns', {})
        correlations = []
        for i, strat1 in enumerate(active_strategies[:-1]):
            for strat2 in active_strategies[i+1:]:
                if strat1 in strategy_returns and strat2 in strategy_returns:
                    corr = np.corrcoef(strategy_returns[strat1], strategy_returns[strat2])[0, 1]
                    correlations.append(abs(corr))
        strategy_correlation = max(correlations) if correlations else 0
        
        return SafetyMetrics(
            drawdown=drawdown,
            volatility=volatility,
            exposure=exposure,
            risk_concentration=risk_concentration,
            strategy_correlation=strategy_correlation
        )
        
    def _evaluate_safety_status(self, metrics: SafetyMetrics) -> SafetyStatus:
        """
        ارزیابی وضعیت ایمنی
        """
        if (metrics.drawdown > self.max_drawdown or
            metrics.volatility > self.max_volatility or
            metrics.exposure > self.max_exposure):
            return SafetyStatus.DANGER
            
        if (metrics.risk_concentration > self.max_risk_concentration or
            metrics.strategy_correlation > self.max_strategy_correlation):
            return SafetyStatus.CAUTION
            
        return SafetyStatus.SAFE
        
    def _determine_safety_actions(self,
                                metrics: SafetyMetrics,
                                status: SafetyStatus) -> Dict:
        """
        تعیین اقدامات ایمنی مورد نیاز
        """
        actions = {}
        
        if status == SafetyStatus.DANGER:
            actions['emergency_actions'] = {
                'reduce_exposure': '-50%',
                'close_risky_positions': True,
                'activate_hedging': True
            }
            self.emergency_stops += 1
            
        if metrics.drawdown > self.max_drawdown:
            actions['risk_reduction'] = {
                'max_position_size': '-30%',
                'stop_loss_tightening': '+20%'
            }
            
        if metrics.volatility > self.max_volatility:
            actions['volatility_control'] = {
                'position_scaling': '-25%',
                'trade_frequency': '-40%'
            }
            
        if metrics.risk_concentration > self.max_risk_concentration:
            actions['diversification'] = {
                'rebalance_strategies': True,
                'add_uncorrelated_strategies': 2
            }
            
        return actions
        
    def _create_safety_report(self,
                            metrics: SafetyMetrics,
                            status: SafetyStatus,
                            actions: Dict) -> Dict:
        """
        ایجاد گزارش ایمنی
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'status': status.value,
            'metrics': {
                'drawdown': metrics.drawdown,
                'volatility': metrics.volatility,
                'exposure': metrics.exposure,
                'risk_concentration': metrics.risk_concentration,
                'strategy_correlation': metrics.strategy_correlation
            },
            'thresholds': {
                'max_drawdown': self.max_drawdown,
                'max_volatility': self.max_volatility,
                'max_exposure': self.max_exposure,
                'max_risk_concentration': self.max_risk_concentration,
                'max_strategy_correlation': self.max_strategy_correlation
            },
            'actions': actions,
            'emergency_stops': self.emergency_stops
        }
        
        return report
        
    def _update_history(self,
                       metrics: SafetyMetrics,
                       status: SafetyStatus,
                       actions: Dict):
        """
        به‌روزرسانی تاریخچه ایمنی
        """
        self.safety_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'drawdown': metrics.drawdown,
                'volatility': metrics.volatility,
                'exposure': metrics.exposure,
                'risk_concentration': metrics.risk_concentration,
                'strategy_correlation': metrics.strategy_correlation
            },
            'status': status.value,
            'actions': actions
        })
        
        # حفظ تاریخچه برای 30 روز
        cutoff_date = datetime.now() - timedelta(days=30)
        self.safety_history = [
            record for record in self.safety_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
    def get_safety_summary(self) -> Dict:
        """
        دریافت خلاصه وضعیت ایمنی
        """
        if not self.safety_history:
            return {"error": "No safety history available"}
            
        recent_metrics = [record['metrics'] for record in self.safety_history]
        
        summary = {
            'current_status': self.safety_history[-1]['status'],
            'emergency_stops': self.emergency_stops,
            'metrics_trend': {
                metric: self._calculate_trend([m[metric] for m in recent_metrics])
                for metric in recent_metrics[0].keys()
            },
            'risk_level': self._calculate_risk_level()
        }
        
        return summary
        
    def _calculate_trend(self, values: List[float]) -> str:
        """
        محاسبه روند یک متریک
        """
        if len(values) < 2:
            return "insufficient_data"
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
            
    def _calculate_risk_level(self) -> str:
        """
        محاسبه سطح ریسک کلی
        """
        if not self.safety_history:
            return "unknown"
            
        recent_statuses = [record['status'] for record in self.safety_history[-10:]]
        danger_count = recent_statuses.count(SafetyStatus.DANGER.value)
        caution_count = recent_statuses.count(SafetyStatus.CAUTION.value)
        
        if danger_count > 3:
            return "high"
        elif danger_count > 1 or caution_count > 4:
            return "elevated"
        else:
            return "normal" 