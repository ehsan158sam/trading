import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSystem:
    def __init__(self, 
                 volatility_threshold: float = 0.3,
                 drawdown_threshold: float = 0.15,
                 volume_spike_threshold: float = 3.0,
                 price_change_threshold: float = 0.1):
        """
        سیستم هشدار خودکار برای شرایط بحرانی بازار
        
        Parameters:
        -----------
        volatility_threshold : float
            آستانه نوسانات برای صدور هشدار
        drawdown_threshold : float
            آستانه افت سرمایه برای صدور هشدار
        volume_spike_threshold : float
            آستانه افزایش حجم معاملات
        price_change_threshold : float
            آستانه تغییر قیمت برای هشدار
        """
        self.logger = self._setup_logger()
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.price_change_threshold = price_change_threshold
        self.alert_history = []
        
    def _setup_logger(self):
        logger = logging.getLogger('AlertSystem')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('debug_logs/alerts.log')
        fh.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def check_market_conditions(self, 
                              market_data: Dict[str, pd.DataFrame],
                              risk_metrics: Optional[Dict] = None) -> List[Dict]:
        """
        بررسی شرایط بازار و صدور هشدار در صورت نیاز
        """
        alerts = []
        
        for symbol, data in market_data.items():
            # بررسی تغییرات قیمت
            returns = data['close'].pct_change()
            latest_return = returns.iloc[-1]
            
            if abs(latest_return) > self.price_change_threshold:
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': 'PRICE_CHANGE',
                    'severity': 'HIGH',
                    'message': f"Significant price change detected: {latest_return:.2%}"
                })
                
            # بررسی نوسانات
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1]
            
            if current_vol > self.volatility_threshold:
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': 'VOLATILITY',
                    'severity': 'MEDIUM',
                    'message': f"High volatility detected: {current_vol:.2%}"
                })
                
            # بررسی حجم معاملات
            volume_ma = data['volume'].rolling(window=20).mean()
            current_volume = data['volume'].iloc[-1]
            
            if current_volume > volume_ma.iloc[-1] * self.volume_spike_threshold:
                alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'type': 'VOLUME_SPIKE',
                    'severity': 'MEDIUM',
                    'message': f"Unusual volume spike detected: {current_volume:,.0f}"
                })
                
            # بررسی معیارهای ریسک اگر در دسترس باشند
            if risk_metrics and symbol in risk_metrics:
                metrics = risk_metrics[symbol]
                
                if metrics['stress_metrics']['worst_drawdown'] > self.drawdown_threshold:
                    alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'type': 'DRAWDOWN',
                        'severity': 'HIGH',
                        'message': f"Critical drawdown level: {metrics['stress_metrics']['worst_drawdown']:.2%}"
                    })
                    
        return alerts
        
    def send_alerts(self, alerts: List[Dict], 
                   email_config: Optional[Dict] = None,
                   webhook_url: Optional[str] = None):
        """
        ارسال هشدارها از طریق ایمیل یا وبهوک
        """
        if not alerts:
            return
            
        # ذخیره هشدارها
        self.alert_history.extend(alerts)
        
        # لاگ کردن هشدارها
        for alert in alerts:
            self.logger.warning(
                f"[{alert['type']}] {alert['symbol']}: {alert['message']}"
            )
            
        # ارسال ایمیل اگر تنظیمات موجود باشد
        if email_config:
            self._send_email_alerts(alerts, email_config)
            
        # ارسال به وبهوک اگر URL موجود باشد
        if webhook_url:
            self._send_webhook_alerts(alerts, webhook_url)
            
    def _send_email_alerts(self, alerts: List[Dict], config: Dict):
        """
        ارسال هشدارها از طریق ایمیل
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']
            msg['Subject'] = "Trading System Alerts"
            
            body = "Trading System Alert Report\n\n"
            for alert in alerts:
                body += f"[{alert['severity']}] {alert['symbol']} - {alert['type']}\n"
                body += f"Message: {alert['message']}\n"
                body += f"Time: {alert['timestamp']}\n\n"
                
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("Alert email sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            
    def _send_webhook_alerts(self, alerts: List[Dict], webhook_url: str):
        """
        ارسال هشدارها به وبهوک
        """
        try:
            import requests
            
            payload = {
                'timestamp': datetime.now().isoformat(),
                'alerts': alerts
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info("Webhook alerts sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
            
    def generate_alert_report(self) -> Dict:
        """
        تولید گزارش از هشدارهای ثبت شده
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alert_history),
            'alerts_by_type': {},
            'alerts_by_severity': {},
            'alerts_by_symbol': {},
            'recent_alerts': self.alert_history[-10:]  # 10 هشدار اخیر
        }
        
        # دسته‌بندی هشدارها
        for alert in self.alert_history:
            # بر اساس نوع
            alert_type = alert['type']
            report['alerts_by_type'][alert_type] = report['alerts_by_type'].get(alert_type, 0) + 1
            
            # بر اساس شدت
            severity = alert['severity']
            report['alerts_by_severity'][severity] = report['alerts_by_severity'].get(severity, 0) + 1
            
            # بر اساس نماد
            symbol = alert['symbol']
            report['alerts_by_symbol'][symbol] = report['alerts_by_symbol'].get(symbol, 0) + 1
            
        # ذخیره گزارش
        with open('debug_logs/alert_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info("Alert report generated successfully")
        return report

if __name__ == "__main__":
    # نمونه استفاده
    alert_system = AlertSystem()
    
    # داده تست
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'close': np.random.normal(100, 2, 1000),
            'volume': np.random.normal(1000000, 200000, 1000)
        })
    }
    
    # تنظیمات ایمیل نمونه
    email_config = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your_email@gmail.com',
        'password': 'your_app_password',
        'from_email': 'your_email@gmail.com',
        'to_email': 'recipient@email.com'
    }
    
    alerts = alert_system.check_market_conditions(test_data)
    alert_system.send_alerts(alerts, email_config=email_config)
    report = alert_system.generate_alert_report() 