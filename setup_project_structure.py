import os
import sys
import json

def create_readme_content():
    return '''# NQTAI (Nexus Quantum Trade AI)

## معرفی
NQTAI یک سیستم معاملاتی پیشرفته مبتنی بر هوش مصنوعی است که قابلیت‌های زیر را ارائه می‌دهد:
- مسیریابی هوشمند سفارشات (Smart Order Routing)
- تحلیل پیشرفته بازار با استفاده از هوش مصنوعی
- مدیریت ریسک و ایمنی پیشرفته
- نظارت خودکار بر فعالیت‌های معاملاتی

## ساختار پروژه
```
NQTAI/
├── modules/          # ماژول‌های اصلی سیستم
├── logs/            # فایل‌های لاگ
├── config/          # تنظیمات و پیکربندی
├── data/            # داده‌های خام و پردازش شده
├── assets/          # فایل‌های استاتیک
├── models_ai/       # مدل‌های هوش مصنوعی
├── tests/           # تست‌ها
├── frontend/        # رابط کاربری
└── docs/            # مستندات
```

## نیازمندی‌ها
- Python 3.8+
- پکیج‌های مورد نیاز در `requirements.txt`

## نصب و راه‌اندازی
1. نصب پکیج‌های مورد نیاز:
```bash
pip install -r requirements.txt
```

2. تنظیم فایل پیکربندی در `config/config.json`

3. اجرای سیستم:
```bash
python main.py
```

## مستندات
مستندات کامل در پوشه `docs` قابل دسترسی است.

## توسعه‌دهندگان
- توسعه اولیه: [نام شما]
- تاریخ شروع: [تاریخ]
'''

def create_requirements_content():
    return '''numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=0.24.2
tensorflow>=2.6.0
statsmodels>=0.13.0
scipy>=1.7.0
logging>=0.5.1.2
python-json-logger>=2.0.2
'''

def create_config_content():
    return {
        "system": {
            "name": "NQTAI",
            "version": "1.0.0",
            "log_level": "INFO",
            "log_dir": "logs/"
        },
        "trading": {
            "max_position_size": 1.0,
            "min_trade_size": 0.01,
            "max_slippage": 0.001,
            "market_impact_threshold": 0.05
        },
        "risk_management": {
            "max_drawdown": 0.1,
            "max_volatility": 0.2,
            "max_exposure": 0.8,
            "max_risk_concentration": 0.3,
            "max_strategy_correlation": 0.7
        },
        "monitoring": {
            "min_daily_trades": 5,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2,
            "monitoring_window": 7,
            "alert_threshold": 0.7,
            "critical_threshold": 0.5,
            "adjustment_cooldown": 4
        }
    }

def create_gitignore_content():
    return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Logs
logs/
*.log

# Local configuration
config/local_config.json

# IDE
.idea/
.vscode/
*.swp
*.swo

# Environment
.env
.venv
venv/
ENV/

# Data and Models
data/raw/*
data/processed/*
models_ai/*.h5
models_ai/*.pkl
'''

def create_project_structure():
    # تعریف مسیر اصلی پروژه
    base_path = "D:/NQTAI"
    
    # لیست تمام دایرکتوری‌های مورد نیاز
    directories = [
        'modules/core',
        'logs',
        'config',
        'data/raw',
        'data/processed',
        'assets',
        'models_ai',
        'tests',
        'frontend',
        'docs'
    ]
    
    try:
        # ایجاد دایرکتوری اصلی اگر وجود نداشته باشد
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            print(f"Created main project directory at: {base_path}")
        
        # ایجاد زیر دایرکتوری‌ها
        for dir_path in directories:
            full_path = os.path.join(base_path, dir_path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"Created directory: {full_path}")
        
        # ایجاد فایل‌های خالی در مسیرهای مربوطه
        core_files = [
            'modules/core/__init__.py',
            'modules/core/safety_manager.py',
            'modules/core/trading_activity_monitor.py',
            'modules/core/smart_order_router.py',
            'modules/core/market_analyzer.py'
        ]
        
        # ایجاد فایل‌های خالی برای کدهای اصلی
        for file_path in core_files:
            full_path = os.path.join(base_path, file_path)
            if not os.path.exists(full_path):
                with open(full_path, 'w', encoding='utf-8') as f:
                    pass  # ایجاد فایل خالی
                print(f"Created file: {full_path}")
        
        # ایجاد فایل‌های پیکربندی و مستندات
        files_to_create = {
            'README.md': create_readme_content(),
            'requirements.txt': create_requirements_content(),
            '.gitignore': create_gitignore_content(),
            'config/config.json': create_config_content()
        }
        
        for file_path, content in files_to_create.items():
            full_path = os.path.join(base_path, file_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(content, f, indent=4, ensure_ascii=False)
                else:
                    f.write(content)
            print(f"Created file with content: {full_path}")
        
        print("\nProject structure created successfully!")
        print(f"Project root: {base_path}")
        print("\nNext steps:")
        print("1. Review and customize config/config.json")
        print("2. Install required packages using: pip install -r requirements.txt")
        print("3. Add your code to the core module files")
        
    except Exception as e:
        print(f"Error creating project structure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    create_project_structure() 