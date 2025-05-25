import os
import json
import shutil
from pathlib import Path

class SystemSetup:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.required_dirs = [
            'debug_logs',
            'tests/test_data',
            'modules/core',
            'modules/debug',
            'modules/ui/components',
            'config'
        ]
        self.config_template = {
            "system": {
                "environment": "development",
                "log_level": "DEBUG"
            },
            "risk_management": {
                "max_position_size": 0.1,
                "max_drawdown_limit": 0.2,
                "var_confidence": 0.95
            },
            "alerts": {
                "email": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "your_email@gmail.com",
                    "password": "your_app_password",
                    "from_email": "your_email@gmail.com",
                    "to_email": "recipient@email.com"
                },
                "thresholds": {
                    "volatility": 0.3,
                    "drawdown": 0.15,
                    "volume_spike": 3.0,
                    "price_change": 0.1
                }
            }
        }
        
    def create_directories(self):
        """ایجاد پوشه‌های مورد نیاز"""
        print("Creating required directories...")
        
        for dir_path in self.required_dirs:
            full_path = os.path.join(self.base_dir, dir_path)
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"✗ Error creating directory {dir_path}: {str(e)}")
                
    def create_config_files(self):
        """ایجاد فایل‌های پیکربندی"""
        print("\nCreating configuration files...")
        
        # ایجاد config.example.json
        config_example_path = os.path.join(self.base_dir, 'config', 'config.example.json')
        try:
            with open(config_example_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_template, f, indent=4)
            print("✓ Created config.example.json")
        except Exception as e:
            print(f"✗ Error creating config.example.json: {str(e)}")
            
        # ایجاد logging.conf
        logging_conf = """[loggers]
keys=root,trading_system

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=DEBUG
handlers=consoleHandler

[logger_trading_system]
level=DEBUG
handlers=fileHandler
qualname=trading_system
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('debug_logs/trading_system.log', 'a')

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
"""
        
        logging_conf_path = os.path.join(self.base_dir, 'config', 'logging.conf')
        try:
            with open(logging_conf_path, 'w') as f:
                f.write(logging_conf)
            print("✓ Created logging.conf")
        except Exception as e:
            print(f"✗ Error creating logging.conf: {str(e)}")
            
    def create_requirements(self):
        """ایجاد فایل requirements.txt"""
        print("\nCreating requirements.txt...")
        
        requirements = """# Core dependencies
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=0.24.0

# Data handling and analysis
matplotlib>=3.4.0
seaborn>=0.11.0
requests>=2.26.0

# Configuration and logging
python-dotenv>=0.19.0
pyyaml>=5.4.1

# Testing
pytest>=6.2.5
pytest-cov>=2.12.1

# UI components
streamlit>=1.0.0
plotly>=5.3.1

# Development tools
black>=21.9b0
flake8>=3.9.2
mypy>=0.910
"""
        
        try:
            with open(os.path.join(self.base_dir, 'requirements.txt'), 'w') as f:
                f.write(requirements)
            print("✓ Created requirements.txt")
        except Exception as e:
            print(f"✗ Error creating requirements.txt: {str(e)}")
            
    def create_gitignore(self):
        """ایجاد فایل .gitignore"""
        print("\nCreating .gitignore...")
        
        gitignore_content = """# Python
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

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs and databases
*.log
*.sqlite
debug_logs/

# Config
config/config.json
.env

# Test coverage
.coverage
htmlcov/

# Distribution
dist/
build/
"""
        
        try:
            with open(os.path.join(self.base_dir, '.gitignore'), 'w') as f:
                f.write(gitignore_content)
            print("✓ Created .gitignore")
        except Exception as e:
            print(f"✗ Error creating .gitignore: {str(e)}")
            
    def setup_system(self):
        """راه‌اندازی کامل سیستم"""
        print("Starting system setup...\n")
        
        self.create_directories()
        self.create_config_files()
        self.create_requirements()
        self.create_gitignore()
        
        print("\nSystem setup completed successfully!")
        print("\nNext steps:")
        print("1. Create and activate a virtual environment:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/macOS")
        print("   venv\\Scripts\\activate    # Windows")
        print("2. Install requirements:")
        print("   pip install -r requirements.txt")
        print("3. Copy and configure config file:")
        print("   cp config/config.example.json config/config.json")
        print("4. Edit config/config.json with your settings")
        print("5. Run the system:")
        print("   python main.py")

if __name__ == "__main__":
    setup = SystemSetup()
    setup.setup_system() 