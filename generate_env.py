import os
from pathlib import Path

# --- تنظیمات پروژه ---
DRIVE_LETTER = "D"  # یا هر درایوی که پروژه شما در آن است
PROJECT_NAME = "AdvancedTradingSystem"
ROOT_PROJECT_DIR = Path(f"{DRIVE_LETTER}:/{PROJECT_NAME}")

# --- اطلاعات حساب دمو (از ورودی شما) ---
MT5_LOGIN_DEMO = "92985590"
MT5_PASSWORD_DEMO = "1dYjHdT*"  # هشدار: به اشتراک گذاری رمز عبور امن نیست
MT5_SERVER_DEMO = "MetaQuotes-Demo"

# --- مسیر پیش فرض برای متاتریدر (در صورت نیاز ویرایش کنید) ---
DEFAULT_MT5_PATH_WINDOWS = "C:/Program Files/MetaTrader 5/terminal64.exe"

def create_env_file(project_root: Path, login: str, password: str, server: str, mt5_path: str):
    """
    فایل .env را در مسیر مشخص شده با اطلاعات داده شده ایجاد یا بازنویسی می کند.
    """
    env_file_path = project_root / ".env"
    
    # اصلاح برای جلوگیری از SyntaxError با بک اسلش در f-string
    formatted_mt5_path = str(Path(mt5_path)).replace('\\', '/') # تبدیل به Path و سپس replace

    content = f"# MetaTrader 5 Credentials\n"
    content += f"MT5_LOGIN={login}\n"
    content += f"MT5_PASSWORD=\"{password}\"  # Ensure password is in quotes if it contains special characters\n"
    content += f"MT5_SERVER=\"{server}\"\n"
    content += f"\n"
    content += f"# Path to MetaTrader 5 terminal executable\n"
    content += f"# Please verify this path is correct for your system\n"
    content += f"MT5_PATH_TERMINAL=\"{formatted_mt5_path}\"\n" # استفاده از متغیر فرمت شده
    content += f"\n"
    content += f"# Weights & Biases API Key (optional)\n"
    content += f"# WANDB_API_KEY=\"YOUR_WANDB_API_KEY_HERE\"\n"

    try:
        with open(env_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Successfully created/updated .env file at: {env_file_path}")
        print("IMPORTANT: Please review the .env file for correctness, especially MT5_PATH_TERMINAL.")
        if password != "YOUR_MT5_PASSWORD_HERE":
            print("\n" + "="*60)
            print("SECURITY WARNING:")
            print("The .env file now contains sensitive login credentials.")
            print("Ensure this file is NOT committed to version control (e.g., Git).")
            print("It is recommended to add '.env' to your .gitignore file.")
            print("Consider changing your demo account password if security is a concern.")
            print("="*60 + "\n")
    except IOError as e:
        print(f"Error: Could not write .env file at {env_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if not ROOT_PROJECT_DIR.exists():
        print(f"Error: Project root directory not found at '{ROOT_PROJECT_DIR}'.")
        print("Please ensure the DRIVE_LETTER and PROJECT_NAME variables are set correctly,")
        print("or that the project structure has been created.")
    else:
        print("\n--- MetaTrader 5 Path Configuration ---")
        print(f"The script will use a default path for MT5 terminal: '{DEFAULT_MT5_PATH_WINDOWS}'")
        user_mt5_path_input = input(f"If this is incorrect, please enter the correct full path to terminal64.exe (or terminal.exe), otherwise press Enter to use default: ").strip()
        
        mt5_terminal_path_to_use = DEFAULT_MT5_PATH_WINDOWS
        if user_mt5_path_input:
            path_candidate = Path(user_mt5_path_input)
            if path_candidate.is_file() and (path_candidate.name == "terminal64.exe" or path_candidate.name == "terminal.exe"):
                mt5_terminal_path_to_use = str(path_candidate)
                print(f"Using user-provided MT5 path: {mt5_terminal_path_to_use}")
            else:
                print(f"Warning: The path you entered ('{user_mt5_path_input}') does not seem to be a valid terminal executable. Falling back to default.")
        else:
             print(f"Using default MT5 path: {mt5_terminal_path_to_use}")

        create_env_file(
            project_root=ROOT_PROJECT_DIR,
            login=MT5_LOGIN_DEMO,
            password=MT5_PASSWORD_DEMO,
            server=MT5_SERVER_DEMO,
            mt5_path=mt5_terminal_path_to_use
        )
        
        print("\n--- Next Steps ---")
        print(f"1. Verify the content of the file: {ROOT_PROJECT_DIR / '.env'}")
        print(f"   Especially ensure that MT5_PATH_TERMINAL points to your actual MetaTrader 5 installation.")
        print(f"2. If you haven't already, run: pip install -r requirements.txt")
        print(f"3. Then try running your main application: python main.py")