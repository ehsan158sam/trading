# D:\AdvancedTradingSystem\main.py

import asyncio
import argparse
from pathlib import Path
import logging # برای لاگ اولیه قبل از تنظیمات کامل

# --- اضافه کردن این بخش برای مدیریت مسیر ---
import sys
# این خط مسیر پوشه ای که main.py در آن قرار دارد (یعنی ریشه پروژه)
# را به sys.path اضافه می کند.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- پایان بخش اضافه شده ---

from modules.system_orchestrator import SystemOrchestrator, _setup_signal_handlers # ایمپورت system_orchestrator

# لاگر اولیه برای مرحله بوت استرپ
bootstrap_logger = logging.getLogger("bootstrap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Advanced Modular Trading System Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml", # مسیر پیش فرض نسبت به ریشه پروژه
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    config_file_path = Path(args.config)
    if not config_file_path.is_absolute():
        # اگر مسیر نسبی است، آن را نسبت به ریشه پروژه در نظر بگیرید
        config_file_path = PROJECT_ROOT / config_file_path # استفاده از PROJECT_ROOT تعریف شده
        config_file_path = config_file_path.resolve()


    if not config_file_path.exists():
        bootstrap_logger.error(f"Configuration file not found: {config_file_path}")
        return

    bootstrap_logger.info(f"Using configuration file: {config_file_path}")

    orchestrator = SystemOrchestrator(config_path=config_file_path)
    loop = asyncio.get_event_loop()

    _setup_signal_handlers(loop, orchestrator)

    try:
        loop.run_until_complete(orchestrator.run())
    except KeyboardInterrupt:
        bootstrap_logger.info("KeyboardInterrupt received. Shutting down...")
        # اطمینان از اینکه shutdown فراخوانی می شود حتی اگر run_until_complete قطع شود
        if not orchestrator.shutdown_event.is_set():
            loop.run_until_complete(orchestrator.shutdown("KeyboardInterrupt"))
    except Exception as e:
        bootstrap_logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        if not orchestrator.shutdown_event.is_set():
            loop.run_until_complete(orchestrator.shutdown("UnhandledException", graceful=False))
    finally:
        bootstrap_logger.info("Application finished.")


if __name__ == "__main__":
    main()