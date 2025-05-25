# D:\AdvancedTradingSystem\modules\system_orchestrator.py
# نسخه کامل با تمام اصلاحات پیشنهادی

import asyncio
import logging
import signal
import os
from pathlib import Path
from typing import Coroutine, Any, Optional, List, Dict, Tuple
from datetime import datetime

import yaml
from pydantic import ValidationError
from dotenv import load_dotenv, find_dotenv

from config.schemas import AppSettings, TimeFrame
from modules.data_nexus import DataNexus
from modules.synapse_ai_core import SynapseAICore
from modules.risk_guardian import RiskGuardian, TradeSignal
from modules.trade_execution_gateway import TradeExecutionGateway
from modules.insight_dashboard import InsightDashboard, DashboardDataProvider

logger = logging.getLogger(__name__)

def _setup_signal_handlers(loop: asyncio.AbstractEventLoop, orchestrator: 'SystemOrchestrator') -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda s=sig: asyncio.create_task(orchestrator.shutdown(signal_name=s.name, graceful=True))
            )
        except (NotImplementedError, RuntimeError) as e:
            if sig == signal.SIGTERM and os.name == 'nt':
                logger.warning(f"SIGTERM handler not fully supported on Windows: {e}")
            elif sig == signal.SIGINT and isinstance(e, RuntimeError) and "Event loop is closed" in str(e).lower():
                logger.debug(f"Could not set SIGINT handler, loop might be closing: {e}")
            else:
                logger.error(f"Failed to set signal handler for {s.name if 's' in locals() else sig}: {e}")
    logger.info("Signal handlers for SIGINT (Ctrl+C) registered (SIGTERM if supported).")


class SystemOrchestrator:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.settings: Optional[AppSettings] = None
        self.running_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        self.data_nexus: Optional[DataNexus] = None
        self.synapse_ai_core: Optional[SynapseAICore] = None
        self.risk_guardian: Optional[RiskGuardian] = None
        self.trade_execution_gateway: Optional[TradeExecutionGateway] = None
        self.dashboard_data_provider: Optional[DashboardDataProvider] = None
        self.insight_dashboard: Optional[InsightDashboard] = None
        
        self._main_trading_loop_task: Optional[asyncio.Task] = None
        self._last_candle_times: Dict[Tuple[str, TimeFrame], datetime] = {}
        self._system_open_positions: Dict[str, Dict[str, Any]] = {}

    def _load_config(self) -> None:
        logger.info(f"Orchestrator: Loading configuration from: {self.config_path}")
        env_vars_loaded = {}
        try:
            env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
            if env_path:
                logger.info(f"Orchestrator: Loading .env file from: {env_path}")
                load_dotenv(env_path, override=True)
                
                env_vars_loaded['MT5_LOGIN'] = os.getenv("MT5_LOGIN")
                env_vars_loaded['MT5_PASSWORD'] = os.getenv("MT5_PASSWORD")
                env_vars_loaded['MT5_SERVER'] = os.getenv("MT5_SERVER")
                env_vars_loaded['MT5_PATH'] = os.getenv("MT5_PATH") 
                env_vars_loaded['WANDB_API_KEY'] = os.getenv("WANDB_API_KEY")

                logger.debug(f"DEBUG_ENV_LOADED: MT5_PATH='{env_vars_loaded.get('MT5_PATH')}'")
                logger.debug(f"DEBUG_ENV_LOADED: MT5_LOGIN='{env_vars_loaded.get('MT5_LOGIN')}'")
            else:
                logger.warning("Orchestrator: .env file not found. Proceeding without it.")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config_from_yaml = yaml.safe_load(f)
            logger.debug(f"DEBUG_YAML: Initial raw_config_from_yaml['metatrader_bridge'].get('mt5_path'): '{raw_config_from_yaml.get('metatrader_bridge', {}).get('mt5_path')}'")

            if 'metatrader_bridge' not in raw_config_from_yaml:
                raw_config_from_yaml['metatrader_bridge'] = {}
            if env_vars_loaded.get('MT5_LOGIN') is not None:
                try: # اطمینان از اینکه لاگین به درستی به int تبدیل می شود
                    raw_config_from_yaml['metatrader_bridge']['mt5_login'] = int(env_vars_loaded['MT5_LOGIN'])
                except ValueError:
                    logger.error(f"Could not convert MT5_LOGIN from ENV ('{env_vars_loaded['MT5_LOGIN']}') to int. Using YAML value if present.")
            if env_vars_loaded.get('MT5_PASSWORD') is not None:
                raw_config_from_yaml['metatrader_bridge']['mt5_password'] = env_vars_loaded['MT5_PASSWORD']
            if env_vars_loaded.get('MT5_SERVER') is not None:
                raw_config_from_yaml['metatrader_bridge']['mt5_server'] = env_vars_loaded['MT5_SERVER']
            if env_vars_loaded.get('MT5_PATH') is not None:
                raw_config_from_yaml['metatrader_bridge']['mt5_path'] = env_vars_loaded['MT5_PATH']
            
            if 'synapse_ai' not in raw_config_from_yaml: raw_config_from_yaml['synapse_ai'] = {}
            if 'training' not in raw_config_from_yaml['synapse_ai']: raw_config_from_yaml['synapse_ai']['training'] = {}
            if env_vars_loaded.get('WANDB_API_KEY') is not None:
                raw_config_from_yaml['synapse_ai']['training']['wandb_api_key'] = env_vars_loaded['WANDB_API_KEY']
            
            logger.debug(f"DEBUG_YAML_MODIFIED: raw_config_from_yaml['metatrader_bridge'].get('mt5_path') after potential ENV override: '{raw_config_from_yaml.get('metatrader_bridge', {}).get('mt5_path')}'")

            self.settings = AppSettings(**raw_config_from_yaml)
            
            logger.debug("--- DEBUG: Values from AppSettings instance (after Pydantic validation) ---")
            if self.settings and self.settings.metatrader_bridge:
                logger.debug(f"DEBUG_PYDANTIC: self.settings.metatrader_bridge.mt5_path: '{self.settings.metatrader_bridge.mt5_path}'")
            logger.debug("--- End DEBUG PYDANTIC ---")

            logger.info("Orchestrator: Configuration loaded and validated successfully.")
        except FileNotFoundError:
            logger.critical(f"Orchestrator: Config file not found: {self.config_path}")
            raise
        except ValidationError as e:
            logger.critical(f"Orchestrator: Config validation error:\n{e}")
            logger.critical(f"Data passed to AppSettings (from YAML, potentially modified by ENV): {raw_config_from_yaml if 'raw_config_from_yaml' in locals() else 'Not loaded'}")
            raise
        except Exception as e:
            logger.critical(f"Orchestrator: Error loading configuration: {e}", exc_info=True)
            raise

    def _setup_logging(self) -> None:
        if not self.settings or not hasattr(self.settings, 'logging'):
            logging.basicConfig(level=logging.INFO, format="%(asctime)s -FALLBACK- %(levelname)s - %(message)s")
            logger.error("Orchestrator: Cannot setup logging from settings. Using default basicConfig.")
            return

        log_cfg = self.settings.logging
        log_file_full_path = None
        if log_cfg.log_file_path:
            log_path_obj = Path(log_cfg.log_file_path)
            if not log_path_obj.is_absolute() and self.config_path.is_file():
                log_file_full_path = (self.config_path.parent / log_path_obj).resolve()
            else:
                log_file_full_path = log_path_obj.resolve()
            
            try:
                log_file_full_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e_mkdir:
                logger.error(f"Could not create log directory {log_file_full_path.parent}: {e_mkdir}")
                log_file_full_path = None

        root_logger_obj = logging.getLogger()
        for handler in root_logger_obj.handlers[:]:
            root_logger_obj.removeHandler(handler)
            handler.close()

        handlers_list = [logging.StreamHandler()]
        if log_file_full_path:
            try:
                file_handler = logging.FileHandler(log_file_full_path, encoding='utf-8')
                file_handler.setFormatter(logging.Formatter(log_cfg.format))
                handlers_list.append(file_handler)
            except OSError as e_fhandler:
                logger.error(f"Could not create file handler for {log_file_full_path}: {e_fhandler}")
        
        logging.basicConfig(level=log_cfg.level.upper(), format=log_cfg.format, handlers=handlers_list, force=True)
        logger.info(f"Orchestrator: Logging setup complete. Level: {log_cfg.level.upper()}")
        if log_file_full_path and any(isinstance(h, logging.FileHandler) for h in handlers_list):
            logger.info(f"Orchestrator: Logging to file: {log_file_full_path}")

    async def _initialize_modules(self) -> None:
        if not self.settings:
            logger.critical("Orchestrator: Cannot initialize modules, settings not loaded.")
            return
        
        logger.info("Orchestrator: Initializing system modules...")
        try:
            logger.debug("Orchestrator: Initializing TradeExecutionGateway...")
            if not self.settings.metatrader_bridge:
                logger.critical("MetatraderBridgeSettings are missing from AppSettings!")
                raise ValueError("MetatraderBridgeSettings missing.")
            self.trade_execution_gateway = TradeExecutionGateway(self.settings.metatrader_bridge)
            if not await self.trade_execution_gateway.initialize():
                logger.error("Orchestrator: TradeExecutionGateway initialization failed.")
            else:
                logger.info("Orchestrator: TradeExecutionGateway initialized.")

            logger.debug("Orchestrator: Initializing DataNexus...")
            if not self.settings.data_nexus or not self.settings.metatrader_bridge:
                 logger.critical("DataNexusSettings or MetatraderBridgeSettings missing for DataNexus init!")
                 raise ValueError("DataNexusSettings or MetatraderBridgeSettings missing.")
            self.data_nexus = DataNexus(self.settings.data_nexus, self.settings.metatrader_bridge)
            if not await self.data_nexus.initialize():
                logger.error("Orchestrator: DataNexus initialization failed.")
            else:
                logger.info("Orchestrator: DataNexus initialized.")

            logger.debug("Orchestrator: Initializing RiskGuardian...")
            if not self.settings.risk_guardian:
                 logger.critical("RiskGuardianSettings missing!")
                 raise ValueError("RiskGuardianSettings missing.")
            self.risk_guardian = RiskGuardian(self.settings.risk_guardian)
            logger.info("Orchestrator: RiskGuardian initialized.")
            
            if self.trade_execution_gateway and self.trade_execution_gateway._is_connected:
                acc_info = await self.trade_execution_gateway.get_account_info()
                if acc_info:
                    open_pos = await self.trade_execution_gateway.get_open_positions() or []
                    self.risk_guardian.update_account_state(acc_info.get('balance',0.0), acc_info.get('equity',0.0), open_pos)
                    self.risk_guardian.new_day_reset(acc_info.get('balance',0.0))
                else:
                    logger.warning("Orchestrator: Could not get initial account info for RiskGuardian during init.")
            
            if self.data_nexus: 
                logger.debug("Orchestrator: Initializing SynapseAICore...")
                if not self.settings.synapse_ai:
                    logger.critical("SynapseAISettings missing!")
                    raise ValueError("SynapseAISettings missing.")
                self.synapse_ai_core = SynapseAICore(self.settings.synapse_ai, self.settings, self.data_nexus)
                logger.info("Orchestrator: SynapseAI Core initialized.")
            else:
                logger.error("Orchestrator: DataNexus not initialized, SynapseAI Core cannot be initialized.")

            if self.settings.insight_dashboard and self.settings.insight_dashboard.enabled:
                logger.debug("Orchestrator: Initializing InsightDashboard components...")
                self.dashboard_data_provider = DashboardDataProvider(orchestrator_ref=self)
                self.insight_dashboard = InsightDashboard(self.settings.insight_dashboard, self.dashboard_data_provider)
                logger.info("Orchestrator: InsightDashboard components initialized.")
            else:
                 logger.info("Orchestrator: InsightDashboard is disabled in settings.")

        except Exception as e:
            logger.critical(f"Orchestrator: Critical error during module initialization: {e}", exc_info=True)
            raise
        logger.info("Orchestrator: System modules initialization process complete.")

    async def _start_module_tasks(self) -> None:
        logger.info("Orchestrator: Starting background module tasks...")
        if self.data_nexus and hasattr(self.data_nexus, '_mt5_initialized') and self.data_nexus._mt5_initialized: # بررسی دقیق تر
            if asyncio.iscoroutinefunction(self.data_nexus.realtime_data_feed_loop) or \
               (hasattr(self.data_nexus.realtime_data_feed_loop, '__call__') and \
                asyncio.iscoroutine(self.data_nexus.realtime_data_feed_loop())): # برای متدی که ممکن است پارامتر نگیرد
                 self.running_tasks.append(asyncio.create_task(self.data_nexus.realtime_data_feed_loop(), name="DataNexusFeedLoop"))
                 logger.info("Orchestrator: DataNexus realtime_data_feed_loop task started.")
            else: 
                 logger.error("DataNexus.realtime_data_feed_loop is not an async coroutine function!")
        elif self.data_nexus:
            logger.warning("Orchestrator: DataNexus realtime_data_feed_loop NOT started (MT5 not initialized in DataNexus).")
        else:
            logger.warning("Orchestrator: DataNexus not available, cannot start data feed loop.")
        
        if self.synapse_ai_core and self.settings.run_initial_training:
            logger.info("Orchestrator: Starting initial AI model training scheduling...")
            training_task = asyncio.create_task(self.synapse_ai_core.schedule_training_for_all_configured(), name="AI_Initial_Training_Scheduler")
            self.running_tasks.append(training_task)
            logger.info("Orchestrator: AI initial training scheduling task created.")
        elif self.synapse_ai_core:
            logger.info("Orchestrator: Attempting to load existing AI models...")
            async def load_all_models_wrapper(): 
                load_model_tasks_list = []
                if self.data_nexus and getattr(self.data_nexus, 'settings', None):
                    for symbol_str_load in self.data_nexus.settings.symbols:
                        for timeframe_enum_obj_load in self.data_nexus.settings.timeframes:
                            load_model_tasks_list.append(
                                asyncio.to_thread(self.synapse_ai_core.load_model, symbol_str_load, timeframe_enum_obj_load)
                            )
                    if load_model_tasks_list:
                        await asyncio.gather(*load_model_tasks_list, return_exceptions=True)
            await load_all_models_wrapper() 
            logger.info("Orchestrator: AI model loading attempt finished.")

        if self.insight_dashboard and self.settings.insight_dashboard and self.settings.insight_dashboard.enabled:
            logger.info("Orchestrator: Starting InsightDashboard server in background...")
            async def run_dashboard_wrapper_task(): 
                try:
                    logger.info(f"InsightDashboard server starting on http://{self.settings.insight_dashboard.host}:{self.settings.insight_dashboard.port}")
                    await asyncio.to_thread(self.insight_dashboard.run) 
                    logger.info("InsightDashboard server stopped.")
                except Exception as e_dash_run:
                    logger.error(f"InsightDashboard server error: {e_dash_run}", exc_info=True)
            asyncio.create_task(run_dashboard_wrapper_task(), name="InsightDashboardServerTask") 
            logger.info("Orchestrator: InsightDashboard server task creation initiated.")

        self._main_trading_loop_task = asyncio.create_task(self._trading_loop(), name="MainTradingLoop")
        self.running_tasks.append(self._main_trading_loop_task)
        logger.info("Orchestrator: Main trading loop task started.")
        logger.info(f"Orchestrator: Total {len(self.running_tasks)} core background tasks managed by orchestrator list.")

    async def _trading_loop(self):
        logger.info("Orchestrator: Trading loop started. Monitoring for trading opportunities...")
        if not all([self.settings, self.data_nexus, self.synapse_ai_core, self.risk_guardian, self.trade_execution_gateway]):
            logger.critical("Orchestrator: One or more critical modules are not initialized. Trading loop cannot start.")
            return

        # اطمینان از اینکه DataNexus settings موجود است
        if not (self.data_nexus and hasattr(self.data_nexus, 'settings') and self.data_nexus.settings):
            logger.critical("DataNexus or its settings not available for trading loop.")
            return

        for symbol_tl in self.data_nexus.settings.symbols:
            for timeframe_tl in self.data_nexus.settings.timeframes:
                self._last_candle_times[(symbol_tl, timeframe_tl)] = datetime.min

        try:
            while not self.shutdown_event.is_set():
                current_time_now = datetime.now()
                
                if not self.trade_execution_gateway or not getattr(self.trade_execution_gateway, '_is_connected', False):
                    logger.warning("Trading Loop: TradeExecutionGateway not connected. Attempting reconnect or skipping cycle.")
                    if self.trade_execution_gateway and hasattr(self.trade_execution_gateway, 'initialize') and not await self.trade_execution_gateway.initialize():
                         logger.error("Trading Loop: Reconnect to TEG failed. Sleeping.")
                         await asyncio.sleep(getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5) * 2)
                         continue
                    elif not self.trade_execution_gateway:
                         logger.error("Trading Loop: TEG is None. Cannot proceed.")
                         await asyncio.sleep(getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5) * 5)
                         continue
                
                if not self.risk_guardian: # بررسی وجود RiskGuardian
                    logger.error("Trading Loop: RiskGuardian not initialized. Skipping cycle.")
                    await asyncio.sleep(getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5))
                    continue

                account_info_loop = await self.trade_execution_gateway.get_account_info()
                if not account_info_loop:
                    logger.error("Trading Loop: Could not get account info. Skipping cycle.")
                    await asyncio.sleep(getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5))
                    continue
                
                broker_open_positions_loop = await self.trade_execution_gateway.get_open_positions() or []
                self.risk_guardian.update_account_state(account_info_loop.get('balance',0.0), account_info_loop.get('equity',0.0), broker_open_positions_loop)
                
                self._system_open_positions.clear()
                for pos_broker in broker_open_positions_loop:
                    if 'symbol' in pos_broker: # اطمینان از وجود کلید symbol
                         self._system_open_positions[pos_broker['symbol']] = pos_broker

                if self.risk_guardian.check_circuit_breaker(account_info_loop.get('balance',0.0), account_info_loop.get('equity',0.0)):
                    if self.settings.risk_guardian.circuit_breaker_on_max_drawdown and self.risk_guardian.circuit_breaker_active:
                        logger.critical("TRADING LOOP: CIRCUIT BREAKER ACTIVE! Attempting to close all positions.")
                        await self.trade_execution_gateway.close_all_open_positions(comment="CB_TradingLoop_CloseAll_v3")
                    await asyncio.sleep(max(60, getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5)))
                    continue

                for symbol_iter in self.data_nexus.settings.symbols:
                    for timeframe_iter in self.data_nexus.settings.timeframes:
                        key_iter = (symbol_iter, timeframe_iter)
                        
                        if not hasattr(self.data_nexus, 'get_market_data'):
                             logger.error("CRITICAL: DataNexus instance has no 'get_market_data' method!")
                             await asyncio.sleep(10) # انتظار زیاد برای جلوگیری از لاگ های مکرر
                             continue

                        latest_candle_df_loop = self.data_nexus.get_market_data(symbol_iter, timeframe_iter, num_candles=1)
                        if latest_candle_df_loop is None or latest_candle_df_loop.empty:
                            continue
                        
                        current_candle_time_loop = latest_candle_df_loop.index[-1].to_pydatetime()
                        
                        if current_candle_time_loop > self._last_candle_times.get(key_iter, datetime.min):
                            logger.info(f"Trading Loop: New candle for {symbol_iter} {timeframe_iter.value} at {current_candle_time_loop}.")
                            self._last_candle_times[key_iter] = current_candle_time_loop

                            if symbol_iter in self._system_open_positions:
                                logger.debug(f"Trading Loop: Position open for {symbol_iter}. Skipping new entry signal. (Exit logic TODO)")
                                continue 
                            
                            if not self.synapse_ai_core: # بررسی وجود SynapseAICore
                                logger.error(f"Trading Loop: SynapseAICore not initialized. Cannot get prediction for {symbol_iter}.")
                                continue

                            logger.debug(f"Trading Loop: Requesting prediction for {symbol_iter} {timeframe_iter.value}...")
                            ai_env_win_size = self.settings.synapse_ai.env_window_size
                            market_data_for_ai_loop = self.data_nexus.get_market_data(symbol_iter, timeframe_iter, num_candles=ai_env_win_size + 5)
                            if market_data_for_ai_loop is None or len(market_data_for_ai_loop) < ai_env_win_size:
                                logger.warning(f"Trading Loop: Not enough data for AI prediction for {symbol_iter} {timeframe_iter.value}. Need >= {ai_env_win_size}, Got {len(market_data_for_ai_loop) if market_data_for_ai_loop is not None else 0}")
                                continue

                            prediction_result_loop = self.synapse_ai_core.get_prediction(symbol_iter, timeframe_iter, market_data_df=market_data_for_ai_loop)

                            if prediction_result_loop:
                                ai_action_code_loop, _ = prediction_result_loop
                                logger.info(f"Trading Loop: AI for {symbol_iter}-{timeframe_iter.value} predicted action code: {ai_action_code_loop}")

                                trade_action_str_loop = None
                                if ai_action_code_loop == 1: trade_action_str_loop = "BUY"
                                elif ai_action_code_loop == 2: trade_action_str_loop = "SELL"

                                if trade_action_str_loop:
                                    symbol_details_loop = await self.trade_execution_gateway.get_symbol_info(symbol_iter)
                                    if not symbol_details_loop:
                                        logger.error(f"Trading Loop: Could not get symbol info for {symbol_iter}."); continue
                                    
                                    entry_price_guess_loop = symbol_details_loop.get('ask') if trade_action_str_loop == "BUY" else symbol_details_loop.get('bid')
                                    if not entry_price_guess_loop:
                                        logger.error(f"Trading Loop: Could not get ask/bid for {symbol_iter}."); continue

                                    raw_signal_loop = TradeSignal(
                                        symbol=symbol_iter, action=trade_action_str_loop, order_type="MARKET",
                                        price=entry_price_guess_loop, signal_source=f"AI_{timeframe_iter.value}"
                                    )
                                    logger.info(f"Trading Loop: Raw signal: {raw_signal_loop}")

                                    validated_signal_loop, reason_loop = self.risk_guardian.validate_new_trade_signal(
                                        raw_signal_loop, account_info_loop['balance'], broker_open_positions_loop, symbol_details_loop
                                    )

                                    if validated_signal_loop:
                                        logger.info(f"Trading Loop: Signal {validated_signal_loop.symbol} ({validated_signal_loop.action}) VALIDATED: {reason_loop}. Vol: {validated_signal_loop.volume_lots}, SL: {validated_signal_loop.stop_loss_price}")
                                        execution_result_loop = await self.trade_execution_gateway.execute_trade_signal(validated_signal_loop)
                                        if execution_result_loop and execution_result_loop.get("success"):
                                            logger.info(f"Trading Loop: Trade EXECUTED for {symbol_iter}: Ticket {execution_result_loop.get('order_ticket')}, Deal: {execution_result_loop.get('deal_ticket')}")
                                            self._system_open_positions[symbol_iter] = {"ticket": execution_result_loop.get("order_ticket") or execution_result_loop.get("deal_ticket"), "type": validated_signal_loop.action, "volume": validated_signal_loop.volume_lots, "price_open": execution_result_loop.get("details_raw", {}).get("price", entry_price_guess_loop)}
                                        else:
                                            logger.error(f"Trading Loop: Trade execution FAILED for {symbol_iter}: {execution_result_loop.get('message') if execution_result_loop else 'Unknown TEG error'}")
                                    else:
                                        logger.warning(f"Trading Loop: AI Signal for {symbol_iter} REJECTED by RiskGuardian: {reason_loop}")
                        await asyncio.sleep(0.01) 
                
                await asyncio.sleep(max(1, getattr(self.settings.data_nexus, 'realtime_update_interval_seconds', 5) // 2 ))
        except asyncio.CancelledError:
            logger.info("Orchestrator: Trading loop was cancelled.")
        except Exception as e_loop: 
            logger.critical(f"Orchestrator: Critical error in trading loop: {e_loop}", exc_info=True)
            await self.shutdown(graceful=False, from_loop_exception=True) 
        finally: 
            logger.info("Orchestrator: Trading loop finished.")

    async def run(self) -> None:
        try:
            self._load_config()
            self._setup_logging()
            logger.info(f"Orchestrator: Starting {self.settings.app_name if self.settings else 'App'} v{self.settings.version if self.settings else 'N/A'}")
            await self._initialize_modules()
            await self._start_module_tasks()
            
            if self.running_tasks:
                wait_for_these_tasks = self.running_tasks + [asyncio.create_task(self.shutdown_event.wait(), name="ShutdownEventWaitTask")]
                done_tasks, pending_tasks = await asyncio.wait(wait_for_these_tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task_done in done_tasks:
                    task_name_done = task_done.get_name() if hasattr(task_done, 'get_name') else "UnnamedTask"
                    exc_done = task_done.exception()
                    if exc_done:
                        logger.error(f"Orchestrator: Core task ({task_name_done}) failed: {exc_done}", exc_info=exc_done)
                        if task_done in self.running_tasks:
                             await self.shutdown(graceful=False, from_loop_exception=isinstance(exc_done, Exception))
                        return 
                    elif task_done in self.running_tasks:
                        logger.info(f"Orchestrator: Core task {task_name_done} completed normally.")
                
                if not self.shutdown_event.is_set():
                    remaining_managed_tasks_list = [t for t in self.running_tasks if not t.done()]
                    if remaining_managed_tasks_list:
                        logger.info(f"Orchestrator: Waiting for {len(remaining_managed_tasks_list)} remaining managed tasks or shutdown signal...")
                        await asyncio.wait(remaining_managed_tasks_list + [asyncio.create_task(self.shutdown_event.wait())], return_when=asyncio.FIRST_COMPLETED)
            else:
                logger.info("Orchestrator: No primary background tasks running. Waiting for shutdown signal.")
                await self.shutdown_event.wait()

        except Exception as e_run_setup:
            logger.critical(f"Orchestrator: Critical error in run setup or main wait: {e_run_setup}", exc_info=True)
            await self.shutdown(graceful=False, from_loop_exception=True)
        finally:
            if not self.shutdown_event.is_set():
                logger.info("Orchestrator: Run method finished unexpectedly. Initiating final shutdown.")
                await self.shutdown(graceful=False)
            
            app_name_at_finish = getattr(self.settings, 'app_name', "System") if self.settings else "System"
            logger.info(f"Orchestrator: {app_name_at_finish} run cycle fully finished.")

    async def shutdown(self, signal_name: Optional[str] = None, graceful: bool = True, from_loop_exception: bool = False) -> None:
        if self.shutdown_event.is_set() and not from_loop_exception: 
            logger.info("Orchestrator: Shutdown already in progress.")
            return

        if signal_name: logger.info(f"Orchestrator: Shutdown initiated by signal: {signal_name} (graceful: {graceful})...")
        else: logger.info(f"Orchestrator: Shutdown initiated (graceful: {graceful}, from_loop_exception: {from_loop_exception})...")
        
        self.shutdown_event.set()

        tasks_to_await = []
        if self.running_tasks:
            logger.info(f"Orchestrator: Cancelling {len(self.running_tasks)} managed task(s)...")
            for task_to_cancel in self.running_tasks:
                if task_to_cancel and not task_to_cancel.done():
                    task_to_cancel.cancel()
                    tasks_to_await.append(task_to_cancel)
        
        if tasks_to_await:
            logger.info(f"Orchestrator: Waiting for {len(tasks_to_await)} task(s) to finish cancellation...")
            results = await asyncio.gather(*tasks_to_await, return_exceptions=True)
            for i, res in enumerate(results):
                task_name_gathered = tasks_to_await[i].get_name() if hasattr(tasks_to_await[i], 'get_name') else f"Task-{i}"
                if isinstance(res, asyncio.CancelledError):
                    logger.info(f"Task {task_name_gathered} was cancelled successfully.")
                elif isinstance(res, Exception):
                    logger.error(f"Task {task_name_gathered} raised an exception during cancellation or completion: {res}", exc_info=res if isinstance(res, BaseException) else None)

        self.running_tasks.clear()
        logger.info("Orchestrator: All managed tasks processed during shutdown.")

        logger.info("Orchestrator: Shutting down modules...")
        if self.synapse_ai_core and hasattr(self.synapse_ai_core, 'shutdown'):
            logger.info("Shutting down SynapseAI Core...")
            await self.synapse_ai_core.shutdown()
        
        if self.risk_guardian and hasattr(self.risk_guardian, 'shutdown'): 
            logger.info("Shutting down RiskGuardian...")
            if asyncio.iscoroutinefunction(self.risk_guardian.shutdown):
                await self.risk_guardian.shutdown()
            else:
                await asyncio.to_thread(self.risk_guardian.shutdown)
        
        if self.data_nexus and hasattr(self.data_nexus, 'shutdown'):
            logger.info("Shutting down DataNexus...")
            await self.data_nexus.shutdown()
        
        if self.trade_execution_gateway and hasattr(self.trade_execution_gateway, 'shutdown'):
            logger.info("Shutting down TradeExecutionGateway (disconnecting MT5 if connected)...")
            if graceful and hasattr(self.trade_execution_gateway, 'get_open_positions') and \
               hasattr(self.trade_execution_gateway, 'close_all_open_positions') and \
               getattr(self.trade_execution_gateway, '_is_connected', False):
                 try:
                     open_pos_on_shutdown = await self.trade_execution_gateway.get_open_positions() or []
                     if open_pos_on_shutdown:
                        logger.info(f"Graceful shutdown: Closing {len(open_pos_on_shutdown)} open position(s)...")
                        await self.trade_execution_gateway.close_all_open_positions(comment="SystemShutdown")
                 except Exception as e_close_pos:
                     logger.error(f"Error closing positions during graceful shutdown: {e_close_pos}")
            await self.trade_execution_gateway.shutdown()
        
        if self.insight_dashboard and hasattr(self.insight_dashboard, 'shutdown'):
            logger.info("Shutting down InsightDashboard...")
            if asyncio.iscoroutinefunction(self.insight_dashboard.shutdown):
                await self.insight_dashboard.shutdown()
            else:
                await asyncio.to_thread(self.insight_dashboard.shutdown)

        logger.info("Orchestrator: All modules shutdown process initiated/completed.")