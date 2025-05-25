# D:\AdvancedTradingSystem\modules\data_nexus.py
# Version: Corrected volume column renaming and selection

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime 

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pandas_ta as pta

try:
    from config.schemas import DataNexusSettings, TimeFrame, MetaTraderBridgeSettings
except ImportError:
    print("CRITICAL WARNING (data_nexus.py): Could not import schemas from config.")
    class DataNexusSettings: historical_data_path="./data_fallback"; processed_data_path=None; symbols=["FALLBACK"]; timeframes=[]; max_historical_candles=100; feature_engineering_enabled=False; indicator_settings={}; realtime_update_interval_seconds=60; env_window_size=60
    class MetaTraderBridgeSettings: pass
    class TimeFrame(str): M1="M1" 

logger = logging.getLogger(__name__)

OHLCV_COLS = ['time', 'open', 'high', 'low', 'close', 'volume']
OHLCV_DTYPES = {'time': 'datetime64[ns]', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': np.int64}


class DataNexus:
    # ... (بخش __init__ و سایر متدها تا _standardize_ohlcv_df بدون تغییر) ...
    def __init__(self, settings: DataNexusSettings, mt5_settings: MetaTraderBridgeSettings):
        self.settings = settings
        self.mt5_settings = mt5_settings
        self._data_store: Dict[str, Dict[TimeFrame, pd.DataFrame]] = {}
        self._mt5_initialized: bool = False
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()

        try:
            hdp_setting = getattr(settings, 'historical_data_path', None)
            pdp_setting = getattr(settings, 'processed_data_path', None)

            hdp_str = str(hdp_setting) if hdp_setting else "./data/dn_hist_fallback"
            self.historical_path = Path(hdp_str).resolve()
            self.historical_path.mkdir(parents=True, exist_ok=True)

            pdp_str = str(pdp_setting) if pdp_setting else None
            if pdp_str:
                self.processed_path = Path(pdp_str).resolve()
                self.processed_path.mkdir(parents=True, exist_ok=True)
            else:
                self.processed_path = None
            logger.info(f"DataNexus paths: Hist='{self.historical_path}', Proc='{self.processed_path}'")
        except Exception as e:
            logger.critical(f"CRITICAL Error initializing DataNexus paths. Settings type: {type(settings)}. Error: {e}", exc_info=True)
            self.historical_path = Path("./data/dn_critical_fallback_hist").resolve()
            self.processed_path = None
            logger.warning(f"Using CRITICAL fallback paths for DataNexus: {self.historical_path}")
            self.historical_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> bool:
        logger.info("Initializing DataNexus...")
        async with self._lock:
            if not await self._connect_to_mt5():
                logger.warning("MT5 connection failed during DataNexus init. Local data only if available.")
            else:
                logger.info("MT5 connection successful for DataNexus.")
                if self._mt5_initialized:
                    await self._ensure_symbols_visible_async()
            await self._load_all_historical_data()
        logger.info("DataNexus initialization complete.")
        return True

    async def _connect_to_mt5(self) -> bool:
        if self._mt5_initialized:
            term_info = await asyncio.to_thread(mt5.terminal_info)
            if term_info and getattr(term_info, 'connected', False):
                 logger.debug("MT5 already initialized and connected.")
                 return True
            logger.warning("MT5 previously initialized, but terminal seems disconnected. Re-initializing.")
            self._mt5_initialized = False

        if not hasattr(self, 'mt5_settings') or not self.mt5_settings:
            logger.error("CRITICAL: mt5_settings not found in DataNexus for MT5 connection.")
            return False

        logger.info(f"Attempting MT5 connection: Login={self.mt5_settings.mt5_login}, Server={self.mt5_settings.mt5_server}")
        try:
            init_success = await asyncio.to_thread(
                mt5.initialize,
                login=self.mt5_settings.mt5_login,
                password=self.mt5_settings.mt5_password,
                server=self.mt5_settings.mt5_server,
                path=str(self.mt5_settings.mt5_path)
            )
            if not init_success:
                logger.error(f"MT5 initialize() call failed. Code: {mt5.last_error()}")
                self._mt5_initialized = False
                return False

            terminal_info = await asyncio.to_thread(mt5.terminal_info)
            if not terminal_info or not getattr(terminal_info, 'connected', False):
                logger.error(f"Failed to get valid terminal_info or not connected after MT5 init. Info: {terminal_info}")
                await asyncio.to_thread(mt5.shutdown)
                self._mt5_initialized = False
                return False
            
            logger.info(f"Connected to MT5: {getattr(terminal_info, 'name', 'N/A')} (Build {getattr(terminal_info, 'build', 'N/A')}), Account Login configured: {self.mt5_settings.mt5_login}")
            self._mt5_initialized = True
            return True
        except Exception as e:
            logger.error(f"Exception during MT5 connection: {e}", exc_info=True)
            self._mt5_initialized = False
            return False
            
    async def _ensure_symbols_visible_async(self) -> None:
        if not self._mt5_initialized: return
        symbols_to_check = getattr(self.settings, 'symbols', [])
        logger.debug(f"Ensuring MT5 symbols visibility for: {symbols_to_check}")
        for symbol_str in symbols_to_check:
            try:
                s_info = await asyncio.to_thread(mt5.symbol_info, symbol_str)
                if s_info is None:
                    logger.warning(f"Symbol {symbol_str} not found on broker (symbol_info is None).")
                    continue
                if not s_info.visible:
                    logger.info(f"Symbol {symbol_str} not visible, attempting to select.")
                    selected = await asyncio.to_thread(mt5.symbol_select, symbol_str, True)
                    if not selected:
                        logger.warning(f"Failed to select symbol {symbol_str}. Error: {mt5.last_error()}")
                    else:
                        asyncio.create_task(self._wait_for_symbol_data_async(symbol_str))
            except Exception as e:
                logger.error(f"Error ensuring symbol {symbol_str} visibility: {e}", exc_info=True)

    async def _wait_for_symbol_data_async(self, symbol: str, timeout_seconds: int = 10):
        loop = asyncio.get_event_loop()
        end_time = loop.time() + timeout_seconds
        logger.debug(f"Waiting for data for symbol {symbol} (timeout: {timeout_seconds}s)")
        while loop.time() < end_time:
            tick = await asyncio.to_thread(mt5.symbol_info_tick, symbol)
            if tick and getattr(tick, 'time', 0) > 0:
                logger.info(f"Symbol {symbol} data available (tick received at {datetime.fromtimestamp(tick.time)}).")
                return
            await asyncio.sleep(0.5)
        logger.warning(f"Timeout waiting for symbol {symbol} data after selection.")

    def _get_historical_data_filepath(self, symbol: str, timeframe: TimeFrame) -> Path:
        symbol_dir = self.historical_path / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{symbol.upper()}_{timeframe.value}.parquet"

    async def _load_historical_data_from_file(self, symbol: str, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        filepath = self._get_historical_data_filepath(symbol, timeframe)
        if not filepath.exists():
            logger.debug(f"Historical file not found: {filepath}")
            return None
        try:
            logger.info(f"Loading historical data: {symbol} {timeframe.value} from {filepath}")
            df = await asyncio.to_thread(pd.read_parquet, filepath)
            
            if df.empty:
                logger.warning(f"Loaded historical file {filepath} is empty.")
                return None

            if df.index.name == 'time':
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index, errors='coerce', utc=False)
            elif 'time' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=False)
                df = df.set_index('time')
            else:
                logger.error(f"'time' column/index missing in {filepath}.")
                return None
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df.sort_index()

            df_standardized = self._standardize_ohlcv_df(df)
            df_final = df_standardized[~df_standardized.index.duplicated(keep='first')]
            logger.info(f"Loaded {len(df_final)} candles for {symbol} {timeframe.value} from file.")
            return df_final
        except Exception as e:
            logger.error(f"Error loading historical data from {filepath}: {e}", exc_info=True)
            return None

    async def _download_and_save_historical_data_mt5(
        self, symbol: str, timeframe: TimeFrame, num_candles: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        if not self._mt5_initialized:
            logger.warning(f"MT5 not initialized for download of {symbol} {timeframe.value}.")
            return None

        mt5_tf_val = timeframe.to_mt5_timeframe()
        if mt5_tf_val is None:
            logger.error(f"Invalid MT5 timeframe for {timeframe.value}. Download aborted for {symbol}.")
            return None

        max_hist_candles_setting = getattr(self.settings, 'max_historical_candles', 20000)
        count = num_candles if num_candles is not None else (max_hist_candles_setting or 20000)
        logger.info(f"DN_DOWNLOAD: Requesting {count} candles for {symbol} {timeframe.value} from MT5.")
        
        try:
            rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, mt5_tf_val, 0, count)
        except Exception as e:
            logger.error(f"Exception in mt5.copy_rates_from_pos for {symbol}: {e}", exc_info=True)
            return None

        if rates is None:
            logger.warning(f"MT5 returned None rates for {symbol} {timeframe.value}. Error: {mt5.last_error()}")
            return None
        if len(rates) == 0:
            logger.warning(f"MT5 returned 0 rates for {symbol} {timeframe.value}. Error: {mt5.last_error()}")
            return None
        logger.info(f"DN_DOWNLOAD: Received {len(rates)} rates from MT5 for {symbol} {timeframe.value}.")

        df = pd.DataFrame(rates)
        if df.empty:
            logger.warning(f"DN_DOWNLOAD: DataFrame from MT5 rates is empty for {symbol} {timeframe.value}.")
            return None
        
        logger.debug(f"DN_DOWNLOAD_V4: Initial columns from MT5 for {symbol} {timeframe.value}: {list(df.columns)}")

        df['time'] = pd.to_datetime(df['time'], unit='s', utc=False)
        df = df.set_index('time').sort_index()
        
        df_standardized = self._standardize_ohlcv_df(df)
        if df_standardized.empty:
             logger.warning(f"DN_DOWNLOAD: DataFrame empty after standardization for {symbol} {timeframe.value}.")
             return None

        df_final = df_standardized[~df_standardized.index.duplicated(keep='first')]
        if df_final.empty:
            logger.warning(f"DN_DOWNLOAD: DataFrame empty after dropping duplicates for {symbol} {timeframe.value}.")
            return None

        filepath = self._get_historical_data_filepath(symbol, timeframe)
        try:
            logger.info(f"Saving {len(df_final)} downloaded candles for {symbol} {timeframe.value} to {filepath}")
            await asyncio.to_thread(df_final.to_parquet, filepath, index=True)
        except Exception as e:
            logger.error(f"Error saving downloaded data to {filepath}: {e}", exc_info=True)
        
        logger.info(f"DN_DOWNLOAD: Returning DataFrame with {len(df_final)} rows for {symbol} {timeframe.value}.")
        return df_final

    def _standardize_ohlcv_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Standardize_df_V4: Input DataFrame columns: {list(df.columns)}")
        df_copy = df.copy()
        
        # --- *** اصلاح کلیدی در rename_map و انتخاب volume *** ---
        # تصمیم می گیریم که tick_volume اولویت دارد. اگر نبود، real_volume.
        # اگر هیچکدام نبود، بعداً ستون volume با صفر ساخته می شود.
        rename_map = {
            'tick_volume': 'volume', # اولویت اول
            # 'real_volume': 'volume', # این را کامنت می کنیم تا تداخل ایجاد نشود
            'Volume': 'volume', 'VOLUME': 'volume', 'vol': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close',
        }
        df_copy = df_copy.rename(columns=rename_map, errors='ignore')
        
        # اگر پس از rename اولیه، ستون 'volume' هنوز وجود ندارد اما 'real_volume' (از rename نشده ها) وجود دارد
        if 'volume' not in df_copy.columns and 'real_volume' in df_copy.columns:
            logger.debug("Standardize_df_V4: 'volume' not found after initial rename, using 'real_volume' instead.")
            df_copy = df_copy.rename(columns={'real_volume': 'volume'}, errors='ignore')
        # --- *** پایان اصلاح rename_map *** ---

        logger.debug(f"Standardize_df_V4: df_copy columns after all renames: {list(df_copy.columns)}")

        for col in ['open', 'high', 'low', 'close']:
            if col not in df_copy.columns:
                logger.error(f"Standardize Error V4: Required column '{col}' missing after rename.")
                return pd.DataFrame(index=df_copy.index) 
        
        if 'volume' not in df_copy.columns:
            logger.warning("Standardize Warning V4: 'volume' column still missing. Adding with zeros.")
            df_copy['volume'] = np.int64(0)
        
        if df_copy.index.name == 'time' and df_copy.index.tz is not None:
             df_copy.index = df_copy.index.tz_localize(None)
        elif 'time' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['time']) and df_copy['time'].dt.tz is not None:
             df_copy['time'] = df_copy['time'].dt.tz_localize(None)

        df_out = pd.DataFrame(index=df_copy.index)
        for col_name_target, dtype_target_str in OHLCV_DTYPES.items():
            if col_name_target == 'time': 
                continue 
            
            dtype_target = eval(dtype_target_str) if isinstance(dtype_target_str, str) and dtype_target_str != 'datetime64[ns]' else dtype_target_str
            logger.debug(f"Standardize_df_V4: Processing target_col='{col_name_target}', target_dtype='{dtype_target_str}'")

            if col_name_target in df_copy.columns:
                try:
                    # حالا باید df_copy[col_name_target] یک Series باشد
                    series_to_convert = df_copy[col_name_target] 
                except KeyError:
                    logger.error(f"Standardize_df_V4: KeyError selecting '{col_name_target}'. Should not happen if logic is correct.")
                    df_out[col_name_target] = pd.Series(0, index=df_copy.index, dtype=dtype_target)
                    continue
                
                logger.debug(f"Standardize_df_V4: For '{col_name_target}', selected data is of type: {type(series_to_convert)}. Shape: {getattr(series_to_convert, 'shape', 'N/A')}")
                if not isinstance(series_to_convert, pd.Series):
                     logger.error(f"Standardize_df_V4: CRITICAL - '{col_name_target}' is still not a Series (type: {type(series_to_convert)}). This indicates a deeper issue. Forcing to 0.")
                     df_out[col_name_target] = pd.Series(0, index=df_copy.index, dtype=dtype_target)
                     continue

                try:
                    if series_to_convert.dtype == object or not pd.api.types.is_numeric_dtype(series_to_convert.dtype):
                        logger.debug(f"Standardize_df_V4: Column '{col_name_target}' (dtype: {series_to_convert.dtype}) is object or non-numeric. Attempting pd.to_numeric.")
                        series_to_convert = pd.to_numeric(series_to_convert, errors='coerce')
                    
                    series_filled = series_to_convert.fillna(0)
                    df_out[col_name_target] = series_filled.astype(dtype_target)
                    logger.debug(f"Standardize_df_V4: Successfully converted '{col_name_target}' to {dtype_target_str}.")

                except Exception as e_astype:
                    logger.warning(f"Standardize_df_V4: astype failed for '{col_name_target}' to {dtype_target_str}: {e_astype}. Original Series Head: \n{series_to_convert.head() if isinstance(series_to_convert, pd.Series) else 'Not a Series'}. Forcing to 0.")
                    df_out[col_name_target] = pd.Series(0, index=df_copy.index, dtype=dtype_target)
            else: 
                logger.warning(f"Standardize_df_V4 Warning: Target column '{col_name_target}' not found in df_copy (available cols: {list(df_copy.columns)}). Adding with zeros.")
                df_out[col_name_target] = pd.Series(0, index=df_copy.index, dtype=dtype_target)
        
        logger.debug(f"Standardize_df_V4: Output df_out columns: {list(df_out.columns)}")
        return df_out.sort_index()

    async def _load_all_historical_data(self) -> None:
        # ... (بدون تغییر) ...
        logger.info("Starting _load_all_historical_data process...")
        symbols_to_load = getattr(self.settings, 'symbols', [])
        timeframes_to_load = getattr(self.settings, 'timeframes', [])

        if not symbols_to_load:
            logger.warning("No symbols configured in DataNexus settings. Historical data load skipped.")
            return

        for symbol_str in symbols_to_load:
            self._data_store[symbol_str] = {}
            for tf_enum in timeframes_to_load:
                logger.info(f"DN_LOAD_ALL: Processing {symbol_str} {tf_enum.value}")
                df = await self._load_historical_data_from_file(symbol_str, tf_enum)
                
                if df is None or df.empty:
                    logger.info(f"DN_LOAD_ALL: No local file for {symbol_str} {tf_enum.value}, trying download.")
                    if self._mt5_initialized:
                        df = await self._download_and_save_historical_data_mt5(symbol_str, tf_enum)
                        if df is not None and not df.empty:
                             logger.info(f"DN_LOAD_ALL: Downloaded {len(df)} candles for {symbol_str} {tf_enum.value} from MT5.")
                        else:
                             logger.warning(f"DN_LOAD_ALL: Download FAILED for {symbol_str} {tf_enum.value} from MT5.")
                    else:
                        logger.error(f"DN_LOAD_ALL: No local data for {symbol_str} {tf_enum.value} & MT5 not connected. Cannot obtain.")
                elif not df.empty:
                    logger.info(f"DN_LOAD_ALL: Loaded {len(df)} candles from file for {symbol_str} {tf_enum.value}.")
                
                if df is not None and not df.empty:
                    if self.settings.feature_engineering_enabled:
                        df = self._calculate_features(df.copy(), tf_enum, symbol_str)
                    self._data_store[symbol_str][tf_enum] = df
                    logger.info(f"DN_LOAD_ALL: Stored {len(df)} candles (features: {self.settings.feature_engineering_enabled}) for {symbol_str} {tf_enum.value}.")
                else:
                    logger.error(f"DN_LOAD_ALL: Ultimately failed to get data for {symbol_str} {tf_enum.value}.")
        logger.info("Historical data loading process finished.")

    def _calculate_features(self, df: pd.DataFrame, timeframe: TimeFrame, symbol: str) -> pd.DataFrame:
        # ... (بدون تغییر) ...
        if df.empty: return df
        if not getattr(self.settings, 'feature_engineering_enabled', False): return df
        
        logger.debug(f"Calculating features for {symbol} {timeframe.value} ({len(df)} candles)...")
        original_cols = df.columns.tolist()

        try:
            ind_cfg = self.settings.indicator_settings if hasattr(self.settings, 'indicator_settings') else {}
            if "SMA" in ind_cfg:
                for p in ind_cfg["SMA"].get("periods", []): df.ta.sma(length=p, append=True, col_names=(f"SMA_{p}",))
            if "EMA" in ind_cfg:
                for p in ind_cfg["EMA"].get("periods", []): df.ta.ema(length=p, append=True, col_names=(f"EMA_{p}",))
            if "RSI" in ind_cfg: 
                p_rsi = ind_cfg["RSI"].get("period",14)
                df.ta.rsi(length=p_rsi, append=True, col_names=(f"RSI_{p_rsi}",))
            if "MACD" in ind_cfg: 
                cfg_macd = ind_cfg["MACD"]
                df.ta.macd(fast=cfg_macd.get("fast_period",12), slow=cfg_macd.get("slow_period",26), signal=cfg_macd.get("signal_period",9), append=True)
            if "BBANDS" in ind_cfg: 
                cfg_bb = ind_cfg["BBANDS"]
                df.ta.bbands(length=cfg_bb.get("period",20), std=cfg_bb.get("std_dev",2), append=True)
            if "ATR" in ind_cfg: 
                p_atr = ind_cfg["ATR"].get("period",14)
                df.ta.atr(length=p_atr, append=True, col_names=(f"ATR_{p_atr}",))
            if "ADX" in ind_cfg: df.ta.adx(length=ind_cfg["ADX"].get("period",14), append=True)
            if "STOCH" in ind_cfg: 
                cfg_stoch = ind_cfg["STOCH"]
                df.ta.stoch(k=cfg_stoch.get("k_period",14), d=cfg_stoch.get("d_period",3), smooth_k=cfg_stoch.get("smooth_k",3), append=True)
            if "CCI" in ind_cfg:
                p_cci = ind_cfg["CCI"].get("period",20) 
                df.ta.cci(length=p_cci, append=True, col_names=(f"CCI_{p_cci}",))
            if "WILLR" in ind_cfg: 
                p_willr = ind_cfg["WILLR"].get("period",14)
                df.ta.willr(length=p_willr, append=True, col_names=(f"WILLR_{p_willr}",))
            
            new_cols = df.columns.difference(original_cols).tolist()
            if new_cols: logger.debug(f"Features added for {symbol} {timeframe.value}: {new_cols}")

        except Exception as e:
            logger.error(f"Error calculating features for {symbol} {timeframe.value}: {e}", exc_info=True)
            return df[original_cols] if all(c in df.columns for c in original_cols) else pd.DataFrame(index=df.index)
        
        df = df.bfill().ffill().fillna(0)
        return df

    async def _update_realtime_data_for_symbol_tf(self, symbol: str, timeframe: TimeFrame) -> bool:
        # ... (بدون تغییر) ...
        if not self._mt5_initialized:
            logger.debug(f"MT5 not init, cannot update RT for {symbol} {timeframe.value}")
            return False
        
        async with self._lock:
            current_df = self._data_store.get(symbol, {}).get(timeframe)
            if current_df is None or current_df.empty:
                logger.warning(f"No existing data for {symbol} {timeframe.value} to RT update. Fetching initial set.")
                synapse_ai_settings_obj = getattr(self.settings, 'synapse_ai', None)
                env_win_size_rt = getattr(synapse_ai_settings_obj, 'env_window_size', 60) if synapse_ai_settings_obj else 60
                
                initial_df = await self._download_and_save_historical_data_mt5(symbol, timeframe, num_candles=env_win_size_rt * 5 + 50)
                if initial_df is not None and not initial_df.empty:
                    if self.settings.feature_engineering_enabled:
                        initial_df = self._calculate_features(initial_df.copy(), timeframe, symbol)
                    if symbol not in self._data_store: self._data_store[symbol] = {}
                    self._data_store[symbol][timeframe] = initial_df
                    return True
                logger.error(f"Failed to fetch initial data for {symbol} {timeframe.value} during RT update attempt.")
                return False

            last_known_time_naive = current_df.index[-1]
            mt5_tf_val = timeframe.to_mt5_timeframe()
            if mt5_tf_val is None: return False
            num_candles_fetch = 200 

            try:
                rates = await asyncio.to_thread(mt5.copy_rates_from_pos, symbol, mt5_tf_val, 0, num_candles_fetch)
            except Exception as e:
                logger.error(f"Exception in RT mt5.copy_rates_from_pos for {symbol}: {e}", exc_info=True)
                return False

            if rates is None or len(rates) == 0:
                logger.debug(f"No new rates from MT5 for RT {symbol} {timeframe.value}. Error: {mt5.last_error()}")
                return False

            df_new = pd.DataFrame(rates)
            df_new['time'] = pd.to_datetime(df_new['time'], unit='s', utc=False)
            df_new = df_new.set_index('time').sort_index()
            df_new = self._standardize_ohlcv_df(df_new)
            
            df_updates = df_new[df_new.index >= last_known_time_naive]

            if df_updates.empty:
                logger.debug(f"No new candle data for RT {symbol} {timeframe.value} beyond {last_known_time_naive}.")
                return False
            
            ohlcv_cols_for_compare = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in current_df.columns and c in df_updates.columns]
            if len(df_updates) == 1 and df_updates.index[0] == last_known_time_naive and \
               df_updates.iloc[0][ohlcv_cols_for_compare].equals(current_df.loc[last_known_time_naive][ohlcv_cols_for_compare]):
                logger.debug(f"Last candle OHLCV for {symbol} {timeframe.value} at {last_known_time_naive} is identical. No update.")
                return False
                
            combined_df = pd.concat([current_df, df_updates])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
            
            max_len = self.settings.max_historical_candles or 500000
            if len(combined_df) > max_len:
                combined_df = combined_df.iloc[-max_len:]

            if self.settings.feature_engineering_enabled:
                self._data_store[symbol][timeframe] = self._calculate_features(combined_df.copy(), timeframe, symbol)
            else:
                self._data_store[symbol][timeframe] = combined_df
            
            logger.info(f"RT Updated data for {symbol} {timeframe.value}. New last: {combined_df.index[-1]}. Total: {len(combined_df)}.")
            return True

    async def realtime_data_feed_loop(self):
        # ... (بدون تغییر) ...
        if not await self._connect_to_mt5():
            logger.error("Cannot start RT feed loop: MT5 connection failed on initial attempt.")
            return

        interval_sec = getattr(self.settings, 'realtime_update_interval_seconds', 60)
        logger.info(f"Starting RT data feed (interval: {interval_sec}s)...")
        while not self._shutdown_event.is_set():
            try:
                if not await self._connect_to_mt5():
                    logger.warning("MT5 connection lost in RT feed loop. Retrying connection...")
                    await asyncio.sleep(max(interval_sec * 2, 30))
                    continue

                symbols_to_update = getattr(self.settings, 'symbols', [])
                timeframes_to_update = getattr(self.settings, 'timeframes', [])

                update_tasks = [
                    self._update_realtime_data_for_symbol_tf(s, tf_e)
                    for s in symbols_to_update for tf_e in timeframes_to_update
                ]
                if update_tasks:
                    results = await asyncio.gather(*update_tasks, return_exceptions=True)
                    for i, res in enumerate(results):
                        if isinstance(res, Exception):
                            coro_obj = update_tasks[i]
                            coro_name = getattr(coro_obj, '__qualname__', 'unknown_coroutine')
                            logger.error(f"Error in RT update task '{coro_name}': {res}")
                
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                logger.info("RT data feed loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in RT data_feed_loop: {e}", exc_info=True)
                await asyncio.sleep(max(interval_sec * 2, 30))
        logger.info("RT data feed loop stopped.")

    def get_market_data(self, symbol: str, timeframe: TimeFrame, num_candles: Optional[int] = None) -> Optional[pd.DataFrame]:
        # ... (بدون تغییر) ...
        try:
            df = self._data_store.get(symbol, {}).get(timeframe)
            if df is None or df.empty:
                logger.warning(f"No market data in store for {symbol} {timeframe.value}")
                return None
            df_copy = df.copy()
            return df_copy.tail(num_candles) if num_candles and num_candles > 0 else df_copy
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol} {timeframe.value}: {e}", exc_info=True)
            return None

    def get_latest_candle_time(self, symbol: str, timeframe: TimeFrame) -> Optional[datetime]:
        # ... (بدون تغییر) ...
        df = self.get_market_data(symbol, timeframe, num_candles=1)
        return df.index[-1].to_pydatetime() if df is not None and not df.empty else None
            
    def get_available_symbols_timeframes(self) -> Dict[str, List[str]]:
        # ... (بدون تغییر) ...
        return {
            s: [tf.value for tf in tf_data.keys()]
            for s, tf_data in self._data_store.items() if tf_data
        }

    async def shutdown(self):
        # ... (بدون تغییر) ...
        logger.info("Shutting down DataNexus...")
        self._shutdown_event.set()
        await asyncio.sleep(0.2)
        if self._mt5_initialized:
            logger.info("Disconnecting DataNexus from MT5...")
            try:
                await asyncio.to_thread(mt5.shutdown)
                self._mt5_initialized = False
                logger.info("DataNexus disconnected from MT5.")
            except Exception as e:
                logger.error(f"Error during MT5 shutdown in DataNexus: {e}", exc_info=True)
        logger.info("DataNexus shutdown complete.")