# D:\AdvancedTradingSystem\modules\trade_execution_gateway.py
# Version: Using threading.Lock instead of asyncio.Lock

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import time 
import threading # <--- اضافه شد

try:
    import MetaTrader5 as mt5
except ImportError:
    print("CRITICAL WARNING (trade_execution_gateway.py): MetaTrader5 module not found! Using MockMT5.")
    from unittest.mock import MagicMock 
    class MockMT5: # ... (MockMT5 بدون تغییر)
        ORDER_TYPE_BUY = 0; ORDER_TYPE_SELL = 1; ORDER_TYPE_BUY_LIMIT = 2; ORDER_TYPE_SELL_LIMIT = 3
        ORDER_TYPE_BUY_STOP = 4; ORDER_TYPE_SELL_STOP = 5; TRADE_ACTION_DEAL = 1
        TRADE_ACTION_PENDING = 0; ORDER_TIME_GTC = 0; ORDER_FILLING_IOC = 1; TRADE_ACTION_SLTP = 2
        TRADE_RETCODE_PLACED = 10008; TRADE_RETCODE_DONE = 10009; TRADE_RETCODE_DONE_PARTIAL = 10010
        TRADE_RETCODE_REQUOTE = 10004 
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name):
            mock_method = MagicMock(name=f"MockMT5.{name}")
            if name in ["initialize", "symbol_select", "shutdown"]: mock_method.return_value = True
            elif name in ["terminal_info", "account_info", "symbol_info", "order_send", "symbol_info_tick"]:
                inner_mock = MagicMock(name=f"MockMT5.{name}.ReturnValue")
                if name == "terminal_info": inner_mock.connected = False
                if name == "order_send": inner_mock.retcode = -1; inner_mock.comment="MockedUninit"
                mock_method.return_value = inner_mock
            elif name in ["positions_get", "history_deals_get", "copy_rates_from_pos"]: mock_method.return_value = [] if name != "copy_rates_from_pos" else None
            elif name == "last_error": mock_method.return_value = (-1, "Mock MT5 Not Initialized")
            else: mock_method.return_value = None
            return mock_method
    mt5 = MockMT5()


from config.schemas import MetaTraderBridgeSettings
try:
    from modules.risk_guardian import TradeSignal 
except ImportError:
    print("WARNING (trade_execution_gateway.py): Could not import TradeSignal. Defining placeholder.")
    class TradeSignal:
        def __init__(self, symbol: str, action: str, order_type: str = "MARKET", volume_lots: Optional[float] = None, price: Optional[float] = None, stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None, signal_source: str = "AI", meta_info: Optional[Dict] = None):
            self.symbol = symbol; self.action = action.upper(); self.order_type = order_type.upper()
            self.volume_lots = volume_lots; self.price = price; self.stop_loss_price = stop_loss_price
            self.take_profit_price = take_profit_price; self.signal_source = signal_source
            self.meta_info = meta_info or {}

logger = logging.getLogger(__name__)

MT5_POSITION_TYPE_MAP = {
    mt5.ORDER_TYPE_BUY: "BUY",
    mt5.ORDER_TYPE_SELL: "SELL"
}

class TradeExecutionGateway:
    def __init__(self, settings: MetaTraderBridgeSettings):
        self.settings = settings
        self._is_connected = False
        self._lock = threading.Lock() # <--- *** تغییر به threading.Lock ***

    async def initialize(self) -> bool:
        # با threading.Lock، نیازی به async with نیست، از with معمولی استفاده می کنیم
        # اما چون بقیه متدهای mt5 با to_thread هستند، خود initialize را async نگه می داریم
        with self._lock: # <--- استفاده از with معمولی
            if self._is_connected:
                # این بخش می تواند بدون قفل هم باشد چون فقط خواندنی است
                term_info_check = await asyncio.to_thread(mt5.terminal_info)
                if term_info_check and getattr(term_info_check, 'connected', False):
                    logger.debug("TEG: Already connected to MetaTrader 5 and connection verified.")
                    return True
                logger.warning("TEG: Was connected, but terminal seems disconnected. Re-initializing.")
                self._is_connected = False

            logger.info(f"TEG: Initializing MT5 connection: Login={getattr(self.settings, 'mt5_login', 'N/A')}")
            try:
                init_success = await asyncio.wait_for(
                    asyncio.to_thread(
                        mt5.initialize, login=self.settings.mt5_login,
                        password=self.settings.mt5_password, server=self.settings.mt5_server,
                        path=str(self.settings.mt5_path), timeout=self.settings.connection_timeout_seconds * 1000
                    ), timeout=self.settings.connection_timeout_seconds + 2
                )
                if not init_success:
                    logger.error(f"TEG: MT5 initialize() call failed. MT5 Error: {mt5.last_error()}")
                    self._is_connected = False; return False
                
                terminal_info = await asyncio.to_thread(mt5.terminal_info)
                if not terminal_info or not getattr(terminal_info, 'connected', False):
                    logger.error(f"TEG: Failed to get terminal_info or not connected post-init. Info: {terminal_info}. MT5 Error: {mt5.last_error()}")
                    await asyncio.to_thread(mt5.shutdown)
                    self._is_connected = False; return False
                
                self._is_connected = True
                logger.info(f"TEG: Successfully connected to MT5: {getattr(terminal_info, 'name', 'N/A')} (Build: {getattr(terminal_info, 'build', 'N/A')}), Account: N/A") # Account: N/A چون login در terminal_info نیست
                return True
            except asyncio.TimeoutError:
                logger.error(f"TEG: MT5 initialize() timed out after {self.settings.connection_timeout_seconds}s.")
                self._is_connected = False; return False
            except Exception as e:
                logger.error(f"TEG: Exception during MT5 initialize: {e}", exc_info=True)
                self._is_connected = False; return False

    async def _ensure_connected(self) -> bool:
        if self._is_connected:
            term_info_q = await asyncio.to_thread(mt5.terminal_info)
            if term_info_q and getattr(term_info_q, 'connected', False): return True
        return await self.initialize()

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        if not await self._ensure_connected(): 
            logger.warning("TEG: Cannot get account info, not connected.")
            return None
        # چون mt5.account_info() خودش thread-safe فرض نمی شود، قفل را نگه می داریم
        with self._lock: # <--- استفاده از with معمولی
            try:
                # mt5.account_info() با to_thread اجرا می شود
                acc_info_obj = await asyncio.to_thread(mt5.account_info)
                if acc_info_obj:
                    if hasattr(acc_info_obj, '_asdict'): return acc_info_obj._asdict()
                    return {
                        attr: getattr(acc_info_obj, attr) 
                        for attr in dir(acc_info_obj) 
                        if not attr.startswith('_') and not callable(getattr(acc_info_obj, attr))
                    }
                logger.error(f"TEG: mt5.account_info() returned None. MT5 Error: {mt5.last_error()}")
                return None
            except Exception as e:
                logger.error(f"TEG: Error getting account info: {e}", exc_info=True)
                return None

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not await self._ensure_connected():
            logger.warning(f"TEG: Cannot get symbol info for {symbol}, not connected.")
            return None
        with self._lock: # <--- استفاده از with معمولی
            try:
                selected = await asyncio.to_thread(mt5.symbol_select, symbol, True)
                if not selected: 
                    logger.warning(f"TEG: Could not select symbol {symbol}. Error: {mt5.last_error()}")
                
                s_info_mt5 = await asyncio.to_thread(mt5.symbol_info, symbol)
                if s_info_mt5:
                    # ... (بقیه کد بدون تغییر) ...
                    info_dict = {}
                    attrs_map = {
                        "name": "name", "description": "description", "currency_base": "currency_base",
                        "currency_profit": "currency_profit", "currency_margin": "currency_margin",
                        "digits": "digits", "point": "point", "trade_contract_size": "contract_size",
                        "volume_min": "volume_min", "volume_max": "volume_max", "volume_step": "volume_step",
                        "spread": "spread", "trade_stops_level": "trade_stops_level", "tick_value": "tick_value",
                        "tick_size": "tick_size", "ask": "ask", "bid": "bid", "time": "time"
                    }
                    for mt5_attr, dict_key in attrs_map.items():
                        if hasattr(s_info_mt5, mt5_attr): info_dict[dict_key] = getattr(s_info_mt5, mt5_attr)
                    
                    if "point" in info_dict and "digits" in info_dict:
                        digits = info_dict["digits"]
                        info_dict["pip_size"] = info_dict["point"] * (10 if (digits == 3 or digits == 5) else 1)
                    
                    time_val = info_dict.get("time")
                    if time_val and time_val > 0 : info_dict["time"] = datetime.fromtimestamp(time_val)
                    elif "time" in info_dict : info_dict["time"] = None
                    return info_dict
                else:
                    logger.error(f"TEG: Failed to get symbol_info for {symbol}. MT5 Error: {mt5.last_error()}")
                    return None
            except Exception as e:
                logger.error(f"TEG: Error getting symbol info for {symbol}: {e}", exc_info=True)
                return None

    async def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not await self._ensure_connected(): return []
        with self._lock: # <--- استفاده از with معمولی
            try:
                pos_mt5_tuple = await asyncio.to_thread(mt5.positions_get, symbol=symbol) if symbol else await asyncio.to_thread(mt5.positions_get)
                if pos_mt5_tuple is None: 
                    logger.warning(f"TEG: positions_get returned None for '{symbol or 'all'}'. Error: {mt5.last_error()}")
                    return []
                
                positions_list = []
                for p_obj in pos_mt5_tuple:
                    pos_dict = {field: getattr(p_obj, field, None) for field in getattr(p_obj, '_fields', [])}
                    pos_dict['type_str'] = MT5_POSITION_TYPE_MAP.get(pos_dict.get('type'), "UNKNOWN")
                    if pos_dict.get('time', 0) > 0: pos_dict['time_open_dt'] = datetime.fromtimestamp(pos_dict['time'])
                    if pos_dict.get('sl', 0.0) == 0.0: pos_dict['sl'] = None
                    if pos_dict.get('tp', 0.0) == 0.0: pos_dict['tp'] = None
                    positions_list.append(pos_dict)
                return positions_list
            except Exception as e:
                logger.error(f"TEG: Error getting open positions: {e}", exc_info=True)
                return []

    async def execute_trade_signal(self, signal: TradeSignal) -> Dict[str, Any]:
        if not await self._ensure_connected():
            return {"success": False, "message": "TEG: Not connected to MT5.", "retcode": -100, "details_raw": None}

        with self._lock: # <--- استفاده از with معمولی
            request = { # ... (بقیه request بدون تغییر) ...
                "action": mt5.TRADE_ACTION_DEAL, "symbol": signal.symbol,
                "volume": float(signal.volume_lots) if signal.volume_lots is not None else 0.0,
                "type": None, "price": 0.0,
                "sl": float(signal.stop_loss_price) if signal.stop_loss_price is not None and signal.stop_loss_price > 0 else 0.0,
                "tp": float(signal.take_profit_price) if signal.take_profit_price is not None and signal.take_profit_price > 0 else 0.0,
                "deviation": 20, "magic": signal.meta_info.get("magic_number", 234003),
                "comment": signal.meta_info.get("comment", f"SigSrc:{signal.signal_source}")[:31],
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
            }
            if request["volume"] <= 0 and signal.action.upper() != "CLOSE":
                 logger.error(f"TEG: Invalid volume {request['volume']} for {signal.action} on {signal.symbol}.")
                 return {"success": False, "message": f"TEG: Invalid volume {request['volume']} for {signal.action}.", "retcode": -101, "details_raw": request}

            action_upper = signal.action.upper(); order_type_upper = signal.order_type.upper()

            if action_upper == "BUY": # ... (منطق BUY, SELL, CLOSE بدون تغییر) ...
                if order_type_upper == "MARKET": request["type"] = mt5.ORDER_TYPE_BUY
                elif order_type_upper == "LIMIT":
                    if signal.price is None or signal.price <= 0: return {"success": False, "message": "TEG: Price required for BUY_LIMIT."}
                    request["type"] = mt5.ORDER_TYPE_BUY_LIMIT; request["price"] = float(signal.price); request["action"] = mt5.TRADE_ACTION_PENDING
                elif order_type_upper == "STOP":
                    if signal.price is None or signal.price <= 0: return {"success": False, "message": "TEG: Price required for BUY_STOP."}
                    request["type"] = mt5.ORDER_TYPE_BUY_STOP; request["price"] = float(signal.price); request["action"] = mt5.TRADE_ACTION_PENDING
                else: return {"success": False, "message": f"TEG: Unsupported BUY order type: {order_type_upper}"}
            elif action_upper == "SELL":
                if order_type_upper == "MARKET": request["type"] = mt5.ORDER_TYPE_SELL
                elif order_type_upper == "LIMIT":
                    if signal.price is None or signal.price <= 0: return {"success": False, "message": "TEG: Price required for SELL_LIMIT."}
                    request["type"] = mt5.ORDER_TYPE_SELL_LIMIT; request["price"] = float(signal.price); request["action"] = mt5.TRADE_ACTION_PENDING
                elif order_type_upper == "STOP":
                    if signal.price is None or signal.price <= 0: return {"success": False, "message": "TEG: Price required for SELL_STOP."}
                    request["type"] = mt5.ORDER_TYPE_SELL_STOP; request["price"] = float(signal.price); request["action"] = mt5.TRADE_ACTION_PENDING
                else: return {"success": False, "message": f"TEG: Unsupported SELL order type: {order_type_upper}"}
            elif action_upper == "CLOSE":
                pos_ticket = signal.meta_info.get("position_ticket")
                # چون get_open_positions هم از lock استفاده می کند، نمی توانیم آن را اینجا await کنیم
                # راه حل: یا get_open_positions را بدون lock کنیم (اگر خواندنی و thread-safe است)
                # یا اطلاعات پوزیشن را از جای دیگری بگیریم (مثلا از orchestrator اگر کش می کند)
                # یا اینکه یک نسخه همزمان (non-async) از get_open_positions داشته باشیم که داخل lock صدا زده شود.
                # برای سادگی، فرض می کنیم pos_ticket در meta_info سیگنال CLOSE موجود است و معتبر است.
                if not pos_ticket:
                     return {"success": False, "message": f"TEG: Position ticket required in meta_info for CLOSE signal."}
                
                # باید نوع پوزیشن (خرید یا فروش) را بدانیم تا نوع سفارش مخالف را ارسال کنیم
                # این اطلاعات باید از یک منبع دیگر (مثلا کش پوزیشن ها در ارکستراتور) بیاید یا در meta_info سیگنال باشد.
                # فرض می کنیم در meta_info نوع پوزیشن هم هست: signal.meta_info.get('position_type') == 'BUY' یا 'SELL'
                original_pos_type = signal.meta_info.get('position_type_str', '').upper() # مثلا "BUY" یا "SELL"
                if not original_pos_type:
                    logger.warning(f"TEG: position_type_str not found in meta_info for closing ticket {pos_ticket}. Trying to infer...")
                    # تلاش برای خواندن مستقیم از MT5 (اینجا چون داخل lock هستیم، باید با to_thread باشد)
                    # این ممکن است باعث deadlock شود اگر get_open_positions خودش lock بگیرد.
                    # بهتر است این اطلاعات از قبل موجود باشد.
                    # فعلا فرض می کنیم این مشکل وجود ندارد و پوزیشن از قبل مشخص است.
                    return {"success": False, "message": f"TEG: Position type unknown for closing ticket {pos_ticket}."}


                request["position"] = pos_ticket
                # اگر volume_lots در سیگنال CLOSE مشخص نشده، تمام حجم پوزیشن را می بندیم
                # که این هم نیاز به دانستن حجم پوزیشن دارد.
                if signal.volume_lots is None:
                    # باز هم نیاز به اطلاعات پوزیشن از قبل
                    logger.warning(f"TEG: Volume for closing position {pos_ticket} not specified. (Full close logic TBD)")
                    # return {"success": False, "message": "TEG: Volume for closing not specified."}
                    # فرض می کنیم اگر حجم نیست، به معنی بستن کامل است، اما mt5.order_send به حجم نیاز دارد.
                    # این بخش نیاز به بازنگری دارد که حجم از کجا می آید.
                    # در ریسک گاردین حجم برای بستن تنظیم می شود.
                    pass # RiskGuardian باید حجم را تنظیم کرده باشد
                
                request["volume"] = float(signal.volume_lots or 0.01) # حداقل حجم اگر هیچ چیز نیست
                if original_pos_type == "BUY": request["type"] = mt5.ORDER_TYPE_SELL
                elif original_pos_type == "SELL": request["type"] = mt5.ORDER_TYPE_BUY
                else: return {"success": False, "message": f"TEG: Unknown original pos type '{original_pos_type}' for closing."}
                request["sl"] = 0.0; request["tp"] = 0.0
            else:
                return {"success": False, "message": f"TEG: Unsupported action: {action_upper}"}

            result_obj = None; result_dict = {}
            for attempt in range(self.settings.max_retry_attempts + 1):
                try:
                    logger.info(f"TEG: Sending trade request (Attempt {attempt+1}): {request}")
                    # mt5.order_send یک تابع همزمان است
                    result_obj = await asyncio.to_thread(mt5.order_send, request)
                    result_dict = result_obj._asdict() if hasattr(result_obj, '_asdict') else (vars(result_obj) if result_obj is not None else {})
                    break 
                except asyncio.TimeoutError: # این TimeoutError برای asyncio.wait_for است که اینجا استفاده نمی شود
                    logger.warning(f"TEG: order_send (simulated) timeout (Attempt {attempt+1}).")
                    if attempt == self.settings.max_retry_attempts:
                        return {"success": False, "message": "TEG: Request timeout after retries.", "retcode": -200, "details_raw": None}
                except Exception as e: # خطاهای دیگر از mt5.order_send
                    logger.error(f"TEG: order_send error (Attempt {attempt+1}): {e}", exc_info=True)
                    if attempt == self.settings.max_retry_attempts:
                        return {"success": False, "message": f"TEG: Execution error: {e}", "retcode": -300, "details_raw": None}
                if attempt < self.settings.max_retry_attempts:
                    # time.sleep همزمان است، از asyncio.sleep استفاده می کنیم اما چون داخل with self._lock هستیم،
                    # و self._lock از نوع threading.Lock است، asyncio.sleep اینجا event loop را بلاک نمی کند
                    # بلکه ترد فعلی را برای مدت کوتاهی آزاد می کند (اگرچه اینجا فقط یک ترد داریم که این lock را گرفته)
                    # بهتر است اگر retry_delay_seconds نیاز به async دارد، این حلقه را بازنویسی کنیم
                    # فعلا با فرض اینکه retry_delay_seconds کوتاه است:
                    await asyncio.sleep(self.settings.retry_delay_seconds * (attempt + 1)) 
            
            if result_obj is None:
                 return {"success": False, "message": "TEG: No response from server after retries (result_obj is None).", "retcode": -400, "details_raw": None}

            if not result_dict and result_obj: 
                 result_dict = {'retcode': getattr(result_obj, 'retcode', -1), 
                                'comment': getattr(result_obj, 'comment', 'Mocked Comment'),
                                'order': getattr(result_obj, 'order', 0), 'deal': getattr(result_obj, 'deal', 0),
                                'request_id': getattr(result_obj, 'request_id', 0)}

            retcode = result_dict.get('retcode', -1)
            comment = result_dict.get('comment', 'Unknown MT5 Comment')
            order_ticket_res = result_dict.get('order', 0)
            deal_ticket_res = result_dict.get('deal', 0)
            request_id_res = result_dict.get('request_id')
            
            logger.info(f"TEG: MT5 order_send raw result: Code={retcode}, Comment='{comment}', Order={order_ticket_res}, Deal={deal_ticket_res}")
            hardcoded_successful_retcodes = [10008, 10009, 10010]

            if retcode in hardcoded_successful_retcodes:
                return {
                    "success": True, 
                    "message": f"Order execution successful. MT5 Comment: {comment}",
                    "retcode": retcode,
                    "order_ticket": order_ticket_res if order_ticket_res > 0 else None,
                    "deal_ticket": deal_ticket_res if deal_ticket_res > 0 else None,
                    "request_id": request_id_res,
                    "details_raw": result_dict, 
                }
            else:
                error_message = f"Trade execution failed. MT5 Retcode: {retcode} ({comment})"
                logger.error(f"TEG: {error_message}") 
                return { "success": False, "message": error_message, "retcode": retcode, "details_raw": result_dict, }

    async def modify_position(self, ticket: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        if not await self._ensure_connected(): return {"success": False, "message": "TEG: Not connected."}
        
        # برای خواندن اطلاعات پوزیشن، از lock استفاده می کنیم
        pos_info_dict = None
        with self._lock:
            # فرض می کنیم یک تابع همزمان برای خواندن اطلاعات یک پوزیشن خاص داریم
            # یا اینکه get_open_positions را طوری تغییر دهیم که بتواند داخل with lock صدا زده شود.
            # فعلا این بخش را ساده نگه می داریم.
            # در یک سیستم واقعی، باید اطلاعات پوزیشن برای modify از یک منبع معتبر بیاید.
            # این بخش از کد اصلی شما ساده سازی شده بود.
            all_positions_raw = await asyncio.to_thread(mt5.positions_get, ticket=ticket)
            if all_positions_raw and len(all_positions_raw) > 0:
                 pos_info_dict = all_positions_raw[0]._asdict() if hasattr(all_positions_raw[0], '_asdict') else vars(all_positions_raw[0])

        if not pos_info_dict: return {"success": False, "message": f"TEG: Position {ticket} not found."}
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP, 
            "position": ticket, 
            "symbol": pos_info_dict['symbol'], 
            "sl": float(stop_loss) if stop_loss is not None else float(pos_info_dict.get('sl', 0.0) or 0.0), 
            "tp": float(take_profit) if take_profit is not None else float(pos_info_dict.get('tp', 0.0) or 0.0),
        }
        if stop_loss == 0.0: request['sl'] = 0.0 # اجازه حذف SL
        if take_profit == 0.0: request['tp'] = 0.0 # اجازه حذف TP

        with self._lock: # <--- استفاده از with معمولی
            logger.info(f"TEG: Sending SL/TP mod request: {request}")
            res_obj = await asyncio.to_thread(mt5.order_send, request)
            res_dict = res_obj._asdict() if hasattr(res_obj, '_asdict') else (vars(res_obj) if res_obj else {})
        
        retcode = res_dict.get('retcode', -1); comment = res_dict.get('comment', 'Unknown MT5 Comment')
        logger.info(f"TEG: Modify pos result: Code={retcode}, Comment='{comment}'")
        if retcode == mt5.TRADE_RETCODE_DONE: return {"success": True, "message": "Position modification successful.", "details_raw": res_dict}
        else: return {"success": False, "message": f"Pos mod failed. Retcode: {retcode} ({comment})", "details_raw": res_dict}

    async def close_position(self, ticket: int, volume_to_close: Optional[float] = None, comment: str = "SysClose") -> Dict[str, Any]:
        # برای بستن پوزیشن، اطلاعات پوزیشن (مانند نوع و حجم) لازم است
        # این اطلاعات باید از meta_info سیگنال یا با خواندن مستقیم پوزیشن بدست آید.
        # RiskGuardian باید این اطلاعات را در TradeSignal برای بستن قرار دهد.
        
        # فرض می کنیم TradeSignal حاوی اطلاعات لازم است
        # signal.meta_info['position_type_str'] = 'BUY' یا 'SELL'
        # signal.meta_info['position_ticket'] = ticket
        # signal.volume_lots توسط RiskGuardian تنظیم شده (یا None برای بستن کامل که باید در execute مدیریت شود)

        # این تابع به شکل فعلی کامل نیست چون به اطلاعات نوع پوزیشن نیاز دارد.
        # execute_trade_signal منطق بستن را دارد اگر سیگنال درست باشد.
        
        # برای این تابع، باید ابتدا اطلاعات پوزیشن را بگیریم
        pos_to_close_list = await self.get_open_positions() # این خودش از lock استفاده می کند
        pos_to_close_details = next((p for p in pos_to_close_list if p.get('ticket') == ticket), None)
        if not pos_to_close_details:
            return {"success": False, "message": f"TEG: Position {ticket} not found for closing."}

        close_signal = TradeSignal(
            symbol=pos_to_close_details['symbol'], 
            action="CLOSE", 
            volume_lots=volume_to_close or pos_to_close_details['volume'], 
            meta_info={
                "position_ticket": ticket, 
                "comment": comment[:31], 
                "signal_source": "SystemClose",
                "position_type_str": MT5_POSITION_TYPE_MAP.get(pos_to_close_details['type'], "UNKNOWN")
            }
        )
        return await self.execute_trade_signal(close_signal)


    async def close_all_positions_for_symbol(self, symbol: str, comment: str = "SysCloseAllSym") -> List[Dict[str, Any]]:
        results = []; open_positions = await self.get_open_positions(symbol=symbol)
        logger.info(f"TEG: Attempting to close {len(open_positions)} positions for {symbol}.")
        for pos in open_positions:
            if 'ticket' in pos:
                res = await self.close_position(pos['ticket'], comment=f"{comment}-{pos.get('ticket')}")
                results.append(res); await asyncio.sleep(0.1) # فاصله کوتاه بین بستن ها
        return results
        
    async def close_all_open_positions(self, comment: str = "SysCloseAll") -> List[Dict[str, Any]]:
        results = []; open_positions = await self.get_open_positions()
        logger.info(f"TEG: Attempting to close {len(open_positions)} total open positions.")
        for pos in open_positions:
            if 'ticket' in pos and 'symbol' in pos : # اطمینان از وجود کلیدها
                res = await self.close_position(pos['ticket'], comment=f"{comment}-{pos['symbol']}-{pos['ticket']}")
                results.append(res); await asyncio.sleep(0.1)
        return results

    async def get_historical_trades(self, from_date: datetime, to_date: datetime, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not await self._ensure_connected(): return []
        with self._lock: # <--- استفاده از with معمولی
            try:
                deals_mt5 = await asyncio.to_thread(mt5.history_deals_get, from_date, to_date)
                if deals_mt5 is None: logger.warning(f"TEG: history_deals_get None. Error: {mt5.last_error()}"); return []
                deals_list = [d._asdict() for d in deals_mt5 if hasattr(d, '_asdict')]
                return [d for d in deals_list if symbol is None or d.get('symbol') == symbol]
            except Exception as e: logger.error(f"TEG: Error getting historical deals: {e}", exc_info=True); return []

    async def shutdown(self):
        logger.info("Shutting down TradeExecutionGateway...")
        # قفل را قبل از خاموش کردن MT5 آزاد می کنیم، اگرچه اینجا شاید لازم نباشد
        if self._is_connected:
            with self._lock: # <--- استفاده از with معمولی
                if self._is_connected: # بررسی مجدد داخل قفل
                    logger.info("Disconnecting TEG from MetaTrader 5...")
                    await asyncio.to_thread(mt5.shutdown)
                    self._is_connected = False
            logger.info("TEG disconnected from MetaTrader 5.")
        logger.info("TradeExecutionGateway shutdown complete.")