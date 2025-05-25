# D:\AdvancedTradingSystem\modules\risk_guardian.py
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import math
from decimal import Decimal, ROUND_DOWN # برای محاسبات دقیق تر

# از config.schemas ایمپورت می کنیم
# اطمینان حاصل کنید که این ایمپورت در محیط تست و اجرا به درستی کار می کند
try:
    from config.schemas import RiskGuardianSettings
except ImportError:
    # این fallback برای زمانی است که ممکن است مستقیم این فایل اجرا شود یا pytest نتواند مسیر را پیدا کند
    # در حالت عادی، SystemOrchestrator باید PYTHONPATH را درست تنظیم کند.
    # برای تست ها، conftest.py این کار را می کند.
    class RiskGuardianSettings: # یک Placeholder ساده
        pass
    print("WARNING: Could not import RiskGuardianSettings from config.schemas in risk_guardian.py")


logger = logging.getLogger(__name__)

class TradeSignal: # این کلاس باید در یک فایل مشترک یا در همینجا باشد اگر فقط اینجا استفاده می شود
    def __init__(self, symbol: str, action: str,
                 order_type: str = "MARKET",
                 volume_lots: Optional[float] = None,
                 price: Optional[float] = None,
                 stop_loss_price: Optional[float] = None,
                 take_profit_price: Optional[float] = None,
                 signal_source: str = "AI_Core",
                 meta_info: Optional[Dict] = None):
        self.symbol = symbol
        self.action = action.upper()
        self.order_type = order_type.upper()
        self.volume_lots = volume_lots
        self.price = price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.signal_source = signal_source
        self.meta_info = meta_info or {}

    def __repr__(self):
        return (f"TradeSignal(symbol='{self.symbol}', action='{self.action}', volume={self.volume_lots}, "
                f"sl={self.stop_loss_price}, tp={self.take_profit_price})")


class RiskGuardian:
    def __init__(self, settings: RiskGuardianSettings):
        self.settings = settings
        self.daily_initial_balance: Optional[float] = None
        self.highest_account_balance: Optional[float] = None
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_end_time: Optional[datetime] = None

        if self.settings and hasattr(self.settings, 'max_risk_per_trade_pct'): # بررسی وجود تنظیمات
            logger.info("RiskGuardian initialized with settings:")
            logger.info(f"  Max Risk/Trade: {self.settings.max_risk_per_trade_pct}%")
            logger.info(f"  Max Daily Drawdown: {self.settings.max_account_drawdown_pct_daily}%")
            logger.info(f"  Max Total Drawdown: {self.settings.max_account_drawdown_pct_total}%")
        else:
            logger.warning("RiskGuardian initialized WITHOUT PROPER SETTINGS!")


    def update_account_state(self, current_balance: float, equity: float, open_positions: List[Dict]):
        if self.daily_initial_balance is None:
            self.daily_initial_balance = current_balance
            if self.settings.log_risk_checks: logger.info(f"Daily initial balance set: {current_balance:.2f}")

        if self.highest_account_balance is None or equity > self.highest_account_balance:
            self.highest_account_balance = equity
            if self.settings.log_risk_checks: logger.info(f"New high water mark: {equity:.2f}")
        
        if self.circuit_breaker_active and self.circuit_breaker_end_time and datetime.now() >= self.circuit_breaker_end_time:
            self.circuit_breaker_active = False
            self.circuit_breaker_end_time = None
            logger.warning("Circuit breaker period ended. Trading re-enabled by RiskGuardian.")

    def new_day_reset(self, current_balance: float):
        self.daily_initial_balance = current_balance
        if self.settings.log_risk_checks: logger.info(f"New trading day. Daily initial balance reset: {current_balance:.2f}")

    def check_circuit_breaker(self, current_balance: float, equity: float) -> bool:
        if self.circuit_breaker_active:
            if self.circuit_breaker_end_time and datetime.now() < self.circuit_breaker_end_time:
                if self.settings.log_risk_checks: logger.warning(f"CB ACTIVE. No new trades until {self.circuit_breaker_end_time}.")
                return True
            else:
                self.circuit_breaker_active = False
                self.circuit_breaker_end_time = None
                logger.info("Circuit breaker period ended.")
        
        if not self.settings.circuit_breaker_on_max_drawdown:
            return False

        if self.daily_initial_balance and self.settings.max_account_drawdown_pct_daily is not None: # بررسی None
            daily_drawdown_pct = (self.daily_initial_balance - equity) / self.daily_initial_balance * 100
            if daily_drawdown_pct >= self.settings.max_account_drawdown_pct_daily:
                logger.critical(f"MAX DAILY DD REACHED: {daily_drawdown_pct:.2f}% >= "
                                f"{self.settings.max_account_drawdown_pct_daily}%. Activating CB!")
                self._activate_circuit_breaker()
                return True

        if self.highest_account_balance and self.settings.max_account_drawdown_pct_total is not None: # بررسی None
            total_drawdown_pct = (self.highest_account_balance - equity) / self.highest_account_balance * 100
            if total_drawdown_pct >= self.settings.max_account_drawdown_pct_total:
                logger.critical(f"MAX TOTAL DD REACHED: {total_drawdown_pct:.2f}% >= "
                                f"{self.settings.max_account_drawdown_pct_total}%. Activating CB!")
                self._activate_circuit_breaker()
                return True
        return False

    def _activate_circuit_breaker(self):
        self.circuit_breaker_active = True
        if self.settings.cool_down_period_minutes_after_circuit_breaker:
            self.circuit_breaker_end_time = datetime.now() + timedelta(
                minutes=self.settings.cool_down_period_minutes_after_circuit_breaker
            )
            logger.warning(f"CB activated. Trading halted until {self.circuit_breaker_end_time}.")
        else:
            logger.warning("CB activated. Trading halted indefinitely.")

    def validate_new_trade_signal(
        self, signal: TradeSignal, current_balance: float, 
        open_positions: List[Dict], symbol_info: Dict
    ) -> Tuple[Optional[TradeSignal], str]:
        
        min_lot = Decimal(str(symbol_info.get('volume_min', self.settings.min_position_size_lots)))
        lot_step = Decimal(str(symbol_info.get('volume_step', 0.01)))
        contract_size = Decimal(str(symbol_info.get('contract_size', 100000))) # مقدار پیش فرض اگر نیست

        if self.circuit_breaker_active:
            return None, f"Trade rejected: Circuit breaker is active until {self.circuit_breaker_end_time or 'further notice'}."

        # 1. بررسی محدودیت تعداد معاملات باز
        if signal.action != "CLOSE": # قوانین تعداد معاملات برای باز کردن پوزیشن جدید اعمال می شود
            num_total_open = len(open_positions)
            if self.settings.max_total_open_trades is not None and num_total_open >= self.settings.max_total_open_trades:
                 is_closing_trade_for_other_symbol = \
                               (signal.action == "BUY" and any(p['symbol'] == signal.symbol and p['type'] == 'SELL' for p in open_positions)) or \
                               (signal.action == "SELL" and any(p['symbol'] == signal.symbol and p['type'] == 'BUY' for p in open_positions))
                 if not is_closing_trade_for_other_symbol: # اگر در حال بستن پوزیشن همان نماد نیست
                    existing_symbol_pos_count = sum(1 for p in open_positions if p['symbol'] == signal.symbol)
                    if not (self.settings.max_open_trades_per_symbol is not None and existing_symbol_pos_count < self.settings.max_open_trades_per_symbol):
                         return None, f"Trade rejected: Max total open trades ({self.settings.max_total_open_trades}) reached."


            num_open_for_symbol = sum(1 for p in open_positions if p['symbol'] == signal.symbol)
            if self.settings.max_open_trades_per_symbol is not None and num_open_for_symbol >= self.settings.max_open_trades_per_symbol:
                # اجازه بستن پوزیشن موجود برای همان نماد (اگر سیگنال خرید برای بستن فروش است یا برعکس)
                is_reversing_position = (signal.action == "BUY" and any(p['type'] == 'SELL' for p in open_positions if p['symbol'] == signal.symbol)) or \
                                        (signal.action == "SELL" and any(p['type'] == 'BUY' for p in open_positions if p['symbol'] == signal.symbol))
                if not is_reversing_position:
                    return None, f"Trade rejected: Max open trades for symbol {signal.symbol} ({self.settings.max_open_trades_per_symbol}) reached."

        if signal.action == "CLOSE":
            existing_pos = next((p for p in open_positions if p['symbol'] == signal.symbol and p['ticket'] == signal.meta_info.get('position_ticket')), 
                                next((p for p in open_positions if p['symbol'] == signal.symbol), None) # اگر تیکت نبود، اولین پوزیشن نماد
                               )
            if existing_pos:
                if signal.volume_lots is None or signal.volume_lots > existing_pos['volume']:
                    if self.settings.log_risk_checks: logger.info(f"Adjusting close signal volume for {signal.symbol} to {existing_pos['volume']}")
                    signal.volume_lots = float(existing_pos['volume'])
            else:
                return None, f"Close signal for {signal.symbol} but no matching open position found."
            return signal, "Close signal validated."

        # --- فقط برای سیگنال های باز کردن پوزیشن جدید (BUY/SELL) ---
        if signal.action not in ["BUY", "SELL"]:
            return None, f"Trade rejected: Invalid action '{signal.action}' for new trade validation."

        entry_price_for_calc = signal.price
        if entry_price_for_calc is None: # برای مارکت اردر، قیمت فعلی را حدس بزنید (این بخش باید بهبود یابد)
            # این باید از ماژول قیمت لحظه ای بیاید. فعلا فرض می کنیم در symbol_info هست
            if signal.action == "BUY" and 'ask' in symbol_info: entry_price_for_calc = symbol_info['ask']
            elif signal.action == "SELL" and 'bid' in symbol_info: entry_price_for_calc = symbol_info['bid']
            else:
                return None, "Trade rejected: Entry price not available for market order risk calculation."
            signal.price = entry_price_for_calc # بروزرسانی قیمت در سیگنال

        # 2. مدیریت حد ضرر
        if self.settings.enforce_stop_loss and signal.stop_loss_price is None:
            if self.settings.default_stop_loss_pips and 'pip_size' in symbol_info:
                pips = Decimal(str(self.settings.default_stop_loss_pips))
                pip_size_dec = Decimal(str(symbol_info['pip_size']))
                price_dec = Decimal(str(entry_price_for_calc))
                if signal.action == "BUY":
                    signal.stop_loss_price = float(price_dec - (pips * pip_size_dec))
                else: # SELL
                    signal.stop_loss_price = float(price_dec + (pips * pip_size_dec))
                if self.settings.log_risk_checks: logger.info(f"Default SL ({pips} pips) set for {signal.symbol} at {signal.stop_loss_price}")
            elif self.settings.default_stop_loss_pct_from_entry:
                pct_sl = Decimal(str(self.settings.default_stop_loss_pct_from_entry)) / Decimal('100.0')
                price_dec = Decimal(str(entry_price_for_calc))
                if signal.action == "BUY":
                    signal.stop_loss_price = float(price_dec * (Decimal('1.0') - pct_sl))
                else: # SELL
                    signal.stop_loss_price = float(price_dec * (Decimal('1.0') + pct_sl))
                if self.settings.log_risk_checks: logger.info(f"Default SL ({self.settings.default_stop_loss_pct_from_entry}%) set: {signal.stop_loss_price}")
            else:
                 return None, "Trade rejected: Stop loss enforced but no default SL mechanism configured or applicable."
        
        if signal.stop_loss_price is None and self.settings.enforce_stop_loss: # بررسی مجدد
             return None, f"Trade rejected: Stop loss enforced but could not be set for {signal.symbol}."

        # اعتبارسنجی اولیه SL (نباید در سمت اشتباه قیمت یا خیلی نزدیک باشد)
        # این بخش نیاز به stops_level از symbol_info دارد
        stops_level_points = symbol_info.get('trade_stops_level', 0)
        point_size = Decimal(str(symbol_info.get('point', 0.00001)))
        min_sl_distance = Decimal(str(stops_level_points)) * point_size

        entry_dec = Decimal(str(entry_price_for_calc))
        sl_dec = Decimal(str(signal.stop_loss_price))

        if signal.action == "BUY" and (sl_dec >= entry_dec or (entry_dec - sl_dec) < min_sl_distance):
            return None, f"Trade rejected: Invalid SL {signal.stop_loss_price} for BUY at {entry_price_for_calc} (too close or wrong side)."
        if signal.action == "SELL" and (sl_dec <= entry_dec or (sl_dec - entry_dec) < min_sl_distance):
            return None, f"Trade rejected: Invalid SL {signal.stop_loss_price} for SELL at {entry_price_for_calc} (too close or wrong side)."


        # 3. محاسبه حجم
        calculated_volume_lots_dec = Decimal('0.0')
        if signal.volume_lots is not None:
            calculated_volume_lots_dec = Decimal(str(signal.volume_lots))
        
        # محاسبه حجم بر اساس ریسک (اگر حجم داده نشده یا برای بررسی حجم داده شده)
        if self.settings.max_risk_per_trade_pct is not None and signal.stop_loss_price:
            risk_per_share_dec = abs(entry_dec - sl_dec)
            risk_per_lot_dec = risk_per_share_dec * contract_size

            if risk_per_lot_dec <= Decimal('1e-9'):
                return None, "Trade rejected: Stop loss too close, risk per lot is near zero."

            max_loss_allowed_dec = Decimal(str(current_balance)) * (Decimal(str(self.settings.max_risk_per_trade_pct)) / Decimal('100.0'))
            
            volume_by_risk_raw = max_loss_allowed_dec / risk_per_lot_dec
            volume_by_risk_adjusted = (volume_by_risk_raw / lot_step).to_integral_value(rounding=ROUND_DOWN) * lot_step
            
            if signal.volume_lots is None: # اگر AI حجم نداده بود
                calculated_volume_lots_dec = volume_by_risk_adjusted
                if self.settings.log_risk_checks: logger.info(f"Volume by risk for {signal.symbol}: {calculated_volume_lots_dec} lots.")
            elif calculated_volume_lots_dec > volume_by_risk_adjusted:
                if self.settings.log_risk_checks: logger.warning(f"AI volume {signal.volume_lots} for {signal.symbol} exceeds risk limit. Adjusting to {volume_by_risk_adjusted}.")
                calculated_volume_lots_dec = volume_by_risk_adjusted
        
        # 4. بررسی حداکثر حجم پوزیشن نسبت به بالانس (اگر فعال است)
        if self.settings.max_position_size_pct_balance is not None and entry_price_for_calc:
            # این قانون باید بعد از محاسبه اولیه حجم بر اساس ریسک اعمال شود
            # یا اینکه حجم را مستقیما بر اساس این قانون هم محاسبه و مینیمم را انتخاب کنیم
            max_pos_value_dec = Decimal(str(current_balance)) * (Decimal(str(self.settings.max_position_size_pct_balance)) / Decimal('100.0'))
            value_per_lot_dec = entry_dec * contract_size
            
            if value_per_lot_dec <= Decimal('1e-9'):
                 return None, "Trade rejected: Contract size or entry price is zero, cannot calculate position value."

            volume_by_balance_limit_raw = max_pos_value_dec / value_per_lot_dec
            volume_by_balance_limit_adjusted = (volume_by_balance_limit_raw / lot_step).to_integral_value(rounding=ROUND_DOWN) * lot_step
            
            if calculated_volume_lots_dec == Decimal('0.0') and signal.volume_lots is None: # اگر حجم هنوز صفر است (مثلا ریسک بالا بود)
                calculated_volume_lots_dec = volume_by_balance_limit_adjusted # حجم را بر اساس این قانون بگذار
                if self.settings.log_risk_checks: logger.info(f"Volume by balance limit for {signal.symbol}: {calculated_volume_lots_dec} lots (as primary calc).")
            elif calculated_volume_lots_dec > volume_by_balance_limit_adjusted:
                if self.settings.log_risk_checks: logger.warning(f"Volume {calculated_volume_lots_dec} for {signal.symbol} exceeds max position size. Adjusting to {volume_by_balance_limit_adjusted}.")
                calculated_volume_lots_dec = volume_by_balance_limit_adjusted

        # 5. بررسی حداقل حجم و اینکه آیا حجم مثبت است
        if calculated_volume_lots_dec < min_lot:
            # اگر calculated_volume_lots_dec صفر شده (مثلا به دلیل محدودیت های بالا)،
            # و min_lot هم مثلا 0.01 است، این شرط برقرار می شود.
            if self.settings.log_risk_checks: logger.warning(f"Calculated volume {calculated_volume_lots_dec} for {signal.symbol} is below min lot {min_lot}. Rejecting.")
            return None, f"Trade rejected: Calculated volume {calculated_volume_lots_dec} is below min lot {min_lot}."

        if calculated_volume_lots_dec <= Decimal('0.0'): # باید مثبت باشد
            return None, f"Trade rejected: Final calculated volume {calculated_volume_lots_dec} is not positive."

        signal.volume_lots = float(calculated_volume_lots_dec)

        # 6. بررسی نهایی ریسک با حجم نهایی (این بررسی باید پس از تمام تعدیلات حجم انجام شود)
        if self.settings.max_risk_per_trade_pct is not None and signal.stop_loss_price:
            final_risk_amount_dec = abs(entry_dec - sl_dec) * calculated_volume_lots_dec * contract_size
            final_risk_pct_dec = (final_risk_amount_dec / Decimal(str(current_balance))) * Decimal('100.0')
            
            # با کمی تلورانس (مثلا 1.01 برای 1%)
            if final_risk_pct_dec > Decimal(str(self.settings.max_risk_per_trade_pct)) * Decimal('1.01'):
                return None, (f"Trade rejected: Final calculated risk {final_risk_pct_dec:.2f}% "
                              f"(for volume {signal.volume_lots} and SL {signal.stop_loss_price}) "
                              f"exceeds max risk per trade {self.settings.max_risk_per_trade_pct}%.")
            if self.settings.log_risk_checks: 
                logger.info(f"Trade for {signal.symbol} validated. Volume: {signal.volume_lots}, SL: {signal.stop_loss_price}, "
                            f"Calculated Risk: {final_risk_amount_dec:.2f} ({final_risk_pct_dec:.2f}% of balance).")
        elif self.settings.log_risk_checks:
            logger.info(f"Trade for {signal.symbol} validated (risk % not fully checked due to missing SL/risk setting). Volume: {signal.volume_lots}")
        
        return signal, "Trade signal validated and potentially adjusted by RiskGuardian."

    async def shutdown(self): # تغییر به async اگر نیاز به عملیات await دارد
        logger.info("RiskGuardian shutting down...")
        logger.info("RiskGuardian shutdown complete.")