# D:\AdvancedTradingSystem\modules\environments\trading_env.py
# نسخه کامل و نهایی شده با اصلاحات برای جلوگیری از هشدارها و بهبود پایداری

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PositionInfo:
    type: str  # 'long' or 'short'
    entry_price: float
    size: float
    entry_time: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

class AdvancedRiskManager:
    def __init__(self,
                 initial_balance: float,
                 max_position_size: float = 1.0,
                 max_leverage: float = 1.0,
                 position_sizing_method: str = 'fixed',
                 risk_per_trade: float = 0.02,
                 max_correlated_trades: int = 2,
                 max_daily_trades: int = 10):
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.position_sizing_method = position_sizing_method
        self.risk_per_trade = risk_per_trade
        self.max_correlated_trades = max_correlated_trades
        self.max_daily_trades = max_daily_trades
        
        self.daily_trades = 0
        self.last_trade_date = None
        self.correlated_positions = {}
        
    def calculate_position_size(self, current_balance: float, entry_price: float, stop_loss: float) -> float:
        if self.position_sizing_method == 'fixed':
            return self.initial_balance * self.max_position_size
        elif self.position_sizing_method == 'risk_based':
            if stop_loss == 0 or entry_price == 0:
                return 0
            risk_amount = current_balance * self.risk_per_trade
            position_size = risk_amount / abs(entry_price - stop_loss)
            return min(position_size, current_balance * self.max_position_size)
        return 0

    def can_open_position(self, symbol: str, current_time: pd.Timestamp) -> bool:
        # Reset daily counters if new day
        if self.last_trade_date is None or current_time.date() != self.last_trade_date:
            self.daily_trades = 0
            self.last_trade_date = current_time.date()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False
            
        # Check correlated positions
        correlated_count = sum(1 for s, count in self.correlated_positions.items() 
                             if s.startswith(symbol[:3]) or s.endswith(symbol[-3:]))
        if correlated_count >= self.max_correlated_trades:
            return False
            
        return True

    def update_trade_metrics(self, symbol: str, current_time: pd.Timestamp):
        self.daily_trades += 1
        self.last_trade_date = current_time.date()
        if symbol not in self.correlated_positions:
            self.correlated_positions[symbol] = 0
        self.correlated_positions[symbol] += 1

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 10}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 window_size: int = 60,
                 commission_pct: float = 0.0002,
                 max_episode_steps: Optional[int] = None,
                 reward_sharpe_ratio: bool = True,
                 normalize_features: bool = True,
                 symbol: str = "UNKNOWN_SYMBOL",
                 timeframe: str = "UNKNOWN_TIMEFRAME",
                 risk_manager_config: Optional[Dict] = None,
                 use_dynamic_features: bool = True,
                 use_multi_timeframe: bool = True):
        super().__init__()

        if df.empty:
            raise ValueError("Input DataFrame (df) cannot be empty.")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        # شرط طول DataFrame: باید حداقل به اندازه window_size برای مشاهده و یک گام برای اقدام و مشاهده بعدی باشد
        # و یک کندل هم برای قیمت بسته شدن در انتهای داده ها
        if len(df) < window_size + 2: 
            raise ValueError(f"DataFrame length ({len(df)}) is too short for window_size ({window_size}). Needs at least {window_size + 2}.")

        self.symbol = symbol
        self.timeframe_str = timeframe
        self.df = df.copy()
        self.initial_balance = float(initial_balance)
        if self.initial_balance <= 0: # جلوگیری از تقسیم بر صفر
            raise ValueError("initial_balance must be positive.")
        self.window_size = int(window_size)
        self.commission_pct = float(commission_pct)
        self.reward_sharpe_ratio = reward_sharpe_ratio
        self.normalize_input_features = normalize_features
        self.use_dynamic_features = use_dynamic_features
        self.use_multi_timeframe = use_multi_timeframe

        # Initialize risk manager
        risk_config = risk_manager_config or {}
        self.risk_manager = AdvancedRiskManager(
            initial_balance=self.initial_balance,
            **risk_config
        )

        if 'time' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['time']):
                self.df['time'] = pd.to_datetime(self.df['time'], errors='coerce')
            self.df = self.df.set_index('time')
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex or have 'time' column.")

        if not self.df.index.is_monotonic_increasing:
            self.df = self.df.sort_index()
            logger.warning(f"DataFrame for {self.symbol}-{self.timeframe_str} sorted by time index.")

        if self.df.index.tz is not None:
            self.df.index = self.df.index.tz_localize(None)
            logger.debug(f"Converted DataFrame index for {self.symbol}-{self.timeframe_str} to naive datetime.")

        # Add technical indicators if using dynamic features
        if self.use_dynamic_features:
            self._add_technical_indicators()

        self.feature_columns = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        if not self.feature_columns:
            raise ValueError("No numerical feature columns found in DataFrame.")
        logger.info(f"TradingEnv for {self.symbol}-{self.timeframe_str} using {len(self.feature_columns)} features: {self.feature_columns}")

        # Normalize features
        self._feature_scalers = {}
        if self.normalize_input_features:
            logger.debug(f"Normalizing input features for {self.symbol}-{self.timeframe_str}...")
            for col in self.feature_columns:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if pd.isna(min_val) or pd.isna(max_val): # اگر ستون فقط NaN دارد
                    logger.warning(f"Column '{col}' for {self.symbol}-{self.timeframe_str} contains all NaNs before normalization. Setting to 0.")
                    self.df[col] = 0.0
                    self._feature_scalers[col] = (0.0, 0.0)
                elif max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    self._feature_scalers[col] = (min_val, max_val)
                elif max_val == min_val and max_val != 0 :
                     self.df[col] = 0.5
                     self._feature_scalers[col] = (min_val, max_val)
                else: 
                    self.df[col] = 0.0
                    self._feature_scalers[col] = (min_val, max_val)
            # پس از نرمال سازی، NaN ها را با 0 پر کن (اگر هنوز وجود دارند)
            self.df[self.feature_columns] = self.df[self.feature_columns].fillna(0.0)

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1, 0, 0]),  # Position size (-1 to 1), SL %, TP %
            high=np.array([1, 0.1, 0.3]),  # Max position, max 10% SL, max 30% TP
            dtype=np.float32
        )

        n_features = len(self.feature_columns)
        if self.use_multi_timeframe:
            n_features *= 3  # Include features from multiple timeframes

        self.observation_shape = (self.window_size * n_features + 5,)  # +5 for account metrics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=self.observation_shape,
            dtype=np.float32
        )
        
        if max_episode_steps is None or max_episode_steps <= 0:
            self._max_episode_steps = len(self.df) - self.window_size - 1 
        else:
            self._max_episode_steps = int(max_episode_steps)
        
        if self._max_episode_steps <=0: # اگر پس از محاسبات، منفی یا صفر شد
            raise ValueError(f"Calculated _max_episode_steps ({self._max_episode_steps}) is not positive. DataFrame too short or window_size too large.")

        self.current_step_in_df: int = 0
        self._current_episode_step_count: int = 0
        self.balance: float = 0.0
        self.net_worth: float = 0.0
        self.current_position: int = 0
        self.entry_price: float = 0.0
        self._episode_rewards_log: List[float] = []
        self._trade_history: List[Dict[str, Any]] = []

        # Add new tracking metrics
        self.trade_metrics = {
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Experience tracking
        self.trade_experience_buffer = deque(maxlen=1000)
        self.market_state_buffer = deque(maxlen=1000)

    def _add_technical_indicators(self):
        """Add advanced technical indicators to the DataFrame"""
        # Trend indicators
        self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['sma_50'] = self.df['close'].rolling(window=50).mean()
        self.df['ema_20'] = self.df['close'].ewm(span=20, adjust=False).mean()
        
        # Volatility indicators
        self.df['atr'] = self._calculate_atr(14)
        
        # Momentum indicators
        self.df['rsi'] = self._calculate_rsi(14)
        self.df['macd'], self.df['macd_signal'] = self._calculate_macd()
        
        # Volume indicators
        if 'volume' in self.df.columns:
            self.df['volume_sma'] = self.df['volume'].rolling(window=20).mean()
            self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma']

        # Fill NaN values
        self.df = self.df.fillna(method='bfill')

    def _calculate_atr(self, period: int) -> pd.Series:
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _calculate_rsi(self, period: int) -> pd.Series:
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def _get_observation(self) -> np.ndarray:
        start_idx = self.current_step_in_df
        end_idx = start_idx + self.window_size
        
        # اطمینان از اینکه end_idx از طول df بیشتر نمی زند
        if end_idx > len(self.df):
            logger.error(f"Observation window end_idx {end_idx} exceeds DataFrame length {len(self.df)} at step {self.current_step_in_df}.")
            # این حالت نباید رخ دهد اگر منطق اتمام اپیزود درست باشد
            # برای جلوگیری از خطا، آخرین پنجره ممکن را برمی گردانیم
            end_idx = len(self.df)
            start_idx = end_idx - self.window_size
            if start_idx < 0: start_idx = 0 # اگر df خیلی کوتاه است

        market_window_data = self.df[self.feature_columns].iloc[start_idx:end_idx].values

        if market_window_data.shape[0] < self.window_size:
             padding_shape = (self.window_size - market_window_data.shape[0], market_window_data.shape[1])
             if padding_shape[0] > 0 : # فقط اگر واقعا نیاز به پدینگ است
                logger.warning(f"Padding observation window at step {self.current_step_in_df}. Shape was {market_window_data.shape}, padding with {padding_shape[0]} rows.")
                padding = np.zeros(padding_shape, dtype=market_window_data.dtype)
                market_window_data = np.vstack((padding, market_window_data)) # پدینگ در ابتدا

        normalized_balance = self.balance / self.initial_balance if self.initial_balance != 0 else 0.0
        position_status = float(self.current_position)

        obs = np.concatenate((market_window_data.flatten(), [normalized_balance, position_status])).astype(np.float32)
        
        if np.isnan(obs).any() or np.isinf(obs).any():
            logger.warning(f"NaN/Inf in observation at step {self.current_step_in_df} for {self.symbol}-{self.timeframe_str}. Replacing with 0.")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.current_position = 0
        self.entry_price = 0.0
        self._episode_rewards_log = []
        self._trade_history = []
        self._current_episode_step_count = 0
        self.current_step_in_df = 0 # همیشه از ابتدای داده های قابل استفاده شروع کن

        logger.debug(f"Env RESET for {self.symbol}-{self.timeframe_str}: Init Balance={self.balance:.2f}, Start DF Step={self.current_step_in_df}")
        return self._get_observation(), {}

    def _get_current_price(self) -> float:
        idx = self.current_step_in_df + self.window_size - 1
        if idx < len(self.df):
            return self.df['close'].iloc[idx]
        logger.error(f"Attempted to get price beyond DataFrame bounds! current_step_in_df={self.current_step_in_df}, window_size={self.window_size}, df_len={len(self.df)}")
        return self.df['close'].iloc[-1] 

    def _update_trade_metrics(self, pnl: float):
        """Update trading metrics after each trade"""
        self.trade_metrics['total_trades'] += 1
        
        if pnl > 0:
            self.trade_metrics['winning_trades'] += 1
            self.trade_metrics['current_consecutive_wins'] += 1
            self.trade_metrics['current_consecutive_losses'] = 0
            self.trade_metrics['max_consecutive_wins'] = max(
                self.trade_metrics['max_consecutive_wins'],
                self.trade_metrics['current_consecutive_wins']
            )
        else:
            self.trade_metrics['losing_trades'] += 1
            self.trade_metrics['current_consecutive_losses'] += 1
            self.trade_metrics['current_consecutive_wins'] = 0
            self.trade_metrics['max_consecutive_losses'] = max(
                self.trade_metrics['max_consecutive_losses'],
                self.trade_metrics['current_consecutive_losses']
            )
        
        # Update win rate
        if self.trade_metrics['total_trades'] > 0:
            self.trade_metrics['win_rate'] = (
                self.trade_metrics['winning_trades'] / self.trade_metrics['total_trades']
            )
        
        # Update average win/loss
        if self.trade_metrics['winning_trades'] > 0:
            self.trade_metrics['avg_win'] = (
                (self.trade_metrics['avg_win'] * (self.trade_metrics['winning_trades'] - 1) + max(0, pnl))
                / self.trade_metrics['winning_trades']
            )
        if self.trade_metrics['losing_trades'] > 0:
            self.trade_metrics['avg_loss'] = (
                (self.trade_metrics['avg_loss'] * (self.trade_metrics['losing_trades'] - 1) + min(0, pnl))
                / self.trade_metrics['losing_trades']
            )
        
        # Update profit factor
        total_profits = self.trade_metrics['avg_win'] * self.trade_metrics['winning_trades']
        total_losses = abs(self.trade_metrics['avg_loss'] * self.trade_metrics['losing_trades'])
        self.trade_metrics['profit_factor'] = total_profits / total_losses if total_losses > 0 else float('inf')

    def _record_trade_experience(self, action: np.ndarray, reward: float, done: bool):
        """Record trading experience for learning"""
        current_state = self._get_observation()
        market_data = self.df.iloc[self.current_step_in_df:self.current_step_in_df + self.window_size].copy()
        
        experience = {
            'state': current_state,
            'action': action,
            'reward': reward,
            'done': done,
            'market_data': market_data,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'position': self.current_position,
            'metrics': self.trade_metrics.copy(),
            'timestamp': self.df.index[self.current_step_in_df + self.window_size - 1]
        }
        
        self.trade_experience_buffer.append(experience)
        self.market_state_buffer.append(market_data)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._current_episode_step_count += 1
        current_price_for_action = self._get_current_price()
        
        reward_this_step = 0.0
        terminated = False
        truncated = False
        previous_net_worth = self.net_worth

        # Parse action
        position_size, stop_loss_pct, take_profit_pct = action
        
        if abs(position_size) > 0.1 and self.risk_manager.can_open_position(self.symbol, self.df.index[self.current_step_in_df + self.window_size - 1]):
            position_type = 'long' if position_size > 0 else 'short'
            size = self.risk_manager.calculate_position_size(
                self.balance,
                current_price_for_action,
                current_price_for_action * (1 - stop_loss_pct if position_type == 'long' else 1 + stop_loss_pct)
            )
            
            if size > 0:
                stop_loss = current_price_for_action * (1 - stop_loss_pct if position_type == 'long' else 1 + stop_loss_pct)
                take_profit = current_price_for_action * (1 + take_profit_pct if position_type == 'long' else 1 - take_profit_pct)
                
                self._trade_history.append({"step": self.current_step_in_df, "type": position_type, "price": current_price_for_action, "pnl": 0, "balance": self.balance})
                logger.debug(f"Opened {position_type} at {current_price_for_action:.5f}")

                self.current_position = 1 if position_type == 'long' else -1
                self.entry_price = current_price_for_action
                commission_open = self.balance * self.commission_pct
                self.balance -= commission_open
                self._trade_history[-1]["pnl"] = -commission_open
                self._trade_history[-1]["balance"] = self.balance
        
        self.net_worth = self.balance
        if self.entry_price != 0: # فقط اگر پوزیشن باز است و قیمت ورود معتبر است
            if self.current_position == 1:
                self.net_worth += (current_price_for_action - self.entry_price) * (self.initial_balance / self.entry_price)
            elif self.current_position == -1:
                self.net_worth += (self.entry_price - current_price_for_action) * (self.initial_balance / self.entry_price)
        
        reward_this_step = (self.net_worth - previous_net_worth) / self.initial_balance if self.initial_balance != 0 else 0.0
        self._episode_rewards_log.append(reward_this_step)

        self.current_step_in_df += 1

        if self.initial_balance != 0 and self.net_worth <= self.initial_balance * 0.5:
            logger.warning(f"Net worth {self.net_worth:.2f} below 50% of initial. Episode terminated.")
            terminated = True; reward_this_step -= 1.0 
        
        if self._current_episode_step_count >= self._max_episode_steps:
            logger.debug(f"Max episode steps ({self._max_episode_steps}) reached. Episode truncated.")
            truncated = True
        
        if self.current_step_in_df + self.window_size >= len(self.df): # به انتهای داده ها رسیده ایم
            logger.info(f"End of DataFrame for {self.symbol}-{self.timeframe_str}. Episode terminated.")
            terminated = True
            if self.current_position != 0 and self.entry_price != 0:
                last_price = self.df['close'].iloc[-1]
                final_pnl = 0
                if self.current_position == 1: final_pnl = (last_price - self.entry_price) * (self.initial_balance / self.entry_price)
                elif self.current_position == -1: final_pnl = (self.entry_price - last_price) * (self.initial_balance / self.entry_price)
                
                self.balance += final_pnl - (self.balance * self.commission_pct)
                self.net_worth = self.balance
                reward_this_step += final_pnl / self.initial_balance if self.initial_balance != 0 else 0.0
                self.current_position = 0; self.entry_price = 0.0
                logger.info(f"End of data: Closed open position. Final PnL: {final_pnl:.2f}. Final Net Worth: {self.net_worth:.2f}")

        final_reward_for_episode = reward_this_step
        if terminated or truncated:
            total_pnl_pct = (self.net_worth - self.initial_balance) / self.initial_balance * 100 if self.initial_balance != 0 else 0.0
            final_reward_for_episode = total_pnl_pct / 100.0 # نرمال شده به درصد

            if self.reward_sharpe_ratio and len(self._episode_rewards_log) > self.window_size : 
                rewards_arr = np.array(self._episode_rewards_log)
                if np.std(rewards_arr) > 1e-9: 
                    # تقریب ساده شارپ ریشیو
                    sharpe_approx = np.mean(rewards_arr) / np.std(rewards_arr) 
                    # ضریب برای سالانه کردن (بسیار تقریبی)
                    annual_factor = np.sqrt(252 if "D" in self.timeframe_str else 252*24 if "H" in self.timeframe_str else 252*24*12 if "M5" in self.timeframe_str else 252*24*60)
                    sharpe_approx *= annual_factor
                    
                    if sharpe_approx > -5: # جلوگیری از مقادیر بسیار منفی شارپ
                        final_reward_for_episode += sharpe_approx * 0.01 # تاثیر کم شارپ
                    logger.info(f"EP END: PnL%={total_pnl_pct:.2f}%, Approx Sharpe={sharpe_approx:.2f}, Final Reward={final_reward_for_episode:.4f}")
                else:
                    logger.info(f"EP END: PnL%={total_pnl_pct:.2f}%, Sharpe N/A (std=0), Final Reward={final_reward_for_episode:.4f}")
            else:
                 logger.info(f"EP END: PnL%={total_pnl_pct:.2f}%, Final Reward={final_reward_for_episode:.4f} (Sharpe not used or not enough data)")
        
        self._record_trade_experience(action, reward_this_step, terminated or truncated)
        
        if self.current_position == 0 and self.entry_price != 0:  # Trade just closed
            pnl = self.net_worth - previous_net_worth
            self._update_trade_metrics(pnl)
        
        next_observation = self._get_observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)
        
        info = {
            "balance": self.balance, "net_worth": self.net_worth, "position": self.current_position,
            "entry_price": self.entry_price if self.current_position != 0 else 0,
            "pnl_pct_episode": total_pnl_pct if 'total_pnl_pct' in locals() and (terminated or truncated) else 0,
            "sharpe_approx_episode": sharpe_approx if 'sharpe_approx' in locals() and (terminated or truncated) else 0,
            "trade_history_len": len(self._trade_history) # فقط طول تاریخچه برای کاهش حجم info
        }
        return next_observation, final_reward_for_episode, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'ansi':
            return (f"Step: {self._current_episode_step_count}, DF_Step: {self.current_step_in_df}, "
                    f"NetWorth: {self.net_worth:.2f}, Pos: {self.current_position}, "
                    f"Entry: {self.entry_price:.5f if self.current_position != 0 else 'N/A'}")
        elif mode == 'human':
            print(self.render(mode='ansi'))

    def close(self):
        logger.debug(f"TradingEnv for {self.symbol}-{self.timeframe_str} closed.")
        pass

    def get_trade_analytics(self) -> Dict[str, Any]:
        """Get detailed trading analytics"""
        return {
            'metrics': self.trade_metrics,
            'recent_experiences': list(self.trade_experience_buffer)[-10:],  # Last 10 trades
            'market_states': list(self.market_state_buffer)[-10:],  # Last 10 market states
        }