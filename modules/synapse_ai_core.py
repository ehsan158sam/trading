# D:\AdvancedTradingSystem\modules\synapse_ai_core.py
# Reverted to asyncio.to_thread for learn, with more detailed logging

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List 
import os
import shutil 
import sys
import time # برای تست تاخیر

import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
from stable_baselines3 import PPO, SAC, A2C 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from modules.fundamental_analyzer import FundamentalAnalyzer
from modules.advanced_analysis.market_regime_analyzer import MarketRegimeAnalyzer
from modules.advanced_analysis.ensemble_predictor import EnsemblePredictor
from modules.advanced_analysis.portfolio_optimizer import PortfolioOptimizer

try:
    from config.schemas import (
        SynapseAISettings, AppSettings, DRLAlgorithmSettings,
        TrainingSettings, PolicyNetworkSettings, TimeFrame 
    )
    from modules.data_nexus import DataNexus
    from modules.environments.trading_env import TradingEnv 
except ImportError as e:
    print(f"CRITICAL WARNING (synapse_ai_core.py): Could not import dependencies: {e}")
    # Fallback classes
    class SynapseAISettings: base_model_path = "./models_ai_fallback"; training = None; algorithm = None; policy_network = None; env_window_size=10; env_max_episode_steps=None; env_initial_balance=10000; env_commission_pct=0.0; env_reward_sharpe_ratio=False; env_normalize_features=False
    class AppSettings: pass
    class DRLAlgorithmSettings: name="PPO"; learning_rate=0.0003; gamma=0.99; gae_lambda=0.95; n_steps=128; batch_size=64; n_epochs=4; clip_range=0.2; ent_coef=0.01
    class TrainingSettings: total_timesteps_per_symbol_timeframe=1000; log_interval=1; save_freq=500; eval_freq=200; n_eval_episodes=1; use_wandb=False; wandb_project_name=None; wandb_api_key=None
    class PolicyNetworkSettings: shared_layers=None; policy_layers=None; value_layers=None; activation_fn="Tanh"
    class DataNexus: settings = None; get_market_data = lambda s,sy,tf,nc=None: pd.DataFrame()
    class TradingEnv: pass
    class TimeFrame(str): M1 = "M1"

logger = logging.getLogger(__name__)

SB3_ALGORITHMS = { "PPO": PPO, "SAC": SAC, "A2C": A2C }
ACTIVATION_FNS = { "Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU }

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))

class AdvancedTradingNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 window_size: int,
                 n_assets: int = 1,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_transformer_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, window_size, d_model))
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.asset_embedding = nn.Parameter(torch.randn(n_assets, d_model))
        
        self.meta_learning_layer = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)  # 3 actions: Buy, Hold, Sell
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, asset_idx: int = 0):
        batch_size = x.shape[0]
        
        # Project input and add positional encoding
        x = self.input_projection(x)
        x = x + self.position_encoding
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Add asset-specific information
        asset_info = self.asset_embedding[asset_idx].unsqueeze(0).expand(batch_size, -1)
        
        # Apply meta-learning layer
        meta_output, _ = self.meta_learning_layer(x)
        
        # Combine features
        combined_features = torch.cat([
            meta_output[:, -1],  # Take the last output from LSTM
            asset_info
        ], dim=1)
        
        # Generate policy and value outputs
        policy_output = self.output_layer(combined_features)
        value_output = self.value_head(combined_features)
        
        return policy_output, value_output

class AdvancedMemoryBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, experience: Dict[str, Any]):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class OnlineLearningOptimizer:
    def __init__(self, 
                 learning_rate: float = 0.001,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 update_frequency: int = 100,
                 min_experiences: int = 500):
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.min_experiences = min_experiences
        self.experience_buffer = []
        self.performance_history = []
        self.update_counter = 0
        
    def add_experience(self, experience: Dict[str, Any]):
        """Add new trading experience to buffer"""
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.memory_size:
            self.experience_buffer.pop(0)
            
    def should_update(self) -> bool:
        """Check if it's time to perform an update"""
        self.update_counter += 1
        return (len(self.experience_buffer) >= self.min_experiences and 
                self.update_counter >= self.update_frequency)
    
    def get_training_batch(self) -> List[Dict[str, Any]]:
        """Get a batch of experiences for training"""
        if len(self.experience_buffer) < self.batch_size:
            return self.experience_buffer
        return random.sample(self.experience_buffer, self.batch_size)
    
    def reset_update_counter(self):
        """Reset the update counter after performing an update"""
        self.update_counter = 0
        
    def add_performance_metric(self, metric: float):
        """Track performance metrics"""
        self.performance_history.append(metric)
        if len(self.performance_history) > 100:  # Keep last 100 metrics
            self.performance_history.pop(0)
            
    def get_performance_trend(self) -> float:
        """Calculate performance trend"""
        if len(self.performance_history) < 2:
            return 0.0
        return np.mean(np.diff(self.performance_history))

class SynapseAICore:
    def __init__(self, settings: SynapseAISettings, app_settings: AppSettings, data_nexus: DataNexus):
        self.settings = settings
        self.app_settings = app_settings
        self.data_nexus = data_nexus
        self.models: Dict[str, Dict[str, Any]] = {}
        base_path_str = getattr(self.settings, 'base_model_path', "./models_ai_fallback_synapse")
        self.base_model_path = Path(base_path_str).resolve()
        self.base_model_path.mkdir(parents=True, exist_ok=True)
        self._active_training_tasks: Dict[str, asyncio.Task] = {}
        self._training_locks: Dict[str, asyncio.Lock] = {} # یک قفل برای هر تسک آموزش
        
        # اضافه کردن حافظه تجربه برای هر نماد و تایم‌فریم
        self.experience_buffers: Dict[str, AdvancedMemoryBuffer] = {}
        
        # تنظیمات پیشرفته برای یادگیری
        self.meta_learning_config = {
            'd_model': 64,
            'nhead': 4,
            'num_transformer_layers': 2,
            'dropout': 0.1,
            'learning_rate': 3e-4,
            'meta_batch_size': 32,
            'update_interval': 100
        }

        self.fundamental_analyzer = FundamentalAnalyzer()
        self.fundamental_data_cache = {}
        self.fundamental_update_interval = 3600  # 1 hour

        # Initialize advanced components
        self.market_regime_analyzer = MarketRegimeAnalyzer()
        self.ensemble_predictor = EnsemblePredictor({
            'input_dim': 13,  # Number of features
            'hidden_dims': [64, 32, 16]
        })
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Add state tracking
        self.current_market_regime = {}
        self.prediction_history = {}
        self.optimization_history = {}

    def _get_model_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        # ... (بدون تغییر) ...
        algo_name = getattr(getattr(self.settings, 'algorithm', object()), 'name', 'unknown_algo')
        tf_value = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        model_dir = self.base_model_path / f"{algo_name.lower()}_{symbol.lower()}_{tf_value.lower()}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / "best_model.zip"

    def _get_vecnormalize_stats_path(self, symbol: str, timeframe: TimeFrame) -> Path:
        # ... (بدون تغییر) ...
        algo_name = getattr(getattr(self.settings, 'algorithm', object()), 'name', 'unknown_algo')
        tf_value = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        model_dir = self.base_model_path / f"{algo_name.lower()}_{symbol.lower()}_{tf_value.lower()}"
        return model_dir / "vec_normalize_stats.pkl"

    def _create_env(self, df_market_data: pd.DataFrame, symbol: str, timeframe: TimeFrame, for_evaluation: bool = False) -> TradingEnv:
        # ... (بدون تغییر) ...
        if df_market_data is None or df_market_data.empty:
            raise ValueError(f"Market data for {symbol} {timeframe.value if hasattr(timeframe, 'value') else str(timeframe)} is empty.")
        
        env_win_size = getattr(self.settings, 'env_window_size', 60)
        if len(df_market_data) < env_win_size + 2: 
            raise ValueError(f"DataFrame length ({len(df_market_data)}) too short for window_size ({env_win_size}). Needs at least {env_win_size + 2}.")

        max_ep_steps_setting = getattr(self.settings, 'env_max_episode_steps', None)
        max_steps = max_ep_steps_setting
        if for_evaluation:
            available_data_for_eval = len(df_market_data) - env_win_size -1 
            max_steps = available_data_for_eval if available_data_for_eval > 0 else 1
        
        df_to_pass = df_market_data.copy()
        if 'time' in df_to_pass.columns and df_to_pass.index.name != 'time':
            df_to_pass = df_to_pass.set_index('time')
        if df_to_pass.index.tz is not None: 
            df_to_pass.index = df_to_pass.index.tz_localize(None)

        logger.debug(f"SAI_CREATE_ENV_V2: Creating TradingEnv for {symbol} {timeframe.value if hasattr(timeframe, 'value') else str(timeframe)}. DF len: {len(df_to_pass)}, WinSize: {env_win_size}, MaxSteps: {max_steps}")
        return TradingEnv(
            df=df_to_pass,
            initial_balance=getattr(self.settings, 'env_initial_balance', 10000.0),
            window_size=env_win_size,
            commission_pct=getattr(self.settings, 'env_commission_pct', 0.0002),
            max_episode_steps=max_steps,
            reward_sharpe_ratio=getattr(self.settings, 'env_reward_sharpe_ratio', True),
            normalize_features=getattr(self.settings, 'env_normalize_features', True),
            symbol=symbol,
            timeframe=timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        )

    async def train_model_for_symbol_timeframe(self, symbol: str, timeframe: TimeFrame, use_subproc_env: bool = False, num_cpu: int = 1):
        tf_value = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        task_key = f"{symbol}_{tf_value}"
        
        # استفاده از قفل برای هر تسک آموزش برای جلوگیری از اجرای همزمان چندباره
        if task_key not in self._training_locks:
            self._training_locks[task_key] = asyncio.Lock()
        
        async with self._training_locks[task_key]: # فقط یک تسک آموزش برای هر task_key می تواند این بخش را اجرا کند
            if task_key in self._active_training_tasks and not self._active_training_tasks[task_key].done():
                logger.warning(f"SAI_TRAIN_V3: Training for {task_key} is ALREADY ACTIVE. Skipping duplicate call (lock acquired).")
                return
            
            # اگر تسک قبلی تمام شده، آن را از لیست حذف کن
            if task_key in self._active_training_tasks and self._active_training_tasks[task_key].done():
                logger.info(f"SAI_TRAIN_V3: Previous training task for {task_key} was done. Removing to allow new training.")
                self._active_training_tasks.pop(task_key, None)

            logger.info(f"SAI_TRAIN_V3: Starting DRL model training process for {task_key}...")
            
            current_task = asyncio.current_task()
            if current_task: # فقط اگر در محیط asyncio هستیم
                self._active_training_tasks[task_key] = current_task
                logger.debug(f"SAI_TRAIN_V3: Task {task_key} (id: {id(current_task)}) added to _active_training_tasks.")
            else: # این حالت نباید رخ دهد اگر از schedule_training_for_all_configured صدا زده شده باشد
                 logger.warning(f"SAI_TRAIN_V3: Could not get current_task for {task_key}. Monitoring might be affected.")


            # ... (بقیه کد تا قبل از model.learn() مانند قبل با لاگ های V3) ...
            algo_settings = getattr(self.settings, 'algorithm', DRLAlgorithmSettings())
            training_settings_obj = getattr(self.settings, 'training', TrainingSettings())
            policy_settings_obj = getattr(self.settings, 'policy_network', PolicyNetworkSettings())
            env_win_size = getattr(self.settings, 'env_window_size', 60)
            env_max_ep_steps = getattr(self.settings, 'env_max_episode_steps', None)

            logger.debug(f"SAI_TRAIN_V3: Fetching market data for {task_key}...")
            full_df = self.data_nexus.get_market_data(symbol, timeframe)
            
            min_len_for_training = env_win_size + (env_max_ep_steps or 100) + 20
            if full_df is None or full_df.empty or len(full_df) < min_len_for_training:
                logger.error(f"SAI_TRAIN_V3: Not enough data for {task_key} to train. Need: {min_len_for_training}, Got: {len(full_df) if full_df is not None else 0}.")
                self._active_training_tasks.pop(task_key, None)
                return

            train_split_idx = int(len(full_df) * 0.8)
            train_df = full_df.iloc[:train_split_idx]
            eval_df = full_df.iloc[train_split_idx:]
            logger.debug(f"SAI_TRAIN_V3: Data split for {task_key}. Train len: {len(train_df)}, Eval len: {len(eval_df)}")

            min_steps_for_one_learn_call = getattr(algo_settings, 'n_steps', 128)
            if len(train_df) < env_win_size + min_steps_for_one_learn_call + 5:
                logger.error(f"SAI_TRAIN_V3: Training data for {task_key} too short. Need df len > {env_win_size + min_steps_for_one_learn_call + 5}, Got: {len(train_df)}. Skipping.")
                self._active_training_tasks.pop(task_key, None)
                return
            
            actual_num_cpu = num_cpu if use_subproc_env and os.name != 'nt' else 1
            train_env = None
            try:
                if actual_num_cpu > 1:
                    logger.info(f"SAI_TRAIN_V3: Using SubprocVecEnv with {actual_num_cpu} CPUs for {task_key}.")
                    train_env = SubprocVecEnv([lambda i=i: Monitor(self._create_env(train_df.copy(), symbol, timeframe), filename=None) for i in range(actual_num_cpu)])
                else:
                    if use_subproc_env and os.name == 'nt': logger.warning("SAI_TRAIN_V3: SubprocVecEnv disabled on Windows, using DummyVecEnv.")
                    logger.info(f"SAI_TRAIN_V3: Using DummyVecEnv for {task_key}.")
                    train_env = DummyVecEnv([lambda: Monitor(self._create_env(train_df.copy(), symbol, timeframe), filename=None)])
            except ValueError as e_env_train:
                logger.error(f"SAI_TRAIN_V3: ValueError creating training environment for {task_key}: {e_env_train}", exc_info=True)
                self._active_training_tasks.pop(task_key, None)
                if train_env and hasattr(train_env, 'close'): train_env.close()
                return
            
            eval_env_for_callback = None
            min_len_eval_df = env_win_size + training_settings_obj.n_eval_episodes * 5 + 2 
            if not eval_df.empty and len(eval_df) > min_len_eval_df:
                logger.info(f"SAI_TRAIN_V3: Creating evaluation environment for {task_key} with {len(eval_df)} candles.")
                try:
                    eval_env_raw = Monitor(self._create_env(eval_df.copy(), symbol, timeframe, for_evaluation=True), filename=None)
                    eval_env_for_callback = eval_env_raw
                except ValueError as e_eval_env_create:
                     logger.error(f"SAI_TRAIN_V3: ValueError creating evaluation environment for {task_key}: {e_eval_env_create}. Eval will be skipped.", exc_info=True)
            else:
                logger.warning(f"SAI_TRAIN_V3: Evaluation skipped for {task_key}. Need eval_df len > {min_len_eval_df}, Got: {len(eval_df)}")

            policy_kwargs = dict(
                net_arch = dict(
                    pi = (policy_settings_obj.shared_layers or []) + (policy_settings_obj.policy_layers or []),
                    vf = (policy_settings_obj.shared_layers or []) + (policy_settings_obj.value_layers or [])
                ) if (policy_settings_obj.policy_layers or policy_settings_obj.value_layers or policy_settings_obj.shared_layers) else None,
                activation_fn=ACTIVATION_FNS.get(getattr(policy_settings_obj, 'activation_fn', "Tanh"), torch.nn.Tanh)
            )
            if policy_kwargs.get('net_arch') is None : del policy_kwargs['net_arch']
            logger.debug(f"SAI_TRAIN_V3: Policy kwargs for {task_key}: {policy_kwargs}")

            AlgorithmClass = SB3_ALGORITHMS.get(algo_settings.name)
            if not AlgorithmClass: logger.error(f"SAI_TRAIN_V3: Unsupported DRL algorithm: {algo_settings.name}"); train_env.close(); self._active_training_tasks.pop(task_key, None); return

            model_save_parent_path = self._get_model_path(symbol, timeframe).parent
            model = None
            try:
                logger.info(f"SAI_TRAIN_V3: Initializing DRL model {algo_settings.name} for {task_key}...")
                model = AlgorithmClass(
                    "MlpPolicy", train_env, policy_kwargs=policy_kwargs,
                    learning_rate=algo_settings.learning_rate, gamma=algo_settings.gamma,
                    gae_lambda=getattr(algo_settings, 'gae_lambda', 0.95), 
                    n_steps=getattr(algo_settings, 'n_steps', 1024),     
                    batch_size=getattr(algo_settings, 'batch_size', 64), 
                    n_epochs=getattr(algo_settings, 'n_epochs', 10),     
                    clip_range=getattr(algo_settings, 'clip_range', 0.2), 
                    ent_coef=getattr(algo_settings, 'ent_coef', 0.005),
                    verbose=1,
                    tensorboard_log=str(self.base_model_path / "tensorboard_logs" / task_key) if not training_settings_obj.use_wandb else None,
                    device="cpu"
                )
            except Exception as e_model_init_v3:
                logger.critical(f"SAI_TRAIN_V3: Failed to init DRL model {algo_settings.name} for {task_key}: {e_model_init_v3}", exc_info=True)
                train_env.close(); self._active_training_tasks.pop(task_key, None); return

            logger.info(f"SAI_TRAIN_V3: DRL Model {algo_settings.name} created for {task_key}. Policy structure: {model.policy}")

            callbacks = []
            if eval_env_for_callback:
                eval_callback = EvalCallback(
                    eval_env_for_callback, best_model_save_path=str(model_save_parent_path),
                    log_path=str(model_save_parent_path / "eval_logs"),
                    eval_freq=max(training_settings_obj.eval_freq // actual_num_cpu, 1),
                    n_eval_episodes=training_settings_obj.n_eval_episodes, deterministic=True, render=False
                )
                callbacks.append(eval_callback)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=max(training_settings_obj.save_freq // actual_num_cpu, 1),
                save_path=str(model_save_parent_path / "checkpoints"),
                name_prefix=f"{algo_settings.name.lower()}_{symbol.lower()}_{tf_value.lower()}"
            )
            callbacks.append(checkpoint_callback)

            wandb_run_for_this_task_v3 = None 
            if training_settings_obj.use_wandb and getattr(training_settings_obj, 'wandb_api_key', None):
                try:
                    import wandb
                    from wandb.integration.sb3 import WandbCallback
                    run_name_wandb_task_v3 = f"{algo_settings.name}_{task_key}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    wandb_config_dict_task_v3 = self.settings.model_dump(mode='json') if hasattr(self.settings, 'model_dump') else self.settings.dict()
                    
                    wandb_run_for_this_task_v3 = wandb.init(
                        project=training_settings_obj.wandb_project_name or "SynapseAI_Trading_Default", 
                        name=run_name_wandb_task_v3, 
                        sync_tensorboard=True, 
                        config=wandb_config_dict_task_v3,
                        save_code=True,
                        reinit=True 
                    )
                    callbacks.append(WandbCallback(
                        model_save_path=str(model_save_parent_path / f"wandb_models/{run_name_wandb_task_v3}"),
                        save_freq=max(training_settings_obj.save_freq // actual_num_cpu, 1),
                        verbose=2,
                    ))
                    logger.info(f"SAI_TRAIN_V3: WandbCallback configured for {task_key} with run name {run_name_wandb_task_v3}.")
                except ImportError: logger.warning("SAI_TRAIN_V3: wandb not installed. Skipping WandbCallback.")
                except Exception as e_wandb_task_v3: logger.error(f"SAI_TRAIN_V3: Error initializing WandbCallback for {task_key}: {e_wandb_task_v3}", exc_info=True)
            elif training_settings_obj.use_wandb:
                 logger.warning("SAI_TRAIN_V3: use_wandb is true but wandb_api_key is not set. Skipping Wandb.")


            try:
                logger.info(f"SAI_TRAIN_V3: >>> Attempting to start ASYNCHRONOUS model.learn() for {task_key} with {training_settings_obj.total_timesteps_per_symbol_timeframe} timesteps.")
                
                # --- بازگشت به اجرای ناهمزمان با asyncio.to_thread ---
                learn_kwargs = {
                    "total_timesteps": training_settings_obj.total_timesteps_per_symbol_timeframe,
                    "callback": callbacks if callbacks else None,
                    "log_interval": training_settings_obj.log_interval, 
                    "reset_num_timesteps": False 
                }
                logger.debug(f"SAI_TRAIN_V3: Calling asyncio.to_thread with model.learn and kwargs: {learn_kwargs}")
                await asyncio.to_thread(model.learn, **learn_kwargs)
                # --- پایان اجرای ناهمزمان ---

                logger.info(f"SAI_TRAIN_V3: <<< model.learn() completed for {task_key}.")
                
                final_model_path = model_save_parent_path / "final_model.zip"
                await asyncio.to_thread(model.save, final_model_path)
                logger.info(f"SAI_TRAIN_V3: Training complete for {task_key}. Final model saved to {final_model_path}")

                if symbol not in self.models: self.models[symbol] = {}
                self.models[symbol][tf_value] = {"model_path": str(self._get_model_path(symbol, timeframe)), "algorithm": algo_settings.name}
            
            except RuntimeError as e_runtime_learn_v3:
                logger.error(f"SAI_TRAIN_V3: RuntimeError during model training for {task_key}: {e_runtime_learn_v3}", exc_info=True)
            except Exception as e_learn_main_v3:
                logger.error(f"SAI_TRAIN_V3: Error during model.learn() or save for {task_key}: {e_learn_main_v3}", exc_info=True)
            finally:
                logger.info(f"SAI_TRAIN_V3: Cleaning up resources for training task {task_key}...")
                if train_env and hasattr(train_env, 'close'):
                    try: train_env.close()
                    except Exception as e_close_train_v3: logger.error(f"Error closing train_env for {task_key}: {e_close_train_v3}")
                if eval_env_for_callback and hasattr(eval_env_for_callback, 'close'):
                    try: eval_env_for_callback.close()
                    except Exception as e_close_eval_final_v3: logger.error(f"Error closing eval_env for {task_key}: {e_close_eval_final_v3}")
                
                if wandb_run_for_this_task_v3: 
                    logger.info(f"SAI_TRAIN_V3: Finishing wandb run for {task_key} ({wandb_run_for_this_task_v3.name})...")
                    try:
                        wandb_run_for_this_task_v3.finish()
                    except Exception as e_wandb_finish_task_v3:
                        logger.error(f"Error finishing wandb run {wandb_run_for_this_task_v3.name}: {e_wandb_finish_task_v3}")
                
                # حذف تسک از لیست فعال فقط اگر واقعاً این تسک است که تمام می شود
                # این کار باید با دقت انجام شود تا تسک های دیگر اشتباهاً حذف نشوند
                # در schedule_training_for_all_configured، ما تسک را در _active_training_tasks قرار می دهیم
                # و خود این تابع (train_model_for_symbol_timeframe) باید آن را پس از اتمام حذف کند.
                current_task_final = asyncio.current_task() # گرفتن تسک فعلی دوباره
                if current_task_final and self._active_training_tasks.get(task_key) is current_task_final:
                    logger.debug(f"SAI_TRAIN_V3: Removing COMPLETED task {task_key} (id: {id(current_task_final)}) from _active_training_tasks.")
                    self._active_training_tasks.pop(task_key, None)
                elif current_task_final:
                     logger.warning(f"SAI_TRAIN_V3: Task {task_key} (id: {id(current_task_final)}) was not the one in _active_training_tasks or not found. Active: {self._active_training_tasks.get(task_key)}")
                else:
                     logger.warning(f"SAI_TRAIN_V3: Could not get current_task in finally block for {task_key}.")

                logger.info(f"SAI_TRAIN_V3: Training task for {task_key} officially finished/terminated.")

        # Add fundamental data to training
        try:
            fundamental_data = await self.fundamental_analyzer.get_fundamental_data(symbol)
            if fundamental_data and fundamental_data['sentiment']['confidence'] > 0.5:
                # Adjust training parameters based on fundamental outlook
                if abs(fundamental_data['sentiment']['sentiment_score']) > 0.7:
                    logger.info(f"Adjusting training for strong fundamental sentiment in {symbol}")
                    # Modify training parameters here
        except Exception as e:
            logger.warning(f"Could not incorporate fundamental data in training for {symbol}: {e}")
            
        # Continue with existing training code ...

    async def schedule_training_for_all_configured(self):
        logger.info("SAI_SCHED_V3: Scheduling training for all configured symbols and timeframes...")
        
        dn_settings = getattr(self.data_nexus, 'settings', None)
        if not dn_settings or not getattr(dn_settings, 'symbols', []) or not getattr(dn_settings, 'timeframes', []):
            logger.error("SAI_SCHED_V3: DataNexus settings for symbols/timeframes not available. Cannot schedule training.")
            return

        created_tasks_count = 0
        for symbol_str_sched in dn_settings.symbols:
            for timeframe_enum_obj_sched in dn_settings.timeframes:
                task_key_to_sched = f"{symbol_str_sched}_{(timeframe_enum_obj_sched.value if hasattr(timeframe_enum_obj_sched, 'value') else str(timeframe_enum_obj_sched))}"
                
                # استفاده از قفل برای اطمینان از اینکه فقط یک تسک برای هر کلید ایجاد می شود
                if task_key_to_sched not in self._training_locks:
                    self._training_locks[task_key_to_sched] = asyncio.Lock()

                # اگر تسک در حال حاضر فعال است (در حال اجرا یا در صف قفل)، دوباره ایجاد نکن
                if self._training_locks[task_key_to_sched].locked() or \
                   (task_key_to_sched in self._active_training_tasks and not self._active_training_tasks[task_key_to_sched].done()):
                    logger.info(f"SAI_SCHED_V3: Training for {task_key_to_sched} is already active or locked. Skipping new task creation.")
                    continue

                logger.info(f"SAI_SCHED_V3: Creating asyncio.Task for training {task_key_to_sched}")
                train_task_obj = asyncio.create_task(
                    self.train_model_for_symbol_timeframe(symbol_str_sched, timeframe_enum_obj_sched, use_subproc_env=False, num_cpu=1),
                    name=f"TrainingTask_{task_key_to_sched}" 
                )
                # self._active_training_tasks[task_key_to_sched] = train_task_obj # این کار داخل train_model_for_symbol_timeframe انجام می شود
                created_tasks_count +=1
        
        if created_tasks_count > 0:
            logger.info(f"SAI_SCHED_V3: {created_tasks_count} new training tasks created and scheduled.")
        else:
            logger.info("SAI_SCHED_V3: No new training tasks were scheduled (all might be active/locked or no valid configs).")


    def load_model(self, symbol: str, timeframe: TimeFrame) -> Optional[Any]:
        # ... (بدون تغییر) ...
        tf_key = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        model_info = self.models.get(symbol, {}).get(tf_key)
        if model_info and "instance" in model_info and model_info["instance"] is not None:
            logger.debug(f"Returning loaded model instance for {symbol} {tf_key}")
            return model_info["instance"]

        model_path = self._get_model_path(symbol, timeframe)
        final_model_path = model_path.parent / "final_model.zip"

        path_to_load = None
        if model_path.exists():
            path_to_load = model_path
            logger.info(f"Found 'best_model.zip' for {symbol} {tf_key} at {path_to_load}.")
        elif final_model_path.exists():
            path_to_load = final_model_path
            logger.info(f"Found 'final_model.zip' for {symbol} {tf_key} at {path_to_load} (best_model was not found).")
        else:
            logger.warning(f"No trained model (best or final) at {model_path.parent} for {symbol} {tf_key}.")
            return None
        
        algo_name_setting = getattr(getattr(self.settings, 'algorithm', object()), 'name', "PPO")
        AlgorithmClass = SB3_ALGORITHMS.get(algo_name_setting)
        if not AlgorithmClass:
            logger.error(f"Unsupported algorithm {algo_name_setting} to load model.")
            return None
        
        try:
            logger.info(f"Loading model for {symbol} {tf_key} from {path_to_load}...")
            loaded_model_obj = AlgorithmClass.load(path_to_load, device="cpu")
            logger.info(f"Model for {symbol} {tf_key} loaded successfully from {path_to_load}.")
            if symbol not in self.models: self.models[symbol] = {}
            self.models[symbol][tf_key] = {
                "model_path": str(path_to_load), "algorithm": algo_name_setting, "instance": loaded_model_obj
            }
            return loaded_model_obj
        except Exception as e:
            logger.error(f"Error loading model for {symbol} {tf_key} from {path_to_load}: {e}", exc_info=True)
            return None

    def get_prediction(self, symbol: str, timeframe: TimeFrame, market_data_df: Optional[pd.DataFrame] = None) -> Optional[Tuple[int, Optional[np.ndarray]]]:
        # ... (بدون تغییر) ...
        model_inst = self.load_model(symbol, timeframe)
        if not model_inst:
            tf_val_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.warning(f"No model for {symbol} {tf_val_str} to make prediction.")
            return None

        env_win_size_setting = getattr(self.settings, 'env_window_size', 60)
        num_candles_needed_for_temp_env = env_win_size_setting + 5 

        if market_data_df is None:
            if self.data_nexus is None: logger.error("DataNexus not available for get_prediction"); return None
            market_data_df = self.data_nexus.get_market_data(symbol, timeframe, num_candles=num_candles_needed_for_temp_env)
        
        if market_data_df is None or market_data_df.empty or len(market_data_df) < env_win_size_setting :
            tf_val_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.error(f"Not enough market data for {symbol} {tf_val_str} for prediction. Need at least: {env_win_size_setting}, Got: {len(market_data_df) if market_data_df is not None else 0}")
            return None
        
        df_for_actual_temp_env = market_data_df.tail(num_candles_needed_for_temp_env).copy()
        if 'time' in df_for_actual_temp_env.columns and df_for_actual_temp_env.index.name != 'time':
             df_for_actual_temp_env = df_for_actual_temp_env.set_index('time')
        if df_for_actual_temp_env.index.tz is not None:
            df_for_actual_temp_env.index = df_for_actual_temp_env.index.tz_localize(None)

        try:
            temp_env_for_obs = TradingEnv(
                df=df_for_actual_temp_env, 
                initial_balance=getattr(self.settings, 'env_initial_balance', 10000.0),
                window_size=env_win_size_setting,
                commission_pct=0.0, 
                max_episode_steps=len(df_for_actual_temp_env) - env_win_size_setting if (len(df_for_actual_temp_env) - env_win_size_setting > 0) else 1,
                reward_sharpe_ratio=False, 
                normalize_features=getattr(self.settings, 'env_normalize_features', True),
                symbol=symbol,
                timeframe=timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            )
            temp_env_for_obs.current_step_in_df = len(df_for_actual_temp_env) - env_win_size_setting
            temp_env_for_obs.balance = getattr(self.settings, 'env_initial_balance', 10000.0) 
            temp_env_for_obs.current_position = 0 
            observation = temp_env_for_obs._get_observation()
            temp_env_for_obs.close()
        except ValueError as ve:
            logger.error(f"ValueError creating temp TradingEnv for prediction: {ve}. DF len: {len(df_for_actual_temp_env)}, WinSize: {env_win_size_setting}", exc_info=True)
            return None
        except Exception as e_obs:
            tf_val_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.error(f"Error creating observation for prediction for {symbol} {tf_val_str}: {e_obs}", exc_info=True)
            return None
            
        try:
            action, states = model_inst.predict(observation, deterministic=True)
            tf_val_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.info(f"Prediction for {symbol} {tf_val_str}: Action={action}")
            return int(action), states
        except Exception as e_pred:
            tf_val_str = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
            logger.error(f"Error during model prediction for {symbol} {tf_val_str}: {e_pred}", exc_info=True)
            return None

    async def online_learning_step(self, symbol: str, timeframe: TimeFrame, experience: Dict):
        """Enhanced online learning implementation"""
        tf_value = timeframe.value if hasattr(timeframe, 'value') else str(timeframe)
        task_key = f"{symbol}_{tf_value}"
        
        if not hasattr(self, '_online_optimizers'):
            self._online_optimizers = {}
        
        if task_key not in self._online_optimizers:
            self._online_optimizers[task_key] = OnlineLearningOptimizer()
        
        optimizer = self._online_optimizers[task_key]
        optimizer.add_experience(experience)
        
        if optimizer.should_update():
            logger.info(f"Starting online learning update for {task_key}")
            try:
                model = self.load_model(symbol, timeframe)
                if model is None:
                    logger.warning(f"No model found for {task_key} to perform online learning")
                    return
                    
                batch = optimizer.get_training_batch()
                if not batch:
                    return
                    
                # Create temporary environment for learning
                temp_env = self._create_env(
                    pd.DataFrame([exp['market_data'] for exp in batch]),
                    symbol, timeframe
                )
                
                # Perform quick learning step
                model.learn(
                    total_timesteps=len(batch),
                    batch_size=min(len(batch), 32),
                    learning_rate=optimizer.learning_rate
                )
                
                # Save updated model
                model_path = self._get_model_path(symbol, timeframe)
                model.save(model_path)
                
                # Update performance metrics
                returns = [exp.get('return', 0) for exp in batch]
                avg_return = np.mean(returns) if returns else 0
                optimizer.add_performance_metric(avg_return)
                
                # Adjust learning parameters based on performance
                trend = optimizer.get_performance_trend()
                if trend < 0:  # If performance is declining
                    optimizer.learning_rate *= 0.9  # Reduce learning rate
                elif trend > 0:  # If performance is improving
                    optimizer.learning_rate = min(optimizer.learning_rate * 1.1, 0.01)
                    
                optimizer.reset_update_counter()
                logger.info(f"Completed online learning update for {task_key}")
                
            except Exception as e:
                logger.error(f"Error during online learning for {task_key}: {e}", exc_info=True)
                
        return

    async def initialize(self):
        """Initialize components including fundamental analyzer"""
        await self.fundamental_analyzer.initialize()

    async def shutdown(self):
        """Shutdown all components"""
        await self.fundamental_analyzer.close()
        logger.info("Shutting down SynapseAI Core...")
        active_tasks_to_cancel = []
        for task_key, task_obj in list(self._active_training_tasks.items()): # کپی برای تغییر در حین پیمایش
            if task_obj and not task_obj.done():
                logger.debug(f"SAI_SHUTDOWN_V2: Marking task {task_key} (id: {id(task_obj)}) for cancellation.")
                active_tasks_to_cancel.append(task_obj)
        
        if active_tasks_to_cancel:
            logger.info(f"SAI_SHUTDOWN_V2: Cancelling {len(active_tasks_to_cancel)} active AI training task(s)...")
            for task_to_cancel_obj in active_tasks_to_cancel:
                task_name_str = task_to_cancel_obj.get_name() if hasattr(task_to_cancel_obj, 'get_name') else "UnknownAITask"
                logger.debug(f"SAI_SHUTDOWN_V2: Attempting to cancel task: {task_name_str}")
                task_to_cancel_obj.cancel()
            
            # منتظر بمان تا همه تسک های کنسل شده واقعا تمام شوند
            # این gather را خارج از حلقه for قرار بده
            results = await asyncio.gather(*active_tasks_to_cancel, return_exceptions=True)
            for i, res in enumerate(results):
                task_name_gathered_shutdown = active_tasks_to_cancel[i].get_name() if hasattr(active_tasks_to_cancel[i], 'get_name') else f"Task-{i}"
                if isinstance(res, asyncio.CancelledError):
                    logger.info(f"SAI_SHUTDOWN_V2: AI Training task {task_name_gathered_shutdown} was cancelled successfully.")
                elif isinstance(res, Exception):
                    logger.error(f"SAI_SHUTDOWN_V2: Task {task_name_gathered_shutdown} raised an exception during cancellation or completion: {res}", exc_info=res if isinstance(res, BaseException) else None)

        self._active_training_tasks.clear() # پاک کردن دیکشنری پس از کنسل کردن همه
        self._training_locks.clear() # پاک کردن قفل ها
        logger.info("SynapseAI Core shutdown complete.")

    async def meta_learning_update(self, symbol: str, timeframe: TimeFrame):
        """بروزرسانی متا-یادگیری برای بهبود عملکرد مدل"""
        buffer_key = f"{symbol}_{timeframe.value}"
        if buffer_key not in self.experience_buffers:
            return
        
        buffer = self.experience_buffers[buffer_key]
        if len(buffer) < self.meta_learning_config['meta_batch_size']:
            return
            
        experiences = buffer.sample(self.meta_learning_config['meta_batch_size'])
        # پیاده‌سازی الگوریتم متا-یادگیری
        # این بخش در آپدیت‌های بعدی تکمیل می‌شود

    async def adaptive_hyperparameter_optimization(self, symbol: str, timeframe: TimeFrame):
        """بهینه‌سازی خودکار هایپرپارامترها بر اساس عملکرد"""
        # این بخش در آپدیت‌های بعدی تکمیل می‌شود

    async def get_prediction_with_fundamentals(self, symbol: str, timeframe: TimeFrame, 
                                            market_data_df: Optional[pd.DataFrame] = None) -> Optional[Tuple[int, Dict[str, Any]]]:
        """Get prediction incorporating both technical and fundamental analysis"""
        # Get technical analysis prediction
        tech_prediction = self.get_prediction(symbol, timeframe, market_data_df)
        if tech_prediction is None:
            return None
            
        action, states = tech_prediction
        
        try:
            # Check if we need to update fundamental data
            current_time = pd.Timestamp.now()
            last_update = self.fundamental_data_cache.get(symbol, {}).get('timestamp', None)
            
            if (last_update is None or 
                (current_time - pd.Timestamp(last_update)).total_seconds() > self.fundamental_update_interval):
                # Get fresh fundamental data
                fundamental_data = await self.fundamental_analyzer.get_fundamental_data(symbol)
                self.fundamental_data_cache[symbol] = fundamental_data
            else:
                fundamental_data = self.fundamental_data_cache[symbol]
            
            # Get fundamental trading signal
            fund_signal = self.fundamental_analyzer.get_trading_signal(fundamental_data)
            
            # Combine technical and fundamental signals
            final_action = self._combine_signals(action, fund_signal)
            
            # Add fundamental data to the return information
            info = {
                'technical_action': action,
                'fundamental_signal': fund_signal,
                'fundamental_data': {
                    'sentiment': fundamental_data['sentiment'],
                    'recent_news': [news['title'] for news in fundamental_data['news'][:3]],
                    'upcoming_events': [
                        f"{event['event']} ({event['impact']})"
                        for event in fundamental_data['economic_events'][:2]
                    ]
                }
            }
            
            return final_action, info
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis for {symbol}: {e}")
            return action, {'technical_action': action, 'error': str(e)}

    def _combine_signals(self, technical_action: int, fundamental_signal: Dict[str, Any]) -> int:
        """Combine technical and fundamental signals"""
        # Convert fundamental direction to numeric
        fund_direction = 0
        if fundamental_signal['direction'] == 'buy':
            fund_direction = 1
        elif fundamental_signal['direction'] == 'sell':
            fund_direction = -1
            
        # If both signals agree, strengthen the signal
        if (technical_action > 0 and fund_direction > 0) or (technical_action < 0 and fund_direction < 0):
            return technical_action  # Keep strong signal
            
        # If fundamental signal is strong but disagrees with technical
        if fundamental_signal['strength'] > 0.7 and fund_direction != 0:
            if technical_action != 0:  # If technical signal exists
                return 0  # Neutralize the signal due to conflict
            return fund_direction  # Use fundamental signal
            
        # If fundamental signal is neutral or weak, use technical signal
        return technical_action

    async def analyze_market_conditions(self, symbol: str, timeframe: TimeFrame) -> Dict[str, Any]:
        """Analyze current market conditions using advanced analytics"""
        try:
            # Get market data
            market_data = self.data_nexus.get_market_data(symbol, timeframe)
            if market_data is None or market_data.empty:
                return None
                
            # Analyze market regime
            regime = self.market_regime_analyzer.analyze_market_regime(market_data)
            
            # Store regime information
            self.current_market_regime[f"{symbol}_{timeframe}"] = regime
            
            return {
                'regime': regime.regime_type,
                'volatility': regime.volatility_level,
                'trend_strength': regime.trend_strength,
                'risk_score': regime.risk_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return None

    async def get_enhanced_prediction(self, symbol: str, timeframe: TimeFrame) -> Dict[str, Any]:
        """Get prediction using ensemble of models"""
        try:
            # Get market data and prepare features
            market_data = self.data_nexus.get_market_data(symbol, timeframe)
            if market_data is None or market_data.empty:
                return None
                
            # Prepare features
            features = self.ensemble_predictor.prepare_features(market_data)
            
            # Get market regime
            regime_key = f"{symbol}_{timeframe}"
            market_regime = self.current_market_regime.get(regime_key, None)
            
            # Get ensemble prediction
            prediction = self.ensemble_predictor.predict(features)
            
            # Adjust prediction based on market regime
            if market_regime and market_regime.regime_type == "crisis":
                # Be more conservative during crisis
                if prediction.confidence < 0.8:
                    prediction.action = 0  # Hold
                    
            # Store prediction history
            if regime_key not in self.prediction_history:
                self.prediction_history[regime_key] = []
            self.prediction_history[regime_key].append(prediction)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history[regime_key]) > 1000:
                self.prediction_history[regime_key].pop(0)
                
            return {
                'action': prediction.action,
                'confidence': prediction.confidence,
                'model_predictions': prediction.model_predictions,
                'feature_importance': prediction.feature_importance,
                'uncertainty': prediction.uncertainty,
                'market_regime': market_regime.regime_type if market_regime else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced prediction: {e}")
            return None

    async def optimize_portfolio_allocation(self, symbols: List[str], timeframe: TimeFrame) -> Dict[str, Any]:
        """Optimize portfolio allocation using advanced techniques"""
        try:
            # Get market data for all symbols
            returns_data = {}
            regimes = {}
            
            for symbol in symbols:
                market_data = self.data_nexus.get_market_data(symbol, timeframe)
                if market_data is not None and not market_data.empty:
                    returns_data[symbol] = np.log(market_data['close'] / market_data['close'].shift(1))
                    regime_key = f"{symbol}_{timeframe}"
                    regimes[symbol] = self.current_market_regime.get(regime_key)
                    
            if not returns_data:
                return None
                
            returns_df = pd.DataFrame(returns_data)
            
            # Define optimization constraints
            constraints = {
                'max_weight': 0.4,
                'max_correlation': 0.7,
                'max_combined_weight': 0.6
            }
            
            # Determine overall market regime
            overall_regime = "normal"
            crisis_count = sum(1 for r in regimes.values() if r and r.regime_type == "crisis")
            if crisis_count > len(regimes) * 0.3:
                overall_regime = "crisis"
            elif sum(1 for r in regimes.values() if r and r.regime_type == "trending") > len(regimes) * 0.5:
                overall_regime = "trending"
                
            # Get optimal allocation
            allocation = self.portfolio_optimizer.optimize_portfolio(
                returns_df, constraints, overall_regime
            )
            
            # Store optimization history
            key = f"portfolio_{timeframe}"
            if key not in self.optimization_history:
                self.optimization_history[key] = []
            self.optimization_history[key].append(allocation)
            
            # Keep only last 100 optimizations
            if len(self.optimization_history[key]) > 100:
                self.optimization_history[key].pop(0)
                
            # Analyze allocation history
            history_analysis = self.portfolio_optimizer.analyze_allocation_history()
            
            return {
                'weights': allocation.weights,
                'expected_return': allocation.expected_return,
                'volatility': allocation.volatility,
                'sharpe_ratio': allocation.sharpe_ratio,
                'max_drawdown': allocation.max_drawdown,
                'market_regime': overall_regime,
                'allocation_stability': history_analysis.get('allocation_stability'),
                'performance_trends': {
                    'return': history_analysis.get('return_trend'),
                    'volatility': history_analysis.get('volatility_trend'),
                    'sharpe': history_analysis.get('sharpe_ratio_trend')
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio allocation: {e}")
            return None

    async def get_rebalancing_recommendations(self, 
                                           current_positions: Dict[str, float],
                                           timeframe: TimeFrame) -> Dict[str, Any]:
        """Get portfolio rebalancing recommendations"""
        try:
            symbols = list(current_positions.keys())
            total_value = sum(current_positions.values())
            
            # Get optimal allocation
            optimal_allocation = await self.optimize_portfolio_allocation(symbols, timeframe)
            if not optimal_allocation:
                return None
                
            # Get required trades
            trades = self.portfolio_optimizer.get_rebalancing_trades(
                current_positions,
                PortfolioAllocation(**optimal_allocation),
                total_value
            )
            
            return {
                'trades': trades,
                'target_allocation': optimal_allocation['weights'],
                'expected_metrics': {
                    'return': optimal_allocation['expected_return'],
                    'risk': optimal_allocation['volatility'],
                    'sharpe': optimal_allocation['sharpe_ratio']
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting rebalancing recommendations: {e}")
            return None
