"""
src/envs.py
-----------
FinRL environment wrapper with RLHF reward augmentation hook.

Usage
-----
# Stage 1 — base training (no RLHF)
env = make_env(df_features, mode="train")

# Stage 3 — RLHF fine-tuning
from src.reward_model import load_reward_model
rm = load_reward_model("results/checkpoints/reward_model_conservative.pt")
env = make_env(df_features, mode="train", reward_model=rm, rlhf_lambda=0.5)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from collections import deque
from typing import Optional, Any

# Trajectory window length for reward model (20 trading days)
TRAJECTORY_WINDOW = 20

# DOW 30 tickers (current constituents)
DOW30_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA",  "CAT",  "CRM",  "CSCO", "CVX",
    "DIS",  "DOW",  "GS",  "HD",  "HON",  "IBM",  "INTC", "JNJ",
    "JPM",  "KO",   "MCD", "MMM", "MRK",  "MSFT", "NKE",  "PG",
    "TRV",  "UNH",  "V",   "VZ",  "AMZN",  "WMT",
]

TRANSACTION_COST = 0.001   # 0.1%
INITIAL_CAPITAL  = 1_000_000.0


# ─── Reward model interface (duck-typed) ──────────────────────────────────────

class RLHFRewardWrapper(gym.Wrapper):
    """
    Wraps any Gymnasium environment and augments the step reward with
    a persona-specific learned reward signal.

    r_total = r_base + lambda * r_theta(trajectory_window)

    The wrapper maintains a rolling window of daily portfolio returns.
    When the window is full, the reward model is called once per step.

    Parameters
    ----------
    env          : base Gymnasium env (FinRL StockTradingEnv)
    reward_model : trained RewardModel (src.reward_model.RewardModel)
    rlhf_lambda  : mixing coefficient (default 0.5)
    """

    def __init__(
        self,
        env: gym.Env,
        reward_model: Any,
        rlhf_lambda: float = 0.5,
    ):
        super().__init__(env)
        self.reward_model = reward_model
        self.rlhf_lambda  = rlhf_lambda
        self._ret_window: deque = deque(maxlen=TRAJECTORY_WINDOW)
        self._wgt_window: deque = deque(maxlen=TRAJECTORY_WINDOW)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ret_window.clear()
        self._wgt_window.clear()
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Extract portfolio value, prices, shares from observation
        # obs layout: [portfolio_value(1), prices(30), shares(30), features(300)]
        n = len(DOW30_TICKERS)
        portfolio_value = float(obs[0])
        prices          = np.array(obs[1:n+1], dtype=float)
        shares          = np.array(obs[n+1:2*n+1], dtype=float)

        # Compute daily return from reward (base_reward = scaled return)
        daily_return = float(base_reward) / 1e-4  # undo reward_scaling

        # Compute portfolio weights from shares and prices
        stock_values = shares * prices
        total_value  = stock_values.sum()
        if total_value > 0:
            weights = stock_values / total_value
        else:
            weights = np.ones(n) / n

        self._ret_window.append(daily_return)
        self._wgt_window.append(weights)

        rlhf_reward = 0.0
        if len(self._ret_window) == TRAJECTORY_WINDOW:
            rlhf_reward = self._compute_rlhf_reward()

        total_reward = base_reward + self.rlhf_lambda * rlhf_reward
        info["base_reward"] = base_reward
        info["rlhf_reward"] = rlhf_reward
        return obs, total_reward, terminated, truncated, info

    def _compute_rlhf_reward(self) -> float:
        """Compute r_theta from the current trajectory window."""
        from src.metrics import trajectory_summary
        daily_returns = np.array(self._ret_window)
        weight_hist   = np.array(self._wgt_window)
        summary = trajectory_summary(daily_returns, weight_hist)
        return self.reward_model.score(summary)


# ─── Environment factory ───────────────────────────────────────────────────────

def make_env(
    df: pd.DataFrame,
    mode: str = "train",
    reward_model: Optional[Any] = None,
    rlhf_lambda: float = 0.5,
    seed: int = 42,
) -> gym.Env:
    """
    Build a (optionally RLHF-wrapped) FinRL StockTradingEnv.

    Parameters
    ----------
    df           : processed feature DataFrame (from notebooks/01_data.ipynb)
    mode         : 'train', 'val', or 'test'
    reward_model : trained RewardModel — if provided, wraps with RLHF reward
    rlhf_lambda  : RLHF mixing coefficient
    seed         : random seed

    Returns
    -------
    gym.Env
    """
    try:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    except ImportError as e:
        raise ImportError(
            "FinRL not installed. Run: pip install git+https://github.com/AI4Finance-Foundation/FinRL"
        ) from e

    env_kwargs = dict(
        df=df,
        stock_dim=len(DOW30_TICKERS),
        hmax=1.0,                           # max weight per asset (enforced via softmax)
        initial_amount=INITIAL_CAPITAL,
        num_stock_shares=[0]*len(DOW30_TICKERS),
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
        reward_scaling=1e-4,
        state_space=1 + len(DOW30_TICKERS) + len(DOW30_TICKERS) + len(DOW30_TICKERS) * 10,  # 361-dim state
        action_space=len(DOW30_TICKERS),
        tech_indicator_list=[
            "close", "volume",
            "close_1d_ret", "close_5d_ret", "close_20d_ret",
            "vol_20d", "vol_60d", "macd", "rsi_14", "volume_ratio",
        ],
        mode=mode,
    )

    base_env = StockTradingEnv(**env_kwargs)
    base_env.reset(seed=seed)

    if reward_model is not None:
        return RLHFRewardWrapper(base_env, reward_model, rlhf_lambda)

    return base_env
