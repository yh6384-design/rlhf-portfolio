"""
src/envs.py
-----------
FinRL environment wrapper with RLHF reward augmentation hook.

Price / normalization design
-----------------------------
FinRL's StockTradingEnv uses self.data.close for TWO things:
  1. Trading price: cash -= price × shares × (1 + cost)  → must be real dollars
  2. State observation: state[1:31] = self.data.close     → agent sees real prices

Our dataframe (from 01_data.ipynb Cell 8) provides:
  - 'close'      = actual dollar prices ($30–$500)   NOT normalized
  - 'close_norm' = z-score normalized close           normalized, for learning
  - all others   = z-score normalized                 normalized, for learning

State layout (361 dimensions):
  state[0]       : cash (real dollars, starts at $1,000,000)
  state[1:31]    : real stock prices ($30–$500)   trading + agent observation
  state[31:61]   : shares held (integer counts)   portfolio tracking
  state[61:361]  : 10 normalized tech features × 30 stocks   learning only
                   (close_norm, volume, close_1d_ret, close_5d_ret,
                    close_20d_ret, vol_20d, vol_60d, macd, rsi_14, volume_ratio)

Why no SafeStockTradingEnv:
  With real prices ($30–$500), FinRL's native guardrail works correctly:
    available_amount = cash // (price × 1.001)
  Cash can never go negative. Portfolio value = cash + Σ(real_price × shares)
  is always ≥ 0 since real prices are always positive.

hmax = 100 shares per stock per step:
  Max trade per stock = 100 × $500 (most expensive) = $50,000 = 5% of $1M.
  Reasonable for daily rebalancing. More conservative than 200 to prevent
  the agent learning to make large single-step concentrated bets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Any

TRAJECTORY_WINDOW = 20

DOW30_TICKERS = [
    "AAPL", "AMGN", "AXP", "BA",  "CAT",  "CRM",  "CSCO", "CVX",
    "DIS",  "DOW",  "GS",  "HD",  "HON",  "IBM",  "INTC", "JNJ",
    "JPM",  "KO",   "MCD", "MMM", "MRK",  "MSFT", "NKE",  "PG",
    "TRV",  "UNH",  "V",   "VZ",  "AMZN", "WMT",
]

TRANSACTION_COST = 0.001        # 0.1% per trade
INITIAL_CAPITAL  = 1_000_000.0  # $1M starting portfolio
HMAX             = 100          # max shares per stock per step

# 10 normalized tech features per stock
# 'close' is NOT here — agent already sees real prices at state[1:31]
# 'close_norm' provides a properly scaled price-level signal for learning
TECH_INDICATOR_LIST = [
    "close_norm",    # z-score normalized close price (learning signal)
    "volume",        # z-score normalized volume
    "close_1d_ret",  # 1-day log return
    "close_5d_ret",  # 5-day log return
    "close_20d_ret", # 20-day log return
    "vol_20d",       # 20-day rolling volatility
    "vol_60d",       # 60-day rolling volatility
    "macd",          # normalized MACD signal
    "rsi_14",        # RSI (14-day)
    "volume_ratio",  # volume / 20-day avg volume
]

# state_space = 1 (cash) + 30 (prices) + 30 (shares) + 30×10 (tech) = 361
STATE_SPACE = 1 + len(DOW30_TICKERS) + len(DOW30_TICKERS) + len(DOW30_TICKERS) * len(TECH_INDICATOR_LIST)


# ─── RLHF reward wrapper ──────────────────────────────────────────────────────

class RLHFRewardWrapper:
    """
    Wraps FinRL's StockTradingEnv and augments the reward with a
    persona-specific learned signal.

        r_total = r_base + rlhf_lambda * r_theta(trajectory_window)

    Daily returns are computed from asset_memory — FinRL's own real-dollar
    portfolio value tracker. This gives correct percentage returns for
    trajectory_summary (Sharpe, drawdown, Calmar etc.).

    Portfolio weights are from real prices × shares — correct economic weights.
    """

    def __init__(self, env: Any, reward_model: Any, rlhf_lambda: float = 0.5):
        self.env          = env
        self.reward_model = reward_model
        self.rlhf_lambda  = rlhf_lambda
        self._ret_window: deque = deque(maxlen=TRAJECTORY_WINDOW)
        self._wgt_window: deque = deque(maxlen=TRAJECTORY_WINDOW)
        self._prev_value: float = float(INITIAL_CAPITAL)
        # Expose gymnasium-required attributes for SB3 compatibility
        self.action_space      = env.action_space
        self.observation_space = env.observation_space

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._ret_window.clear()
        self._wgt_window.clear()
        self._prev_value = float(INITIAL_CAPITAL)
        return result

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        n = len(DOW30_TICKERS)

        # ── Real dollar portfolio value from FinRL's own tracker ─────────────
        # asset_memory[-1] = cash + Σ(real_price × shares) after this step
        current_value = (
            float(self.env.asset_memory[-1])
            if self.env.asset_memory
            else self._prev_value
        )

        # ── Percentage daily return in real dollar terms ──────────────────────
        daily_return = (
            current_value / self._prev_value - 1.0
            if self._prev_value > 0 else 0.0
        )
        self._prev_value = current_value

        # ── Portfolio weights from real prices × shares ───────────────────────
        state      = self.env.state
        prices     = np.array(state[1:n+1],     dtype=float)  # real dollar prices
        shares     = np.array(state[n+1:2*n+1], dtype=float)
        stock_vals = prices * shares
        total      = stock_vals.sum()
        weights    = stock_vals / total if total > 0 else np.ones(n) / n

        self._ret_window.append(daily_return)
        self._wgt_window.append(weights)

        # ── RLHF reward (only when trajectory window is full) ─────────────────
        rlhf_reward = 0.0
        if len(self._ret_window) == TRAJECTORY_WINDOW:
            rlhf_reward = self._compute_rlhf_reward()

        total_reward = base_reward + self.rlhf_lambda * rlhf_reward

        if isinstance(info, dict):
            info["base_reward"]     = base_reward
            info["rlhf_reward"]     = rlhf_reward
            info["portfolio_value"] = current_value  # real dollars

        return obs, total_reward, terminated, truncated, info

    def _compute_rlhf_reward(self) -> float:
        from src.metrics import trajectory_summary
        summary = trajectory_summary(
            np.array(self._ret_window),
            np.array(self._wgt_window),
        )
        return self.reward_model.score(summary)

    def close(self):
        return self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


# ─── Environment factory ───────────────────────────────────────────────────────

def make_env(
    df: pd.DataFrame,
    mode: str = "train",
    reward_model: Optional[Any] = None,
    rlhf_lambda: float = 0.5,
    seed: int = 42,
) -> Any:
    """
    Build a FinRL StockTradingEnv, optionally RLHF-wrapped.

    Requirements for df (long format from 01_data.ipynb)
    -----------------------------------------------------
    'close'       : actual dollar prices — NOT normalized
    'close_norm'  : z-score normalized close price
    'volume'      : z-score normalized
    'close_1d_ret': z-score normalized
    'close_5d_ret': z-score normalized
    'close_20d_ret': z-score normalized
    'vol_20d'     : z-score normalized
    'vol_60d'     : z-score normalized
    'macd'        : z-score normalized
    'rsi_14'      : z-score normalized
    'volume_ratio': z-score normalized
    'date', 'tic' : required by FinRL

    Parameters
    ----------
    df           : long-format feature DataFrame
    mode         : 'train', 'val', or 'test'
    reward_model : trained RewardModel — if provided, wraps with RLHFRewardWrapper
    rlhf_lambda  : RLHF mixing coefficient (default 0.5)
    seed         : random seed
    """
    try:
        from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    except ImportError as e:
        raise ImportError(
            "FinRL not installed. "
            "Run: pip install git+https://github.com/AI4Finance-Foundation/FinRL"
        ) from e

    env_kwargs = dict(
        df                  = df,
        stock_dim           = len(DOW30_TICKERS),
        hmax                = HMAX,
        initial_amount      = INITIAL_CAPITAL,
        num_stock_shares    = [0] * len(DOW30_TICKERS),
        buy_cost_pct        = [TRANSACTION_COST] * len(DOW30_TICKERS),
        sell_cost_pct       = [TRANSACTION_COST] * len(DOW30_TICKERS),
        reward_scaling      = 1e-4,
        state_space         = STATE_SPACE,           # 361
        action_space        = len(DOW30_TICKERS),
        tech_indicator_list = TECH_INDICATOR_LIST,   # 10 normalized features
        mode                = mode,
    )

    base_env = StockTradingEnv(**env_kwargs)
    base_env.reset(seed=seed)

    if reward_model is not None:
        return RLHFRewardWrapper(base_env, reward_model, rlhf_lambda)

    return base_env
