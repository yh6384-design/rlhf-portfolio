"""
src/metrics.py
--------------
Portfolio performance metric computation.

All functions accept daily returns as a 1-D array-like (numpy or pandas Series).
"""

import numpy as np
import pandas as pd
from typing import Union

Array = Union[np.ndarray, pd.Series]

TRADING_DAYS = 252


# ─── Core metrics ──────────────────────────────────────────────────────────────

def annualized_return(daily_returns: Array) -> float:
    """Compound annualized return from daily log or simple returns."""
    daily_returns = np.asarray(daily_returns)
    n = len(daily_returns)
    cumulative = (1 + daily_returns).prod()
    return float(cumulative ** (TRADING_DAYS / n) - 1)


def annualized_volatility(daily_returns: Array) -> float:
    """Annualized standard deviation of daily returns."""
    return float(np.std(daily_returns, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: Array, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio (default: risk-free = 0)."""
    daily_returns = np.asarray(daily_returns)
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    if excess.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std(ddof=1))


def max_drawdown(daily_returns: Array) -> float:
    """Maximum peak-to-trough drawdown (positive value, e.g. 0.15 = 15%)."""
    daily_returns = np.asarray(daily_returns)
    cum_values = (1 + daily_returns).cumprod()
    rolling_peak = np.maximum.accumulate(cum_values)
    drawdowns = (rolling_peak - cum_values) / rolling_peak
    return float(drawdowns.max())


def calmar_ratio(daily_returns: Array) -> float:
    """Calmar ratio = annualized return / |max drawdown|. Returns 0 if drawdown is 0."""
    mdd = max_drawdown(daily_returns)
    if mdd == 0:
        return 0.0
    return float(annualized_return(daily_returns) / mdd)


def average_daily_turnover(weight_history: np.ndarray) -> float:
    """
    Mean L1 norm of daily weight changes.

    Parameters
    ----------
    weight_history : np.ndarray, shape (T, N)
        Portfolio weight vectors for each day.

    Returns
    -------
    float
        Average absolute weight change per day.
    """
    diffs = np.abs(np.diff(weight_history, axis=0))
    return float(diffs.sum(axis=1).mean())


# ─── Trajectory summary (used by reward model) ─────────────────────────────────

def trajectory_summary(daily_returns: Array, weight_history: np.ndarray = None) -> dict:
    """
    Compute the 6-feature trajectory summary vector used by the reward model.

    Returns
    -------
    dict with keys: annualized_return, sharpe, max_drawdown, volatility, calmar, turnover
    """
    daily_returns = np.asarray(daily_returns)
    summary = {
        "annualized_return": annualized_return(daily_returns),
        "sharpe":            sharpe_ratio(daily_returns),
        "max_drawdown":      max_drawdown(daily_returns),
        "volatility":        annualized_volatility(daily_returns),
        "calmar":            calmar_ratio(daily_returns),
        "turnover":          average_daily_turnover(weight_history) if weight_history is not None else 0.0,
    }
    return summary


def full_metrics_table(agents: dict, daily_returns_map: dict, weight_history_map: dict = None) -> pd.DataFrame:
    """
    Build the summary metrics table for all agents.

    Parameters
    ----------
    agents : dict
        {agent_name: ...}  (just used for row labels)
    daily_returns_map : dict
        {agent_name: np.ndarray of daily returns}
    weight_history_map : dict, optional
        {agent_name: np.ndarray of shape (T, N)}

    Returns
    -------
    pd.DataFrame — one row per agent, seven metric columns.
    """
    rows = []
    for name in agents:
        r = daily_returns_map[name]
        w = weight_history_map.get(name) if weight_history_map else None
        rows.append({
            "agent":             name,
            "cumulative_return": float((1 + np.asarray(r)).prod() - 1),
            "annualized_return": annualized_return(r),
            "sharpe_ratio":      sharpe_ratio(r),
            "max_drawdown":      max_drawdown(r),
            "calmar_ratio":      calmar_ratio(r),
            "volatility":        annualized_volatility(r),
            "avg_turnover":      average_daily_turnover(w) if w is not None else float("nan"),
        })
    return pd.DataFrame(rows).set_index("agent")
