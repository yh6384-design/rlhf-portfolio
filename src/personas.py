"""
src/personas.py
---------------
Deterministic persona preference functions for trajectory pair labeling.

Each function takes two trajectory summary dicts (from src/metrics.trajectory_summary)
and returns 1 if traj_A is preferred, 0 if traj_B is preferred.

Summary dict keys: annualized_return, sharpe, max_drawdown, volatility, calmar, turnover
"""

from typing import Dict, Tuple

TrajSummary = Dict[str, float]

PERSONAS = ["conservative", "balanced", "aggressive"]


# ─── Individual persona functions ──────────────────────────────────────────────

def conservative_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    mdd_a = traj_a["max_drawdown"]
    mdd_b = traj_b["max_drawdown"]

    if mdd_a < mdd_b - 0.02:
        return 1
    if mdd_b < mdd_a - 0.02:
        return 0

    if traj_a["volatility"] < traj_b["volatility"] - 1e-8:
        return 1
    if traj_b["volatility"] < traj_a["volatility"] - 1e-8:
        return 0

    return int(traj_a["sharpe"] >= traj_b["sharpe"])


def balanced_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    """
    Balanced (Moderate) persona.

    Primary : maximize Sharpe ratio (risk-adjusted return).
    Secondary: among similar Sharpe (<= 0.1 difference), prefer higher return.
    """
    sharpe_a = traj_a["sharpe"]
    sharpe_b = traj_b["sharpe"]

    if abs(sharpe_a - sharpe_b) <= 0.1:
        return 1 if traj_a["annualized_return"] >= traj_b["annualized_return"] else 0

    return 1 if sharpe_a >= sharpe_b else 0


def aggressive_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    """
    Aggressive (Growth) persona.

    Primary : maximize annualized return, subject to max drawdown <= 30% cap.
    Secondary: among similar returns (<= 1% absolute difference), prefer higher Calmar.
    """
    ret_a = traj_a["annualized_return"]
    ret_b = traj_b["annualized_return"]
    mdd_a = traj_a["max_drawdown"]
    mdd_b = traj_b["max_drawdown"]

    DRAWDOWN_CAP = 0.30

    # Disqualify trajectories exceeding drawdown cap
    a_ok = mdd_a <= DRAWDOWN_CAP
    b_ok = mdd_b <= DRAWDOWN_CAP

    if a_ok and not b_ok:
        return 1
    if b_ok and not a_ok:
        return 0
    # Both exceed or both within cap — compare on return

    if abs(ret_a - ret_b) <= 0.01:
        return 1 if traj_a["calmar"] >= traj_b["calmar"] else 0

    return 1 if ret_a >= ret_b else 0


# ─── Dispatch table ────────────────────────────────────────────────────────────

PREFERENCE_FN = {
    "conservative": conservative_preference,
    "balanced":     balanced_preference,
    "aggressive":   aggressive_preference,
}


def label_pair(
    traj_a: TrajSummary,
    traj_b: TrajSummary,
    persona: str,
) -> int:
    """
    Return preference label (1 = A preferred, 0 = B preferred) for a given persona.

    Parameters
    ----------
    traj_a, traj_b : TrajSummary
        Trajectory summary dicts from metrics.trajectory_summary()
    persona : str
        One of 'conservative', 'balanced', 'aggressive'
    """
    if persona not in PREFERENCE_FN:
        raise ValueError(f"Unknown persona '{persona}'. Choose from {PERSONAS}.")
    return PREFERENCE_FN[persona](traj_a, traj_b)


def label_all_personas(
    traj_a: TrajSummary,
    traj_b: TrajSummary,
) -> Dict[str, int]:
    """Return labels for all three personas at once."""
    return {p: label_pair(traj_a, traj_b, p) for p in PERSONAS}
