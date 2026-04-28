"""
src/personas.py
---------------
Deterministic persona preference functions for trajectory pair labeling.
Must match nb04 cell 15 v5 label functions exactly.

Each function takes two trajectory summary dicts (from src/metrics.trajectory_summary)
and returns 1 if traj_A is preferred, 0 if traj_B is preferred.

Summary dict keys: annualized_return, sharpe, max_drawdown, volatility, calmar, turnover
"""
from typing import Dict
TrajSummary = Dict[str, float]

PERSONAS = ["conservative", "balanced", "aggressive"]


# ─── Individual persona functions (match nb04 v5 cell 15) ────────────────────

def conservative_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    """
    Conservative (Retiree) persona.
    Dead-trajectory guard, then lower DD → lower vol → higher Sharpe.
    """
    # Guard against degenerate "do-nothing" trajectories
    a_dead = (traj_a['turnover'] < 0.01) and (abs(traj_a['annualized_return']) < 0.01)
    b_dead = (traj_b['turnover'] < 0.01) and (abs(traj_b['annualized_return']) < 0.01)
    if a_dead and not b_dead:
        return 0
    if b_dead and not a_dead:
        return 1

    # Primary: prefer lower drawdown
    if traj_a['max_drawdown'] < traj_b['max_drawdown'] - 0.02:
        return 1
    if traj_b['max_drawdown'] < traj_a['max_drawdown'] - 0.02:
        return 0

    # Secondary: prefer lower volatility
    if traj_a['volatility'] < traj_b['volatility'] - 1e-8:
        return 1
    if traj_b['volatility'] < traj_a['volatility'] - 1e-8:
        return 0

    # Tertiary: prefer higher Sharpe
    return int(traj_a['sharpe'] >= traj_b['sharpe'])


def balanced_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    """
    Balanced (Moderate) persona.
    Higher Sharpe (0.10 threshold) → higher return → higher Calmar.
    """
    if traj_a['sharpe'] > traj_b['sharpe'] + 0.10:
        return 1
    if traj_b['sharpe'] > traj_a['sharpe'] + 0.10:
        return 0

    if traj_a['annualized_return'] > traj_b['annualized_return'] + 1e-8:
        return 1
    if traj_b['annualized_return'] > traj_a['annualized_return'] + 1e-8:
        return 0

    return int(traj_a['calmar'] >= traj_b['calmar'])


def aggressive_preference(traj_a: TrajSummary, traj_b: TrajSummary) -> int:
    """
    Aggressive (Growth) persona.
    DD cap 0.30, then higher return (0.01 threshold), then higher Calmar.
    """
    a_ok = traj_a['max_drawdown'] <= 0.30
    b_ok = traj_b['max_drawdown'] <= 0.30
    if a_ok and not b_ok:
        return 1
    if b_ok and not a_ok:
        return 0

    if traj_a['annualized_return'] > traj_b['annualized_return'] + 0.01:
        return 1
    if traj_b['annualized_return'] > traj_a['annualized_return'] + 0.01:
        return 0

    return int(traj_a['calmar'] >= traj_b['calmar'])


# ─── Dispatch table ───────────────────────────────────────────────────────────

PREFERENCE_FN = {
    "conservative": conservative_preference,
    "balanced":     balanced_preference,
    "aggressive":   aggressive_preference,
}


def label_pair(traj_a: TrajSummary, traj_b: TrajSummary, persona: str) -> int:
    if persona not in PREFERENCE_FN:
        raise ValueError(f"Unknown persona '{persona}'. Choose from {PERSONAS}.")
    return PREFERENCE_FN[persona](traj_a, traj_b)


def label_all_personas(traj_a: TrajSummary, traj_b: TrajSummary) -> Dict[str, int]:
    return {p: label_pair(traj_a, traj_b, p) for p in PERSONAS}
