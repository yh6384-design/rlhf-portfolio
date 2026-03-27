#!/usr/bin/env python
"""
scripts/verify_env.py
---------------------
Quick sanity-check: imports all src modules and runs unit tests on metrics + personas.
Run after setting up the environment to confirm everything works.

Usage: python scripts/verify_env.py
"""

import sys
import numpy as np

print("=" * 55)
print("RLHF-Portfolio environment verification")
print("=" * 55)

errors = []

# ── 1. Python version ──────────────────────────────────────────
v = sys.version_info
print(f"\n[1] Python {v.major}.{v.minor}.{v.micro}")
if v.major < 3 or (v.major == 3 and v.minor < 9):
    errors.append("Python >= 3.9 required")

# ── 2. Core library imports ────────────────────────────────────
libs = [
    ("numpy",           "np"),
    ("pandas",          "pd"),
    ("torch",           "torch"),
    ("gymnasium",       "gym"),
    ("stable_baselines3","sb3"),
    ("yfinance",        "yf"),
    ("matplotlib",      "plt"),
]
print("\n[2] Library imports:")
for pkg, alias in libs:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"    ✓  {pkg:<22} {ver}")
    except ImportError:
        print(f"    ✗  {pkg:<22} NOT FOUND")
        errors.append(f"Missing: {pkg}")

# Optional: FinRL
try:
    import finrl
    print(f"    ✓  finrl                {getattr(finrl, '__version__', '?')}")
except ImportError:
    print("    ⚠  finrl                not installed (needed for env)")

# ── 3. src module imports ──────────────────────────────────────
print("\n[3] src module imports:")
src_modules = ["src.metrics", "src.personas", "src.reward_model", "src.envs"]
for mod_name in src_modules:
    try:
        __import__(mod_name)
        print(f"    ✓  {mod_name}")
    except Exception as e:
        print(f"    ✗  {mod_name} — {e}")
        errors.append(f"Import error: {mod_name}")

# ── 4. Metrics smoke test ──────────────────────────────────────
print("\n[4] Metrics smoke test:")
try:
    from src.metrics import trajectory_summary, sharpe_ratio, max_drawdown
    rng = np.random.default_rng(42)
    fake_returns = rng.normal(0.0005, 0.01, 252)
    fake_weights = rng.dirichlet(np.ones(30), size=252)
    summary = trajectory_summary(fake_returns, fake_weights)
    assert set(summary.keys()) == {"annualized_return","sharpe","max_drawdown","volatility","calmar","turnover"}
    print(f"    ✓  trajectory_summary: {summary}")
except Exception as e:
    print(f"    ✗  {e}")
    errors.append(str(e))

# ── 5. Personas smoke test ─────────────────────────────────────
print("\n[5] Personas smoke test:")
try:
    from src.personas import label_all_personas
    safe_traj  = {"annualized_return": 0.08, "sharpe": 0.9, "max_drawdown": 0.07,
                  "volatility": 0.10, "calmar": 1.1, "turnover": 0.05}
    risky_traj = {"annualized_return": 0.20, "sharpe": 0.7, "max_drawdown": 0.25,
                  "volatility": 0.28, "calmar": 0.8, "turnover": 0.15}
    labels = label_all_personas(safe_traj, risky_traj)
    print(f"    ✓  labels = {labels}")
    assert labels["conservative"] == 1, "Conservative should prefer safe trajectory"
    assert labels["aggressive"]   == 0, "Aggressive should prefer risky trajectory"
    print("    ✓  Persona logic assertions passed")
except Exception as e:
    print(f"    ✗  {e}")
    errors.append(str(e))

# ── 6. Reward model smoke test ─────────────────────────────────
print("\n[6] Reward model smoke test:")
try:
    import torch
    from src.reward_model import RewardModel, bradley_terry_loss
    model = RewardModel()
    x = torch.randn(4, 6)
    scores = model(x)
    assert scores.shape == (4, 1)
    loss = bradley_terry_loss(scores[:2], scores[2:], torch.tensor([1.0, 0.0]))
    print(f"    ✓  forward pass shape: {scores.shape}")
    print(f"    ✓  Bradley-Terry loss: {loss.item():.4f}")
except Exception as e:
    print(f"    ✗  {e}")
    errors.append(str(e))

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 55)
if errors:
    print(f"FAILED — {len(errors)} issue(s):")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — environment is ready.")
print("=" * 55)
