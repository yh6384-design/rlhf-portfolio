"""
src/reward_model.py
-------------------
MLP reward model for RLHF persona preference learning.

Architecture:
  Input  → 6 trajectory summary features
  Hidden → 32 units, ReLU
  Hidden → 16 units, ReLU
  Output → 1 scalar preference score (unbounded)

Loss: Bradley-Terry binary cross-entropy over trajectory pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


# ─── Feature order (must match metrics.trajectory_summary key order) ──────────

FEATURE_KEYS = ["annualized_return", "sharpe", "max_drawdown", "volatility", "calmar", "turnover"]
N_FEATURES = len(FEATURE_KEYS)  # 6


# ─── Model definition ──────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    Small MLP that scores a trajectory summary vector.

    Input  : (batch, 6) — trajectory summary features
    Output : (batch, 1) — scalar preference score
    """

    def __init__(self, input_dim: int = N_FEATURES, hidden1: int = 32, hidden2: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, 1)

    def score(self, summary_dict: dict) -> float:
        """Score a single trajectory summary dict. Inference utility."""
        features = torch.tensor(
            [[summary_dict[k] for k in FEATURE_KEYS]], dtype=torch.float32
        )
        self.eval()
        with torch.no_grad():
            return self.forward(features).item()


# ─── Bradley-Terry loss ────────────────────────────────────────────────────────

def bradley_terry_loss(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy under the Bradley-Terry model.

    P(A ≻ B) = σ(r(A) − r(B))

    Parameters
    ----------
    score_a, score_b : (batch, 1) tensors — scalar scores for each trajectory
    labels           : (batch,) — 1 if A preferred, 0 if B preferred
    """
    logit = (score_a - score_b).squeeze(1)          # (batch,)
    return nn.functional.binary_cross_entropy_with_logits(logit, labels.float())


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_reward_model(
    df: pd.DataFrame,
    persona: str,
    n_epochs: int = 50,
    lr: float = 3e-4,
    batch_size: int = 64,
    val_split: float = 0.2,
    seed: int = 42,
    save_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[RewardModel, dict]:
    """
    Train a RewardModel on a preference pair DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
          traj_a_<feature>, traj_b_<feature> for each feature in FEATURE_KEYS
          label_<persona>  (1 if A preferred, 0 if B preferred)
    persona : str
        One of 'conservative', 'balanced', 'aggressive'
    save_path : str, optional
        Path to save the trained model weights (.pt)

    Returns
    -------
    model : trained RewardModel
    history : dict with keys 'train_loss', 'val_loss', 'val_accuracy'
    """
    torch.manual_seed(seed)

    # ── Build tensors ──
    feat_a = torch.tensor(
        df[[f"traj_a_{k}" for k in FEATURE_KEYS]].values, dtype=torch.float32
    )
    feat_b = torch.tensor(
        df[[f"traj_b_{k}" for k in FEATURE_KEYS]].values, dtype=torch.float32
    )
    labels = torch.tensor(df[f"label_{persona}"].values, dtype=torch.float32)

    # ── Train / val split ──
    n = len(labels)
    n_val = int(n * val_split)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(feat_a[train_idx], feat_b[train_idx], labels[train_idx])
    val_ds   = TensorDataset(feat_a[val_idx],   feat_b[val_idx],   labels[val_idx])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Model & optimizer ──
    model = RewardModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_losses = []
        for a, b, lbl in train_dl:
            a, b, lbl = a.to(device), b.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = bradley_terry_loss(model(a), model(b), lbl)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for a, b, lbl in val_dl:
                a, b, lbl = a.to(device), b.to(device), lbl.to(device)
                sa, sb = model(a), model(b)
                val_losses.append(bradley_terry_loss(sa, sb, lbl).item())
                preds = (sa.squeeze() > sb.squeeze()).float()
                correct += (preds == lbl).sum().item()
                total   += lbl.size(0)

        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_accuracy"].append(correct / total if total > 0 else 0.0)

        if (epoch + 1) % 10 == 0:
            print(
                f"[{persona}] epoch {epoch+1:3d}/{n_epochs} | "
                f"train_loss={history['train_loss'][-1]:.4f} | "
                f"val_loss={history['val_loss'][-1]:.4f} | "
                f"val_acc={history['val_accuracy'][-1]:.3f}"
            )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Saved {persona} reward model → {save_path}")

    final_acc = history["val_accuracy"][-1]
    print(f"\n[{persona}] Final val accuracy: {final_acc:.3f} (target ≥ 0.75)")
    if final_acc < 0.75:
        print("  ⚠ Below target — consider more training data or epochs.")

    return model, history


def load_reward_model(path: str, device: str = "cpu") -> RewardModel:
    """Load a saved RewardModel from disk."""
    model = RewardModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
