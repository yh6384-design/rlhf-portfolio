# RLHF Portfolio Management
**DS-GA 3001 · Reinforcement Learning · Spring 2026 · Team Wall St RL**

Investor-aligned portfolio management via Reinforcement Learning with Human Feedback (RLHF).  
An RL agent learns investor-specific risk preferences from pairwise trajectory comparisons — no manual reward engineering needed.

---

## Pipeline Overview

| Stage | Notebook | Output |
|---|---|---|
| 1 · Data & features | `notebooks/01_data.ipynb` | `data/features_{train,val,test}.parquet` |
| 2 · Env validation | `notebooks/02_env.ipynb` | Validated `StockTradingEnv`, unit tests |
| 3 · Base PPO training | `notebooks/03_base_training.ipynb` | `results/checkpoints/base_agent_seed{1,2,3}.zip` |
| 4 · Preference data + reward models | `notebooks/04_rlhf_data.ipynb` | `data/preferences.parquet`, `results/checkpoints/reward_model_{conservative,balanced,aggressive}.pt` |
| 5 · RLHF fine-tuning | `notebooks/05_finetuning.ipynb` | `results/checkpoints/rlhf_agent_{conservative,balanced,aggressive}.zip` |
| 6 · Evaluation | `notebooks/06_evaluation.ipynb` | `results/figures/`, `results/metrics_summary.csv` |

## Key source modules

| File | Role |
|---|---|
| `src/envs.py` | FinRL wrapper with RLHF reward hook |
| `src/reward_model.py` | Bradley-Terry MLP reward model |
| `src/personas.py` | Persona preference functions (Conservative / Balanced / Aggressive) |
| `src/metrics.py` | Portfolio performance metric helpers |

## Quickstart (Colab)

```python
# Mount Drive & clone
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install -r requirements.txt

# Run setup check
!python scripts/verify_env.py
```

## Assets

- **Dow 30** constituents, daily OHLCV, 2010–2024 via `yfinance`
- Train: 2015–2022 · Val: Jan–Jun 2023 · Test: Jul 2023–Dec 2024
- 0.1% transaction cost, daily rebalancing, $1M initial capital

## Team

| Member | Primary Role |
|---|---|
| Teammate A | Repo lead · `src/` architecture · PPO base training |
| Teammate B | Data pipeline · Env validation · Preference labeling |
| Teammate C | Reward model training · RLHF data generation |
| Teammate D | RLHF fine-tuning · Evaluation · Visualization · Report |

## Due dates

| Milestone | Date |
|---|---|
| Env + data complete | Mar 30 |
| Base PPO trained (3 seeds) | Apr 6 |
| Reward models trained | Apr 13 |
| RLHF fine-tuning complete | Apr 20 |
| Full evaluation + report | **Apr 27** |
| Presentation | May 1 or May 4 |
