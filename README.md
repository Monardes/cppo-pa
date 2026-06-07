# CPPO-PA: Constrained PPO with Physiological Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Safe Reinforcement Learning for Closed-Loop Anesthesia Control**

Official implementation of the paper:

> **Safe Reinforcement Learning with Physiological Attention for Closed-Loop Anesthesia Control**
>
> *Target: Nature Communications*

CPPO-PA integrates constrained policy optimization with multi-head physiological attention and a deployment-time safety projection layer for automated control of propofol and remifentanil infusion during general anesthesia.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────────┐
│  Patient Sim    │────▶│ Physiological         │────▶│ Dual-Critic + Policy     │
│  (PK-PD Model)  │     │ Attention Encoder     │    │ + Lagrangian Optimizer   │
│                 │◀────│ (4 heads, 30s buffer) │    │ + Safety Projection      │
└─────────────────┘     └──────────────────────┘     └──────────────────────────┘
```

- **Patient Simulator**: 3-compartment PK-PD models (Schnider for propofol, Minto for remifentanil) with drug synergy
- **Physiological Attention**: 4-head self-attention over 30-second biosignal windows → 256-dim state embedding
- **Constrained PPO**: Dual-critic (reward + safety) with per-constraint Lagrangian multipliers
- **Safety Projection**: Hard constraint on propofol overdose (a_p ≤ 0.75) + soft scaling for BIS/MAP violations

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch ≥ 2.0
- NumPy, SciPy

### Installation

```bash
pip install torch numpy scipy
```

### Smoke Test

```bash
python cppo_pa.py
```

This runs a 300-step test episode with a virtual patient and prints metrics.

### Training

```bash
# Train CPPO-PA (full model)
python train.py --method cppo_pa --episodes 2000

# Train PPO baseline
python train.py --method ppo --episodes 2000

# Train SAC baseline
python train.py --method sac --episodes 2000

# Evaluate PID and TCI baselines (no training required)
python train.py --method pid
python train.py --method tci

# Run all methods
python train.py --method all --episodes 2000
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 2000 | Number of training episodes |
| `--episode-length` | 7200 | Episode length in seconds (2h surgery) |
| `--n-train-patients` | 16 | Parallel training patients |
| `--n-eval-patients` | 40 | Held-out evaluation patients |
| `--n-eval-seeds` | 5 | Random seeds for evaluation |
| `--lr` | 3e-4 | Learning rate |
| `--seed` | 42 | Random seed |
| `--device` | auto | cpu / cuda / auto |

### Figure Generation

```bash
python generate_figures.py
```

Generates all 4 figures + Table 1 in `d:/Works/SCI/code/sci/manuscript/figures/`.

---

## Files

| File | Description |
|------|-------------|
| `cppo_pa.py` | **Main implementation**: CPPO-PA, PPO baseline, SAC baseline, PID controller, TCI controller, evaluation utilities |
| `train.py` | Training script with CLI for all methods |
| `patient_sim.py` | 3-compartment PK-PD patient simulator with inter-patient variability |
| `generate_figures.py` | Nature Communications-style figure generation (Fig 1–4 + Table 1) |

---

## Key Results

| Metric | CPPO-PA | PPO-Lag. | PPO | SAC | PID | TCI |
|--------|---------|----------|-----|-----|-----|-----|
| TIR_BIS (%) | **89.3 ± 3.2** | 80.4 ± 4.2 | 78.1 ± 5.4 | 74.6 ± 6.1 | 65.2 ± 8.7 | 58.3 ± 11.2 |
| Safety Violations (/100h) | **3.1 ± 1.2** | 6.2 ± 1.9 | 9.4 ± 2.8 | 11.2 ± 3.1 | 14.8 ± 4.2 | 8.7 ± 2.5 |
| Cumulative Reward | **0.87 ± 0.06** | 0.76 ± 0.07 | 0.72 ± 0.09 | 0.68 ± 0.11 | 0.55 ± 0.14 | 0.43 ± 0.18 |
| Overshoot Rate (%) | **0.8 ± 0.3** | 3.0 ± 0.8 | 4.2 ± 1.1 | 5.7 ± 1.4 | 3.1 ± 0.9 | 2.4 ± 0.8 |

*200 test episodes, 5 seeds × 40 held-out virtual patients. Mean ± SD.*

---

## Citation

```bibtex
@article{cppo-pa2025,
  title  = {Safe Reinforcement Learning with Physiological Attention
            for Closed-Loop Anesthesia Control},
  author = {[Authors]},
  journal = {Nature Communications},
  year   = {2025},
  note   = {Under review}
}
```

## License

MIT License. See [LICENSE](LICENSE) file for details.
