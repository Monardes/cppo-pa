# CPPO-PA

**Constrained Proximal Policy Optimisation with Physiological Attention** — Safe reinforcement learning for closed-loop anaesthesia control.

https://github.com/Monardes/cppo-pa

## Files

| File | Description |
|------|-------------|
| `cppo_pa.py` | CPPO-PA, PPO, SAC, PID, TCI implementations + evaluation utilities |
| `train.py` | Training CLI (`--method cppo_pa/ppo/sac/pid/tci/all`) |
| `patient_sim.py` | 3-compartment PK-PD simulator (Schnider + Minto models) |
| `generate_figures.py` | Figure generation (Nature Communications style) |

## Quick Start

```bash
pip install torch numpy scipy
python cppo_pa.py          # Smoke test (300-step episode)
python train.py --method cppo_pa --episodes 2000   # Full training
python generate_figures.py # Generate figures
```

## Architecture

- **Physiological Attention**: 4-head self-attention over 30s × 7-channel biosignal buffer → 256-dim state embedding
- **Dual-Critic**: Shared-trunk reward critic V_r + safety critic V_c
- **Constrained PPO**: Per-constraint Lagrangian multipliers (3 × λᵢ) with clipped surrogate objective
- **Safety Projection**: Hard constraint (propofol ≤ 0.75) + soft scaling for BIS/MAP violations

## Baselines

PPO, SAC (off-policy + auto temperature tuning), PPO-Lagrangian, PID (Ziegler–Nichols), TCI (Marsh model)

## License

MIT
