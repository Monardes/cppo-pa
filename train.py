"""
Training script for CPPO-PA and baseline methods on anaesthesia control.

Usage:
    python train.py --method cppo_pa --episodes 2000
    python train.py --method ppo --episodes 2000
    python train.py --method all --episodes 2000

Evaluates on held-out patients after training.
"""

import argparse
import numpy as np
import torch
import json
import os
import time
from collections import defaultdict

from patient_sim import AnesthesiaEnv
from cppo_pa import (CPPOPA, PPOAgent, SACAgent, PIDController, TCIController,
                      RolloutBuffer, compute_metrics)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CPPO-PA and baselines for anaesthesia control')
    parser.add_argument('--method', type=str, default='cppo_pa',
                        choices=['cppo_pa', 'ppo', 'sac', 'pid', 'tci', 'all'],
                        help='Method to train/evaluate')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of training episodes '
                             '(2000 episodes × 2h surgery = 14.4M steps/patient)')
    parser.add_argument('--episode-length', type=int, default=7200,
                        help='Episode length in seconds (7200 s = 2 h surgery)')
    parser.add_argument('--n-train-patients', type=int, default=16,
                        help='Number of parallel training patients')
    parser.add_argument('--n-eval-patients', type=int, default=40,
                        help='Number of evaluation patients')
    parser.add_argument('--n-eval-seeds', type=int, default=5,
                        help='Number of random seeds for evaluation')
    parser.add_argument('--buffer-size', type=int, default=2048,
                        help='Rollout buffer size (steps per update)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Minibatch size for PPO update')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='PPO update epochs per rollout')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str,
                        default='d:/Works/SCI/code/sci/results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, or auto)')
    return parser.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(agent, env, n_steps: int = 7200,
                   use_safe: bool = False) -> dict:
    """Evaluate an agent on a full episode.

    Args:
        agent: CPPOPA, PPOAgent, PIDController, or TCIController.
        env: AnesthesiaEnv instance.
        n_steps: Maximum evaluation steps.
        use_safe: Use act_safe() for CPPOPA.

    Returns:
        Dictionary of evaluation metrics.
    """
    obs = env.reset()
    history = [obs[0]]  # patient 0
    actions_taken = []

    for t in range(n_steps):
        hist_array = np.array(history)

        if isinstance(agent, CPPOPA):
            if use_safe:
                action = agent.act_safe(hist_array, t)
            else:
                action, _, _ = agent.act(hist_array, t, deterministic=True)
        elif isinstance(agent, (PPOAgent, SACAgent)):
            action, _ = agent.act(hist_array, t, deterministic=True)
        elif isinstance(agent, (PIDController, TCIController)):
            action = agent.act(history[-1])
        else:
            raise ValueError(f"Unknown agent type: {type(agent)}")

        obs_new, reward, done, info = env.step(np.array([action]))
        history.append(obs_new[0])
        actions_taken.append(action)

    return compute_metrics(np.array(history), np.array(actions_taken))


def train_cppo_pa(args) -> dict:
    """Train CPPO-PA and return evaluation results."""
    print("\n" + "=" * 60)
    print("Training CPPO-PA")
    print("=" * 60)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(args.seed)

    # Create environments
    train_env = AnesthesiaEnv(
        n_patients=args.n_train_patients,
        episode_length=args.episode_length,
        seed=args.seed
    )

    # Create agent
    agent = CPPOPA(device=device, lr=args.lr)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in agent.attention.parameters()) + sum(p.numel() for p in agent.policy.parameters()) + sum(p.numel() for p in agent.critic.parameters()):,}")

    # Training loop
    buffer = RolloutBuffer(
        buffer_size=args.buffer_size,
        state_shape=(30, 7),
        action_dim=2,
        device=device,
    )

    train_metrics = defaultdict(list)
    t_start = time.time()

    for episode in range(args.episodes):
        obs = train_env.reset()
        histories = [[] for _ in range(args.n_train_patients)]
        episode_reward = np.zeros(args.n_train_patients)
        episode_safety = np.zeros((args.n_train_patients, 3))

        for t in range(args.episode_length):
            # Collect actions for all parallel patients
            actions = np.zeros((args.n_train_patients, 2))
            log_probs = np.zeros(args.n_train_patients)
            safety_preds = np.zeros((args.n_train_patients, 3))

            for i in range(args.n_train_patients):
                histories[i].append(obs[i])
                hist_array = np.array(histories[i])
                action, log_prob, safety_pred = agent.act(hist_array, t)
                actions[i] = action
                log_probs[i] = log_prob
                safety_preds[i] = safety_pred

            # Step environment
            obs_new, rewards, dones, info = train_env.step(actions)

            # Safety costs
            safety_costs = np.zeros((args.n_train_patients, 3))
            for i in range(args.n_train_patients):
                safety_costs[i, 0] = float(obs_new[i, 0] < 40)
                safety_costs[i, 1] = float(obs_new[i, 1] < 55)
                safety_costs[i, 2] = float(actions[i, 0] > 0.75)

            # Get values for GAE
            for i in range(args.n_train_patients):
                hist_array = np.array(histories[i])
                x = torch.FloatTensor(
                    agent._build_state_buffer(hist_array, t)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = agent.attention(x)
                    v_r, v_c = agent.critic(h)

                buffer.add(
                    agent._build_state_buffer(hist_array, t),
                    actions[i],
                    log_probs[i],
                    rewards[i],
                    v_r.item(),
                    safety_costs[i],
                    v_c.cpu().numpy().flatten(),
                    dones[i],
                )

            episode_reward += rewards
            episode_safety += safety_costs
            obs = obs_new

        # Compute GAE and update
        mean_reward = episode_reward.mean()
        mean_safety = episode_safety.mean(axis=0)

        # Estimate last values for GAE
        with torch.no_grad():
            last_h = agent.attention(
                torch.FloatTensor(histories[0][-30:]).unsqueeze(0).to(device)
            )
            last_vr, last_vc = agent.critic(last_h)

        buffer.compute_gae(
            agent.gamma, agent.gae_lambda,
            last_vr.item(), last_vc.cpu().numpy().flatten()
        )

        if len(buffer) >= args.batch_size:
            update_metrics = agent.update(
                buffer, batch_size=args.batch_size, n_epochs=args.n_epochs
            )
            buffer = RolloutBuffer(
                buffer_size=args.buffer_size,
                state_shape=(30, 7),
                action_dim=2,
                device=device,
            )

        # Logging
        train_metrics['reward'].append(mean_reward)
        train_metrics['safety_bis'].append(mean_safety[0])
        train_metrics['safety_map'].append(mean_safety[1])
        train_metrics['safety_overdose'].append(mean_safety[2])
        # Per-constraint Lagrangian multipliers (λ₁, λ₂, λ₃)
        lambdas = agent.get_lambda()
        train_metrics['lagrangian_mean'].append(float(np.mean(lambdas)))
        for i in range(len(lambdas)):
            train_metrics[f'lagrangian_{i}'].append(float(lambdas[i]))

        if (episode + 1) % 100 == 0:
            elapsed = time.time() - t_start
            recent_reward = np.mean(train_metrics['reward'][-100:])
            recent_safety = np.mean(train_metrics['safety_bis'][-100:])
            lambdas = agent.get_lambda()
            lambda_str = '/'.join(f'{l:.3f}' for l in lambdas)
            print(f"Ep {episode + 1:4d}/{args.episodes} | "
                  f"Reward: {recent_reward:.3f} | "
                  f"BIS<40: {recent_safety:.4f} | "
                  f"λ=[{lambda_str}] | "
                  f"Time: {elapsed:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    # --- Evaluation ---
    print("\nEvaluating CPPO-PA on held-out patients...")
    eval_metrics = defaultdict(list)

    for seed in range(args.n_eval_seeds):
        eval_env = AnesthesiaEnv(
            n_patients=1,
            episode_length=args.episode_length,
            seed=args.seed + 1000 + seed
        )
        metrics = evaluate_agent(agent, eval_env, args.episode_length)
        for k, v in metrics.items():
            eval_metrics[k].append(v)

    # Summary
    results = {}
    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    for k in sorted(eval_metrics.keys()):
        mean_v = np.mean(eval_metrics[k])
        std_v = np.std(eval_metrics[k])
        results[k] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f"{k:<30} {mean_v:8.2f} {std_v:8.2f}")

    results['train_time'] = train_time
    results['train_metrics'] = {k: [float(x) for x in v[-100:]]
                                 for k, v in train_metrics.items()}

    return results


def evaluate_baseline(args, method: str) -> dict:
    """Evaluate a non-learning baseline (PID, TCI)."""
    print(f"\nEvaluating {method.upper()}...")

    eval_metrics = defaultdict(list)

    for seed in range(args.n_eval_seeds):
        eval_env = AnesthesiaEnv(
            n_patients=1,
            episode_length=args.episode_length,
            seed=args.seed + 1000 + seed
        )

        if method == 'pid':
            agent = PIDController()
        elif method == 'tci':
            agent = TCIController()
        else:
            raise ValueError(f"Unknown method: {method}")

        agent.reset()
        metrics = evaluate_agent(agent, eval_env, args.episode_length)
        for k, v in metrics.items():
            eval_metrics[k].append(v)

    results = {}
    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    for k in sorted(eval_metrics.keys()):
        mean_v = np.mean(eval_metrics[k])
        std_v = np.std(eval_metrics[k])
        results[k] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f"{k:<30} {mean_v:8.2f} {std_v:8.2f}")

    return results


def train_ppo(args) -> dict:
    """Train standard PPO baseline."""
    print("\n" + "=" * 60)
    print("Training PPO Baseline")
    print("=" * 60)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(args.seed)

    train_env = AnesthesiaEnv(
        n_patients=args.n_train_patients,
        episode_length=args.episode_length,
        seed=args.seed
    )

    agent = PPOAgent(device=device, lr=args.lr)
    print(f"Device: {device}")

    buffer = RolloutBuffer(
        buffer_size=args.buffer_size,
        state_shape=(30, 7),
        action_dim=2,
        device=device,
    )

    train_metrics = defaultdict(list)
    t_start = time.time()

    for episode in range(args.episodes):
        obs = train_env.reset()
        histories = [[] for _ in range(args.n_train_patients)]
        episode_reward = np.zeros(args.n_train_patients)

        for t in range(args.episode_length):
            actions = np.zeros((args.n_train_patients, 2))
            log_probs = np.zeros(args.n_train_patients)

            for i in range(args.n_train_patients):
                histories[i].append(obs[i])
                action, log_prob = agent.act(np.array(histories[i]), t)
                actions[i] = action
                log_probs[i] = log_prob

            obs_new, rewards, dones, info = train_env.step(actions)

            safety_costs = np.zeros((args.n_train_patients, 3))
            for i in range(args.n_train_patients):
                safety_costs[i, 0] = float(obs_new[i, 0] < 40)
                safety_costs[i, 1] = float(obs_new[i, 1] < 55)
                safety_costs[i, 2] = float(actions[i, 0] > 0.75)

            for i in range(args.n_train_patients):
                hist_array = np.array(histories[i])
                x = torch.FloatTensor(
                    agent._build_state_buffer(hist_array, t)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    h = agent.attention(x)
                    v = agent.critic(h)

                buffer.add(
                    agent._build_state_buffer(hist_array, t),
                    actions[i], log_probs[i], rewards[i],
                    v.item(), safety_costs[i],
                    np.zeros(3),  # PPO doesn't use safety value
                    dones[i],
                )

            episode_reward += rewards
            obs = obs_new

        with torch.no_grad():
            last_h = agent.attention(
                torch.FloatTensor(histories[0][-30:]).unsqueeze(0).to(device)
            )
            last_v = agent.critic(last_h)

        buffer.compute_gae(
            agent.gamma, agent.gae_lambda,
            last_v.item(), np.zeros(3)
        )

        if len(buffer) >= args.batch_size:
            agent.update(buffer, args.batch_size, args.n_epochs)
            buffer = RolloutBuffer(
                buffer_size=args.buffer_size,
                state_shape=(30, 7),
                action_dim=2,
                device=device,
            )

        train_metrics['reward'].append(episode_reward.mean())

        if (episode + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"Ep {episode + 1:4d}/{args.episodes} | "
                  f"Reward: {np.mean(train_metrics['reward'][-100:]):.3f} | "
                  f"Time: {elapsed:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.0f}s")

    # Evaluation
    print("\nEvaluating PPO on held-out patients...")
    eval_metrics = defaultdict(list)

    for seed in range(args.n_eval_seeds):
        eval_env = AnesthesiaEnv(
            n_patients=1,
            episode_length=args.episode_length,
            seed=args.seed + 1000 + seed
        )
        metrics = evaluate_agent(agent, eval_env, args.episode_length)
        for k, v in metrics.items():
            eval_metrics[k].append(v)

    results = {}
    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    for k in sorted(eval_metrics.keys()):
        mean_v = np.mean(eval_metrics[k])
        std_v = np.std(eval_metrics[k])
        results[k] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f"{k:<30} {mean_v:8.2f} {std_v:8.2f}")

    results['train_time'] = train_time
    return results


def train_sac(args) -> dict:
    """Train SAC baseline."""
    print("\n" + "=" * 60)
    print("Training SAC Baseline")
    print("=" * 60)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(args.seed)

    train_env = AnesthesiaEnv(
        n_patients=1,  # SAC trains on single patient (off-policy)
        episode_length=args.episode_length,
        seed=args.seed
    )

    agent = SACAgent(device=device, replay_size=500000)
    print(f"Device: {device}")

    train_metrics = defaultdict(list)
    t_start = time.time()

    total_steps = 0
    for episode in range(args.episodes):
        obs = train_env.reset()
        history = [obs[0]]
        episode_reward = 0.0

        for t in range(args.episode_length):
            hist_array = np.array(history)
            action, _ = agent.act(hist_array, t)
            obs_new, reward, done, info = train_env.step(np.array([action]))
            history.append(obs_new[0])

            # Store transition
            agent.store_transition(
                agent._build_state_buffer(hist_array, t),
                action, reward[0],
                agent._build_state_buffer(np.array(history), t + 1),
                float(t >= args.episode_length - 1)
            )

            episode_reward += reward[0]
            total_steps += 1

            # Update every 4 steps
            if total_steps % 4 == 0:
                update_metrics = agent.update(batch_size=256)
                for k, v in update_metrics.items():
                    train_metrics[k].append(v)

            obs = obs_new

        train_metrics['reward'].append(episode_reward)

        if (episode + 1) % 50 == 0:
            elapsed = time.time() - t_start
            recent_reward = np.mean(train_metrics['reward'][-50:])
            alpha = train_metrics['alpha'][-1] if train_metrics['alpha'] else 0
            print(f"Ep {episode + 1:4d}/{args.episodes} | "
                  f"Reward: {recent_reward:.3f} | "
                  f"α: {alpha:.3f} | "
                  f"Time: {elapsed:.0f}s")

    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.0f}s ({train_time/60:.1f} min)")

    # Evaluation
    print("\nEvaluating SAC on held-out patients...")
    eval_metrics = defaultdict(list)

    for seed in range(args.n_eval_seeds):
        eval_env = AnesthesiaEnv(
            n_patients=1,
            episode_length=args.episode_length,
            seed=args.seed + 1000 + seed
        )
        metrics = evaluate_agent(agent, eval_env, args.episode_length)
        for k, v in metrics.items():
            eval_metrics[k].append(v)

    results = {}
    print(f"\n{'Metric':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    for k in sorted(eval_metrics.keys()):
        mean_v = np.mean(eval_metrics[k])
        std_v = np.std(eval_metrics[k])
        results[k] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f"{k:<30} {mean_v:8.2f} {std_v:8.2f}")

    results['train_time'] = train_time
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    if args.method in ('cppo_pa', 'all'):
        results = train_cppo_pa(args)
        all_results['CPPO-PA'] = results
        with open(f'{args.output_dir}/results_cppo_pa.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.method in ('ppo', 'all'):
        results = train_ppo(args)
        all_results['PPO'] = results
        with open(f'{args.output_dir}/results_ppo.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.method in ('sac', 'all'):
        results = train_sac(args)
        all_results['SAC'] = results
        with open(f'{args.output_dir}/results_sac.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.method in ('pid', 'all'):
        results = evaluate_baseline(args, 'pid')
        all_results['PID'] = results
        with open(f'{args.output_dir}/results_pid.json', 'w') as f:
            json.dump(results, f, indent=2)

    if args.method in ('tci', 'all'):
        results = evaluate_baseline(args, 'tci')
        all_results['TCI'] = results
        with open(f'{args.output_dir}/results_tci.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary comparison
    if args.method == 'all':
        print("\n" + "=" * 60)
        print("METHOD COMPARISON")
        print("=" * 60)
        metrics_to_show = ['tir_bis', 'violations_per_100h',
                           'cumulative_reward', 'induction_time']
        for metric in metrics_to_show:
            print(f"\n{metric}:")
            for method, results in all_results.items():
                if metric in results:
                    m = results[metric]
                    print(f"  {method:<10}: {m['mean']:8.2f} ± {m['std']:6.2f}")

        with open(f'{args.output_dir}/results_all.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
