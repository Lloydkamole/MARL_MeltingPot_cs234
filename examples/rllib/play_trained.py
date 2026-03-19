"""Play a MeltingPot substrate using trained RLlib agent checkpoints and record video.

Usage examples:
  # Load a specific iteration from a run directory:
  python -m examples.rllib.play_trained --run-dir run_checkpoint/commons_harvest__open_phi70_... --iteration 30

  # Load the best checkpoint:
  python -m examples.rllib.play_trained --run-dir run_checkpoint/commons_harvest__open_phi70_... --iteration best

  # Load from an explicit checkpoint path (legacy):
  python -m examples.rllib.play_trained --checkpoint run_checkpoint/.../checkpoint_00030
"""

import argparse
import csv
import glob
import json
import os
import random
import sys

import cv2
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

from meltingpot import substrate

from . import utils
from .custom_model import MeltingPotModel, MeltingPotCNNModel

# Action index for fireZap (same across commons_harvest and clean_up)
FIRE_ZAP_ACTION = 7

# Agent display colours (BGR for OpenCV)
AGENT_COLORS = [
    (50, 200, 50),    # green
    (50, 50, 200),    # red
    (200, 200, 50),   # cyan
    (50, 200, 200),   # yellow
    (200, 50, 200),   # magenta
    (200, 150, 50),   # teal
    (100, 100, 255),  # salmon
]


def draw_hud(frame, step, agent_ids, total_rewards, zap_counts):
    """Draw a translucent HUD bar with per-agent zaps and total reward."""
    h, w = frame.shape[:2]
    num = len(agent_ids)
    bar_h = 18 + num * 16 + 4
    # Translucent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    total_r = sum(total_rewards.values())
    total_z = sum(zap_counts.values())
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Step {step}  |  Total R: {total_r:.0f}  |  Zaps: {total_z}',
                (6, 14), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    for j, aid in enumerate(agent_ids):
        color = AGENT_COLORS[j % len(AGENT_COLORS)]
        y = 30 + j * 16
        txt = f'{aid}: R={total_rewards[aid]:.0f}  Z={zap_counts[aid]}'
        cv2.putText(frame, txt, (10, y), font, 0.34, color, 1, cv2.LINE_AA)
    return frame


def resolve_checkpoint(args):
    """Resolve the checkpoint path from --run-dir/--iteration or --checkpoint."""
    if args.run_dir is not None:
        run_dir = os.path.abspath(args.run_dir)
        if not os.path.isdir(run_dir):
            sys.exit(f"Error: --run-dir '{run_dir}' does not exist.")

        if args.iteration == 'best':
            ckpt = os.path.join(run_dir, 'checkpoint_best')
            if not os.path.isdir(ckpt):
                sys.exit(f"Error: No checkpoint_best/ found in {run_dir}. "
                         f"Was training run with best-checkpoint tracking?")
            return ckpt
        elif args.iteration == 'latest' or args.iteration is None:
            # Find the highest-numbered checkpoint
            pattern = os.path.join(run_dir, 'checkpoint_[0-9]*')
            candidates = sorted(glob.glob(pattern))
            if candidates:
                return candidates[-1]
            # Fall back to root checkpoint
            if os.path.exists(os.path.join(run_dir, 'rllib_checkpoint.json')):
                return run_dir
            sys.exit(f"Error: No checkpoints found in {run_dir}")
        else:
            # Numeric iteration
            try:
                it = int(args.iteration)
            except ValueError:
                sys.exit(f"Error: --iteration must be a number, 'best', or 'latest'. "
                         f"Got '{args.iteration}'.")
            ckpt = os.path.join(run_dir, f'checkpoint_{it:05d}')
            if not os.path.isdir(ckpt):
                # List available checkpoints
                pattern = os.path.join(run_dir, 'checkpoint_[0-9]*')
                available = [os.path.basename(p) for p in sorted(glob.glob(pattern))]
                sys.exit(f"Error: checkpoint_{it:05d}/ not found in {run_dir}.\n"
                         f"Available: {available or 'none'}")
            return ckpt
    elif args.checkpoint is not None:
        return os.path.abspath(args.checkpoint)
    else:
        sys.exit("Error: Provide either --run-dir or --checkpoint.")


def main():
    parser = argparse.ArgumentParser(
        description='Run trained agents on a MeltingPot substrate and record video')

    # Checkpoint selection (two modes)
    ckpt_group = parser.add_argument_group('checkpoint selection')
    ckpt_group.add_argument('--run-dir', type=str, default=None,
                        help='Path to the training run directory '
                             '(e.g. run_checkpoint/commons_harvest__open_phi70_...)')
    ckpt_group.add_argument('--iteration', type=str, default='latest',
                        help='Which checkpoint to load: a number (e.g. 30), '
                             '"best", or "latest" (default: latest)')
    ckpt_group.add_argument('--checkpoint', type=str, default=None,
                        help='Explicit path to checkpoint directory (alternative to --run-dir)')

    parser.add_argument('--substrate', type=str, default='commons_harvest__open',
                        help='MeltingPot substrate name')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of environment steps to run (episode = 5000)')
    parser.add_argument('--output-dir', type=str, default='trained_gameplay_videos',
                        help='Directory to save the output video')
    parser.add_argument('--fps', type=int, default=12,
                        help='Output video FPS')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Output video resolution (square)')
    parser.add_argument('--output-name', type=str, default=None,
                        help='Custom output filename (without .mp4). '
                             'Defaults to {substrate}_{checkpoint}.mp4')
    parser.add_argument('--model', type=str, default=None,
                        choices=['lstm', 'cnn'],
                        help='Model type used during training. Auto-detected '
                             'from run_config.json if --run-dir is provided. '
                             'Required if using --checkpoint without a config.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility. Fixes numpy/random '
                             'and enables deterministic action selection. '
                             'Without this, each run produces a different episode.')
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint(args)

    # ---- Auto-detect model type from run_config.json if available ----
    model_type = args.model
    if model_type is None:
        # Try to find run_config.json in the run directory
        search_dirs = []
        if args.run_dir:
            search_dirs.append(os.path.abspath(args.run_dir))
        # Walk up from checkpoint path
        ckpt_abs = os.path.abspath(checkpoint_path)
        search_dirs.append(os.path.dirname(ckpt_abs))
        search_dirs.append(os.path.dirname(os.path.dirname(ckpt_abs)))
        for d in search_dirs:
            cfg_path = os.path.join(d, 'run_config.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    model_type = json.load(f).get('model_type', 'lstm')
                print(f'Auto-detected model type: {model_type} (from {cfg_path})')
                break
        if model_type is None:
            model_type = 'lstm'  # default fallback

    # ---- Initialise Ray and register env / model ----
    ray.init(ignore_reinit_error=True)
    tune.register_env('meltingpot', utils.env_creator)
    ModelCatalog.register_custom_model('meltingpot_model', MeltingPotModel)
    ModelCatalog.register_custom_model('meltingpot_cnn_model', MeltingPotCNNModel)

    # ---- Restore trained algorithm from checkpoint ----
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f'Loading checkpoint from {checkpoint_path} ...')
    algo = PPO.from_checkpoint(checkpoint_path)
    print('Checkpoint restored successfully.')

    # ---- Build the environment (same wrappers as training) ----
    substrate_name = args.substrate
    player_roles = substrate.get_config(substrate_name).default_player_roles
    num_agents = len(player_roles)

    env_config = {'substrate': substrate_name, 'roles': player_roles}
    env = utils.env_creator(env_config)  # MeltingPotObsWrapper(MeltingPotEnv(...))

    agent_ids = [f'agent_{i}' for i in range(num_agents)]
    player_ids = [f'player_{i}' for i in range(num_agents)]
    player_to_agent = {p: a for p, a in zip(player_ids, agent_ids)}

    # ---- Initialise per-agent LSTM states / prev action / prev reward ----
    states = {}
    prev_actions = {}
    prev_rewards = {}
    for agent_id in agent_ids:
        pol = algo.get_policy(agent_id)
        states[agent_id] = pol.get_initial_state()
        prev_actions[agent_id] = 0
        prev_rewards[agent_id] = 0.0

    # ---- Video setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_name:
        fname = args.output_name if args.output_name.endswith('.mp4') else args.output_name + '.mp4'
    else:
        ckpt_label = os.path.basename(checkpoint_path)
        fname = f'{substrate_name}_{ckpt_label}.mp4'
    output_video = os.path.join(args.output_dir, fname)
    output_size = (args.resolution, args.resolution)

    # ---- Seed for reproducibility ----
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f'Random seed set to {args.seed} (deterministic episode)')
    else:
        print('No seed set — episode will differ on each run (use --seed N to fix)')

    # ---- Run simulation ----
    obs, info = env.reset()

    writer = None
    frames_written = 0
    total_rewards = {aid: 0.0 for aid in agent_ids}
    zap_counts = {aid: 0 for aid in agent_ids}
    # Track READY_TO_SHOOT from last step (last element of flat obs)
    ready_to_shoot = {pid: 0.0 for pid in player_ids}

    # ---- CSV log setup ----
    csv_base = os.path.splitext(output_video)[0] + '_rewards.csv'
    csv_fields = (['step']
                  + [f'reward_{aid}' for aid in agent_ids]
                  + ['reward_total']
                  + [f'cumul_{aid}' for aid in agent_ids]
                  + ['cumul_total']
                  + [f'zaps_{aid}' for aid in agent_ids]
                  + ['zaps_total'])
    csv_file = open(csv_base, 'w', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    print(f'Running {num_agents} trained agents for {args.steps} steps ...')

    try:
        for step in range(args.steps):
            # Compute actions for every agent
            actions = {}
            for player_id in player_ids:
                agent_id = player_to_agent[player_id]
                action, state, _ = algo.compute_single_action(
                    obs[player_id],
                    state=states[agent_id],
                    policy_id=agent_id,
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    full_fetch=True,
                    explore=(args.seed is None),  # deterministic when seed is set
                )
                actions[player_id] = action
                states[agent_id] = state
                prev_actions[agent_id] = action

                # Count zap: agent chose fireZap AND was ready to shoot
                if action == FIRE_ZAP_ACTION and ready_to_shoot[player_id] >= 0.5:
                    zap_counts[agent_id] += 1

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Update prev rewards, accumulators, and READY_TO_SHOOT
            step_rewards = {}
            for player_id in player_ids:
                agent_id = player_to_agent[player_id]
                r = rewards.get(player_id, 0.0)
                prev_rewards[agent_id] = r
                total_rewards[agent_id] += r
                step_rewards[agent_id] = r
                # READY_TO_SHOOT is always the last element of the flat obs
                flat = obs[player_id]
                ready_to_shoot[player_id] = float(flat[-1]) if len(flat) > 0 else 0.0

            # Write CSV row
            row = {'step': step + 1}
            total_z = sum(zap_counts.values())
            total_r = sum(total_rewards.values())
            step_r_total = sum(step_rewards.values())
            for aid in agent_ids:
                row[f'reward_{aid}'] = f'{step_rewards[aid]:.4f}'
                row[f'cumul_{aid}'] = f'{total_rewards[aid]:.4f}'
                row[f'zaps_{aid}'] = zap_counts[aid]
            row['reward_total'] = f'{step_r_total:.4f}'
            row['cumul_total'] = f'{total_r:.4f}'
            row['zaps_total'] = total_z
            csv_writer.writerow(row)

            # Capture WORLD.RGB frame via render()
            frame = env.render()
            if frame is not None:
                frame = frame.astype(np.uint8)
                frame = cv2.resize(frame, output_size,
                                   interpolation=cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Draw HUD overlay
                frame = draw_hud(frame, step + 1, agent_ids,
                                 total_rewards, zap_counts)

                if writer is None:
                    writer = cv2.VideoWriter(
                        output_video,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        args.fps,
                        output_size,
                    )
                writer.write(frame)
                frames_written += 1

            # Check episode end
            if terminated.get('__all__', False) or truncated.get('__all__', False):
                print(f'Episode ended at step {step + 1}')
                break

            # Progress update every 100 steps
            if (step + 1) % 100 == 0:
                print(f'  Step {step + 1}/{args.steps}')

        # ---- Finish ----
        csv_file.close()
        if writer is not None:
            writer.release()
            writer = None

        print(f'\nDone! Wrote {frames_written} frames to {output_video}')
        print(f'Rewards CSV saved to {csv_base}')
        print('Cumulative rewards / zaps per agent:')
        for agent_id in agent_ids:
            print(f'  {agent_id}: reward={total_rewards[agent_id]:.1f}  '
                  f'zaps={zap_counts[agent_id]}')
        print(f'  TOTAL:    reward={sum(total_rewards.values()):.1f}  '
              f'zaps={sum(zap_counts.values())}')

    finally:
        if writer is not None:
            writer.release()
        env.close()
        algo.stop()
        ray.shutdown()


if __name__ == '__main__':
    main()
