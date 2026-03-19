# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs an example of a self-play training experiment."""

import argparse
import csv
import gc
import json
import math
import os
from datetime import datetime

import numpy as np
import psutil
from meltingpot import substrate
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import policy

from . import utils
from .custom_model import MeltingPotModel, MeltingPotCNNModel


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def gini_coefficient(values):
  """Gini coefficient in [0, 1]. 0 = perfect equality."""
  arr = np.asarray(values, dtype=np.float64)
  if arr.size == 0:
    return 0.0
  if np.min(arr) < 0:
    arr = arr - np.min(arr)
  total = np.sum(arr)
  if total <= 0:
    return 0.0
  arr = np.sort(arr)
  n = arr.size
  idx = np.arange(1, n + 1, dtype=np.float64)
  return float(np.clip((2.0 * np.sum(idx * arr) / (n * total)) - ((n + 1.0) / n), 0.0, 1.0))


def linear_phi_deg(iteration, total_iterations, start_deg, end_deg):
  """Linearly interpolate phi in degrees for a given iteration (1-based)."""
  if total_iterations <= 1:
    return start_deg
  t = (iteration - 1) / (total_iterations - 1)
  return start_deg + t * (end_deg - start_deg)


def log_phi_deg(iteration, total_iterations, start_deg, end_deg):
  """Logarithmic phi schedule (fast ramp early, slowing later)."""
  if total_iterations <= 1:
    return start_deg
  t_lin = (iteration - 1) / (total_iterations - 1)
  t = math.log(1.0 + t_lin * (math.e - 1.0))  # 0 → 1, concave
  return start_deg + t * (end_deg - start_deg)


def exp_phi_deg(iteration, total_iterations, start_deg, end_deg):
  """Exponential phi schedule (slow ramp early, accelerating later)."""
  if total_iterations <= 1:
    return start_deg
  t_lin = (iteration - 1) / (total_iterations - 1)
  t = (math.exp(t_lin) - 1.0) / (math.e - 1.0)  # 0 → 1, convex
  return start_deg + t * (end_deg - start_deg)


def sigmoid_phi_deg(iteration, total_iterations, start_deg, end_deg):
  """Sigmoid phi schedule: sigmoid((x - N/2) / scale), S-curve centered at midpoint.

  Uses sigmoid(12*(t - 0.5)) where t = (iter-1)/(N-1) so the S-curve
  spans the full training range rather than collapsing to a step function.
  """
  if total_iterations <= 1:
    return start_deg
  # def _sig(x):
  #   return 1.0 / (1.0 + math.exp(-x))
  # t_lin = (iteration - 1) / (total_iterations - 1)  # 0 → 1
  # raw = _sig(12.0 * (t_lin - 0.5))  # sigmoid centered at midpoint
  # # Normalize so t=0 at iter 1, t=1 at iter N
  # raw_start = _sig(12.0 * (0.0 - 0.5))   # = _sig(-6)
  # raw_end = _sig(12.0 * (1.0 - 0.5))     # = _sig(6)
  # t = (raw - raw_start) / (raw_end - raw_start)
  # return start_deg + t * (end_deg - start_deg)
  def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))
  mid = total_iterations / 2.0
  raw = _sig(iteration - mid)
  # Normalize so t=0 at iter 1, t=1 at iter N
  raw_start = _sig(1 - mid)
  raw_end = _sig(total_iterations - mid)
  t = (raw - raw_start) / (raw_end - raw_start)
  return start_deg + t * (end_deg - start_deg)


PHI_SCHEDULES = {
    "linear": linear_phi_deg,
    "log": log_phi_deg,
    "exp": exp_phi_deg,
    "sigmoid": sigmoid_phi_deg,
}


def get_config(
    substrate_name: str = "bach_or_stravinsky_in_the_matrix__repeated",
    num_rollout_workers: int = 4,
    rollout_fragment_length: int = 100,
    train_batch_size: int = 12800,
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
  cnn_channels=(16, 128),
  cnn_first_kernel: int = 8,
  cnn_first_stride: int = 8,
  cnn_second_kernel=None,
  cnn_second_stride: int = 1,
    sgd_minibatch_size: int = 128,
    num_sgd_iter: int = 10,
    entropy_coeff: float = 0.003,
    lr: float = 3e-4,
    kl_coeff: float = 1e-8,
    kl_target: float = 0.01,
    vf_clip_param: float = 100.0,
    clip_param: float = 0.2,
    grad_clip: float = 0.5,
    model_type: str = "lstm",
):
  """Get the configuration for running an agent on a substrate using RLLib.

  We need the following 2 pieces to run the training:

  Args:
    substrate_name: The name of the MeltingPot substrate, coming from
      `substrate.AVAILABLE_SUBSTRATES`.
    num_rollout_workers: The number of workers for playing games.
    rollout_fragment_length: Unroll time for learning.
    train_batch_size: Batch size (batch * rollout_fragment_length)
    fcnet_hiddens: Fully connected layers.
    post_fcnet_hiddens: Layer sizes after the fully connected torso.
    lstm_cell_size: Size of the LSTM.
    cnn_channels: Two channel sizes for conv layers, e.g. (16, 128).
    cnn_first_kernel: Kernel size for first conv layer (square).
    cnn_first_stride: Stride for first conv layer.
    cnn_second_kernel: Second conv kernel as (h, w), or None for auto-fit.
    cnn_second_stride: Stride for second conv layer.
    sgd_minibatch_size: Size of the mini-batch for learning.
    num_sgd_iter: Number of SGD epochs per training iteration (PPO default 30
      is wasteful for multi-agent; 5–10 is usually sufficient).
    entropy_coeff: Entropy bonus coefficient. 0.0 = no exploration pressure
      (PPO default), agents quickly converge to deterministic policies.
      0.003–0.01 is standard for multi-agent environments.
    lr: Learning rate. PPO default is 5e-5 which is very conservative.
    kl_coeff: Adaptive KL penalty coefficient. RLLib default 0.2 causes KL
      divergence explosion — it grows unboundedly when KL exceeds kl_target,
      eventually freezing the policy. Set 1e-8 to effectively disable
      but still compute KL for diagnostics (0.0 skips computation entirely).
    kl_target: Target KL divergence for adaptive penalty (only matters if
      kl_coeff > 0). RLLib default 0.01.
    vf_clip_param: Value function clip range. RLLib default 10.0 is far too
      small for environments with reward magnitudes 100-500+, causing the
      value function to learn incorrect estimates. 100.0 or float('inf').
    clip_param: PPO surrogate objective clip parameter. Standard PPO uses
      0.2; RLLib defaults to 0.3 which allows excessively large updates.
    grad_clip: Global gradient norm clip. RLLib default None (no clipping)
      can cause instability. 0.5 is standard.
    model_type: "lstm" for CNN+LSTM (default), "cnn" for CNN-only baseline.

  Returns:
    The configuration for running the experiment.
  """
  # Resolve model registration name
  if model_type == "cnn":
    custom_model_name = "meltingpot_cnn_model"
  else:
    custom_model_name = "meltingpot_model"
  # Gets the default training configuration
  config = ppo.PPOConfig()
  # Number of arenas.
  config.num_rollout_workers = num_rollout_workers
  # This is to match our unroll lengths.
  config.rollout_fragment_length = rollout_fragment_length
  # Total (time x batch) timesteps on the learning update.
  config.train_batch_size = train_batch_size
  # Mini-batch size.
  config.sgd_minibatch_size = sgd_minibatch_size
  # SGD epochs per iteration. PPO default (30) is too aggressive for
  # multi-agent: most benefit comes in the first few passes and extra
  # epochs dominate wall-clock time (~86% of iteration at 30 epochs).
  config.num_sgd_iter = num_sgd_iter
  # Entropy bonus — critical for exploration. PPO default is 0.0 which
  # means zero incentive to explore; agents quickly converge to a bad
  # deterministic policy. 0.003–0.01 is standard for multi-agent.
  config.entropy_coeff = entropy_coeff
  # Learning rate.
  config.lr = lr
  # KL penalty — RLLib default 0.2 is the #1 cause of "KL explosion" in
  # multi-agent PPO. The adaptive coefficient grows every time KL exceeds
  # kl_target, eventually preventing any policy updates. Disable it and
  # rely solely on clipping (PPO-Clip), which is what most successful
  # implementations (CleanRL, SB3, OpenAI Five) do.
  config.kl_coeff = kl_coeff
  config.kl_target = kl_target
  # Value function clip — RLLib default 10.0 is way too small when
  # episode rewards are 100-500+. The VF can't track the true value,
  # leading to bad advantage estimates and cascading policy degradation.
  config.vf_clip_param = vf_clip_param
  # PPO clip parameter — 0.2 is standard (OpenAI's original).
  config.clip_param = clip_param
  # Gradient clipping — prevents exploding gradients.
  config.grad_clip = grad_clip
  # Observations are already flat Box (via MeltingPotObsWrapper).
  # Use PyTorch as the tensor framework (GPU-compatible).
  config = config.framework("torch")
  # Use 1 GPU for the learner, rollout workers stay on CPU.
  config.num_gpus = 1
  config.log_level = "INFO"

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  player_roles = substrate.get_config(substrate_name).default_player_roles
  config.env_config = {"substrate": substrate_name, "roles": player_roles}

  config.env = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config.env_config)

  # Setup PPO with policies, one per entry in default player roles.
  policies = {}
  player_to_agent = {}
  c1, c2 = int(cnn_channels[0]), int(cnn_channels[1])
  for i in range(len(player_roles)):
    # Get RGB shape from the underlying MeltingPotEnv (before flattening)
    rgb_shape = test_env._rgb_shape[f"player_{i}"]  # (H, W, C)
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8
    second_kernel = [sprite_x, sprite_y] if cnn_second_kernel is None else [
      int(cnn_second_kernel[0]), int(cnn_second_kernel[1])
    ]
    conv_filters = [
      [c1, [int(cnn_first_kernel), int(cnn_first_kernel)], int(cnn_first_stride)],
      [c2, second_kernel, int(cnn_second_stride)],
    ]

    policies[f"agent_{i}"] = policy.PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"],
        config={
            "model": {
                "custom_model": custom_model_name,
                "custom_model_config": {
                "conv_filters": conv_filters,
                    "rgb_shape": list(rgb_shape),
                },
            },
        })
    player_to_agent[f"player_{i}"] = f"agent_{i}"

  def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    return player_to_agent[agent_id]

  # 5. Configuration for multi-agent setup with one policy per role:
  config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

  # 6. Set the agent architecture — handled by custom model (MeltingPotModel).
  # The custom model internally builds CNN + FC + LSTM, so we disable
  # RLlib's built-in LSTM wrapper. Config values are passed through to the
  # custom model via model_config.
  config.model["fcnet_hiddens"] = fcnet_hiddens
  config.model["fcnet_activation"] = "relu"
  config.model["conv_activation"] = "relu"
  config.model["post_fcnet_hiddens"] = post_fcnet_hiddens
  config.model["post_fcnet_activation"] = "relu"
  # LSTM is handled inside the custom model — do NOT set use_lstm=True
  # to avoid RLlib's buggy LSTMWrapper with Dict observations.
  config.model["use_lstm"] = False
  config.model["max_seq_len"] = 100
  config.model["lstm_use_prev_action"] = True
  config.model["lstm_use_prev_reward"] = False
  config.model["lstm_cell_size"] = lstm_cell_size

  # 7. Compress observations to reduce object store / network pressure.
  config["compress_observations"] = True

  # 8. Worker fault tolerance — survive OOM kills on shared machines.
  # When Ray's OOM monitor kills a worker, RLlib will automatically
  # recreate it instead of crashing the whole training run.
  config = config.fault_tolerance(
      recreate_failed_workers=True,
      num_consecutive_worker_failures_tolerance=100,
      restart_failed_sub_environments=True,
  )

  return config


def train(config, num_iterations=1, output_dir=None,
          phi_mode="fixed", phi_start_deg=0.0, phi_end_deg=45.0,
          phis_deg=None, num_agents=7, checkpoint_freq=5,
          resume_from=None, object_store_gb=8.0,
          phi_schedule="linear"):
  """Trains a model with optional SVO (phi) reward shaping.

  Args:
    config: PPO config.
    num_iterations: number of training iterations.
    output_dir: directory to save checkpoints and logs.
    phi_mode: "fixed" or "curriculum".
    phi_start_deg: starting phi (curriculum mode).
    phi_end_deg: ending phi (curriculum mode).
    phis_deg: list of per-agent phi in degrees (current values for fixed mode).
    num_agents: number of agents in the substrate.
    checkpoint_freq: save checkpoint every N iterations.
    resume_from: path to checkpoint directory to restore from.

  Returns:
    Last training result dict.
  """
  from tqdm import tqdm

  tune.register_env("meltingpot", utils.env_creator)
  ModelCatalog.register_custom_model("meltingpot_model", MeltingPotModel)
  ModelCatalog.register_custom_model("meltingpot_cnn_model", MeltingPotCNNModel)

  # ---- Memory-safe Ray init ----
  # Raise OOM kill threshold to 0.98 (default 0.95 is too aggressive
  # when other users consume RAM).
  os.environ.setdefault("RAY_memory_usage_threshold", "0.98")
  obj_store_bytes = int(object_store_gb * 1024**3)
  print(f"Ray object store: {object_store_gb:.1f} GB")
  ray.init(
      object_store_memory=obj_store_bytes,
      _system_config={
          "automatic_object_spilling_enabled": True,
          "object_spilling_threshold": 0.9,
      },
  )

  algo = config.build()

  # ---- Resume from checkpoint if requested ----
  start_iteration = 1
  if resume_from is not None:
    print(f"Restoring from checkpoint: {resume_from}")
    algo.restore(resume_from)
    # Try to determine which iteration we're resuming from by reading the CSV
    existing_csv = os.path.join(output_dir, "training_log.csv")
    if os.path.exists(existing_csv):
      with open(existing_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
          start_iteration = int(rows[-1]["iteration"]) + 1
          print(f"Resuming from iteration {start_iteration}")

  # ---- Prepare CSV log ----
  agent_ids = [f"agent_{i}" for i in range(num_agents)]
  csv_path = os.path.join(output_dir, "training_log.csv")
  csv_fields = (
      ["iteration", "timestamp", "phi_mode",
       "phi_start_deg", "phi_end_deg"]
      + [f"phi_agent_{i}_deg" for i in range(num_agents)]
      + ["reward_mean", "reward_min", "reward_max",
         "episode_len_mean", "total_env_steps",
         "ram_used_gb", "ram_pct"]
      + [f"reward_agent_{i}" for i in range(num_agents)]
      + ["gini"]
      + [f"kl_agent_{i}" for i in range(num_agents)]
      + [f"entropy_agent_{i}" for i in range(num_agents)]
      + [f"policy_loss_agent_{i}" for i in range(num_agents)]
      + [f"vf_loss_agent_{i}" for i in range(num_agents)]
      + [f"cur_kl_coeff_agent_{i}" for i in range(num_agents)]
      + ["kl_mean", "entropy_mean", "policy_loss_mean", "vf_loss_mean"]
      + ["substrate", "num_rollout_workers", "train_batch_size",
         "sgd_minibatch_size", "num_sgd_iter", "rollout_fragment_length",
         "fcnet_hiddens", "post_fcnet_hiddens", "lstm_cell_size",
         "num_gpus"]
  )
  # Append mode if resuming, otherwise create fresh
  if start_iteration > 1 and os.path.exists(csv_path):
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
  else:
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
  csv_file.flush()

  # Static config values (same every row)
  static_cfg = {
      "substrate": config.env_config["substrate"],
      "num_rollout_workers": config.num_rollout_workers,
      "train_batch_size": config.train_batch_size,
      "sgd_minibatch_size": config.sgd_minibatch_size,
      "num_sgd_iter": config.num_sgd_iter,
      "rollout_fragment_length": config.rollout_fragment_length,
      "fcnet_hiddens": str(list(config.model["fcnet_hiddens"])),
      "post_fcnet_hiddens": str(list(config.model["post_fcnet_hiddens"])),
      "lstm_cell_size": config.model["lstm_cell_size"],
      "num_gpus": config.num_gpus,
      "phi_mode": phi_mode,
      "phi_start_deg": phi_start_deg,
      "phi_end_deg": phi_end_deg,
  }

  # ---- Training loop ----
  current_phis_deg = list(phis_deg)  # mutable copy
  prev_total_steps = 0
  stall_count = 0
  best_reward = float("-inf")
  best_iteration = 0
  iters_remaining = num_iterations - start_iteration + 1
  pbar = tqdm(range(start_iteration, num_iterations + 1),
              desc="Training", unit="iter",
              initial=start_iteration - 1, total=num_iterations)
  try:
    for i in pbar:
      # -- Curriculum: update phi on all worker environments --
      if phi_mode == "curriculum":
        schedule_fn = PHI_SCHEDULES[phi_schedule]
        base_phi = schedule_fn(i, num_iterations, phi_start_deg, phi_end_deg)
        current_phis_deg = [base_phi] * num_agents
        new_phis_rad = [math.radians(base_phi)] * num_agents
        try:
          algo.workers.foreach_worker(
              lambda w, phis=new_phis_rad: w.foreach_env(
                  lambda e: e.update_phis(phis)),
              local_worker=False)
        except Exception as e:
          tqdm.write(f"  [WARN] Failed to update phis on workers: {e}")

      result = algo.train()

      # -- Force garbage collection to limit memory growth --
      gc.collect()

      # -- Extract metrics --
      reward_mean = result.get("episode_reward_mean", float("nan"))
      reward_min = result.get("episode_reward_min", float("nan"))
      reward_max = result.get("episode_reward_max", float("nan"))
      ep_len = result.get("episode_len_mean", float("nan"))
      total_steps = result.get("num_env_steps_sampled",
                               result.get("num_env_steps_sampled_lifetime", 0))

      # -- Memory monitoring --
      mem = psutil.virtual_memory()
      ram_used_gb = mem.used / (1024**3)
      ram_pct = mem.percent

      # Detect stalled training (no new data collected)
      if i > start_iteration and total_steps == prev_total_steps:
        stall_count += 1
        tqdm.write(f"  [WARN] iter {i}: no new env steps (stall #{stall_count})!"
                   f" RAM: {ram_used_gb:.1f}GB ({ram_pct:.0f}%)")
        if stall_count >= 5:
          tqdm.write(f"  [ERROR] {stall_count} consecutive stalls — workers "
                     f"are likely stuck in OOM kill loop. Consider reducing "
                     f"--num-rollout-workers or freeing RAM.")
      else:
        stall_count = 0
      prev_total_steps = total_steps

      # Warn if memory is dangerously high
      if ram_pct > 90:
        tqdm.write(f"  [WARN] RAM {ram_used_gb:.1f}GB ({ram_pct:.0f}%) — "
                   f"approaching OOM threshold")

      # Per-agent rewards from policy_reward_mean
      policy_rewards = result.get("policy_reward_mean", {})
      per_agent = [policy_rewards.get(aid, float("nan")) for aid in agent_ids]
      gini = gini_coefficient([r for r in per_agent if not math.isnan(r)])

      # -- Extract per-policy learner stats (KL, entropy, losses) --
      learner_info = result.get("info", {}).get("learner", {})
      per_kl = []
      per_entropy = []
      per_policy_loss = []
      per_vf_loss = []
      per_kl_coeff = []
      for aid in agent_ids:
        stats = learner_info.get(aid, {}).get("learner_stats", {})
        per_kl.append(stats.get("kl", float("nan")))
        per_entropy.append(stats.get("entropy", float("nan")))
        per_policy_loss.append(stats.get("policy_loss", float("nan")))
        per_vf_loss.append(stats.get("vf_loss", float("nan")))
        per_kl_coeff.append(stats.get("cur_kl_coeff", float("nan")))

      # Compute means (ignoring NaN)
      def _nanmean(vals):
        valid = [v for v in vals if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")
      kl_mean = _nanmean(per_kl)
      entropy_mean = _nanmean(per_entropy)
      policy_loss_mean = _nanmean(per_policy_loss)
      vf_loss_mean = _nanmean(per_vf_loss)

      # -- Detect KL explosion and entropy collapse --
      if not math.isnan(kl_mean) and kl_mean > 0.1:
        tqdm.write(f"  [ALERT] KL EXPLOSION @ iter {i}: kl_mean={kl_mean:.4f} "
                   f"(healthy < 0.02, dangerous > 0.1)")
      if not math.isnan(entropy_mean) and entropy_mean < 0.1:
        tqdm.write(f"  [ALERT] ENTROPY COLLAPSE @ iter {i}: "
                   f"entropy_mean={entropy_mean:.4f} "
                   f"(policy is near-deterministic, no exploration)")
      # Warn if KL coeff is growing (adaptive penalty active)
      kl_coeff_vals = [v for v in per_kl_coeff if not math.isnan(v)]
      if kl_coeff_vals and max(kl_coeff_vals) > 1.0:
        tqdm.write(f"  [ALERT] KL COEFF RUNAWAY @ iter {i}: "
                   f"max_kl_coeff={max(kl_coeff_vals):.4f} "
                   f"(policy updates being penalized heavily)")

      # -- Update tqdm --
      postfix = {"reward": f"{reward_mean:.1f}", "steps": total_steps,
                 "RAM": f"{ram_pct:.0f}%"}
      if not math.isnan(kl_mean):
        postfix["KL"] = f"{kl_mean:.4f}"
      if not math.isnan(entropy_mean):
        postfix["ent"] = f"{entropy_mean:.3f}"
      if phi_mode == "curriculum":
        postfix["phi"] = f"{current_phis_deg[0]:.1f}"
      pbar.set_postfix(**postfix)

      # -- Write CSV row --
      row = dict(static_cfg)
      row["iteration"] = i
      row["timestamp"] = datetime.now().isoformat()
      for j in range(num_agents):
        row[f"phi_agent_{j}_deg"] = f"{current_phis_deg[j]:.4f}"
      row["reward_mean"] = reward_mean
      row["reward_min"] = reward_min
      row["reward_max"] = reward_max
      row["episode_len_mean"] = ep_len
      row["total_env_steps"] = total_steps
      row["ram_used_gb"] = f"{ram_used_gb:.2f}"
      row["ram_pct"] = f"{ram_pct:.1f}"
      for j in range(num_agents):
        row[f"reward_agent_{j}"] = per_agent[j]
      row["gini"] = f"{gini:.6f}"
      for j in range(num_agents):
        row[f"kl_agent_{j}"] = per_kl[j]
        row[f"entropy_agent_{j}"] = per_entropy[j]
        row[f"policy_loss_agent_{j}"] = per_policy_loss[j]
        row[f"vf_loss_agent_{j}"] = per_vf_loss[j]
        row[f"cur_kl_coeff_agent_{j}"] = per_kl_coeff[j]
      row["kl_mean"] = kl_mean
      row["entropy_mean"] = entropy_mean
      row["policy_loss_mean"] = policy_loss_mean
      row["vf_loss_mean"] = vf_loss_mean
      csv_writer.writerow(row)
      csv_file.flush()

      # -- Checkpoint --
      if i % checkpoint_freq == 0:
        ckpt_dir = os.path.join(output_dir, f"checkpoint_{i:05d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt = algo.save(ckpt_dir)
        tqdm.write(f"  Checkpoint saved @ iter {i}: {ckpt_dir}")

      # -- Best checkpoint --
      if not math.isnan(reward_mean) and reward_mean > best_reward:
        best_reward = reward_mean
        best_iteration = i
        best_dir = os.path.join(output_dir, "checkpoint_best")
        os.makedirs(best_dir, exist_ok=True)
        algo.save(best_dir)
        # Write a small metadata file so we know which iteration this was
        with open(os.path.join(best_dir, "best_info.json"), "w") as bf:
          json.dump({"iteration": i, "reward_mean": reward_mean,
                     "timestamp": datetime.now().isoformat()}, bf, indent=2)
        tqdm.write(f"  ★ New best @ iter {i}: reward={reward_mean:.2f} "
                   f"(saved to checkpoint_best/)")

    # Final checkpoint — also save in run root for easy access
    ckpt_dir = os.path.join(output_dir, f"checkpoint_{num_iterations:05d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    algo.save(ckpt_dir)
    ckpt = algo.save(output_dir)  # overwrite root for --resume-from convenience
    tqdm.write(f"Final checkpoint: {ckpt_dir} (also at {output_dir})")
    tqdm.write(f"Best reward: {best_reward:.2f} @ iteration {best_iteration} "
               f"(checkpoint_best/)")

  finally:
    csv_file.close()
    tqdm.write(f"Training log saved: {csv_path}")

  algo.stop()
  return result


def main():
  parser = argparse.ArgumentParser(description="MeltingPot self-play training")

  # --- Environment ---
  parser.add_argument("--substrate", type=str,
                       default="commons_harvest__open",
                       help="MeltingPot substrate name")

  # --- Training ---
  parser.add_argument("--num-iterations", type=int, default=20,
                       help="Number of training iterations")
  parser.add_argument("--num-rollout-workers", type=int, default=12,
                       help="Number of rollout workers (default 4; keep low "
                            "on shared machines to avoid OOM)")
  parser.add_argument("--train-batch-size", type=int, default=12800,
                       help="Training batch size")
  parser.add_argument("--sgd-minibatch-size", type=int, default=256,
                       help="SGD mini-batch size")
  parser.add_argument("--num-sgd-iter", type=int, default=10,
                       help="Number of SGD epochs per PPO iteration (default 10; "
                            "PPO default is 30 but that dominates wall-clock time "
                            "in multi-agent — most gain comes in first 5-10 passes)")
  parser.add_argument("--entropy-coeff", type=float, default=0.01,
                       help="Entropy bonus coefficient for exploration (default "
                            "0.003; PPO default 0.0 gives no exploration "
                            "pressure — try 0.003-0.01 for multi-agent)")
  parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate (default 3e-4; PPO default 5e-5 is "
                            "very conservative)")
  parser.add_argument("--kl-coeff", type=float, default=1e-8,
                       help="Adaptive KL penalty coefficient (default 1e-8 = "
                            "effectively disabled but KL still computed for "
                            "diagnostics; RLLib default 0.2 causes KL "
                            "explosion in multi-agent; 0.0 skips KL "
                            "computation entirely)")
  parser.add_argument("--kl-target", type=float, default=0.01,
                       help="Target KL divergence for adaptive penalty "
                            "(only relevant if --kl-coeff > 0)")
  parser.add_argument("--vf-clip-param", type=float, default=100.0,
                       help="Value function clip range (default 100.0; RLLib "
                            "default 10.0 is too small for envs with reward "
                            "magnitudes > 10)")
  parser.add_argument("--clip-param", type=float, default=0.2,
                       help="PPO surrogate objective clip (default 0.2; "
                            "RLLib default 0.3 is too permissive)")
  parser.add_argument("--grad-clip", type=float, default=0.5,
                       help="Global gradient norm clip (default 0.5; RLLib "
                            "default None = no clipping)")
  parser.add_argument("--rollout-fragment-length", type=int, default=100,
                       help="Rollout fragment length")
  parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs for the learner")
  parser.add_argument("--checkpoint-freq", type=int, default=10,
                       help="Save a checkpoint every N iterations")

  # --- Model architecture ---
  parser.add_argument("--model", type=str, default="lstm",
                       choices=["lstm", "cnn"],
                       help="Model type: 'lstm' for CNN+LSTM (default), "
                            "'cnn' for CNN-only feed-forward baseline")
  parser.add_argument("--lstm-cell-size", type=int, default=256,
                       help="LSTM hidden size (memory) for --model lstm")
  parser.add_argument("--cnn-channels", type=str, default="16,128",
                       help="CNN channels as 'C1,C2' (default: 16,128)")
  parser.add_argument("--cnn-first-kernel", type=int, default=8,
                       help="Kernel size for first conv layer (square)")
  parser.add_argument("--cnn-first-stride", type=int, default=8,
                       help="Stride for first conv layer")
  parser.add_argument("--cnn-second-kernel", type=str, default="auto",
                       help="Second conv kernel as 'H,W' or 'auto' to use "
                            "(rgb_h//8, rgb_w//8)")
  parser.add_argument("--cnn-second-stride", type=int, default=1,
                       help="Stride for second conv layer")

  # --- SVO / Phi ---
  parser.add_argument("--phi", type=str, default="0",
                       help="Phi angle(s) in degrees. Single value applies to "
                            "all agents; comma-separated for per-agent, e.g. "
                            "'30,15,45,0,30,10,20'")
  parser.add_argument("--phi-mode", type=str, default="fixed",
                       choices=["fixed", "curriculum"],
                       help="fixed: constant phi. curriculum: linearly ramp "
                            "phi from --phi-start to --phi-end over training.")
  parser.add_argument("--phi-start", type=float, default=0.0,
                       help="Starting phi in degrees (curriculum mode)")
  parser.add_argument("--phi-end", type=float, default=45.0,
                       help="Ending phi in degrees (curriculum mode)")
  parser.add_argument("--phi-schedule", type=str, default="linear",
                       choices=["linear", "log", "exp", "sigmoid"],
                       help="Curriculum schedule shape: linear (constant rate), "
                            "log (fast early, slow late), exp (slow early, "
                            "fast late), sigmoid (S-curve centered at midpoint)")

  # --- Memory ---
  parser.add_argument("--object-store-gb", type=float, default=8.0,
                       help="Ray object store size in GB (default 8). Increase "
                            "if you see heavy spilling; decrease on low-RAM "
                            "machines.")

  # --- Resilience ---
  parser.add_argument("--max-failures", type=int, default=100,
                       help="Max consecutive worker failures before abort")
  parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to a checkpoint directory to resume from")

  # --- Output ---
  parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save checkpoints (default: auto)")

  args = parser.parse_args()

  # ---- Parse model size args ----
  try:
    cnn_channels = tuple(int(x.strip()) for x in args.cnn_channels.split(","))
  except ValueError as e:
    parser.error(f"--cnn-channels must be two integers like '16,128' ({e})")
  if len(cnn_channels) != 2:
    parser.error("--cnn-channels must have exactly two values, e.g. '16,128'")

  if args.cnn_second_kernel.lower() == "auto":
    cnn_second_kernel = None
  else:
    try:
      kernel_vals = tuple(int(x.strip()) for x in args.cnn_second_kernel.split(","))
    except ValueError as e:
      parser.error(f"--cnn-second-kernel must be 'auto' or 'H,W' ({e})")
    if len(kernel_vals) != 2:
      parser.error("--cnn-second-kernel must be 'auto' or two ints like '11,11'")
    cnn_second_kernel = kernel_vals

  # ---- Build config ----
  config = get_config(
      substrate_name=args.substrate,
      num_rollout_workers=args.num_rollout_workers,
      rollout_fragment_length=args.rollout_fragment_length,
      train_batch_size=args.train_batch_size,
      lstm_cell_size=args.lstm_cell_size,
      cnn_channels=cnn_channels,
      cnn_first_kernel=args.cnn_first_kernel,
      cnn_first_stride=args.cnn_first_stride,
      cnn_second_kernel=cnn_second_kernel,
      cnn_second_stride=args.cnn_second_stride,
      sgd_minibatch_size=args.sgd_minibatch_size,
      num_sgd_iter=args.num_sgd_iter,
      entropy_coeff=args.entropy_coeff,
      lr=args.lr,
      kl_coeff=args.kl_coeff,
      kl_target=args.kl_target,
      vf_clip_param=args.vf_clip_param,
      clip_param=args.clip_param,
      grad_clip=args.grad_clip,
      model_type=args.model,
  )
  config.num_gpus = args.num_gpus

  # Override fault tolerance with CLI value
  config = config.fault_tolerance(
      recreate_failed_workers=True,
      num_consecutive_worker_failures_tolerance=args.max_failures,
      restart_failed_sub_environments=True,
  )

  # ---- Resolve phi values ----
  num_agents = len(config.env_config["roles"])

  if args.phi_mode == "fixed":
    phi_values_deg = [float(x) for x in args.phi.split(",")]
    if len(phi_values_deg) == 1:
      phi_values_deg = phi_values_deg * num_agents
    if len(phi_values_deg) != num_agents:
      parser.error(f"--phi has {len(phi_values_deg)} values but substrate "
                   f"has {num_agents} agents. Provide 1 or {num_agents} values.")
  else:  # curriculum — start at phi_start
    phi_values_deg = [args.phi_start] * num_agents

  # Inject phis into env_config so workers create PhiRewardWrapper
  phis_rad = [math.radians(p) for p in phi_values_deg]
  config.env_config["phis"] = phis_rad

  # ---- Auto-generate run directory ----
  if args.resume_from is not None and args.output_dir is None:
    # When resuming, default to the same directory as the checkpoint
    output_dir = os.path.dirname(args.resume_from)
    # If checkpoint was saved in a subdirectory, go up to the run dir
    if os.path.basename(output_dir).startswith("checkpoint_"):
      output_dir = os.path.dirname(output_dir)
  elif args.output_dir is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    phi_tag = f"phi{args.phi}" if args.phi_mode == "fixed" else \
              f"curriculum_{args.phi_start}-{args.phi_end}"
    run_name = f"{args.substrate}_{phi_tag}_{timestamp}"
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = os.path.join(project_root, 'run_checkpoint', run_name)
  else:
    output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # ---- Save run config as JSON ----
  run_config = {
      "substrate": args.substrate,
      "num_agents": num_agents,
      "num_iterations": args.num_iterations,
      "num_rollout_workers": args.num_rollout_workers,
      "train_batch_size": args.train_batch_size,
      "sgd_minibatch_size": args.sgd_minibatch_size,
      "num_sgd_iter": args.num_sgd_iter,
      "entropy_coeff": args.entropy_coeff,
      "lr": args.lr,
      "kl_coeff": args.kl_coeff,
      "kl_target": args.kl_target,
      "vf_clip_param": args.vf_clip_param,
      "clip_param": args.clip_param,
      "grad_clip": args.grad_clip,
      "rollout_fragment_length": args.rollout_fragment_length,
      "num_gpus": args.num_gpus,
      "checkpoint_freq": args.checkpoint_freq,
      "phi_mode": args.phi_mode,
      "phi_schedule": args.phi_schedule,
      "phi_start_deg": args.phi_start,
      "phi_end_deg": args.phi_end,
      "phi_per_agent_deg": phi_values_deg,
        "cnn_channels": list(cnn_channels),
        "cnn_first_kernel": args.cnn_first_kernel,
        "cnn_first_stride": args.cnn_first_stride,
        "cnn_second_kernel": (
          "auto" if cnn_second_kernel is None else list(cnn_second_kernel)
        ),
        "cnn_second_stride": args.cnn_second_stride,
      "fcnet_hiddens": list(config.model["fcnet_hiddens"]),
      "post_fcnet_hiddens": list(config.model["post_fcnet_hiddens"]),
      "lstm_cell_size": config.model["lstm_cell_size"],
      "model_type": args.model,
      "object_store_gb": args.object_store_gb,
      "timestamp": datetime.now().isoformat(),
  }
  config_path = os.path.join(output_dir, "run_config.json")
  with open(config_path, "w") as f:
    json.dump(run_config, f, indent=2)

  print(f"Run directory: {output_dir}")
  print(f"Substrate: {args.substrate} ({num_agents} agents)")
  print(f"Model: {args.model.upper()} ({'CNN+LSTM' if args.model == 'lstm' else 'CNN-only baseline'})")
  print("Model params: "
      f"lstm_cell_size={args.lstm_cell_size}, "
      f"cnn_channels={list(cnn_channels)}, "
      f"cnn1=({args.cnn_first_kernel}x{args.cnn_first_kernel}, s={args.cnn_first_stride}), "
      f"cnn2=({'auto' if cnn_second_kernel is None else list(cnn_second_kernel)}, s={args.cnn_second_stride})")
  phi_sched_str = f" ({args.phi_schedule})" if args.phi_mode == "curriculum" else ""
  print(f"Phi mode: {args.phi_mode}{phi_sched_str} | "
        f"phi={phi_values_deg if args.phi_mode == 'fixed' else f'{args.phi_start}° → {args.phi_end}°'}")

  # ---- Memory check ----
  mem = psutil.virtual_memory()
  print(f"System RAM: {mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB "
        f"({mem.percent:.0f}%) | Swap: {psutil.swap_memory().percent:.0f}%")
  if mem.percent > 85:
    print(f"WARNING: RAM already at {mem.percent:.0f}% before training. "
          f"Consider using fewer workers (--num-rollout-workers {max(2, args.num_rollout_workers - 2)}) "
          f"or freeing memory.")

  # ---- Train ----
  results = train(
      config,
      num_iterations=args.num_iterations,
      output_dir=output_dir,
      phi_mode=args.phi_mode,
      phi_start_deg=args.phi_start,
      phi_end_deg=args.phi_end,
      phis_deg=phi_values_deg,
      num_agents=num_agents,
      checkpoint_freq=args.checkpoint_freq,
      resume_from=args.resume_from,
      object_store_gb=args.object_store_gb,
      phi_schedule=args.phi_schedule,
  )
  print("Training complete!")
  print(f"Final reward: {results.get('episode_reward_mean', 'N/A')}")
  print(f"Checkpoints + logs saved to: {output_dir}")


if __name__ == "__main__":
  main()
