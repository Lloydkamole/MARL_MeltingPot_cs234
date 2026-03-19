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
"""MeltingPotEnv as a MultiAgentEnv wrapper to interface with RLLib."""

from typing import Tuple

import dm_env
import dmlab2d
from gymnasium import spaces
from meltingpot import substrate
from meltingpot.utils.policies import policy
from ml_collections import config_dict
import numpy as np
from ray.rllib import algorithms
from ray.rllib.env import multi_agent_env
from ray.rllib.policy import sample_batch

from ..gym import utils

PLAYER_STR_FORMAT = 'player_{index}'


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    """Initializes the instance.

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
    """
    self._env = env
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    # RLLib requires environments to have the following member variables:
    # observation_space, action_space, and _agent_ids
    self._agent_ids = set(self._ordered_agent_ids)
    # RLLib expects a dictionary of agent_id to observation or action,
    # Melting Pot uses a tuple, so we convert
    self.observation_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.observation_spec()),
        remove_world_observations=True)
    self.action_space = self._convert_spaces_tuple_to_dict(
        utils.spec_to_space(self._env.action_spec()))
    super().__init__()

  def reset(self, *args, **kwargs):
    """See base class."""
    timestep = self._env.reset()
    return utils.timestep_to_observations(timestep), {}

  def step(self, action_dict):
    """See base class."""
    actions = [action_dict[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    done = {'__all__': timestep.last()}
    info = {}

    observations = utils.timestep_to_observations(timestep)
    return observations, rewards, done, done, info

  def close(self):
    """See base class."""
    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""
    return self._env

  # Metadata is required by the gym `Env` class that we are extending, to show
  # which modes the `render` method supports.
  metadata = {'render.modes': ['rgb_array']}

  def render(self) -> np.ndarray:
    """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """
    observation = self._env.observation()
    world_rgb = observation[0]['WORLD.RGB']

    # RGB mode is used for recording videos
    return world_rgb

  def _convert_spaces_tuple_to_dict(
      self,
      input_tuple: spaces.Tuple,
      remove_world_observations: bool = False) -> spaces.Dict:
    """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """
    return spaces.Dict({
        agent_id: (utils.remove_world_observations_from_space(input_tuple[i])
                   if remove_world_observations else input_tuple[i])
        for i, agent_id in enumerate(self._ordered_agent_ids)
    })


class MeltingPotObsWrapper(multi_agent_env.MultiAgentEnv):
  """Wraps MeltingPotEnv to flatten Dict obs into a single Box per player.

  Layout: [RGB_flat (H*W*3, uint8)] ++ [scalar1, scalar2, ...] as float32.
  Scalars are sorted alphabetically. RGB is placed first so the model can
  split it deterministically.
  """

  def __init__(self, env: MeltingPotEnv, phi_wrapper=None):
    self._env = env
    self._phi_wrapper = phi_wrapper
    self._agent_ids = env._agent_ids
    self._ordered_agent_ids = env._ordered_agent_ids

    # Build new flat observation space
    self.observation_space = spaces.Dict()
    self.action_space = env.action_space
    self._rgb_keys = {}    # per-agent
    self._scalar_keys = {} # per-agent
    self._rgb_shape = {}   # per-agent: (H, W, C)

    for agent_id in self._ordered_agent_ids:
      orig = env.observation_space[agent_id]
      rgb_keys = sorted(k for k, s in orig.spaces.items()
                        if isinstance(s, spaces.Box) and len(s.shape) == 3)
      scalar_keys = sorted(k for k in orig.spaces
                           if k not in rgb_keys)
      self._rgb_keys[agent_id] = rgb_keys
      self._scalar_keys[agent_id] = scalar_keys

      flat_size = 0
      for k in rgb_keys:
        s = orig[k]
        self._rgb_shape[agent_id] = s.shape  # (H, W, C)
        flat_size += int(np.prod(s.shape))
      for k in scalar_keys:
        s = orig[k]
        flat_size += int(np.prod(s.shape)) if len(s.shape) > 0 else 1

      self.observation_space[agent_id] = spaces.Box(
          low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32)

    super().__init__()

  def _flatten_obs(self, obs_dict):
    """Convert dict of {agent_id: {key: array}} to {agent_id: flat_array}."""
    flat = {}
    for agent_id in self._ordered_agent_ids:
      if agent_id not in obs_dict:
        continue
      obs = obs_dict[agent_id]
      parts = []
      # RGB first (as float32)
      for k in self._rgb_keys[agent_id]:
        parts.append(obs[k].flatten().astype(np.float32))
      # Then scalars
      for k in self._scalar_keys[agent_id]:
        v = obs[k]
        if np.ndim(v) == 0:
          parts.append(np.array([float(v)], dtype=np.float32))
        else:
          parts.append(v.flatten().astype(np.float32))
      flat[agent_id] = np.concatenate(parts)
    return flat

  def reset(self, *args, **kwargs):
    obs, info = self._env.reset(*args, **kwargs)
    return self._flatten_obs(obs), info

  def step(self, action_dict):
    obs, rewards, terminated, truncated, info = self._env.step(action_dict)
    return self._flatten_obs(obs), rewards, terminated, truncated, info

  def close(self):
    self._env.close()

  def render(self):
    return self._env.render()

  def get_dmlab2d_env(self):
    return self._env.get_dmlab2d_env()

  def update_phis(self, phis):
    """Update phi angles (radians) on the PhiRewardWrapper mid-training."""
    if self._phi_wrapper is not None:
      self._phi_wrapper._phis = np.asarray(phis, dtype=np.float64)
      self._phi_wrapper._n = len(self._phi_wrapper._phis)
      self._phi_wrapper._cos_phi = np.cos(self._phi_wrapper._phis)
      self._phi_wrapper._sin_phi = np.sin(self._phi_wrapper._phis)

  metadata = {'render.modes': ['rgb_array']}


def env_creator(env_config):
  """Outputs an environment for registering."""
  env_config = config_dict.ConfigDict(env_config)
  env = substrate.build(env_config['substrate'], roles=env_config['roles'])

  # Optionally wrap with PhiRewardWrapper for SVO reward shaping.
  phi_wrapper = None
  phis = env_config.get('phis', None)
  if phis is not None:
    from phy_reward_warper import PhiRewardWrapper
    phi_wrapper = PhiRewardWrapper(env, list(phis))
    env = phi_wrapper

  env = MeltingPotEnv(env)
  env = MeltingPotObsWrapper(env, phi_wrapper=phi_wrapper)
  return env


class RayModelPolicy(policy.Policy[policy.State]):
  """Policy wrapping an RLLib model for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               model: algorithms.Algorithm,
               policy_id: str = sample_batch.DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      model: An rllib.trainer.Trainer checkpoint.
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    self._model = model
    self._prev_action = 0
    self._policy_id = policy_id

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key
    }

    action, state, _ = self._model.compute_single_action(
        observations,
        prev_state,
        policy_id=self._policy_id,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._model.get_policy(self._policy_id).get_initial_state()

  def close(self) -> None:
    """See base class."""
