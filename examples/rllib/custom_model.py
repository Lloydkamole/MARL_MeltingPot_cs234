# Copyright 2024 — Custom models for MeltingPot + PyTorch.
#
# Two model variants:
#   1. MeltingPotModel — CNN + FC + LSTM (recurrent, default)
#   2. MeltingPotCNNModel — CNN + FC only (feed-forward baseline)
#
# Both receive a flat Box observation (created by MeltingPotObsWrapper)
# and unfold it back into RGB + scalar parts.
#
# This bypasses RLlib's buggy Dict-obs + LSTM pipeline by making the
# observation a simple Box at the RLlib level.

import numpy as np
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from typing import Dict, List, Tuple

torch, nn = try_import_torch()


# ---------------------------------------------------------------------------
# Helper: build a CNN + scalar-FC from config (shared by both models)
# ---------------------------------------------------------------------------

def _build_vision_torso(obs_space, model_config):
    """Build CNN for RGB and FC for scalars. Returns (cnn, scalar_fc,
    rgb_shape, rgb_flat_size, scalar_size, concat_size)."""
    custom_cfg = model_config.get("custom_model_config", {})
    conv_filters = custom_cfg.get("conv_filters",
                                  [[16, [8, 8], 8], [128, [5, 5], 1]])
    rgb_shape = tuple(custom_cfg.get("rgb_shape", [40, 40, 3]))
    fc_hiddens = list(model_config.get("fcnet_hiddens", [64, 64]))
    activation_fn = nn.ReLU

    h, w, c = rgb_shape
    rgb_flat_size = h * w * c
    total_obs = int(np.prod(obs_space.shape))
    scalar_size = total_obs - rgb_flat_size

    # CNN
    cnn_layers = []
    in_channels = c
    out_h, out_w = h, w
    for out_channels, kernel, stride in conv_filters:
        kh = kernel[0] if isinstance(kernel, (list, tuple)) else kernel
        kw = kernel[1] if isinstance(kernel, (list, tuple)) else kernel
        cnn_layers.append(nn.Conv2d(in_channels, out_channels,
                                    (kh, kw), stride=stride))
        cnn_layers.append(activation_fn())
        in_channels = out_channels
        out_h = (out_h - kh) // stride + 1
        out_w = (out_w - kw) // stride + 1
    cnn_layers.append(nn.Flatten())
    cnn = nn.Sequential(*cnn_layers)
    cnn_out_size = in_channels * out_h * out_w

    # Scalar FC
    concat_size = cnn_out_size
    scalar_fc = None
    if scalar_size > 0:
        fc_layers = []
        prev = scalar_size
        for h_size in fc_hiddens:
            fc_layers.append(nn.Linear(prev, h_size))
            fc_layers.append(activation_fn())
            prev = h_size
        scalar_fc = nn.Sequential(*fc_layers)
        concat_size += prev

    return cnn, scalar_fc, rgb_shape, rgb_flat_size, scalar_size, concat_size


def _process_flat_obs(flat_obs, cnn, scalar_fc, rgb_shape, rgb_flat_size,
                      scalar_size):
    """Split flat obs -> CNN(rgb) [+ FC(scalars)] -> concatenated features."""
    rgb_flat = flat_obs[:, :rgb_flat_size]
    h, w, c = rgb_shape
    rgb = rgb_flat.reshape(-1, h, w, c).permute(0, 3, 1, 2).float() / 255.0
    cnn_out = cnn(rgb)
    if scalar_fc is not None and scalar_size > 0:
        scalars = flat_obs[:, rgb_flat_size:].float()
        scalar_out = scalar_fc(scalars)
        return torch.cat([cnn_out, scalar_out], dim=-1)
    return cnn_out


# =========================================================================
# CNN-only baseline (no LSTM) — MeltingPotCNNModel
# =========================================================================

class MeltingPotCNNModel(TorchModelV2, nn.Module):
    """Feed-forward CNN + FC baseline for MeltingPot (no recurrence).

    Same vision torso as MeltingPotModel but replaces the LSTM with a
    simple FC layer so it can serve as a non-recurrent baseline.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)

        # Build shared vision torso
        (self.cnn, self.scalar_fc, self.rgb_shape,
         self.rgb_flat_size, self.scalar_size,
         concat_size) = _build_vision_torso(obs_space, model_config)

        # Post-concat FC
        post_fc_hiddens = list(model_config.get("post_fcnet_hiddens", [256]))
        activation_fn = nn.ReLU
        post_layers = []
        prev = concat_size
        for h_size in post_fc_hiddens:
            post_layers.append(nn.Linear(prev, h_size))
            post_layers.append(activation_fn())
            prev = h_size
        self.post_fc = nn.Sequential(*post_layers)

        # Output heads
        self._logits = nn.Linear(prev, num_outputs)
        self._value_branch = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(self._logits.weight)
        nn.init.constant_(self._logits.bias, 0.0)
        nn.init.xavier_uniform_(self._value_branch.weight)
        nn.init.constant_(self._value_branch.bias, 0.0)

        self._cur_value = None

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        flat_obs = (input_dict["obs_flat"].float()
                    if "obs_flat" in input_dict
                    else input_dict[SampleBatch.OBS].float())
        features = _process_flat_obs(
            flat_obs, self.cnn, self.scalar_fc,
            self.rgb_shape, self.rgb_flat_size, self.scalar_size)
        features = self.post_fc(features)
        logits = self._logits(features)
        self._cur_value = self._value_branch(features).squeeze(-1)
        return logits, state  # state pass-through (empty list)

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "Must call forward() first"
        return self._cur_value


# =========================================================================
# LSTM model (existing) — MeltingPotModel
# =========================================================================


class MeltingPotModel(RecurrentNetwork, nn.Module):
    """Custom model for MeltingPot with CNN + FC + LSTM.

    Expects a flat Box observation where the first H*W*3 elements are a
    flattened RGB image, and the remaining elements are scalar features.
    The wrapper (MeltingPotObsWrapper in utils.py) produces this layout.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # ---- Config ----
        self.cell_size = model_config.get("lstm_cell_size", 256)
        self.use_prev_action = model_config.get("lstm_use_prev_action", True)
        self.use_prev_reward = model_config.get("lstm_use_prev_reward", False)

        # ---- Shared vision torso (CNN + scalar FC) ----
        (self.cnn, self.scalar_fc, self.rgb_shape,
         self.rgb_flat_size, self.scalar_size,
         concat_size) = _build_vision_torso(obs_space, model_config)

        # ---- Post-concat FC ----
        post_fc_hiddens = list(model_config.get("post_fcnet_hiddens", [256]))
        activation_fn = nn.ReLU
        post_layers = []
        prev = concat_size
        for h_size in post_fc_hiddens:
            post_layers.append(nn.Linear(prev, h_size))
            post_layers.append(activation_fn())
            prev = h_size
        self.post_fc = nn.Sequential(*post_layers)
        lstm_input_size = prev

        # ---- Prev action/reward ----
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
        else:
            self.action_dim = int(np.prod(action_space.shape))

        if self.use_prev_action:
            lstm_input_size += self.action_dim
        if self.use_prev_reward:
            lstm_input_size += 1

        # ---- LSTM ----
        self.lstm = nn.LSTM(lstm_input_size, self.cell_size, batch_first=True)

        # ---- Output heads ----
        self._logits = nn.Linear(self.cell_size, num_outputs)
        self._value_branch = nn.Linear(self.cell_size, 1)
        nn.init.xavier_uniform_(self._logits.weight)
        nn.init.constant_(self._logits.bias, 0.0)
        nn.init.xavier_uniform_(self._value_branch.weight)
        nn.init.constant_(self._value_branch.bias, 0.0)

        self._cur_value = None

        # ---- View requirements ----
        if self.use_prev_action:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        if self.use_prev_reward:
            self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        """Return initial LSTM (h, c) as torch tensors."""
        return [
            torch.zeros(self.cell_size, dtype=torch.float32),
            torch.zeros(self.cell_size, dtype=torch.float32),
        ]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._cur_value is not None, "Must call forward() first"
        return self._cur_value

    def _process_obs(self, flat_obs: TensorType) -> TensorType:
        """Split flat obs into RGB + scalars, process through CNN + FC, concat."""
        return _process_flat_obs(
            flat_obs, self.cnn, self.scalar_fc,
            self.rgb_shape, self.rgb_flat_size, self.scalar_size)

    @override(RecurrentNetwork)
    def forward_rnn(
        self,
        inputs: TensorType,
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """LSTM forward: inputs (B, T, features) -> logits (B, T, num_outputs)."""
        h_in = state[0].unsqueeze(0)  # (1, B, cell_size)
        c_in = state[1].unsqueeze(0)
        lstm_out, (h_out, c_out) = self.lstm(inputs, (h_in, c_in))
        logits = self._logits(lstm_out)
        self._cur_value = self._value_branch(lstm_out).squeeze(-1)
        return logits, [h_out.squeeze(0), c_out.squeeze(0)]

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Full forward: flat_obs -> CNN/FC -> [prev_act] -> LSTM -> logits."""
        flat_obs = input_dict["obs_flat"].float() if "obs_flat" in input_dict else input_dict[SampleBatch.OBS].float()

        # CNN + FC
        features = self._process_obs(flat_obs)
        features = self.post_fc(features)

        # Prev action
        if self.use_prev_action:
            prev_a = input_dict.get(SampleBatch.PREV_ACTIONS)
            if prev_a is None:
                prev_a = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            if isinstance(self.action_space, gym.spaces.Discrete):
                prev_a_onehot = torch.nn.functional.one_hot(
                    prev_a.long(), self.action_dim
                ).float()
            else:
                prev_a_onehot = prev_a.float()
            features = torch.cat([features, prev_a_onehot], dim=-1)

        # Prev reward
        if self.use_prev_reward:
            prev_r = input_dict.get(SampleBatch.PREV_REWARDS)
            if prev_r is None:
                prev_r = torch.zeros(features.shape[0], 1, device=features.device)
            elif prev_r.dim() == 1:
                prev_r = prev_r.unsqueeze(-1)
            features = torch.cat([features, prev_r.float()], dim=-1)

        # Time dimension + LSTM
        features_td = add_time_dimension(
            features, seq_lens=seq_lens, framework="torch", time_major=False
        )
        output, new_state = self.forward_rnn(features_td, state, seq_lens)

        # Flatten to (B*T, num_outputs)
        output = output.reshape(-1, self.num_outputs)
        self._cur_value = self._cur_value.reshape(-1)

        return output, new_state
