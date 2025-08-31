# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Short-Horizon Actor Critic.
See: https://arxiv.org/pdf/2204.07137.pdf
"""

from typing import Any, Tuple

from brax.training import types
from shac import networks as shac_networks
from brax.training.types import Params
import flax
import jax
import jax.numpy as jnp

@flax.struct.dataclass
class SHACNetworkParams:
  """Contains training state for the learner."""
  policy: Params
  value: Params


def compute_shac_policy_loss(
    policy_params: Params,
    value_params: Params,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    shac_network: shac_networks.SHACNetworks,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes SHAC critic loss.
  This implements Eq. 5 of 2204.07137.
  Args:
    policy_params: Policy network parameters
    value_params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    shac_network: SHAC networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.
  Returns:
    A scalar loss
  """

  parametric_action_distribution = shac_network.parametric_action_distribution
  policy_apply = shac_network.policy_network.apply
  value_apply = shac_network.value_network.apply

  # Put the time dimension first.
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

  # this is a redundant computation with the critic loss function
  # but there isn't a straighforward way to get these values when
  # they are used in that step
  values = value_apply(normalizer_params, value_params, data.observation)
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  terminal_values = value_apply(normalizer_params, value_params, terminal_obs)

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  # Append terminal values to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(terminal_values, 0)], axis=0)

  # jax implementation of https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/algorithms/shac.py#L227
  def sum_step(carry, target_t):
    gam, rew_acc = carry
    reward, termination = target_t

    # clean up gamma and rew_acc for done envs, otherwise update
    rew_acc = jnp.where(termination, 0, rew_acc + gam * reward)
    gam = jnp.where(termination, 1.0, gam * discounting)

    return (gam, rew_acc), (gam, rew_acc)

  rew_acc = jnp.zeros_like(terminal_values)
  gam = jnp.ones_like(terminal_values)
  (gam, last_rew_acc), (gam_acc, rew_acc) = jax.lax.scan(
    sum_step,
    (gam, rew_acc),
    (rewards, termination)
  )

  policy_loss = jnp.sum(-last_rew_acc - gam * terminal_values)
  # for trials that are truncated (i.e. hit the episode length) include reward for
  # terminal state. otherwise, the trial was aborted and should receive zero additional
  # policy_loss = policy_loss + jnp.sum(
  #   (-rew_acc - gam_acc * jnp.where(truncation, values_t_plus_1, 0)) * termination)
  policy_loss = policy_loss + jnp.sum(
    (-rew_acc - gam_acc * values_t_plus_1) * truncation
  )
  policy_loss = policy_loss / values.shape[0] / values.shape[1]

  # Entropy reward
  policy_logits = policy_apply(normalizer_params, policy_params,
                               data.observation)
  entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
  entropy_loss = entropy_cost * -entropy

  total_loss = policy_loss + entropy_loss

  return total_loss, {
    'policy_loss': policy_loss,
    'entropy_loss': entropy_loss
  }


def compute_shac_critic_loss(
    params: Params,
    normalizer_params: Any,
    target_value_params: Params,
    data: types.Transition,
    shac_network: shac_networks.SHACNetworks,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    td_lambda: bool = True
) -> Tuple[jnp.ndarray, types.Metrics]:
  """Computes SHAC critic loss.
  This implements Eq. 7 of 2204.07137
  https://github.com/NVlabs/DiffRL/blob/main/algorithms/shac.py#L349
  Args:
    params: Value network parameters,
    normalizer_params: Parameters of the normalizer.
    target_value_params: Target value network parameters.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
        ['policy_extras']['log_prob']
    rng: Random key
    shac_network: SHAC networks.
    entropy_cost: entropy cost.
    discounting: discounting,
    reward_scaling: reward multiplier.
    lambda_: Lambda for TD value updates
    td_lambda: whether to use a TD-Lambda value target
  Returns:
    A tuple (loss, metrics)
  """

  value_apply = shac_network.value_network.apply

  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  T, B = data.reward.shape[:2]

  v_pred = value_apply(normalizer_params, params, data.observation)
  v_tp1_all = value_apply(normalizer_params, target_value_params, data.next_observation)
  # Rewards & masks
  rew = data.reward * reward_scaling
  trunc = data.extras['state_extras']['truncation'].astype(bool)   # time-limit
  # termination: true end of episode (not time-limit). Requires discount∈{0,1}.
  term = ((1.0 - data.discount) * (1.0 - trunc.astype(v_pred.dtype))).astype(bool)

  # Mask bootstrap at true terminations; allow at truncations and normal steps
  v_tp1_masked = jnp.where(term, 0.0, v_tp1_all)  # [T, B]

  if td_lambda:
    # Standard TD(λ) backward recursion:
    #   G_t = r_t + γ * [ (1-λ) V_{t+1} + λ G_{t+1} ], with G_t = r_t if termination.
    def step(g_tp1, x):
      r_t, v_tp1_t, term_t = x
      # When term_t, the bootstrap term is zero
      mix = (1.0 - lambda_) * v_tp1_t + lambda_ * g_tp1
      g_t = r_t + discounting * jnp.where(term_t, 0.0, mix)
      return g_t, g_t

    g_T = jnp.zeros((B,), v_pred.dtype)
    _, g_seq = jax.lax.scan(
        step,
        g_T,
        (rew, v_tp1_masked, term),
        reverse=True)
    target_values = jax.lax.stop_gradient(g_seq)          # [T, B]
  else:
    # One-step TD: V^target_t = r_t + γ * V_{t+1}; zero out on true terminations
    target_values = rew + discounting * v_tp1_masked
    target_values = jax.lax.stop_gradient(target_values)

  v_loss = jnp.mean((target_values - v_pred) ** 2)

  return v_loss, {
      'total_loss': v_loss,
      'policy_loss': jnp.array(0.0),
      'v_loss': v_loss,
      'entropy_loss': jnp.array(0.0),
  }
