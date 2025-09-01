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


# def compute_shac_policy_loss(
#     policy_params: Params,
#     value_params: Params,
#     normalizer_params: Any,
#     data: types.Transition,
#     rng: jnp.ndarray,
#     shac_network: shac_networks.SHACNetworks,
#     entropy_cost: float = 1e-4,
#     discounting: float = 0.9,
#     reward_scaling: float = 1.0) -> Tuple[jnp.ndarray, types.Metrics]:
#   """Computes SHAC critic loss.
#   This implements Eq. 5 of 2204.07137.
#   Args:
#     policy_params: Policy network parameters
#     value_params: Value network parameters,
#     normalizer_params: Parameters of the normalizer.
#     data: Transition that with leading dimension [B, T]. extra fields required
#       are ['state_extras']['truncation'] ['policy_extras']['raw_action']
#         ['policy_extras']['log_prob']
#     rng: Random key
#     shac_network: SHAC networks.
#     entropy_cost: entropy cost.
#     discounting: discounting,
#     reward_scaling: reward multiplier.
#   Returns:
#     A scalar loss
#   """

#   parametric_action_distribution = shac_network.parametric_action_distribution
#   policy_apply = shac_network.policy_network.apply
#   value_apply = shac_network.value_network.apply

#   # Put the time dimension first.
#   data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

#   # this is a redundant computation with the critic loss function
#   # but there isn't a straighforward way to get these values when
#   # they are used in that step
#   values = value_apply(normalizer_params, value_params, data.observation)
#   terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
#   terminal_values = value_apply(normalizer_params, value_params, terminal_obs)

#   rewards = data.reward * reward_scaling
#   truncation = data.extras['state_extras']['truncation']
#   termination = (1 - data.discount) * (1 - truncation)

#   # Append terminal values to get [v1, ..., v_t+1]
#   values_t_plus_1 = jnp.concatenate(
#       [values[1:], jnp.expand_dims(terminal_values, 0)], axis=0)

#   # jax implementation of https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/algorithms/shac.py#L227
#   def sum_step(carry, target_t):
#     gam, rew_acc = carry
#     reward, termination = target_t

#     # clean up gamma and rew_acc for done envs, otherwise update
#     rew_acc = jnp.where(termination, 0, rew_acc + gam * reward)
#     gam = jnp.where(termination, 1.0, gam * discounting)

#     return (gam, rew_acc), (gam, rew_acc)

#   rew_acc = jnp.zeros_like(terminal_values)
#   gam = jnp.ones_like(terminal_values)
#   (gam, last_rew_acc), (gam_acc, rew_acc) = jax.lax.scan(
#     sum_step,
#     (gam, rew_acc),
#     (rewards, termination)
#   )

#   policy_loss = jnp.sum(-last_rew_acc - gam * terminal_values)
#   # for trials that are truncated (i.e. hit the episode length) include reward for
#   # terminal state. otherwise, the trial was aborted and should receive zero additional
#   # policy_loss = policy_loss + jnp.sum(
#   #   (-rew_acc - gam_acc * jnp.where(truncation, values_t_plus_1, 0)) * termination)
#   policy_loss = policy_loss + jnp.sum(
#     (-rew_acc - gam_acc * values_t_plus_1) * truncation
#   )
#   policy_loss = policy_loss / values.shape[0] / values.shape[1]

#   # Entropy reward
#   policy_logits = policy_apply(normalizer_params, policy_params,
#                                data.observation)
#   entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
#   entropy_loss = entropy_cost * -entropy

#   total_loss = policy_loss + entropy_loss

#   return total_loss, {
#     'policy_loss': policy_loss,
#     'entropy_loss': entropy_loss
#   }

# def compute_shac_policy_loss(
#   policy_params: Params,
#   value_params: Params,          # <-- this should be TARGET value params (you already pass target in train.py)
#   normalizer_params: Any,
#   data: types.Transition,
#   rng: jnp.ndarray,
#   shac_network: shac_networks.SHACNetworks,
#   entropy_cost: float = 1e-4,
#   discounting: float = 0.9,
#   reward_scaling: float = 1.0
# ) -> Tuple[jnp.ndarray, types.Metrics]:
#   """SHAC actor loss over a short horizon H.

#   J = E[ sum_{t=0}^{H-1} gamma^t r_t  + gamma^H * 1{no true term in [0,H-1]} * V_target(s_H) ]
#   """

#   parametric_action_distribution = shac_network.parametric_action_distribution
#   policy_apply = shac_network.policy_network.apply
#   value_apply  = shac_network.value_network.apply

#   # [B, T] -> [T, B]
#   data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
#   rewards = data.reward * reward_scaling                      # [T, B]
#   trunc = data.extras['state_extras']['truncation'].astype(jnp.bool_)  # [T, B]
#   # true term = discount==0 AND not a time-limit truncation
#   term = ((1.0 - data.discount) * (1.0 - trunc.astype(rewards.dtype))).astype(jnp.bool_)  # [T, B]
#   T, B = rewards.shape[:2]
#   # alive prefix mask: 1 until the first true termination, else 0.
#   # alive[0] = 1
#   # alive[t] = prod_{k=0..t-1} (1 - term[k])
#   # Implement exclusive cumprod:
#   one = jnp.ones((1, B), dtype=rewards.dtype)
#   alive = jnp.cumprod(1.0 - term.astype(rewards.dtype), axis=0)         # [T, B], inclusive
#   alive = jnp.concatenate([one, alive[:-1]], axis=0)                    # make it exclusive

#   # discounted factors gamma^t
#   gammas = (discounting ** jnp.arange(T, dtype=rewards.dtype))[:, None] # [T, 1]

#   # window return (stop adding after true termination)
#   window_return = jnp.sum(gammas * rewards * alive, axis=0)             # [B]

#   # Bootstrap only if no true termination happened anywhere in the window
#   no_term_in_window = jnp.prod(1.0 - term.astype(rewards.dtype), axis=0)  # [B]

#   # V_target(s_H) using "next_observation" at the last step
#   s_H = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)    # [B, ...]
#   v_H = value_apply(normalizer_params, value_params, s_H)                 # [B]
#   v_H = jax.lax.stop_gradient(v_H)

#   actor_obj = window_return + (discounting ** T) * no_term_in_window * v_H  # [B]
#   policy_loss = -jnp.mean(actor_obj) / T

#   # Entropy bonus
#   logits = policy_apply(normalizer_params, policy_params, data.observation)  # [T,B,...]
#   entropy = jnp.mean(parametric_action_distribution.entropy(logits, rng))
#   entropy_loss =  -entropy_cost * entropy
#   total_loss = policy_loss + entropy_loss

#   return total_loss, {
#     'policy_loss': policy_loss,
#     'entropy_loss': entropy_loss,
#   }

def compute_shac_policy_loss(
    policy_params, target_value_params, normalizer_params, data, rng,
    shac_network, entropy_cost=1e-4, discounting=0.99, reward_scaling=1.0):

  param_dist = shac_network.parametric_action_distribution
  policy_apply = shac_network.policy_network.apply
  value_apply  = shac_network.value_network.apply

  # [B,T] -> [T,B]
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  rew   = data.reward * reward_scaling                             # [T,B]
  trunc = data.extras['state_extras']['truncation'].astype(jnp.bool_)  # [T,B]
  term  = ((1.0 - data.discount) * (1.0 - trunc.astype(rew.dtype))).astype(jnp.bool_)  # [T,B]
  reset_event = jnp.logical_or(trunc, term)                        # time-limit or true terminal
  T, B = rew.shape[:2]

  # Sum discounted rewards, but reset γ to 1 right after any reset_event.
  def scan_sum(carry, x):
    gam, ret = carry                      # [B], [B]
    r_t, reset_t = x                      # [B], [B]
    ret = ret + gam * r_t
    gam = jnp.where(reset_t, 1.0, gam * discounting)
    return (gam, ret), None

  init = (jnp.ones((B,), rew.dtype), jnp.zeros((B,), rew.dtype))
  (_, window_return), _ = jax.lax.scan(scan_sum, init, (rew, reset_event))
  window_return = window_return  # [B]

  # Only suppress bootstrap if a true termination happened inside the window.
  no_true_term = jnp.prod(1.0 - term.astype(rew.dtype), axis=0)     # [B]

  s_H = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  v_H = value_apply(normalizer_params, target_value_params, s_H)     # keep ∂V/∂s_H

  actor_obj = window_return + (discounting ** T) * no_true_term * v_H
  policy_loss = -jnp.mean(actor_obj) / T

  logits = policy_apply(normalizer_params, policy_params, data.observation)
  entropy = jnp.mean(param_dist.entropy(logits, rng))
  total_loss = policy_loss - entropy_cost * entropy

  return total_loss, {'policy_loss': policy_loss, 'entropy_loss': -entropy_cost * entropy}

def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
  """Calculates the Generalized Advantage Estimation (GAE).

  Args:
    truncation: A float32 tensor of shape [T, B] with truncation signal.
    termination: A float32 tensor of shape [T, B] with termination signal.
    rewards: A float32 tensor of shape [T, B] containing rewards generated by
      following the behaviour policy.
    values: A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value: A float32 of shape [B] with the value function estimate at
      time T.
    lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
      lambda_=1.
    discount: TD discount.

  Returns:
    A float32 tensor of shape [T, B]. Can be used as target to
      train a baseline (V(x_t) - vs_t)^2.
    A float32 tensor of shape [T, B] of advantages.
  """

  truncation_mask = 1 - truncation
  # Append bootstrapped value to get [v1, ..., v_t+1]
  values_t_plus_1 = jnp.concatenate(
      [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
  )
  deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
  deltas *= truncation_mask

  acc = jnp.zeros_like(bootstrap_value)

  def compute_vs_minus_v_xs(carry, target_t):
    lambda_, acc = carry
    truncation_mask, delta, termination = target_t
    acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
    return (lambda_, acc), (acc)

  (_, _), (vs_minus_v_xs) = jax.lax.scan(
      compute_vs_minus_v_xs,
      (lambda_, acc),
      (truncation_mask, deltas, termination),
      length=int(truncation_mask.shape[0]),
      reverse=True,
  )
  # Add V(x_s) to get v_s.
  vs = jnp.add(vs_minus_v_xs, values)

  # vs_t_plus_1 = jnp.concatenate(
  #     [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
  # )
  # advantages = (
  #     rewards + discount * (1 - termination) * vs_t_plus_1 - values
  # ) * truncation_mask
  return jax.lax.stop_gradient(vs) #, jax.lax.stop_gradient(advantages)


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
  # Rewards & masks
  rew = data.reward * reward_scaling
  trunc = data.extras['state_extras']['truncation'].astype(bool)   # time-limit
  # termination: true end of episode (not time-limit). Requires discount∈{0,1}.
  term = ((1.0 - data.discount) * (1.0 - trunc.astype(v_pred.dtype))).astype(bool)

  values_target = value_apply(
      normalizer_params, target_value_params, data.observation
  )  # [T,B]
  bootstrap_value = value_apply(
      normalizer_params, target_value_params,
      jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  )  # [B]

  target_values = compute_gae(
      truncation=trunc,
      termination=term,
      rewards=rew,
      values=values_target,
      bootstrap_value=bootstrap_value,
      lambda_=lambda_,
      discount=discounting,
  )  # [T,B]; stop_gradient already applied

  v_loss = jnp.mean((target_values - v_pred) ** 2)

  # numerically-stable variance
  def var(x):
    x = x - jnp.mean(x)
    return jnp.mean(x * x)

  tgt_var = var(target_values)
  resid_var = var(target_values - v_pred)
  ev = jnp.where(tgt_var > 0, 1.0 - resid_var / tgt_var, 0.0)

  return v_loss, {
      'total_loss': v_loss,
      'policy_loss': jnp.array(0.0),
      'v_loss': v_loss,
      'entropy_loss': jnp.array(0.0),

      # scale/fit diagnostics
      'critic/ev': ev,
      'critic/bootstrap_value_mean': jnp.mean(bootstrap_value),
      'critic/bootstrap_value_std': jnp.sqrt(var(bootstrap_value) + 1e-8),
      'critic/target_mean': jnp.mean(target_values),
      'critic/target_std': jnp.sqrt(tgt_var + 1e-8),
      'critic/v_mean': jnp.mean(v_pred),
      'critic/v_std': jnp.sqrt(var(v_pred) + 1e-8),
      # how often we actually bootstrap (not a true termination)
      'critic/bootstrap_frac': jnp.mean((~term).astype(jnp.float32)),
  }
