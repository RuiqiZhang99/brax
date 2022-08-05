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

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from brax.training import types
from brax.training.agents.sac2 import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training import distribution
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(sac_network: sac_networks.SACNetworks, reward_scaling: float,
                discounting: float, action_size: int):
  """Creates the SAC losses."""

  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution

  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params,
                  transitions: Transition,
                  key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    # next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1) # - 0.01 * next_log_prob
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling + transitions.discount * discounting *next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def actor_loss(policy_params: Params, normalizer_params: Any,
                 q_params: Params, transitions: Transition,
                 key: PRNGKey, alpha, beta, lock_variance=False) -> jnp.ndarray:
    
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    # action_raw = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
    indiff_origin_action = transitions.extras['policy_extras']['origin_action']
    if lock_variance:
        dist_mean = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        dist_std = alpha * jnp.ones_like(dist_mean)
    else:
        dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    
    diff_epsilon = (indiff_origin_action - dist_mean) / (dist_std + 0.001)
    epsilon = jax.lax.stop_gradient(diff_epsilon)
    diff_action_raw = dist_mean + jax.lax.stop_gradient(epsilon * dist_std)

    diff_action = parametric_action_distribution.postprocess(diff_action_raw)
    # log_prob = parametric_action_distribution.log_prob(dist_params, diff_action_raw)
    q_action = q_network.apply(normalizer_params, q_params,transitions.observation, diff_action)
    truncation_mask = 1 - transitions.extras['state_extras']['truncation']
    min_q = jnp.min(q_action, axis=-1) * truncation_mask
    
    reward_action_grad = transitions.extras['reward_action_grad']
    partial_reward_mul_action = jnp.sum(reward_action_grad * diff_action, axis=-1)

    actor_loss = partial_reward_mul_action + discounting * min_q
    
    actor_loss = -jnp.mean(actor_loss) + beta * jnp.mean(jnp.absolute(diff_epsilon))
    return actor_loss, {'Q_bootstrap': jnp.mean(min_q),
                        'raw_action_mean': jnp.mean(diff_action_raw),
                        'raw_action_std': jnp.std(diff_action_raw),
                        'sampled_action_mean': jnp.mean(diff_action),
                        'sampled_action_std': jnp.std(diff_action),
                        'reward_grad_mean': jnp.mean(reward_action_grad),
                        'reward_grad_std': jnp.std(reward_action_grad),
                        'max_reward_grad': jnp.max(reward_action_grad),
                        'min_reward_grad': jnp.min(reward_action_grad),
                        'partial_reward_mul_action': jnp.mean(partial_reward_mul_action),
                        'epsilon_mean': jnp.mean(epsilon),
                        'epsilon_std': jnp.std(epsilon),}

  return critic_loss, actor_loss
