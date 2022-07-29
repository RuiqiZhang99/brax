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
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey, Transition
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(sac_network: sac_networks.SACNetworks, reward_scaling: float,
                discounting: float, action_size: int):
  """Creates the SAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution

  def actor_loss(policy_params: Params,
                 target_q_params: Params, normalizer_params: Any, 
                 transitions: Transition, key: PRNGKey, min_std=0.001) -> jnp.ndarray:

    # dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    # real_action = transitions.action

    def differentialize_action(trans: Transition):
      dist_params = policy_network.apply(normalizer_params, policy_params, trans.observation)
      dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
      indiff_action = trans.action
      nor_tanh_std = jax.nn.softplus(dist_std) + min_std
      epsilon = jax.lax.stop_gradient((indiff_action - dist_mean) / (nor_tanh_std))
      diff_action = dist_mean + nor_tanh_std * epsilon
      return diff_action
  
    diff_action = differentialize_action(transitions)
    rew2act_grads = transitions.extras['reward_grads']
    
    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1)
    reward_item = rew2act_grads * diff_action * reward_scaling
    # truncation = jnp.expand_dims(1 - transitions.extras['state_extras']['truncation'], axis=-1)
    
    actor_loss = (reward_item + jnp.expand_dims(transitions.discount * discounting * next_v, -1))
    actor_loss = -jnp.mean(actor_loss)
    return actor_loss


  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params, # alpha: jnp.ndarray,  
                  transitions: Transition, key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    # next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
    # next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling + transitions.discount * discounting * next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  return actor_loss, critic_loss




