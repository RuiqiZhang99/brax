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

"""
Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any
from brax.jumpy import ones_like

from brax.training import types
from brax.training.agents.sac2 import networks2 as sac_networks
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
                 transitions: Transition, key: PRNGKey) -> jnp.ndarray:

    # dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    # real_action = transitions.action

    indiff_action = transitions.extras['policy_extras']['non_tanh_action']
    # dist_params = policy_network.apply(normalizer_params, policy_params, trans.observation)
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)

    # Note: Here, dist_params_std = jnp.ones_like(dist_params_mean)
    dist_params_mean = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    epsilon = jax.lax.stop_gradient(indiff_action - dist_params_mean)

    diff_action = dist_params_mean + jnp.ones_like(dist_params_mean) * epsilon
    diff_action = parametric_action_distribution.postprocess(diff_action)
    
    rew2act_grads = transitions.extras['reward_grads']
    empty_acc = jnp.zeros_like(transitions.action[0])

    def compute_disc_action(carry, target_t):
      discount, reward_scaling, acc = carry
      action, rew2act_grad, trans_discount, truncation_mask, transition = target_t
      # print(transition.observation, transition.next_observation)
      acc = action + rew2act_grad * discount * trans_discount * truncation_mask * acc
      return (discount, reward_scaling, acc), (acc)

    (_, _, _), (disc_actions) = jax.lax.scan(
      compute_disc_action,
      (discounting, reward_scaling, empty_acc),
      (diff_action, rew2act_grads, 
      transitions.discount, 1-transitions.extras['state_extras']['truncation'],
      transitions),
      length = int(transitions.action.shape[0]),
      reverse=True)

    
    disc_action_sum = jnp.sum(disc_actions[0])
    end_next_obs = transitions.next_observation[-1].reshape(1, -1)
    next_dist_params = policy_network.apply(normalizer_params, policy_params, end_next_obs)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    next_action = parametric_action_distribution.postprocess(next_action)
    real_next_q = q_network.apply(normalizer_params, target_q_params, end_next_obs, next_action)
    next_q = discounting ** transitions.reward.shape[0] * jnp.min(real_next_q, axis=-1)
    next_q = jnp.mean(next_q)

    actor_loss = - (next_q + disc_action_sum)
    # reward_term = rew2act_grads * diff_action * reward_scaling
    # truncation = jnp.expand_dims(1 - transitions.extras['state_extras']['truncation'], axis=-1)
    correction = jnp.mean(transitions.observation[1:] - transitions.next_observation[:-1])
    
    
    return actor_loss, {'reward_term': disc_action_sum, 
                        'Q_bootstrap_pi': real_next_q,
                        'epsilon_avg': jnp.mean(epsilon),
                        'epsilon_norm': jnp.std(epsilon),
                        'trans_rank_correct': correction}


  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params, # alpha: jnp.ndarray,  
                  transitions: Transition, key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling + transitions.discount * discounting * next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss, {"Q_old_action": jnp.mean(q_old_action),
                    "Q_bootstrap_q": jnp.mean(next_v)}

  return actor_loss, critic_loss




