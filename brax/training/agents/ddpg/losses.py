from typing import Any

from brax.training import types
from brax.training.agents.ddpg import networks as ddpg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training import distribution
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(ddpg_network: ddpg_networks.DDPGNetworks, reward_scaling: float,
                discounting: float):

  policy_network = ddpg_network.policy_network
  q_network = ddpg_network.q_network
  parametric_action_distribution = ddpg_network.parametric_action_distribution

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
                 key: PRNGKey) -> jnp.ndarray:
    
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    # action_raw = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
    dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    diff_action_raw = parametric_action_distribution.sample_no_postprocessing(dist_params, key)

    diff_action = parametric_action_distribution.postprocess(diff_action_raw)
    # log_prob = parametric_action_distribution.log_prob(dist_params, diff_action_raw)
    q_action = q_network.apply(normalizer_params, q_params,transitions.observation, diff_action)
    truncation_mask = 1 - transitions.extras['state_extras']['truncation']
    min_q = jnp.min(q_action, axis=-1) * truncation_mask

    actor_loss = -jnp.mean(min_q)
    return actor_loss, {
                        'raw_action_mean': jnp.mean(diff_action_raw),
                        'raw_action_std': jnp.std(diff_action_raw),
                        'sampled_action_mean': jnp.mean(diff_action),
                        'sampled_action_std': jnp.std(diff_action),
                        }

  return critic_loss, actor_loss
