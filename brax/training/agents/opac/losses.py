from typing import Any

from brax.training import types
from brax.training.agents.opac import networks as sac_networks
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

    indiff_action = transitions.extras['policy_extras']['non_tanh_action']
    # dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    # dist_mean, dist_std = jnp.split(dist_params, 2, axis=-1)
    # nor_tanh_std = jax.nn.softplus(dist_std) + min_std
    dist_mean = policy_network.apply(normalizer_params, policy_params, transitions.observation)
    dist_std = jnp.ones_like(dist_mean)
    epsilon = jax.lax.stop_gradient(indiff_action - dist_mean)

    # diff_action_raw = dist_mean + nor_tanh_std * epsilon
    diff_action_raw = dist_mean + dist_std * epsilon
    diff_action = parametric_action_distribution.postprocess(diff_action_raw)
  
    rew2act_grads = transitions.extras['reward_grads']
    
    next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1)
    reward_term = rew2act_grads * diff_action * reward_scaling
    # truncation = jnp.expand_dims(1 - transitions.extras['state_extras']['truncation'], axis=-1)
    
    actor_loss = (jnp.sum(reward_term, axis=-1) + transitions.discount * discounting * next_v)
    actor_loss = -jnp.mean(actor_loss)
    return actor_loss, {'reward_term': jnp.mean(jnp.sum(reward_term, axis=-1)), 
                        'Q_bootstrap_pi': jnp.mean(next_v),
                        'epsilon_avg': jnp.mean(epsilon),
                        'epsilon_norm': jnp.std(epsilon),
                        'raw_action_avg': jnp.mean(diff_action_raw),
                        'raw_action_norm': jnp.std(diff_action_raw),
                        'action_transfer_error': jnp.max(indiff_action-diff_action_raw)}


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