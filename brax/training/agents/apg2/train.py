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

"""Analytic policy gradient training."""

import functools
import time
from typing import Callable, Optional, Tuple, Any

from absl import logging
from brax import envs
from brax.envs import wrappers
from brax.training import acting
from brax.training import pmap
from brax.training import types
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.apg2 import networks as apg_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import numpy as np
import tensorflow as tf

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  policy_params: Params
  value_optimizer_state: optax.OptState
  value_params: Params
  target_value_params: Params
  normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
  return jax.tree_map(lambda x: x[0], v)


def train(environment: envs.Env,
          episode_length: int,
          action_repeat: int = 1,
          num_envs: int = 1,
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 128,
          learning_rate: float = 6e-4,
          seed: int = 0,
          truncation_length: Optional[int] = None,
          max_gradient_norm: float = 1e9,
          num_evals: int = 1,
          discount = 0.99,
          lambda_ = 0.95,
          alpha = 0.95,
          normalize_observations: bool = True,
          deterministic_eval: bool = False,
          tensorboard_flag = True,
          logdir = './logs',
          network_factory: types.NetworkFactory[apg_networks.APGNetworks] = apg_networks.make_apg_networks,
          value_network_factory: types.NetworkFactory[apg_networks.APGNetworks] = apg_networks.make_value_networks,
          progress_fn: Callable[[int, Metrics], None] = lambda *args: None):
  """Direct trajectory optimization training."""
  xt = time.time()

  if tensorboard_flag:
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count

  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  if truncation_length is not None:
    assert truncation_length > 0

  num_evals_after_init = max(num_evals - 1, 1)

  assert num_envs % device_count == 0
  env = environment
  env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  env = wrappers.VmapWrapper(env)
  env = wrappers.AutoResetWrapper(env)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  apg_network = network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=normalize)
  make_policy = apg_networks.make_inference_fn(apg_network)

  optimizer = optax.adam(learning_rate=learning_rate)

#================================================ S T A R T =======================================================#
  value_network = value_network_factory(
      env.observation_size,
      env.action_size,
      preprocess_observations_fn=normalize
  )
  value_optimizer = optax.adam(learning_rate=learning_rate)
#==================================================================================================================#

  def env_step(carry: Tuple[envs.State, PRNGKey], step_index: int,
               policy: types.Policy):
    env_state, key = carry
    key, key_sample = jax.random.split(key)
    actions = policy(env_state.obs, key_sample)[0]
    nstate = env.step(env_state, actions)
    if truncation_length is not None:
      nstate = jax.lax.cond(
          jnp.mod(step_index + 1, truncation_length) == 0.,
          jax.lax.stop_gradient, lambda x: x, nstate)

    return (nstate, key), (nstate.reward, env_state.obs, nstate.obs)

  def loss(policy_params, target_value_params, normalizer_params, key):
    key_reset, key_scan = jax.random.split(key)
    env_state = env.reset(
        jax.random.split(key_reset, num_envs // process_count))
    f = functools.partial(
        env_step, policy=make_policy((normalizer_params, policy_params)))
    (rewards, obs, next_obs) = jax.lax.scan(f, (env_state, key_scan), (jnp.array(range(episode_length // action_repeat))))[1]

    #================================================ S T A R T =======================================================#
    def compute_discount_return(carry, target_t):
      discount, acc = carry
      reward = target_t
      acc = reward + discount * acc
      return (discount, acc), acc
    
    acc = jnp.zeros_like(rewards[0])
    (_, discount_returns) = jax.lax.scan(
      compute_discount_return,
      (discount, acc),
      rewards,
      length = int(rewards.shape[0]),
      reverse=True)
    
    value_apply = value_network.value_network.apply
    target_boostrap = value_apply(normalizer_params, target_value_params, next_obs[-1])
    policy_loss = -jnp.mean(discount_returns[0] + target_boostrap)

    return policy_loss, (rewards, obs, next_obs, policy_loss)
    #==================================================================================================================#

  loss_grad = jax.grad(loss, has_aux=True)

  def clip_by_global_norm(updates):
    g_norm = optax.global_norm(updates)
    trigger = g_norm < max_gradient_norm
    return jax.tree_map(
        lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm), updates)

  #================================================ S T A R T =======================================================#
  def compute_v_loss(v_params: Params, target_v_params: Params, normalizer_params: Any, 
                rewards, obs, next_obs, lambda_: float = lambda_, discount: float = discount):
    value_apply = value_network.value_network.apply
    values = value_apply(normalizer_params, v_params, obs)
    target_values = value_apply(normalizer_params, target_v_params, obs)
    target_bootstrap_value = value_apply(normalizer_params, target_v_params, next_obs[-1])

    # Calculate Value Esitimation MSE Loss
    values_t_plus_1 = jnp.concatenate([target_values[1:], jnp.expand_dims(target_bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * values_t_plus_1 - target_values

    v_acc = jnp.zeros_like(target_bootstrap_value)
    td_error = []

    def compute_td_error(carry, target_t):
        lambda_, acc = carry
        delta = target_t
        acc = delta + discount * lambda_ * acc
        return (lambda_, acc), (acc)
    (_, _), (td_error) = jax.lax.scan(compute_td_error, (lambda_, v_acc), (deltas), length=int(rewards.shape[0]), reverse=True)
    # Add V(x_s) to get v_s.
    vs = jax.lax.stop_gradient(td_error + target_values) 
    v_error = vs - values
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5
    return v_loss

  value_update_fn = gradients.gradient_update_fn(compute_v_loss, value_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
  #==================================================================================================================#

  def training_epoch(training_state: TrainingState, key: PRNGKey):
    key, key_grad = jax.random.split(key)
    grad, (rewards, obs, next_obs, policy_loss) = loss_grad(training_state.policy_params,
                                                  training_state.target_value_params, training_state.normalizer_params, key_grad)
    grad = clip_by_global_norm(grad)
    grad = jax.lax.pmean(grad, axis_name='i')
    params_update, optimizer_state = optimizer.update(grad, training_state.optimizer_state)
    policy_params = optax.apply_updates(training_state.policy_params, params_update)

    #================================================ S T A R T =======================================================#
    v_loss, value_params, value_optimizer_state = value_update_fn(training_state.value_params,
                                                        training_state.target_value_params,
                                                        training_state.normalizer_params, 
                                                        rewards, obs, next_obs,
                                                        optimizer_state = training_state.value_optimizer_state)
    
    target_value_params = jax.tree_map(lambda x, y: x*alpha+y*(1-alpha), training_state.target_value_params, value_params)
    #==================================================================================================================#

    normalizer_params = running_statistics.update(
        training_state.normalizer_params, obs, pmap_axis_name=_PMAP_AXIS_NAME)

    metrics = {
        'pi_grad_norm': optax.global_norm(grad),
        'pi_params_norm': optax.global_norm(policy_params),
    #================================================ S T A R T =======================================================#
        'v_params_norm': optax.global_norm(value_params),
        'target_v_params_norm': optax.global_norm(target_value_params),
        'policy_loss': policy_loss,
        'value_loss': v_loss,
    #==================================================================================================================#
    }
    return TrainingState(
        optimizer_state = optimizer_state,
        normalizer_params = normalizer_params,
        policy_params = policy_params,
        value_optimizer_state = value_optimizer_state,
        value_params = value_params,
        target_value_params = target_value_params), metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  training_walltime = 0

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(training_state: TrainingState,
                                 key: PRNGKey) -> Tuple[TrainingState, Metrics]:
    nonlocal training_walltime
    t = time.time()
    (training_state, metrics) = training_epoch(training_state, key)
    metrics = jax.tree_map(jnp.mean, metrics)
    jax.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (episode_length * num_envs) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, metrics

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, eval_key = jax.random.split(local_key)

  # The network key should be global, so that networks are initialized the same
  # way for different processes.
  policy_params = apg_network.policy_network.init(global_key)
  value_params = value_network.value_network.init(global_key)
  del global_key

  training_state = TrainingState(
      optimizer_state=optimizer.init(policy_params),
      policy_params=policy_params,
      #================================================ S T A R T =======================================================#
      value_optimizer_state=value_optimizer.init(value_params),
      value_params=value_params,
      target_value_params=value_params,
      #==================================================================================================================#
      normalizer_params=running_statistics.init_state(
          specs.Array((env.observation_size,), jnp.float32)))
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  evaluator = acting.Evaluator(
      env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.policy_params)),
        training_metrics={})
    logging.info(metrics)
    progress_fn(0, metrics)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    # optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, training_metrics) = training_epoch_with_timing(training_state, epoch_keys)
    tf.summary.scalar('pi_grad_norm', data=np.array(training_metrics['training/pi_grad_norm']), step=it*episode_length*num_envs)
    tf.summary.scalar('pi_params_norm', data=np.array(training_metrics['training/pi_params_norm']), step=it*episode_length*num_envs)
    #================================================ S T A R T =======================================================#
    tf.summary.scalar('v_params_norm', data=np.array(training_metrics['training/v_params_norm']), step=it*episode_length*num_envs)
    tf.summary.scalar('target_v_params_norm', data=np.array(training_metrics['training/target_v_params_norm']), step=it*episode_length*num_envs)
    tf.summary.scalar('policy_loss', data=np.array(training_metrics['training/policy_loss']), step=it*episode_length*num_envs)
    tf.summary.scalar('value_loss', data=np.array(training_metrics['training/value_loss']), step=it*episode_length*num_envs)
    #==================================================================================================================#

    if process_id == 0:
      # Run evals.
      metrics = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.policy_params)),
          training_metrics)
      logging.info(metrics)
      progress_fn(it + 1, metrics)

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.policy_params))
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
