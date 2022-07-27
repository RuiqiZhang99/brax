import functools
import time
from typing import Any, Callable, Optional, Tuple, Sequence

from absl import logging
from brax import envs
from brax.envs import wrappers
from brax.io import model
from brax.jumpy import array
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.types import Policy
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.sac2 import losses as sac_losses
from brax.training.agents.sac2 import networks as sac_networks
from brax.training.types import Params, PRNGKey, Transition
import tensorflow as tf
import numpy as np
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey, obs_size: int, local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0., dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.float32))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params)
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])

def train(environment: envs.Env,
          num_timesteps: int = 250_0000,
          episode_length: int = 1000,
          action_repeat: int = 1,
          num_envs: int = 16,
          num_eval_envs: int = 8,
          learning_rate: float = 1e-4,
          discounting: float = 0.9,
          seed: int = 0,
          batch_size: int = 128,
          num_evals: int = 1,
          normalize_observations: bool = True,
          max_devices_per_host: Optional[int] = None,
          reward_scaling: float = 1.,
          tau: float = 0.005,
          min_replay_size: int = 0,
          max_replay_size: Optional[int] = 10_0000,
          grad_updates_per_step: int = 1,
          deterministic_eval: bool = False,
          max_gradient_norm: float = 1e2,
          tensorboard_flag = True,
          logdir = './logs',
          network_factory: types.NetworkFactory[sac_networks.SACNetworks] = sac_networks.make_sac_networks,
          progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
          checkpoint_logdir: Optional[str] = None):

    if tensorboard_flag:
        file_writer = tf.summary.create_file_writer(logdir)
        file_writer.set_as_default()

    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info('local_device_count: %s; total_device_count: %s', local_devices_to_use, device_count)

    if min_replay_size >= num_timesteps:
        raise ValueError('No training will happen because min_replay_size >= num_timesteps')
    if max_replay_size is None:
        max_replay_size = num_timesteps

    env_steps_per_actor_step = action_repeat * num_envs
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(-(num_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step))

    '''
    assert num_envs % device_count == 0
    env = environment
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    env = wrappers.VmapWrapper(env)
    env = wrappers.AutoResetWrapper(env)
    '''
    obs_size = environment.observation_size
    action_size = environment.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn)
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.,
        discount=0.,
        next_observation=dummy_obs,
        extras={
            'state_extras': {'truncation': 0.},
            'policy_extras': {},
            'rew2act_grad': jnp.zeros((action_size,))})

    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count)

    alpha_loss, critic_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size)

    alpha_update = gradients.gradient_update_fn(alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
    critic_update = gradients.gradient_update_fn(critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

    def clip_by_global_norm(updates):
        g_norm = optax.global_norm(updates)
        trigger = g_norm < max_gradient_norm
        return jax.tree_map(lambda t: jnp.where(trigger, t, (t / g_norm) * max_gradient_norm), updates)

# ===================================== Single-Env, Single-Step Updating, Storing and Training ===================================== #
    def sgd_step(carry: Tuple[TrainingState, PRNGKey], transitions: Transition) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state)

        alpha = jnp.exp(training_state.alpha_params)

        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state)
        
        policy_grad, (alpha, log_prob, min_q, actor_loss) = policy_loss_grad(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor)
        policy_grad = clip_by_global_norm(policy_grad)
        policy_grad = jax.lax.pmean(policy_grad, axis_name='i')
        params_update, policy_optimizer_state = policy_optimizer.update(policy_grad, training_state.policy_optimizer_state)
        policy_params = optax.apply_updates(training_state.policy_params, params_update)
        
        new_target_q_params = jax.tree_map(lambda x, y: x * (1 - tau) + y * tau, training_state.target_q_params, q_params)

        metrics = {'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'alpha_loss': alpha_loss,
                    'alpha': jnp.exp(alpha_params),
                    'policy_grad_norm': optax.global_norm(policy_grad),
                    'policy_params_norm': optax.global_norm(policy_params)}

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params)
        return (new_training_state, key), metrics

    def actor_step(env_state: envs.State, actions, 
                 policy_extras=None, extra_fields: Sequence[str] = ()) -> Tuple[envs.State, Transition]:
        nstate = environment.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        reward = nstate.reward.at[0].get()
        return reward, (nstate, Transition(observation=env_state.obs,
                                            action=actions,
                                            reward=nstate.reward,
                                            discount=1 - nstate.done,
                                            next_observation=nstate.obs,
                                            extras={'policy_extras': policy_extras, 'state_extras': state_extras}))
    
    rew2act_grad_fn = jax.grad(actor_step, argnums=1, has_aux=True)

    def get_experience(normalizer_params: running_statistics.RunningStatisticsState,
                        policy_params: Params, env_state: envs.State,
                        buffer_state: ReplayBufferState, key: PRNGKey
                        ) -> Tuple[running_statistics.RunningStatisticsState, envs.State, ReplayBufferState]:
        policy = make_policy((normalizer_params, policy_params))
        actions, policy_extras = policy(env_state.obs, key)
        rew2act_grad, (nstate, transition) = rew2act_grad_fn(env_state, actions, policy_extras, extra_fields=('truncation',))
        transition.extras["rew2act_grad"] = rew2act_grad
        normalizer_params = running_statistics.update(normalizer_params, transition.observation, pmap_axis_name=_PMAP_AXIS_NAME)
        return normalizer_params, env_state, transition

    
# ===================================== Single-Env, Single-Step Updating, Storing and Training ===================================== #
    def training_step(training_state: TrainingState, env_state: envs.State,
                        buffer_state: ReplayBufferState, key: PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(training_state.normalizer_params, training_state.policy_params,
                                                                    env_state, buffer_state, experience_key)
        training_state = training_state.replace(normalizer_params=normalizer_params, env_steps=training_state.env_steps + env_steps_per_actor_step)
        training_state = training_state.replace(normalizer_params=normalizer_params, env_steps=training_state.env_steps + env_steps_per_actor_step)

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_map(lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]), transitions)
        (training_state, _), metrics = jax.lax.scan(sgd_step,(training_state, training_key), transitions)

        metrics['buffer_current_size'] = buffer_state.current_size
        metrics['buffer_current_position'] = buffer_state.current_position
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state: TrainingState, env_state: envs.State,
                              buffer_state: ReplayBufferState, key: PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params, training_state.policy_params,
                env_state, buffer_state, key)
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step)
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def training_epoch(training_state: TrainingState, env_state: envs.State,
                        buffer_state: ReplayBufferState, key: PRNGKey) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:

        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), (),
            length=num_training_steps_per_epoch)
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)