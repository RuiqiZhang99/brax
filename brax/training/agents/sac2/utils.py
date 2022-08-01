import abc
import jax
import jax.numpy as jnp
import numpy as np
import abc
from typing import Generic, Tuple, TypeVar
import random
from brax.training.types import PRNGKey
import flax
from jax import flatten_util




State = TypeVar('State')
Sample = TypeVar('Sample')


class ReplayBuffer(abc.ABC, Generic[State, Sample]):
  """Contains replay buffer methods."""

  @abc.abstractmethod
  def init(self, key: PRNGKey) -> State:
    """Init the replay buffer."""

  @abc.abstractmethod
  def insert(self, buffer_state: State, samples: Sample) -> State:
    """Insert data in the replay buffer."""

  @abc.abstractmethod
  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data."""

  @abc.abstractmethod
  def size(self, buffer_state: State) -> int:
    """Total amount of elements that are sampleable."""


@flax.struct.dataclass
class _ReplayBufferState:
  """Contains data related to a replay buffer."""
  data: jnp.ndarray
  current_position: jnp.ndarray
  current_size: jnp.ndarray
  key: PRNGKey


class UniformSamplingQueue(ReplayBuffer, Generic[Sample]):
  """Replay buffer with uniform sampling.

  * It behaves as a limited size queue (if buffer is full it removes the oldest
    elements when new one is inserted).
  * It supports batch insertion only (no single element)
  * It performs uniform random sampling with replacement of a batch of size
    `sample_batch_size`
  """

  def __init__(self, max_replay_size: int, dummy_data_sample: Sample,
               sample_batch_size: int, num_envs: int):
    self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])

    dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
    self._unflatten_fn = jax.vmap(self._unflatten_fn)
    data_size = len(dummy_flatten)

    self._data_shape = (max_replay_size, data_size)
    self._data_dtype = dummy_flatten.dtype
    self._sample_batch_size = sample_batch_size
    self._num_envs = num_envs

  def init(self, key: PRNGKey) -> _ReplayBufferState:
    return _ReplayBufferState(
        data=jnp.zeros(self._data_shape, self._data_dtype),
        current_size=jnp.zeros((), jnp.int32),
        current_position=jnp.zeros((), jnp.int32),
        key=key)

  def insert(self, buffer_state: _ReplayBufferState,
             samples: Sample) -> _ReplayBufferState:
    """Insert data in the replay buffer.

    Args:
      buffer_state: Buffer state
      samples: Sample to insert with a leading batch size.

    Returns:
      New buffer state.
    """
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'buffer_state.data.shape ({buffer_state.data.shape}) '
          f'doesn\'t match the expected value ({self._data_shape})')

    update = self._flatten_fn(samples)
    data = buffer_state.data

    # Make sure update is not larger than the maximum replay size.
    if len(update) > len(data):
      raise ValueError(
          'Trying to insert a batch of samples larger than the maximum replay '
          f'size. num_samples: {len(update)}, max replay size {len(data)}')

    # If needed, roll the buffer to make sure there's enough space to fit
    # `update` after the current position.
    position = buffer_state.current_position
    roll = jnp.minimum(0, len(data) - position - len(update))
    data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0),
                        lambda: data)
    position = position + roll

    # Update the buffer and the control numbers.
    data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
    position = (position + len(update)) % len(data)
    size = jnp.minimum(buffer_state.current_size + len(update), len(data))

    return _ReplayBufferState(
        data=data,
        current_position=position,
        current_size=size,
        key=buffer_state.key)

  def sample(
      self, buffer_state: _ReplayBufferState
  ) -> Tuple[_ReplayBufferState, Sample]:
    """Sample a batch of data.

    Args:
      buffer_state: Buffer state

    Returns:
      New buffer state and a batch with leading dimension 'sample_batch_size'.
    """
    assert buffer_state.data.shape == self._data_shape
    # assert buffer_state.current_size.at[0].get() - self._num_envs * self._sample_batch_size > 0
    key, sample_key = jax.random.split(buffer_state.key)
    
    start_idx = jax.random.randint(
                sample_key, (1,), 
                minval=0,
                maxval=buffer_state.current_size-(self._num_envs*self._sample_batch_size))
    
    def generate_index(carry, counter):
        seed, num_envs = carry
        idx = seed + num_envs * counter
        return (start_idx, num_envs), (idx)
    
    (_, _), (index) = jax.lax.scan(
        generate_index,
        (start_idx, self._num_envs),
        (jnp.arange(1, self._sample_batch_size+1, step=1, dtype=np.uint)))
    index = jnp.squeeze(index)
    # idx = jnp.arange(self._seed, (self._seed+self._num_envs*self._sample_batch_size), step=self._num_envs, dtype=np.uint)
    batch = jnp.take(buffer_state.data, index, axis=0, mode='clip')
    return buffer_state.replace(key=key), self._unflatten_fn(batch)

  def size(self, buffer_state: _ReplayBufferState) -> int:
    return buffer_state.current_size
    
'''
class ParametricDistribution(abc.ABC):
  """Abstract class for parametric (action) distribution."""

  def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
    """Abstract class for parametric (action) distribution.

    Specifies how to transform distribution parameters (i.e. actor output)
    into a distribution over actions.

    Args:
      param_size: size of the parameters for the distribution
      postprocessor: bijector which is applied after sampling (in practice, it's
        tanh or identity)
      event_ndims: rank of the distribution sample (i.e. action)
      reparametrizable: is the distribution reparametrizable
    """
    self._param_size = param_size
    self._postprocessor = postprocessor
    self._event_ndims = event_ndims  # rank of events
    self._reparametrizable = reparametrizable
    assert event_ndims in [0, 1]

  @abc.abstractmethod
  def create_dist(self, parameters):
    """Creates distribution from parameters."""
    pass

  @property
  def param_size(self):
    return self._param_size

  @property
  def reparametrizable(self):
    return self._reparametrizable

  def postprocess(self, event):
    return self._postprocessor.forward(event)

  def inverse_postprocess(self, event):
    return self._postprocessor.inverse(event)

  def sample_no_postprocessing(self, parameters, seed):
    return self.create_dist(parameters).sample(seed=seed)

  def sample(self, parameters, seed):
    """Returns a sample from the postprocessed distribution."""
    return self.postprocess(self.sample_no_postprocessing(parameters, seed))

  def mode(self, parameters):
    """Returns the mode of the postprocessed distribution."""
    return self.postprocess(self.create_dist(parameters).mode())

  def log_prob(self, parameters, actions):
    """Compute the log probability of actions."""
    dist = self.create_dist(parameters)
    log_probs = dist.log_prob(actions)
    log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
    if self._event_ndims == 1:
      log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
    return log_probs

  def entropy(self, parameters, seed):
    """Return the entropy of the given distribution."""
    dist = self.create_dist(parameters)
    entropy = dist.entropy()
    entropy += self._postprocessor.forward_log_det_jacobian(
        dist.sample(seed=seed))
    if self._event_ndims == 1:
      entropy = jnp.sum(entropy, axis=-1)
    return entropy


class NormalDistribution:
  """Normal distribution."""

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

  def sample(self, seed):
    return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

  def mode(self):
    return self.loc

  def log_prob(self, x):
    log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
    log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
    return log_unnormalized - log_normalization

  def entropy(self):
    log_normalization = 0.5 * jnp.log(2. * jnp.pi) + jnp.log(self.scale)
    entropy = 0.5 + log_normalization
    return entropy * jnp.ones_like(self.loc)


class TanhBijector:
  """Tanh Bijector."""

  def forward(self, x):
    return jnp.tanh(x)

  def inverse(self, y):
    return jnp.arctanh(y)

  def forward_log_det_jacobian(self, x):
    return 2. * (jnp.log(2.) - x - jax.nn.softplus(-2. * x))


class NormalTanhDistribution(ParametricDistribution):
  """Normal distribution followed by tanh."""

  def __init__(self, event_size, min_std=0.001):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
    """
    # We apply tanh to gaussian actions to bound them.
    # Normally we would use TransformedDistribution to automatically
    # apply tanh to the distribution.
    # We can't do it here because of tanh saturation
    # which would make log_prob computations impossible. Instead, most
    # of the code operate on pre-tanh actions and we take the postprocessor
    # jacobian into account in log_prob computations.
    super().__init__(
        param_size=event_size,
        postprocessor=TanhBijector(),
        event_ndims=1,
        reparametrizable=True)
    self._min_std = min_std

  def create_dist(self, parameters):
    loc = parameters
    scale = jnp.ones_like(loc)
    # scale = jax.nn.softplus(scale) + self._min_std
    return NormalDistribution(loc=loc, scale=scale)
'''