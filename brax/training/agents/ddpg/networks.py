from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks

from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class DDPGNetworks:
  policy_network: networks.FeedForwardNetwork
  q_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ddpg_networks: DDPGNetworks):

  def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = ddpg_networks.policy_network.apply(*params, observations)
      if deterministic:
        return ddpg_networks.parametric_action_distribution.mode(logits), {}
      origin_action = ddpg_networks.parametric_action_distribution.sample_no_postprocessing(logits, key_sample)
      return ddpg_networks.parametric_action_distribution.postprocess(origin_action), {}

    return policy

  return make_policy


def make_ddpg_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.elu) -> DDPGNetworks:

  parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  q_network = networks.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  return DDPGNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution)
