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

"""SAC networks."""

from typing import Sequence, Tuple

from brax.training import distribution

from brax.training.agents.sac2 import utils

from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen


@flax.struct.dataclass
class SACNetworks:
  policy_network: utils.FeedForwardNetwork
  q_network: utils.FeedForwardNetwork
  parametric_action_distribution: utils.ParametricDistribution


def make_inference_fn(sac_networks: SACNetworks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:

    def policy(observations: types.Observation,
               key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      logits = sac_networks.policy_network.apply(*params, observations)
      param_act_dist = sac_networks.parametric_action_distribution
      if deterministic:
        return param_act_dist.mode(logits), {}
      origin_action = param_act_dist.sample_no_postprocessing(logits, key_sample)
      log_prob = param_act_dist.log_prob(logits, origin_action)
      return sac_networks.parametric_action_distribution.postprocess(origin_action), \
      {'origin_action': origin_action, 'log_prob': log_prob}

    return policy

  return make_policy


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: utils.ActivationFn = linen.elu, alpha = 0.2, lock_variance = False) -> SACNetworks:
  """Make SAC networks."""
  parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
  if lock_variance:
    parametric_action_distribution = utils.NormalTanhDistribution(event_size=action_size, alpha=alpha)
  policy_network = utils.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  q_network = utils.make_q_network(
      observation_size,
      action_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  return SACNetworks(
      policy_network=policy_network,
      q_network=q_network,
      parametric_action_distribution=parametric_action_distribution)
