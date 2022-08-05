from datetime import datetime
import functools
import os

from IPython.display import HTML, clear_output

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from brax import envs
from brax import jumpy as jp
from brax.io import html
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ddpg import train as ddpg
from brax.training.agents.sac import train as sac
from brax.training.agents.apg2 import train as apg2
from brax.training.agents.sac2 import train as offpolicy
from brax.training.agents.sac2 import td_train as offpolicy_v2
import warnings
warnings.filterwarnings("ignore")

env_name = "walker2d"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']
env = envs.get_environment(env_name=env_name)
state = env.reset(rng=jp.random_prngkey(seed=0))

train_fn = functools.partial(ddpg.train, num_timesteps = 20_0000, episode_length=1000, num_evals=50, num_envs=4, logdir='./logs')

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  clear_output(wait=True)
  plt.xlabel('environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.show()

train_fn(environment=env)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')