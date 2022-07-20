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
from brax.training.agents.sac import train as sac
from brax.training.agents.apg2 import train as apg2

env_name = "walker2d"  # @param ['ant', 'fetch', 'grasp', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'pusher', 'reacher', 'walker2d', 'grasp', 'ur5e']
env = envs.get_environment(env_name=env_name)
state = env.reset(rng=jp.random_prngkey(seed=0))

train_fn = {
  'walker2d': functools.partial(apg2.train, episode_length=1000, num_envs=8, num_evals=200, logdir='./logs'),
}[env_name]

max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'fetch': 15, 'grasp': 100, 'ur5e': 10, 'pusher': 0}[env_name]
min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)

xdata, ydata = [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  times.append(datetime.now())
  xdata.append(num_steps)
  ydata.append(metrics['eval/episode_reward'])
  clear_output(wait=True)
  # plt.xlim([0, train_fn.keywords['num_timesteps']])
  # plt.ylim([min_y, max_y])
  plt.xlabel('environment steps')
  plt.ylabel('reward per episode')
  plt.plot(xdata, ydata)
  plt.show()

train_fn(environment=env)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')