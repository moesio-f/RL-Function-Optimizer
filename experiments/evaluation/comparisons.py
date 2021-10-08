"""Realiza a comparação da convergência com os diferentes algoritmos."""

import os
import collections
from typing import List, Tuple
import functools

import numpy as np
import tensorflow as tf
import pandas as pd

from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.policies import tf_policy
from tf_agents.utils import common

from src.single_agent.environments import py_function_environment as py_fun_env
from src.single_agent.metrics import tf_custom_metrics
from src.single_agent.typing.types import TFMetric
from src.functions import core
from src.functions import numpy_functions as npf
from src import config

from experiments.evaluation import utils as eval_utils

MODELS_DIR = config.POLICIES_DIR


class DummyClass:
  def __init__(self,
               policy: tf_policy.TFPolicy,
               driver: dy_ed.DynamicEpisodeDriver,
               tf_env: tf_py_environment.TFPyEnvironment,
               metrics: List[TFMetric],
               algorithm_name: str):
    self._policy = policy
    self._driver = driver
    self._tf_env = tf_env
    self._algorithm_name = algorithm_name
    self._metrics = metrics
    self._last_results = None
    self._convergence_metric_idx = None

    for idx, metric in enumerate(self._metrics):
      if isinstance(metric, tf_custom_metrics.ConvergenceMultiMetric):
        self._convergence_metric_idx = idx

    assert self._convergence_metric_idx is not None

  def run(self, initial_time_step):
    self._update_current_time_step(initial_time_step, 0)
    self._driver.run(time_step=initial_time_step)

  def average_trajectory(self):
    if self._last_results is None:
      self.compute_results()
    return self._last_results.get(
      self._metrics[self._convergence_metric_idx].name)[0].numpy()

  def compute_results(self):
    self._last_results = collections.OrderedDict(
      [(metric.name, metric.result()) for metric in
       self._metrics])
    return self._last_results

  def _update_current_time_step(self, new_time_step, num_steps: int):
	# Erro atual: Trajetória inicial possui 2 time_steps iniciais (função reset() sendo chamada)
	# Possibilidade 1: Alterar demais time_steps dos ambientes envolvidos (PyEnv, TimeLimit)
	#	bem como suas variáveis de controle (duration, episode_ended, ...)
	# Possibilidade 2: Alterar estrutura da classe para criar um novo
	#	ambiente sempre que for rodar um episódio de teste (Armazenar dados num TFDeque criado pela
	#	classe)
    self._tf_env._time_step = new_time_step
    self._tf_env._num_steps = num_steps
    self._driver.env._time_step = new_time_step


def write_to_csv(trajectories: List[Tuple],
                 function: core.Function,
                 dims: int):
  file_name = f'{function.name}_{dims}D_convergence.csv'
  data = pd.DataFrame({t[0]: t[1] for t in trajectories})
  data.to_csv(file_name, index_label='iteration')


def create_evaluator(algorithm_name: str,
                     function: core.Function,
                     tf_function: core.Function,
                     dims: int,
                     steps: int,
                     episodes: int,
                     use_tf_function: bool = False):
  policy = tf.compat.v2.saved_model.load(
    policy_path(algorithm_name,
                dims,
                function))

  tf_env = tf_py_environment.TFPyEnvironment(
    environment=wrappers.TimeLimit(
      py_fun_env.PyFunctionEnv(function, dims),
      duration=steps))

  eval_metrics = [tf_custom_metrics.ConvergenceMultiMetric(
    trajectory_size=steps + 1,
    function=tf_function,
    buffer_size=episodes)]

  driver = dy_ed.DynamicEpisodeDriver(env=tf_env,
                                      policy=policy,
                                      observers=eval_metrics,
                                      num_episodes=1)

  if use_tf_function:
    driver.run = common.function(driver.run)

  return DummyClass(policy=policy,
                    driver=driver,
                    metrics=eval_metrics,
                    tf_env=tf_env,
                    algorithm_name=algorithm_name)


def policy_path(agent: str, dims: int, fun: core.Function):
  return os.path.join(MODELS_DIR, agent, str(dims) + 'D', fun.name)


if __name__ == '__main__':
  # Dimensões das funções.
  DIMS = 30
  # Quantidade de episódios para o cálculo das medidas.
  EPISODES = 2
  # Quantidade de interações agente-ambiente.
  STEPS = 4
  # Lista com as funções que serão testadas.
  FUNCTIONS = [npf.Sphere(), npf.Ackley()]

  tf.config.run_functions_eagerly(True)

  for FUNCTION in FUNCTIONS:
    ENV = py_fun_env.PyFunctionEnv(FUNCTION, DIMS)
    ENV = wrappers.TimeLimit(ENV, duration=STEPS)
    TF_ENV = tf_py_environment.TFPyEnvironment(environment=ENV)
    TF_FUNCTION = npf.get_tf_function(FUNCTION)

    create_evaluator_ = functools.partial(create_evaluator,
                                          function=FUNCTION,
                                          tf_function=TF_FUNCTION,
                                          dims=DIMS,
                                          steps=STEPS,
                                          episodes=EPISODES)

    reinforce_evaluator = create_evaluator_('ReinforceAgent')
    sac_evaluator = create_evaluator_('SacAgent')
    td3_evaluator = create_evaluator_('Td3Agent')
    ppo_evaluator = create_evaluator_('PPOClipAgent')

    for _ in range(EPISODES):
      initial_ts = TF_ENV.reset()

      reinforce_evaluator.run(initial_ts)
      sac_evaluator.run(initial_ts)
      td3_evaluator.run(initial_ts)
      ppo_evaluator.run(initial_ts)

    avg_reinforce = ('REINFORCE', reinforce_evaluator.average_trajectory())
    avg_sac = ('SAC', sac_evaluator.average_trajectory())
    avg_td3 = ('TD3', td3_evaluator.average_trajectory())
    avg_ppo = ('PPO', ppo_evaluator.average_trajectory())

    write_to_csv([avg_reinforce, avg_sac, avg_td3, avg_ppo],
                 function=FUNCTION,
                 dims=DIMS)
