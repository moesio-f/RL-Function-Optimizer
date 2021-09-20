"""Funções utilitárias para avaliação dos algoritmos."""
import collections
import typing
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_environment
from tf_agents.drivers import dynamic_episode_driver as dy_ed
from tf_agents.utils import common
from tf_agents.policies import tf_policy

from src.functions import core
from src.functions import tensorflow_functions as tff
from src.functions import numpy_functions as npf
from src.single_agent.metrics import tf_custom_metrics

from experiments.training import utils as training_utils


# Agrupa as informações necessárias sobre a avaliação de um baseline.
class BaselineEvalData(typing.NamedTuple):
  baseline_name: str
  function_name: str
  avg_best_solution: float
  stddev_best_solutions: float
  avg_best_solution_iteration: int


def evaluate_agent(eval_env: tf_environment.TFEnvironment,
                   policy_eval: tf_policy.TFPolicy,
                   function: core.Function,
                   dims,
                   steps,
                   algorithm_name,
                   agent_dir,
                   save_to_file=False,
                   episodes=100):
  tf_function = npf.get_tf_function(function)
  eval_metrics = [tf_custom_metrics.ConvergenceMultiMetric(steps + 1,
                                                           tf_function)]

  eval_driver = dy_ed.DynamicEpisodeDriver(env=eval_env,
                                           policy=policy_eval,
                                           observers=eval_metrics,
                                           num_episodes=episodes)
  eval_driver.run = common.function(eval_driver.run)

  results = eager_compute(eval_metrics,
                          eval_driver)
  mean = results.get(eval_metrics[0].name)[0]
  best = results.get(eval_metrics[0].name)[1]

  _, ax = plt.subplots(figsize=(18.0, 10.0,))

  ax.plot(mean, 'r', label='Best mean value: {0}'.format(mean[-1]))
  ax.plot(best, 'g', label='Best value: {0}'.format(best[-1]))

  ax.set(xlabel="Iterations",
         ylabel="Best objective value",
         title="{0} on {1} ({2} Dims)".format(algorithm_name,
                                              function.name,
                                              dims))

  ax.set_yscale('log')
  ax.set_xlim(left=0)

  ax.legend()
  ax.grid()

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
           rotation_mode="anchor")
  if save_to_file:
    filename = os.path.join(agent_dir,
                            '{0}-{1}D-{2}.png'.format(function.name,
                                                      dims,
                                                      algorithm_name))
    plt.savefig(fname=filename,
                bbox_inches='tight')
  plt.show()


def eager_compute(metrics,
                  driver,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix=''):
  for metric in metrics:
    metric.reset()

  environment = driver.env
  policy = driver.policy

  time_step = environment.reset()
  policy_state = policy.get_initial_state(environment.batch_size)
  driver.run(time_step, policy_state)
  results = [(metric.name, metric.result()) for metric in metrics]

  if train_step is not None and summary_writer:
    with summary_writer.as_default():
      for result in results:
        m_name, m_result = result
        tag = common.join_scope(summary_prefix, m_name)
        tf.compat.v2.summary.scalar(name=tag, data=m_result, step=train_step)

  return collections.OrderedDict(results)


def evaluate_baselines(functions: typing.List[core.Function],
                       dims: int,
                       steps=500,
                       episodes=100):
  baseline_eval_data: typing.List[BaselineEvalData] = []

  for fun in functions:
    rng = rand.default_rng()
    tf_function = npf.get_tf_function(fun)

    gd_bs = []
    gd_bs_it = []

    nag_bs = []
    nag_bs_it = []

    for ep in range(episodes):
      gd_pos = rng.uniform(size=(dims,),
                           low=fun.domain.min,
                           high=fun.domain.max)

      nag_pos = rng.uniform(size=(dims,),
                            low=fun.domain.min,
                            high=fun.domain.max)

      gd = GD(tf_function,
              pos=gd_pos,
              steps=steps)
      gd_bs.append(gd[0][-1])
      gd_bs_it.append(gd[1])

      nag = NAG(tf_function,
                pos=nag_pos,
                steps=steps)
      nag_bs.append(nag[0][-1])
      nag_bs_it.append(nag[1])

    data_gd = BaselineEvalData('GD',
                               tf_function.name,
                               np.mean(gd_bs).astype(np.float32).item(),
                               np.std(gd_bs).astype(np.float32).item(),
                               int(np.rint(np.mean(gd_bs_it))))
    baseline_eval_data.append(data_gd)

    data_nag = BaselineEvalData('NAG',
                                tf_function.name,
                                np.mean(nag_bs).astype(np.float32).item(),
                                np.std(nag_bs).astype(np.float32).item(),
                                int(np.rint(np.mean(nag_bs_it))))
    baseline_eval_data.append(data_nag)

  with open(f'baselines_{dims}D_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Baseline',
                     'Function',
                     'Dims',
                     'Avg Best Solution',
                     'Stddev of best solutions',
                     'Avg Best Solution Iteration'])
    for data in baseline_eval_data:
      writer.writerow([data.baseline_name,
                       data.function_name,
                       dims,
                       data.avg_best_solution,
                       data.stddev_best_solutions,
                       data.avg_best_solution_iteration])


# Parâmetros para o GD em cada função.
# Valores escolhidos após realizar 20 testes distintos (500 iterações)
#   e selecionando o parâmetro que obteve a melhor solução final (média).
# Conjunto de busca: {1e-1, 1e-2, 1e-3, 1e-4}
gd_lrs = {'F1': 1e-1,
          'F2': 1e-2,
          'F3': 1e-4,
          'F4': 1e-4,
          'F5': 1e-2,
          'F6': 1e-1,
          'F7': 1e-1,
          'F8': 1e-4}


# Baseline: GD (Gradient Descent)
def GD(function: core.Function,
       pos: typing.Union[tf.Tensor, np.ndarray],
       steps=500):
  lr = gd_lrs.get(function.name, 1e-2)
  pos = tf.convert_to_tensor(pos, dtype=tf.float32)
  best_solutions = [function(pos)]
  best_it = 0
  domain = function.domain

  for t in range(steps):
    grads, _ = tff.get_grads(function, pos)
    pos = tf.clip_by_value(pos - lr * grads,
                           clip_value_min=domain.min,
                           clip_value_max=domain.max)

    y = function(pos)
    if y < best_solutions[-1]:
      best_it = t
    best_solutions.append(min(y, best_solutions[-1]))
  return best_solutions, best_it


# Parâmetros para o NAG em cada função.
# Valores escolhidos após realizar 20 testes distintos e selecionando o par
#   (lr, momentum) de parâmetros que encontraram a melhor solução final (média)
# Conjunto de busca: {1e-1, 1e-2, 1e-3, 1e-4} X {0.5, 0.8, 0.9}
nag_params = {'F1': (1e-1, 0.5),
              'F2': (1e-3, 0.5),
              'F3': (1e-4, 0.9),
              'F4': (1e-4, 0.9),
              'F5': (1e-1, 0.8),
              'F6': (1e-3, 0.9),
              'F7': (1e-1, 0.9),
              'F8': (1e-4, 0.9)}


# Baseline: NAG (Nesterov accelerated gradient)
def NAG(function: core.Function,
        pos: typing.Union[tf.Tensor, np.ndarray],
        steps=500):
  lr, momentum = nag_params.get(function.name, (1e-2, 0.8))
  pos = tf.convert_to_tensor(pos, dtype=tf.float32)
  velocity = tf.zeros(shape=pos.shape, dtype=tf.float32)
  domain = function.domain

  best_solutions = [function(pos)]
  best_it = 0

  for t in range(steps):
    projected = tf.clip_by_value(pos + momentum * velocity,
                                 clip_value_min=domain.min,
                                 clip_value_max=domain.max)
    grads, _ = tff.get_grads(function, projected)

    velocity = momentum * velocity - lr * grads
    pos = tf.clip_by_value(pos + velocity,
                           clip_value_min=domain.min,
                           clip_value_max=domain.max)

    current_pos = function(pos)
    if current_pos < best_solutions[-1]:
      best_it = t
    best_solutions.append(min(current_pos, best_solutions[-1]))
  return best_solutions, best_it


def plot_convergence(functions: typing.List[str],
                     show=False,
                     dpi=300,
                     style=None):
  if style is None:
    style = ['-', '-', '-', '-', '--', '--']

  for F in functions:
    file = F + '_30D_convergence.csv'
    data = pd.read_csv(file)
    del data['iteration']
    data.plot(logx='sym', logy='sym', grid=True, fontsize=9, style=style)
    plt.xlabel('Iterações')
    plt.ylabel('Melhor Valor')
    plt.savefig(F + '_30D_plot', dpi=dpi)
  if show:
    plt.show()
