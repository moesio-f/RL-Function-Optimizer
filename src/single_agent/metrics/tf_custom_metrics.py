"""Custom TF Metrics."""

import tensorflow as tf
import numpy as np

from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from src.functions import core


# AverageBestObjectiveValue: Calcula o valor objetivo médio entre
#   todos episódios.
# MinBestObjectiveValue: Calcula o menor valor objetivo de todos episódios
# MaxBestObjectiveValue: Calcula o maior valor objetivo de todos episódios
# Average???: Calcula a iteração média onde o agente parou de melhorar

class AverageBestObjectiveValueMetric(tf_metric.TFStepMetric):
  """Metric to compute the average return."""

  def __init__(self,
               function: core.Function,
               name='AverageBestValue',
               prefix='Metrics',
               dtype=tf.float32,
               buffer_size=10):
    super(AverageBestObjectiveValueMetric, self).__init__(name=name,
                                                          prefix=prefix)
    self._function = common.function(function)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._best_value = common.create_variable(initial_value=0, dtype=dtype,
                                              shape=(), name='BestValue')
    self._current_value = common.create_variable(initial_value=0, dtype=dtype,
                                                 shape=(), name='CurrentValue')

  @common.function(autograph=True)
  def call(self, trajectory):
    self._current_value.assign(self._function(trajectory.observation[0]))

    if trajectory.is_first():
      self._best_value.assign(self._current_value)
    else:
      if tf.reduce_any(tf.math.less_equal(self._current_value,
                                          self._best_value)):
        self._best_value.assign(self._current_value)

      if trajectory.is_last():
        self._buffer.add(self._best_value)

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._best_value.assign(0)
