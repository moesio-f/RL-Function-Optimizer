"""TFEnvironments that implement mathematical functions as environments."""

import tensorflow as tf
from tensorflow.python.autograph.impl import api as autograph
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from src.single_agent.environments import py_function_environment
import src.functions.base

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST
MAX_STEPS = 50000


class TFFunctionEnvironment(tf_environment.TFEnvironment):
  """Single-agent function environment."""
  def __init__(self, function: src.functions.base.Function, dims,
               clip_actions: bool = False):
    self._function = function
    self._domain_min = tf.cast(function.domain.min, tf.float32)
    self._domain_max = tf.cast(function.domain.max, tf.float32)
    self._dims = dims

    action_spec = specs.BoundedTensorSpec(shape=(self._dims,), dtype=tf.float32,
                                          minimum=-1.0,
                                          maximum=1.0,
                                          name='action')
    if not clip_actions:
      action_spec = specs.TensorSpec.from_spec(action_spec)

    observation_spec = specs.BoundedTensorSpec(shape=(self._dims,),
                                               dtype=tf.float32,
                                               minimum=self._domain_min,
                                               maximum=self._domain_max,
                                               name='observation')

    time_step_spec = ts.time_step_spec(observation_spec)
    super().__init__(time_step_spec, action_spec)

    self._generator = tf.random.Generator.from_non_deterministic_state()

    self._episode_ended = common.create_variable(name='episode_ended',
                                                 initial_value=False,
                                                 dtype=tf.bool)
    self._steps_taken = common.create_variable(name='steps_taken',
                                               initial_value=0, dtype=tf.int32)
    self._state = common.create_variable(name='state',
                                         initial_value=self._generator.uniform(
                                           shape=observation_spec.shape,
                                           minval=self._domain_min,
                                           maxval=self._domain_max,
                                           dtype=tf.float32),
                                         dtype=tf.float32)

    self._last_objective_value = common.create_variable(
      name='last_objective_value',
      initial_value=self._function(self._state),
      dtype=tf.float32)
    self._last_position = common.create_variable(name='last_position',
                                                 initial_value=self._state,
                                                 dtype=tf.float32)

  def _current_time_step(self) -> ts.TimeStep:
    state = self._state.value()

    def first():
      return (tf.constant(FIRST, dtype=tf.int32),
              tf.constant(0.0, dtype=tf.float32))

    def mid():
      return (tf.constant(MID, dtype=tf.int32),
              tf.math.negative(self._function(state)))

    def last():
      return (tf.constant(LAST, dtype=tf.int32),
              tf.math.negative(self._function(state)))

    discount = tf.constant(1.0, dtype=tf.float32)
    step_type, reward = tf.case(
      [(tf.math.less_equal(self._steps_taken, 0), first),
       (tf.math.reduce_any(self._episode_ended), last)],
      default=mid,
      exclusive=True, strict=True)

    return ts.TimeStep(step_type=step_type,
                       reward=reward,
                       discount=discount,
                       observation=state)

  def _reset(self) -> ts.TimeStep:
    reset_ended = self._episode_ended.assign(value=False)
    reset_steps = self._steps_taken.assign(value=0)

    with tf.control_dependencies([reset_ended, reset_steps]):
      state_reset = self._state.assign(
        value=self._generator.uniform(shape=self.observation_spec().shape,
                                      minval=self._domain_min,
                                      maxval=self._domain_max,
                                      dtype=tf.float32))
    with tf.control_dependencies([state_reset]):
      self._last_position.assign(self._state)
      self._last_objective_value.assign(self._function(self._state))
      time_step = self.current_time_step()

    return time_step

  def _step(self, action):
    action = tf.convert_to_tensor(value=action)

    def take_step():
      with tf.control_dependencies(tf.nest.flatten(action)):
        new_state = tf.clip_by_value(self._state + action,
                                     clip_value_min=self._domain_min,
                                     clip_value_max=self._domain_max)

      with tf.control_dependencies([new_state]):
        state_update = self._state.assign(new_state)
        self._steps_taken.assign_add(1)
        episode_finished = tf.cond(
          pred=tf.math.greater_equal(self._steps_taken, MAX_STEPS),
          true_fn=lambda: self._episode_ended.assign(True),
          false_fn=self._episode_ended.value)

      with tf.control_dependencies([state_update, episode_finished]):
        self._last_position.assign(self._state)
        self._last_objective_value.assign(self._function(self._state))
        return self.current_time_step()

    def reset_env():
      return self.reset()

    return tf.cond(pred=tf.math.reduce_any(self._episode_ended),
                   true_fn=reset_env,
                   false_fn=take_step)

  @autograph.do_not_convert()
  def get_info(self, to_numpy=False):
    if to_numpy:
      return py_function_environment.FunctionEnvironmentInfo(
        position=self._last_position.value().numpy(),
        objective_value=self._last_objective_value.value().numpy())

    return py_function_environment.FunctionEnvironmentInfo(
      position=self._last_position,
      objective_value=self._last_objective_value)

  def render(self):
    raise ValueError('Environment does not support render yet.')
