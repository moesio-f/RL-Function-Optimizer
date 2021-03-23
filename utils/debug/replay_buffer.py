import tensorflow as tf
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories.trajectory import Trajectory
import numpy as np


class ReplayBufferState(object):
    def __init__(self, replay_buffer: TFUniformReplayBuffer):
        self._data_spec = replay_buffer.data_spec

        self._list_trajectories = list(replay_buffer.as_dataset(single_deterministic_pass=True).as_numpy_iterator())
        self._num_samples = len(self._list_trajectories)

        self._avg_obs = None
        self._avg_act = None
        self._avg_rew = None

        self._initialize_statistics()

    def num_samples(self):
        return self._num_samples

    @property
    def average_observation(self):
        return self._avg_obs

    @property
    def average_action(self):
        return self._avg_act

    @property
    def average_reward(self):
        return self._avg_rew

    def num_observations_close_to(self, value: np.ndarray, ref_l2_norm=1.0):
        self._validate_observation(value)
        num = 0

        for item in self._list_trajectories:
            observation = item[0].observation
            if np.linalg.norm(value - observation) <= ref_l2_norm:
                num = num + 1

        return num, 100 * (num / self._num_samples)

    def _initialize_statistics(self):
        sum_obs = 0
        sum_act = 0
        sum_rew = 0

        for item in self._list_trajectories:
            trajectory = item[0]
            sum_obs = sum_obs + trajectory.observation
            sum_act = sum_act + trajectory.action
            sum_rew = sum_rew + trajectory.reward

        self._avg_obs = sum_obs / self._num_samples
        self._avg_act = sum_act / self._num_samples
        self._avg_rew = sum_rew / self._num_samples

    def _validate_observation(self, observation):
        obs_shape = self._data_spec.observation.shape[0]
        obs_min = self._data_spec.observation.minimum
        obs_max = self._data_spec.observation.maximum
        obs_dtype = self._data_spec.observation.dtype.as_numpy_dtype

        if observation.shape[0] != obs_shape:
            raise ValueError("Observation shape doesn't match spec: {0} != {1}".format(observation.shape[0],
                                                                                       obs_shape))
        elif observation.dtype != obs_dtype:
            raise ValueError("Observation dtype doesn't match spec: {0} != {1}".format(observation.dtype,
                                                                                       obs_dtype))
        elif np.any(np.less(observation, obs_min)) or np.any(np.greater(observation, obs_max)):
            raise ValueError("Observation is out of spec bounds")

