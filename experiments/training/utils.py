"""Funções utilitárias para o treinamento dos agentes."""

import os

from tensorflow import io as tf_io

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_saver

from src.functions import core
from src import config

ROOT_DIR = config.ROOT_DIR


def create_logs_dir(algorithm_name: str,
                    function: core.Function,
                    dims: int):
  agent_dir = path_agent_dir(algorithm_name, function, dims)

  log_dir = os.path.join(agent_dir, 'logs')
  log_eval_dir = os.path.join(log_dir, 'eval')
  log_train_dir = os.path.join(log_dir, 'train')

  tf_io.gfile.makedirs(log_eval_dir)
  tf_io.gfile.makedirs(log_train_dir)

  return log_dir, log_eval_dir, log_train_dir


def save_policy(algorithm_name: str,
                function: core.Function,
                dims: int,
                policy: tf_policy.TFPolicy):
  agent_dir = path_agent_dir(algorithm_name, function, dims)
  output_dir = os.path.join(agent_dir, 'policy')
  tf_io.gfile.makedirs(output_dir)

  tf_policy_saver = policy_saver.PolicySaver(policy)
  tf_policy_saver.save(output_dir)


def path_agent_dir(algorithm_name: str,
                   function: core.Function,
                   dims: int) -> str:
  agent_dir = os.path.join(ROOT_DIR, 'output')
  agent_dir = os.path.join(agent_dir,
                           f'{algorithm_name}-{str(dims)}D-{function.name}')
  return agent_dir
