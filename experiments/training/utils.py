"""Funções utilitárias para o treinamento dos agentes."""

import os
import json
import typing

from tensorflow import io as tf_io

from tf_agents.policies import tf_policy
from tf_agents.policies import policy_saver

from src.functions import core
from src import config

ROOT_DIR = config.ROOT_DIR


def create_logs_dir(agent_dir: str):
  log_dir = os.path.join(agent_dir, 'logs')
  log_eval_dir = os.path.join(log_dir, 'eval')
  log_train_dir = os.path.join(log_dir, 'train')

  tf_io.gfile.makedirs(log_eval_dir)
  tf_io.gfile.makedirs(log_train_dir)

  return log_dir, log_eval_dir, log_train_dir


def save_policy(agent_dir: str,
                policy: tf_policy.TFPolicy):
  output_dir = os.path.join(agent_dir, 'policy')
  tf_io.gfile.makedirs(output_dir)

  tf_policy_saver = policy_saver.PolicySaver(policy)
  tf_policy_saver.save(output_dir)


def save_specs(agent_dir: str,
               specs_dict: typing.Dict):
  with open(os.path.join(agent_dir, 'specs.json'), 'w') as specs_file:
    json.dump(specs_dict, specs_file, indent=True)


def create_agent_dir(algorithm_name: str,
                     function: core.Function,
                     dims: int) -> str:
  str_dims = str(dims)
  agent_identifier = f'{algorithm_name}-{str_dims}D-{function.name}-0'
  output_dir = os.path.join(ROOT_DIR, 'output')
  agent_dir = os.path.join(output_dir, agent_identifier)

  # TODO: Diminuir complexidade, atualmente O(n)
  i = 0
  while os.path.exists(agent_dir):
    i += 1
    agent_identifier = f'{algorithm_name}-{str_dims}D-{function.name}-{i}'
    agent_dir = os.path.join(output_dir, agent_identifier)

  tf_io.gfile.makedirs(agent_dir)
  return agent_dir


def json_pretty_string(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))
