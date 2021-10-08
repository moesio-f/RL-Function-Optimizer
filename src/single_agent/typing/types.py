"""Common types used."""

import typing

from tf_agents.metrics import tf_metric
from tf_agents.typing import types as tf_types

LayerParam = typing.Union[typing.List, typing.Tuple]
TFMetric = typing.Union[tf_metric.TFStepMetric,
                        tf_metric.TFMultiMetricStepMetric,
                        tf_metric.TFHistogramStepMetric]
