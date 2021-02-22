import matplotlib as mpl
import tensorflow as tf
import numpy as np


# === Multi-modal ===
# D dimensões - Múltiplos mínimos
def ackley(x, a=20, b=0.2, c=2 * np.math.pi):
    if isinstance(x, tf.Tensor):
        d = x.shape.as_list()[0]
    else:
        d = len(x)
    return -a * tf.exp(-b * tf.sqrt(tf.reduce_sum(x * x) / d)) - \
           tf.exp(tf.reduce_sum(tf.cos(c * x)) / d) + a + np.math.e


# D dimensões - Múltiplos mínimos
def griewank(x: tf.Tensor):
    sum = tf.reduce_sum(x ** 2) / 4000.0
    den = tf.range(1, x.shape[0], dtype=x.dtype)
    prod = tf.cos(x / tf.sqrt(den))
    prod = tf.reduce_prod(prod)
    return sum - prod + 1


# D dimensões - Múltiplos mínimos
def rastrigin(x):
    if isinstance(x, tf.Tensor):
        d = x.shape.as_list()[0]
    else:
        d = len(x)
    return 10 * d + tf.reduce_sum(x ** 2 - 10 * tf.cos(x * 2 * np.math.pi))


# === Valley-shaped ===
# D dimensões
def rosenbrock(x):
    x = tf.cast(x, dtype=tf.float32)
    rosen_sum = 0.0

    if isinstance(x, tf.Tensor):
        d = x.shape[0]
    else:
        d = len(x)

    for i in range(d - 1):
        rosen_sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1.0) ** 2

    return rosen_sum


# === Plate-shaped ===
# D dimensões
def zakharov(x):
    x = tf.cast(x, dtype=tf.float32)

    if isinstance(x, tf.Tensor):
        d = x.shape[0]
    else:
        d = len(x)

    sum1 = tf.reduce_sum(x * x)
    sum2 = tf.reduce_sum(tf.constant([0.5 * (i + 1) for i in range(d)]) * x)
    return sum1 + sum2 ** 2 + sum2 ** 4


# === Bowl-shaped ===
# 2 Dimensões
def bohachevsky(x):
    x = tf.cast(x, dtype=tf.float32)

    if tf.is_tensor(x):
        d = x.shape[0]
    else:
        d = len(x)

    assert d == 2

    return x[0] ** 2 + 2 * (x[1] ** 2) - 0.3 * tf.cos(3 * np.pi * x[0]) - 0.4 * tf.cos(4 * np.pi * x[1]) + 0.7


# D dimensões
def sum_squares(x):
    x = tf.cast(x, dtype=tf.float32)

    if tf.is_tensor(x):
        d = x.shape[0]
    else:
        d = len(x)

    mul = tf.constant([(i + 1) for i in range(d)], dtype=tf.float32)
    return tf.reduce_sum((x ** 2) * mul)


# D dimensões
def sphere(x):
    return tf.reduce_sum(x * x)


# === Funções Utilitárias ===
# Recebe uma função como argumenta
# Retorna o nome dessa função
def get_function_name(function):
    if function is sphere:
        return 'sphere'
    elif function is ackley:
        return 'ackley'
    elif function is rastrigin:
        return 'rastrigin'
    elif function is rosenbrock:
        return 'rosenbrock'
    elif function is bohachevsky:
        return 'bohachevsky'
    elif function is sum_squares:
        return 'sum_squares'
    elif function is zakharov:
        return 'zakharov'
    elif function is griewank:
        return 'griewank'


# Recebe uma função como argumento
# Retorna o 'limite inferior e superior' da função
def get_low_and_high(function):
    if function is sphere:
        return -5.12, 5.12
    elif function is ackley:
        return -32.768, 32.768
    elif function is rastrigin:
        return -5.12, 5.12
    elif function is rosenbrock:
        return -5.0, 10.0
    elif function is bohachevsky:
        return -100.0, 100.0
    elif function is sum_squares:
        return -10.0, 10.0
    elif function is zakharov:
        return -5.0, 10.0
    elif function is griewank:
        return -600.0, 600.0
