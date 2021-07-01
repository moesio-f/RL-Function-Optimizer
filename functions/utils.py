import numpy as np
import functions.numpy_functions as functions_np
import functions.tensorflow_functions as functions_tf
from functions.function import Function


def getCommonFunction(name, in_tensorflow=False) -> Function:
    if in_tensorflow:
        function = next((f for f in functions_tf.list_all_functions() if f.name == name), None)
    else:
        function = next((f for f in functions_np.list_all_functions() if f.name == name), None)

    return function

def test_functions():
    for func in functions_np.list_all_functions():
        min, max = func.domain
        pos = np.random.uniform(min, max, (2,10))
        result = func(pos)
        expected_shape = (10,)
        if result.shape != expected_shape:
            raise ValueError(f'Function {func.name} is broken! result shape is {result.shape} instead of: {expected_shape}')
        else:
            print(f'Function {func.name}: Passed test!')

if __name__ == '__main__':
    test_functions()