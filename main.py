import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def my_sum(a: float, b: float) -> float:
    return a + b


if __name__ == '__main__':
    np.random.seed(0)
    print(my_sum(3, 5))
