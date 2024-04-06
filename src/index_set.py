import random

import numpy as np


def create_index_sets(n: int, cardinality_K: int, uniform: bool = True, seed: int = None):
    if seed:
        random.seed(seed)
    Is = []
    ks = list(np.arange(n))
    if uniform:
        cardinality_I = n // cardinality_K
        for _ in range(cardinality_K + 1):
            I = []
            if n == 0:
                break
            if n < cardinality_I:
                cardinality_I = n
            for _ in range(cardinality_I):
                i = random.choice(ks)
                I.append(i)
                ks.remove(i)
            Is.append(I)
            n -= cardinality_I
    else:
        while n > 0:
            if n == 1:
                cardinality = 1
            else:
                cardinality = random.choice(range(1, n))
            n -= cardinality
            I = []
            for _ in range(cardinality):
                i = random.choice(ks)
                I.append(i)
                ks.remove(i)
            Is.append(I)

    return Is


def create_feasible_point(n: int, Is: list):
    x = np.zeros(n)
    for I in Is:
        x[I] = 1 / len(I)
    return x
