import random

import numpy as np


def create_index_sets(n: int, cardinality_K: int = 1, uniform: bool = True, seed: int = None) -> list[int]:
    """
    Create index sets for a given number of elements.

    Args:
        n (int): The total number of elements.
        cardinality_K (int): The desired cardinality of each index set.
        It will be ignored if uniform is False.
        uniform (bool, optional): If True, create index sets with uniform cardinality. 
            If False, create index sets with random cardinality.
            Defaults to True.
        seed (int, optional): The seed value for the random number generator.
        Defaults to None.

    Returns:
        list: A list of index sets, where each index set is represented as a list of integers.

    """
    if seed:
        random.seed(seed)
    ks = set(np.arange(n))
    Is = []
    if uniform:
        cardinality_I = n // cardinality_K
        for _ in range(cardinality_K + 1):
            if n == 0:
                break
            if n < cardinality_I:
                cardinality_I = n
            I = random.sample(list(ks), cardinality_I)
            ks -= set(I)
            Is.append(I)
            n -= cardinality_I
    else:
        while n > 0:
            if n == 1:
                cardinality = 1
            else:
                cardinality = random.randint(1, n)
            n -= cardinality
            I = random.sample(list(ks), cardinality)
            ks -= set(I)
            Is.append(I)
    return Is
