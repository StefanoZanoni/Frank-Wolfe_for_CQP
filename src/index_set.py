import random

import numpy as np


def create_index_sets(n: int, cardinality_K: int = 1, uniform: bool = True, seed: int = None) -> list[int]:
    """
    Creates index sets (with a minimum cardinality of 2) for a specified number of elements,
     with options for uniform or random cardinality.

    Parameters:
    - n (int): The total number of elements to create index sets for.
    - cardinality_K (int, optional): The desired cardinality of each index set.
     This parameter is ignored if `uniform` is False. Default to 1.
    - uniform (bool, optional): Determines the uniformity of the index sets' cardinality.
     If True, index sets will have uniform cardinality. If False, index sets will have random cardinality.
      Defaults to True.
    - seed (int, optional): The seed value for the random number generator, ensuring reproducibility. Defaults to None.

    Returns:
    - list[int]: A list of index sets, where each index set is represented as a list of integers.
    """

    if seed:
        random.seed(seed)

    ks = set(np.arange(n))
    Is = []
    if uniform:
        cardinality_I = max(n // cardinality_K, 2)
        for _ in range(cardinality_K + 1):
            if n == 0:
                break
            if n < cardinality_I + 2:
                cardinality_I = n
            I = random.sample(list(ks), cardinality_I)
            ks -= set(I)
            Is.append(I)
            n -= cardinality_I
    else:
        while n > 0:
            cardinality = random.randint(2, n)
            while n - cardinality == 1:
                cardinality = random.randint(2, n)
            n -= cardinality
            I = random.sample(list(ks), cardinality)
            ks -= set(I)
            Is.append(I)

    if seed:
        random.seed()

    return Is
