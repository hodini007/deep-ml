import numpy as np


def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    a, b = np.array(a), np.array(b)
    am, an = a.shape
    bm, bn = b.shape

    if an != bm:
        return -1

    c = a @ b
    return c.tolist()