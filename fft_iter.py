import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from math import pi, cos, sin

def iter_fft(x):
    n = len(x)
    if n == 1:
        return x

    log_n = int(np.log2(n))
    rev = [0] * n
    for i in range(n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (log_n - 1))

    x = [x[rev[i]] for i in range(n)]

    size = 2
    while size <= n:
        half = size // 2
        w_step = np.exp(-2j * pi / size)
        for k in range(0, n, size):
            w = 1
            for j in range(half):
                even = x[k + j]
                odd = x[k + j + half] * w
                x[k + j] = even + odd
                x[k + j + half] = even - odd
                w *= w_step
        size *= 2

    return x
