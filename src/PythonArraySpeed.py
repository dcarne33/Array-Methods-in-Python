# Array iteration methods in Python 3.10
# Copyright (C) 2023 Daniel Carne <dandaman35@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from numba import njit, prange
import time

# Base python
def func1(a, b, size, iter):
    for i in range(iter):
        for j in range(size):
            for k in range(size):
                a[j, k] += b[j, k]
    return a

# Numpy vectorized
def func2(a, b, iter):
    for i in range(iter):
        a[:, :] += b[:, :]
    return a

# Numba compiled
@njit()
def func3(a, b, size, iter):
    for i in range(iter):
        for j in range(size):
            for k in range(size):
                a[j, k] += b[j, k]
    return a

# Numba compiled in parallel
@njit(parallel=True)
def func4(a, b, size, iter):
    for i in range(iter):
        for j in prange(size):
            for k in range(size):
                a[j, k] += b[j, k]
    return a


if __name__ == "__main__":
    # iterations to loop through
    iter = 5
    # size of 2d array
    size = 1000
    # two arrays for computation
    a = b = np.ones((size, size))

    # func1
    start = time.time()
    a = func1(a, b, size, iter)
    print("Base python time: ", (time.time()-start), "\n")

    # func2
    start = time.time()
    a = func2(a, b, iter)
    print("NumPy vectorized time: ", (time.time() - start), "\n")

    # func3. call compiled function once before timing so compilation is not included in time.
    a = func3(a, b, size, iter)
    start = time.time()
    a = func3(a, b, size, iter)
    print("Numba compiled time: ", (time.time() - start), "\n")

    # func4. call compiled function once before timing so compilation is not included in time.
    a = func4(a, b, size, iter)
    start = time.time()
    a = func4(a, b, size, iter)
    print("Numba compiled in parallel time: ", (time.time() - start), "\n")
