import numpy
import numpy as np
import scipy.linalg as sp

A = numpy.array([[1, 2], [2, 3]])

inv = sp.inv(A)

trace = np.trace(inv)

print(trace)