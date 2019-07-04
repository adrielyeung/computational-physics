import numpy as np
# A very crude way to calculate machine accuracy is to start from a value and
# then keep reducing it until it cannot be meaningfully added to 1 (i.e.
# 1 + the number = 1)

def machacc():
    i = np.float(1.)
    while 1. + i / 2 != 1:
        i = i / 2
    return i

# This function calculates the machine accuracy for 32- and 64-bits. However,
# 128-bit is not available using NumPy.
# Input: bit is an integer specifying the number of bits
def precacc(bit):
    if bit != 32 and bit != 64 and bit != 128:
        print("Precision input error: please use 32 for 'single', 64 for 'double'"
              " or 128 for 'extended' to represent the storage format.")
    elif bit == 32:
        i = np.float32(1.)
        while np.float32(1) + i / np.float32(2) != np.float32(1):
            i = i / np.float32(2)
        return i
    elif bit == 64:
        i = np.float64(1.)
        while np.float64(1) + i / np.float64(2) != np.float64(1):
            i = i / np.float64(2)
        return i
    elif bit == 128:
        i = np.longdouble(1.)
        while np.longdouble(1) + i / np.longdouble(2) != np.longdouble(1):
            i = i / 2
        return i
    # Even longdouble is still 64-bit precision.
    # Read up online: NumPy does not provide a dtype with more precision than C
    # long double``s; in particular, the 128-bit IEEE quad precision data type
    # (FORTRAN's ``REAL*16) is not available.
    # Expect 1.93e-34 ~ 2^-112.