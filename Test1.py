import Floating_1 as F1

print(F1.machacc())

# Returns 2.22e-16 ~ 2^-52, so conclude that it is a 64-bit storage with a hidden
# bit.

print(F1.precacc(32))

# Returns 1.19e-7 ~ 2^-23, 32-bit with hidden bit

print(F1.precacc(64))

# Same as using machacc

print(F1.precacc(128))

# Did not work because even longdouble is still 64-bit precision.
# Read up online: NumPy does not provide a dtype with more precision than C
# long double``s; in particular, the 128-bit IEEE quad precision data type
# (FORTRAN's ``REAL*16) is not available.
# Expect 1.93e-34 ~ 2^-112.