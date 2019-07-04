import Matrix_2 as M2
import imp
imp.reload(M2)
import numpy as np
import scipy as sp

A = np.matrix([[2,1,0,0,0],[3,8,4,0,0],[0,9,20,10,0],[0,0,22,51,-25],[0,0,0,-55,60]],dtype=float)

print(M2.LU(A, False))

#[[  2.           1.           0.           0.           0.        ]
# [  1.5          6.5          4.           0.           0.        ]
# [  0.           1.38461538  14.46153846  10.           0.        ]
# [  0.           0.           1.5212766   35.78723404 -25.        ]
# [  0.           0.           0.          -1.53686088  21.578478  ]]

L, U = M2.LU(A)
print(L, U)

#[[ 1.          0.          0.          0.          0.        ]
# [ 1.5         1.          0.          0.          0.        ]
# [ 0.          1.38461538  1.          0.          0.        ]
# [ 0.          0.          1.5212766   1.          0.        ]
# [ 0.          0.          0.         -1.53686088  1.        ]]
#[[  2.           1.           0.           0.           0.        ]
# [  0.           6.5          4.           0.           0.        ]
# [  0.           0.          14.46153846  10.           0.        ]
# [  0.           0.           0.          35.78723404 -25.        ]
# [  0.           0.           0.           0.          21.578478  ]]

# Check to see if L*U = A

print(np.dot(L, U))

#[[  2.   1.   0.   0.   0.]
# [  3.   8.   4.   0.   0.]
# [  0.   9.  20.  10.   0.]
# [  0.   0.  22.  51. -25.]
# [  0.   0.   0. -55.  60.]]

# Check with scipy function (permute_l = True means the permutation matrix is 
# multiplied into L so A = (PL)*U)

sp.linalg.lu(A, permute_l = True)

# Using this function, the L and U matrices returned multiply to give the actual
# matrix. Although using scipy.linalg.lu gives a different result because there
# is pivoting. After selecting permute_l = True, the two matrices matched.

print(M2.det(A))

print(np.linalg.det(A))

# Returns 145180.0 for given matrix, which matches the value calculated by numpy.

b = np.array([2,5,-4,8,9], dtype=float)
x = M2.solve(A, b)
print(x)

# x = [ 0.33764981  1.32470037 -1.6526381   1.71304587  1.72029205]

print(np.dot(A, x))

# [[ 2.  5. -4.  8.  9.]]

Ain = M2.inv(A)
print(Ain)

# Returns array([[ 0.71180603, -0.14120402,  0.04642513, -0.0165312 , -0.006888  ],
#       [-0.42361207,  0.28240805, -0.09285025,  0.03306241,  0.013776  ],
#       [ 0.31336961, -0.20891307,  0.15088166, -0.05372641, -0.022386  ],
#       [-0.24548836,  0.16365891, -0.1181981 ,  0.07769665,  0.03237361],
#       [-0.225031  ,  0.15002066, -0.10834826,  0.07122193,  0.04634247]])

print(np.dot(A, Ain))

#[[  1.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00]
# [ -2.22044605e-16   1.00000000e+00   0.00000000e+00  -2.77555756e-17
#    0.00000000e+00]
# [  3.33066907e-16   1.11022302e-16   1.00000000e+00   0.00000000e+00
#   -1.38777878e-17]
# [  7.49400542e-16  -1.38777878e-15   9.99200722e-16   1.00000000e+00
#   -2.35922393e-16]
# [  3.33066907e-16   1.55431223e-15  -1.33226763e-15   4.44089210e-16
#    1.00000000e+00]]

# All off-diagonal elements are of order e-15 to e-17 so negligible
# so this has multiplied with A to give the identity matrix.