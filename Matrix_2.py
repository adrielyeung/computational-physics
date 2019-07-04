import numpy as np

# LU Decomposition
# Inputs: A = matrix to be decomposed, sep (boolean) determines whether an LU combined
# matrix or two separate matrices.
# Outputs: LU is a matrix with all elements of U and all non-diagonal elements
# of L, or L and U which are two matrices separated into lower and upper triangular
# matrices.

def LU(A, sep = True):
    if A.shape[0] != A.shape[1]:
        print("Not a square matrix")
    else:
        LU = np.zeros([A.shape[0], A.shape[1]])
        L = np.zeros([A.shape[0], A.shape[1]])
        U = np.zeros([A.shape[0], A.shape[1]])
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                LU[i, j] = A[i, j] # since every element in the LU matrix is
                # derived from the respective element in the input matrix.
                count = 0
                if i <= j: # means on the upper portion, use beta formula
                    while count < i:
                        LU[i, j] = LU[i, j] - LU[i, count]*LU[count, j]
                        count = count + 1
                    U[i, j] = LU[i, j]
                else: # on the lower portion, use alpha formula
                    while count < j:
                        LU[i, j] = LU[i, j] - LU[i, count]*LU[count, j]
                        count = count + 1
                    LU[i, j] = LU[i, j] / LU[j, j] # this step is not in the
                    L[i, j] = LU[i, j]
                    # while loop because ALL alpha's need to take, even if
                    # nothing is subtracted.
                L[i, i] = 1 # diagonal values
        if sep == True:
            return L, U
        else:
            return LU

# Calculates the determinant of a matrix using LU decomposition.
# Input: A = matrix for which the determinant is to be found

def det(A):
    L, U = LU(A)
    det = 1.
    for i in np.arange(A.shape[0]):
        det = det * U[i, i] # multiply by all diagonal elements in the upper matrix
    return det

# Solves a matrix equation using LU decomposition.
# Inputs: L and U are the decomposed lower and upper triangular matrices
# respectively, and b is the vector with all the constants.
# Output: vector containing all the x values

def solve(A, b):
    if A.shape[0] == np.size(b):
        L, U = LU(A)
        y = b.copy() # so that the value of b does not change with y
        for i in np.arange(np.size(b)):
            count = 0
            while count < i:
                y[i] = y[i] - L[i, count]*y[count]
                count = count + 1
        x = y.copy()
        for j in np.arange(1, np.size(b) + 1):
            count = j
            while count > 1:
                x[-j] = x[-j] - U[-j, -count + 1]*x[-count + 1]
                count = count - 1
            x[-j] = x[-j] / U[-j, -j]
        return x
    else:
        print("Size mismatch")

# To solve for A^-1 using LU decomposition we just combine the solutions to
# vectors(1, 0, 0), (0, 1, 0) and (0, 0, 1).

def inv(A):
    I = np.identity(A.shape[0])
    inv = np.zeros([A.shape[0], A.shape[1]])
    for j in np.arange(A.shape[1]):
        x = solve(A, I[j])
        for i in np.arange(A.shape[0]):
            inv[i, j] = x[i]
    return inv