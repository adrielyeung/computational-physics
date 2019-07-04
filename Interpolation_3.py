import numpy as np
import Matrix_2 as M2

"""
lin - Linear interpolation
Input: inp = a matrix containing all the x and y values (columns 0 and 1),
sorted in ascending order in x.
Output: array f whose elements i provide the y values for any x between x_i and
x_i+1 using a linear interpolation method.
"""

def lin(inp, stepsize = 0.01):
    x = inp[:, 0]
    y = inp[:, 1]
    f = np.array([])
    xsamp = np.array([])
    for i in np.arange(len(x)-1):
        xnew = np.arange(start = x[i], stop = x[i+1], step = stepsize)
        xsamp = np.append(xsamp, xnew)
        # Samples points at step size 0.01 between two adjacent data points
        # Note: last point of each section is not included, but it does not
        # matter because it is automatically the first point of the next,
        # EXCEPT the very last point
        for j in np.arange(len(xnew)):
            f = np.append(f, (((x[i+1] - xsamp[j + len(xsamp) - len(xnew)])*y[i]
            + (xsamp[j + len(xsamp) - len(xnew)] - x[i])*y[i+1]) / (x[i+1]-x[i])))
    xsamp = np.append(xsamp, x[-1]) # Adds in the last point because it is not
    # included in any interpolation
    f = np.append(f, y[-1])
    return xsamp, f

"""
cubspline - Cublic spline
Input: inp = a matrix containing all the x and y values (columns 0 and 1),
sorted in ascending order in x.
Output: array f whose elements i provide the y values for any x between x_i and
x_i+1 using a cubic spline method.
"""

def cubspline(inp, stepsize = 0.01, showcoeff = False, showb = False): # show the coefficient matrix
# or the array of constants
    x = inp[:, 0]
    y = inp[:, 1]
    fdashdash = np.array([0.]) # first second derivative = 0
    # Calculate the array contatining all the second derivatives of f at each
    # data point.
    coeff = np.zeros([len(x)-2, len(x)-2]) # Contains all the coefficients in
    # the fundamental equation in finding f". If we assume f"_0 = f"_n = 0
    # (natural spline), then this leaves us with n-1 unknowns in n-1 eqs.
    b = np.zeros(len(x)-2) # Contains the constant values
    for i in np.arange(len(x)-2):
        coeff[i, i] = (x[i+2] - x[i]) / 3 # Valid for all points      
        if i > 0: # First point (i = 0) only has 2 coefficients as f"_0 = 0
            coeff[i, i-1] = (x[i+1] - x[i]) / 6
        if i < len(x)-3: # Last point (i = len(x) - 3) only has 2 coefficients as f"_n = 0
            coeff[i, i+1] = (x[i+2] - x[i+1]) / 6
        b[i] = (y[i+2] - y[i+1])/(x[i+2] - x[i+1]) - (y[i+1] -y[i])/(x[i+1] - x[i])
    if showcoeff == True:
        print(coeff)
    if showb == True:
        print(b)
    fdashdash = np.append(fdashdash, M2.solve(coeff, b))
    fdashdash = np.append(fdashdash, 0.) # last second derivative = 0

    # Now we solve for the cubic spline function f(x).
    f = np.array([])
    xsamp = np.array([])
    for i in np.arange(len(x)-1):
        xnew = np.arange(start = x[i], stop = x[i+1], step = stepsize)
        #xnew = np.linspace(x[i], x[i+1], 500)
        xsamp = np.append(xsamp, xnew)
        for j in np.arange(len(xnew)):
            A = (x[i+1] - xnew[j])/(x[i+1] - x[i])
            B = (xnew[j] - x[i])/(x[i+1] - x[i])
            C = 1./6.*(A**3-A)*((x[i+1] - x[i])**2)
            D = 1./6.*(B**3-B)*((x[i+1] - x[i])**2)
            f = np.append(f, A*y[i] + B*y[i+1] + C*fdashdash[i] + D*fdashdash[i+1])
    xsamp = np.append(xsamp, x[-1]) # Adds in the last point because it is not
    # included in any interpolation
    f = np.append(f, y[-1])
    return xsamp, f