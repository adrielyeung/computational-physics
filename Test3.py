import numpy as np
import Interpolation_3 as I3
import imp
imp.reload(I3)
import pylab as pl

inp = np.matrix([[-2.1, 0.012155],[-1.45, 0.122151],[-1.3, 0.184520],[-0.2,0.960789],
                 [0.1, 0.990050],[0.15, 0.977751],[0.8, 0.527292],[1.1, 0.298197],
                 [1.5, 0.105399],[2.8, 3.936690e-4],[3.8, 5.355348e-7]], dtype=float)

stepsize = 0.005

xlin, ylin = I3.lin(inp, stepsize)
pl.figure()
pl.plot(xlin, ylin, 'b-', label="Linear")
pl.grid()

# Returns expected shape.

xcub, ycub = I3.cubspline(inp, stepsize)

pl.plot(xcub, ycub, 'r-', label="Cubic spline")
pl.title("Interpolation using linear and cubic spline interpolation methods")
pl.xlabel("x")
pl.ylabel("y")
pl.legend()