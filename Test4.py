import FT_4 as F4
import imp
imp.reload(F4)
import pylab as pl
import numpy as np
import time

numpts = 1025. # Number of sample points
samplerange = [-10., 10.] # Sampling range in time

# Rectangular function
th, h, deltath = F4.rect(5, samplerange[0], samplerange[1], 2, 4, numpts)

pl.figure()
pl.plot(th, h)
pl.title("Rectangular function, h")

# Gaussian function
tg, g, deltatg = F4.Gauss(samplerange[0], samplerange[1], numpts)
pl.figure()
pl.plot(tg, g)
pl.title("Gaussian function, g")

# Fourier transform the Gaussian and rectangular functions - also measuring runtime
start_time = time.clock() # returns the time since started running
hf = np.fft.fft(h)
FThtime = time.clock() - start_time # difference in time is the time used for FT
gf = np.fft.fft(g)
FTgtime = time.clock() - FThtime - start_time
print(FThtime, FTgtime)

# Sampling frequencies used
freq = np.fft.fftfreq(h.size, d = deltath) # Same for g if g.size == h.size, i.e.
# number of sampling points the same.

# Plotting the Fourier transformed function of g, h and g*h in frequency domain
# Also calculating the y-values in the convoluted graph
con = F4.conv(th, g, h, deltatg, True, True, True)

# Convoluted result
pl.figure()
pl.plot(th, h, label = "Rectangular function")
pl.plot(tg, g, label = "Gaussian function")
pl.plot(th, con, label = "Convolution")
pl.xlabel("t")
pl.ylabel("Function")
pl.legend()
pl.grid()
pl.title("Convolution of a rectangular function and a Gaussian")

# Calculate the area under the curves using trapezoidal rule
harea = np.trapz(h, th)
garea = np.trapz(g, tg)
conarea = np.trapz(con, th)
print (harea, garea, conarea)
