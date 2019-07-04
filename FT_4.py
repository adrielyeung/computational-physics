import numpy as np
import pylab as pl

"""
rect - rectangular function
Inputs:
val : y-value of the rectangular function.
start : time (x-value) to start sampling from.
stop : time (x-value) to stop sampling.
(REQUIRE start <= stop)
rise : where the y-value of the rectangular function starts being non-zero.
fall : where the y-value of the rectangular function returns to zero.
(REQUIRE rise <= fall)
numpts = number of sampling points.

Outputs:
t = array containing all 'numpts' sampling times within start and stop.
h = array containing all values of the rectangular function whose value equals 
'val' if the corresponding value in array t is within start and stop.
deltat = time interval between each sampling time.
"""
def rect(val, start, stop, rise, fall, numpts):
    t = np.linspace(start, stop, numpts, dtype=float)
    h = np.zeros_like(t)
    deltat = (stop - start) / (numpts-1.) # Sampling time interval
    for i in np.arange(0, len(t)):
        if t[i] >= rise and t[i] <= fall: # Within rectangle
            h[i] = val
    return t, h, deltat

"""
Gauss - Gaussian function
Inputs:
start : time (x-value) to start sampling from.
stop : time (x-value) to stop sampling.
(REQUIRE start <= stop)
numpts : number of sampling points.

Outputs:
t : array containing all 'numpts' sampling times within start and stop.
g : array containing all values of the Gaussian function given by g(t) =
1/sqrt(2*pi)*e^(-t^2/2)
deltat = time interval between each sampling time.
"""
def Gauss(start, stop, numpts):
    t = np.linspace(start, stop, numpts, dtype=float)
    deltat = (stop - start) / (numpts-1) # Sampling time interval
    g = 1./np.sqrt(2*np.pi)*np.exp(-t**2 / 2)
    return t, g, deltat

"""
conv - Convolution function
Inputs:
g : array containing the x-values of function (g).
h : array containing the x-values of function (h).
(REQUIRE sizes of g and h match)
deltat = time interval between each sampling time (this should match for both functions).

Optional inputs:
plotftg = False: whether to plot the FT form of the first function (g).
plotfth = False: whether to plot the FT form of the second function (h).
plotftcon = False: whether to plot the FT form of the convolution.

Output:
con : array containing the y-values of the convoluted function in time domain.
"""
def conv(t, g, h, deltat, plotftg = False, plotfth = False, plotftcon = False):
    hf = np.fft.fft(h)
    gf = np.fft.fft(g)
    freq = np.fft.fftfreq(h.size, d=deltat)
    if plotfth == True:
        pl.figure()
        pl.plot(freq, np.abs(hf)) # Magnitude used as the values are complex
        pl.title("Fourier transform of function 2 (h)")
    if plotftg == True:
        pl.figure()
        pl.plot(freq, np.abs(gf))
        pl.title("Fourier transform of function 1 (g)")
    conf = gf * hf
    if plotftcon == True:
        pl.figure()
        pl.plot(freq, np.abs(conf))
        pl.title("Fourier transform of the convoluted function (g*h)")
    con = np.abs(np.fft.ifft(conf))*deltat # When the FT is performed, 
    # a normalisation factor deltat was divided for both g and h. However, the
    # inverse was only done once, so only one factor of deltat has been multiplied.
    # To normalise the inverse FT we need to multiply by deltat again.
    return con
