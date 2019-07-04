import numpy as np
import pylab as pl
import random as rd
import time

# Part a
# First we use the 'random' package
# Single seed used for generating a sequence of random numbers
seed = 1
rd.seed(seed)

num_sample = 10**5
# Count the number of numbers generated
count = 0

# Contains the numbers generated, starting the seed
rannumrd = np.array([seed])

while count < num_sample:
    rannumrd = np.append(rannumrd, rd.random())
    count = count + 1

pl.figure()
pl.hist(rannumrd, bins = np.linspace(0., 1., 100)) # 100 bins between 0 and 1
pl.xlabel("x")
pl.ylabel("Number of counts in bin")
pl.title("Distribution of 1e5 random numbers using 'random' package")

# Compare with 'numpy.random.uniform' for uniform deviation
# numpy.seed does not allow seed dtype = float, must be integers
# Also all numerical inputs below must be integers
np.random.seed(seed)

rannumnp = np.array([seed])
rannumnp = np.append(rannumnp, np.random.uniform(low = 0, high = 1, size = num_sample))

pl.figure()
pl.hist(rannumnp, bins = np.linspace(0., 1., 100)) # 100 bins
pl.xlabel("x")
pl.ylabel("Number of counts in bin")
pl.title("Distribution of 1e5 random numbers using 'numpy.random.uniform' package")

# Conclusion: random.random is a better method to use

# Part b - transformation method
# Transforming from uniform P(x) into the PDF given, integrating the PDF between 
# 0 and y and equating to x,
# the relation between x and y was found to be y = cos^-1 (1 - 2*x)

# Using x values computed from 'random' package in part a, compute corresponding y
start_time_trans = time.clock() # Records the start time
y = np.arccos(1-2*rannumrd)
time_trans = time.clock() - start_time_trans # Records the end time and subtract from
# start to check running time for this method

# Plotting sine function to check
check = np.linspace(0., np.pi, 500) # Used for plotting comparison function and predicted PDF
pl.figure()
(count, bins, patches) = pl.hist(y, bins = np.linspace(0., np.pi, 100),
normed=True, label = "Histogram of sinusoidal distribution")
# 100 bins between 0 and pi
pl.plot(check, 0.5*np.sin(check), 'r-', label = "Predicted PDF")
pl.xlabel("y")
pl.ylabel("Normalised PDF")
pl.legend()
pl.title("Distribution of 1e5 random numbers using a sinusoidal distribution")

# Part c - rejection method
# PDF (y) takes the values of the sin function in part b and substitute into
# the new PDF given.
PDF = 2./np.pi*(np.sin(y))**2.
comp = np.linspace(0., np.pi, 10**5.+1.)
PDFcomp = 2./np.pi*(np.sin(y))

# Accept = array containing all accepted values
accept = np.array([])
# Counters to keep track of when to stop iterating
num_accept = 0.
num_trials = 0.

start_time_rej = time.clock() # Records the start time
while num_accept < num_sample:
    indexP = rd.randrange(PDF.size)
    compval = rd.uniform(0, PDFcomp[indexP])
    if PDF[indexP] >= compval:
        accept = np.append(accept, y[indexP])
        num_accept += 1.
    num_trials += 1.
time_rej = time.clock() - start_time_rej # Records the end time and subtract from
# start to check running time for this method

pl.figure()
(count2, bins2, patches2) = pl.hist(accept, bins = np.linspace(0., np.pi, 100),
normed=True, label = "Histogram of sine squared distribution") # 100 bins between 0 and pi
pl.plot(check, 2./np.pi*(np.sin(check))**2, 'r-', label = "Predicted PDF")
pl.plot(check, 2./np.pi*(np.sin(check)), 'k-', label = "Comparison function")
pl.legend()
pl.xlabel("y")
pl.ylabel("Normalised PDF")
pl.title("Distribution of 1e5 random numbers using a sine squared distribution")

# Check measuring times
print(time_trans, time_rej, time_trans/time_rej)