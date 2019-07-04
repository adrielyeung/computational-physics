# coding: utf-8

# In[3]:

import numpy as np
from matplotlib import pyplot as plt


# In[4]:

# Set up basic step sizes
N = 50
deltat = 1.0/N # increment from 0 to 1


# In[5]:

# the DFT function of p
def ftilde(p):
    sum = 0.0j
    for n in range(N):
        sum += f(n*deltat)*np.exp(1j*2.0*np.pi*p*n/N)   
    return sum


# In[6]:

# the inverse DFT
def finverse(n):
    sum = 0.0j
    for p in range(0, N):
        sum += ftilde(p)*np.exp(-1j*2.0*np.pi*p*n/N)
    return sum/N


# In[7]:

# "Simple" function 
def f(x):
    #val = -100*x**2+100*x**4
    val = -100*x**2+100*x**4+10*np.sin(x*100) # high-freq complication
    if(x>0.4 and x<0.6):
        return val*10*abs(x-0.5) # minor blip in the middle
        #return val
    return val


# In[8]:

# plot f(t) with N points
tarray = np.zeros(N)
farray = np.zeros(N)
for i in range(0, N):
    tarray[i] = i*deltat
    farray[i] = f(i*deltat)
plt.clf()    
plt.plot(tarray,farray,'.-',label="Discrete f(t) with "+str(N)+" points",alpha=0.2)
plt.legend()
plt.show()


# In[9]:

# plot ftilde (both Real and Imaginary parts)
parray = np.arange(0,N)
ftildearrayR = np.zeros(parray.size)
for p in range(0, parray.size):
    ftildearrayR[p] = np.real(ftilde(p))
ftildearrayI = np.zeros(parray.size)
for p in range(0, parray.size):
    ftildearrayI[p] = np.imag(ftilde(p))


# In[10]:

plt.clf()
plt.plot(parray, ftildearrayR,'.-',label="ftilde Real",
         alpha=0.2)
plt.legend()
plt.show()


# In[11]:

plt.plot(parray, ftildearrayI,'.--',label="ftilde Imag",
         alpha=0.6)
plt.legend()
plt.show()


# In[12]:

tarray = np.zeros(N)
farrayR = np.zeros(N)
farrayI = np.zeros(N)
for i in range(0, N):
    tarray[i] = i*deltat
    farrayR[i] = np.real(finverse(i))
    farrayI[i] = np.imag(finverse(i))


# In[13]:

plt.clf()    
plt.plot(tarray,farrayR,'.-',label="fInvReal",alpha=0.4)
plt.legend()
plt.show()


# In[14]:

plt.plot(tarray,farrayI,'.-',label="fInvImag",alpha=0.4)
plt.legend()
plt.show()


# In[ ]:




# In[ ]:



