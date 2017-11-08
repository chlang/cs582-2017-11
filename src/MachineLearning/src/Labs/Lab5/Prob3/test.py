
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib notebook')

import matplotlib.pyplot as pl
import numpy as np

p = np.array([[2,2],[2,-2],[-2,-2],[-2,2]])
n = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])

pl.plot(p[:,0],p[:,1],'ro')
pl.plot(n[:,0],n[:,1],'bx')
pl.show()


# In[2]:

def phi(x):
    if np.transpose(x).dot(x) > 4:
        k = abs(x[0] - x[1])
        return np.array([4-x[1]+k,4-x[0]+k])
    else:
        return x


# In[3]:

def phis(x):
    return np.reshape([phi(x[i]) for i in range(np.shape(x)[0])], (4,2))


# In[4]:

pp = phis(p)
nn = phis(n)

print(pp)
print(nn)


# In[5]:

pl.figure()
pl.plot(pp[:,0],pp[:,1],'ro')
pl.plot(nn[:,0],nn[:,1],'bx')
pl.plot([-1, 4], [4, -1], 'k-')
pl.show()

