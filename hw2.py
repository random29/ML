import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg  as lalg
from numpy import random

x=np.linspace(0, 1, 1000).reshape(1000,1)
y=20*np.sin(2*np.pi*3*x)+100*np.exp(x)
e=10*np.random.rand(1000).reshape(1000,1)
t=y+e
mix = np.arange(1000)
np.random.shuffle(mix)
tr = mix[0 : 600]
vald = mix[600 : 800]
test = mix[800 :]


def matr(x, n):
    z=np.ones((x.size, n))
    for i in range(x.size):
        for j in range (n):
            z[i][j]=x[i]**j
    return z         

edm=np.ones(10)
er=np.zeros(10)
k=np.linspace(0, 10, 10)
for i in range(10):
    z=matr(x, i)
    w=((lalg.inv(z.T.dot(z))).dot(z.T)).dot(t)
    pr=z.dot(w)
    for j in range(t.size):
        er[i]+=(t[j]-pr[j])**2
    er[i]=0.5 * er[i]
    
    
plt.figure()
plt.plot(x[tr], t[tr], '.r')
plt.plot(x[vald], t[vald], '.g')
plt.plot(x[test], t[test], '.b')
plt.show()


#plt.title('Data')
#plt.show()
#plt.figure()
#plt.plot(x[tr], t[tr], '.g')
#plt.plot(x, pr, 'r')
#plt.show()
#plt.figure()
#plt.plot(k, er, 'b')
#plt.show()
