import numpy as np
from matplotlib import pyplot as plt

x0 = 100 * (np.random.rand(100)) + 300
y0 = 100 * (np.random.rand(100)) + 200 
data0 = np.column_stack((x0, y0))           
x1 = 100 * (np.random.rand(100)) + 200
y1 = 100 * (np.random.rand(100)) + 100 
data1 = np.column_stack((x1, y1))   
x2 = 100 * (np.random.rand(100)) + 100
y2 = 100 * (np.random.rand(100)) 
data2 = np.column_stack((x2, y2))   
         
data = np.vstack((data0,np.vstack((data1,data2))))
np.random.shuffle(data)


def kmeans(data, K, maxIters):
    centroids = data[np.random.choice(np.arange(len(data)), K)]
    for i in range(maxIters): 
        C = np.array([np.argmin([np.dot(xi-yk, xi-yk) for yk in centroids]) for xi in data])
        centroids = [data[C == k].mean(axis = 0) for k in range(K)]       
    return np.array(centroids) , C

def show(X, C, centroids):   
    plt.plot(X[C == 0, 0], X[C == 0, 1], '.b')
    plt.plot(X[C == 1, 0], X[C == 1, 1], '.r')
    plt.plot(X[C == 2, 0], X[C == 2, 1], '.g')


    plt.plot(centroids[:,0], centroids[:,1], '.', markersize=20) 
    plt.show()

centroids, C = kmeans(data, K = 3, maxIters=20)
show(data, C, centroids)

plt.plot(x0,y0,'.g')
plt.plot(x1,y1,'.r')
plt.plot(x2,y2,'.b')
plt.show()






















