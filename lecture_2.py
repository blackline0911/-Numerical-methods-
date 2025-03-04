from scipy import *
import numpy as np
import matplotlib.pyplot as plt
from utility import *
from math import *

# note x does not include boundary
N = 11
x_min = -5
x_max = 5
dx = (x_max-x_min)/(N-1)
x = np.zeros(N)
for i in range(N):
    x[i] = x_min + dx*i

def f(x):
    return x**2



A  = [ [1,1,1,1,1],
       [0,-1,-2,1,2],
       [0,0.5,2,0.5,2],
       [0,-1/6,-8/6,1/6,8/6],
       [0,1/24,16/24,1/24,16/24]]

b = [[1],
     [0],
     [0],
     [0],
     [0]]
a = np.dot(np.linalg.inv(A),b)



U_start = -asinh(0) 
U_end = -asinh(0) 
# suppose ND = NA

Up = np.array([])
for i in x:
    Up = np.append(Up,-np.arcsinh(i))

ploting(x,Up,'x','Up_initital guess',filename='Up')
print(Up)

iter = 1
for j in range(iter):

    # building dF/dU
    D = np.zeros((N,N))
    for r in range(N):
        print(r)
        for c in range(N):
            
            print(c)
            if (r==c ):
                D[r,c] = -2/dx**2 - np.sinh(Up[r])
            if (r==(c-1) ):
                D[r,c] = 1/dx**2
            if (r==(c+1) and (not r==0) and (not r==N-1) ):
                D[r,c] = 1/dx**2

    # Gaussian Elimination
    for p in range(1,N):
        D[p,:] = D[p-1,:] * (-D[p,p-1]/D[p-1,p-1]) + D[p,:]
    
    # solving equation
    dU = np.zeros(N)
    dU[N-1] = Up[N-1] / D[N-1,N-1]
    for i in range(1,N):
        s = 0
        for k in range(i):
            s += D[N-i-1,N-i-k]*dU[N-i-k]
        print(s)
        dU[N-i-1] = 1/D[N-2,N-2] * (Up[N-2] -s )
    
    # update Up
    Up += dU
    print(Up)
    np.savetxt("D.txt", D, fmt="%.8f", delimiter="\t")


# np.savetxt("D.txt", D, fmt="%.8f", delimiter="\t")

