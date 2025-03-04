from scipy import *
import numpy as np
import matplotlib.pyplot as plt
from utility import *
from math import *

# note x does not include boundary
N = 101
x_min = -0.1
x_max = 0.1
dx = (x_max-x_min)/(N-1)
x = np.zeros(N)
for i in range(N):
    x[i] = x_min + dx*i

# def f(x):
#     return x**2



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

# boundary condition
f= (10**5)*np.sign(x) 
ploting(x,f,'x','f',filename='f')
U_start = -np.arcsinh(-10**5) 
U_end = -np.arcsinh(10**5) 
# suppose ND = NA

Up = np.array([])
for i in range(len(x)):
    Up = np.append(Up,-np.arcsinh(f[i]))

ploting(x,Up,'x','Up_initital guess',filename='Up')
# print(Up)

iter = 20
for j in range(iter):

    # building dF/dU
    D = np.zeros((N,N))
    for r in range(N):
        # print(r)
        for c in range(N):
            # print(c)
            if (r==c ):
                D[r,c] = -2/dx**2 - np.cosh(Up[r])
            if (r==(c-1) ):
                D[r,c] = 1/dx**2
            if (r==(c+1) and (not r==0) and (not r==N-1) ):
                D[r,c] = 1/dx**2
    D = (-D)

    # Gaussian Elimination
    for p in range(1,N):
        D[p,:] = D[p-1,:] * (-D[p,p-1]/D[p-1,p-1]) + D[p,:]
    
    F = np.zeros(N)
    F[0] = ( 1/dx**2)*(Up[1]-2*Up[0]+U_start)-np.sinh(Up[0])-f[0]
    F[N-1] = ( 1/dx**2)*(U_end-2*Up[N-1]+Up[N-2])-np.sinh(Up[N-1])-f[N-1]
    for i in range(2,N-1):
        F[i] = 1/dx**2 * ( Up[i+1]-2*Up[i]+Up[i-1] ) - np.sinh(Up[i]) - f[i]

    # solving equation
    dU = np.zeros(N)
    dU[N-1] = F[N-1] / D[N-1,N-1]
    for i in range(1,N):
        dU[N-i-1] = 1/D[N-i-1,N-i-1] * (F[N-i-1] - D[N-i-1,N-i]*dU[N-i])
    
    # update Up
    Up = Up+dU
    np.savetxt("D.txt", D, fmt="%.8f", delimiter="\t")

ploting(x,Up,'x',"Up","Up_renew.png")
# np.savetxt("D.txt", D, fmt="%.8f", delimiter="\t")

