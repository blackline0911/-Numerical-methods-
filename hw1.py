import numpy as np
import matplotlib.pyplot as plt
import scipy 
from utility import *

wl = 1.5
k0 = 2*np.pi/wl
theta1 = 0
ky = k0*np.sin(theta1)
kx = k0*np.cos(theta1)
n1 = 1.0
n2 = 1.0
d = 5.0
dx = 0.5
pad = 2.5
u0 = 1
x_min = -pad
x_max = pad+d
n_layer = 1.5

x = create_x(x_min,x_max,dx)
n = np.where( x<0, n1, np.where(x>d, n2, n_layer))
n[x==0] = ( n[x==-dx]**2 + n[x==dx]**2 )/2

N = len(x)-2   # vector dimension
M = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i==j:
            M[i,j] = -2/dx**2 + n[j]**2*k0**2-ky**2
        if j==i+1 and (not i==N-1):
            M[i,j] = 1/dx**2
        if j==i-1 and (not i==0):
            M[i,j] = 1/dx**2

for p in range(1,N):
        M[p,:] = M[p-1,:] * (-M[p,p-1]/M[p-1,p-1]) + M[p,:]

U = np.zeros((N,1))
F = np.zeros((N,1))
U_start = (2*1j*kx*n[0]*u0 + 2/dx*U[1] - 1/2/dx*U[2])/( 1j*kx*n[0] + 3/2/dx)
U_end = ( 2/dx*U[-1] - 1/2/dx*U[-2] )/( 1j*kx*n[-1] + 3/2/dx )
F[0] = -1/dx**2 * U_start
F[N-1] = -1/dx**2 * U_end

np.savetxt("M.txt", M, fmt="%.8f", delimiter="\t")

