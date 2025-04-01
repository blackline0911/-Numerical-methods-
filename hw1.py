import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy 
from utility import *

wl = 1.5
k0 = 2*np.pi/wl
theta1 = 0
ky = k0*np.sin(theta1)
kx = k0*np.cos(theta1)
n1 = 1.0
n2 = 1.5
d = 5.0
dx = 0.001
pad = 2.5
u0 = 1
x_min = -pad
x_max = pad+d
n_layer = 1.5

x = create_x(x_min,x_max,dx)
N = len(x)   # vector dimension

def n_function(n1,n2,n_medium,x,d):
    if np.isinstance(x,np.ndarray):
        return np.where(x>d,n2,np.where(x<0,n1,n_medium))
    else:
        if x>d:
            return n2
        if x<0:
            return n1
        if x<=d and x>=0:
            return n_medium

n = create_n(n_function,n1,n2,n_layer,x,d)

M = np.zeros((N,N)) + 1j*np.zeros((N,N))
for i in range(1,N-1):
    for j in range(N):
        if i==j:
            M[i,j] = -2/dx**2 + n[j]**2*k0**2-ky**2
        if j==i+1 :
            M[i,j] = 1/dx**2
        if j==i-1 :
            M[i,j] = 1/dx**2

M[0,0] = 1/2 + 3/(2*dx*2*1j*kx*n[0])
M[0,1] = -4/(2*dx*2*1j*kx*n[0])
M[0,2] = 1/(2*dx*2*1j*kx*n[0])
M[N-1,N-3] = 1/(2*dx*2*1j*kx*n[N-1])
M[N-1,N-2] = -4/(2*dx*2*1j*kx*n[N-1])
M[N-1,N-1] = 1/2 + 3/(2*dx*2*1j*kx*n[N-1])

U = np.zeros((N,1)) + 1j*np.zeros((N,1))
F = np.zeros((N,1)) + 1j*np.zeros((N,1))
F[0] = u0


U = np.linalg.solve(M, F)

# np.savetxt("M.txt", M, fmt="%.8f", delimiter="\t")
np.savetxt("n.txt", n, fmt="%.8f", delimiter="\t")
# np.savetxt("F.txt", F, fmt="%.8f", delimiter="\t")
np.savetxt("U.txt", U, fmt="%.8f", delimiter="\t")

kx_pround = kx*n
du_dx = np.zeros((N,1)) + 1j* np.zeros((N,1))
for i in range(N):
    if i==0:
        du_dx[i] = 1/2/dx*(-3*U[i]+4*U[i+1]-U[i+2])
    elif i==1:
        du_dx[i] = 1/2/dx*(-3*U[i]+4*U[i+1]-U[i+2])
    else :
        du_dx[i] = 1/2/dx*(3*U[i]-4*U[i-1]+U[i-2])
    # elif i==N-2:
    #     du_dx[i] = 1/2/dx*(3*U[i]-4*U[i-1]+U[i-2])
    # else:
    #     du_dx[i] = 1/2/dx*(U[i+1]-U[i-1])

U_pos = U/2 - du_dx/(2*1j*kx_pround)
U_neg = U/2 + du_dx/(2*1j*kx_pround)

ploting(x,n,x_label="x (um)",title="Index",filename="index")
ploting(x,np.real(U),x_label="x (um)",title="E field_real",filename="E_real")
ploting(x,np.imag(U),x_label="x (um)",title="E field_imag",filename="E_imag")
ploting(x,U_pos,x_label="x (um)",title="U_pos",filename="U_pos")
ploting(x,abs(U_pos)**2,x_label="x (um)",title=r'$|U pos|^2$',filename="U_pos_power")
ploting(x,abs(U_neg)**2,x_label="x (um)",title=r'$|U neg|^2$',filename="U_neg_power")

