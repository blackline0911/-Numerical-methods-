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
dx = 0.002
pad = 2.5
u0 = 1
x_min = -pad
x_max = pad+d
n_layer = 1.5

x = create_x(x_min,x_max,dx)
N = len(x)   # vector dimension

n = np.zeros((N,1))
n = np.where( x<0, n1, np.where(x>d, n2, n_layer))

idx_0 = np.argmin(np.abs(x - 0))
idx_d = np.argmin(np.abs(x - d))
idx_m1 = np.argmin(np.abs(x + dx))
idx_p1 = np.argmin(np.abs(x - dx))
idx_d_m1 = np.argmin(np.abs(x - (d - dx)))
idx_d_p1 = np.argmin(np.abs(x - (d + dx)))

n[idx_0] = (n[idx_m1]**2 + n[idx_p1]**2) / 2
if (not n[idx_d_m1]==n[idx_d_p1]):
    n[idx_d] = (n[idx_d_m1]**2 + n[idx_d_p1]**2) / 2
n = n.reshape((N,1))

M = np.zeros((N,N)) + 1j*np.zeros((N,N))
for i in range(1,N-1):
    for j in range(N):
        if i==j:
            M[i,j] = -2/dx**2 + n[j]**2*k0**2-ky**2
        if j==i+1 :
            M[i,j] = 1/dx**2
        if j==i-1 :
            M[i,j] = 1/dx**2
# M[0,0] = 1/2 + 1/(2*dx*2*1j*kx*n[0])
# M[0,1] = -1/(2*dx*2*1j*kx*n[0])
# M[N-1,N-2] = 1/(2*dx*2*1j*kx*n[N-1])
# M[N-1,N-1] = 1/2 - 1/(2*dx*2*1j*kx*n[N-1])
M[0,0] = 1/2 + 3/(2*dx*2*1j*kx*n[0])
M[0,1] = -4/(2*dx*2*1j*kx*n[0])
M[0,2] = 1/(2*dx*2*1j*kx*n[0])
M[N-1,N-3] = 1/(2*dx*2*1j*kx*n[N-1])
M[N-1,N-2] = -4/(2*dx*2*1j*kx*n[N-1])
M[N-1,N-1] = 1/2 + 3/(2*dx*2*1j*kx*n[N-1])

# for p in range(1,N):
#         M[p,:] = M[p-1,:] * (-M[p,p-1]/M[p-1,p-1]) + M[p,:]

U = np.zeros((N,1)) + 1j*np.zeros((N,1))
F = np.zeros((N,1)) + 1j*np.zeros((N,1))
F[0] = u0
try:
    M_inv = inv(M)
except:
    assert False ,"\nInversion Error\n"

U = np.dot(M_inv,F)
# U[N-1] = F[N-1] / M[N-1,N-1]
# for i in range(1,N):
#     U[N-i-1] = 1/M[N-i-1,N-i-1] * (F[N-i-1] - M[N-i-1,N-i]*U[N-i])

np.savetxt("M.txt", M, fmt="%.8f", delimiter="\t")
np.savetxt("n.txt", n, fmt="%.8f", delimiter="\t")
np.savetxt("F.txt", F, fmt="%.8f", delimiter="\t")
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
print(np.shape(U_pos))
print(np.shape(U_neg))
print(np.shape(du_dx))
print(np.shape(kx_pround))
print(np.shape(n))
print((n))
print(np.shape(U))
ploting(x,n,x_label="x (um)",title="Index",filename="index")
ploting(x,kx_pround,x_label="x (um)",title="kx",filename="kx")
ploting(x,np.real(U),x_label="x (um)",title="E field_real",filename="E_real")
ploting(x,np.imag(U),x_label="x (um)",title="E field_imag",filename="E_imag")
ploting(x,U_pos,x_label="x (um)",title="U_pos",filename="U_pos")
ploting(x,abs(U_pos)**2,x_label="x (um)",title=r'$|U pos|^2$',filename="U_pos_power")
ploting(x,abs(U_neg)**2,x_label="x (um)",title=r'$|U neg|^2$',filename="U_neg_power")

