from utility import *
import numpy as np
from math import *
from numpy.linalg import inv
import matplotlib.pyplot as plt

dz = 0.05
dt = 0.05
Nt = 500+2
nz = 200

Ln = np.zeros((Nt,1)) + 1j*np.zeros((Nt,1))
t = np.linspace(-dt*Nt/2,dt*Nt/2,Nt)
z = np.linspace(0,dz*nz,nz)
u0 = 1
NL = 1.0
U =  u0*1/np.cosh(t)*np.exp(1j*z[0]/2)
U.reshape(Nt,1)
def B(U,dz,dt,Ln):
    B1 = np.zeros((Nt,1)) + 1j*np.zeros((Nt,1))
    B1[0]  = U[0] - dz/2*Ln[0]*U[0] + 1j*dz/(4*dt**2)*(U[1]-2*U[0]+0) + NL*1j*dz/2*np.conj(U[0])*U[0]*U[0]
    B1[-1] = U[-1] - dz/2*Ln[-1]*U[-1] + 1j*dz/(4*dt**2)*(0-2*U[-1]+U[-2]) + NL*1j*dz/2*np.conj(U[-1])*U[-1]*U[-1]
    for i in range(1,len(U)-1):
        B1[i] = U[i] - dz/2*Ln[i]*U[i] + 1j*dz/(4*dt**2)*(U[i+1]-2*U[i]+U[i-1]) + NL*1j*dz/2*np.conj(U[i])*U[i]*U[i]
    return B1

def update(U0,U1,Ln,dt,dz,Nt):
    B0 = B(U0,dz,dt,Ln)
    M = np.zeros((Nt,Nt)) + 1j*np.zeros((Nt,Nt))
    M[0,0] = 1 + dz/2*Ln[0] - 1j*dz/(4*dt**2)*(-2) - NL*1j*dz/2*np.conj(U1[0])*U1[0]
    M[0,1] = (-1j*dz/(4*dt**2))
    M[Nt-1,Nt-1] = 1 + dz/2*Ln[Nt-1] - 1j*dz/(4*dt**2)*(-2) - NL*1j*dz/2*np.conj(U1[Nt-1])*U1[Nt-1]
    M[Nt-1,Nt-2] = (-1j*dz/(4*dt**2))

    for i in range(1,Nt-1):
        for j in range(Nt):
            if i==j:
                M[i,j] = 1 + dz/2*Ln[i] - 1j*dz/(4*dt**2)*(-2) - NL*1j*dz/2*np.conj(U1[i])*U1[i]
            if i==j-1:
                M[i,j] = (-1j*dz/(4*dt**2))
            if i==j+1:
                M[i,j] = (-1j*dz/(4*dt**2))
    return np.linalg.solve(M, B0)

def step(U0,L0,dt,dz,Nt):
    U1 = update(U0,U0,L0,dt,dz,Nt)
    U1 = update(U0,U1,L0,dt,dz,Nt)
    return U1

U_record = np.zeros((Nt,nz)) + 1j*np.zeros((Nt,nz))
for i in range(nz):
    for j in range(Nt):
        U_record[j,i] = U[j]
    # plt.plot(U)
    # plt.show()
    U = step(U,Ln,dt,dz,Nt)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
T, Z = np.meshgrid(t, z)
ax.plot_surface(T,Z,np.transpose(abs(U_record)**2))
plt.show()
