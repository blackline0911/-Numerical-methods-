from utility import *
import numpy as np
from math import *
import matplotlib.pyplot as plt

# Setting physical simulation arguments
dt = 0.01           # minimum time resolution
Num = 100
dT = dt*Num         # Round trip resolution
n = 512*Num         #Time array length
nT = 1000           #Round Trip array length
g0 = 4.0            #Gain Coefficient
Es = 0.5            #Saturation absorber Rate
A0 = 1.0            # Input Optical Field Amplitude
iter = 2            #iteration number of Split Step
kc = 0.0+1.0*1j     #Kerr effect Coefficient
dc = 0.05+0.5*1j     #Dispersion Coefficient
dw = 2*np.pi/(n*dt) # Minimum Radian Frequency Step 
L0 = 1.0            #Amplitude Loss Coefficient
wM = 2*np.pi/100
M = 0.8


# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Creating t and T array

def creat_array(N,dx,start = 0):
    array = np.zeros(abs(N))
    if N>0:
        step = 1
    else:
        step = -1
    for i in range(0,N,step):
        array[abs(i)] = i*dx + start*dx
    return array
t = creat_array(n,dt)-n*dt/2   # Time center at t=0
T = creat_array(nT,dT)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Setting dispersion phase shift

u0 = A0*(1/np.cosh(t)) + 0.0*1j                         #Initial Solution at t=0 
wpos = np.exp(-dc*dT*creat_array(int(n/2),dw)**2)
wneg = np.exp(-dc*dT*creat_array(-int(n/2),dw,start = -1)**2)
pshift = np.append(np.flip(wneg),wpos)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Writing split-step function
def stepfft(u0:np.ndarray,
            pshift:np.ndarray,
            dT:float,dt:float,
            n:float,g0:float,Es:float,
            kc:np.complex128,L0:float,iter:int):
    u1 = u0

    # Process integration of saturate absorber
    Ep = 0
    Ep = np.sum(abs(u1)**2,dtype=np.complex128) *dt
    g = g0/(1+Ep/Es)

    # Process FFT
    B = (g + kc*abs(u1)**2 - L0 + 1j*M*np.cos(wM*t))
    F = 1/n*np.fft.fft(u1*(1 + B*dT/2 ))
    F = np.fft.fftshift(F)
    F = pshift * F
    F_temp = np.fft.ifftshift(F)
    u1 = n*np.fft.ifft(F_temp)
    utemp = u1

    # Processing Split-step
    j = 0
    while (j<iter):
        # 需要重新計算U(T+dT)**2的積分
        Ep = 0
        Ep = np.sum(abs(u1)**2,dtype=np.complex128) *dt
        g = g0/(1+Ep/Es)
        u1 = utemp / (1-dT/2* ( g - L0 + kc*abs(u1)**2 + 1j*M*np.cos(wM*t)) )
        j += 1
    return u1 

U_record = np.zeros((nT,n)) + np.zeros((nT,n)) *1j
U_record[0,:] = u0 
for r in range(1,nT):
    u0 = stepfft(u0,pshift,dT,dt,n,g0,Es,kc,L0,iter)
    U_record[r,:] = u0

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
Z_big, T_big = np.meshgrid(T, t)
ax.plot_surface(T_big,Z_big,np.transpose(abs(U_record)**2))
plt.xlabel("time")
plt.ylabel("Round Trip (T)")
plt.show()

ploting(T,abs(U_record[:,int(n/2-1)])**2,x_label=":Round Trip (T)",title="|U|^2 at t=0")



