import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy 
from utility import *

def N(n1,n2):
    return [ [(n2+n1)/2/n2, (n2-n1)/2/n2] ,[(n2-n1)/2/n2, (n2+n1)/2/n2]]

def averaging(f):
    return sum(f)/len(f)

def create_x(x_min,x_max,dx):
    N = int((x_max-x_min)/dx+1)
    x = np.zeros(N)
    for i in range(N):
        x[i] = i*dx+x_min
    return x

def create_n(n_function,x,x_min,x_max,*args,threshold=1):
    N = len(x)
    n = np.zeros((len(x),1))
    for i in range(N):
        n[i] = n_function(x[i],x_min,x_max,*args)
        if ((n[i]-n[i-1]>=threshold) and (not i==0)):
            n[i] = ( (n[i]**2+n[i-1]**2)/2 )**0.5
    n = n.reshape((N,1))
    return n
def n_function(x,x_min,x_max,n1,n2,n_medium,pad):
    x1 = (x_min+pad)
    x2 = (x_max-pad)
    if isinstance(x,np.ndarray):
        return np.where(x>x2,n2,np.where(x<x1,n1,n_medium))
    else:
        if x>x2:
            return n2
        if x< x1:
            return n1
        if x<=x2 and x>=x1:
            return n_medium
        
def solve(n_function,wl,theta,x_min,x_max,dx,u0,*args):
    k0 = 2*np.pi/wl
    ky = k0*np.sin(theta)
    kx = k0*np.cos(theta)

    x = create_x(x_min,x_max,dx)
    N = len(x)   # vector dimension

    n = create_n(n_function,x,x_min,x_max,*args)

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

    return U, U_pos, U_neg, n, x 

def calculate_RT(n1,n2,U_pos,U_neg,u0,x_min,x_max,pad,dx):
    R = averaging((abs(U_neg[0:int(np.floor(abs(x_min+pad-x_min)/dx))]))**2)/u0**2
    T = n2*averaging(abs(U_pos[int(np.ceil(abs(x_max-pad+pad)/dx)) : len(x)-1])**2)/(n1*u0**2)
    return R[0],T[0]


if __name__=='__main__':
    wl = 1.5
    theta1 = 0
    n1 = 1.0
    n2 = 1.0
    # n2 = 1.0
    d = 5.0
    dx = 0.0025
    pad = d/2+d/4
    u0 = 1
    x_min = -pad
    x_max = d/2+pad
    n_layer = 3.0
    U, U_pos, U_neg, n ,x = solve(n_function,wl,
                                theta1,
                                x_min,x_max,dx,
                                u0,
                                n1,
                                n2,
                                n_layer,
                                pad,    
                                )
    ploting(x,n,x_label="x (um)",title="Index",filename="index")
    ploting(x,np.real(U),x_label="x (um)",title="E field_real",filename="E_real")
    ploting(x,np.imag(U),x_label="x (um)",title="E field_imag",filename="E_imag")
    ploting(x,U_pos,x_label="x (um)",title="U_pos",filename="U_pos")
    ploting(x,U_neg,x_label="x (um)",title="U_neg",filename="U_neg")
    ploting(x,abs(U_pos)**2,x_label="x (um)",title=r'$|U pos|^2$',filename="U_pos_power")
    ploting(x,abs(U_neg)**2,x_label="x (um)",title=r'$|U neg|^2$',filename="U_neg_power")

    # np.savetxt("n.txt", n, fmt="%.8f", delimiter="\t")
    # np.savetxt("U.txt", U, fmt="%.8f", delimiter="\t")
    
    R,T = calculate_RT(n1,n2,U_pos,U_neg,u0,x_min,x_max,pad,dx)
    print("R = ", R)
    print("T = ", T)
    
    wl_nums = np.linspace(1/1.4,1/1.7,100) # 1/um
    Rs = np.zeros(len(wl_nums))
    Ts = np.zeros(len(wl_nums))
    Ans_R = np.zeros(len(wl_nums))
    Ans_T = np.zeros(len(wl_nums))
    i=0
    
    for wl_num in wl_nums:
        U, U_pos, U_neg, n ,x = solve(n_function,1/wl_num,
                                theta1,
                                x_min,x_max,dx,
                                u0,
                                n1,
                                n2,
                                n_layer,
                                pad,
                                )
        R,T = calculate_RT(n1,n2,U_pos,U_neg,u0,x_min,x_max,pad,dx)
        Rs[i] = R
        Ts[i] = T
        # analytic solution
        N1 = N(n1,n_layer)
        N3 = N(n_layer,n2)
        N2 = [ [np.exp(-1j*2*np.pi*wl_num*d/2*n_layer),0], [0, np.exp(1j*2*np.pi*wl_num*d/2*n_layer)] ] 
        N_total = np.dot(np.dot(N3,N2),N1)
        r = -N_total[1,0]/N_total[1,1]*u0
        t = N_total[0,0]*u0 + N_total[0,1]*r
        Ans_R[i] = abs(r)**2
        Ans_T[i] = abs(t)**2
        i+=1

    
    # ploting(wl_nums,Rs,Ans_R,Ts,Ans_T,x_label="wave number (1/um)",title="Transfer function",filename="T",leg=['R(FDM)','Ans R','T(FDM)','Ans T'])
    ploting(wl_nums,Rs,Ts,x_label="wave number (1/um)",title="Transfer function",filename="T",leg=['R(FDM)','T(FDM)'])
  
    
    