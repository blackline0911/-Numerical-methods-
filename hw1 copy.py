import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy 
from utility import *
import hw1

def n_function(x,x_min,x_max,n1,n2,n_medium,pad):
    x1 = (x_min+pad)
    x2 = (x_max-pad)
    if isinstance(x,np.ndarray):
        return np.where(x>x2,n2,np.where(x<x1,n1,1+1.25*(1-np.cos(np.pi*x/( (x_max-x_min)-2*pad) ) ) ) )
    else:
        if x>x2:
            return n2
        if x< x1:
            return n1
        if x<=x2 and x>=x1:
            return 1+1.25*(1-np.cos(np.pi*x/( (x_max-x_min)-2*pad) ) )
     
# def calculate_RT(n1,n2,U_pos,U_neg,x_min,x_max,pad):
    # R = averaging((abs(U_neg[0:int(np.floor(abs(x_min+pad-x_min)/dx))]))**2)/u0**2
    # T = n2*averaging(abs(U_pos[int(np.ceil(abs(x_max-pad+pad)/dx)) : len(x)-1])**2)/(n1*u0**2)
    # return R[0],T[0]

  
if __name__=='__main__':
    wl = 1.5
    theta1 = 0
    n1 = 1.0
    n2 = 3.5
    # n2 = 1.0
    d = 5.0
    dx = 0.0025
    pad = d/2
    u0 = 1
    x_min = -pad
    x_max = d+pad
    n_layer = 3.0
    U, U_pos, U_neg, n ,x = hw1.solve(n_function,
                                wl,
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
    ploting(x,abs(U_pos)**2,x_label="x (um)",title=r'$|U pos|^2$',filename="U_pos_power")
    ploting(x,abs(U_neg)**2,x_label="x (um)",title=r'$|U neg|^2$',filename="U_neg_power")

    # np.savetxt("n.txt", n, fmt="%.8f", delimiter="\t")
    # np.savetxt("U.txt", U, fmt="%.8f", delimiter="\t")
    
    R,T = hw1.calculate_RT(n1,n2,U_pos,U_neg,u0,x_min,x_max,pad,dx)
    print("R = ", R)
    print("T = ", T)
    
    wl_nums = np.linspace(1/1.4,1/1.7,100) # 1/um
    Rs = np.zeros(len(wl_nums))
    Ts = np.zeros(len(wl_nums))
    Ans_R = np.zeros(len(wl_nums))
    Ans_T = np.zeros(len(wl_nums))
    i=0
    
    for wl_num in wl_nums:
        U, U_pos, U_neg, n ,x = hw1.solve(1/wl_num,
                                theta1,
                                x_min,x_max,dx,
                                u0,
                                n1,
                                n2,
                                n_layer,
                                pad,
                                )
        R,T = hw1.calculate_RT(n1,n2,U_pos,U_neg,u0,x_min,x_max,pad,dx)
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
  
    
    