from scipy import *
import numpy as np
import matplotlib.pyplot as plt
from utility import *

N = 101
x_min = -10
x_max = 10
dx = (x_max-x_min)/(N-1)
x = np.zeros(N)
for i in range(N):
    x[i] = x_min + dx*i

def f(x):
    return x**2


print(x)


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
print(a)

