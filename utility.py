from scipy import *
import numpy as np
import matplotlib.pyplot as plt

c=299792458e6

def ploting(x,y,x_label, title,filename='',figcolor='w',line_color='b',
            grid_color='g',grid_style='--',grid_alpha=0.5):
       """
       input argments:
              x:x
              y:data you want to plot
              x_label:x_label
              title: figure title
              figcolor: set the color in figure
              line_color:set the color of line you plot
              grid_color: set the color of grid
              grid_style: set the style of grid
              grid_alpha: set opacity of grid
              filename: set the file name of the figure you plot. 
                        If unspecified, the figure will not be saved.
       """
       plt.figure()
       plt.plot(x,y,color=line_color)
       plt.xlabel(x_label)
       plt.title(title)
       plt.grid(color=grid_color,linestyle=grid_style, alpha=grid_alpha)
       ax = plt.gca()
       ax.set_facecolor(figcolor)
       if(filename!=''):
              plt.savefig(filename)
       plt.show()
       return

def create_x(x_min,x_max,dx):
    N = int((x_max-x_min)/dx+1)
    x = np.zeros(N)
    for i in range(N):
        x[i] = i*dx+x_min
    return x
if __name__=='__main__':
    x = np.linspace(0,10,1000)
    y = x**2
    ploting(x,y,r"$\alpha$",r'$|\alpha|^2$','testing.png')