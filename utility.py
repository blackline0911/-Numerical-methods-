from scipy import *
import numpy as np
import matplotlib.pyplot as plt

c=299792458e6

def ploting(x,*arg,x_label, title,filename='',figcolor='w',line_color='b',
            grid_color='g',grid_style='--',grid_alpha=0.5,leg=['']):
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
       n=0
       for i in arg:
            if (not leg==['']):
                plt.plot(x,i,label=leg[n])
                plt.legend()
            else:
                plt.plot(x,i)
            n+=1
       plt.xlabel(x_label)
       plt.title(title)
       plt.grid(color=grid_color,linestyle=grid_style, alpha=grid_alpha)
       ax = plt.gca()
       ax.set_facecolor(figcolor)
       if(filename!=''):
              plt.savefig(filename)
       plt.show()
       return




if __name__=='__main__':
    x = np.linspace(0,10,1000)
    y = x**2
    ploting(x,y,r"$\alpha$",r'$|\alpha|^2$','testing.png')