# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:33:54 2018

@author: xavier.mouy
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import mpl_toolkits.mplot3d
import numpy as np
import pickle
import pandas as pd
import localizationlib as loclib

def getArrayUncertainties(R,radius,spacing,V,NoiseSTD,contoursValues):
    
    # Virtual sources coordinates -> Cube of points (Cartesian coordinates)
    vec = np.arange(-radius, radius+spacing, spacing)
    X, Y, Z = np.meshgrid(vec, vec, vec, indexing='ij')
    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
    # find location of slice
    ind = np.argmin(abs(vec))
    sliceValue = vec[ind]
    # Nb of receivers    
    nReceivers = R.shape[0]
    
    # Variance of TDOA measurement errors
    NoiseVariance = NoiseSTD**2
    
    # Define receiver pairs for TDOAs
    Rpairs = loclib.defineReceiverPairs (nReceivers)

    # Get list of Jacobian matrice for each source
    J = loclib.defineJacobian(R, S, V, Rpairs) 
    
    # Calculates localization uncertainty for each source
    Uncertainties = loclib.getUncertainties(J, NoiseVariance)   

    # Plots unceratinties of optimized array
    loclib.plotArrayUncertainties(R, S, Uncertainties)

    # PLot hydrophone locations
    f0 = plt.figure()
    ax0 = f0.add_subplot(111, projection='3d')
    ax0.scatter(R['x'], R['y'], R['z'], s=30, c='black')
    ax0.set_xlabel('X (m)', labelpad=10)
    ax0.set_ylabel('Y (m)', labelpad=10)
    ax0.set_zlabel('Z (m)', labelpad=10)
    plt.show()

    # Define and plot plane slices
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False,figsize=(16, 5))
    ## XY plane
    XY=np.zeros([len(vec),len(vec)])
    for i in range(len(vec)):
        for jj in range(len(vec)):
            idx = S.index[(S['x']==vec[i]) & (S['y']==vec[jj]) & (S['z']==sliceValue)][0]
            XY[i, jj] = Uncertainties['rms'][idx]              
    CS_XY = ax1.contour(vec, vec, XY, levels=contoursValues, colors=['k'])
    # Receivers
    ax1.plot(R['x'], R['y'], 'go')
    ax1.set_xlabel('X(m)')
    ax1.set_ylabel('Y(m)')
    ax1.grid(True)
    im=ax1.imshow(XY, interpolation='bilinear', origin='lower',
                    cmap=cm.jet, extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 10))
    ax1.set_aspect('auto')
    cbar = f.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel('Uncertainty (m)')
    
    ## XZ plane
    XZ=np.zeros([len(vec),len(vec)])
    for i in range(len(vec)):
        for jj in range(len(vec)):
            idx = S.index[(S['x']==vec[i]) & (S['z']==vec[jj]) & (S['y']==sliceValue)][0]
            XZ[i, jj] = Uncertainties['rms'][idx]    
    CS_XZ = ax2.contour(vec, vec, XZ, levels=contoursValues, colors=['k'])
    # Receivers
    ax2.plot(R['x'], R['z'], 'go')
    ax2.set_xlabel('X(m)')
    ax2.set_ylabel('Z(m)')
    ax2.grid(True)
    im=ax2.imshow(XZ, interpolation='bilinear', origin='lower',
                    cmap=cm.jet, extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 10))
    ax2.set_aspect('auto')
    cbar = f.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel('Uncertainty (m)')
    
    ## YZ plane
    YZ=np.zeros([len(vec),len(vec)])
    for i in range(len(vec)):
        for jj in range(len(vec)):
            idx = S.index[(S['y']==vec[i]) & (S['z']==vec[jj]) & (S['x']==sliceValue)][0]
            YZ[i, jj] = Uncertainties['rms'][idx]
    CS_YZ = ax3.contour(vec, vec, YZ, levels=contoursValues, colors=['k'])
    # Receivers
    ax3.plot(R['y'], R['z'], 'go')
    ax3.set_xlabel('Y(m)')
    ax3.set_ylabel('Z(m)')
    ax3.grid(True)
    im = ax3.imshow(YZ, interpolation='bilinear', origin='lower',
                    cmap='jet', extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 10))
    cbar = f.colorbar(im, ax=ax3)
    cbar.ax.set_ylabel('Uncertainty (m)')
    
#    from mpl_toolkits.axes_grid1 import make_axes_locatable
#    divider = make_axes_locatable(plt.gca())
#    cax = divider.append_axes("right", "5%", pad="3%")
#    plt.colorbar(im, cax=cax)

    #plt.colorbar(im,ax=ax3)
    ax3.set_aspect('auto')
    #plt.tight_layout()
    plt.show()
    #plt.tight_layout()

        
    # Extract contours lines
    coord_CS_XY=[]
    coord_CS_XZ=[]
    coord_CS_YZ=[]
    for i in range(len(contoursValues)):
        p1 = CS_XY.collections[i].get_paths()[0]
        coord_CS_XY.append(p1.vertices)
        p2 = CS_XZ.collections[i].get_paths()[0]
        coord_CS_XZ.append(p2.vertices)
        p3 = CS_YZ.collections[i].get_paths()[0]
        coord_CS_YZ.append(p3.vertices)
    
    return coord_CS_XY, coord_CS_XZ, coord_CS_YZ



## Grid/uncertainties parameters
radius = 7 # meters
spacing = 0.6 # meters 0.3
V = 1488 # Sound speed (m/s)
NoiseSTD = 0.00012 # standard deviation of TDOAs
contoursValues = [0.5,1,3]
Colors=['k','r','b','g']

### Optimal configuration from simulated annealing
#x=[-1,0,-1,-1,1,1]
#y=[1,0,1,-1,-1,1]
#z=[1,0,-1,1,-1,1]
#R1 = pd.DataFrame({'x':x,'y':y, 'z':z})
#XY1, XZ1, YZ1 = getArrayUncertainties(R1,radius,spacing,V,NoiseSTD,contoursValues)

## Optimal configuration from simulated annealing WITH 6ft XY and 5 ft Z 
x=[-0.9144,0,-0.9144,-0.9144,0.9144,0.9144]
y=[0.9144,0,0.9144,-0.9144,-0.9144,0.9144]
z=[0.762,0,-0.762,0.762,-0.762,0.762]
R1 = pd.DataFrame({'x':x,'y':y, 'z':z})
XY1, XZ1, YZ1 = getArrayUncertainties(R1,radius,spacing,V,NoiseSTD,contoursValues)


### Optimal configuration from simulated annealing
#x=[0,-1,1,0]
#y=[-1,1,1,0]
#z=[-1,-1,-1,1]
#R1 = pd.DataFrame({'x':x,'y':y, 'z':z})
#XY1, XZ1, YZ1 = getArrayUncertainties(R1,radius,spacing,V,NoiseSTD,contoursValues)

### Rodney's configuration
#x=[1,0,0,-1,0,0]
#y=[0,1,0,0,-1,0]
#z=[0,0,-1,0,0,1]
#R2 = pd.DataFrame({'x':x,'y':y, 'z':z})
#XY2, XZ2, YZ2 = getArrayUncertainties(R2,radius,spacing,V,NoiseSTD,contoursValues)

## Optimal configuration from simulated annealing WITH 5ft instead of 2 m
x=[-0.9144,0,-0.9144,-0.9144,0.9144,0.9144]
y=[0.9144,-0.9144,0.9144,-0.9144,-0.9144,0.9144]
z=[0.762,0,-0.762,0.762,-0.762,0.762]
R2 = pd.DataFrame({'x':x,'y':y, 'z':z})
XY2, XZ2, YZ2 = getArrayUncertainties(R2,radius,spacing,V,NoiseSTD,contoursValues)

### 6 HP optim configuration using cubic volume sources
#picklefile=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\20180801132326_Receivers6_Bounds1m_Sources729_Radius3m\ArrayOptimizationResults_iteration-3.pickle'
#data = pickle.load(open(picklefile, 'rb')) # load  pickle file
#R2 = data["R"] # Receivers coordinates
#XY2, XZ2, YZ2 = getArrayUncertainties(R2,radius,spacing,V,NoiseSTD,contoursValues)

### 5 HP optim configuration
#x=[-1,-1,-1,1,1]
#y=[1,1,-1,-1,1]
#z=[1,-1,1,-1,1]
#R2 = pd.DataFrame({'x':x,'y':y, 'z':z})
#XY2, XZ2, YZ2 = getArrayUncertainties(R2,radius,spacing,V,NoiseSTD,contoursValues)

### 4 HP optim configuration
#picklefile=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\20180711213934_Receivers4_Bounds1m_Sources500_Radius2m\ArrayOptimizationResults_iteration-2.pickle'
#data = pickle.load(open(picklefile, 'rb')) # load  pickle file
#R2 = data["R"] # Receivers coordinates
#XY2, XZ2, YZ2 = getArrayUncertainties(R2,radius,spacing,V,NoiseSTD,contoursValues)

### 4 HP tetrahedral
#x=[0,-1,1,0]
#y=[-1,1,1,0]
#z=[-1,-1,-1,1]
#R1 = pd.DataFrame({'x':x,'y':y, 'z':z})
#XY1, XZ1, YZ1 = getArrayUncertainties(R1,radius,spacing,V,NoiseSTD,contoursValues)

## PLot both contours
f2, (AX1, AX2, AX3) = plt.subplots(1, 3, sharey=False,figsize=(16, 5))

#XY
for i in range(len(contoursValues)):
   AX1.plot(XY1[i][:,0],XY1[i][:,1],Colors[i])
   AX1.plot(XY2[i][:,0],XY2[i][:,1],'--'+Colors[i])
   AX1.set_xlabel('X(m)')
   AX1.set_ylabel('Y(m)')
   AX1.grid(True)
   #AX1.set_aspect('auto')

#XZ
for i in range(len(contoursValues)):
   AX2.plot(XZ1[i][:,0],XZ1[i][:,1],Colors[i])
   AX2.plot(XZ2[i][:,0],XZ2[i][:,1],'--'+Colors[i])
   AX2.set_xlabel('X(m)')
   AX2.set_ylabel('Z(m)')
   AX2.grid(True)
   #AX2.set_aspect('auto')

#YZ
for i in range(len(contoursValues)):
   AX3.plot(YZ1[i][:,0],YZ1[i][:,1],Colors[i],label=str(contoursValues[i]))
   AX3.plot(YZ2[i][:,0],YZ2[i][:,1],'--'+Colors[i])
   AX3.set_xlabel('Y(m)')
   AX3.set_ylabel('Z(m)')
   AX3.grid(True)
   #AX3.set_aspect('auto')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=Colors[0], lw=2),
                Line2D([0], [0], color=Colors[1], lw=2),
                Line2D([0], [0], color=Colors[2], lw=2)]
AX3.legend(custom_lines, [str(contoursValues[0])+' m', str(contoursValues[1])+' m', str(contoursValues[2])+' m'],loc='upper right')