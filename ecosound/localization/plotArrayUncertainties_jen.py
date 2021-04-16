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
                    cmap=cm.jet, extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 20))
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
                    cmap=cm.jet, extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 20))
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
                    cmap='jet', extent=(-radius, radius, -radius, radius),norm=colors.Normalize(vmin = 0, vmax = 20))
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
radius = 3 # meters
spacing = 0.2 # meters 0.3
V = 1488 # Sound speed (m/s)
NoiseSTD = 3.1717e-5 # standard deviation of TDOAs
contoursValues = [0.1,0.2,0.4,0.5]
Colors=['k','r','b','g']

## Hydrophone coordinates (m)
#x=[-0.51,  0.095,  2.430, -0.53,  2.41] Jen hydrophones
#y=[-0.10, -0.125, -0.115, -0.12, -0.10]
#z=[-1.82,  0.570, -0.050, -0.05, -1.82]

x=[-0.42,0.03,0,0.47]
y=[0,0.19,0.51,0]
z=[0,0.51,0,0]
R1 = pd.DataFrame({'x':x,'y':y, 'z':z})

## Runs the processing:
XY1, XZ1, YZ1 = getArrayUncertainties(R1,radius,spacing,V,NoiseSTD,contoursValues)

