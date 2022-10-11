# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:22:32 2019

@author: xavier.mouy
"""

# C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\run10_mean_6hp_sphere-2m

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.animation as animation
import os
from PIL import Image
import glob
import pickle



picklefile=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\20191118092620_Receivers6_Bounds1m_Sources500_Radius2m\ArrayOptimizationResults_iteration-1.pickle'
outdir1 = r"C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\20191118092620_Receivers6_Bounds1m_Sources500_Radius2m\animation\frames"


data = pickle.load(open(picklefile, 'rb'))
outroot = data["outroot"]
outdir = data["outdir"]
nIter = data["nIter"]
nsources = data["nsources"]
radius = data["radius"]
origin = data["origin"]
spacing = data["spacing"]
NoiseSTD = data["NoiseSTD"]
V = data["V"]
nReceivers = data["nReceivers"]
ReceiverBoundValue = data["ReceiverBoundValue"]
ReceiverBounds = data["ReceiverBounds"]
AnnealingSchedule = data["AnnealingSchedule"]
R = data["R"]
Rchanges = data["Rchanges"]
acceptRateChanges = data["acceptRateChanges"]
Cost = data["Cost"]
S = data["S"]
Uncertainties2 = data["Uncertainties2"]
                  

# set a list with acceptance rate for each iteration and add to the Cost dataFrane
acRate=[]
for t in Cost['T']:
    acRate.append(acceptRateChanges[acceptRateChanges['T']==t]['acceptRate'].iloc[0])
Cost['acceptRate']=acRate


def update(i,Cost):
    plt.close('all')
    gridsize = (3, 3)
    fig = plt.figure(figsize=(20, 10))
    fig.set_tight_layout(True)
    
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=3, projection='3d')
    ax2 = plt.subplot2grid(gridsize, (0, 2))
    ax3 = plt.subplot2grid(gridsize, (1, 2))
    ax4 = plt.subplot2grid(gridsize, (2, 2))
    
    Vmin=-1
    Vmax = 1
    #ax1.scatter(Rchanges[:,0,i], Rchanges[:,1,i], Rchanges[:,2,i], c=['r','y','k','b','g','c'],s=40)
    ax1.scatter(Rchanges[0,0,i], Rchanges[0,1,i], Rchanges[0,2,i], c=['r'],s=60,label='Hydrophone 1')
    ax1.scatter(Rchanges[1,0,i], Rchanges[1,1,i], Rchanges[1,2,i], c=['y'],s=60,label='Hydrophone 2')
    ax1.scatter(Rchanges[2,0,i], Rchanges[2,1,i], Rchanges[2,2,i], c=['k'],s=60,label='Hydrophone 3')
    ax1.scatter(Rchanges[3,0,i], Rchanges[3,1,i], Rchanges[3,2,i], c=['b'],s=60,label='Hydrophone 4')
    ax1.scatter(Rchanges[4,0,i], Rchanges[4,1,i], Rchanges[4,2,i], c=['g'],s=60,label='Hydrophone 5')
    ax1.scatter(Rchanges[5,0,i], Rchanges[5,1,i], Rchanges[5,2,i], c=['c'],s=60,label='Hydrophone 6')            
    ax1.set_xlim([Vmin,Vmax])
    ax1.set_ylim([Vmin,Vmax])
    ax1.set_zlim([Vmin,Vmax])
    ax1.set_xlabel('X (m)',fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y (m)',fontsize=14, fontweight='bold')
    ax1.set_zlabel('Z (m)',fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left',bbox_to_anchor= (0.0, 1.01),frameon=False,fontsize=12)
    ax1.view_init(30, 80)
    
    ax2.semilogy(Cost['T'][0:i])
    ax2.set_ylabel('Temperature',fontsize=14, fontweight='bold')
    ax2.set_xlim([0,13000])
    ax2.set_ylim([2.464046623054599e-05,200])
    
    ax3.plot(Cost['acceptRate'][0:i])
    ax3.set_ylabel('Acceptance rate',fontsize=14, fontweight='bold')
    ax3.set_xlim([0,13000])
    ax3.set_ylim([0,1])
    
    ax4.plot(Cost['cost'][0:i])
    ax4.set_xlabel('Iteration',fontsize=14, fontweight='bold')
    ax4.set_ylabel('Cost',fontsize=14, fontweight='bold')
    ax4.set_xlim([0,13000])
    ax4.set_ylim([0,50])

    filename = str(i) + '.png'
    fname = os.path.join(outdir1, filename)
    plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)


for i in range(0,12588,50):
    update(i,Cost)

#for i in range(12500,12588,1):
#    update(i,Cost)

# Open all the frames
files = glob.glob(outdir1 + '/*.png')
files = os.listdir(outdir1)
filessorted = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))

images = []
for n in filessorted:
    frame = Image.open(os.path.join(outdir1,n))
    images.append(frame)

# Save the frames as an animated GIF
images[0].save(os.path.join(outdir1,'animation.gif'),
               save_all=True,
               append_images=images[1:],
               duration=200,
               loop=1)
    