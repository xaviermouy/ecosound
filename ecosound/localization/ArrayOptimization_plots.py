# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:26:32 2017

@author: xavier
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import plotly
import plotly.graph_objs as go
import pickle

  

picklefile=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization\20180711093903_Receivers5_Bounds1m_Sources500_Radius2m\ArrayOptimizationResults_iteration-2.pickle'

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
                  

nReceivers = R.shape[0]

# plot Parameters evolution with Temperature
f1 = plt.figure(1)
for Ridx in range(nReceivers):
    plt.subplot(nReceivers,1, Ridx+1)
    plt.plot(Rchanges[Ridx,0,:], label='X(m)', color='black')
    plt.plot(Rchanges[Ridx,1,:], label='Y(m)', color='red')
    plt.plot(Rchanges[Ridx,2,:], label='Z(m)', color='green')
    plt.grid('on')
    plt.ylabel('H' + str(Ridx+1) )
    if Ridx == nReceivers -1:
        plt.xlabel('Temperature step')
    #if Ridx == 0:
    #    plt.legend(loc="best", labels=['X(m)','Y(m)','Z(m)'], bbox_to_anchor=(0.5,-0.1))

# plot cost evolution with Temperature
f2 = plt.figure(2)
plt.plot(Cost['cost'], color = 'black')
plt.grid('on')
plt.xlabel('Temperature step')
plt.ylabel('Cost')

# plot acceptance rate with Temperature
f3 = plt.figure(3)
plt.semilogx(acceptRateChanges['T'],acceptRateChanges['acceptRate'], color = 'black')
plt.grid('on')
plt.xlabel('Temperature')
plt.ylabel('Acceptance rate')
plt.semilogx

# plot Parameters evolution with Temperature
f4 = plt.figure(4)
ax1 = f4.add_subplot(111, projection='3d')
# Receivers
ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
# Axes labels
ax1.set_xlabel('X (m)', labelpad=10)
ax1.set_ylabel('Y (m)', labelpad=10)
ax1.set_zlabel('Z (m)', labelpad=10)
plt.show()




#f1.legend(loc="upper right", labels=['X(m)','Y(m)','Z(m)'], bbox_to_anchor=(0.5,-0.1))
trace = go.Scatter3d(
        x=R['x'],
        y=R['y'],
        z=R['z'],
        mode='markers',
        marker=dict(size=10,color='red',colorscale='Viridis'),
        text= ['H1','H2','H3','H4','H5','H6']
    )


data = [trace]
plotly.offline.plot(data,show_link=False,config=dict(displayModeBar=False), filename='OptimalReceiversLocation.html')




trace1 = go.Scatter3d(
        x=S['x'],
        y=S['y'],
        z=S['z'],
        mode='markers',
        marker=dict(size=8,color=Uncertainties2['rms'],colorscale='Viridis'),    
        text = Uncertainties2['rms']
    )


data = [trace1, trace]
plotly.offline.plot(data, show_link=False,config=dict(displayModeBar=False), filename='OptimalReceiversUncertainties.html' )

