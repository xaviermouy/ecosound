# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 09:35:38 2017

@author: xavier
"""
import numpy as np
import pandas as pd
import pickle
import os
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import datetime

import localizationlib as loclib

#def defineSphereSurfaceGrid(npoints, radius, origin=[0, 0, 0]):
#    # Using the golden spiral method
#    # ------------------
#    # inputs:
#    #   npoints =>  nb of points on the sphere - integer
#    #   radius  => radius of the sphere - float
#    #   origin  => origin of teh sphere in catesian coordinates- 3 element list
#    # ------------------
#    # sampling in spherical coordinates
#    indices = np.arange(0, npoints, dtype=float) + 0.5
#    phi = np.arccos(1 - 2*indices/npoints)
#    theta = np.pi * (1 + 5**0.5) * indices
#    # convert to cartesian coordinates
#    Sx, Sy, Sz = radius*np.cos(theta) * np.sin(phi), radius*np.sin(theta) * np.sin(phi), radius*np.cos(phi)
#    # Adjust origin
#    Sx = Sx + origin[0]
#    Sy = Sy + origin[1]
#    Sz = Sz + origin[2]
#    # package in a datafrane
#    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
#    return S
#
#
#def defineSphereVolumeGrid(spacing, radius, origin=[0, 0, 0]):
#    # ------------------
#    # inputs:
#    #   spacing =>  distance in meters separatying each receiver - float
#    #   radius  => radius of the sphere - float
#    #   origin  => origin of the sphere in catesian coordinates- 3 element list
#    # ------------------
#    # Cube of points (Cartesian coordinates)
#    vec = np.arange(-radius, radius, spacing)
#    X, Y, Z = np.meshgrid(vec, vec, vec, indexing='ij')
#    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
#    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
#    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
#    # Convert to spherical coordinates and remove points with r < radius
#    Sr = np.sqrt(Sx**2 + Sy**2 + Sz**2)
#    Sr_sphere = Sr <= radius
#    Sx_sphere = Sx[Sr_sphere]
#    Sy_sphere = Sy[Sr_sphere]
#    Sz_sphere = Sz[Sr_sphere]
#    # Adjust origin
#    Sx_sphere = Sx_sphere + origin[0]
#    Sy_sphere = Sy_sphere + origin[1]
#    Sz_sphere = Sz_sphere + origin[2]
#    # package in a datafrane
#    S = pd.DataFrame({'x': Sx_sphere, 'y': Sy_sphere, 'z': Sz_sphere})
#    return S
#
#
#def defineReceiverPairs (nReceivers):
#    refReceiver = 1
#    Rpairs = []
#    for i in range(nReceivers):
#        if i != refReceiver:
#            pair = [refReceiver, i]
#            Rpairs.append(pair)
#    return Rpairs
#
#
#def defineJacobian(R, S, V, Rpairs):
#    N = R.shape[0] - 1          # nb of measurements (TDOAs)
#    M = 3                       # number of model parameters (unknowns)
#    nsources = S.shape[0]       # number of sources
#    J = [None] * nsources       # initiaization
#    # for each source location
#    for idx in range(nsources):
#        s = S.iloc[idx]
#        j = np.full([N, M], np.nan)  # initialization of Jacobian for that source location
#        for i in range(N):
#            p1 = Rpairs[i][0]        # receiver #1 ID
#            p2 = Rpairs[i][1]        # receiver #2 ID
#            for kk in range(M):
#                Term1 = (1/V)*0.5*((((s.x-R.x[p1])**2)+((s.y-R.y[p1])**2)+((s.z-R.z[p1])**2))**(-0.5))*2*(s.iloc[kk]-R.iloc[p1][kk])
#                Term2 = (1/V)*0.5*((((s.x-R.x[p2])**2)+((s.y-R.y[p2])**2)+((s.z-R.z[p2])**2))**(-0.5))*2*(s.iloc[kk]-R.iloc[p2][kk])
#                j[i][kk] = Term2 - Term1
#        J[idx] = j  # stacks jacobians for each source
#    return J
#
#
#def getUncertainties(J, NoiseVariance):
#    nsources = len(J)
#    errLoc_X = [None] * nsources
#    errLoc_Y = [None] * nsources
#    errLoc_Z = [None] * nsources
#    errLoc_RMS = [None] * nsources
#    for i in range(nsources):
#        Cm = NoiseVariance * np.linalg.pinv(np.dot(np.transpose(J[i]), J[i]))  # covariance matrix of the model
#        errLoc_X[i], errLoc_Y[i], errLoc_Z[i] = np.sqrt(np.diag(Cm))  # uncertainty (std) along each axis
#        errLoc_RMS[i] = np.sqrt(errLoc_X[i]**2 + errLoc_Y[i]**2 + errLoc_Z[i]**2)  # overall uncertainty (RMS)
#    Uncertainty = pd.DataFrame({'x': errLoc_X, 'y': errLoc_Y, 'z': errLoc_Z, 'rms': errLoc_RMS})
#    return Uncertainty
#
#
#def plotArrayUncertainties(R, S, Uncertainties):
#    fig1 = plt.figure()
#    ax1 = fig1.add_subplot(111, projection='3d')
#    # Sources
#    points = ax1.scatter(S['x'], S['y'], S['z'], s=10, c=Uncertainties.rms.values, cmap='spectral')
#    cbar = plt.colorbar(points)
#    cbar.ax.set_ylabel('RMS uncertainty (m)')
#    # Receivers
#    ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
#    # Axes labels
#    ax1.set_xlabel('X (m)', labelpad=10)
#    ax1.set_ylabel('Y (m)', labelpad=10)
#    ax1.set_zlabel('Z (m)', labelpad=10)
#    plt.show()
#
#
#def getCost(R, S, Rpairs, V, NoiseVariance):
#    # Get list of Jacobian matrice for each source
#    J = defineJacobian(R, S, V, Rpairs)
#    # Calculates localization uncertainty for each source
#    Uncertainties = getUncertainties(J, NoiseVariance)
#    # Get max uncertainty
#    #E = max(Uncertainties.rms)
#    E = np.mean(Uncertainties.rms)
#    #E = np.median(Uncertainties.rms)
#    return E
#
#
#def getReceiverBoundsWidth(ReceiverBounds):
#    ReceiverBoundsWidth = ReceiverBounds.applymap(lambda x: max(x)-min(x))
#    return ReceiverBoundsWidth
#
#
#def initializeReceivers(nReceivers, ReceiverBounds):
#    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)
#    R1 = [None] * ReceiverBounds.shape[1]
#    R1[0] = [None] * nReceivers  # x
#    R1[1] = [None] * nReceivers  # y
#    R1[2] = [None] * nReceivers  # z
#    for r in range(nReceivers):  # random location for each receiver and axis
#        for dim in range(ReceiverBounds.shape[1]):
#            R1[dim][r] = np.random.rand(1)[0]*ReceiverBoundsWidth.iloc[r][dim] + min(ReceiverBounds.iloc[r][dim])
#    Receivers = pd.DataFrame({'x': R1[0], 'y': R1[1], 'z': R1[2]})
#    return Receivers
#
#
#def getParamsLinearMapping(R):
#    Rindices = [None] * R.shape[0] * R.shape[1]
#    midx = 0
#    for ridx in range(R.shape[0]):
#        for dimidx in range(R.shape[1]):
#            Rindices[midx] = [ridx, dimidx]
#            midx += 1
#    return Rindices
#
#
#def perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, PerturbSTD, T0, T):
#    # goes back to first parameters if reached the end of the list of parameters
#    if PerturbParamIdx > len(MappedParamsIdx)-1:
#        PerturbParamIdx = 0
#    # Identifies from MappedParamsIdx which Receiver and Dimension to perturb
#    rid = MappedParamsIdx[PerturbParamIdx][0]    # Receiver ID
#    dimid = MappedParamsIdx[PerturbParamIdx][1]  # Dimension ID
#    # Add perturbation to parameter
#    perturb = ((PerturbSTD*ReceiverBoundsWidth.iloc[rid][dimid])*np.random.normal(loc=0))  # Gaussian distributed perturbation
#    # nu = np.random.random()
#    # gamma = PerturbSTD*(T/T0)*np.tan(np.pi*(nu-0.5))
#    # bandwidth = ReceiverBoundsWidth.iloc[rid][dimid]
#    # perturb = gamma*bandwidth
#    newparam = R.iloc[rid][dimid] + perturb
#    # print(perturb)
#    # Checks that perturbed parameter lies within the bounds
#    isinbound = (newparam >= min(ReceiverBounds.iloc[rid][dimid])) & (newparam <= max(ReceiverBounds.iloc[rid][dimid]))
#    # updates receiver parameter (only if new paramater fall within parameter bounds)
#    R_prime = pd.DataFrame.copy(R)
#    if isinbound == True:
#        R_prime.iloc[rid][dimid] = newparam        
#    return R_prime, isinbound, PerturbParamIdx
#
#
#def optimizeArray(ReceiverBounds, nReceivers, AnnealingSchedule, S, Rpairs, V, NoiseVariance):
#    # start clock
#    start = time.time()
#    
#    # Defines width of parameters bounds
#    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)
#    
#    # initialization of variables    
#    Cost = pd.DataFrame({'T': [], 'cost': []})
#    acceptRateChanges = pd.DataFrame({'T': [], 'acceptRate': []})
#    acceptRate = 1
#    PerturbParamIdx = -1
#    Tidx = 0  # temperature step index
#    LoopStopFlag = 0
#    Rchanges = []
#    while LoopStopFlag == 0:  # Temperature loop. Keeps iterating until acceptance rate is too low
#        # for nnn in range(1):
#        # First iteration
#        if (Tidx == 0):
#            R = initializeReceivers(nReceivers, ReceiverBounds)  # random initialization of receivers locations (whithin the bounds)
#            E_m = getCost(R, S, Rpairs, V, NoiseVariance)        # Calculates max RMS uncertainty
#            T = AnnealingSchedule['Start']                       # initial temperature
#            tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
#            Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
#            Rchanges = R.as_matrix()                                         # Keeps track of model paraneters at each iteration
#            MappedParamsIdx = getParamsLinearMapping(R)          # linear list of each elements to optimize (for the perturnation phase)
#            #plotArrayUncertainties(R, S, Uncertainties)        
#
#        # Checks that starting temperature is not set to low
#        if (Tidx == 1):
#            if (acceptRate < AnnealingSchedule['StartAcceptanceRate']):
#                raise ValueError(['The acceptance rate during the melting phase is too low.Please adjust starting temperature (' + str(acceptRate) + ')'])    
#            else:
#                print('Melting temperature valid (' + str(acceptRate) + ').')
#
#        # Perturb paraneters
#        nAccepted = 0  # keeps track of accepted perturbations
#        for j in range(AnnealingSchedule['nPerturb']):  # perturbations of parameters for temperature T
#            # Perturb one receiver parameter
#            PerturbParamIdx += 1  # increment to next parameter in line
#            R_prime, isinbound, PerturbParamIdx = perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, AnnealingSchedule['PerturbSTD'], AnnealingSchedule['Start'], T)
#            # Acceptance tests
#            acceptedFlag = []
#            if isinbound:
#                E_mprime = getCost(R_prime, S, Rpairs, V, NoiseVariance)       # Calculates max RMS uncertainty
#                deltaE = E_mprime-E_m
#                if (deltaE <= 0):   # accept change
#                    acceptedFlag = True
#                elif (deltaE > 0):
#                    psy = np.random.uniform()
#                    P = np.exp(-deltaE/T)
#                    if (psy <= P):  # accept change
#                        acceptedFlag = True
#                    else:           # reject change
#                        acceptedFlag = False
#            else:
#                acceptedFlag = False
#            # Updates parmaters for accepted changes
#            if acceptedFlag:  # accepted change
#                nAccepted += 1
#                # update paramters
#                R = pd.DataFrame.copy(R_prime)
#                E_m = E_mprime
#                del R_prime, E_mprime   # delete variables
#                # saves cost and model parameters
#                tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
#                Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
#                Rchanges = np.dstack((Rchanges, R.as_matrix()))
#            elif acceptedFlag is not False:  # sanity check
#                raise ValueError(['Perturbation did go throught the acceptance test. Check the code for errors!'])
#    
#        # Calculates acceptance rate for that temperature value
#        acceptRate = nAccepted / AnnealingSchedule['nPerturb']
#        tmp2 = pd.DataFrame({'T': [T], 'acceptRate': [acceptRate]})
#        acceptRateChanges = acceptRateChanges.append(tmp2, ignore_index=True)
#        print('Temperature: %.3f - Acceptance rate: %.2f - Cost: %.2f' % (T, acceptRate, E_m))
#    
#        # Stopping conditions
#        if (acceptRate < AnnealingSchedule['StopAcceptanceRate']):
#            LoopStopFlag = 1
#            print('Optimization complete (acceptance rate threshold reached)')
#        if (E_m <= AnnealingSchedule['StopCost']):
#            LoopStopFlag = 1
#            print('Optimization complete (cost objective reached)')
#    
#        # Update temperature
#        T = T * AnnealingSchedule['ReducFactor']  # decrease temperature
#        Tidx += 1  # next temperature step
#
#    end = time.time()
#    elapsedTime = end - start
#    return R, Rchanges, acceptRateChanges, Cost, elapsedTime
#
#def plotOptimizationResults(outdir, nReceivers, Rchanges, Cost, acceptRateChanges, R):
#    
#    # plot Parameters evolution with Temperature
#    f1 = plt.figure(1)
#    for Ridx in range(nReceivers):
#        plt.subplot(nReceivers,1, Ridx+1)
#        plt.plot(Rchanges[Ridx,0,:], label='X(m)', color='black')
#        plt.plot(Rchanges[Ridx,1,:], label='Y(m)', color='red')
#        plt.plot(Rchanges[Ridx,2,:], label='Z(m)', color='green')
#        plt.grid('on')
#        plt.ylabel('H' + str(Ridx+1) )
#        if Ridx == nReceivers -1:
#            plt.xlabel('Temperature step')
#        #if Ridx == 0:
#        #    plt.legend(loc="best", labels=['X(m)','Y(m)','Z(m)'], bbox_to_anchor=(0.5,-0.1))
#    f1.savefig(os.path.join(outdir, 'ReceiversPositionVsTemperature' + '_iteration-' + str(i+1) + '.png'), bbox_inches='tight')
#    
#    # plot cost evolution with Temperature
#    f2 = plt.figure(2)
#    plt.plot(Cost['cost'], color = 'black')
#    plt.grid('on')
#    plt.xlabel('Temperature step')
#    plt.ylabel('Cost')
#    f2.savefig(os.path.join(outdir, 'CostVsTemperature' + '_iteration-' + str(i+1) + '.png'), bbox_inches='tight')
#
#    
#    # plot acceptance rate with Temperature
#    f3 = plt.figure(3)
#    plt.semilogx(acceptRateChanges['T'],acceptRateChanges['acceptRate'], color = 'black')
#    plt.grid('on')
#    plt.xlabel('Temperature')
#    plt.ylabel('Acceptance rate')
#    plt.semilogx
#    f3.savefig(os.path.join(outdir, 'AcceptanceRateVsTemperature' + '_iteration-' + str(i+1) + '.png'), bbox_inches='tight')
#
#    
#    # plot Final receivers positions
#    f4 = plt.figure(4)
#    ax1 = f4.add_subplot(111, projection='3d')
#    # Receivers
#    ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
#    # Axes labels
#    ax1.set_xlabel('X (m)', labelpad=10)
#    ax1.set_ylabel('Y (m)', labelpad=10)
#    ax1.set_zlabel('Z (m)', labelpad=10)
#    plt.show()
#    f4.savefig(os.path.join(outdir, 'FinalReceiversPosition' + '_iteration-' + str(i+1) + '.png'), bbox_inches='tight')

# ------------------------------------------------------
# input parameters --------------------------------------

# Results directory
outroot = r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\Fish-localization\results\ArrayOptimization'

# Number of iterations (nb of time the process is repeated to ensure stability of the solution)
nIter = 1

## Spherical grid (sources location)
nsources = 500 #300
radius = 2
origin = [0, 0, 0]
spacing = 0.5

## cubic grid
#radius = 2
#origin = [0, 0, 0]
#spacing = 0.1 # 0.5

# Measurements errors (TDOA)
NoiseSTD = 5.0714e-04  # standard deviation of TDOAs

# Sound speed (m/s)
V = 1488

# number of receivers (for optimization)
nReceivers = 6

# Optimization
ReceiverBoundValue = 1
tmp = [[-ReceiverBoundValue, ReceiverBoundValue]] * nReceivers  # boundaries of receiver location (m)
ReceiverBounds = pd.DataFrame({'x': tmp, 'y': tmp, 'z': tmp})
#
#tmpx = [[-0.6, 0.6]] * nReceivers  # boundaries of receiver location (m)
#tmpy = [[-0.6, 0.6]] * nReceivers  # boundaries of receiver location (m)
#tmpz = [[0.6, 0.6]] * nReceivers  # boundaries of receiver location (m)
#ReceiverBounds = pd.DataFrame({'x': tmpx, 'y': tmpy, 'z': tmpz})

AnnealingSchedule = ({          # defines annealing schedule
    'Start': 200, #200
    'ReducFactor': 0.9, # before was 0.9
    'nPerturb': 100, # 30
    'PerturbSTD': 1/8,  #  1/8
    'StartAcceptanceRate': 0.8, # 0.8
    'StopAcceptanceRate': 0.001,  # 0.001
    'StopCost': 0
     })
# ------------------------------------------------------
# ------------------------------------------------------

# prelim
NoiseVariance = NoiseSTD**2

# Set virtual sources location (spherical grid)
S = loclib.defineSphereSurfaceGrid(nsources, radius, origin)
#S = loclib.defineCubeVolumeGrid(spacing, radius, origin)
nsources = S.shape[0]

# creates output folder
StartTimestamp_obj = datetime.datetime.now()
StartTimestamp_str = StartTimestamp_obj.strftime("%Y%m%d%H%M%S")
outdir = os.path.join(outroot,StartTimestamp_str + '_' + 'Receivers' + str(nReceivers) + '_' + 'Bounds' + str(ReceiverBoundValue) + 'm_' + 'Sources' + str(nsources)  + '_' + 'Radius' + str(radius) + 'm' )
os.mkdir(outdir)

# Define receiver pairs for TDOAs
Rpairs = loclib.defineReceiverPairs (nReceivers)

# Repeats optimization nIter times to ensure stability
for i in range(nIter):    
       
    # Closes all open figures
    plt.close("all")
    # Optimize array configuration 
    R, Rchanges, acceptRateChanges, Cost, processingTime = loclib.optimizeArray(ReceiverBounds, nReceivers, AnnealingSchedule, S, Rpairs, V, NoiseVariance)       
    # Get list of Jacobian matrice for each source
    J2 = loclib.defineJacobian(R, S, V, Rpairs) 
    # Calculates localization uncertainty for each source
    Uncertainties2 = loclib.getUncertainties(J2, NoiseVariance)   
    # Plots unceratinties of optimized array
    loclib.plotArrayUncertainties(R, S, Uncertainties2)
    plt.savefig(os.path.join(outdir, 'UncertaintiesPlot' + '_iteration-' + str(i+1) + '.png'), bbox_inches='tight')
    # Plots Optimization results
    loclib.plotOptimizationResults(outdir, nReceivers, Rchanges, Cost, acceptRateChanges, R, i)

    # Save paraneters and results to pickle file
    outfilename = os
    data = {"outroot": outroot , 
             "outdir": outdir , 
             "nIter": nIter , 
             "nsources": nsources , 
             "radius": radius, 
             "origin": origin, 
             "spacing": spacing, 
             "NoiseSTD": NoiseSTD, 
             "V": V, 
             "nReceivers": nReceivers, 
             "ReceiverBoundValue": ReceiverBoundValue,
             "ReceiverBounds": ReceiverBounds, 
             "AnnealingSchedule": AnnealingSchedule, 
             "R": R, 
             "Rchanges": Rchanges, 
             "acceptRateChanges": acceptRateChanges, 
             "Cost": Cost, 
             "S": S, 
             "Uncertainties2": Uncertainties2,
             "processingTime": processingTime
             }
    pickle.dump(data, open(os.path.join(outdir, 'ArrayOptimizationResults' + '_iteration-' + str(i+1) + '.pickle'), "wb"))
	



## ------------------------------------------------------
## Original array configuration -------------------------
#
## Original receivers location
#Rx = [1.185, 0, -1.185, 0]          # X values
#Ry = [0, 1.190, 0, -1.190]  # Y values
#Rz = [0, 0, 0, 0]               # Z values
#
#
## set receivers location in pandas dataframe
#R0 = pd.DataFrame({'x': Rx, 'y': Ry, 'z': Rz})
#
## Set sources location (spherical grid)
#S = defineSphereSurfaceGrid(nsources, radius, origin)
##S2 = pd.DataFrame.append(S,S1, ignore_index = True)
##S = defineSphereVolumeGrid(spacing, radius, origin)
#
## Get list of Jacobian matrice for each source
#J = defineJacobian(R0, S, V, Rpairs)
## Calculates localization uncertainty for each source
#Uncertainties = getUncertainties(J, NoiseVariance)
## plot Sources, Receivers and uncertainties
#plotArrayUncertainties(R0, S, Uncertainties)

## ------------------------------------------------------
