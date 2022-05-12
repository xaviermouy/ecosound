# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:04:19 2018

@author: xavier.mouy
"""

import numpy as np
import pandas as pd
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import time
import os
from ecosound.core.audiotools import upsample
import scipy.signal
from numba import njit


def defineSphereSurfaceGrid(npoints, radius, origin=[0, 0, 0]):
    # Using the golden spiral method
    # ------------------
    # inputs:
    #   npoints =>  nb of points on the sphere - integer
    #   radius  => radius of the sphere - float
    #   origin  => origin of teh sphere in catesian coordinates- 3 element list
    # ------------------
    # sampling in spherical coordinates
    indices = np.arange(0, npoints, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/npoints)
    theta = np.pi * (1 + 5**0.5) * indices
    # convert to cartesian coordinates
    Sx, Sy, Sz = radius*np.cos(theta) * np.sin(phi), radius*np.sin(theta) * np.sin(phi), radius*np.cos(phi)
    # Adjust origin
    Sx = Sx + origin[0]
    Sy = Sy + origin[1]
    Sz = Sz + origin[2]
    # package in a datafrane
    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
    return S


def defineSphereVolumeGrid(spacing, radius, origin=[0, 0, 0]):
    # ------------------
    # inputs:
    #   spacing =>  distance in meters separatying each receiver - float
    #   radius  => radius of the sphere - float
    #   origin  => origin of the sphere in catesian coordinates- 3 element list
    # ------------------
    # Cube of points (Cartesian coordinates)
    vec = np.arange(-radius, radius+spacing, spacing)
    X, Y, Z = np.meshgrid(vec, vec, vec, indexing='ij')
    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
    # Convert to spherical coordinates and remove points with r < radius
    Sr = np.sqrt(Sx**2 + Sy**2 + Sz**2)
    Sr_sphere = Sr <= radius
    Sx_sphere = Sx[Sr_sphere]
    Sy_sphere = Sy[Sr_sphere]
    Sz_sphere = Sz[Sr_sphere]
    # Adjust origin
    Sx_sphere = Sx_sphere + origin[0]
    Sy_sphere = Sy_sphere + origin[1]
    Sz_sphere = Sz_sphere + origin[2]
    # package in a datafrane
    S = pd.DataFrame({'x': Sx_sphere, 'y': Sy_sphere, 'z': Sz_sphere})
    return S


def defineCubeVolumeGrid(spacing, radius, origin=[0, 0, 0]):
    # ------------------
    # inputs:
    #   spacing =>  distance in meters separatying each receiver - float
    #   radius  => radius of the sphere - float
    #   origin  => origin of the sphere in catesian coordinates- 3 element list
    # ------------------
    # Cube of points (Cartesian coordinates)
    vec = np.arange(-radius, radius+spacing, spacing)
    X, Y, Z = np.meshgrid(vec, vec, vec, indexing='ij')
    Sx = np.reshape(X, X.shape[0]*X.shape[1]*X.shape[2])
    Sy = np.reshape(Y, Y.shape[0]*Y.shape[1]*Y.shape[2])
    Sz = np.reshape(Z, Z.shape[0]*Z.shape[1]*Z.shape[2])
    # Adjust origin
    Sx = Sx + origin[0]
    Sy = Sy + origin[1]
    Sz = Sz + origin[2]
    # package in a datafrane
    S = pd.DataFrame({'x': Sx, 'y': Sy, 'z': Sz})
    return S


def defineReceiverPairs (n_receivers, ref_receiver=0):
    Rpairs = []
    for i in range(n_receivers):
        if i != ref_receiver:
            pair = [ref_receiver, i]
            Rpairs.append(pair)
    return Rpairs

def defineJacobian(R, S, V, Rpairs):
    unknowns = ['x','y','z']    # unknowns: 3D coordinates of sound source
    N = R.shape[0] - 1          # nb of measurements (TDOAs)
    M = len(unknowns)           # number of model parameters (unknowns)
    nsources = S.shape[0]       # number of sources
    J = [None] * nsources       # initiaization
    # for each source location
    for idx in range(nsources):
        s = S.iloc[idx]
        j = np.full([N, M], np.nan)  # initialization of Jacobian for that source location
        for i in range(N):
            p1 = Rpairs[i][0]        # receiver #1 ID
            p2 = Rpairs[i][1]        # receiver #2 ID
            for kk, unknown in enumerate(unknowns):
                Term1 = (1/V)*0.5*((((s.x-R.x[p1])**2)+((s.y-R.y[p1])**2)+((s.z-R.z[p1])**2))**(-0.5))*2*(s[unknown]-R[unknown][p1])
                Term2 = (1/V)*0.5*((((s.x-R.x[p2])**2)+((s.y-R.y[p2])**2)+((s.z-R.z[p2])**2))**(-0.5))*2*(s[unknown]-R[unknown][p2])
                j[i][kk] = Term2 - Term1
        J[idx] = j  # stacks jacobians for each source
    if nsources == 1:
        J = J[0]
    return J

def predict_tdoa(m, V, hydrophones_coords, hydrophone_pairs):
    """ Create data from the forward problem.
        Generate TDOAs based on location of hydrophones and source.
    """
    N = len(hydrophone_pairs)
    dt = np.full([N,1], np.nan)
    for idx, hydrophone_pair in enumerate(hydrophone_pairs):
        p1=hydrophone_pair[0]
        p2=hydrophone_pair[1]
        t1=((1/V)*np.sqrt( ((m.x-hydrophones_coords.x[p1])**2)+((m.y-hydrophones_coords.y[p1])**2)+((m.z-hydrophones_coords.z[p1])**2)) )
        t2=((1/V)*np.sqrt(((m.x-hydrophones_coords.x[p2])**2)+((m.y-hydrophones_coords.y[p2])**2)+((m.z-hydrophones_coords.z[p2])**2)))
        dt[idx] = np.array(t2-t1) # noiseless data
    return dt


def getUncertainties(J, NoiseVariance):
    nsources = len(J)
    errLoc_X = [None] * nsources
    errLoc_Y = [None] * nsources
    errLoc_Z = [None] * nsources
    errLoc_RMS = [None] * nsources
    for i in range(nsources):
        Cm = NoiseVariance * np.linalg.inv(np.dot(np.transpose(J[i]), J[i]))  # covariance matrix of the model
        errLoc_X[i], errLoc_Y[i], errLoc_Z[i] = np.sqrt(np.diag(Cm))  # uncertainty (std) along each axis
        errLoc_RMS[i] = np.sqrt(errLoc_X[i]**2 + errLoc_Y[i]**2 + errLoc_Z[i]**2)  # overall uncertainty (RMS)
    Uncertainty = pd.DataFrame({'x': errLoc_X, 'y': errLoc_Y, 'z': errLoc_Z, 'rms': errLoc_RMS})
    return Uncertainty


def plotArrayUncertainties(R, S, Uncertainties):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    # Sources
    points = ax1.scatter(S['x'], S['y'], S['z'], s=10, c=Uncertainties.rms.values, cmap='Spectral')
    cbar = plt.colorbar(points)
    cbar.ax.set_ylabel('Uncertainty (m)')
    # Receivers
    ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
    # Axes labels
    ax1.set_xlabel('X (m)', labelpad=10)
    ax1.set_ylabel('Y (m)', labelpad=10)
    ax1.set_zlabel('Z (m)', labelpad=10)
    plt.show()


def getCost(R, S, Rpairs, V, NoiseVariance):
    # Get list of Jacobian matrice for each source
    J = defineJacobian(R, S, V, Rpairs)
    # Calculates localization uncertainty for each source
    Uncertainties = getUncertainties(J, NoiseVariance)
    # Get max uncertainty
    #E = max(Uncertainties.rms)
    E = np.mean(Uncertainties.rms)
    #E = np.median(Uncertainties.rms)
    return E


def getReceiverBoundsWidth(ReceiverBounds):
    ReceiverBoundsWidth = ReceiverBounds.applymap(lambda x: max(x)-min(x))
    return ReceiverBoundsWidth


def initializeReceivers(nReceivers, ReceiverBounds):
    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)
    R1 = [None] * ReceiverBounds.shape[1]
    R1[0] = [None] * nReceivers  # x
    R1[1] = [None] * nReceivers  # y
    R1[2] = [None] * nReceivers  # z
    for r in range(nReceivers):  # random location for each receiver and axis
        for dim in range(ReceiverBounds.shape[1]):
            R1[dim][r] = np.random.rand(1)[0]*ReceiverBoundsWidth.iloc[r][dim] + min(ReceiverBounds.iloc[r][dim])
    Receivers = pd.DataFrame({'x': R1[0], 'y': R1[1], 'z': R1[2]})
    return Receivers


def getParamsLinearMapping(R):
    Rindices = [None] * R.shape[0] * R.shape[1]
    midx = 0
    for ridx in range(R.shape[0]):
        for dimidx in range(R.shape[1]):
            Rindices[midx] = [ridx, dimidx]
            midx += 1
    return Rindices


def perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, PerturbSTD, T0, T):
    # goes back to first parameters if reached the end of the list of parameters
    if PerturbParamIdx > len(MappedParamsIdx)-1:
        PerturbParamIdx = 0
    # Identifies from MappedParamsIdx which Receiver and Dimension to perturb
    rid = MappedParamsIdx[PerturbParamIdx][0]    # Receiver ID
    dimid = MappedParamsIdx[PerturbParamIdx][1]  # Dimension ID
    # Add perturbation to parameter
    perturb = ((PerturbSTD*ReceiverBoundsWidth.iloc[rid][dimid])*np.random.normal(loc=0))  # Gaussian distributed perturbation
    # nu = np.random.random()
    # gamma = PerturbSTD*(T/T0)*np.tan(np.pi*(nu-0.5))
    # bandwidth = ReceiverBoundsWidth.iloc[rid][dimid]
    # perturb = gamma*bandwidth
    newparam = R.iloc[rid][dimid] + perturb
    # print(perturb)
    # Checks that perturbed parameter lies within the bounds
    isinbound = (newparam >= min(ReceiverBounds.iloc[rid][dimid])) & (newparam <= max(ReceiverBounds.iloc[rid][dimid]))
    # updates receiver parameter (only if new paramater fall within parameter bounds)
    R_prime = pd.DataFrame.copy(R)
    if isinbound == True:
        R_prime.iloc[rid][dimid] = newparam
    return R_prime, isinbound, PerturbParamIdx


def optimizeArray(ReceiverBounds, nReceivers, AnnealingSchedule, S, Rpairs, V, NoiseVariance):
    # start clock
    start = time.time()

    # Defines width of parameters bounds
    ReceiverBoundsWidth = getReceiverBoundsWidth(ReceiverBounds)

    # initialization of variables
    Cost = pd.DataFrame({'T': [], 'cost': []})
    acceptRateChanges = pd.DataFrame({'T': [], 'acceptRate': []})
    acceptRate = 1
    PerturbParamIdx = -1
    Tidx = 0  # temperature step index
    LoopStopFlag = 0
    Rchanges = []
    while LoopStopFlag == 0:  # Temperature loop. Keeps iterating until acceptance rate is too low
        # for nnn in range(1):
        # First iteration
        if (Tidx == 0):
            R = initializeReceivers(nReceivers, ReceiverBounds)  # random initialization of receivers locations (whithin the bounds)
            E_m = getCost(R, S, Rpairs, V, NoiseVariance)        # Calculates max RMS uncertainty
            T = AnnealingSchedule['Start']                       # initial temperature
            tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
            Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
            Rchanges = R.as_matrix()                                         # Keeps track of model paraneters at each iteration
            MappedParamsIdx = getParamsLinearMapping(R)          # linear list of each elements to optimize (for the perturnation phase)
            #plotArrayUncertainties(R, S, Uncertainties)

        # Checks that starting temperature is not set to low
        if (Tidx == 1):
            if (acceptRate < AnnealingSchedule['StartAcceptanceRate']):
                raise ValueError(['The acceptance rate during the melting phase is too low.Please adjust starting temperature (' + str(acceptRate) + ')'])
            else:
                print('Melting temperature valid (' + str(acceptRate) + ').')

        # Perturb paraneters
        nAccepted = 0  # keeps track of accepted perturbations
        for j in range(AnnealingSchedule['nPerturb']):  # perturbations of parameters for temperature T
            # Perturb one receiver parameter
            PerturbParamIdx += 1  # increment to next parameter in line
            R_prime, isinbound, PerturbParamIdx = perturbReceivers(R, PerturbParamIdx, MappedParamsIdx, ReceiverBounds, ReceiverBoundsWidth, AnnealingSchedule['PerturbSTD'], AnnealingSchedule['Start'], T)
            # Acceptance tests
            acceptedFlag = []
            if isinbound:
                E_mprime = getCost(R_prime, S, Rpairs, V, NoiseVariance)       # Calculates max RMS uncertainty
                deltaE = E_mprime-E_m
                if (deltaE <= 0):   # accept change
                    acceptedFlag = True
                elif (deltaE > 0):
                    psy = np.random.uniform()
                    P = np.exp(-deltaE/T)
                    if (psy <= P):  # accept change
                        acceptedFlag = True
                    else:           # reject change
                        acceptedFlag = False
            else:
                acceptedFlag = False
            # Updates parmaters for accepted changes
            if acceptedFlag:  # accepted change
                nAccepted += 1
                # update paramters
                R = pd.DataFrame.copy(R_prime)
                E_m = E_mprime
                del R_prime, E_mprime   # delete variables
                # saves cost and model parameters
                #tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
                #Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
                #Rchanges = np.dstack((Rchanges, R.as_matrix()))
            elif acceptedFlag is not False:  # sanity check
                raise ValueError(['Perturbation did go throught the acceptance test. Check the code for errors!'])
            # saves cost and model parameters
            tmp1 = pd.DataFrame({'T': [T], 'cost': [E_m]})
            Cost = pd.DataFrame.append(Cost, tmp1, ignore_index=True)
            Rchanges = np.dstack((Rchanges, R.as_matrix()))

        # Calculates acceptance rate for that temperature value
        acceptRate = nAccepted / AnnealingSchedule['nPerturb']
        tmp2 = pd.DataFrame({'T': [T], 'acceptRate': [acceptRate]})
        acceptRateChanges = acceptRateChanges.append(tmp2, ignore_index=True)
        print('Temperature: %.3f - Acceptance rate: %.2f - Cost: %.2f' % (T, acceptRate, E_m))

        # Stopping conditions
        if (acceptRate < AnnealingSchedule['StopAcceptanceRate']):
            LoopStopFlag = 1
            print('Optimization complete (acceptance rate threshold reached)')
        if (E_m <= AnnealingSchedule['StopCost']):
            LoopStopFlag = 1
            print('Optimization complete (cost objective reached)')

        # Update temperature
        T = T * AnnealingSchedule['ReducFactor']  # decrease temperature
        Tidx += 1  # next temperature step

    end = time.time()
    elapsedTime = end - start
    return R, Rchanges, acceptRateChanges, Cost, elapsedTime

def plotOptimizationResults(outdir, nReceivers, Rchanges, Cost, acceptRateChanges, R, iteridx=1):

    # plot Parameters evolution with Temperature
    f1 = plt.figure(1)
    for Ridx in range(nReceivers):
        plt.subplot(nReceivers,1, Ridx+1)
        plt.plot(Rchanges[Ridx,0,:], label='X(m)', color='black')
        plt.plot(Rchanges[Ridx,1,:], label='Y(m)', color='red')
        plt.plot(Rchanges[Ridx,2,:], label='Z(m)', color='green')
        plt.grid(True)
        plt.ylabel('H' + str(Ridx+1) )
        if Ridx == nReceivers -1:
            plt.xlabel('Temperature step')
        #if Ridx == 0:
        #    plt.legend(loc="best", labels=['X(m)','Y(m)','Z(m)'], bbox_to_anchor=(0.5,-0.1))
    f1.savefig(os.path.join(outdir, 'ReceiversPositionVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')

    # plot cost evolution with Temperature
    f2 = plt.figure(2)
    plt.plot(Cost['cost'], color = 'black')
    plt.grid(True)
    plt.xlabel('Temperature step')
    plt.ylabel('Cost')
    f2.savefig(os.path.join(outdir, 'CostVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')


    # plot acceptance rate with Temperature
    f3 = plt.figure(3)
    plt.semilogx(acceptRateChanges['T'],acceptRateChanges['acceptRate'], color = 'black')
    plt.grid(True)
    plt.xlabel('Temperature')
    plt.ylabel('Acceptance rate')
    plt.semilogx
    f3.savefig(os.path.join(outdir, 'AcceptanceRateVsTemperature' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')


    # plot Final receivers positions
    f4 = plt.figure(4)
    ax1 = f4.add_subplot(111, projection='3d')
    # Receivers
    ax1.scatter(R['x'], R['y'], R['z'], s=30, c='black')
    # Axes labels
    ax1.set_xlabel('X (m)', labelpad=10)
    ax1.set_ylabel('Y (m)', labelpad=10)
    ax1.set_zlabel('Z (m)', labelpad=10)
    plt.show()
    f4.savefig(os.path.join(outdir, 'FinalReceiversPosition' + '_iteration-' + str(iteridx+1) + '.png'), bbox_inches='tight')


def euclidean_dist(df1, df2, cols=['x', 'y', 'z']):
    """
    Calculate euclidean distance between two Pandas dataframes.

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is ['x','y','z'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.linalg.norm(df1[cols].values - df2[cols].values, axis=0)


def calc_hydrophones_distances(hydrophones_coords):
    """
    Calculate Euclidiean distance between each hydrophone of an array.

    Parameters
    ----------
    hydrophones_coords : TYPE
        DESCRIPTION.

    Returns
    -------
    hydrophones_dist_matrix : TYPE
        DESCRIPTION.

    """
    hydrophones_dist_matrix = np.empty((len(hydrophones_coords),len(hydrophones_coords)))
    for index1, row1 in hydrophones_coords.iterrows():
        for index2, row2 in hydrophones_coords.iterrows():
            dist = euclidean_dist(row1, row2)
            hydrophones_dist_matrix[index1, index2] = dist
    return hydrophones_dist_matrix


def calc_tdoa(waveform_stack, hydrophone_pairs, sampling_frequency, TDOA_max_sec=None, upsample_res_sec=None, normalize=False, hydrophones_name=None, doplot=False):
    """
    TDOA measurements

    Calculates the time-difference of orrival (TDOA) between signals from
    different hydrophones by cross-correlation.

    Parameters
    ----------
    waveform_stack : list of numpy arrays
        Wavforms with amplitude values of the signal for each hydrophone.
        Each wavform is a numpy array which are stored in a list
        e.g. waveform_stack[0] contains a numpy array with the wavform from
         the first hydrophone.
    hydrophone_pairs : list
        Defines the pair of hydrophones for the TDOA measurements. Each element
        of hydrophones_pairs is a list with index values of the hydrophone in
        waveform_stack.
        e.g. hydrophones_pairs = [[3, 0], [3, 1], [3, 2], [3, 4], [3, 5]].
    sampling_frequency : float
        Sampling frequency of the waveform signals in  waveform_stack in Hz.
    TDOA_max_sec : float, optional
        Restricts the TDOA search to TDOA_max_sec seconds. The default is None.
    upsample_res_sec : float, optional
        If set, upsamples the wavforms in waveform_stack before the cross-
        correlation to have a time resolution of upsample_res_sec seconds.
        The default is None.
    normalize : bool, optional
        If set to True, normalizes the wavforms in waveform_stack to have a
        maximum amplitude of 1. The default is False.
    hydrophones_name : list, optional
        list of string with the name of each hydrophone. Only used for plots.
        The default is None.
    doplot : bool, optional
        If set to True, displays cross correlation plots for each hydrophone
        pair. The default is False.

    Returns
    -------
    tdoa_sec : 1-D numpy array
        Time-difference of arrival in seconds, for each hydrophone pair.
    tdoa_corr : 1-D numpy array
        Maximum cross-correlation value for each hydrophone pair (between 0
        and 1).

    """
    tdoa_sec = []
    tdoa_corr = []
    # Upsampling
    if upsample_res_sec:
        if upsample_res_sec < (1/sampling_frequency):
            for chan_id, waveform in enumerate(waveform_stack):
                waveform_stack[chan_id], new_sampling_frequency = upsample(
                    waveform, 1/sampling_frequency, upsample_res_sec)
            sampling_frequency = new_sampling_frequency
        else:
            print('Warning: upsampling not applied because the requested time'
                  ' resolution (upsample_res_sec) is larger than the current'
                  ' time resolution of the signal.')
    # Normalize max amplitude to 1
    if normalize:
        for chan_id, waveform in enumerate(waveform_stack):
                    waveform_stack[chan_id] = waveform / np.max(waveform)
    # Constrains to a max TDOA (based on array geometry)
    if TDOA_max_sec:
        TDOA_max_samp = int(np.round(TDOA_max_sec*sampling_frequency))

    # cross correlation
    for hydrophone_pair in hydrophone_pairs:
        # signal from each hydrophone
        s1 = waveform_stack[hydrophone_pair[0]]
        s2 = waveform_stack[hydrophone_pair[1]]
        # cross correlation
        corr = scipy.signal.correlate(s2,s1, mode='full', method='auto')
        corr = corr/(np.linalg.norm(s1)*np.linalg.norm(s2))
        lag_array = scipy.signal.correlation_lags(s2.size,s1.size, mode="full")
        # Identify correlation peak within the TDOA search window (SW)
        if TDOA_max_sec:
            SW_start_idx = np.where(lag_array == -TDOA_max_samp)[0][0] # search window start idx
            SW_stop_idx = np.where(lag_array == TDOA_max_samp)[0][0] # search window stop idx
        else:
            SW_start_idx=0
            SW_stop_idx=len(corr)-1
        corr_max_idx = np.argmax(corr[SW_start_idx:SW_stop_idx]) + SW_start_idx # idx of max corr value
        delay = lag_array[corr_max_idx]
        corr_value = corr[corr_max_idx]
        tdoa_sec.append(delay/sampling_frequency)
        tdoa_corr.append(corr_value)

        if doplot:
            fig, ax = plt.subplots(nrows=2, sharex=False)
            if hydrophones_name:
                label1= hydrophones_name[hydrophone_pair[0]] + ' (ref)'
                label2= hydrophones_name[hydrophone_pair[1]]
            else:
                 label1 = 'Hydrophone ' + str(hydrophone_pair[0]) + ' (ref)'
                 label2 = 'Hydrophone ' + str(hydrophone_pair[1])
            ax[0].plot(s1, color='red', label=label1)
            ax[0].plot(s2, color='black',label=label2)
            ax[0].set_xlabel('Time (sample)')
            ax[0].set_ylabel('Amplitude')
            ax[0].legend()
            ax[0].grid()
            ax[0].set_title('TDOA: ' + str(delay) + ' samples')
            ax[1].plot(lag_array, corr)
            ax[1].plot(delay, corr_value ,marker = '.', color='r', label='TDOA')
            if TDOA_max_sec:
                width = 2*TDOA_max_samp
                height = 2
                rect = plt.Rectangle((-TDOA_max_samp, -1), width, height,
                                             linewidth=1,
                                             edgecolor='green',
                                             facecolor='green',
                                             alpha=0.3,
                                             label='Search window')
                ax[1].add_patch(rect)
            ax[1].set_xlabel('Lag (sample)')
            ax[1].set_ylabel('Correlation')
            ax[1].set_title('Correlation: ' + str(corr_value))
            ax[1].set_ylim(-1,1)
            ax[1].grid()
            ax[1].legend()
            plt.tight_layout()
    return np.array([tdoa_sec]).transpose(), np.array([tdoa_corr]).transpose()


def solve_iterative_ML(d, hydrophones_coords, hydrophone_pairs, m, V, damping_factor):
    # Define the Jacobian matrix: Eq. (5) in Mouy et al. (2018)
    A = defineJacobian(hydrophones_coords, m, V, hydrophone_pairs)
    # Predicted TDOA at m (forward problem): Eq. (1) in Mouy et al. (2018)
    d0 = predict_tdoa(m, V, hydrophones_coords, hydrophone_pairs)
    # Reformulation of the problem
    delta_d = d-d0 # Delta d: measured data - predicted data
    try:
        # Resolving by creeping approach(ML inverse for delta m): Eq. (6) in Mouy et al. (2018)
        delta_m = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),delta_d) #general inverse
        # New model: Eq. (7) in Mouy et al. (2018)
        m_new = m.iloc[0].values + (damping_factor*delta_m.transpose())
        m['x'] = m_new[0,0]
        m['y'] = m_new[0,1]
        m['z'] = m_new[0,2]
        # ## jumping approach
        # #m=inv(A'*CdInv*A)*A'*CdInv*dprime; % retrieved model using ML
        # Data misfit
        part1 = np.dot(A,delta_m)-delta_d
        data_misfit = np.dot(part1.transpose(), part1)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('Error when inverting for delta_m: Singular Matrix')
            m['x'] = np.nan
            m['y'] = np.nan
            m['z'] = np.nan
            data_misfit = np.array([[np.nan]])
        else:
            raise err

    return m, data_misfit[0, 0]


def linearized_inversion(d, hydrophones_coords,hydrophone_pairs,inversion_params, sound_speed_mps, doplot=False):
    # convert parameters to numpy arrays
    damping_factor = np.array(inversion_params['damping_factor'])
    Tdelta_m = np.array(inversion_params['stop_delta_m'])
    V = np.array(sound_speed_mps)
    max_iteration = np.array(inversion_params['stop_max_iteration'])
    # define starting model(s)
    start_models = [inversion_params['start_model']]
    if inversion_params['start_model_repeats'] > 1:
        x_bounds = [hydrophones_coords['x'].min(), hydrophones_coords['x'].max()]
        y_bounds = [hydrophones_coords['y'].min(), hydrophones_coords['y'].max()]
        z_bounds = [hydrophones_coords['z'].min(), hydrophones_coords['z'].max()]
        for m_nb in range(1, inversion_params['start_model_repeats']):
            x_tmp = np.random.uniform(low=x_bounds[0], high=x_bounds[1])
            y_tmp = np.random.uniform(low=y_bounds[0], high=y_bounds[1])
            z_tmp = np.random.uniform(low=z_bounds[0], high=z_bounds[1])
            start_models.append([x_tmp, y_tmp, z_tmp])
    m_stack = []
    iterations_logs_stack = []
    data_misfit_stack = []
    for start_model in start_models:
        # current starting model
        m = pd.DataFrame({'x': [start_model[0]], 'y': [start_model[1]], 'z': [start_model[2]]})
        # Keeps track of values for each iteration
        iterations_logs = pd.DataFrame({
            'x': [start_model[0]],
            'y': [start_model[1]],
            'z': [start_model[2]],
            'norm': np.nan,
            'data_misfit': np.nan,
            })
        # Start iterations
        stop = False
        idx=0
        while stop == False:
            idx=idx+1
            # Linear inversion
            m_it, data_misfit_it = solve_iterative_ML(d, hydrophones_coords, hydrophone_pairs, m, V, damping_factor)
            # Save model and data misfit for each iteration
            iterations_logs = iterations_logs.append({
                'x': m_it['x'].values[0],
                'y': m_it['y'].values[0],
                'z': m_it['z'].values[0],
                #'norm': np.sqrt(np.square(m_it).sum(axis=1)).values[0],
                'norm': np.linalg.norm(m_it.iloc[0].to_numpy() - iterations_logs.iloc[-1][['x','y','z']].to_numpy(),np.inf),
                'data_misfit': data_misfit_it,
                }, ignore_index=True)
            # Update m
            m = m_it
            # stopping criteria
            if idx>1:
                 # norm of model diference
                #if (iterations_logs['norm'][idx] - iterations_logs['norm'][idx-1] <= Tdelta_m):
                if iterations_logs['norm'][idx] <= Tdelta_m:
                    stop = True
                    if np.isnan(m['x'].values[0]):
                        converged=False # Due to singular matrix
                        print('Singular matrix - inversion hasn''t converged.')
                    else:
                        converged=True
                elif idx > max_iteration: # max iterations exceeded
                    stop = True
                    converged=False
                    print('Max iterations reached - inversion hasn''t converged.')
                # elif iterations_logs['norm'][idx] > iterations_logs['norm'][idx-1]: # if norm starts increasing -> then stop there
                #     stop = True
                #     converged=False
                #     print('Norm increasing - inversion hasn''t converged.')
                elif np.isnan(m['x'].values[0]):
                    stop = True
                    converged=False
                    print('Singular matrix - inversion hasn''t converged.')

        if not converged:
            m['x']=np.nan
            m['y']=np.nan
            m['z']=np.nan
        # Save results for each starting model
        m_stack.append(m)
        iterations_logs_stack.append(iterations_logs)
        data_misfit_stack.append(iterations_logs['data_misfit'].iloc[-1])
    # Select results for starting model with best ending data misfit
    min_data_misfit = np.nanmin(data_misfit_stack)
    if np.isnan(min_data_misfit):
        min_idx=0
    else:
        min_idx = data_misfit_stack.index(min_data_misfit)
    return m_stack[min_idx], iterations_logs_stack[min_idx]