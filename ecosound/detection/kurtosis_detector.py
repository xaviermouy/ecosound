# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:54:11 2017

-------------------------
TODO:

    - fix plot of timeseries
    - plotDetectionFunction with unit='samp' not implemented yet
    - 
--------------------------

@author: xavier
"""
import pandas as pd
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
import os
from numba import jit

class KurtosisDetector:

    def __init__(self, SoundObj, Kurtframe_sec, Kurtth, Kurtdelta_sec):        
        self.Kurtframe_sec = Kurtframe_sec
        self.Kurtth = Kurtth
        self.Kurtdelta_sec = Kurtdelta_sec
        self.SoundObj = SoundObj
        self.Kurtframe_samp = int(self.Kurtframe_sec*SoundObj.getSamplingFrequencyHz())
        self.Kurtdelta_samp = int(self.Kurtdelta_sec*SoundObj.getSamplingFrequencyHz())

    def run(self):
        Kurt = self.calcKurtosis()
        Peaks = findPeaks(Kurt, self.Kurtth, self.Kurtdelta_sec)
        detec = Detections()
        detec.output['fileName'] = self.SoundObj.getFileName()
        detec.output['filePath'] = self.SoundObj.getFilePath()
        detec.output['fileExtension'] = self.SoundObj.getFileExtension()
        detec.output['startTimeSec'] = Peaks['Tsec']
        detec.output['stopTimeSec'] = Peaks['Tsec']
        detec.output['startTimeSamp'] = Peaks['Tsamp']
        detec.output['stopTimeSamp'] = Peaks['Tsamp']
        detec.output['confidence'] = Peaks['Value']
        detec.output['detectorName'] = 'KurtosisDetector'
        detec.output['type'] = 'detec'
        detec.output['channel'] = self.SoundObj.getSelectedChannel()
        detec.output['fileDurationSec'] = self.SoundObj.getFileDur_sec()
        self.detecFunction = Kurt
        self.detections = detec.output
        return self.detections

    @jit
    def calcKurtosis(self):
        sig = self.SoundObj.getWaveform()
        K = np.array(sig)
        Tidx = np.array([])
        ss = pd.Series(sig)
        K = ss.rolling(window=self.Kurtframe_samp, center=False).kurt()
        K.fillna(0, inplace=True)
        Tidx = np.arange(0, len(K), 1)
        Tsec = Tidx/self.SoundObj.getSamplingFrequencyHz()
        Kurt = pd.DataFrame({'Value': K, 'Tsec': Tsec, 'Tsamp': Tidx})
        return Kurt
    
    def plot(self, displayDetections=True, unit='sec', newFig=True):
        if displayDetections:
            detections = self.detections
        else:
            detections = []        
        ylabel = 'Kurtosis'
        fig = Figure(self.detecFunction, detections=detections, ylabel=ylabel)
        fig.plotDetectionFunction(unit='sec', newFig=newFig)


class Detections:

    def __init__(self):
        data = pd.DataFrame({
            'type': 'detec',
            'channel': [],
            'detectorName': [],
            'fileName': [],
            'filePath': [],
            'fileExtension': [],
            'startTimeSec': [],
            'stopTimeSec': [],
            'startTimeSamp': [],
            'stopTimeSamp': [],
            'freqMinHz': [],
            'freqMaxHz': [],
            'confidence': [],
            'species': [],
            'call': [],
            'fileDurationSec': []
            })
        self.output = data

    def save2Pamlab(self, outdir):
        cols = ['fieldkey:', 'Soundfile', 'Channel', 'Sampling freq (Hz)', 'Latitude (deg)', 'Longitude (deg)', 'Recorder ID', 'Recorder depth', 'Start date and time (UTC)', 'Annotation date and time (local)', 'Recorder type', 'Deployment', 'Station', 'Operator', 'Left time (sec)', 'Right time (sec)', 'Top freq (Hz)', 'Bottom freq (Hz)', 'Species', 'Call type', 'rms SPL', 'SEL', '', '']
        annot = pd.DataFrame({'fieldkey:': 0, 'Soundfile': 0, 'Channel': 0, 'Sampling freq (Hz)': 0, 'Latitude (deg)': 0, 'Longitude (deg)': 0, 'Recorder ID': 0, 'Recorder depth': 0, 'Start date and time (UTC)': 0, 'Annotation date and time (local)': 0, 'Recorder type': 0, 'Deployment': 0, 'Station': 0, 'Operator': 0, 'Left time (sec)': 0, 'Right time (sec)': 0, 'Top freq (Hz)': 0, 'Bottom freq (Hz)': 0, 'Species': 0, 'Call type': 0, 'rms SPL': 0, 'SEL': 0, '': 0, '': 0}, index=list(range(self.output.shape[0])))    
        annot['fieldkey:'] = 'an:'
        annot['Species'] = self.output['species']
        annot['Call type'] = self.output['call']
        annot['Left time (sec)'] = self.output['startTimeSec']
        annot['Right time (sec)'] = self.output['stopTimeSec']
        annot['Top freq (Hz)'] = self.output['freqMaxHz']
        annot['Bottom freq (Hz)'] = self.output['freqMinHz']
        annot['rms SPL'] = self.output['confidence']
        annot['Operator'] =self.output['detectorName']
        annot['Channel'] =self.output['channel']
        if len(self.output.fileName) == 0:
            annot['Soundfile'] = os.path.join(str(self.output.filePath[0]), str(self.output.fileName[0])) + str(self.output.fileExtension[0])
        else:
            filenames=[]
            for i in range(0,len(self.output.fileName)):
                filenames.append(os.path.join(str(self.output.filePath[i]), str(self.output.fileName[i])) + str(self.output.fileExtension[i]))
            annot['Soundfile'] = filenames             
        annot.to_csv(os.path.join(outdir, str(self.output.fileName[0])) + str(self.output.fileExtension[0]) + ' chan' +  str(self.output.channel[0]) + ' annotations.log', sep='\t', encoding='utf-8', header=True, columns=cols, index=False)
                
    
    def save2Raven(self, outdir):      
        cols = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)', 'Begin Path', 'File Offset (s)', 'Begin File', 'File Duration (s)', 'Species', 'Sound type', 'Detector', 'Distortion (DTW)']
        annot = pd.DataFrame({'Selection': 0, 'View': 0, 'Channel': 0, 'Begin Time (s)': 0, 'End Time (s)': 0, 'Low Freq (Hz)': 0, 'High Freq (Hz)': 0, 'Begin Path': 0, 'File Offset (s)': 0, 'Begin File': 0, 'File Duration (s)': 0, 'Species': 0, 'Sound type': 0, 'Detector': 0, 'Distortion (DTW)': 0}, index=list(range(self.output.shape[0])))    
        annot['Selection'] = range(1,self.output.shape[0]+1)
        annot['View'] = 'Spectrogram 1'
        annot['Channel'] = self.output['channel'] + 1
        annot['Begin Time (s)'] = self.output['startTimeSec']
        annot['End Time (s)'] = self.output['stopTimeSec']
        annot['Low Freq (Hz)'] = self.output['freqMinHz']
        annot['High Freq (Hz)'] = self.output['freqMaxHz']
        annot['Begin Path'] = os.path.join(str(self.output.filePath[0]), str(self.output.fileName[0])) + str(self.output.fileExtension[0])
        annot['File Offset (s)'] = self.output['startTimeSec']
        annot['Begin File'] = str(self.output.fileName[0]) + str(self.output.fileExtension[0])
        annot['File Duration (s)'] = self.output['fileDurationSec']
        annot['Species'] = self.output['species']
        annot['Sound type'] = self.output['call']
        annot['Detector'] =self.output['detectorName']
        annot['Distortion (DTW)'] = self.output['confidence']
        annot.to_csv(os.path.join(outdir, str(self.output.fileName[0])) + str(self.output.fileExtension[0]) + '.chan' +  str(self.output.channel[0]) + '.Table.1.selections.txt', sep='\t', encoding='utf-8', header=True, columns=cols, index=False)

class Figure:
    def __init__(self, detecFunction, detections=[], ylabel='Detection value'):
        self.axis_t_sec = detecFunction['Tsec'].values
        self.axis_t_samp = detecFunction['Tsamp'].values
        self.timeseries = detecFunction['Value'].values
        self.ylabel = ylabel
        self.detections = detections

    def plotDetectionFunction(self, unit='sec', newFig=True):
        if newFig:
            plt.figure()
        if unit == 'sec':
            xaxis = self.axis_t_sec
            xlabel = 'Time (sec)'
        elif unit == 'samp':
            xaxis = self.axis_t_samp
            xlabel = 'Time (sample)'
        else:
            raise ValueError('Invalid unit value. Should be "sec" or "samp"')
        # plot figure
        plt.plot(xaxis, self.timeseries, color='black')
        plt.xlabel(xlabel)
        plt.ylabel(self.ylabel)
        plt.axis([xaxis[0], xaxis[-1], min(self.timeseries), max(self.timeseries)])
        if len(self.detections) > 0:
            plt.plot(self.detections['startTimeSec'], self.detections['confidence'], '.r')
            plt.legend('Detections')
        plt.grid()
        plt.show()


@jit
def findPeaks(Kurt, threshold, delta_sec):
    # find peaks
    Pidx = spsig.argrelmax(Kurt['Value'].values)
    Pidx = Pidx[0][:]
    Pval = Kurt['Value'].values[Pidx]
    # only keep peaks >= threshold
    idx = Pval >= threshold
    Pidx = Pidx[idx]
    Pval = Pval[idx]
    Psec = Kurt.ix[Pidx]['Tsec'].values
    Psamp = Kurt.ix[Pidx]['Tsamp'].values
    Peaks = pd.DataFrame()
    Peaks['Value'] = []
    Peaks['Tsamp'] = []
    Peaks['Tsec'] = []
    while len(Pval) > 0:
        dt = Psec-Psec[0]
        idx = dt < delta_sec
        tmp = pd.DataFrame({'Value': [Pval[0]], 'Tsamp': [Psamp[0]], 'Tsec': [Psec[0]]})
        Peaks = pd.DataFrame.append(Peaks, tmp, ignore_index=True)
        for i in range(sum(idx)):
            Pidx = np.delete(Pidx, [0])
            Pval = np.delete(Pval, [0])
            Psec = np.delete(Psec, [0])
            Psamp = np.delete(Psamp, [0])
    return Peaks