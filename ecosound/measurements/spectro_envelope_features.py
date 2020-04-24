# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""

from .measurer_builder import BaseClass
from ecosound.core.annotation import Annotation
from ecosound.core.spectrogram import Spectrogram
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.core.tools import resample_1D_array

import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
#from datetime import datetime
#import uuid
import matplotlib.pyplot as plt

class SpectroEnvelopeFeatures(BaseClass):

    measurer_parameters = ()
    
    def __init__(self, *args, **kwargs):
        # Initialize all measurer parameters to None
        self.__dict__.update(dict(zip(self.measurer_parameters,
                                      [None]*len(self.measurer_parameters))))
        # Unpack kwargs as measurer parameters if provided on instantiation
        self.__dict__.update(**kwargs)
    
    @property
    def name(self):
        """Return name of the measurer."""
        measurer_name = 'SpectroEnvelopeFeatures'
        return measurer_name
    
    @property
    def version(self):
        """Return version of the measurer."""
        version = '0.1'
        return version
    
    def _prerun_check(self, spectrogram, annotations):
        # check that all required arguments are defined
        if True in [self.__dict__.get(keys) is None for keys in self.measurer_parameters]:
            raise ValueError('Not all measurer parameters have been defined.'
                             + ' Required parameters: '
                             + str(self.measurer_parameters))
        # check that spectrogram is a spectrogram class
        if not isinstance(spectrogram, Spectrogram):
            raise ValueError('Input must be an ecosound Spectrogram object'
                             + '(ecosound.core.spectrogram).')
            
        # check that annotations is an Annotation class
        if not isinstance(annotations, Annotation):
            raise ValueError('Input must be an ecosound Annotation object'
                             + '(ecosound.core.annotation).')
    
    def compute(self, spectro, annotations, resolution_freq=0.1, resolution_time=0.001 ,debug=False):
        
        self._prerun_check(spectro, annotations)
        # ref for a single detection
        annot = annotations.data.iloc[9]
        tmin = annot['time_min_offset']
        tmax = annot['time_max_offset']
        fmin = annot['frequency_min']
        fmax = annot['frequency_max']
        # extract minmgram for that detection
        minigram = spectro.crop(frequency_min=fmin,frequency_max=fmax,time_min=tmin,time_max=tmax)
        # display detection gram
        graph = GrapherFactory('SoundPlotter', title='Detection', frequency_max=1000)
        graph.add_data(minigram)
        graph.show()
        
        # extract time and frequency envelops
        envelop_time, envelop_freq = self._get_envelops(minigram)
        
        # interpolate each envelop
        axis_t, envelop_time2 = resample_1D_array(minigram.axis_times, envelop_time, resolution=resolution_time, kind='quadratic')
        axis_f, envelop_freq2 = resample_1D_array(minigram.axis_frequencies, envelop_freq, resolution=resolution_freq,kind='quadratic')

        
        # envelop stats
        features_freq = self.envelop_features(axis_f, envelop_freq2)

        # plot - for debuging
        fig, ax = plt.subplots(1,2)
        ax[0].plot(axis_f,envelop_freq2,'.r')
        ax[0].plot(minigram.axis_frequencies,envelop_freq,'.g')

        # hide axes
        # fig.patch.set_visible(False)
        ax[1].axis('off')
        ax[1].axis('tight')
        ax[1].table(cellText=features_freq.values.T, rowLabels=features_freq.columns, loc='center', colWidths=[0.6,0.8])
        fig.tight_layout()
        # #plt.show()

        # # hide axes
        # fig.patch.set_visible(False)
        #ax[1].axis('off')
        #ax[1].axis('tight')
        # df = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD'))
        # ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        # fig.tight_layout()
        # plt.show()

        # plt.figure(4)
        # plt.plot(axis_t,envelop_time2,'.r')
        # plt.plot(minigram.axis_times,envelop_time,'.g')
        # plt.show()
        
        print('Works')
    
    
    def _get_envelops(self, minigram):
        envelop_freq = np.sum(minigram.spectrogram, axis=1)
        envelop_time = np.sum(minigram.spectrogram, axis=0)
        
        return envelop_time, envelop_freq

    def envelop_features(self, axis, values):

        # duration/width
        length= axis[-1] # feat

        # peak
        peak_value = np.amax(values)
        peak_location_unit = axis[np.where(values == peak_value)][0] # feat
        peak_location_relative = (peak_location_unit/length)*100 #feat

        # quartiles
        values_sum = np.sum(values)
        pct25 = 0.25 * values_sum
        pct50 = 0.5 * values_sum
        pct5 = 0.05 * values_sum
        pct95 = 0.95 * values_sum
        values_cumsum = np.cumsum(values)
        pct50_location_idx = np.where(values_cumsum > pct50)[0][0]
        pct50_location_unit = axis[pct50_location_idx] # feat
        pct75_location_unit = axis[np.where(values_cumsum > values_cumsum[pct50_location_idx]+pct25)][0] # feat
        pct25_location_unit = axis[np.where(values_cumsum > values_cumsum[pct50_location_idx]-pct25)][0] # feat
        inter_quart_range = pct75_location_unit - pct25_location_unit # feat
        asymmetry = (pct25_location_unit + pct75_location_unit-(2*pct50_location_unit))/(pct25_location_unit+pct75_location_unit) # feat

        pct5_location_unit = axis[np.where(values_cumsum > pct5)][0] #feat
        pct95_location_unit = axis[np.where(values_cumsum > pct95)][0] #feat
        length_90 =  pct95_location_unit - pct5_location_unit #feat

        # concentration
        sort_idx = np.argsort(-values)
        values_sorted = values[sort_idx]
        axis_sorted = axis[sort_idx]
        values_sorted_cumsum = np.cumsum(values_sorted)
        idx_pct50 = np.where(values_sorted_cumsum > pct50)[0][0]
        unit_min = np.min(axis_sorted[0:idx_pct50])
        unit_max = np.max(axis_sorted[0:idx_pct50])
        concentration_unit = unit_max - unit_min # feat
        
        # other stats
        std = np.std(values) # feat
        kurt = kurtosis(values) # feat
        skewness = skew(values) # feat
        
        # entropy
        aggregate_entropy = self.entropy(values)
        
        # gather all feature into DataFrame
        features = pd.DataFrame({
            'peak_location': [peak_location_unit],
            'peak_location_relative': [peak_location_relative],
            'length': [length],
            'length_90': [length_90],
            'pct5_location': [pct5_location_unit],
            'pct25_location': [pct25_location_unit],
            'pct50_location': [pct50_location_unit],
            'pct75_location': [pct75_location_unit],
            'pct95_location': [pct95_location_unit],
            'IQR': [inter_quart_range],
            'asymmetry': [asymmetry],
            'concentration': [concentration_unit],
            'std': [std],
            'kurtosis': [kurt],
            'skewness': [skewness],
            'entropy': [aggregate_entropy],
            })
        return features

    def entropy(self, array_1d, apply_square=False):
        """ 
        Aggregate entropy as defined in the Raven manual 
        apply_square = True, suqares the array value before calculation 
        """
        if apply_square:
            array_1d = np.square(array_1d)
        values_sum = np.sum(array_1d)
        H = 0
        for value in array_1d:
            ratio = (value/values_sum)
            H += ratio*np.log2(ratio)
        return H
        