# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""

from .measurer_builder import BaseClass
from ecosound.core.annotation import Annotation
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement
# from ecosound.visualization.grapher_builder import GrapherFactory
import ecosound.core.tools
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats.mstats import gmean
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numba import njit
from dask import delayed, compute, visualize

class SpectrogramFeatures(BaseClass):
    """Spectrogram features.

    This class extracts a set of spectral and temporal features from a
    spectrogram. It is based on measurements extracted by the software Raven
    and other measurements described in the R package Seewave by Jerome Sueur.

    The SpectrogramFeatures measurer must be instantiated using the
    MeasurerFactory with the positional argument 'SpectrogramFeatures':

    from ecosound.measurements.measurer_builder import MeasurerFactory
    spectro_features = MeasurerFactory('SpectrogramFeatures',
                                       resolution_time=0.001,
                                       resolution_freq=0.1,
                                       interp='linear')
    measurements = spectro_features.compute(spectro, detections,
                                            debug=False,
                                            verbose=False)

    The Measurement object returned has all the features appended to the
    original annotation fields in the pandas datafrane measurment.data.
    Measurer's name, version and features' name are in the pandas Dataframe
    measurement.metadata. Spectrogram features include:

        1- 'freq_peak': peak frequency in the frequency envelop, in Hz.
        2- 'freq_bandwidth': Bandwidth of the frequency envelop, in Hz.
        3- 'freq_bandwidth90': 90% bandwidth of the frequency envelop, in Hz.
        4- 'freq_pct5': frequency of the 5th percentile, in the frequency envelope, in Hz.
        5- 'freq_pct25': frequency of the 25th percentile, in the frequency envelope, in Hz.
        6- 'freq_pct50': frequency of the 50th percentile, in the frequency envelope, in Hz.
        7- 'freq_pct75': frequency of the 75th percentile, in the frequency envelope, in Hz.
        8- 'freq_pct95': frequency of the 95th percentile, in the frequency envelope, in Hz.
        9- 'freq_iqr': inter quartile range of the frequency envelope, in Hz.
       10- 'freq_asymmetry': symmetry of the frequency envelope.
       11- 'freq_concentration': concentration of the frequency envelope.
       12- 'freq_std': standard deciation of the frequency envelope.
       13- 'freq_kurtosis': kurtosis of the frequency envelope.
       14- 'freq_skewness': skewness of the frequency envelope.
       15- 'freq_entropy': Shannon's entropy of the frequency envelope.
       16- 'freq_flatness': flatness of the frequency envelope.
       17- 'freq_roughness': roughness of the frequency envelope.
       18- 'freq_centroid': centroid of the frequency envelope, in Hz.
       19- 'freq_overall_peak': overall peak frequency in the spectrogram, in Hz.
       20- 'freq_median_mean': mean of the median frequency through the spectrogram time slices, Hz.
       21- 'freq_median_std': standard deviation of the median frequency through the spectrogram time slices, Hz.
       22- 'freq_entropy_mean': mean of entropy through the spectrogram time slices, Hz.
       23- 'freq_entropy_std': std of the entropy through the spectrogram time slices, Hz.
       24- 'freq_upsweep_mean': frequency upsweep mean index
       25- 'freq_upsweep_fraction': frequency upsweep fraction
       26- 'snr': signal to noise ratio, in dB
       27- 'time_peak_sec': time of peak in the time envelope, in sec.
       28- 'time_peak_perc': relative time of peak in the time envelope.
       29- 'time_duration': duration of the time envelope, in sec.
       30- 'time_duration90': 90% duration of the time envelop, in sec.
       31- 'time_pct5': time of the 5th percentile, in the time envelope, in sec.
       32- 'time_pct25': time of the 25th percentile, in the time envelope, in sec.
       33- 'time_pct50': time of the 50th percentile, in the time envelope, in sec.
       34- 'time_pct75': time of the 75th percentile, in the time envelope, in sec.
       35- 'time_pct95': time of the 95th percentile, in the time envelope, in sec.
       36- 'time_iqr': inter quartile range of the time envelope, in sec.
       37- 'time_asymmetry': symmetry of the time envelope.
       38- 'time_concentration': concentration of the time envelope.
       39- 'time_std': standard deciation of the time envelope.
       40- 'time_kurtosis': kurtosis of the time envelope.
       41- 'time_skewness': skewness of the time envelope.
       42- 'time_entropy': Shannon's entropy of the time envelope.
       43- 'time_flatness': flatness of the time envelope.
       44- 'time_roughness': roughness of the time envelope.
       45- 'time_centroid': centroid of the time envelope, in sec.

    Attributes
    ----------
    name : str
        Name of the measurer
    version : str
        Version of the measurer
    resolution_freq : float
        frequency resolution of the interpolated spectral envelope, in Hz.
        Default is 0.1.
    resolution_time : float
        Time resolution of the interpolated temporal envelope, in seconds.
        Default is 0.001.
    interp : str
        Type of interpolation method for interpolating time and frequency
        envelopes. Can be 'linear' or 'quadratic'. Default is 'linear'.

    Methods
    -------
    run(spectrogram, detections, debug=True, verbose=False debug=False)
        Calculate spectrogram features each detection in the spectrogram.

    """

    measurer_parameters = ('resolution_freq',
                           'resolution_time',
                           'interp',
                           )

    def __init__(self, *args, **kwargs):
        """
        Initialize the measurer.

        Parameters
        ----------
        *args : str
            Do not use. Only used by the MeasurerFactory.
        resolution_freq : float, optional
            frequency resolution of the interpolated spectral envelope, in Hz.
            Default is 0.1.
        resolution_time : float, optional
            Time resolution of the interpolated temporal envelope, in seconds.
            Default is 0.001.
        interp : str, optional
            Type of interpolation method for interpolating time and frequency
            envelopes. Can be 'linear' or 'quadratic'. Default is 'linear'.

        Returns
        -------
        None. Measurer object.

        """
        # Initialize all measurer parameters to None
        self.__dict__.update(dict(zip(self.measurer_parameters,
                                      [None]*len(self.measurer_parameters))))
        # default values:
        self.resolution_time = 0.001
        self.resolution_freq = 0.1
        self.interp = 'linear'
        # Unpack kwargs as measurer parameters if provided on instantiation
        self.__dict__.update(**kwargs)

    @property
    def name(self):
        """Return name of the measurer."""
        measurer_name = 'SpectrogramFeatures'
        return measurer_name

    @property
    def version(self):
        """Return version of the measurer."""
        version = '0.1'
        return version

    def _prerun_check(self, spectrogram, annotations):
        """Run several verifications before the run."""
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

    def compute_old(self, spectro, annotations, debug=False, verbose=False):
        """ Compute spectrogram features.

        Goes through each annotation and compute features from the spectrogram.

        Parameters
        ----------
        spectro : ecosound Spectrogram object
            Spectrogram of the recording to analyze.
        annotations : ecosound Annotation object
            Annotations of the sounds to measure. Can be from manual analysis
            or from an automatic detector.
        debug : bool, optional
            Displays figures for each annotation with the spectrogram, spectral
            and time envelopes, and tables with all associated measurements.
            The default is False.
        verbose : bool, optional
            Prints in the console the annotation being processed. The default
            is False.

        Returns
        -------
        measurements : ecosound Measurement object
            Measurement object containing the measurements appended to the
            original annotation fields. Measurements are in the .data data
            frame. Metadata with mearurer name, version and measurements names
            are in the .metadata datafreame.

        """
        self._prerun_check(spectro, annotations)
        # loop through each annotation
        for index, annot in annotations.data.iterrows():
            if verbose:
                print('processing annotation ', index)
            tmin = annot['time_min_offset']
            tmax = annot['time_max_offset']
            fmin = annot['frequency_min']
            fmax = annot['frequency_max']
            # extract minmgram for that detection
            minigram = spectro.crop(frequency_min=fmin,
                                    frequency_max=fmax,
                                    time_min=tmin,
                                    time_max=tmax)
            # extract time and frequency envelops
            envelop_time, envelop_freq = SpectrogramFeatures.get_envelops(minigram,
                                                            normalize=True)
            # interpolate each envelop
            axis_t, envelop_time2 = ecosound.core.tools.resample_1D_array(
                minigram.axis_times,
                envelop_time,
                resolution=self.resolution_time,
                kind=self.interp)
            axis_f, envelop_freq2 = ecosound.core.tools.resample_1D_array(
                minigram.axis_frequencies,
                envelop_freq,
                resolution=self.resolution_freq,
                kind=self.interp)
            if sum(envelop_freq2)==0:
                print('here')
            if sum(envelop_time2)==0:
                print('here')
            # Frequency envelop features
            features_envelop_freq = self.envelop_features(axis_f, envelop_freq2)
            if debug:
                axis_orig = minigram.axis_frequencies
                envelop_orig = envelop_freq
                axis_interp = axis_f
                envelop_interp = envelop_freq2
                features = features_envelop_freq
                title = 'Frequency envelop'
                SpectrogramFeatures._plot_envelop_features(axis_orig,
                                           envelop_orig,
                                           axis_interp,
                                           envelop_interp,
                                           features,
                                           title=title)
            # Time envelop features
            features_envelop_time = self.envelop_features(axis_t, envelop_time2)
            if debug:
                axis_orig = minigram.axis_times
                envelop_orig = envelop_time
                axis_interp = axis_t
                envelop_interp = envelop_time2
                features = features_envelop_time
                title = 'Time envelop'
                SpectrogramFeatures._plot_envelop_features(axis_orig,
                                           envelop_orig,
                                           axis_interp,
                                           envelop_interp,
                                           features,
                                           title=title)
            # Amplitude modulation features
            # TO DO
            # Full spectrogram matrix features
            adjusted_bounds = [features_envelop_time['pct5_position'].values[0],
                               features_envelop_time['pct95_position'].values[0],
                               features_envelop_freq['pct5_position'].values[0],
                               features_envelop_freq['pct95_position'].values[0],
                               ]
            features_spectrogram, frequency_points = self.spectrogram_features(minigram,adjusted_bounds=adjusted_bounds)
            if debug:
                SpectrogramFeatures._plot_spectrogram_features(minigram,
                                               features_spectrogram,
                                               adjusted_bounds,
                                               frequency_points,
                                               title='spectrogram features')
            # stack all features
            tmp = pd.DataFrame({
                'uuid': [annot['uuid']],
                # from frequency envelop
                'freq_peak': features_envelop_freq['peak_position'],
                'freq_bandwidth': features_envelop_freq['length'],
                'freq_bandwidth90': features_envelop_freq['length_90'],
                'freq_pct5': features_envelop_freq['pct5_position'],
                'freq_pct25': features_envelop_freq['pct25_position'],
                'freq_pct50': features_envelop_freq['pct50_position'],
                'freq_pct75': features_envelop_freq['pct75_position'],
                'freq_pct95': features_envelop_freq['pct95_position'],
                'freq_iqr': features_envelop_freq['IQR'],
                'freq_asymmetry': features_envelop_freq['asymmetry'],
                'freq_concentration': features_envelop_freq['concentration'],
                'freq_std': features_envelop_freq['std'],
                'freq_kurtosis': features_envelop_freq['kurtosis'],
                'freq_skewness': features_envelop_freq['skewness'],
                'freq_entropy': features_envelop_freq['entropy'],
                'freq_flatness': features_envelop_freq['flatness'],
                'freq_roughness': features_envelop_freq['roughness'],
                'freq_centroid': features_envelop_freq['centroid'],
                # from full spectrogram
                'freq_overall_peak': features_spectrogram['freq_peak'],
                'freq_median_mean': features_spectrogram['freq_median_mean'],
                'freq_median_std': features_spectrogram['freq_median_std'],
                'freq_entropy_mean': features_spectrogram['freq_entropy_mean'],
                'freq_entropy_std': features_spectrogram['freq_entropy_std'],
                'freq_upsweep_mean': features_spectrogram['freq_upsweep_mean'],
                'freq_upsweep_fraction': features_spectrogram['freq_upsweep_fraction'],
                'snr': features_spectrogram['snr'],
                # from time envelop
                'time_peak_sec': features_envelop_time['peak_position'],
                'time_peak_perc': features_envelop_time['peak_position_relative'],
                'time_duration': features_envelop_time['length'],
                'time_duration90': features_envelop_time['length_90'],
                'time_pct5': features_envelop_time['pct5_position'],
                'time_pct25': features_envelop_time['pct25_position'],
                'time_pct50': features_envelop_time['pct50_position'],
                'time_pct75': features_envelop_time['pct75_position'],
                'time_pct95': features_envelop_time['pct95_position'],
                'time_iqr': features_envelop_time['IQR'],
                'time_asymmetry': features_envelop_time['asymmetry'],
                'time_concentration': features_envelop_time['concentration'],
                'time_std': features_envelop_time['std'],
                'time_kurtosis': features_envelop_time['kurtosis'],
                'time_skewness': features_envelop_time['skewness'],
                'time_entropy': features_envelop_time['entropy'],
                'time_flatness': features_envelop_time['flatness'],
                'time_roughness': features_envelop_time['roughness'],
                'time_centroid': features_envelop_time['centroid'],
                })
            # stack features for each annotation
            if index == 0:
                features = tmp
                features_name = list(features.columns)
                features_name.remove('uuid')
            else:
                features = pd.concat([features, tmp], ignore_index=False)
        # merge with annotation fields
        annotations.data.set_index('uuid', inplace=True, drop=False)
        features.set_index('uuid', inplace=True, drop=True)
        meas = pd.concat([annotations.data, features], axis=1, join='inner')
        meas.reset_index(drop=True, inplace=True)
        # create Measurement object
        measurements = Measurement(measurer_name=self.name,
                                   measurer_version=self.version,
                                   measurements_name=features_name)
        measurements.data = meas
        return measurements
    
    def compute(self, spectro, annotations, debug=False, verbose=False, use_dask=False):
        """ Compute spectrogram features.

        Goes through each annotation and compute features from the spectrogram.

        Parameters
        ----------
        spectro : ecosound Spectrogram object
            Spectrogram of the recording to analyze.
        annotations : ecosound Annotation object
            Annotations of the sounds to measure. Can be from manual analysis
            or from an automatic detector.
        debug : bool, optional
            Displays figures for each annotation with the spectrogram, spectral
            and time envelopes, and tables with all associated measurements.
            The default is False.
        verbose : bool, optional
            Prints in the console the annotation being processed. The default
            is False.

        Returns
        -------
        measurements : ecosound Measurement object
            Measurement object containing the measurements appended to the
            original annotation fields. Measurements are in the .data data
            frame. Metadata with mearurer name, version and measurements names
            are in the .metadata datafreame.

        """
        self._prerun_check(spectro, annotations)
        
        #init
        features = self._init_dataframe()
        features_name = list(features.columns)
        # loop through each annotation
        df_list=[]
        for index, annot in annotations.data.iterrows():
            if verbose:
                print('processing annotation ', index, annot['time_min_offset'], '-' ,annot['time_max_offset'])
            # feature for 1 annot
            # tmp = self.compute_single_annot(annot, spectro, debug, verbose)
            # # stack features for each annotation
            # features = pd.concat([features, tmp], ignore_index=False)
            
            # feature for 1 annot
            if use_dask:
                df = delayed(self.compute_single_annot)(annot, spectro, debug)
            else:
                df = self.compute_single_annot(annot, spectro, debug)
            # stack features for each annotation
            df_list.append(df)
        if use_dask:
            features = delayed(pd.concat)(df_list, ignore_index=False)
            #features.visualize('measuremnets')
            features = features.compute()
        else:
            features = pd.concat(df_list, ignore_index=False)
        # merge with annotation fields
        annotations.data.set_index('uuid', inplace=True, drop=False)
        features.set_index('uuid', inplace=True, drop=True)
        meas = pd.concat([annotations.data, features], axis=1, join='inner')
        meas.reset_index(drop=True, inplace=True)
        # create Measurement object
        measurements = Measurement(measurer_name=self.name,
                                   measurer_version=self.version,
                                   measurements_name=features_name)
        measurements.data = meas
        return measurements
    
    def _init_dataframe(self):
        tmp = pd.DataFrame({
                'uuid': [],
                # from frequency envelop
                'freq_peak': [],
                'freq_bandwidth': [],
                'freq_bandwidth90': [],
                'freq_pct5': [],
                'freq_pct25': [],
                'freq_pct50': [],
                'freq_pct75': [],
                'freq_pct95': [],
                'freq_iqr': [],
                'freq_asymmetry': [],
                'freq_concentration': [],
                'freq_std': [],
                'freq_kurtosis': [],
                'freq_skewness': [],
                'freq_entropy': [],
                'freq_flatness': [],
                'freq_roughness': [],
                'freq_centroid': [],
                # from full spectrogram
                'freq_overall_peak': [],
                'freq_median_mean': [],
                'freq_median_std': [],
                'freq_entropy_mean': [],
                'freq_entropy_std': [],
                'freq_upsweep_mean': [],
                'freq_upsweep_fraction': [],
                'snr': [],
                # from time envelop
                'time_peak_sec': [],
                'time_peak_perc': [],
                'time_duration': [],
                'time_duration90': [],
                'time_pct5': [],
                'time_pct25': [],
                'time_pct50': [],
                'time_pct75': [],
                'time_pct95': [],
                'time_iqr': [],
                'time_asymmetry': [],
                'time_concentration': [],
                'time_std': [],
                'time_kurtosis': [],
                'time_skewness': [],
                'time_entropy': [],
                'time_flatness': [],
                'time_roughness': [],
                'time_centroid': [],
                })
        return tmp
    def compute_single_annot(self, annot, spectro, debug):
            tmin = annot['time_min_offset']
            tmax = annot['time_max_offset']
            fmin = annot['frequency_min']
            fmax = annot['frequency_max']
            # extract minmgram for that detection
            minigram = spectro.crop(frequency_min=fmin,
                                    frequency_max=fmax,
                                    time_min=tmin,
                                    time_max=tmax)
            if minigram.spectrogram.any():
                # extract time and frequency envelops
                envelop_time, envelop_freq = SpectrogramFeatures.get_envelops(minigram,
                                                                normalize=True)
                # interpolate each envelop
                axis_t, envelop_time2 = ecosound.core.tools.resample_1D_array(
                    minigram.axis_times,
                    envelop_time,
                    resolution=self.resolution_time,
                    kind=self.interp)
                axis_f, envelop_freq2 = ecosound.core.tools.resample_1D_array(
                    minigram.axis_frequencies,
                    envelop_freq,
                    resolution=self.resolution_freq,
                    kind=self.interp)
                if sum(envelop_freq2)==0:
                    print('here')
                if sum(envelop_time2)==0:
                    print('here')
                # Frequency envelop features
                features_envelop_freq = self.envelop_features(axis_f, envelop_freq2)
                if debug:
                    axis_orig = minigram.axis_frequencies
                    envelop_orig = envelop_freq
                    axis_interp = axis_f
                    envelop_interp = envelop_freq2
                    features = features_envelop_freq
                    title = 'Frequency envelop'
                    SpectrogramFeatures._plot_envelop_features(axis_orig,
                                               envelop_orig,
                                               axis_interp,
                                               envelop_interp,
                                               features,
                                               title=title)
                # Time envelop features
                features_envelop_time = self.envelop_features(axis_t, envelop_time2)
                if debug:
                    axis_orig = minigram.axis_times
                    envelop_orig = envelop_time
                    axis_interp = axis_t
                    envelop_interp = envelop_time2
                    features = features_envelop_time
                    title = 'Time envelop'
                    SpectrogramFeatures._plot_envelop_features(axis_orig,
                                               envelop_orig,
                                               axis_interp,
                                               envelop_interp,
                                               features,
                                               title=title)
                # Amplitude modulation features
                # TO DO
                # Full spectrogram matrix features
                adjusted_bounds = [features_envelop_time['pct5_position'].values[0],
                                   features_envelop_time['pct95_position'].values[0],
                                   features_envelop_freq['pct5_position'].values[0],
                                   features_envelop_freq['pct95_position'].values[0],
                                   ]
                features_spectrogram, frequency_points = self.spectrogram_features(minigram,adjusted_bounds=adjusted_bounds)
                if debug:
                    SpectrogramFeatures._plot_spectrogram_features(minigram,
                                                   features_spectrogram,
                                                   adjusted_bounds,
                                                   frequency_points,
                                                   title='spectrogram features')
                # stack all features
                tmp = pd.DataFrame({
                    'uuid': [annot['uuid']],
                    # from frequency envelop
                    'freq_peak': features_envelop_freq['peak_position'],
                    'freq_bandwidth': features_envelop_freq['length'],
                    'freq_bandwidth90': features_envelop_freq['length_90'],
                    'freq_pct5': features_envelop_freq['pct5_position'],
                    'freq_pct25': features_envelop_freq['pct25_position'],
                    'freq_pct50': features_envelop_freq['pct50_position'],
                    'freq_pct75': features_envelop_freq['pct75_position'],
                    'freq_pct95': features_envelop_freq['pct95_position'],
                    'freq_iqr': features_envelop_freq['IQR'],
                    'freq_asymmetry': features_envelop_freq['asymmetry'],
                    'freq_concentration': features_envelop_freq['concentration'],
                    'freq_std': features_envelop_freq['std'],
                    'freq_kurtosis': features_envelop_freq['kurtosis'],
                    'freq_skewness': features_envelop_freq['skewness'],
                    'freq_entropy': features_envelop_freq['entropy'],
                    'freq_flatness': features_envelop_freq['flatness'],
                    'freq_roughness': features_envelop_freq['roughness'],
                    'freq_centroid': features_envelop_freq['centroid'],
                    # from full spectrogram
                    'freq_overall_peak': features_spectrogram['freq_peak'],
                    'freq_median_mean': features_spectrogram['freq_median_mean'],
                    'freq_median_std': features_spectrogram['freq_median_std'],
                    'freq_entropy_mean': features_spectrogram['freq_entropy_mean'],
                    'freq_entropy_std': features_spectrogram['freq_entropy_std'],
                    'freq_upsweep_mean': features_spectrogram['freq_upsweep_mean'],
                    'freq_upsweep_fraction': features_spectrogram['freq_upsweep_fraction'],
                    'snr': features_spectrogram['snr'],
                    # from time envelop
                    'time_peak_sec': features_envelop_time['peak_position'],
                    'time_peak_perc': features_envelop_time['peak_position_relative'],
                    'time_duration': features_envelop_time['length'],
                    'time_duration90': features_envelop_time['length_90'],
                    'time_pct5': features_envelop_time['pct5_position'],
                    'time_pct25': features_envelop_time['pct25_position'],
                    'time_pct50': features_envelop_time['pct50_position'],
                    'time_pct75': features_envelop_time['pct75_position'],
                    'time_pct95': features_envelop_time['pct95_position'],
                    'time_iqr': features_envelop_time['IQR'],
                    'time_asymmetry': features_envelop_time['asymmetry'],
                    'time_concentration': features_envelop_time['concentration'],
                    'time_std': features_envelop_time['std'],
                    'time_kurtosis': features_envelop_time['kurtosis'],
                    'time_skewness': features_envelop_time['skewness'],
                    'time_entropy': features_envelop_time['entropy'],
                    'time_flatness': features_envelop_time['flatness'],
                    'time_roughness': features_envelop_time['roughness'],
                    'time_centroid': features_envelop_time['centroid'],
                    })
            else:
                tmp = pd.DataFrame({
                    'uuid': [annot['uuid']],
                    # from frequency envelop
                    'freq_peak': np.nan,
                    'freq_bandwidth': np.nan,
                    'freq_bandwidth90': np.nan,
                    'freq_pct5': np.nan,
                    'freq_pct25': np.nan,
                    'freq_pct50': np.nan,
                    'freq_pct75': np.nan,
                    'freq_pct95': np.nan,
                    'freq_iqr': np.nan,
                    'freq_asymmetry': np.nan,
                    'freq_concentration': np.nan,
                    'freq_std': np.nan,
                    'freq_kurtosis': np.nan,
                    'freq_skewness': np.nan,
                    'freq_entropy': np.nan,
                    'freq_flatness': np.nan,
                    'freq_roughness': np.nan,
                    'freq_centroid': np.nan,
                    # from full spectrogram
                    'freq_overall_peak': np.nan,
                    'freq_median_mean': np.nan,
                    'freq_median_std': np.nan,
                    'freq_entropy_mean': np.nan,
                    'freq_entropy_std': np.nan,
                    'freq_upsweep_mean': np.nan,
                    'freq_upsweep_fraction': np.nan,
                    'snr': np.nan,
                    # from time envelop
                    'time_peak_sec': np.nan,
                    'time_peak_perc': np.nan,
                    'time_duration': np.nan,
                    'time_duration90': np.nan,
                    'time_pct5': np.nan,
                    'time_pct25': np.nan,
                    'time_pct50': np.nan,
                    'time_pct75': np.nan,
                    'time_pct95': np.nan,
                    'time_iqr': np.nan,
                    'time_asymmetry': np.nan,
                    'time_concentration': np.nan,
                    'time_std': np.nan,
                    'time_kurtosis': np.nan,
                    'time_skewness': np.nan,
                    'time_entropy': np.nan,
                    'time_flatness': np.nan,
                    'time_roughness': np.nan,
                    'time_centroid': np.nan,
                    })
            return tmp

    def envelop_features(self, axis, values):
        """Extract fetaures from time or frequency envelop.

        These measurements are mostly based on Mellinger and Bradbury, 2007:
        Mellinger, D.K. and J.W. Bradbury. 2007. Acoustic measurement of marine
        mammal sounds in noisy environments. Proceedings of the Second
        International Conference on Underwater Acoustic Measurements:
        Technologies and Results, Heraklion, Greece, pp. 273-280. ftp://ftp.
        pmel.noaa.gov/newport/mellinger/papers/Mellinger+Bradbury07-Bioacoustic
        MeasurementInNoise-UAM,Crete.pdf.

        Measurements include:
            1- peak_position
            2- peak_position_relative
            3- length
            4- length_90
            5- pct5_position
            6- pct25_position
            7- pct50_position
            8- pct75_position
            9- pct95_position
           10- IQR
           11- asymmetry
           12- concentration
           13- std
           14- kurtosis
           15- skewness
           16- entropy
           17- flatness
           18- roughness
           19- centroid

        Parameters
        ----------
        axis : numpy array
            axis of the envelope in Hz or seconds.
        values : numpy array
            time of frequency envelope. Has the same length as axis.

        Returns
        -------
        features : pandas dataframe
            Dataframe with measurmenets of the envelope.

        """
        # peak
        peak_value, peak_position_unit, peak_position_relative = SpectrogramFeatures.peak(values, axis)
        # Position of percentiles
        percentiles_value = [5, 25, 50, 75, 95]
        percentiles_position = SpectrogramFeatures.percentiles_position(values, percentiles_value, axis=axis)
        # Inter quartile range
        inter_quart_range = percentiles_position['75'] - percentiles_position['25'] 
        # Asymetry
        asymmetry = SpectrogramFeatures.asymmetry(percentiles_position['25'], percentiles_position['50'], percentiles_position['75'])
        # duration/width
        length = SpectrogramFeatures.length(values, axis[1]-axis[0])  
        # duration/width containing 90% of magnitude
        length_90 = percentiles_position['95'] - percentiles_position['5']  
        # concentration
        concentration_unit = SpectrogramFeatures.concentration(values, axis)
        # standard deviation
        std = np.std(values)  # feat
        # kusrtosis
        kurt = kurtosis(values)  # feat
        # skewness
        skewness = skew(values)  # feat
        # entropy
        entropy = ecosound.core.tools.entropy(values)
        # flatness - spectral flatness (0: tone, 1: white noise) (see seewave)
        flatness = SpectrogramFeatures.flatness(values)
        # roughness or total curvature of a curve
        roughness = SpectrogramFeatures.roughness(values)
        # centroid
        centroid = SpectrogramFeatures.centroid(values, axis)
        # gather all feature into DataFrame
        features = pd.DataFrame({
            'peak_position': [peak_position_unit],
            'peak_position_relative': [peak_position_relative],
            'length': [length],
            'length_90': [length_90],
            'pct5_position': [percentiles_position['5']],
            'pct25_position': [percentiles_position['25']],
            'pct50_position': [percentiles_position['50']],
            'pct75_position': [percentiles_position['75']],
            'pct95_position': [percentiles_position['95']],
            'IQR': [inter_quart_range],
            'asymmetry': [asymmetry],
            'concentration': [concentration_unit],
            'std': [std],
            'kurtosis': [kurt],
            'skewness': [skewness],
            'entropy': [entropy],
            'flatness': [flatness],
            'roughness': [roughness],
            'centroid': [centroid],
            })
        return features

    def spectrogram_features(self, minigram1, adjusted_bounds=None):
        """Extract fetaures from the spectrogram.

        These measurements are mostly based on Mellinger and Bradbury, 2007:
        Mellinger, D.K. and J.W. Bradbury. 2007. Acoustic measurement of marine
        mammal sounds in noisy environments. Proceedings of the Second
        International Conference on Underwater Acoustic Measurements:
        Technologies and Results, Heraklion, Greece, pp. 273-280. ftp://ftp.
        pmel.noaa.gov/newport/mellinger/papers/Mellinger+Bradbury07-Bioacoustic
        MeasurementInNoise-UAM,Crete.pdf.

        Measurements include:
           1- freq_peak
           2- freq_median_mean
           3- freq_median_std
           4- freq_entropy_mean
           5- freq_entropy_std
           6- freq_upsweep_mean
           7- freq_upsweep_fraction
           8- snr

        Parameters
        ----------
        minigram1 : ecosound Spectrogram object
            Spectrogram of the sound to analyse.
        adjusted_bounds : list, optional
            List with defining the 90% energy time-frequency window for the 
            measurmenets.
            adjusted_bounds = [Time min., Time max., Freq. min., Freq. max.]. 
            Times is seconds, frequencies in Hz. The default is None.

        Returns
        -------
        features : pandas dataframe
            dataframe with spectrogram measuremnets.
        frequency_points : pandas dataframe
            Dataframe with the median and peak frequency vectors with their
            time axis vector. Only used for plotting and debugging purposes.

        """
        if adjusted_bounds:
            minigram = minigram1.crop(time_min=adjusted_bounds[0],
                                      time_max=adjusted_bounds[1],
                                      frequency_min=adjusted_bounds[2],
                                      frequency_max=adjusted_bounds[3],
                                      inplace=False,
                                      )
        else:
            minigram = minigram1
        spectro = minigram.spectrogram.transpose()
        # Spectrum for each time framee
        peak_f = []
        peak_amp = []
        median_f = []
        entropy_agg = []
        
        if spectro.shape[1] > 1: #  must be at least 1 bin of bandwidth
            #root4_magnitude = []
            for spectrum in spectro:
                axis_f, spectrum2 = ecosound.core.tools.resample_1D_array(
                    minigram.axis_frequencies,
                    spectrum,
                    resolution=self.resolution_freq,
                    kind=self.interp)
                if sum(spectrum)>0:
                    # track peak frequency
                    peak_value, peak_position, _ = SpectrogramFeatures.peak(spectrum2, axis_f)
                    peak_amp.append(peak_value)
                    peak_f.append(peak_position)
                    # track median frequency
                    pct50_position = SpectrogramFeatures.percentiles_position(spectrum2,[50],axis_f)['50']
                    median_f.append(pct50_position)
                    # entropy
                    entropy_agg.append(ecosound.core.tools.entropy(spectrum))
                    #root4_magnitude.append(np.power(np.sum(spectrum2), 1/4))
            if len(median_f) > 1:
                # overall frequency peak
                _, freq_peak, _ = SpectrogramFeatures.peak(peak_amp, peak_f)
                # mean of median frequency track
                freq_median_mean = np.nanmean(median_f)
                # standard deviation of median frequency track
                freq_median_std = np.nanstd(median_f)
                # mean of spectral entropy track
                freq_entropy_mean = np.nanmean(entropy_agg)
                # mean of spectral entropy track
                freq_entropy_std = np.nanstd(entropy_agg)
                # Upsweep mean/fraction
                upsweep_mean, upsweep_fraction = SpectrogramFeatures.upsweep_index(median_f)
            else:
                freq_peak = np.nan
                freq_median_mean = np.nan
                freq_median_std = np.nan
                freq_entropy_mean = np.nan
                freq_entropy_std = np.nan
                upsweep_mean = np.nan
                upsweep_fraction = np.nan     
        elif spectro.shape[1] == 1: # only 1 bin of bandwidth
            freq_peak = minigram.axis_frequencies[0]
            freq_median_mean = minigram.axis_frequencies[0]
            freq_median_std = 0
            freq_entropy_mean = np.nan
            freq_entropy_std = np.nan
            upsweep_mean = 0
            upsweep_fraction = 0
        else:
            freq_peak = np.nan
            freq_median_mean = np.nan
            freq_median_std = np.nan
            freq_entropy_mean = np.nan
            freq_entropy_std = np.nan
            upsweep_mean = np.nan
            upsweep_fraction = np.nan
        # signal to noise ratio
        snr = SpectrogramFeatures.snr(spectro)
        # FM features
        # med_freq_offset = np.dot((median_f -  np.mean(median_f)),root4_magnitude)
        # gather all feature into DataFrame
        features = pd.DataFrame({
            'freq_peak': [freq_peak],
            'freq_median_mean': [freq_median_mean],
            'freq_median_std': [freq_median_std],
            'freq_entropy_mean': [freq_entropy_mean],
            'freq_entropy_std': [freq_entropy_std],
            'freq_upsweep_mean': [upsweep_mean],
            'freq_upsweep_fraction': [upsweep_fraction],
            'snr': [snr],
            })
        frequency_points = pd.DataFrame({
            'axis_times': [minigram.axis_times+adjusted_bounds[0]],
            'freq_median': [median_f],
            'freq_peak': [peak_f],
            })
        return features, frequency_points

    @staticmethod
    def get_envelops(minigram, normalize=False):
        """Extract time and frequency envelop from spectrogram."""
        envelop_freq = np.sum(minigram.spectrogram, axis=1)
        envelop_time = np.sum(minigram.spectrogram, axis=0)
        if normalize:
            envelop_freq = envelop_freq/sum(envelop_freq)
            envelop_time = envelop_time/sum(envelop_time)
        return envelop_time, envelop_freq

    @staticmethod
    def _plot_envelop_features(axis_orig, envelop_orig, axis_interp, envelop_interp, features, title):
        """Plot envelope along with measurments table."""
        # plot - for debuging
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        ax[0].plot(axis_interp, envelop_interp, '.r')
        ax[0].plot(axis_orig, envelop_orig, '.g')
        ax[0].legend(['Interpolated', 'Original'])
        ax[0].grid()
        table = ax[1].table(cellText=features.values.T,
                            rowLabels=features.columns,
                            loc='center',
                            colWidths=[0.8,0.4])
        table.set_fontsize(20)
        ax[1].axis('off')
        fig.suptitle(title)
        fig.patch.set_visible(False)

    @staticmethod
    def _plot_spectrogram_features(minigram, features, adjusted_bounds, frequency_points, title=''):
        """"Plot spectrogram along with measurments table."""
        # plot - for debuging
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        ax[0].pcolormesh(minigram.axis_times,
                         minigram.axis_frequencies,
                         minigram.spectrogram,
                         cmap='jet',
                         vmin=np.percentile(minigram.spectrogram, 50),
                         vmax=np.percentile(minigram.spectrogram, 99.9)
                         )
        # ax[0].grid()
        ax[0].add_patch(Rectangle((adjusted_bounds[0], adjusted_bounds[2]),
                                  adjusted_bounds[1]-adjusted_bounds[0],
                                  adjusted_bounds[3]-adjusted_bounds[2],
                                  linewidth=2,
                                  edgecolor='white',
                                  facecolor='white',
                                  fill=False,
                                  alpha=0.8,
                                  )
                        )
        ax[0].plot(frequency_points['axis_times'].values[0],
                   frequency_points['freq_median'].values[0],
                   'xr')
        ax[0].plot(frequency_points['axis_times'].values[0],
                   frequency_points['freq_peak'].values[0],
                   'xk')
        ax[0].legend(['Median frequency', 'Peak frequency'])
        table = ax[1].table(cellText=features.values.T,
                            rowLabels=features.columns,
                            loc='center',
                            colWidths=[0.8, 0.4]
                            )
        table.set_fontsize(20)
        ax[1].axis('off')
        fig.suptitle(title)
        fig.patch.set_visible(False)

    @staticmethod
    @njit
    def length(array, resolution):
        """Duration/bandwidth of a time/frequency envelop."""
        return len(array)*resolution

    @staticmethod
    def peak(array, axis):
        """Return peak value, poistion and relative position of a 
        time/frequency envelop."""
        peak_value = np.amax(array)
        idxmax = np.where(array == peak_value)[0][0]
        peak_position_unit = axis[idxmax]
        peak_position_relative = (idxmax/len(array))*100
        return peak_value, peak_position_unit, peak_position_relative

    @staticmethod
    def percentiles_position_old(array, percentiles, axis=None):
        """Provide position of a percentile in an array of values.

        Parameters
        ----------
        array : numpy array
            array with values.
        percentiles : list
            List with the percentiles to "find" (e.g. [50, 75]).
        axis : numpy array, optional
            array with axis for the array values. The default is None.

        Returns
        -------
        pct_position : dict
            Dictionary with position of the percentile. Dict keys are the 
            values of the percentiles requested (e.g. pct_position['50']).
        """
        if axis is None:
            axis = range(0, len(array), 1)
        pct_position = dict()
        values_sum = np.sum(array)
        values_cumsum = np.cumsum(array)
        for pct in percentiles:
            pct_val = pct/100*values_sum
            pct_val_idx = np.where(values_cumsum > pct_val)[0][0]
            pct_val_unit = axis[pct_val_idx]
            pct_position[str(pct)] = pct_val_unit
        return pct_position

    @staticmethod
    def percentiles_position(array, percentiles, axis=None):
        """Provide position of a percentile in an array of values.

        Parameters
        ----------
        array : numpy array
            array with values.
        percentiles : list
            List with the percentiles to "find" (e.g. [50, 75]).
        axis : numpy array, optional
            array with axis for the array values. The default is None.

        Returns
        -------
        pct_position : dict
            Dictionary with position of the percentile. Dict keys are the 
            values of the percentiles requested (e.g. pct_position['50']).
        """
        if axis is None:
            axis = range(0, len(array), 1)
        pct_position = dict()
        values_sum = np.sum(array)
        values_cumsum = np.cumsum(array)
        for pct in percentiles:
            pct_val = pct/100*values_sum
            pct_val_idx = np.where(values_cumsum > pct_val)[0][0]
            pct_val_unit = axis[pct_val_idx]
            pct_position[str(pct)] = pct_val_unit
        return pct_position

    @staticmethod
    @njit
    def asymmetry(pct25, pct50, pct75):
        """Calculate envelope assymetry."""
        return (pct25+pct75-(2*pct50))/(pct25+pct75)  # feat

    @staticmethod
    def concentration(array, axis):
        """Calculate envelope concentration."""
        sort_idx = np.argsort(-array)
        values_sorted = array[sort_idx]
        axis_sorted = axis[sort_idx]
        idx_pct50 = SpectrogramFeatures.percentiles_position(values_sorted, [50])['50']
        idx_pct50 = np.max([idx_pct50, 1])  # in case idx50 == 0
        unit_min = np.min(axis_sorted[0:idx_pct50])
        unit_max = np.max(axis_sorted[0:idx_pct50])
        concentration_unit = unit_max - unit_min  # feat
        return concentration_unit

    @staticmethod
    @njit
    def flatness(array):
        """Calculate envelope flatness."""
        # normalize and add 1 to account for zero values
        array = array/max(array)+1
        # arithmetic mean
        arithmetic_mean = np.mean(array)
        # geometric mean
        n = len(array)
        multiply = np.prod(array)
        geometric_mean = (multiply)**(1/n)
        #geometric_mean = gmean(array)
        return geometric_mean/arithmetic_mean

    @staticmethod
    def roughness(array):
        """Calculate envelope roughness."""
        array_norm = array/max(array)
        deriv2 = ecosound.core.tools.derivative_1d(array_norm, order=2)
        return np.sum(np.power(deriv2, 2))

    @staticmethod
    def centroid(array, axis):
        """Calculate envelope centroid."""
        return np.dot(axis, array) / np.sum(array)  # feat

    @staticmethod
    def upsweep_index(array):
        """Calculate envelope upsweep mean and upsweep fraction."""
        freq_median_delta = np.subtract(array[1:], array[0:-1])
        upsweep_mean = np.mean(freq_median_delta)
        upsweep_fraction = len(np.where(freq_median_delta >= 0)[0]) / len(freq_median_delta)
        return upsweep_mean, upsweep_fraction

    @staticmethod
    def snr(array):
        """Calculate signal to noise ratio."""
        sig = np.amax(array)
        noise = np.percentile(array, 25)
        if noise > 0:
            snr = 10*np.log10(sig/noise)
        else:
            #snr = 10*np.log10(sig)  #feat
            snr = np.nan  #feat
        return snr
