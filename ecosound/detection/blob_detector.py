# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""
from .detector_builder import BaseClass
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation

from scipy import signal, ndimage
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import uuid
from numba import njit
import dask
import dask_image.ndfilters
import dask.array

class BlobDetector(BaseClass):
    """Blob detector.

    A detector to find transient events in a spectrogram object. The local
    variance is calculated in the spectrogram for each time-frequency bin using
    a local area (e.g. kernel) defined by 'kernel_duration' and
    'kernel_bandwidth'. Bins or the spectrogram with a local variance less than
    'threshold' are set to zero, while all the bins greater than 'threshold'
    are set to one. The Moors Neighborhood  algorithm is then used to define
    the time and frequency boudaries of the adjacent spectrogram bins taht
    equal one. All detections with a duration less than 'duration_min' and a
    bandwidth less than 'bandwidth_min' are discarded.

    The BlobDetector detector must be instantiated using the DetectorFactory
    with the positional argument 'BlobDetector':

    from ecosound.detection.detector_builder import DetectorFactory
    detector = DetectorFactory('BlobDetector', args)

    Attributes
    ----------
    name : str
        Name of the detector
    version : str
        Version of the detector
    kernel_duration : float
        Duration of the kernel, in seconds.
    kernel_bandwidth : float
        Bandwidth of teh kernel, in Hz.
    threshold : float
        Variance threshold for teh binarization.
    duration_min : float
        Minimum duration of detection accepted,in seconds.
    bandwidth_min : float
        Minimum bandwidth of detection accepted,in seconds.

    Methods
    -------
    run(spectro, debug=False)
        Run the detector on a spectrogram object.
    """

    detector_parameters = ('kernel_duration',
                           'kernel_bandwidth',
                           'threshold',
                           'duration_min',
                           'bandwidth_min')

    def __init__(self, *args, **kwargs):
        """
        Initialize the detector.

        Parameters
        ----------
        *args : str
            Do not use. Only used by the DetectorFactory.
        kernel_duration : float
            Duration of the kernel, in seconds.
        kernel_bandwidth : float
            Bandwidth of teh kernel, in Hz.
        threshold : float
            Variance threshold for the binarization.
        duration_min : float
            Minimum duration of detection accepted,in seconds.
        bandwidth_min : float
            Minimum bandwidth of detection accepted,in seconds.

        Returns
        -------
        None. Detector object.

        """
        # Initialize all detector parameters to None
        self.__dict__.update(dict(zip(self.detector_parameters,
                                      [None]*len(self.detector_parameters))))
        # Unpack kwargs as detector parameters if provided on instantiation
        self.__dict__.update(**kwargs)

    @property
    def name(self):
        """Return name of the detector."""
        detector_name = 'BlobDetector'
        return detector_name

    @property
    def version(self):
        """Return version of the detector."""
        version = '0.1'
        return version

    def _prerun_check(self, spectrogram):
        """Run several verifications before the run."""
        # check that all required arguments are defined
        if True in [self.__dict__.get(keys) is None for keys in self.detector_parameters]:
            raise ValueError('Not all detector parameters have been defined.'
                             + ' Required parameters: '
                             + str(self.detector_parameters))
        # check that spectrogram is a spectrogram class
        if not isinstance(spectrogram, Spectrogram):
            raise ValueError('Input must be an ecosound Spectrogram object'
                             + '(ecosound.core.spectrogram).')

    def _plot_matrix(self, Matrix, title):
        """Plot spectyrogram matrix when in debug mode."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
            figsize=(16, 4),
            sharex=True)
        ax.pcolormesh(Matrix, cmap='jet')
        ax.set_title(title)

    def run(self, spectro, start_time=None, use_dask=False, dask_chunks=(1000,1000), debug=False):
        """Run detector.

        Runs the detector on the spectrogram object.

        Parameters
        ----------
        spectro : Spectrogram
            Spectrogram object to detect from.
        debug : bool, optional
            Displays binarization results for debugging purpused.The default
            is False.
        start_time : datetime.datetime, optional
            Start time/date of the signal being processed. If defined, the
            fields 'time_min_date' and 'time_max_date' of the detection
            annotation object are populated. The default is None.
        use_dask, bool, optional
            If True, runs the detector in parallel using Dask. The default is
            False.
        dask_chunks, tuple -> (int, int), optional
            Tuple of two int defining the size of the spectrogram chunks to use
            for the parallel processing: dask_chunks=(number of frequency bins,
             number of time bbins). Only used in use_dask is True. The default
            is (1000, 1000).

        Returns
        -------
        detec : Annotation
            Annotation object with the detection results.

        """
        # Pre-run verifications
        self._prerun_check(spectro)
        # Convert units to spectrogram bins
        kernel_duration = max(
            round(self.kernel_duration/spectro.time_resolution), 1)
        kernel_bandwidth = max(
            round(self.kernel_bandwidth/spectro.frequency_resolution), 1)
        duration_min = max(
            round(self.duration_min/spectro.time_resolution), 1)
        bandwidth_min = max(
            round(self.bandwidth_min/spectro.frequency_resolution), 1)
        # # Apply filter
        if use_dask:
            dask_spectro = dask.array.from_array(spectro.spectrogram, chunks=dask_chunks)
            Svar = dask_image.ndfilters.generic_filter(dask_spectro,
                                                       calcVariance2D,
                                                       size=(kernel_bandwidth, kernel_duration),
                                                       mode='mirror')
            Svar = Svar.compute()
        else:
            Svar = ndimage.generic_filter(spectro.spectrogram,
                                          calcVariance2D,
                                          (kernel_bandwidth, kernel_duration),
                                          mode='mirror')

        # binarization
        Svar[Svar < self.threshold] = 0
        Svar[Svar > 0] = 1
        if debug:
            self._plot_matrix(Svar, 'Binarized spectrogram matrix')

        #new
        #Svar = cv2.cvtColor(cv2.UMat(Svar), cv2.COLOR_RGB2GRAY)
        # Define contours
        Svar_gray = cv2.normalize(src=Svar,
                                  dst=None,
                                  alpha=0,
                                  beta=255,
                                  norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
        (cnts, hierarchy) = cv2.findContours(Svar_gray.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        isdetec = False
        t1 = []
        t2 = []
        fmin = []
        fmax = []
        for c in cnts:
            # Compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is too small, ignore it
            if w < duration_min or h < bandwidth_min:
                continue
            else:
                isdetec = True
                # box coord
                t1.append(x)
                t2.append(x+w)
                fmin.append(y)
                fmax.append(y+h)
        # Insert results in an Annotation object
        detec = Annotation()
        detec.data['time_min_offset'] = [t*spectro.time_resolution for t in t1]
        detec.data['time_max_offset'] = [t*spectro.time_resolution for t in t2]
        detec.data['frequency_min'] = [f*spectro.frequency_resolution for f in fmin]
        detec.data['frequency_max'] = [f*spectro.frequency_resolution for f in fmax]
        detec.data['duration'] = detec.data['time_max_offset'] - detec.data['time_min_offset']
        detec.data['from_detector'] = True
        detec.data['software_name'] = self.name
        detec.data['software_version'] = self.version
        detec.data['entry_date'] = datetime.now()
        detec.data['uuid'] = detec.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        if start_time:
            detec.data['time_min_date']= pd.to_datetime(start_time + pd.to_timedelta(detec.data['time_min_offset'], unit='s'))
            detec.data['time_max_date']= pd.to_datetime(start_time + pd.to_timedelta(detec.data['time_max_offset'], unit='s'))
        return detec

@njit()
def calcVariance2D(buffer):
    """Calculate the 2D variance."""
    return np.var(buffer)
    # return np.median(buffer.ravel())
    # return np.mean(buffer.ravel())
