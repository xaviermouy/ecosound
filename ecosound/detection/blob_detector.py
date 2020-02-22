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
import cv2
import uuid


class BlobDetector(BaseClass):
    detector_parameters = ('kernel_duration','kernel_bandwidth', 'threshold','duration_min','bandwidth_min')

    def __init__(self, *args, **kwargs):
        # Initialize all detector parameters to None
        self.__dict__.update(dict(zip(self.detector_parameters,
                                      [None]*len(self.detector_parameters))))
        # Unpack kwargs as detector parameters if provided on instantiation
        self.__dict__.update(**kwargs)

    @property
    def name(self):
        """Return name of teh detector."""
        detector_name = 'BlobDetector'
        return detector_name

    @property
    def version(self):
        """Return version of the detector."""
        version = '0.1'
        return version

    def _prerun_check(self, spectrogram):
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
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(
        figsize=(16,4),
        sharex=True
        )
        im = ax.pcolormesh(Matrix, cmap = 'jet')
        ax.set_title(title)

    def run(self, spectro, debug=False):
        # Pre-run verifications
        self._prerun_check(spectro)
        # Convert units to spectrogram bins
        kernel_duration = max(round(self.kernel_duration/spectro.time_resolution), 1)
        kernel_bandwidth = max(round(self.kernel_bandwidth/spectro.frequency_resolution), 1)
        duration_min = max(round(self.duration_min/spectro.time_resolution), 1)
        bandwidth_min = max(round(self.bandwidth_min/spectro.frequency_resolution), 1)
        # Apply filter
        Svar = ndimage.generic_filter(spectro.spectrogram, calcVariance2D, (kernel_bandwidth, kernel_duration), mode='mirror')
        # binarization
        Svar[Svar<self.threshold] = 0
        Svar[Svar>0] = 1
        if debug:
             self._plot_matrix(Svar, 'Binarized spectrogram matrix')
        # Define contours
        Svar_gray = cv2.normalize(src=Svar, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        (im2, cnts, hierarchy) = cv2.findContours(Svar_gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        isdetec=False
        t1 = []
        t2 = []
        fmin = []
        fmax = []
        for c in cnts:
            # Compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is too small, ignore it
            if w < duration_min or  h < bandwidth_min:
                continue
            else:
                isdetec=True
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

        return detec

def calcVariance2D(buffer):
    return np.var(buffer)
    #return np.median(buffer.ravel())
    #return np.mean(buffer.ravel())