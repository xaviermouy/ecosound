# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""

from .detector_builder import BaseClass
from ecosound.core.annotation import Annotation
from datetime import datetime
import uuid

class Detector1(BaseClass):

    detector_parameters = ('kernel_duration','kernel_bandwidth', 'threshold','duration_min','bandwidth_min')
    
    def __init__(self, *args, **kwargs):
        # Initialize all detector parameters to None
        self.__dict__.update(dict(zip(self.detector_parameters,
                                      [None]*len(self.detector_parameters))))
        # Unpack kwargs as detector parameters if provided on instantiation
        self.__dict__.update(**kwargs)
    
    @property
    def name(self):
        """Return name of the detector."""
        detector_name = 'DetectorTemplate'
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
    
    def run(self, spectro, debug=False):
        
        self._prerun_check(self, spectrogram)
        
        detec = Annotation()
        detec.data['from_detector'] = True
        detec.data['software_name'] = self.name
        detec.data['software_version'] = self.version
        detec.data['entry_date'] = datetime.now()
        detec.data['uuid'] = detec.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        return detec