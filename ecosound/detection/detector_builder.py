# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:02:51 2020

@author: xavier.mouy
"""


class BaseClass(object):
    """
    Base class for all detectors.

    All detectors need to inheritate from this BaseClass in order to be built
    by the DetectorFactory.
    """

    def __init__(self, detector_name):
        self.detector_name = detector_name
        print('yeah')

    @classmethod
    def is_detector_for(cls, detector_name):
        """
        Check detector name.

        Compare the requested detector_name with each detector classe
        available.

        Parameters
        ----------
        detector_name : str
        Name of the detector class.

        Returns
        -------
        bool
        """
        return detector_name == cls.__name__


def DetectorFactory(detector_name, *args, **kwargs):
    """
    Detector Factory.

    Loads the detector class defined by detector_name. Each detector class must
    be added to the __init__.py file.

    For example :
    from .detector1 import Detector1

    where Detector1 is the name of the detector class to load from teh .py
    file detection1

    Parameters
    ----------
    detector_name : str
        Name of the detector class.
    *args : any
        Input arguments for the detector selected.
    **kwargs : any
        Keyword arguments for the detector selected.

    Raises
    ------
    ValueError
        If detector_name doesn't correspond to any of the detectors available.

    Returns
    -------
    Detector object

    """
    detec_list = []
    for cls in BaseClass.__subclasses__():
        detec_list.append(cls.__name__)
        if cls.is_detector_for(detector_name):
            return cls(detector_name, *args, **kwargs)
    raise ValueError("Invalid detector name. Detectors available: "
                     + str(detec_list))
