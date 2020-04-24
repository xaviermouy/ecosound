# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:02:51 2020

@author: xavier.mouy
"""


class BaseClass(object):
    """
    Base class for all measurers.

    All measurers need to inheritate from this BaseClass in order to be built
    by the MeasurerFactory.
    """

    def __init__(self, measurer_name):
        self.measurer_name = measurer_name

    @classmethod
    def is_measurer_for(cls, measurer_name):
        """
        Check measurer name.

        Compare the requested measurer_name with each measurer class
        available.

        Parameters
        ----------
        measurer_name : str
        Name of the detector class.

        Returns
        -------
        bool
        """
        return measurer_name == cls.__name__


def MeasurerFactory(measurer_name, *args, **kwargs):
    """
    Measurer Factory.

    Loads the measurer class defined by measurer_name. Each measurer class must
    be added to the __init__.py file.

    For example :
    from .measurer1 import Measurer1

    where Measurer1 is the name of the measurer class to load from the .py
    file measurer1

    Parameters
    ----------
    measurer_name : str
        Name of the measurer class.
    *args : any
        Input arguments for the measurer selected.
    **kwargs : any
        Keyword arguments for the measurer selected.

    Raises
    ------
    ValueError
        If measurer_name doesn't correspond to any of the measurers available.

    Returns
    -------
    Measurer object

    """
    meas_list = []
    for cls in BaseClass.__subclasses__():
        meas_list.append(cls.__name__)
        if cls.is_measurer_for(measurer_name):
            return cls(measurer_name, *args, **kwargs)
    raise ValueError("Invalid measurer name. Measurers available: "
                     + str(meas_list))
