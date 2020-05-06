# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
import pandas as pd


class Measurement(Annotation):

    def __init__(self, measurer_name, measurer_version, measurements_name):
        """ Measurement object.

        Inheritate all methods from the ecosound Annotaion class.

        Parameters
        ----------
        measurer_name : str
            Name of the measurer that was used to calculate the measurements.
        measurer_version : str
            Version of the measurer that was used to calculate the measurements
        measurements_name : list of str
            List with the name of each measurement.

        Returns
        -------
        None. ecosound Measurement object with a .data and .metadata dataframes

        """
        super(Measurement, self).__init__()
        metadata = {'measurer_name': measurer_name,
                    'measurer_version': measurer_version,
                    'measurements_name': [measurements_name],
                    }
        self._metadata = pd.DataFrame(metadata)
        self.data = pd.concat([self.data,pd.DataFrame(columns=metadata['measurements_name'][0])])

    @property
    def metadata(self):
        """Return the metadata attribute."""
        return self._metadata
