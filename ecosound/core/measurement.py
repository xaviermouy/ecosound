# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
import pandas as pd
import xarray as xr

class Measurement(Annotation):

    def __init__(self, measurer_name=None, measurer_version=None, measurements_name=None):
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

    def to_netcdf(self, file):
        if file.endswith('.nc') == False:
            file = file + '.nc'
        meas = self.data
        meas.set_index('time_min_date', drop=False, inplace=True)
        meas.index.name = 'date'
        dxr1 = meas.to_xarray()
        dxr1.attrs['datatype'] = 'Measurement'
        dxr1.attrs['measurements_name'] = self.metadata.measurements_name.values[0]
        dxr1.attrs['measurer_name'] = self.metadata.measurer_name.values[0]
        dxr1.attrs['measurer_version'] = self.metadata.measurer_version.values[0]
        dxr1.to_netcdf(file, engine='netcdf4', format='NETCDF4')
     
    def from_netcdf(self, file):
        dxr = xr.open_dataset(file)
        if dxr.attrs['datatype'] == 'Measurement': 
            self.data = dxr.to_dataframe()
            self.data.reset_index(inplace=True)
            self._metadata['measurer_name'][0] = dxr.attrs['measurer_name']
            self._metadata['measurer_version'][0] = dxr.attrs['measurer_version']
            self._metadata['measurements_name'][0] = dxr.attrs['measurements_name']
        else:
            print('Not a Measurement file.')
