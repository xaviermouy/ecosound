# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
import ecosound.core.tools
import pandas as pd
import xarray as xr
import os


class Measurement(Annotation):
    def __init__(
        self,
        measurer_name=None,
        measurer_version=None,
        measurements_name=None,
        measurements_parameters=None,
    ):
        """Measurement object.

        Object to "store" sound measurements. Inheritate all methods from the
        ecosound Annotaion class.

        Parameters
        ----------
        measurer_name : str, optional
            Name of the measurer that was used to calculate the measurements.
            The default is None.
        measurer_version : str, optional
            Version of the measurer that was used to calculate the measurements.
            The default is None.
        measurements_name : list of str, optional
            List with the name of each measurement. The default is None.
        measurements_parameters: dict, optional
            dict with lists of measurement parameters
        Returns
        -------
        None. ecosound Measurement object with a .data and .metadata dataframes

        """
        super(Measurement, self).__init__()
        metadata = {
            "measurer_name": measurer_name,
            "measurer_version": measurer_version,
            "measurements_name": [measurements_name],
            "measurements_parameters": [measurements_parameters],
        }
        self._metadata = pd.DataFrame(metadata)
        self.data = pd.concat(
            [self.data, pd.DataFrame(columns=metadata["measurements_name"][0])]
        )

    @property
    def metadata(self):
        """
        Return the metadata attribute.

        Includes adictionary with the measurer_name, measurer_version, and
        measurements_name.
        """
        return self._metadata

    def to_netcdf(self, file):
        """
        Write measurement data to a netcdf file.

        Write measurementss as .nc file. This format works well with xarray
        and Dask.

        Parameters
        ----------
        file : str
            Path of the netcdf file (.nc) to be written.

        Returns
        -------
        None.
        """

        if file.endswith(".nc") == False:
            file = file + ".nc"
        self._enforce_dtypes()
        meas = self.data
        meas.set_index("time_min_date", drop=False, inplace=True)
        meas.index.name = "date"
        dxr1 = meas.to_xarray()
        dxr1.attrs["datatype"] = "Measurement"
        dxr1.attrs[
            "measurements_name"
        ] = self.metadata.measurements_name.values[0]
        dxr1.attrs["measurer_name"] = self.metadata.measurer_name.values[0]
        dxr1.attrs["measurer_version"] = self.metadata.measurer_version.values[
            0
        ]
        try:
            dxr1.attrs["measurements_parameters"] = str(
                self.metadata.measurements_parameters[0]
            )
        except:
            pass
        dxr1.to_netcdf(file, engine="netcdf4", format="NETCDF4")

    def from_netcdf(self, file, verbose=False):
        """
        Import measurement data from a netcdf file.

        Load measurements from a .nc file. This format works well with xarray
        and Dask. Only "Measurement" netcdf files created with to_netcdf can
        be imported. all other netcdf files will return an error.

        Parameters
        ----------
        file : str
            Path of the nc file to import. Can be a str if importing a single
            file or entire folder. Needs to be a list if importing multiple
            files. If 'files' is a folder, all files in that folder ending with
            '.nc' will be imported.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        if type(file) is str:
            if os.path.isdir(file):
                file = ecosound.core.tools.list_files(
                    file,
                    ".nc",
                    recursive=False,
                    case_sensitive=True,
                )
                if verbose:
                    print(len(file), "files found.")
            else:
                file = [file]
        self.data, self._metadata = self._import_netcdf_files(file)
        self.check_integrity(verbose=verbose)

    def _import_netcdf_files(self, files):
        """Import one or several netcdf files to a Panda datafrane."""
        assert type(files) in (
            str,
            list,
        ), "Input must be of type str (single \
            file or directory) or list (multiple files)"
        # Import all files to a dataframe
        tmp = []
        for idx, file in enumerate(files):
            dxr = xr.open_dataset(file)
            if dxr.attrs["datatype"] == "Measurement":
                if idx == 0:
                    measurer_name = dxr.measurer_name
                    measurer_version = dxr.measurer_version
                    measurements_name = dxr.measurements_name
                    try:
                        measurements_parameters = eval(
                            dxr.measurements_parameters
                        )
                    except:
                        measurements_parameters = None
                ## check measurere name and version
                if (dxr.measurer_name == measurer_name) & (
                    dxr.measurer_version == measurer_version
                ):
                    tmp2 = dxr.to_dataframe()
                    tmp2.reset_index(inplace=True)
                else:
                    raise ValueError(
                        file
                        + "Not all files were not generated from the same measurer type and version."
                    )
            else:
                raise ValueError(file + "Not a Measurement file.")
            tmp.append(tmp2)
        data = pd.concat(tmp, ignore_index=True, sort=False)
        data.reset_index(inplace=True, drop=True)
        metadata = {
            "measurer_name": measurer_name,
            "measurer_version": measurer_version,
            "measurements_name": [measurements_name],
            "measurements_parameters": [measurements_parameters],
        }
        metadata = pd.DataFrame(metadata)
        return data, metadata

    def __add__(self, other):
        """Concatenate data from several Measurement objects."""
        assert (
            type(other) is ecosound.core.measurement.Measurement
        ), "Object type not\
            supported. Can only concatenate Measurement objects together."
        assert (
            other.metadata["measurer_name"].values[0]
            == self.metadata["measurer_name"].values[0]
        ), "Can't concatenate measurements made from different measurers."
        assert (
            other.metadata["measurer_version"].values[0]
            == self.metadata["measurer_version"].values[0]
        ), "Can't concatenate measurements made from different versions of measurers."
        self._enforce_dtypes()
        other._enforce_dtypes()
        self.data = pd.concat(
            [self.data, other.data], ignore_index=True, sort=False
        )
        return self
