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

    def to_raven(
        self, outdir, outfile="Raven.Table.1.selections.txt", single_file=False
    ):
        """
        Write data to 1 or several Raven files.

        Write measurements as .txt files readable by the software Raven. Output
        files can be written in a single txt file or in several txt files (one
        per audio recording). In the latter case, output file names are
        automatically generated based on the audio file's name.

        Parameters
        ----------
        outdir : str
            Path of the output directory where the Raven files are written.
        outfile : str
            Name of the output file. Only used is single_file is True. The
            default is 'Raven.Table.1.selections.txt'.
        single_file : bool, optional
            If set to True, writes a single output file with all annotations.
            The default is False.

        Returns
        -------
        None.

        """
        cols = [
            "Selection",
            "View",
            "Channel",
            "Begin Time (s)",
            "End Time (s)",
            "Delta Time (s)",
            "Low Freq (Hz)",
            "High Freq (Hz)",
            "Begin Path",
            "File Offset (s)",
            "Begin File",
            "Class",
            "Sound type",
            "Software",
            "Confidence",
        ]
        
        # add measurements names
        cols = cols + self.metadata['measurements_name'][0]
        
        if len(self) > 0:
            if single_file:
                annots = [self.data]
            else:
                annots = [
                    pd.DataFrame(y)
                    for x, y in self.data.groupby(
                        "audio_file_name", as_index=False
                    )
                ]
            for annot in annots:
                annot.reset_index(inplace=True, drop=True)
                
                outdf = pd.DataFrame(columns =cols)
                # outdf = pd.DataFrame(
                #     {
                #         "Selection": 0,
                #         "View": 0,
                #         "Channel": 0,
                #         "Begin Time (s)": 0,
                #         "End Time (s)": 0,
                #         "Delta Time (s)": 0,
                #         "Low Freq (Hz)": 0,
                #         "High Freq (Hz)": 0,
                #         "Begin Path": 0,
                #         "File Offset (s)": 0,
                #         "Begin File": 0,
                #         "Class": 0,
                #         "Sound type": 0,
                #         "Software": 0,
                #         "Confidence": 0,
                #     },
                #     index=list(range(annot.shape[0])),
                # )
                
                
                outdf["Selection"] = range(1, annot.shape[0] + 1)
                outdf["View"] = "Spectrogram 1"
                outdf["Channel"] = annot["audio_channel"]
                outdf["Begin Time (s)"] = annot["time_min_offset"]
                outdf["End Time (s)"] = annot["time_max_offset"]
                outdf["Delta Time (s)"] = annot["duration"]
                outdf["Low Freq (Hz)"] = annot["frequency_min"]
                outdf["High Freq (Hz)"] = annot["frequency_max"]
                outdf["File Offset (s)"] = annot["time_min_offset"]
                outdf["Class"] = annot["label_class"]
                outdf["Sound type"] = annot["label_subclass"]
                outdf["Software"] = annot["software_name"]
                outdf["Confidence"] = annot["confidence"]
                outdf["Begin Path"] = [
                    os.path.join(x, y) + z
                    for x, y, z in zip(
                        annot["audio_file_dir"],
                        annot["audio_file_name"],
                        annot["audio_file_extension"],
                    )
                ]
                outdf["Begin File"] = [
                    x + y
                    for x, y in zip(
                        annot["audio_file_name"], annot["audio_file_extension"]
                    )
                ]
                outdf = outdf.fillna(0)
                # add neasurements
                outdf[self.metadata['measurements_name'][0]]=annot[self.metadata['measurements_name'][0]]                
                if single_file:
                    outfilename = os.path.join(outdir, outfile)
                else:
                    outfilename = os.path.join(
                        outdir,
                        str(annot["audio_file_name"].iloc[0])
                        + str(annot["audio_file_extension"].iloc[0])
                        + ".chan"
                        + str(annot["audio_channel"].iloc[0])
                        + ".Table.1.selections.txt",
                    )                                                
                outdf.to_csv(
                    outfilename,
                    sep="\t",
                    encoding="utf-8",
                    header=True,
                    columns=cols,
                    index=False,
                )
        else:
            # No annotation => write file with header only
            outfilename = os.path.join(outdir, outfile)
            header = "\t".join(cols)
            f = open(outfilename, "w")
            f.write(header)
            f.close()
    

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
