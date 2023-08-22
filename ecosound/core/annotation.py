# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""

import pandas as pd
import xarray as xr
import numpy as np
import os
import uuid
import warnings
import ecosound.core.tools
import ecosound.core.decorators
import sqlite3
from ecosound.core.metadata import DeploymentInfo
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.audiotools import Sound
import copy
import csv
import datetime
import re
import warnings
from tqdm import tqdm


class Annotation:
    """
    A class used for manipulating annotation data.

    The Annotation object stores from both manual analysis annotations
    collected with software like PAMlab and Raven, and outputs from
    automated detectors and classifiers.

    Attributes
    ----------
    data : pandas DataFrame
        Annotation DataFranme.

    Methods
    -------
    check_integrity(verbose=False, time_duplicates_only=False)
        Check integrity of Annotation object.
    from_raven(files, class_header='Sound type',subclass_header=None,
               verbose=False)
        Import annotation data from 1 or several Raven files.
    to_raven(outdir, single_file=False)
        Write annotation data to one or several Raven files.
    from_pamlab(files, verbose=False)
        Import annotation data from 1 or several PAMlab files.
    to_pamlab(outdir, single_file=False)
        Write annotation data to one or several Raven files.
    from_parquet(file)
        Import annotation data from a Parquet file.
    to_parquet(file)
        Write annotation data to a Parquet file.
    from_netcdf(file)
        Import annotation data from a netCDF4 file.
    to_netcdf(file)
        Write annotation data to a netCDF4 file.
    insert_values(**kwargs)
        Manually insert values for given Annotation fields.
    insert_metadata(deployment_info_file)
        Insert metadata information to the annotation from a
        deployment_info_file.
    filter_overlap_with(annot, freq_ovp=True, dur_factor_max=None,
                        dur_factor_min=None,ovlp_ratio_min=None,
                        remove_duplicates=False,inherit_metadata=False,
                        filter_deploymentID=True, inplace=False)
        Filter annotations overalaping with another set of annotations.
    update_audio_dir(new_data_dir)
        Update path of audio files.
    get_labels_class()
        Return all unique class labels.
    get_labels_subclass()
        Return all unique subclass labels.
    get_fields()
        Return list with all annotations fields.
    summary(rows='deployment_ID',columns='label_class')
        Produce a summary pivot table with the number of annotations for two
        given annotation fields.
    __add__()
        Concatenate data from annotation objects uisng the + sign.
    __len__()
        Return number of annotations.
    """

    def __init__(self):
        """
        Initialize Annotation object.

        Sets all the annotation fields.:
            -'uuid': UUID,
                Unique identifier code
            -'from_detector': bool,
                True if data comes from an automatic process.
            -'software_name': str,
                Software name. Can be Raven or PAMlab for manual analysis.
            -'software_version': str,
                Version of the software used to create the annotations.
            -'operator_name': str,
                Name of the person responsible for the creation of the
                annotations.
            -'UTC_offset': float,
                Offset hours to UTC.
            -'entry_date': datetime,
                Date when the annotation was created.
            -'audio_channel': int,
                Channel number.
            -'audio_file_name': str,
                Name of the audio file.
            -'audio_file_dir': str,
                Directory where the audio file is.
            -'audio_file_extension': str,
                Extension of teh audio file.
            -'audio_file_start_date': datetime,
                Date of the audio file start time.
            -'audio_sampling_frequency': int,
                Sampling frequecy of the audio data.
            -'audio_bit_depth': int,
                Bit depth of the audio data.
            -'mooring_platform_name': str,
                Name of the moorig platform (e.g. 'glider','Base plate').
            -'recorder_type': str,
                Name of the recorder type (e.g., 'AMAR'), 'SoundTrap'.
            -'recorder_SN': str,
                Serial number of the recorder.
            -'hydrophone_model': str,
                Model of the hydrophone.
            -'hydrophone_SN': str,
                Serial number of the hydrophone.
            -'hydrophone_depth': float,
                Depth of the hydrophone in meters.
            -'location_name': str,
                Name of the deploymnet location.
            -'location_lat': float,
                latitude of the deployment location in decimal degrees.
            -'location_lon': float,
                longitude of the deployment location in decimal degrees.
            -'location_water_depth': float,
                Water depth at the deployment location in meters.
            -'deployment_ID': str,
                Unique ID of the deployment.
            -'frequency_min': float,
                Minimum frequency of the annotaion in Hz.
            -'frequency_max': float,
                Maximum frequency of the annotaion in Hz.
            -'time_min_offset': float,
                Start time of the annotaion, in seconds relative to the
                begining of the audio file.
            -'time_max_offset': float,
                Stop time of the annotaion, in seconds relative to the
                begining of the audio file.
            -'time_min_date': datetime,
                Date of the annotation start time.
            -'time_max_date': datetime,
                Date of the annotation stop time.
            -'duration': float,
                Duration of the annotation in seconds.
            -'label_class': str,
                label of the annotation class (e.g. 'fish').
            -'label_subclass': str,
                label of the annotation subclass (e.g. 'grunt')
            'confidence': float,
                Confidence of the classification.

        Returns
        -------
        Annotation object.

        """
        self.data = pd.DataFrame(
            {
                "uuid": [],
                "from_detector": [],  # True, False
                "software_name": [],
                "software_version": [],
                "operator_name": [],
                "UTC_offset": [],
                "entry_date": [],
                "audio_channel": [],
                "audio_file_name": [],
                "audio_file_dir": [],
                "audio_file_extension": [],
                "audio_file_start_date": [],
                "audio_sampling_frequency": [],
                "audio_bit_depth": [],
                "mooring_platform_name": [],
                "recorder_type": [],
                "recorder_SN": [],
                "hydrophone_model": [],
                "hydrophone_SN": [],
                "hydrophone_depth": [],
                "location_name": [],
                "location_lat": [],
                "location_lon": [],
                "location_water_depth": [],
                "deployment_ID": [],
                "frequency_min": [],
                "frequency_max": [],
                "time_min_offset": [],
                "time_max_offset": [],
                "time_min_date": [],
                "time_max_date": [],
                "duration": [],
                "label_class": [],
                "label_subclass": [],
                "confidence": [],
            }
        )
        self._enforce_dtypes()

    def check_integrity(
        self, verbose=False, ignore_frequency_duplicates=False
    ):
        """
        Check integrity of Annotation object.

        Tasks performed:
            1- Check that start time < stop time
            2- Check that min frequency < max frequency
            3- Remove duplicate entries based on time and frequency, filename,
               labels and filenames

        Parameters
        ----------
        verbose : bool, optional
            Print summary of the duplicate entries deleted.
            The default is False.
        ignore_frequency_duplicates : bool, optional
            If set to True, doesn't consider frequency values when deleting
            duplicates. It is useful when data are imported from Raven.
            The default is False.

        Raises
        ------
        ValueError
            If annotations have a start time > stop time
            If annotations have a min frequency > max frequency

        Returns
        -------
        None.

        """
        # Drop all duplicates
        count_start = len(self.data)
        if ignore_frequency_duplicates:  # doesn't use frequency boundaries
            self.data = self.data.drop_duplicates(
                subset=[
                    "time_min_offset",
                    "time_max_offset",
                    "label_class",
                    "label_subclass",
                    "audio_file_name",
                ],
                keep="first",
            ).reset_index(drop=True)
        else:  # remove annot with exact same time AND frequency boundaries
            self.data = self.data.drop_duplicates(
                subset=[
                    "time_min_offset",
                    "time_max_offset",
                    "frequency_min",
                    "frequency_max",
                    "label_class",
                    "label_subclass",
                    "audio_file_name",
                ],
                keep="first",
            ).reset_index(drop=True)
        count_stop = len(self.data)
        if verbose:
            print("Duplicate entries removed:", str(count_start - count_stop))
        # Check that start and stop times are coherent (i.e. t2 > t1)
        time_check = self.data.index[
            self.data["time_max_offset"] < self.data["time_min_offset"]
        ].tolist()
        if len(time_check) > 0:
            raise ValueError(
                "Incoherent annotation times (time_min > time_max). \
                 Problematic annotations:"
                + str(time_check)
            )
        # Check that min and max frequencies are coherent (i.e. fmin < fmax)
        freq_check = self.data.index[
            self.data["frequency_max"] < self.data["frequency_min"]
        ].tolist()
        if len(freq_check) > 0:
            raise ValueError(
                "Incoherent annotation frequencies (frequency_min > \
                frequency_max). Problematic annotations:"
                + str(freq_check)
            )

        # check that there are not uuid duplicates
        idx = self.data.duplicated(subset=["uuid"])
        dup_idxs = idx[idx == True].index
        for dup_idx in dup_idxs:
            self.data.loc[dup_idx, "uuid"] = str(uuid.uuid4())
        if len(dup_idxs) > 0:
            if verbose:
                print(
                    len(dup_idxs),
                    " UUID duplicates were found and regenerated.",
                )
        if verbose:
            print("Integrity test succesfull")

    def from_raven(
        self,
        files,
        class_header="Sound type",
        subclass_header=None,
        is_file_sequence=False,
        recursive=False,
        verbose=False,
    ):
        """
        Import data from 1 or several Raven files.

        Load annotation tables from .txt files generated by the software Raven.

        Parameters
        ----------
        files : str, list
            Path of the txt file(s) to import. Can be a str if importing a single
            file. Needs to be a list if importing multiple files. If 'files' is
            a folder, all files in that folder ending with '.selections.txt'
            will be imported.
        class_header : str, optional
            Name of the header in the Raven file corresponding to the class
            name. The default is 'Sound type'.
        subclass_header : str, optional
            Name of the header in the Raven file corresponding to the subclass
            name. The default is None.
        recursive : bool, optional
            If set to True, goes rcursively through all subfolders. The default
            is False.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        if type(files) is not list:
            if os.path.isdir(files):
                files = ecosound.core.tools.list_files(
                    files,
                    ".selections.txt",
                    recursive=recursive,
                    case_sensitive=True,
                )
                if verbose:
                    print(len(files), "annotation files found.")
        data = Annotation._import_csv_files(files)
        columns = data.columns.to_list()

        if "Begin Path" in columns:
            files_timestamp = ecosound.core.tools.filename_to_datetime(
                data["Begin Path"].tolist()
            )
            self.data["audio_file_name"] = data["Begin Path"].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
            self.data["audio_file_dir"] = data["Begin Path"].apply(
                lambda x: os.path.dirname(x)
            )
            self.data["audio_file_extension"] = data["Begin Path"].apply(
                lambda x: os.path.splitext(x)[1]
            )
        elif "Begin File" in columns:
            if verbose:
                print(
                    "'Begin Path' not found using 'Begin File' to retriev timestamps"
                )
            files_timestamp = ecosound.core.tools.filename_to_datetime(
                data["Begin File"].tolist()
            )
            self.data["audio_file_name"] = data["Begin File"].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
            self.data["audio_file_dir"] = None
            self.data["audio_file_extension"] = data["Begin File"].apply(
                lambda x: os.path.splitext(x)[1]
            )
        else:
            files_timestamp = None
            if verbose:
                print("Name of annotated audio files could not be found")
        self.data["audio_file_start_date"] = files_timestamp
        self.data["audio_channel"] = data["Channel"]
        self.data["time_min_offset"] = data["Begin Time (s)"]
        self.data["time_max_offset"] = data["End Time (s)"]
        self.data["time_min_date"] = pd.to_datetime(
            self.data["audio_file_start_date"]
            + pd.to_timedelta(self.data["time_min_offset"], unit="s")
        )
        self.data["time_max_date"] = pd.to_datetime(
            self.data["audio_file_start_date"]
            + pd.to_timedelta(self.data["time_max_offset"], unit="s")
        )
        self.data["frequency_min"] = data["Low Freq (Hz)"]
        self.data["frequency_max"] = data["High Freq (Hz)"]
        if class_header is not None:
            self.data["label_class"] = data[class_header]
        if subclass_header is not None:
            self.data["label_subclass"] = data[subclass_header]
        self.data["from_detector"] = False
        self.data["software_name"] = "raven"
        self.data["uuid"] = self.data.apply(
            lambda _: str(uuid.uuid4()), axis=1
        )
        self.data["duration"] = (
            self.data["time_max_offset"] - self.data["time_min_offset"]
        )
        self.check_integrity(verbose=verbose, ignore_frequency_duplicates=True)
        if verbose:
            print(len(self), "annotations imported.")

    def to_raven(
        self, outdir, outfile="Raven.Table.1.selections.txt", single_file=False
    ):
        """
        Write data to 1 or several Raven files.

        Write annotations as .txt files readable by the software Raven. Output
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
                outdf = pd.DataFrame(
                    {
                        "Selection": 0,
                        "View": 0,
                        "Channel": 0,
                        "Begin Time (s)": 0,
                        "End Time (s)": 0,
                        "Delta Time (s)": 0,
                        "Low Freq (Hz)": 0,
                        "High Freq (Hz)": 0,
                        "Begin Path": 0,
                        "File Offset (s)": 0,
                        "Begin File": 0,
                        "Class": 0,
                        "Sound type": 0,
                        "Software": 0,
                        "Confidence": 0,
                    },
                    index=list(range(annot.shape[0])),
                )
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

    def to_sqlite(self, file):
        """
        Write data to a sqlite database file.

        Write annotations as .sqlite file.

        Parameters
        ----------
        file : str
            Path of the output file (.sqlite) to be written.

        Returns
        -------
        None.

        """
        if file.endswith(".sqlite") is False:
            file = file + ".sqlite"
        self._enforce_dtypes()

        conn = sqlite3.connect(file)
        self.data.to_sql(
            name="detections", con=conn, if_exists="append", index=False
        )
        conn.close()

    def from_sqlite(self, files, table_name="detections", verbose=False):
        """
        Import data from 1 or several sqlite files.

        Load annotation or detection tables from .sqlite files created by the
        method annotation.to_sqlite

        Parameters
        ----------
        files : str, list
            Path of the sqlite file(s) to import. Can be a str if importing a
            single file. Needs to be a list if importing multiple files. If
            'files' is a folder, all files in that folder ending with '.sqlite'
            will be imported.
        table_name : str, optional
            Name of the sql table name containing the annotations. The default
            is 'detections'.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        assert type(files) in (
            str,
            list,
        ), "Input must be of type str (single \
            file or directory) or list (multiple files)"
        files = Annotation._make_list_from_input(
            files, ".sqlite", verbose=verbose
        )
        """Import one or several sqlite files to a Panda datafrane."""
        tmp = []
        for idx, file in enumerate(files):
            conn = sqlite3.connect(file)
            tmp2 = pd.read_sql_query(
                "SELECT * FROM " + table_name,
                conn,
                parse_dates={
                    "entry_date":'ISO8601',
                    "audio_file_start_date":'ISO8601',
                    "time_min_date":'ISO8601',
                    "time_max_date":'ISO8601',
                },
            )
            conn.close()

            tmp.append(tmp2)
        data = pd.concat(tmp, ignore_index=True, sort=False)
        data.reset_index(inplace=True, drop=True)
        self.data = data
        self.check_integrity(verbose=verbose, ignore_frequency_duplicates=True)
        if verbose:
            print(len(self), "annotations imported.")

    def from_pamlab(self, files, verbose=False):
        """
        Import data from 1 or several PAMlab files.

        Load annotation data from .log files generated by the software PAMlab.

        Parameters
        ----------
        files : str, list
            Path of the txt file to import. Can be a str if importing a single
            file or entire folder. Needs to be a list if importing multiple
            files. If 'files' is a folder, all files in that folder ending with
            'annotations.log' will be imported.
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        if type(files) is str:
            if os.path.isdir(files):
                files = ecosound.core.tools.list_files(
                    files,
                    " annotations.log",
                    recursive=False,
                    case_sensitive=True,
                )
                if verbose:
                    print(len(files), "annotation files found.")
        data = Annotation._import_csv_files(files)
        files_timestamp = ecosound.core.tools.filename_to_datetime(
            data["Soundfile"].tolist()
        )
        self.data["audio_file_start_date"] = files_timestamp
        self.data["operator_name"] = data["Operator"]
        self.data["entry_date"] = pd.to_datetime(
            data["Annotation date and time (local)"],
            format="%Y-%m-%d %H:%M:%S.%f",
        )
        self.data["audio_channel"] = data["Channel"]
        self.data["audio_file_name"] = data["Soundfile"].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        self.data["audio_file_dir"] = data["Soundfile"].apply(
            lambda x: os.path.dirname(x)
        )
        self.data["audio_file_extension"] = data["Soundfile"].apply(
            lambda x: os.path.splitext(x)[1]
        )
        self.data["audio_sampling_frequency"] = data["Sampling freq (Hz)"]
        self.data["recorder_type"] = data["Recorder type"]
        self.data["recorder_SN"] = data["Recorder ID"]
        self.data["hydrophone_depth"] = data["Recorder depth"]
        self.data["location_name"] = data["Station"]
        self.data["location_lat"] = data["Latitude (deg)"]
        self.data["location_lon"] = data["Longitude (deg)"]
        self.data["time_min_offset"] = data["Left time (sec)"]
        self.data["time_max_offset"] = data["Right time (sec)"]
        self.data["time_min_date"] = pd.to_datetime(
            self.data["audio_file_start_date"]
            + pd.to_timedelta(self.data["time_min_offset"], unit="s")
        )
        self.data["time_max_date"] = pd.to_datetime(
            self.data["audio_file_start_date"]
            + pd.to_timedelta(self.data["time_max_offset"], unit="s")
        )
        self.data["frequency_min"] = data["Bottom freq (Hz)"]
        self.data["frequency_max"] = data["Top freq (Hz)"]
        self.data["label_class"] = data["Species"]
        self.data["label_subclass"] = data["Call type"]
        self.data["from_detector"] = False
        self.data["software_name"] = "pamlab"
        self.data["uuid"] = self.data.apply(
            lambda _: str(uuid.uuid4()), axis=1
        )
        self.data["duration"] = (
            self.data["time_max_offset"] - self.data["time_min_offset"]
        )
        self.check_integrity(verbose=verbose)
        if verbose:
            print(len(self), "annotations imported.")

    def to_pamlab(
        self, outdir, outfile="PAMlab annotations.log", single_file=False
    ):
        """
        Write data to 1 or several PAMlab files.

        Write annotations as .log files readable by the software PAMlab. Output
        files can be written in a single txt file or in several txt files (one
        per audio recording). In teh latter case, output file names are
        automatically generated based on the audio file's name and the name
        format required by PAMlab.

        Parameters
        ----------
        outdir : str
            Path of the output directory where the Raven files are written.
        outfile : str
            Name of teh output file. Only used is single_file is True. The
            default is 'PAMlab annotations.log'.
        single_file : bool, optional
            If set to True, writes a single output file with all annotations.
            The default is False.

        Returns
        -------
        None.

        """
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
            cols = [
                "fieldkey:",
                "Soundfile",
                "Channel",
                "Sampling freq (Hz)",
                "Latitude (deg)",
                "Longitude (deg)",
                "Recorder ID",
                "Recorder depth",
                "Start date and time (UTC)",
                "Annotation date and time (local)",
                "Recorder type",
                "Deployment",
                "Station",
                "Operator",
                "Left time (sec)",
                "Right time (sec)",
                "Top freq (Hz)",
                "Bottom freq (Hz)",
                "Species",
                "Call type",
                "rms SPL",
                "SEL",
                "",
                "",
            ]
            outdf = pd.DataFrame(
                {
                    "fieldkey:": 0,
                    "Soundfile": 0,
                    "Channel": 0,
                    "Sampling freq (Hz)": 0,
                    "Latitude (deg)": 0,
                    "Longitude (deg)": 0,
                    "Recorder ID": 0,
                    "Recorder depth": 0,
                    "Start date and time (UTC)": 0,
                    "Annotation date and time (local)": 0,
                    "Recorder type": 0,
                    "Deployment": 0,
                    "Station": 0,
                    "Operator": 0,
                    "Left time (sec)": 0,
                    "Right time (sec)": 0,
                    "Top freq (Hz)": 0,
                    "Bottom freq (Hz)": 0,
                    "Species": "",
                    "Call type": "",
                    "rms SPL": 0,
                    "SEL": 0,
                    "": "",
                    "": "",
                },
                index=list(range(annot.shape[0])),
            )
            outdf["fieldkey:"] = "an:"
            outdf["Species"] = annot["label_class"]
            outdf["Call type"] = annot["label_subclass"]
            outdf["Left time (sec)"] = annot["time_min_offset"]
            outdf["Right time (sec)"] = annot["time_max_offset"]
            outdf["Top freq (Hz)"] = annot["frequency_max"]
            outdf["Bottom freq (Hz)"] = annot["frequency_min"]
            outdf["rms SPL"] = annot["confidence"]
            outdf["Operator"] = annot["operator_name"]
            outdf["Channel"] = annot["audio_channel"]
            outdf["Annotation date and time (local)"] = annot["entry_date"]
            outdf["Sampling freq (Hz)"] = annot["audio_sampling_frequency"]
            outdf["Recorder type"] = annot["recorder_type"]
            outdf["Recorder ID"] = annot["recorder_SN"]
            outdf["Recorder depth"] = annot["hydrophone_depth"]
            outdf["Station"] = annot["location_name"]
            outdf["Latitude (deg)"] = annot["location_lat"]
            outdf["Longitude (deg)"] = annot["location_lon"]
            outdf["Soundfile"] = [
                os.path.join(x, y) + z
                for x, y, z in zip(
                    annot["audio_file_dir"],
                    annot["audio_file_name"],
                    annot["audio_file_extension"],
                )
            ]
            outdf = outdf.fillna(0)
            if single_file:
                outfilename = os.path.join(outdir, outfile)
            else:
                outfilename = os.path.join(
                    outdir,
                    str(annot["audio_file_name"].iloc[0])
                    + str(annot["audio_file_extension"].iloc[0])
                    + " annotations.log",
                )
            outdf.to_csv(
                outfilename,
                sep="\t",
                encoding="utf-8",
                header=True,
                columns=cols,
                index=False,
            )

    def from_parquet(self, file, verbose=False):
        """
        Import data from a Parquet file.

        Load annotations from a .parquet file. This format allows for fast and
        efficient data storage and access.

        Parameters
        ----------
        file : str
            Path of the input parquet file.

        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        self.data = pd.read_parquet(file)
        self.check_integrity(verbose=verbose)
        if verbose:
            print(len(self), "annotations imported.")

    def to_parquet(self, file):
        """
        Write data to a Parquet file.

        Write annotations as .parquet file. This format allows for fast and
        efficient data storage and access.

        Parameters
        ----------
        file : str
            Path of the output directory where the parquet files is written.

        Returns
        -------
        None.

        """
        # make sure the HP SN column are strings
        self.data.hydrophone_SN = self.data.hydrophone_SN.astype(str)
        # save
        self.data.to_parquet(
            file, coerce_timestamps="ms", allow_truncated_timestamps=True
        )

    def from_netcdf(self, files, verbose=False):
        """
        Import data from a netcdf file.

        Load annotations from a .nc file. This format works well with xarray
        and Dask.

        Parameters
        ----------
        files : str
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
        assert type(files) in (
            str,
            list,
        ), "Input must be of type str (single \
            file or directory) or list (multiple files)"
        files = Annotation._make_list_from_input(files, ".nc", verbose=verbose)
        # Import all files to a dataframe
        tmp = []
        for idx, file in enumerate(files):
            dxr = xr.open_dataset(file)
            if dxr.attrs["datatype"] == "Annotation":
                tmp2 = dxr.to_dataframe()
                tmp2.reset_index(inplace=True)
            elif dxr.attrs["datatype"] == "Measurement":
                tmp2 = dxr.to_dataframe()
                tmp2.reset_index(inplace=True)
                tmp2 = tmp2[self.get_fields()]
                warnings.warn(
                    "Importing Measurement data as Annotation >> Not all Measurement data are loaded."
                )
            else:
                raise ValueError(file + "Not an Annotation file.")
            tmp.append(tmp2)

        data = pd.concat(tmp, ignore_index=True, sort=False)
        data.reset_index(inplace=True, drop=True)
        self.data = data
        self.check_integrity(verbose=verbose)
        if verbose:
            print(len(self), "annotations imported.")

    def to_netcdf(self, file):
        """
        Write data to a netcdf file.

        Write annotations as .nc file. This format works well with xarray
        and Dask.

        Parameters
        ----------
        file : str
            Path of the output file (.nc) to be written.

        Returns
        -------
        None.

        """
        if file.endswith(".nc") is False:
            file = file + ".nc"
        self._enforce_dtypes()
        meas = self.data
        meas.set_index("time_min_date", drop=False, inplace=True)
        meas.index.name = "date"
        dxr1 = meas.to_xarray()
        dxr1.attrs["datatype"] = "Annotation"
        dxr1.to_netcdf(file, engine="netcdf4", format="NETCDF4")

    def to_csv(self, file):
        """
        Writes data from the Annoatation object to a csv file.

        Parameters
        ----------
        file : str
            Name and path of the output csv file.

        Returns
        -------
        None.

        """
        self.data.to_csv(file, index=False)

    def insert_values(self, **kwargs):
        """
        Insert constant values for given Annotation fields.

        Fill in entire columns of the annotation dataframe with constant
        values. It is usefull for adding project related informations that may
        not be included in data imported from Raven or PAMlab files (e.g.,
        'location_lat', 'location_lon'). Values can be inserted for several
        annotations fields at a time by setting several keywords. This should
        only be used for filling in static values (i.e., not for variable
        values such as time/frequency boundaries of the annotations). Keywords
        must have the exact same name as the annotation field (see method
        .get_fields). For example: (location_lat=48.6, recorder_type='AMAR')

        Parameters
        ----------
        **kwargs : annotation filed name
            Keyword and value of the annotation field to fill in. Keywords must
            have the exact same name as the annotation field.

        Raises
        ------
        ValueError
            If keyword doesn't match any annotation field name.

        Returns
        -------
        None.

        """
        for key, value in kwargs.items():
            if key in self.data:
                self.data[key] = value
            else:
                raise ValueError(
                    "The annotation object has no field: " + str(key)
                )

    def insert_metadata(self, deployment_info_file, channel=0):
        """
        Insert metadata infortion to the annotation.

        Uses the Deployment_info_file to fill in the metadata of the annotation
        . The deployment_info_file must be created using the DeploymentInfo
        class from ecosound.core.metadata using DeploymentInfo.write_template.

        Parameters
        ----------
        deployment_info_file : str
            Csv file readable by ecosound.core.meta.DeploymentInfo.read(). It
            contains all the deployment metadata.
        channel : int
            Channel number of the recorder for the metadata to insert.
            Default is 0

        Returns
        -------
        None.

        """
        channel = int(channel)
        dep_info = DeploymentInfo()
        dep_info.read(deployment_info_file)
        self.insert_values(
            UTC_offset=dep_info.data["UTC_offset"].values[channel],
            audio_channel=dep_info.data["audio_channel_number"].values[
                channel
            ],
            audio_sampling_frequency=dep_info.data[
                "sampling_frequency"
            ].values[channel],
            audio_bit_depth=dep_info.data["bit_depth"].values[channel],
            mooring_platform_name=dep_info.data[
                "mooring_platform_name"
            ].values[channel],
            recorder_type=dep_info.data["recorder_type"].values[channel],
            recorder_SN=dep_info.data["recorder_SN"].values[channel],
            hydrophone_model=dep_info.data["hydrophone_model"].values[channel],
            hydrophone_SN=dep_info.data["hydrophone_SN"].values[channel],
            hydrophone_depth=dep_info.data["hydrophone_depth"].values[channel],
            location_name=dep_info.data["location_name"].values[channel],
            location_lat=dep_info.data["location_lat"].values[channel],
            location_lon=dep_info.data["location_lon"].values[channel],
            location_water_depth=dep_info.data["location_water_depth"].values[
                channel
            ],
            deployment_ID=dep_info.data["deployment_ID"].values[channel],
        )

    def filter_overlap_with(
        self,
        annot,
        freq_ovp=True,
        dur_factor_max=None,
        dur_factor_min=None,
        ovlp_ratio_min=None,
        remove_duplicates=False,
        inherit_metadata=False,
        filter_deploymentID=True,
        inplace=False,
    ):
        """
        Filter overalaping annotations.

        Only keep annotations that overlap in time and/or frequency with the
        annotation object "annot".

        Parameters
        ----------
        annot : ecosound.annotation.Annotation object
            Annotation object used to filter the current annotations.
        freq_ovp : bool, optional
            If set to True, filters not only annotations that overlap in time
            but also overlap in frequency. The default is True.
        dur_factor_max : float, optional
            Constraint dictating the maximum duration overlapped
            annotations must not exceed in order to be "kept". Any annotations
            whose duration exceed dur_factor_max*annot.duration are discareded,
            even if they overlap in time/frequency. If set to None, no maximum
            duration constraints are applied. The default is None.
        dur_factor_min : float, optional
            Constraint dictating the minimum duration overlapped
            annotations must exceed in order to be "kept". Any annotations
            whose duration does not exceed dur_factor_min*annot.duration are
            discareded, even if they overlap in time/frequency. If set to None,
            no minimum duration constraints are applied. The default is None.
        ovlp_ratio_min : float, optional
            Constraint dictating the minimum amount (percentage) of overlap in
            time annotations must have in order to be "kept". If set to None,
            no minimum time overlap constraints are applied. The default is
            None.
        remove_duplicates : bool, optional
            If set to True, only selects a single annotation overlaping with
            annotations from the annot object. This is relevant only if several
            annotations overlap with an annotation from the annot object. The
            default is False.
        inherit_metadata : bool, optional
            If set to True, the filtered annotations inherit all the metadata
            information from the matched annotations in the annot object. It
            includes 'label_class', 'label_subclass', 'mooring_platform_name',
            'recorder_type', 'recorder_SN', 'hydrophone_model', 'hydrophone_SN'
            , 'hydrophone_depth', 'location_name', 'location_lat',
            'location_lon', 'location_water_depth', and 'deployment_ID'. The
            default is False.
        filter_deploymentID : bool, optional
            If set to False, doesn't use the deploymentID to match annotations
            together but just the frequency and time offset boundaries of the
            annotations. The default is True.
        inplace : bool, optional
            If set to True, updates the urrent object with the filter results.
            The default is False.

        Returns
        -------
        out_object : ecosound.annotation.Annotation
            Filtered Annotation object.

        """
        stack = []
        det = self.data
        for index, an in annot.data.iterrows():  # for each annotation
            # restrict to the specific deploymnetID of the annotation if file names are not unique
            if filter_deploymentID:
                df = det[det.deployment_ID == an.deployment_ID]
            else:
                df = det
            ## filter detections to same file and deployment ID as the current annotation
            df = df[df.audio_file_name == an.audio_file_name]
            ## check overlap in time first
            if len(df) > 0:
                df = df[
                    (
                        (df.time_min_offset <= an.time_min_offset)
                        & (df.time_max_offset >= an.time_max_offset)
                    )
                    | (  # 1- annot inside detec
                        (df.time_min_offset >= an.time_min_offset)
                        & (df.time_max_offset <= an.time_max_offset)
                    )
                    | (  # 2- detec inside annot
                        (df.time_min_offset < an.time_min_offset)
                        & (df.time_max_offset < an.time_max_offset)
                        & (df.time_max_offset > an.time_min_offset)
                    )
                    | (  # 3- only the end of the detec overlaps with annot
                        (df.time_min_offset > an.time_min_offset)
                        & (df.time_min_offset < an.time_max_offset)
                        & (df.time_max_offset > an.time_max_offset)
                    )  # 4- only the begining of the detec overlaps with annot
                ]
            # then looks at frequency overlap. Can be turned off if freq bounds are not reliable
            if (len(df) > 0) & freq_ovp:
                df = df[
                    (
                        (df.frequency_min <= an.frequency_min)
                        & (df.frequency_max >= an.frequency_max)
                    )
                    | (  # 1- annot inside detec
                        (df.frequency_min >= an.frequency_min)
                        & (df.frequency_max <= an.frequency_max)
                    )
                    | (  # 2- detec inside annot
                        (df.frequency_min < an.frequency_min)
                        & (df.frequency_max < an.frequency_max)
                        & (df.frequency_max > an.frequency_min)
                    )
                    | (  # 3- only the top of the detec overlaps with annot
                        (df.frequency_min > an.frequency_min)
                        & (df.frequency_min < an.frequency_max)
                        & (df.frequency_max > an.frequency_max)
                    )  # 4- only the bottom of the detec overlaps with annot
                ]
            # discard if durations are too different
            if (len(df) > 0) & (dur_factor_max is not None):
                df = df[df.duration < an.duration * dur_factor_max]
            if (len(df) > 0) & (dur_factor_min is not None):
                df = df[df.duration > an.duration * dur_factor_min]

            # discard if they don't overlap enough
            if (len(df) > 0) & (ovlp_ratio_min is not None):
                df_ovlp = (
                    df["time_max_offset"].apply(
                        lambda x: min(x, an.time_max_offset)
                    )
                    - df["time_min_offset"].apply(
                        lambda x: max(x, an.time_min_offset)
                    )
                ) / an.duration
                df = df[df_ovlp >= ovlp_ratio_min]
                df_ovlp = df_ovlp[df_ovlp >= ovlp_ratio_min]

            if (len(df) > 1) & remove_duplicates:
                try:
                    df = df.iloc[
                        [df_ovlp.values.argmax()]
                    ]  # pick teh one with max time overlap
                except:
                    print("asas")

            if len(df) > 0:
                if inherit_metadata:
                    df["mooring_platform_name"] = an["mooring_platform_name"]
                    df["recorder_type"] = an["recorder_type"]
                    df["recorder_SN"] = an["recorder_SN"]
                    df["hydrophone_model"] = an["hydrophone_model"]
                    df["hydrophone_SN"] = an["hydrophone_SN"]
                    df["hydrophone_depth"] = an["hydrophone_depth"]
                    df["location_name"] = an["location_name"]
                    df["location_lat"] = an["location_lat"]
                    df["location_lon"] = an["location_lon"]
                    df["location_water_depth"] = an["location_water_depth"]
                    df["deployment_ID"] = an["deployment_ID"]
                    df["label_class"] = an["label_class"]
                    df["label_subclass"] = an["label_subclass"]
                stack.append(df)
        if len(stack) > 0:
            ovlp = pd.concat(stack, ignore_index=True)
        else:
            ovlp = self.data[0:0]
        if inplace:
            self.data = ovlp
            self.check_integrity()
            out_object = None
        else:
            out_object = copy.copy(self)
            out_object.data = ovlp
            out_object.check_integrity()
        return out_object

    def calc_time_aggregate_1D(
        self,
        integration_time="1H",
        resampler="count",
        is_binary=False,
        start_date=None,
        end_date=None,
    ):
        """
        Calculate the 1D time aggregate of annotations.

        Calculate the time aggregate of annotations over the defined
        integration time.

        Parameters
        ----------
        integration_time : str, optional
            Integration time for the aggregate. Uses the pandas offset aliases
            (i.e. '2H'-> 2 hours, '15min'=> 15 minutes, '1D'-> 1 day) see pandas
            documnentation here:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            The default is '1H'.
        resampler : str, optional
            Defines method to combine aggregates. Currently only 'count' is
            implemented. The default is 'count'.
        start_date : None, str, optional
            Defines at which date the aggregate should start. If set to None,
            the min. date will be automatically chosen. If str must be in the
            format: yyyy-mm-dd HH:MM:SS. The default is None.
        end_date : None, str, optional
            Defines at which date the aggregate should end. If set to None,
            the max. date will be automatically chosen. If str must be in the
            format: yyyy-mm-dd HH:MM:SS. The default is None.
        is_binary : bool, optional
            If set to True, calculates the aggregates in term on presence (1)
            or absence (0). The default is False.

        Returns
        -------
        data_resamp : Pandas DataFrame
            1D DataFrame with datetime as the index and a 'value' column with the
            result of the aggregates for each time frane.
        """
        # calulate 1D aggreagate
        data = copy.copy(self.data)
        data.set_index("time_min_date", inplace=True)
        data_resamp = Annotation._resample(
            data,
            integration_time=integration_time,
            resampler=resampler,
            start_date=start_date,
            end_date=end_date,
        )
        data_resamp.set_index("datetime", inplace=True)
        if is_binary:
            data_resamp[data_resamp > 0] = 1
        return data_resamp

    def calc_time_aggregate_2D(
        self, integration_time="1H", resampler="count", is_binary=False
    ):
        """
        Calculate the 2D time aggregate of annotations.

        Calculate the time aggregate of annotations for each day and over each
        time of day defined by the integration time (i.e. Time of day vs Date).

        Parameters
        ----------
        integration_time : str, optional
            Integration time for the aggregate. Uses the pandas offset aliases
            (i.e. '2H'-> 2 hours, '15min'=> 15 minutes, '1D'-> 1 day) see pandas
            documnentation here:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
            The default is '1H'.
        resampler : str, optional
            Defines method to combine aggregates. Currently only 'count' is
            implemented. The default is 'count'.
        is_binary : bool, optional
            If set to True, calculates the aggregates in term on presence (1)
            or absence (0). The default is False.

        Returns
        -------
        data_resamp : Pandas DataFrame
            2D DataFrame with the time od day as the index and the date as
            columns. The resulting table contains the aggregate of annotations
            for each time of day and date.
        """
        # calulate 1D aggreagate
        data_resamp = self.calc_time_aggregate_1D(
            integration_time=integration_time, is_binary=is_binary
        )
        data_resamp.reset_index(inplace=True)
        data_resamp["date"] = data_resamp["datetime"].dt.date
        data_resamp["time"] = data_resamp["datetime"].dt.time
        # Create 2D matrix
        axis_date = sorted(data_resamp["date"].unique())
        data_grid = pd.pivot_table(
            data_resamp,
            values="value",
            index="time",
            columns="date",
            aggfunc=np.sum,
        )
        data_grid = data_grid.fillna(0)  # replaces NaNs by zeros
        if is_binary:
            data_grid[data_grid > 0] = 1
        return data_grid

    def heatmap(self, **kwargs):
        """
        Display heatmap of annotations (date vs time-of-day).

        Parameters
        ----------
        **kwargs :
            see documentation of ecosound.visualization.annotation_plotter.AnnotHeatmap

        Returns
        -------
        None.

        """
        print(kwargs)
        graph = GrapherFactory("AnnotHeatmap", **kwargs)
        graph.add_data(self)
        graph.show()

    def filter(self, query_str, inplace=False, **kwargs):
        """
        Filter data based on user-defined criteria.

        Uses the pandas dataframe.query method to filter rows of the annotation
        object (in annot.data datafrane). Filtering conditions are defined
        as query string (query_str) indicating the fields and value conditions.
        For example: query_str = 'label_class == "MW" & confidence >= 90'. See
        documentation of pandas.DataFrame.query for more details.

        Parameters
        ----------
        query_str : str
            query string defining annotations fiels and value conditions. For
            exemple: query_str = 'label_class == "MW" & confidence >= 90'.
        inplace : bool, optional
            Whether to modify the DataFrame rather than creating a new one.
            The default is True.

        Returns
        -------
        Annotation object
            Updated with the filtered data.

        """
        # unpack kwargs as separate variables
        for key, val in kwargs.items():
            exec(key + "=val")
        # del key, val
        # filter
        filt = self.data.query(query_str, inplace=inplace)
        # create output obj
        if inplace:
            out_object = None
        else:
            out_object = copy.copy(self)
            out_object.data = filt
            out_object.check_integrity()
        return out_object

    def merge_overlapped(self, time_tolerance_sec=None, inplace=False):

        # add temporary time shift
        if time_tolerance_sec:
            self.data.time_min_offset = self.data.time_min_offset - time_tolerance_sec
            self.data.time_max_offset = self.data.time_max_offset + time_tolerance_sec

        # get index of overlapped annots
        ovlp_idx_list = self._identify_ovlp_annot()

        # # merge
        # for annot_idx in ovlp_idx_list:
        #     # adjust t1 and t2, fmin, fmax etc
        #
        #     # apply merge rules for given columns (e.g. SNR, confidence)
        #
        #     # create new dataframe
        # print('here')


        # # remove temporary time shift
        # if time_tolerance_sec:
        #     self.data.time_min_offset = self.data.time_min_offset - time_tolerance_sec
        #     self.data.time_max_offset = self.data.time_max_offset + time_tolerance_sec
        #
        # if inplace:
        #     self.data = ovlp
        #     self.check_integrity()
        #     out_object = None
        # else:
        #     out_object = copy.copy(self)
        #     out_object.data = ovlp
        #     out_object.check_integrity()
        # return out_object

    def update_audio_dir(self, new_data_dir, verbose=False):
        """
        Update path of audio files

        Recursively finds the path of the annotations audio files in the folder
        provided in new_data_dir and automatically updates the annotation field
        "audio_file_dir". It is useful when the location of the audio data has
        moved or if using annotations on a different computer.

        Parameters
        ----------
        new_data_dir : str
            Path of the parent directory where the audio files are.
        verbose : bool
            Printprocess logs in command window if set to True. The defaut is
            False.

        Returns
        -------
        None.

        """
        # list name of all audio files in dataset
        dataset_files_list = set(
            self.data["audio_file_dir"]
            + os.path.sep
            + self.data["audio_file_name"]
            + self.data["audio_file_extension"]
        )

        # list extension of all audio files in dataset
        dataset_ext_list = set(
            [os.path.splitext(file)[1] for file in dataset_files_list]
        )
        if verbose:
            print(len(dataset_files_list), " audio files.")

        # list all audio files in new folder (only for the target file extensions)
        new_dir_files_list = []
        for ext in dataset_ext_list:
            new_dir_files_list = (
                new_dir_files_list
                + ecosound.core.tools.list_files(
                    new_data_dir, ext, recursive=True
                )
            )

        # go through each file in dataset and try to find in in new data folder
        missing_files_list = []
        for file in dataset_files_list:
            # if verbose:
            # print(file)
            res = [
                idx
                for idx, new_dir_file in enumerate(new_dir_files_list)
                if re.search(os.path.split(file)[1], new_dir_file)
            ]
            if len(res) == 0:
                missing_files_list.append(file)
            else:
                new_path = os.path.split(new_dir_files_list[res[0]])[0]
                self.data.loc[
                    self.data["audio_file_name"]
                    == os.path.splitext(os.path.split(file)[1])[0],
                    "audio_file_dir",
                ] = new_path

        if len(missing_files_list) > 0:
            warnings.warn(
                str(len(missing_files_list)) + " files could not be found."
            )
            if verbose:
                print("")
                print("List of audio files not found: ")
                print("")
                for ff in missing_files_list:
                    print(ff)
        else:
            if verbose:
                print("Audio paths succesfully updated.")

    def export_spectrograms(
        self,
        out_dir,
        time_buffer_sec=1,
        spectro_unit="samp",
        spetro_nfft=256,
        spetro_frame=256,
        spetro_inc=5,
        freq_min_hz=None,
        freq_max_hz=None,
        sanpling_rate_hz=None,
        filter_order=8,
        filter_type="iir",
        fig_size=(15, 10),
        deployment_subfolders=False,
        date_subfolders=False,
        file_name_field="uuid",
        file_prefix_field=None,
        channel=None,
        colormap="viridis",
        save_wav=False,
    ):

        # define the different class names and create separate folders
        if os.path.isdir(out_dir) == False:
            os.mkdir(out_dir)
        labels = list(set(self.data["label_class"]))
        # labels.reverse()

        # initialize spectrogram
        Spectro = Spectrogram(
            spetro_frame,
            "hann",
            spetro_nfft,
            spetro_inc,
            sanpling_rate_hz,
            unit=spectro_unit,
        )

        # loop through each class_labels
        for label in labels:
            # print(label)
            current_dir = os.path.join(out_dir, label)
            if os.path.isdir(current_dir) == False:
                os.mkdir(current_dir)
            annot_sp = self.data[self.data["label_class"] == label]

            # loop through is annot for that class label
            for idx, annot in tqdm(
                annot_sp.iterrows(),
                desc=label,
                leave=True,
                position=0,
                miniters=1,
                total=len(annot_sp),
                colour="green",
            ):

                # annot = annot_sp.iloc[1969]
                # idx = 1969
                # output file name
                F = self._convert_to_str(annot[file_name_field])

                # create subfolder for each deployment and each day if option selected
                if deployment_subfolders:
                    current_dir2 = os.path.join(
                        current_dir, str(annot.deployment_ID)
                    )
                else:
                    current_dir2 = current_dir
                if date_subfolders:
                    current_date = annot.time_min_date.strftime("%Y-%m-%d")
                    current_dir2 = os.path.join(current_dir2, current_date)
                if os.path.isdir(current_dir2) == False:
                    os.mkdir(current_dir2)

                # only if file doesn't exist already
                if os.path.isfile(os.path.join(current_dir2, F)) == False:
                    # print("Processing file", F)

                    # Load info from audio file
                    audio_data = Sound(
                        os.path.join(
                            annot["audio_file_dir"], annot["audio_file_name"]
                        )
                        + annot["audio_file_extension"]
                    )

                    # define start/stop times +/- buffer
                    t1 = annot.time_min_offset - time_buffer_sec
                    if t1 <= 0:
                        t1 = 0
                    t2 = annot.time_max_offset + time_buffer_sec
                    if t2 > audio_data.file_duration_sec:
                        t2 = audio_data.file_duration_sec
                    duration = t2 - t1

                    # load audio data
                    if channel != None:
                        chan = int(channel)
                    else:
                        chan = annot["audio_channel"] - 1
                    audio_data.read(
                        channel=chan,
                        chunk=[t1, t2],
                        unit="sec",
                        detrend=True,
                    )

                    # decimate
                    audio_data.decimate(sanpling_rate_hz)

                    # normalize
                    audio_data.normalize()

                    # compute spectrogram
                    _ = Spectro.compute(audio_data, dB=True, use_dask=False)

                    # crop if needed
                    if freq_min_hz != None or freq_max_hz != None:
                        Spectro.crop(
                            frequency_min=freq_min_hz,
                            frequency_max=freq_max_hz,
                            inplace=True,
                        )

                    # display/save spectrogram as image file
                    graph = GrapherFactory(
                        "SoundPlotter",
                        title=annot["audio_file_name"]
                        + " - "
                        + str(abs(round(annot["time_min_offset"], 2))),
                        fig_size=fig_size,
                        colormap=colormap,
                    )
                    # crop plot if needed
                    if freq_min_hz != None:
                        graph.frequency_min = freq_min_hz
                    if freq_max_hz != None:
                        graph.frequency_max = freq_max_hz

                    # output file name
                    graph.add_data(Spectro)
                    if file_prefix_field:
                        prefix = self._convert_to_str(annot[file_prefix_field])
                        full_out_file = os.path.join(
                            current_dir2, prefix + "_" + F
                        )
                    else:
                        full_out_file = os.path.join(current_dir2, F)

                    graph.to_file(full_out_file + ".png")

                    if save_wav:
                        audio_data.write(full_out_file + ".wav")

                    # graph.show()

                    # if params["spetro_on_npy"]:
                    #    np.save(os.path.splitext(outfilename)[0] + ".npy", S)
                    # annot_unique_id += 1
                # else:
                # print("file ", F, " already processed.")

    def get_labels_class(self):
        """
        Get all the unique class labels of the annotations.

        Returns
        -------
        classes : list
            List of unique class labels.

        """
        if len(self.data) > 0:
            classes = list(self.data["label_class"].unique())
        else:
            classes = []
        return classes

    def get_labels_subclass(self):
        """
        Get all the unique subclass labels of the annotations.

        Returns
        -------
        classes : list
            List of unique subclass labels.

        """
        if len(self.data) > 0:
            subclasses = list(self.data["label_subclass"].unique())
        else:
            subclasses = []
        return subclasses

    def get_fields(self):
        """
        Get all the annotations fields.

        Returns
        -------
        classes : list
            List of annotation fields.

        """
        return list(self.data.columns)

    def summary(self, rows="deployment_ID", columns="label_class"):
        """
        Produce a summary table of the number of annotations.

        Create a pivot table summarizing the number of annotations for each
        deployment and each label class. The optional arguments 'rows' and
        'columns' can be used to change the fields of the annotations to be
        displayed in the table.

        Parameters
        ----------
        rows : 'str', optional
            Name of the annotation field for the rows of the table. The default
            is 'deployment_ID'.
        columns : 'str', optional
            Name of the annotation field for the columns of the table. The
            default default is 'label_class'.

        Returns
        -------
        summary : pandas DataFrame
            Pivot table with the number of annotations in each category.

        """
        summary = self.data.pivot_table(
            index=rows, columns=columns, aggfunc="size", fill_value=0
        )
        # Add a "Total" row and column
        summary.loc["Total"] = summary.sum()
        summary["Total"] = summary.sum(axis=1)
        return summary

    def _identify_ovlp_annot(self):
        stack = []
        data = self.data
        files_list = list(set(data.audio_file_name))
        for file in files_list:  # for each audio file
            file_data = data.query('audio_file_name==@file')  # data for a single file
            while len(file_data) > 0:
                t1 = file_data.iloc[0].time_min_offset
                t2 = file_data.iloc[0].time_max_offset
                # stack index + update dataframe
                tmp = []
                index_id = file_data.iloc[0].name
                tmp.append(index_id)
                file_data = file_data.drop([index_id], axis=0)  # delete annot alreday stacked in tmp
                while True:
                    # find other annot overlaping with curent annot
                    ovlp = file_data[
                        (
                            (file_data.time_min_offset <= t1)
                            & (file_data.time_max_offset >= t2)
                        )
                        | (  # 1- annot inside detec
                            (file_data.time_min_offset >= t1)
                            & (file_data.time_max_offset <= t2)
                        )
                        | (  # 2- detec inside annot
                            (file_data.time_min_offset < t1)
                            & (file_data.time_max_offset < t2)
                            & (file_data.time_max_offset > t1)
                        )
                        | (  # 3- only the end of the detec overlaps with annot
                            (file_data.time_min_offset > t1)
                            & (file_data.time_min_offset < t2)
                            & (file_data.time_max_offset > t2)
                        )  # 4- only the begining of the detec overlaps with annot
                        ]
                    if len(ovlp) == 0:  # no overlap
                        stack.append(tmp)
                        break
                    elif len(ovlp) > 0:  # 1 of more overlaps
                        index_ids = list(ovlp.index.values)
                        for index_id in index_ids:
                            tmp.append(index_id)
                            t1 = min([t1, ovlp.time_min_offset.values[0]])
                            t2 = max([t2, ovlp.time_max_offset.values[0]])
                            file_data = file_data.drop([index_id], axis=0)  # delete annot alreday stacked in tmp
        #print('done')
        # Sanity check
        annot_count = sum([len(ovlp) for ovlp in stack])
        if annot_count != len(data):
            print('overlapped annotation identified do not add up to the total number of annotations')
        return stack

    @staticmethod
    def _resample(
        data,
        integration_time="1H",
        resampler="count",
        start_date=None,
        end_date=None,
    ):

        # define start time of the aggregate
        if start_date is None:
            start_date = min(data.index)
        elif type(start_date) is str:
            start_date = pd.Timestamp(start_date)

        # define end time of the aggregate
        if end_date is None:
            end_date = max(data.index)
        elif type(end_date) is str:
            end_date = pd.Timestamp(end_date)

        start_date = start_date.to_period(integration_time).to_timestamp()
        end_date = end_date.to_period(integration_time).to_timestamp()

        t_index = pd.DatetimeIndex(
            pd.date_range(
                start=start_date,
                end=end_date,
                freq=integration_time,
            )
        )
        if resampler == "count":
            # Lmean = data.resample(integration_time, loffset=None, label='left').apply(count)
            data_new = data.resample(
                integration_time,
                #loffset=None,
                origin="start_day",
                label="left",
            ).count()

        data_out = pd.DataFrame(
            {"datetime": data_new.index, "value": data_new["uuid"]}
        )
        data_out = data_out.reindex(t_index).fillna(0)
        data_out["datetime"] = data_out.index
        data_out.reset_index(drop=True, inplace=True)
        return data_out

    def _enforce_dtypes(self):
        self.data = self.data.astype(
            {
                "uuid": "str",
                "from_detector": "bool",  # True, False
                "software_name": "str",
                "software_version": "str",
                "operator_name": "str",
                "UTC_offset": "float",
                "entry_date": "datetime64[ns]",
                "audio_channel": "int",
                "audio_file_name": "str",
                "audio_file_dir": "str",
                "audio_file_extension": "str",
                "audio_file_start_date": "datetime64[ns]",
                "audio_sampling_frequency": "int",
                "audio_bit_depth": "int",
                "mooring_platform_name": "str",
                "recorder_type": "str",
                "recorder_SN": "str",
                "hydrophone_model": "str",
                "hydrophone_SN": "str",
                "hydrophone_depth": "float",
                "location_name": "str",
                "location_lat": "float",
                "location_lon": "float",
                "location_water_depth": "float",
                "deployment_ID": "str",
                "frequency_min": "float",
                "frequency_max": "float",
                "time_min_offset": "float",
                "time_max_offset": "float",
                "time_min_date": "datetime64[ns]",
                "time_max_date": "datetime64[ns]",
                "duration": "float",
                "label_class": "str",
                "label_subclass": "str",
                "confidence": "float",
            }
        )

    @staticmethod
    @ecosound.core.decorators.listinput
    def _import_csv_files(files):
        """Import one or several text files with header to a Panda datafrane."""
        assert type(files) in (
            str,
            list,
        ), "Input must be of type str (single \
            file) or list (multiple files)"
        # Import all files to a dataframe
        for idx, file in enumerate(files):
            # Extract header first due to formating issues in PAMlab files
            header = pd.read_csv(file, delimiter="\t", header=None, nrows=1)
            headerLength = header.shape[1]
            # Get all data and only keep values correpsonding to header labels
            # tmp = pd.read_csv(file,
            #                   delimiter='\t',
            #                   header=None,
            #                   #header=True,
            #                   #skiprows=1,
            #                   na_values=None,
            #                   )
            # tmp = tmp.iloc[:, 0:headerLength]
            # Put header back
            # tmp = tmp.set_axis(list(header.values[0]), axis=1, inplace=False)
            tmp = pd.read_csv(file, delimiter="\t", na_values=None)
            if idx == 0:
                data = tmp
            else:
                data = pd.concat([data, tmp], ignore_index=True, sort=False)
        return data

    @staticmethod
    def _make_list_from_input(files, file_ext, verbose=True):
        if type(files) is str:
            if os.path.isdir(files):
                files = ecosound.core.tools.list_files(
                    files,
                    file_ext,
                    recursive=False,
                    case_sensitive=True,
                )
                if verbose:
                    print(len(files), "files found.")
            else:
                files = [files]
        return files

    def _convert_to_str(self, value):
        if type(value) is str:
            F = value
        elif type(value) is float:
            if value < 0:
                F = "minus-" + str(abs(round(value, 2)))
            else:
                F = str(round(value, 2))
        elif type(value) is pd.Timestamp:
            if self.data.UTC_offset.iloc[0] >= 0:
                sign_str = "+"
            else:
                sign_str = "-"
            tz_str = datetime.datetime(
                year=1,
                month=1,
                day=1,
                hour=int(abs(self.data.UTC_offset.iloc[0])),
            ).strftime("%H%M")

            F = value.strftime("%Y%m%dT%H%M%S.%f") + sign_str + tz_str
        elif np.isnan(value):
            F = "nan"
        return F

    def __add__(self, other):
        """Concatenate data from several annotation objects."""
        assert (
            type(other) is ecosound.core.annotation.Annotation
        ), "Object type not \
            supported. Can only concatenate Annotation objects together."
        self._enforce_dtypes()
        other._enforce_dtypes()
        self.data = pd.concat(
            [self.data, other.data], ignore_index=True, sort=False
        )
        return self

    def __repr__(self):
        """Return the type of object."""
        return f"{self.__class__.__name__} object (" f"{len(self.data)})"

    def __str__(self):
        """Return string when used with print of str."""
        return f"{len(self.data)} annotation(s)"

    def __len__(self):
        """Return number of annotations."""
        return len(self.data)
