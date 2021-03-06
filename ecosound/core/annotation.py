# -*- coding: utf-8 -*-.
"""
Created on Tue Jan 21 13:04:58 2020

@author: xavier.mouy
"""

import pandas as pd
import xarray as xr
import os
import uuid
import warnings
import ecosound.core.tools
import ecosound.core.decorators
from ecosound.core.metadata import DeploymentInfo
import copy


class Annotation():
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
        self.data = pd.DataFrame({
            'uuid': [],
            'from_detector': [],  # True, False
            'software_name': [],
            'software_version': [],
            'operator_name': [],
            'UTC_offset': [],
            'entry_date': [],
            'audio_channel': [],
            'audio_file_name': [],
            'audio_file_dir': [],
            'audio_file_extension': [],
            'audio_file_start_date': [],
            'audio_sampling_frequency': [],
            'audio_bit_depth': [],
            'mooring_platform_name': [],
            'recorder_type': [],
            'recorder_SN': [],
            'hydrophone_model': [],
            'hydrophone_SN': [],
            'hydrophone_depth': [],
            'location_name': [],
            'location_lat': [],
            'location_lon': [],
            'location_water_depth': [],
            'deployment_ID': [],
            'frequency_min': [],
            'frequency_max': [],
            'time_min_offset': [],
            'time_max_offset': [],
            'time_min_date': [],
            'time_max_date': [],
            'duration': [],
            'label_class': [],
            'label_subclass': [],
            'confidence': []
            })
        self._enforce_dtypes()

    def check_integrity(self, verbose=False, ignore_frequency_duplicates=False):
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
                subset=['time_min_offset',
                        'time_max_offset',
                        'label_class',
                        'label_subclass',
                        'audio_file_name',
                        ], keep="first",).reset_index(drop=True)
        else:  # remove annot with exact same time AND frequency boundaries
            self.data = self.data.drop_duplicates(
                subset=['time_min_offset',
                        'time_max_offset',
                        'frequency_min',
                        'frequency_max',
                        'label_class',
                        'label_subclass',
                        'audio_file_name',
                        ], keep="first",).reset_index(drop=True)
        count_stop = len(self.data)
        if verbose:
            print('Duplicate entries removed:', str(count_start-count_stop))
        # Check that start and stop times are coherent (i.e. t2 > t1)
        time_check = self.data.index[
            self.data['time_max_offset'] <
            self.data['time_min_offset']].tolist()
        if len(time_check) > 0:
            raise ValueError(
                'Incoherent annotation times (time_min > time_max). \
                 Problematic annotations:' + str(time_check))
        # Check that min and max frequencies are coherent (i.e. fmin < fmax)
        freq_check = self.data.index[
            self.data['frequency_max'] < self.data['frequency_min']].tolist()
        if len(freq_check) > 0:
            raise ValueError(
                'Incoherent annotation frequencies (frequency_min > \
                frequency_max). Problematic annotations:' + str(freq_check))
        if verbose:
            print('Integrity test succesfull')

    def from_raven(self, files, class_header='Sound type', subclass_header=None, verbose=False):
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
        verbose : bool, optional
            If set to True, print the summary of the annatation integrity test.
            The default is False.

        Returns
        -------
        None.

        """
        if os.path.isdir(files):
            files = ecosound.core.tools.list_files(files,
                                                   '.selections.txt',
                                                   recursive=False,
                                                   case_sensitive=True,
                                                   )
            if verbose:
                print(len(files), 'annotation files found.')
        data = Annotation._import_csv_files(files)
        files_timestamp = ecosound.core.tools.filename_to_datetime(
            data['Begin Path'].tolist())
        self.data['audio_file_start_date'] = files_timestamp
        self.data['audio_channel'] = data['Channel']
        self.data['audio_file_name'] = data['Begin Path'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        self.data['audio_file_dir'] = data['Begin Path'].apply(
            lambda x: os.path.dirname(x))
        self.data['audio_file_extension'] = data['Begin Path'].apply(
            lambda x: os.path.splitext(x)[1])
        self.data['time_min_offset'] = data['Begin Time (s)']
        self.data['time_max_offset'] = data['End Time (s)']
        self.data['time_min_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] + pd.to_timedelta(
                self.data['time_min_offset'], unit='s'))
        self.data['time_max_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] +
            pd.to_timedelta(self.data['time_max_offset'], unit='s'))
        self.data['frequency_min'] = data['Low Freq (Hz)']
        self.data['frequency_max'] = data['High Freq (Hz)']
        if class_header is not None:
            self.data['label_class'] = data[class_header]
        if subclass_header is not None:
            self.data['label_subclass'] = data[subclass_header]
        self.data['from_detector'] = False
        self.data['software_name'] = 'raven'
        self.data['uuid'] = self.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=verbose, ignore_frequency_duplicates=True)
        if verbose:
                print(len(self), 'annotations imported.')

    def to_raven(self, outdir, outfile='Raven.Table.1.selections.txt', single_file=False):
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
        if single_file:
            annots = [self.data]
        else:
            annots = [pd.DataFrame(y) for x, y in self.data.groupby(
                'audio_file_name', as_index=False)]
        for annot in annots:
            annot.reset_index(inplace=True, drop=True)
            cols = ['Selection', 'View', 'Channel', 'Begin Time (s)',
                    'End Time (s)', 'Delta Time (s)', 'Low Freq (Hz)',
                    'High Freq (Hz)', 'Begin Path', 'File Offset (s)',
                    'Begin File', 'Class', 'Sound type', 'Software',
                    'Confidence']
            outdf = pd.DataFrame({'Selection': 0, 'View': 0, 'Channel': 0,
                                  'Begin Time (s)': 0, 'End Time (s)': 0,
                                  'Delta Time (s)': 0, 'Low Freq (Hz)': 0,
                                  'High Freq (Hz)': 0, 'Begin Path': 0,
                                  'File Offset (s)': 0, 'Begin File': 0,
                                  'Class': 0, 'Sound type': 0, 'Software': 0,
                                  'Confidence': 0},
                                 index=list(range(annot.shape[0])))
            outdf['Selection'] = range(1, annot.shape[0]+1)
            outdf['View'] = 'Spectrogram 1'
            outdf['Channel'] = annot['audio_channel']
            outdf['Begin Time (s)'] = annot['time_min_offset']
            outdf['End Time (s)'] = annot['time_max_offset']
            outdf['Delta Time (s)'] = annot['duration']
            outdf['Low Freq (Hz)'] = annot['frequency_min']
            outdf['High Freq (Hz)'] = annot['frequency_max']
            outdf['File Offset (s)'] = annot['time_min_offset']
            outdf['Class'] = annot['label_class']
            outdf['Sound type'] = annot['label_subclass']
            outdf['Software'] = annot['software_name']
            outdf['Confidence'] = annot['confidence']
            outdf['Begin Path'] = [os.path.join(x, y) + z for x, y, z
                                   in zip(annot['audio_file_dir'],
                                          annot['audio_file_name'],
                                          annot['audio_file_extension'])]
            outdf['Begin File'] = [x + y for x, y
                                   in zip(annot['audio_file_name'],
                                          annot['audio_file_extension'])]
            outdf = outdf.fillna(0)
            if single_file:
                outfilename = os.path.join(outdir, outfile)
            else:
                outfilename = os.path.join(
                    outdir, str(annot['audio_file_name'].iloc[0]))
                + str(annot['audio_file_extension'].iloc[0])
                + '.chan' + str(annot['audio_channel'].iloc[0])
                + '.Table.1.selections.txt'
            outdf.to_csv(outfilename,
                         sep='\t',
                         encoding='utf-8',
                         header=True,
                         columns=cols,
                         index=False)

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
                files = ecosound.core.tools.list_files(files,
                                                       ' annotations.log',
                                                       recursive=False,
                                                       case_sensitive=True,
                                                       )
                if verbose:
                    print(len(files), 'annotation files found.')
        data = Annotation._import_csv_files(files)
        files_timestamp = ecosound.core.tools.filename_to_datetime(
            data['Soundfile'].tolist())
        self.data['audio_file_start_date'] = files_timestamp
        self.data['operator_name'] = data['Operator']
        self.data['entry_date'] = pd.to_datetime(
            data['Annotation date and time (local)'],
            format='%Y-%m-%d %H:%M:%S.%f')
        self.data['audio_channel'] = data['Channel']
        self.data['audio_file_name'] = data['Soundfile'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0])
        self.data['audio_file_dir'] = data['Soundfile'].apply(
            lambda x: os.path.dirname(x))
        self.data['audio_file_extension'] = data['Soundfile'].apply(
            lambda x: os.path.splitext(x)[1])
        self.data['audio_sampling_frequency'] = data['Sampling freq (Hz)']
        self.data['recorder_type'] = data['Recorder type']
        self.data['recorder_SN'] = data['Recorder ID']
        self.data['hydrophone_depth'] = data['Recorder depth']
        self.data['location_name'] = data['Station']
        self.data['location_lat'] = data['Latitude (deg)']
        self.data['location_lon'] = data['Longitude (deg)']
        self.data['time_min_offset'] = data['Left time (sec)']
        self.data['time_max_offset'] = data['Right time (sec)']
        self.data['time_min_date'] = pd.to_datetime(
            self.data['audio_file_start_date']
            + pd.to_timedelta(self.data['time_min_offset'], unit='s'))
        self.data['time_max_date'] = pd.to_datetime(
            self.data['audio_file_start_date'] +
            pd.to_timedelta(self.data['time_max_offset'], unit='s'))
        self.data['frequency_min'] = data['Bottom freq (Hz)']
        self.data['frequency_max'] = data['Top freq (Hz)']
        self.data['label_class'] = data['Species']
        self.data['label_subclass'] = data['Call type']
        self.data['from_detector'] = False
        self.data['software_name'] = 'pamlab'
        self.data['uuid'] = self.data.apply(lambda _: str(uuid.uuid4()), axis=1)
        self.data['duration'] = self.data['time_max_offset'] - self.data['time_min_offset']
        self.check_integrity(verbose=verbose)
        if verbose:
            print(len(self), 'annotations imported.')

    def to_pamlab(self, outdir, outfile='PAMlab annotations.log', single_file=False):
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
            annots = [pd.DataFrame(y)
                      for x, y in self.data.groupby(
                              'audio_file_name', as_index=False)]
        for annot in annots:
            annot.reset_index(inplace=True, drop=True)
            cols = ['fieldkey:', 'Soundfile', 'Channel', 'Sampling freq (Hz)',
                    'Latitude (deg)', 'Longitude (deg)', 'Recorder ID',
                    'Recorder depth', 'Start date and time (UTC)',
                    'Annotation date and time (local)', 'Recorder type',
                    'Deployment', 'Station', 'Operator', 'Left time (sec)',
                    'Right time (sec)', 'Top freq (Hz)', 'Bottom freq (Hz)',
                    'Species', 'Call type', 'rms SPL', 'SEL', '', '']
            outdf = pd.DataFrame({'fieldkey:': 0, 'Soundfile': 0, 'Channel': 0,
                                  'Sampling freq (Hz)': 0, 'Latitude (deg)': 0,
                                  'Longitude (deg)': 0, 'Recorder ID': 0,
                                  'Recorder depth': 0,
                                  'Start date and time (UTC)': 0,
                                  'Annotation date and time (local)': 0,
                                  'Recorder type': 0, 'Deployment': 0,
                                  'Station': 0, 'Operator': 0,
                                  'Left time (sec)': 0, 'Right time (sec)': 0,
                                  'Top freq (Hz)': 0, 'Bottom freq (Hz)': 0,
                                  'Species': '', 'Call type': '', 'rms SPL': 0,
                                  'SEL': 0, '': '', '': ''},
                                 index=list(range(annot.shape[0])))
            outdf['fieldkey:'] = 'an:'
            outdf['Species'] = annot['label_class']
            outdf['Call type'] = annot['label_subclass']
            outdf['Left time (sec)'] = annot['time_min_offset']
            outdf['Right time (sec)'] = annot['time_max_offset']
            outdf['Top freq (Hz)'] = annot['frequency_max']
            outdf['Bottom freq (Hz)'] = annot['frequency_min']
            outdf['rms SPL'] = annot['confidence']
            outdf['Operator'] = annot['operator_name']
            outdf['Channel'] = annot['audio_channel']
            outdf['Annotation date and time (local)'] = annot['entry_date']
            outdf['Sampling freq (Hz)'] = annot['audio_sampling_frequency']
            outdf['Recorder type'] = annot['recorder_type']
            outdf['Recorder ID'] = annot['recorder_SN']
            outdf['Recorder depth'] = annot['hydrophone_depth']
            outdf['Station'] = annot['location_name']
            outdf['Latitude (deg)'] = annot['location_lat']
            outdf['Longitude (deg)'] = annot['location_lon']
            outdf['Soundfile'] = [os.path.join(x, y) + z for x, y, z
                                  in zip(annot['audio_file_dir'],
                                         annot['audio_file_name'],
                                         annot['audio_file_extension']
                                         )
                                  ]
            outdf = outdf.fillna(0)
            if single_file:
                outfilename = os.path.join(outdir, outfile)
            else:
                outfilename = os.path.join(
                    outdir,
                    str(annot['audio_file_name'].iloc[0]))
                + str(annot['audio_file_extension'].iloc[0])
                + ' annotations.log'
            outdf.to_csv(outfilename,
                         sep='\t',
                         encoding='utf-8',
                         header=True,
                         columns=cols,
                         index=False)

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
            print(len(self), 'annotations imported.')

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
        self.data.to_parquet(file,
                             coerce_timestamps='ms',
                             allow_truncated_timestamps=True)

    def from_netcdf(self, file, verbose=False):
        """
        Import data from a netcdf file.

        Load annotations from a .nc file. This format works well with xarray
        and Dask.

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
                    '.nc',
                    recursive=False,
                    case_sensitive=True,)
                if verbose:
                    print(len(file), 'files found.')
            else:
                file = [file]
        self.data = self._import_netcdf_files(file)
        self.check_integrity(verbose=verbose)
        if verbose:
            print(len(self), 'annotations imported.')

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
        if file.endswith('.nc') is False:
            file = file + '.nc'
        self._enforce_dtypes()
        meas = self.data
        meas.set_index('time_min_date', drop=False, inplace=True)
        meas.index.name = 'date'
        dxr1 = meas.to_xarray()
        dxr1.attrs['datatype'] = 'Annotation'
        dxr1.to_netcdf(file, engine='netcdf4', format='NETCDF4')

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
                raise ValueError('The annotation object has no field: '
                                 + str(key))

    def insert_metadata(self, deployment_info_file):
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

        Returns
        -------
        None.

        """
        dep_info = DeploymentInfo()
        dep_info.read(deployment_info_file)
        self.insert_values(UTC_offset=dep_info.data['UTC_offset'].values[0],
                           audio_channel=dep_info.data['audio_channel_number'].values[0],
                           audio_sampling_frequency=dep_info.data['sampling_frequency'].values[0],
                           audio_bit_depth=dep_info.data['bit_depth'].values[0],
                           mooring_platform_name = dep_info.data['mooring_platform_name'].values[0],
                           recorder_type=dep_info.data['recorder_type'].values[0],
                           recorder_SN=dep_info.data['recorder_SN'].values[0],
                           hydrophone_model=dep_info.data['hydrophone_model'].values[0],
                           hydrophone_SN=dep_info.data['hydrophone_SN'].values[0],
                           hydrophone_depth=dep_info.data['hydrophone_depth'].values[0],
                           location_name=dep_info.data['location_name'].values[0],
                           location_lat=dep_info.data['location_lat'].values[0],
                           location_lon=dep_info.data['location_lon'].values[0],
                           location_water_depth=dep_info.data['location_water_depth'].values[0],
                           deployment_ID=dep_info.data['deployment_ID'].values[0],
                          )

    def filter_overlap_with(self, annot, freq_ovp=True,dur_factor_max=None,dur_factor_min=None,ovlp_ratio_min=None,remove_duplicates=False,inherit_metadata=False,filter_deploymentID=True, inplace=False):
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
                df = df[((df.time_min_offset <= an.time_min_offset) & (df.time_max_offset >= an.time_max_offset)) |  # 1- annot inside detec
                         ((df.time_min_offset >= an.time_min_offset) & (df.time_max_offset <= an.time_max_offset)) |  # 2- detec inside annot
                         ((df.time_min_offset < an.time_min_offset) & (df.time_max_offset < an.time_max_offset) & (df.time_max_offset > an.time_min_offset)) | # 3- only the end of the detec overlaps with annot
                         ((df.time_min_offset > an.time_min_offset) & (df.time_min_offset < an.time_max_offset) & (df.time_max_offset > an.time_max_offset)) # 4- only the begining of the detec overlaps with annot
                          ]
            # then looks at frequency overlap. Can be turned off if freq bounds are not reliable
            if (len(df) > 0) & freq_ovp:
                df = df[((df.frequency_min <= an.frequency_min) & (df.frequency_max >= an.frequency_max)) | # 1- annot inside detec
                        ((df.frequency_min >= an.frequency_min) & (df.frequency_max <= an.frequency_max)) | # 2- detec inside annot
                        ((df.frequency_min < an.frequency_min) & (df.frequency_max < an.frequency_max) & (df.frequency_max > an.frequency_min)) | # 3- only the top of the detec overlaps with annot
                        ((df.frequency_min > an.frequency_min) & (df.frequency_min < an.frequency_max) & (df.frequency_max > an.frequency_max)) # 4- only the bottom of the detec overlaps with annot
                        ]
            # discard if durations are too different
            if (len(df) > 0) & (dur_factor_max is not None):
                df = df[df.duration < an.duration*dur_factor_max]
            if (len(df) > 0) & (dur_factor_min is not None):
                df = df[df.duration > an.duration*dur_factor_min]

            # discard if they don't overlap enough
            if (len(df) > 0) & (ovlp_ratio_min is not None):
                df_ovlp = (df['time_max_offset'].apply(lambda x: min(x,an.time_max_offset)) - df['time_min_offset'].apply(lambda x: max(x,an.time_min_offset))) / an.duration
                df = df[df_ovlp>=ovlp_ratio_min]
                df_ovlp = df_ovlp[df_ovlp>=ovlp_ratio_min]

            if (len(df) > 1) & remove_duplicates:
                try:
                    df = df.iloc[[df_ovlp.values.argmax()]] # pick teh one with max time overlap
                except:
                    print('asas')

            if len(df) > 0:
                if inherit_metadata:
                    df['mooring_platform_name'] = an['mooring_platform_name']
                    df['recorder_type'] = an['recorder_type']
                    df['recorder_SN'] = an['recorder_SN']
                    df['hydrophone_model'] = an['hydrophone_model']
                    df['hydrophone_SN'] = an['hydrophone_SN']
                    df['hydrophone_depth'] = an['hydrophone_depth']
                    df['location_name'] = an['location_name']
                    df['location_lat'] = an['location_lat']
                    df['location_lon'] = an['location_lon']
                    df['location_water_depth'] = an['location_water_depth']
                    df['deployment_ID'] = an['deployment_ID']
                    df['label_class'] = an['label_class']
                    df['label_subclass'] = an['label_subclass']
                stack.append(df)
        ovlp = pd.concat(stack, ignore_index=True)
        if inplace:
            self.data = ovlp
            self.check_integrity()
            out_object = None
        else:
            out_object = copy.copy(self)
            out_object.data = ovlp
            out_object.check_integrity()
        return out_object

    def get_labels_class(self):
        """
        Get all the unique class labels of the annotations.

        Returns
        -------
        classes : list
            List of unique class labels.

        """
        if len(self.data) > 0:
            classes = list(self.data['label_class'].unique())
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
            subclasses = list(self.data['label_subclass'].unique())
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

    def summary(self, rows='deployment_ID', columns='label_class'):
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
        summary = self.data.pivot_table(index=rows,
                                        columns=columns,
                                        aggfunc='size',
                                        fill_value=0)
        # Add a "Total" row and column
        summary.loc['Total'] = summary.sum()
        summary['Total'] = summary.sum(axis=1)
        return summary

    def _enforce_dtypes(self):
        self.data = self.data.astype({
            'uuid': 'str',
            'from_detector': 'bool',  # True, False
            'software_name': 'str',
            'software_version': 'str',
            'operator_name': 'str',
            'UTC_offset': 'float',
            'entry_date': 'datetime64[ns]',
            'audio_channel': 'int',
            'audio_file_name': 'str',
            'audio_file_dir': 'str',
            'audio_file_extension': 'str',
            'audio_file_start_date': 'datetime64[ns]',
            'audio_sampling_frequency': 'int',
            'audio_bit_depth': 'int',
            'mooring_platform_name': 'str',
            'recorder_type': 'str',
            'recorder_SN': 'str',
            'hydrophone_model': 'str',
            'hydrophone_SN': 'str',
            'hydrophone_depth': 'float',
            'location_name': 'str',
            'location_lat': 'float',
            'location_lon': 'float',
            'location_water_depth': 'float',
            'deployment_ID': 'str',
            'frequency_min': 'float',
            'frequency_max': 'float',
            'time_min_offset': 'float',
            'time_max_offset': 'float',
            'time_min_date': 'datetime64[ns]',
            'time_max_date': 'datetime64[ns]',
            'duration': 'float',
            'label_class': 'str',
            'label_subclass': 'str',
            'confidence': 'float',
            })

    @staticmethod
    @ecosound.core.decorators.listinput
    def _import_csv_files(files):
        """Import one or several text files with header to a Panda datafrane."""
        assert type(files) in (str, list), "Input must be of type str (single \
            file) or list (multiple files)"
        # Import all files to a dataframe
        for idx, file in enumerate(files):
            # Extract header first due to formating issues in PAMlab files
            header = pd.read_csv(file,
                                 delimiter='\t',
                                 header=None,
                                 nrows=1)
            headerLength = header.shape[1]
            # Get all data and only keep values correpsonding to header labels
            tmp = pd.read_csv(file,
                              delimiter='\t',
                              header=None,
                              skiprows=1,
                              na_values=None)
            tmp = tmp.iloc[:, 0:headerLength]
            # Put header back
            tmp = tmp.set_axis(list(header.values[0]), axis=1, inplace=False)
            if idx == 0:
                data = tmp
            else:
                data = pd.concat([data, tmp], ignore_index=True, sort=False)
        return data

    def _import_netcdf_files(self, files):
        """Import one or several netcdf files to a Panda datafrane."""
        assert type(files) in (str, list), "Input must be of type str (single \
            file or directory) or list (multiple files)"
        # Import all files to a dataframe
        tmp = []
        for idx, file in enumerate(files):
            dxr = xr.open_dataset(file)
            if dxr.attrs['datatype'] == 'Annotation':
                tmp2 = dxr.to_dataframe()
                tmp2.reset_index(inplace=True)
            elif dxr.attrs['datatype'] == 'Measurement':
                tmp2 = dxr.to_dataframe()
                tmp2.reset_index(inplace=True)
                tmp2 = tmp2[self.get_fields()]
                warnings.warn('Importing Measurement data as Annotation >> Not all Measurement data are loaded.')
            else:
                raise ValueError(file + 'Not an Annotation file.')
            tmp.append(tmp2)

        data = pd.concat(tmp, ignore_index=True, sort=False)
        data.reset_index(inplace=True, drop=True)
        return data

    def __add__(self, other):
        """Concatenate data from several annotation objects."""
        assert type(other) is ecosound.core.annotation.Annotation, "Object type not \
            supported. Can only concatenate Annotation objects together."
        self._enforce_dtypes()
        other._enforce_dtypes()
        self.data = pd.concat([self.data, other.data],
                              ignore_index=True,
                              sort=False)
        return self

    def __repr__(self):
        """Return the type of object."""
        return (f'{self.__class__.__name__} object ('
                f'{len(self.data)})')

    def __str__(self):
        """Return string when used with print of str."""
        return f'{len(self.data)} annotation(s)'

    def __len__(self):
        """Return number of annotations."""
        return len(self.data)
