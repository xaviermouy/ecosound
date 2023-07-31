# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:27:39 2020

@author: xavier.mouy
"""

from .measurer_builder import BaseClass
from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
from ecosound.core.audiotools import Sound
import numpy as np
import pandas as pd
from dask import delayed, compute, visualize
import os


class SNR(BaseClass):
    """ """

    measurer_parameters = ("noise_win_sec",)

    def __init__(self, *args, **kwargs):
        """
        Initialize the measurer.

        Parameters
        ----------
        *args : str
            Do not use. Only used by the MeasurerFactory.
        noise_win_sec : float, optional
            Duration of window to use on either side of the signal to estimate
            noise, in seconds.

        Returns
        -------
        None. Measurer object.

        """
        # Initialize all measurer parameters to None
        self.__dict__.update(
            dict(
                zip(
                    self.measurer_parameters,
                    [None] * len(self.measurer_parameters),
                )
            )
        )

        # Unpack kwargs as measurer parameters if provided on instantiation
        self.__dict__.update(**kwargs)

    @property
    def name(self):
        """Return name of the measurer."""
        measurer_name = "SNR"
        return measurer_name

    @property
    def version(self):
        """Return version of the measurer."""
        version = "0.1"
        return version

    def _prerun_check(self, annotations):
        """Run several verifications before the run."""
        # check that all required arguments are defined
        if True in [
            self.__dict__.get(keys) is None
            for keys in self.measurer_parameters
        ]:
            raise ValueError(
                "Not all measurer parameters have been defined."
                + " Required parameters: "
                + str(self.measurer_parameters)
            )
        # check that annotations is an Annotation class
        if not isinstance(annotations, Annotation):
            raise ValueError(
                "Input must be an ecosound Annotation object"
                + "(ecosound.core.annotation)."
            )

    def compute(self, annotations, debug=False, verbose=False, use_dask=False):
        """Compute signal-to-noise-ratio of annotations.

        Goes through each annotation and computes the SNR by estinating the
        power of the noise before and after the annotation. Measurements are
        performed on the band-pass filtered waveform.

        Parameters
        ----------
        annotations : ecosound Annotation object
            Annotations of the sounds to measure. Can be from manual analysis
            or from an automatic detector.
        use_dask : bool, optional
            If True, run the measurer in parallele using Dask. The default is
            False.
        debug : bool, optional
            Displays figures for each annotation with the spectrogram, spectral
            and time envelopes, and tables with all associated measurements.
            The default is False.
        verbose : bool, optional
            Prints in the console the annotation being processed. The default
            is False.

        Returns
        -------
        measurements : ecosound Measurement object
            Measurement object containing the measurements appended to the
            original annotation fields. Measurements are in the .data data
            frame. Metadata with mearurer name, version and measurements names
            are in the .metadata datafreame.

        """
        self._prerun_check(annotations)

        # init
        features = self._init_dataframe()
        features_name = list(features.columns)
        # loop through each annotation
        df_list = []
        for index, annot in annotations.data.iterrows():
            if verbose:
                print(
                    "processing annotation ",
                    index,
                    annot["time_min_offset"],
                    "-",
                    annot["time_max_offset"],
                )

            # feature for 1 annot
            if use_dask:
                df = delayed(self.compute_single_annot)(annot, debug)
            else:
                df = self.compute_single_annot(annot, debug)
            # stack features for each annotation
            df_list.append(df)
        if use_dask:
            features = delayed(pd.concat)(df_list, ignore_index=False)
            # features.visualize('measuremnets')
            features = features.compute()
        else:
            features = pd.concat(df_list, ignore_index=False)
        # merge with annotation fields
        annotations.data.set_index("uuid", inplace=True, drop=False)
        features.set_index("uuid", inplace=True, drop=True)
        meas = pd.concat([annotations.data, features], axis=1, join="inner")
        meas.reset_index(drop=True, inplace=True)

        params_dict = dict()
        for param in self.measurer_parameters:
            params_dict[param] = eval("self." + param)

        # create Measurement object
        measurements = Measurement(
            measurer_name=self.name,
            measurer_version=self.version,
            measurements_name=features_name,
            measurements_parameters=params_dict,
        )
        measurements.data = meas
        return measurements

    def _init_dataframe(self):
        tmp = pd.DataFrame(
            {
                "snr": [],
            }
        )
        return tmp

    def compute_single_annot(self, annot, debug):
        # load sound file properties
        sound = Sound(
            os.path.join(annot["audio_file_dir"], annot["audio_file_name"])
            + annot["audio_file_extension"]
        )

        # define duration of noise window
        if self.noise_win_sec == "auto":
            half_noise_win_dur = annot.duration / 2
        else:
            half_noise_win_dur = self.noise_win_sec / 2

        # define left noise window
        noise_left_start = annot["time_min_offset"] - half_noise_win_dur
        if noise_left_start < 0:
            noise_left_end = annot["time_min_offset"]
            noise_left_start = 0
        else:
            noise_left_end = annot["time_min_offset"]

        # define right noise window
        noise_right_start = annot["time_max_offset"]
        noise_right_end = noise_right_start + half_noise_win_dur
        if noise_right_end > sound.file_duration_sec:
            noise_right_end = sound.file_duration_sec

        # load sound data chunk
        try:
            sound.read(chunk=[noise_left_start, noise_right_end], unit="sec")
        except:
            print(annot)
            raise Exception("error with time boundaries")
        
        # bandpass filter
        try:
            sound.filter(
                "bandpass",
                cutoff_frequencies=[
                    annot["frequency_min"],
                    annot["frequency_max"],
                ],
                order=10,
                verbose=False,
            )
        except:
            print(annot)
            raise Exception("error with frequency filtering")
        sound.normalize()

        # calculate energies
        times_samp = np.round(
            np.dot(
                [
                    noise_left_start,
                    noise_left_end,
                    noise_right_start,
                    noise_right_end,
                ],
                sound.waveform_sampling_frequency,
            )
        )
        times_samp = times_samp - times_samp[0]
        noise_left = sound.waveform[int(times_samp[0]) : int(times_samp[1])]
        sig = sound.waveform[int(times_samp[1]) : int(times_samp[2])]
        noise_right = sound.waveform[int(times_samp[2]) : int(times_samp[3])]
        # noise_pw = (sum(noise_left**2) + sum(noise_right**2)) / (
        #     len(noise_left) + len(noise_right)
        # )
        # sig_pw = sum(sig**2) / len(sig)

        noise_rms = np.sqrt(
            (sum(noise_left**2) + sum(noise_right**2))
            / (len(noise_left) + len(noise_right))
        )
        sig_rms = np.sqrt(sum(sig**2) / len(sig))

        # noise_var = np.sqrt((sum(noise_left**2) + sum(noise_right**2)))

        # try:
        snr = 20 * np.log10(sig_rms / noise_rms)
        # snr = 10 * np.log10(sig_pw / noise_pw)
        # except:
        #    print("Error")
        #    snr = np.nan

        if debug:
            sound.plot(newfig=True, title=str(round(snr, 1)))

        # stack all features
        tmp = pd.DataFrame(
            {
                "uuid": [annot["uuid"]],
                "snr": [snr],
            }
        )
        return tmp
