# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:21:40 2020

@author: xavier.mouy
"""
import json
import re
from datetime import datetime
import ecosound.core.decorators
import numpy as np
from scipy import interpolate
from scipy.signal import argrelmax
import os, sys
from numba import njit
import pkg_resources
import yaml


def read_json(file):
    """
    Load a JSON file as a dictionary.

    Parameters
    ----------
    file : str
        Path of the JSON file to load.

    Returns
    -------
    data : dict
        Contents of the JSON file as a Python dictionary.

    """
    with open(file, "r") as read_file:
        data = json.load(read_file)
    return data


def read_yaml(file):
    """
    Load config file.

    Parameters
    ----------
    file : str
        Path of the yaml file with all the parameters.

    Returns
    -------
    config : dict
        Parsed parameters.

    """
    # Loads config  files
    yaml_file = open(file)
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


@ecosound.core.decorators.listinput
def filename_to_datetime(files):
    """
    Extract datetime from audio filenames.

    Parses timestamps embedded in audio file names or paths using a set of
    known timestamp patterns defined in ``timestamp_formats.json``. Accepts a
    single filename string or a list of filename strings.

    Parameters
    ----------
    files : str or list of str
        Filename(s) or full file path(s) containing an embedded timestamp.

    Raises
    ------
    ValueError
        If the timestamp format in a filename is not recognized.

    Returns
    -------
    timestamps : list of datetime
        List of :class:`datetime.datetime` objects parsed from each filename.

    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    patterns = read_json(os.path.join(current_dir, r"timestamp_formats.json"))

    # stream = pkg_resources.resource_stream(__name__, 'core/timestamp_formats.json')
    # patterns = read_json(os.path.join(stream)

    regex_string = "|".join(
        [pattern["string_pattern"] for pattern in patterns]
    )
    time_formats = [pattern["time_format"] for pattern in patterns]
    timestamps = [None] * len(files)
    p = re.compile(regex_string)
    for idx, file in enumerate(files):
        datestr = p.search(file)
        for time_format in time_formats:
            ok_flag = False
            try:
                timestamps[idx] = datetime.strptime(datestr[0], time_format)
                ok_flag = True
            except:
                ok_flag = False
            if ok_flag is True:
                break
        if ok_flag is False:
            raise ValueError("Time format not recognized:" + file)
    return timestamps


# @njit
def normalize_vector(vec):
    """
    Normalize the amplitude of a 1-D vector.

    Removes the mean (DC offset) and scales the vector so that its maximum
    absolute value is 1.

    Parameters
    ----------
    vec : numpy.ndarray
        1-D array to normalize.

    Returns
    -------
    normVec : numpy.ndarray
        Mean-subtracted and amplitude-normalized version of ``vec``.

    """
    # vec = vec+abs(min(vec))
    # normVec = vec/max(vec)
    # normVec = (normVec - 0.5)*2
    vec = vec - np.mean(vec)
    normVec = vec / max(vec)
    return normVec


# @njit
def tighten_signal_limits(signal, energy_percentage):
    """
    Tighten signal limits based on cumulative energy.

    Redefines the start and stop sample indices to retain the central
    ``energy_percentage`` percent of the signal's cumulative energy. Equal
    amounts of energy are trimmed from both ends.

    Parameters
    ----------
    signal : numpy.ndarray
        1-D waveform array.
    energy_percentage : float
        Percentage of the total signal energy to retain (0–100). Higher values
        keep more of the signal; lower values crop more aggressively.

    Returns
    -------
    chunk : list of int
        Two-element list ``[start_sample, stop_sample]`` with the new sample
        indices bounding the requested energy percentage.

    """
    cumul_energy = np.cumsum(np.square(signal))
    cumul_energy = cumul_energy / max(cumul_energy)
    percentage_begining = (1 - (energy_percentage / 100)) / 2
    percentage_end = 1 - percentage_begining
    chunk = [
        np.nonzero(cumul_energy > percentage_begining)[0][0],
        np.nonzero(cumul_energy > percentage_end)[0][0],
    ]
    return chunk


def tighten_signal_limits_peak(signal, percentage_max_energy):
    """
    Tighten signal limits based on peak energy samples.

    Redefines the start and stop sample indices by identifying the smallest
    contiguous window that contains the highest-energy samples, representing
    ``percentage_max_energy`` percent of the total signal energy. Samples are
    ranked by energy (highest first) rather than by position. Smaller values of
    ``percentage_max_energy`` produce a tighter (shorter) window.

    Parameters
    ----------
    signal : numpy.ndarray
        1-D waveform array.
    percentage_max_energy : float
        Percentage of total signal energy to capture (0–100). Smaller values
        yield tighter windows around the most energetic part of the signal.

    Returns
    -------
    chunk : list of int
        Two-element list ``[start_sample, stop_sample]`` spanning the window
        that contains the requested percentage of peak energy.

    """
    squared_signal = np.square(signal)
    norm_factor = sum(squared_signal)
    squared_signal_normalized = squared_signal / norm_factor
    sort_idx = np.argsort(-squared_signal_normalized)
    sort_val = squared_signal_normalized[sort_idx]
    sort_val_cum = np.cumsum(sort_val)
    id_limit = np.where(sort_val_cum > (percentage_max_energy / 100))
    id_limit = id_limit[0][0]
    min_idx_limit = np.min(sort_idx[0:id_limit])
    max_idx_limit = np.max(sort_idx[0:id_limit])
    chunk = [min_idx_limit, max_idx_limit]

    return chunk


def resample_1D_array(x, y, resolution, kind="linear"):
    """
    Resample a 1-D array to a new uniform resolution via interpolation.

    Interpolates the values of ``y`` over a new ``x`` axis with uniform
    spacing defined by ``resolution``.

    Parameters
    ----------
    x : numpy.ndarray
        1-D array of original x-axis coordinates (must be monotonically
        increasing).
    y : numpy.ndarray
        1-D array of values corresponding to ``x``.
    resolution : float
        Desired spacing between consecutive points on the new x-axis. Must
        use the same units as ``x``.
    kind : str, optional
        Interpolation method passed to :func:`scipy.interpolate.interp1d`.
        Common options are ``'linear'``, ``'nearest'``, ``'cubic'``.
        The default is ``'linear'``.

    Returns
    -------
    xnew : numpy.ndarray
        New uniformly-spaced x-axis array from ``x[0]`` to ``x[-1]`` with
        step ``resolution``.
    ynew : numpy.ndarray
        Interpolated y values at each point in ``xnew``.

    """
    f = interpolate.interp1d(x, y, kind=kind, fill_value="extrapolate")
    xnew = np.arange(x[0], x[-1] + resolution, resolution)
    ynew = f(xnew)
    return xnew, ynew


@njit
def entropy(array_1d, apply_square=False):
    """
    Calculate the aggregate Shannon entropy of a 1-D array.

    Computes the Shannon entropy as defined in the Raven bioacoustics software
    manual. Each element is treated as a probability mass after normalizing by
    the total sum of the array. Optionally squares each element before
    calculation to weight higher-amplitude values more strongly.

    Parameters
    ----------
    array_1d : numpy.ndarray
        1-D array of non-negative values (e.g., a power spectrum).
    apply_square : bool, optional
        If ``True``, squares each element of ``array_1d`` before computing
        the entropy. The default is ``False``.

    Returns
    -------
    H : float
        Shannon entropy of the (optionally squared) normalized array.
        Lower values indicate a more peaked (tonal) distribution; higher
        values indicate a flatter (broadband) distribution.

    """
    if apply_square:
        array_1d = np.square(array_1d)
    values_sum = np.sum(array_1d)
    H = 0
    for value in array_1d:
        ratio = value / values_sum
        if ratio <= 0:
            H += 0
        else:
            H += ratio * np.log2(ratio)
    return H


@njit
def derivative_1d(array, order=1):
    """
    Compute the discrete derivative of a 1-D array.

    Calculates the finite difference (element i+1 minus element i) of the
    array. If ``order`` is greater than 1 the differencing is applied
    iteratively, so the output is shorter than the input by ``order`` elements.

    Parameters
    ----------
    array : numpy.ndarray
        1-D input array.
    order : int, optional
        Order of the derivative (number of successive differences). The
        default is 1.

    Returns
    -------
    array : numpy.ndarray
        Derivative array of length ``len(array) - order``.

    """
    for n in range(0, order, 1):
        array = np.subtract(array[1:], array[0:-1])
    return array


def list_files(indir, suffix, case_sensitive=True, recursive=False):
    """
    List files in folder whose name ends with a given suffix/extension.

    Parameters
    ----------
    indir : str
        Path of the folder to search.
    suffix : str
        Suffix of the filename.
    case_sensitive : bool, optional
        If set to True, search using case sensitive filenames. The default is
        True.
    recursive : bool, optional
        If set to True, search in parent folder but also in all its subfolders.
        The default is False.

    Returns
    -------
    files_list : list
        List of strings with full path of files found.

    """
    if os.path.isdir(indir):
        files_list = []
        if case_sensitive is False:
            suffix = suffix.lower()
        if recursive:  # scans subfolders recursively
            for root, dirs, files in os.walk(indir):
                for file in files:
                    if case_sensitive is False:
                        file = file.lower()
                    if file.endswith(suffix):
                        files_list.append(os.path.join(root, file))
                        # print(os.path.join(root, file))
        else:  # only scans parent folder
            for file in os.listdir(indir):
                if case_sensitive is False:
                    file = file.lower()
                if file.endswith(suffix):
                    files_list.append(os.path.join(indir, file))
                    # print(os.path.join(indir, file))
    else:
        raise Exception("The indir folder given does not exist.")
    return files_list


@njit
def find_peaks(array, troughs=False):
    """
    Find peaks or troughs in an 1-D array.

    Parameters
    ----------
    array : numpy array or list
        1-dimensional array.
    troughs : bool, optional
        If set to True, finds troughs instead of peaks in the input array.
        The default is False.

    Returns
    -------
    x : list
        Indices of peaks or troughs
    y : list
        Values of peaks or troughs

    """

    x = [
        0,
    ]
    y = [
        array[0],
    ]
    for k in range(1, len(array) - 1):
        if troughs:
            if (np.sign(array[k] - array[k - 1]) == -1) and (
                (np.sign(array[k] - array[k + 1])) == -1
            ):
                x.append(k)
                y.append(array[k])
        else:
            if (np.sign(array[k] - array[k - 1]) == 1) and (
                np.sign(array[k] - array[k + 1]) == 1
            ):
                x.append(k)
                y.append(array[k])
    return x, y


def envelope(array, interp="cubic"):
    """
    Compute the upper and lower amplitude envelopes of a 1-D signal.

    Detects local peaks and troughs in ``array`` and interpolates between
    them to produce smooth upper and lower envelope curves. Both envelopes
    are anchored to the first and last values of the array.

    Parameters
    ----------
    array : numpy.ndarray
        1-D signal array.
    interp : str, optional
        Interpolation method passed to :func:`scipy.interpolate.interp1d`.
        Common options are ``'linear'``, ``'nearest'``, ``'cubic'``.
        The default is ``'cubic'``.

    Returns
    -------
    env_high : numpy.ndarray
        Upper envelope of ``array``, same length as ``array``.
    env_low : numpy.ndarray
        Lower envelope of ``array``, same length as ``array``.

    """
    # initialize output arrays
    env_high = np.zeros(array.shape)
    env_low = np.zeros(array.shape)
    # Prepend the first value of (s) to the interpolating values. This forces
    # the model to use the same starting point for both the upper and lower
    # envelope models.
    u_x = [
        0,
    ]
    u_y = [
        array[0],
    ]
    l_x = [
        0,
    ]
    l_y = [
        array[0],
    ]
    # Detect peaks and troughs
    l_x, l_y = find_peaks(array, troughs=True)
    u_x, u_y = find_peaks(array, troughs=False)
    # Append the last value of (s) to the interpolating values. This forces the
    # model to use the same ending point for both the upper and lower envelope
    # models.
    u_x.append(len(array) - 1)
    u_y.append(array[-1])
    l_x.append(len(array) - 1)
    l_y.append(array[-1])

    # Interpolate between peaks/troughs
    u_p = interpolate.interp1d(
        u_x, u_y, kind=interp, bounds_error=False, fill_value=0.0
    )
    l_p = interpolate.interp1d(
        l_x, l_y, kind=interp, bounds_error=False, fill_value=0.0
    )
    for k in range(0, len(array)):
        env_high[k] = u_p(k)
        env_low[k] = l_p(k)
    return env_high, env_low
