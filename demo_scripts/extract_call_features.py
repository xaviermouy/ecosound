'''
This script measures a set of features from annotations stored Raven table files.
It loads the annotations, calculates the spectrogram, applies a median filter (otional)
measures the set of features described in Mouy et al. 2024 (https://doi.org/10.3389/frsen.2024.1439995)
for each annotation and saves the measurements as csv, nc and raven table file

To run this script:
- create python 3.9 environment
- install the required libraries with: pip install ecosound, PyQt5

Xavier Mouy - 26 Jan 2025
'''

from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation
from ecosound.measurements.measurer_builder import MeasurerFactory
from ecosound.core.tools import filename_to_datetime, list_files
from ecosound.visualization.grapher_builder import GrapherFactory
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
matplotlib.use('Agg')

## Input parameters ################################################################
annot_dir = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Sound_files_RF_feature_extraction' # folder where the raven tables are
audio_dir = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Sound_files_RF_feature_extraction' # folder where the audio files are
output_dir = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Sound_files_RF_feature_extraction\measurements' # folder where the results/measurements will be saved

# Spectrogram parameters (in samples)
frame = 3000
nfft = 4096
step = 500
window_type = 'hann'
fmin = 0 # freq min (Hz)
fmax = 1000 # freq max (Hz)
dB = False
# Denoising:
remove_background = True # True or False => subtract average spectrum before doing measurements

# for loading just a section of the file (do not use - just for debugging)
t1=None
t2=None
## ###################################################################################

# list raven tabel files
annot_files = list_files(annot_dir,'.chan0.Table.1.selections.FS.txt')

#annot_file = annot_files[0]
for annot_file in annot_files:
    print(annot_file)

    # load manual annotations
    annot = Annotation()
    annot.from_raven(annot_file)

    # load audio data
    audio_file =  os.path.join(audio_dir, annot.data.iloc[0].audio_file_name + annot.data.iloc[0].audio_file_extension)
    sound = Sound(audio_file)
    if (t1 is None) or (t2 is None):
        sound.read(channel=0, unit='sec', detrend=True)
    else:
        sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

    # Fill in missing fields in annotations
    annot.insert_values(
        audio_sampling_frequency=sound.waveform_sampling_frequency,
        audio_channel=sound.channel_selected,
        audio_bit_depth=24,
    )

    # Calculates  spectrogram
    spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
    spectro.compute(sound, dB=dB, use_dask=True, dask_chunks=40)

    # Crop unused frequencies
    spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

    # make spectrogram values positive
    spectro._spectrogram = spectro.spectrogram + abs(spectro.spectrogram.min())

    # denoising = subtract mean spectrum of the recording
    if remove_background:
        bkg_spectrum = spectro.spectrogram.mean(axis=1)
        spectro._spectrogram = spectro.spectrogram - bkg_spectrum[:, np.newaxis]

    # apply frequency independent amplitude offset to avoid negative values
    spectro._spectrogram = spectro.spectrogram + abs(spectro._spectrogram.min())

    # # Plot - uncomment to look at denoising spectrogram on a single file
    # graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=fmax)
    # graph.add_data(sound)
    # graph.add_annotation(annot, panel=0, color='red')
    # graph.add_data(spectro)
    # graph.add_annotation(annot, panel=1)
    # graph.colormap = 'jet'
    # graph.show()
    # plt.show()

    # # for debugging only
    # annot.data =  annot.data.iloc[6:7] #1385.85
    # annot.data.reset_index(inplace=True)

    # Measurements
    spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.01, resolution_freq=0.1, interp='linear')
    measurements = spectro_features.compute(spectro,
                                            annot,
                                            debug=False,
                                            verbose=True,
                                            use_dask=False)

    # Save measurements as .nc, .csv, and raven table (.txt)
    measurements.to_netcdf(os.path.join(output_dir,annot.data.iloc[0].audio_file_name + annot.data.iloc[0].audio_file_extension + '.nc'))
    measurements.to_csv(os.path.join(output_dir,annot.data.iloc[0].audio_file_name + annot.data.iloc[0].audio_file_extension + '.csv'))
    measurements.to_raven(os.path.join(output_dir))

print('Done!')
