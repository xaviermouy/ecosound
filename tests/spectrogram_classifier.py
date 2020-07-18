# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy

"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import ecosound.core.tools
import time
import pickle
import copy

## Input paraneters ##########################################################

single_channel_file = r"../ecosound/resources/67674121.181018013806.wav"
#single_channel_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\audio_data\AMAR173.4.20190916T004248Z.wav"
#single_channel_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\DFO_snake-island_rca-in_20181017\audio_data\67674121.181017060806.wav'
#single_channel_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\DFO_snake-island_rca-in_20181017\audio_data\noise\67674121.181121150813.wav'

# Spectrogram parameters
frame = 0.0625 #3000
nfft = 0.0853 # 4096
step = 0.01 # 5
fmin = 0
fmax = 1000
window_type = 'hann'

# start and stop time of wavfile to analyze
t1 = 0#141#24
t2 = 60#167#40
## ###########################################################################
tic = time.perf_counter()

# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
spectro1 = copy.deepcopy(spectro)

# Denoise
spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(2048,1000), inplace=True)
spectro2 = copy.deepcopy(spectro)

# Detector
detector = DetectorFactory('BlobDetector', kernel_duration=0.1, kernel_bandwidth=300, threshold=10, duration_min=0.05, bandwidth_min=40)
detections = detector.run(spectro, use_dask=True, dask_chunks=(2048,1000), debug=False)

# Measurements
spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.001, resolution_freq=1, interp='linear')
measurements = spectro_features.compute(spectro,
                                        detections,
                                        debug=False,
                                        verbose=False,
                                        use_dask=False)


# Classification
model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\RF50_model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))
features = loaded_model['features']
model = loaded_model['model']
data = measurements.data
X = data[features]
pred_class = model.predict(X)
pred_prob = model.predict_proba(X)
#pred_prob = pred_prob[:,1]
pred_prob = pred_prob[range(0,len(pred_class)),pred_class]

#relabel
#pred_class2 = l = [None] * len(pred_class)
pred_class2 = pred_class
pred_class2 = ['Noise' if i==0 else 'Fish' for i in pred_class2]
# update measuremnets
data['label_class'] = pred_class2
data['confidence'] = pred_prob

data_fish = data[data['label_class']=='Fish']
data_noise = data[data['label_class']=='Noise']

classif_fish = copy.deepcopy(measurements)
classif_fish.data = data_fish

classif_noise = copy.deepcopy(measurements)
classif_noise.data = data_noise



# Plot
graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
graph.add_data(sound)
#graph.add_annotation(classif_fish, panel=0,color='red', label='Fish', tag=True)
graph.add_data(spectro1)
graph.add_data(spectro2)
graph.add_data(spectro2)
graph.add_data(spectro2)
graph.add_annotation(detections, panel=3,color='black', label='Detections')
graph.add_annotation(classif_fish, panel=4,color='red', label='Fish', tag=True)
graph.add_annotation(classif_noise, panel=4,color='blue', label='Noise',tag=True)
graph.colormap = 'binary'
#graph.colormap = 'jet'
graph.show()


#measurements.to_netcdf('test.nc')



toc = time.perf_counter()
print(f"Executed in {toc - tic:0.4f} seconds")
# #########################################
