# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy

"""
import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound")  # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
from ecosound.classification.classification import Classifier

import ecosound.core.tools
import time
import pickle
import copy

## Input paraneters ##########################################################

single_channel_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Herring_DFO\5042.200306203002.wav"
#single_channel_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\datasets\DFO_snake-island_rca-in_20190410\noise\1342218252.190430230159.wav"
#single_channel_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\problematic_files\1342218252.190415230156.wav"
#model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\problematic_files\RF50_model_20201208T223420.sav'
#model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\problematic_files\XGBoost_model_20201209T132812.sav'
model_filename = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\problematic_files\RF50_model_20201209T134646.sav'

# Spectrogram parameters
frame = 0.02 #0.0625
nfft = 0.02 # 4096
step = 0.005 # 5
fmin = 500
fmax = 5000
window_type = 'hann'


# start and stop time of wavfile to analyze
t1 = 0#141#24
t2 = 300#167#40
## ###########################################################################
tic = time.perf_counter()

# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
spectro.compute(sound, dB=True, use_dask=True, dask_chunks=100) #40

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
spectro1 = copy.deepcopy(spectro)

# Denoise
#spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(2048,1000), inplace=True)
spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(50,50000), inplace=True)
spectro2 = copy.deepcopy(spectro)

# Detector
detector = DetectorFactory('BlobDetector', kernel_duration=0.1, kernel_bandwidth=500, threshold=10, duration_min=0.05, bandwidth_min=40)
#detections = detector.run(spectro, use_dask=True, dask_chunks=(2048,1000), debug=False)
detections = detector.run(spectro, use_dask=True, dask_chunks=(4096,50000), debug=False)

# Measurements
spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.001, resolution_freq=0.1, interp='linear')
measurements = spectro_features.compute(spectro,
                                        detections,
                                        debug=False,
                                        verbose=False,
                                        use_dask=True)


# detections.insert_values(audio_file_name='1342218252.190430230159')
# detections.insert_values(audio_file_dir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\datasets\DFO_snake-island_rca-in_20190410\noise')
# detections.insert_values(audio_file_extension='.wav')
# detections.insert_values(label_class='NN')
# detections.to_pamlab(r"C:\Users\xavier.mouy\Documents\PhD\Projects\Detector\datasets\DFO_snake-island_rca-in_20190410\noise\annotations", outfile='1342218252.190430230159.wav annotations.log', single_file=True)


# # Classification
# loaded_model = pickle.load(open(model_filename, 'rb'))
# features = loaded_model['features']
# model = loaded_model['model']
# Norm_mean = loaded_model['normalization_mean']
# Norm_std = loaded_model['normalization_std']
# classes_encoder = loaded_model['classes']

# data = measurements.data
# X = data[features]
# X = data[features]
# X = (X-Norm_mean)/Norm_std
# pred_class =list(model.predict(X))
# pred_prob = model.predict_proba(X)
# #pred_prob = pred_prob[:,1]
# pred_prob = pred_prob[range(0,len(pred_class)),pred_class]
# #relabel
# for index, row in classes_encoder.iterrows():
#     pred_class = [row['label'] if i==row['ID'] else i for i in pred_class]
# # update measuremnets
# data['label_class'] = pred_class
# data['confidence'] = pred_prob
# data_fish = data[data['label_class']=='FS']
# data_noise = data[data['label_class']=='NN']
# classif_fish = copy.deepcopy(measurements)
# classif_fish.data = data_fish
# classif_noise = copy.deepcopy(measurements)
# classif_noise.data = data_noise



# # Plot
graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=5000)
# graph.add_data(sound)
# #graph.add_annotation(classif_fish, panel=0,color='red', label='Fish', tag=True)
# graph.add_data(spectro1)
# graph.add_data(spectro2)
graph.add_data(spectro2)
# graph.add_data(spectro2)
graph.add_annotation(detections, panel=0,color='red', label='Detections')
# graph.add_annotation(classif_fish, panel=4,color='red', label='Fish', tag=True)
# graph.add_annotation(classif_noise, panel=4,color='blue', label='Noise',tag=True)
# graph.colormap = 'binary'
graph.colormap = 'jet'
graph.show()



# classifier = Classifier()
# classifier.load_model(model_filename)
# classif2 = classifier.classify(measurements)

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound)
# #graph.add_annotation(classif_fish, panel=0,color='red', label='Fish', tag=True)
# graph.add_data(spectro1)
# graph.add_data(spectro2)
# graph.add_data(spectro2)
# graph.add_data(spectro2)
# graph.add_annotation(detections, panel=3,color='black', label='Detections')
# graph.add_annotation(classif2, panel=4,color='red', tag=True)
# #graph.add_annotation(classif2, panel=4,color='blue', label='Noise',tag=True)
# graph.colormap = 'binary'
# #graph.colormap = 'jet'
# graph.show()


# toc = time.perf_counter()
# print(f"Executed in {toc - tic:0.4f} seconds")
# # #########################################
