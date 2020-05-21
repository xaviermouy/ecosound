# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:54:29 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import ecosound.core.tools
import time
import os

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster,processes=False)

    
def run_detector(infile, outdir):
    ## Input paraneters ##########################################################   
    
    
    # Spectrogram parameters
    frame = 3000
    nfft = 4096
    step = 500
    fmin = 0
    fmax = 1000
    window_type = 'hann'

    # start and stop time of wavfile to analyze
    #t1 = 0 # 24
    #t2 = 60 # 40
    ## ###########################################################################
    outfile = os.path.join(outdir,os.path.split(file)[1]+'.nc')

    # load audio data
    sound = Sound(infile)
    #sound.read(channel=0, chunk=[t1, t2], unit='sec')
    sound.read(channel=0, unit='sec')
    # Calculates  spectrogram
    spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
    spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)
    # Crop unused frequencies
    spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
    # Denoise
    spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(2048,1000), inplace=True)
    # Detector
    file_timestamp = ecosound.core.tools.filename_to_datetime(infile)[0]
    detector = DetectorFactory('BlobDetector', kernel_duration=0.1, kernel_bandwidth=300, threshold=10, duration_min=0.05, bandwidth_min=40)
    detections = detector.run(spectro, start_time=file_timestamp, use_dask=True, dask_chunks=(2048,1000), debug=False)
    # Maasurements
    spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.001, resolution_freq=0.1, interp='linear')
    measurements = spectro_features.compute(spectro,
                                            detections,
                                            debug=False,
                                            verbose=False,
                                            use_dask=False)
    measurements.to_netcdf(outfile)



indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets'
#outdir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\UVIC_hornby-island_2019'
outdir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset'
ext='.wav'

files = ecosound.core.tools.list_files(indir,
                                        ext,
                                        recursive=True,
                                        case_sensitive=True)

for idx,  file in enumerate(files):
    print(str(idx)+r'/'+str(len(files))+': '+ file)
    try:
        tic = time.perf_counter()
        run_detector(file, outdir)
        toc = time.perf_counter()
    except:
        print('ERROR HERE --------------------------------------')
            
    print(f"Executed in {toc - tic:0.4f} seconds")

