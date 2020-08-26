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
import pickle
import platform
import numpy as np
# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster,processes=False)

    
def run_detector(infile, outdir, classif_model=None, deployment_file=None):
    ## Input paraneters ##########################################################   
    
    
    # Spectrogram parameters
    frame = 0.0625 #3000
    nfft = 0.0853 # 4096
    step = 0.01 # 5
    fmin = 0
    fmax = 1000
    window_type = 'hann'

    # start and stop time of wavfile to analyze
    #t1 = 0 # 24
    #t2 = 60 # 40
    ## ###########################################################################
    outfile = os.path.join(outdir, os.path.split(infile)[1]+'.nc')
    
    if os.path.exists(outfile) is False:
        # load audio data
        sound = Sound(infile)
        #sound.read(channel=0, chunk=[t1, t2], unit='sec')
        sound.read(channel=0, unit='sec')
        # Calculates  spectrogram
        spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
        spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)
        # Crop unused frequencies
        spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
        # Denoise
        spectro.denoise('median_equalizer',
                        window_duration=3,
                        use_dask=True,
                        dask_chunks= 'auto',#(87,10000),
                        inplace=True)
        # Detector
        file_timestamp = ecosound.core.tools.filename_to_datetime(infile)[0]
        detector = DetectorFactory('BlobDetector',
                                   kernel_duration=0.1,
                                   kernel_bandwidth=300,
                                   threshold=10,
                                   duration_min=0.05,
                                   bandwidth_min=40)
        detections = detector.run(spectro,
                                  start_time=file_timestamp,
                                  use_dask=True,
                                  dask_chunks='auto',
                                  debug=False)
        # Maasurements
        spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.001, resolution_freq=0.1, interp='linear')
        measurements = spectro_features.compute(spectro,
                                                detections,
                                                debug=False,
                                                verbose=False,
                                                use_dask=True)
        
        # Add metadata
        if deployment_file:
            measurements.insert_metadata(deployment_file)
        
        # Add file informations
        file_name = os.path.splitext(os.path.basename(infile))[0]
        file_dir = os.path.dirname(infile)
        file_ext = os.path.splitext(infile)[1]
        measurements.insert_values(operator_name=platform.uname().node,
                                   audio_file_name=file_name,
                                   audio_file_dir=file_dir,
                                   audio_file_extension=file_ext,
                                   audio_file_start_date= ecosound.core.tools.filename_to_datetime(infile)[0]
                                   )
    
    
    
        # Classification
        if classif_model:
            features = classif_model['features']
            model = classif_model['model']
            Norm_mean = classif_model['normalization_mean']
            Norm_std = classif_model['normalization_std']
            classes_encoder = classif_model['classes']
            # data dataframe
            data = measurements.data
            n1=len(data)
            # drop observations/rows with NaNs
            data = data.replace([np.inf, -np.inf], np.nan)
            data.dropna(subset=features, axis=0, how='any', thresh=None, inplace=True)
            n2=len(data)
            print('Deleted observations (due to NaNs): ' + str(n1-n2))
            # Classification - predictions
            X = data[features]
            X = (X-Norm_mean)/Norm_std
            pred_class = model.predict(X)
            pred_prob = model.predict_proba(X)
            pred_prob = pred_prob[range(0,len(pred_class)),pred_class]
            # Relabel
            for index, row in classes_encoder.iterrows():
                pred_class = [row['label'] if i==row['ID'] else i for i in pred_class]
            # update measurements
            data['label_class'] = pred_class
            data['confidence'] = pred_prob
            measurements.data = data

        # save result as NetCDF file
        measurements.to_netcdf(outfile)
    else:
        print('Recording already processed.')


def main():
    indir = r'D:\RCA_IN\April_July2019\1342218252'
    outdir=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\April_July2019_1342218252'    
    deployment_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\April_July2019_1342218252\deployment_info.csv'
    classif_model_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\bkp\RF300_model.sav'
    ext='.wav' 
    
    # load classif model
    classif_model = pickle.load(open(classif_model_file, 'rb'))

    # list files to process
    files = ecosound.core.tools.list_files(indir,
                                            ext,
                                            recursive=False,
                                            case_sensitive=True)
    # process each file
    for idx,  file in enumerate(files):
        print(str(idx)+r'/'+str(len(files))+': '+ file)
        tic = time.perf_counter()
        run_detector(file, outdir, classif_model=classif_model, deployment_file=deployment_file)
        toc = time.perf_counter()
        print(f"Executed in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    main()