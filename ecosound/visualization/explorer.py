import hvplot.pandas
import panel as pn
import holoviews as hv
import numpy as np
import sounddevice as sd
import pandas as pd
import os
import sys
import copy
sys.path.append(r"C:\Users\xavier.mouy\Documents\Workspace\GitHub\ecosound")
from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.visualization.grapher_builder import GrapherFactory
pn.extension()

# load dataset
data_file=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN.nc'
dataset = Measurement()
dataset.from_netcdf(data_file)
data_all = dataset.data
data = copy.deepcopy(data_all)
#data = data.sample(n = 10000)

# Widgets
inputs = list(data.columns)
X_variable  = pn.widgets.Select(name='X axis',
                                value='frequency_min',
                                options=inputs)
Y_variable  = pn.widgets.Select(name='Y axis',
                                value='frequency_max',
                                options=inputs)
Color  = pn.widgets.Select(name='Color',
                           value='label_class',
                           options=inputs)
alpha_slider = pn.widgets.FloatSlider(name='Point transparency',
                                      start=0, end=1,
                                      step=0.1,
                                      value=0.2,
                                     )
size_slider = pn.widgets.IntSlider(name='Point size',
                                   start=1,
                                   end=40,
                                   step=1,
                                   value=6,
                                  )

sampling_slider = pn.widgets.FloatSlider(name='% data loaded',
                                      start=0, end=100,
                                      step=10,
                                      value=20,
                                        )

markdown = pn.pane.Markdown("<b>Markdown display</b>", width=400)

sound_checkbox = pn.widgets.Checkbox(name='Automatically play selected sound', value=False)

delete_button = pn.widgets.Button(name='Delete selection', button_type='warning',width=150)
save_button = pn.widgets.Button(name='Save', button_type='default',width=150)


# Color value multi selector
color_values_multi_select = pn.widgets.MultiSelect(name='Selected color values', value= list(data[Color.value].unique()), options=list(data[Color.value].unique()), size=8)
def callback_color_selection(target, event):
    target.options = list(data_all[event.new].unique())#    target.value = list(data_all[event.new].unique())  # selects everything by default.
Color.link(color_values_multi_select, callbacks={'value': callback_color_selection})

# Stream
tap = hv.streams.Selection1D(index=[data.index.min()])

# multi selector for user Selection 
selection_multi_select = pn.widgets.MultiSelect(name='Selected points', value= [data.index.min()], options=[data.index.min()], size=8, jslink=True)
def callback_points_selection(*events):
    for event in events:
        if event.name == 'index':
            if event.new:
                print(event.new)
                #selection_multi_select.options = event.new
                selection_multi_select.options = list(data.iloc[event.new].index)
                #list(data.iloc[[6670, 11469]].index)
                if selection_multi_select.options:
                    selection_multi_select.value = [selection_multi_select.options[0]]
                    selection_multi_select.name = 'Selected points (' + str(len(selection_multi_select.options)) + ')'
watcher = tap.param.watch(callback_points_selection, ['index'], onlychanged=False)

# Subsample
def subsample(data1, sampling_slider):
    n_points = int(len(data1)*sampling_slider)
    return data1.sample(n = n_points)

# Filter color values based on multi selector
def filter_color_values(data1,Color, color_values_multi_select):
    return data1[data1[Color].isin(color_values_multi_select)]


def delete_selection(event):
    # delete data
    #global data, data_all
    #data = data.drop(selection_multi_select.value, axis=0)
    #data_all = data_all.drop(selection_multi_select.value, axis=0)
    # update selection list
    selec_list = selection_multi_select.options
    [selec_list.remove(x) for x in selection_multi_select.value]
    #print(selec_list)
    if len(selec_list) == 0: # if list is now empty
        selec_list = data.index[0] # select first row of data
    selection_multi_select.name = 'Selected points (' + str(len(selection_multi_select.options)) + ')'
    selection_multi_select.value = [selec_list[0]] #might need to do better here later (i.e. not reseting to index = 0)
    selection_multi_select.options = selec_list
    
    # update scatter plot by resetting point alpha
    #alpha_slider.value = 1

delete_button.on_click(delete_selection)

# Scatter plot
@pn.depends(X_variable, Y_variable, Color, alpha_slider, size_slider,sampling_slider, color_values_multi_select)
def scatterplot(X_variable, Y_variable, Color, alpha_slider, size_slider,sampling_slider, color_values_multi_select):
    global data
    # subsampling (from slider)
    data = subsample(data_all, sampling_slider/100)
    # filter based on color values selected
    if len(color_values_multi_select)>0:
        data = filter_color_values(data, Color, color_values_multi_select)
    # Scatter plot
    scatter_plot = data.hvplot.scatter(x=X_variable,
                                       y=Y_variable,
                                       c=Color,
                                       s=size_slider,
                                       alpha=alpha_slider,
                                       title= str(len(data)) + '/' + str(len(data_all)) + ' points',
                                       #datashade=True,
                                       tools=['tap', 'box_select', 'lasso_select'],
                                       #active_tools=['wheel_zoom'],
                                       hover_cols = ['index','uuid'],
                                       responsive=True,
                                       nonselection_alpha = 0.1,
                                       selection_alpha = 1,
                                       min_width=800,
                                       min_height=450,
                                      )
    tap.source = scatter_plot   
    #scatter_plot.opts(click_policy="hide")
    return scatter_plot 

# Measurement table
@pn.depends(index=selection_multi_select)
def table(index):  
    if index:
        return pn.pane.DataFrame(data.loc[index[0]])

    
# Spectrogram
@pn.depends(index=selection_multi_select, play_sound=sound_checkbox)
def spectrogram_plot(index, play_sound):
    if index:
        frame = 0.0625 #3000
        nfft = 0.0853 # 4096
        step = 0.01 # 5
        fmin = 0
        fmax = 1000
        window_type = 'hann'
        time_buffer = 1
        palet = 'jet' # 'binary'

        #data_selection = data.iloc[index[0]]
        #index_selection = data.index[index[0]]
        data_selection = data.loc[index[0]]
        index_selection = index[0]

        wavfilename = os.path.join(data_selection.audio_file_dir, data_selection.audio_file_name + data_selection.audio_file_extension)
        t1 = data_selection.time_min_offset - time_buffer
        t2 = data_selection.time_max_offset + time_buffer
        # load audio data
        sound = Sound(wavfilename)
        sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)
        # Calculates  spectrogram
        spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
        spectro.compute(sound, dB=True, use_dask=False, dask_chunks=40)
        # Define annotation box
        annot = Annotation()
        annot.data = annot.data.append({'time_min_offset': time_buffer,
                                        'time_max_offset': time_buffer + data_selection.duration,
                                        'frequency_min': data_selection.frequency_min,
                                        'frequency_max': data_selection.frequency_max,
                                        'duration':data_selection.duration,
                                       },
                                       ignore_index=True)

        # Plot
        graph = GrapherFactory('SoundPlotter', title=str(index_selection) + ': ' +data_selection.label_class + ' - ' +data_selection.label_subclass, frequency_max=fmax)
        graph.add_data(spectro)
        graph.add_annotation(annot, panel=0, color='black', label='Detections')

        #graph.colormap = 'binary'
        graph.colormap = palet
        fig, ax = graph.show(display=False)

        # play sound
        if play_sound:
            sd.play(sound.waveform/max(sound.waveform), sound.waveform_sampling_frequency)
        
        return pn.pane.Matplotlib(fig)


# Dashboard
#gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=800)
tabs = pn.Tabs(('Axes',pn.Column(X_variable, Y_variable, Color, color_values_multi_select)))
tabs.append(('Style',pn.Column(alpha_slider, size_slider)))
widgets = pn.Column(sampling_slider, tabs)
tabs2 = pn.Tabs(('Spectrogram',spectrogram_plot))
tabs2.append(('Measurements',table))

dashboard = pn.Column(pn.Row(widgets, scatterplot),pn.Row(tabs2, pn.Column(selection_multi_select,sound_checkbox, pn.Row(delete_button,save_button))))
dashboard.show(threaded=True)