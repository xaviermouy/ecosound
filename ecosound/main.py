from core.audiotools import Sound, Filter

single_channel_file = r"./resources/AMAR173.4.20190916T061248Z.wav"
#multi_channel_file = r"./resources/671404070.190722162836.wav.wav"


# load part of the file and plot
print('------------------------------')
#sig = Sound() # should return error
sig = Sound(single_channel_file)
print(len(sig))
sig.read(channel=0, chunk=[0, 12000])
print('------------------------------')
print(len(sig))
print('start sample: ', sig.waveform_start_sample)
print('stop sample: ', sig.waveform_stop_sample)
print('duration:: ', sig.waveform_duration_sample)
print(len(sig))

# extract a sinppet from the data
sig2 = sig.select_snippet([100,1000])
sig2.plot_waveform(newfig=True)
print('------------------------------')
print('start sample: ', sig2.waveform_start_sample)
print('stop sample: ', sig2.waveform_stop_sample)
print('duration:: ', sig2.waveform_duration_sample)
print(len(sig2))

# try filter
filter_type = ['bandpass', 'lowpass', 'highpass']
cutoff_frequencies = [100, 1000]
sig2.filter(filter_type[0], cutoff_frequencies, order=4)
# try agfain -> should retrun error
#sig2.filter(filter_type[0], cutoff_frequencies, order=4)
sig2.plot_waveform(newfig=True)
print('------------------------------')
print('start sample: ', sig2.waveform_start_sample)
print('stop sample: ', sig2.waveform_stop_sample)
print('duration: ', sig2.waveform_duration_sample)
print(len(sig2))

# re-adjust seletec waveform based on energy
energy_percentage = 80
sig2.tighten_waveform_window(energy_percentage)
sig2.plot_waveform(newfig=True)
print('------------------------------')
print('start sample: ', sig2.waveform_start_sample)
print('start sample: ', sig2.waveform_stop_sample)
print('duration:: ', sig2.waveform_duration_sample)
print(len(sig2))
