import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings, os
warnings.filterwarnings('ignore')

from ecosound.core.audiotools import Sound, Filter, upsample
from ecosound.core.spectrogram import Spectrogram
import scipy.signal as spsig

DATA = r'C:/Users/xavier.mouy/Documents/GitHub/ecosound/data/wav_files'
FIG  = r'C:/Users/xavier.mouy/Documents/GitHub/ecosound/docs/source/tutorials/figures'
WAV  = DATA + '/AMAR173.4.20190916T061248Z.wav'

# === SOUND: __init__ ===
print('='*62)
print('SOUND: __init__ - file attributes')
print('='*62)
sound = Sound(WAV)
print('file_dir              :', sound.file_dir)
print('file_name             :', sound.file_name)
print('file_extension        :', sound.file_extension)
print('file_duration_sec     :', round(sound.file_duration_sec, 3), 's')
print('file_duration_sample  :', sound.file_duration_sample, 'samples')
print('file_sampling_frequency:', sound.file_sampling_frequency, 'Hz')
print('channels              :', sound.channels)

# === SOUND: read whole file ===
print()
print('='*62)
print('SOUND: read - whole file')
print('='*62)
sound.read(channel=0, chunk=[], unit='sec', detrend=True)
print('waveform shape             :', sound.waveform.shape)
print('waveform_sampling_frequency:', sound.waveform_sampling_frequency, 'Hz')
print('waveform_start_sample      :', sound.waveform_start_sample)
print('waveform_stop_sample       :', sound.waveform_stop_sample)
print('waveform_duration_sec      :', round(sound.waveform_duration_sec, 3), 's')

# === SOUND: read chunk in seconds ===
print()
print('='*62)
print('SOUND: read - chunk in seconds')
print('='*62)
sound.read(channel=0, chunk=[0, 10], unit='sec', detrend=True)
print('waveform shape        :', sound.waveform.shape)
print('waveform_start_sample :', sound.waveform_start_sample)
print('waveform_stop_sample  :', sound.waveform_stop_sample)
print('waveform_duration_sec :', round(sound.waveform_duration_sec, 3), 's')

# === SOUND: filter ===
print()
print('='*62)
print('SOUND: filter')
print('='*62)
sound.read(channel=0, chunk=[0, 30], unit='sec', detrend=True)
print('filter_applied before:', sound.filter_applied)
sound.filter('bandpass', [100, 1000], order=4)
print('filter_applied after :', sound.filter_applied)

# waveform comparison figure
s1 = Sound(WAV); s1.read(channel=0, chunk=[0, 10], unit='sec', detrend=True)
s2 = Sound(WAV); s2.read(channel=0, chunk=[0, 10], unit='sec', detrend=True)
s2.filter('bandpass', [100, 1000], order=4)
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
t = np.arange(len(s1.waveform)) / s1.waveform_sampling_frequency
axes[0].plot(t, s1.waveform, lw=0.5, color='steelblue')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Raw waveform (no filter)')
axes[1].plot(t, s2.waveform, lw=0.5, color='coral')
axes[1].set_ylabel('Amplitude')
axes[1].set_title('After bandpass filter 100-1000 Hz')
axes[1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(FIG + '/sound_waveform.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved sound_waveform.png')

# === SOUND: select_snippet ===
print()
print('='*62)
print('SOUND: select_snippet')
print('='*62)
sound.read(channel=0, chunk=[0, 30], unit='sec', detrend=True)
snippet = sound.select_snippet([5, 10], unit='sec')
print('snippet.waveform_duration_sec:', round(snippet.waveform_duration_sec, 3))
print('snippet.waveform_start_sample :', snippet.waveform_start_sample)
print('snippet.waveform shape        :', snippet.waveform.shape)

# === SOUND: normalize ===
print()
print('='*62)
print('SOUND: normalize')
print('='*62)
s3 = Sound(WAV); s3.read(channel=0, chunk=[0, 5], unit='sec', detrend=True)
print('max abs before normalize:', round(float(np.max(np.abs(s3.waveform))), 6))
s3.normalize(method='amplitude')
print('max abs after  normalize:', round(float(np.max(np.abs(s3.waveform))), 6))

# === SOUND: tighten_waveform_window ===
print()
print('='*62)
print('SOUND: tighten_waveform_window')
print('='*62)
s4 = Sound(WAV); s4.read(channel=0, chunk=[0, 30], unit='sec', detrend=True)
print('Duration before tighten:', round(s4.waveform_duration_sec, 3), 's')
s4.tighten_waveform_window(energy_percentage=80)
print('Duration after  tighten (80%):', round(s4.waveform_duration_sec, 3), 's')

# === FILTER ===
print()
print('='*62)
print('FILTER: design and attributes')
print('='*62)
filt = Filter('bandpass', [100, 1000], order=4)
print('type               :', filt.type)
print('cutoff_frequencies :', filt.cutoff_frequencies)
print('order              :', filt.order)
sos = filt.coefficients(4096)
print('coefficients shape :', sos.shape, '(second-order sections)')

# filter frequency response figure
fs = 4096
sos_bp = spsig.butter(4, [100, 1000], btype='bandpass', fs=fs, output='sos')
w, h = spsig.sosfreqz(sos_bp, worN=4096, fs=fs)
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogx(w, 20*np.log10(np.abs(h)+1e-12), color='steelblue', lw=2)
ax.axvline(100, color='gray', ls='--', lw=1, label='Cutoff frequencies')
ax.axvline(1000, color='gray', ls='--', lw=1)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Gain (dB)')
ax.set_title('Frequency response - bandpass filter 100-1000 Hz, order 4')
ax.set_xlim(10, fs/2)
ax.set_ylim(-80, 5)
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(FIG + '/filter_response.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved filter_response.png')

# === upsample ===
print()
print('='*62)
print('upsample function')
print('='*62)
s5 = Sound(WAV); s5.read(channel=0, chunk=[0, 5], unit='sec', detrend=True)
print('Original fs  :', s5.waveform_sampling_frequency, 'Hz')
print('Original len :', len(s5.waveform), 'samples')
up_waveform, up_fs = upsample(s5.waveform, 1/s5.waveform_sampling_frequency, 1/8192)
print('Upsampled fs :', up_fs, 'Hz')
print('Upsampled len:', len(up_waveform), 'samples')

# === SPECTROGRAM ===
print()
print('='*62)
print('SPECTROGRAM: __init__ in samples')
print('='*62)
sound5 = Sound(WAV)
sound5.read(channel=0, chunk=[0, 30], unit='sec', detrend=True)
spectro = Spectrogram(frame=3000, window_type='hann', fft=4096, step=500,
                      sampling_frequency=sound5.waveform_sampling_frequency,
                      unit='samp', verbose=True)
print('frame_samp      :', spectro.frame_samp)
print('fft_samp        :', spectro.fft_samp)
print('step_samp       :', spectro.step_samp)
print('frame_sec       :', round(spectro.frame_sec, 5), 's')
print('step_sec        :', round(spectro.step_sec, 5), 's')
print('freq_resolution :', round(spectro.frequency_resolution, 4), 'Hz')
print('time_resolution :', round(spectro.time_resolution, 5), 's')

print()
print('='*62)
print('SPECTROGRAM: compute')
print('='*62)
spectro.compute(sound5, dB=True, use_dask=False)
print('spectrogram shape   :', spectro.spectrogram.shape, '(freq bins x time bins)')
print('axis_frequencies[:5]:', np.round(spectro.axis_frequencies[:5], 2), 'Hz')
print('axis_times[:5]      :', np.round(spectro.axis_times[:5], 4), 's')

print()
print('='*62)
print('SPECTROGRAM: crop')
print('='*62)
spec_c = spectro.crop(frequency_min=0, frequency_max=1000, inplace=False)
print('Original freq range:', round(spectro.axis_frequencies[0],1), '-', round(spectro.axis_frequencies[-1],1), 'Hz')
print('Cropped  freq range:', round(spec_c.axis_frequencies[0],1), '-', round(spec_c.axis_frequencies[-1],1), 'Hz')
print('Original shape:', spectro.spectrogram.shape)
print('Cropped  shape:', spec_c.spectrogram.shape)

print()
print('='*62)
print('SPECTROGRAM: denoise')
print('='*62)
import copy
spec_dn = copy.copy(spec_c)
spec_dn.denoise('median_equalizer', window_duration=3, use_dask=False, inplace=True)
print('Denoised spectrogram shape:', spec_dn.spectrogram.shape)

# spectrogram comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im0 = axes[0].pcolormesh(spec_c.axis_times, spec_c.axis_frequencies,
                          spec_c.spectrogram, cmap='inferno', shading='auto')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title('Spectrogram (dB, 0-1000 Hz)')
plt.colorbar(im0, ax=axes[0], label='dB')
im1 = axes[1].pcolormesh(spec_dn.axis_times, spec_dn.axis_frequencies,
                          spec_dn.spectrogram, cmap='inferno', shading='auto', vmin=0)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('After median_equalizer denoising')
plt.colorbar(im1, ax=axes[1], label='dB')
plt.tight_layout()
plt.savefig(FIG + '/spectrogram_raw_denoised.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved spectrogram_raw_denoised.png')

print('ALL DONE')
