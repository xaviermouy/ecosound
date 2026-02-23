import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings, tempfile, os
warnings.filterwarnings('ignore')

from ecosound.core import tools
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation
import pandas as pd

DATA   = r'C:/Users/xavier.mouy/Documents/GitHub/ecosound/data'
FIG    = r'C:/Users/xavier.mouy/Documents/GitHub/ecosound/docs/source/tutorials/figures'

# =====================================================================
# TOOLS: list_files
# =====================================================================
print('='*62)
print('TOOLS: list_files')
print('='*62)
wav_files = tools.list_files(DATA + '/wav_files', '.wav', recursive=False)
print('WAV files found:')
for f in wav_files:
    print(' ', os.path.basename(f))

txt_files = tools.list_files(DATA + '/Raven_annotations', '.txt', recursive=False)
print('Raven .txt files found:')
for f in txt_files:
    print(' ', os.path.basename(f))

# =====================================================================
# TOOLS: filename_to_datetime
# =====================================================================
print()
print('='*62)
print('TOOLS: filename_to_datetime')
print('='*62)
fnames = [
    'data/wav_files/AMAR173.4.20190916T061248Z.wav',
    'data/wav_files/67674121.181018013806.wav',
    'data/wav_files/JASCOAMARHYDROPHONE742_20140913T084017.774Z.wav',
]
timestamps = tools.filename_to_datetime(fnames)
for fname, ts in zip(fnames, timestamps):
    print(' ', os.path.basename(fname), '->', ts)

# =====================================================================
# TOOLS: normalize_vector
# =====================================================================
print()
print('='*62)
print('TOOLS: normalize_vector')
print('='*62)
vec = np.array([3.0, 1.0, -2.0, 5.0, -1.0])
print('Input      :', vec)
normed = tools.normalize_vector(vec)
print('Normalized :', np.round(normed, 4))
print('Mean       :', round(float(np.mean(normed)), 6), '(approx 0)')
print('Max abs    :', round(float(np.max(np.abs(normed))), 6), '(approx 1)')

# =====================================================================
# TOOLS: tighten_signal_limits
# =====================================================================
print()
print('='*62)
print('TOOLS: tighten_signal_limits')
print('='*62)
rng = np.random.default_rng(42)
t = np.linspace(0, 1, 8000)
noise = rng.normal(0, 0.05, len(t))
burst = np.exp(-((t - 0.5)**2) / (2 * 0.01**2))
signal = noise + burst
chunk = tools.tighten_signal_limits(signal, energy_percentage=90)
print('Signal length   :', len(signal))
print('chunk (90 pct)  :', chunk)
print('Duration before :', len(signal), 'samples')
print('Duration after  :', chunk[1] - chunk[0], 'samples')

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t, signal, lw=0.5)
axes[0].set_title('Original signal')
axes[0].set_ylabel('Amplitude')
axes[1].plot(t, signal, lw=0.5)
axes[1].axvspan(t[chunk[0]], t[chunk[1]], color='gold', alpha=0.4, label='90% energy window')
axes[1].set_title('tighten_signal_limits - 90% energy window highlighted')
axes[1].set_ylabel('Amplitude')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
plt.tight_layout()
plt.savefig(FIG + '/tools_tighten.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved tools_tighten.png')

# =====================================================================
# TOOLS: resample_1D_array
# =====================================================================
print()
print('='*62)
print('TOOLS: resample_1D_array')
print('='*62)
x = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 10.0])
y = np.array([0.0, 2.0, 1.5, 3.0, 1.0, 2.5])
xnew, ynew = tools.resample_1D_array(x, y, resolution=0.5)
print('Original x      :', x)
print('Original y      :', y)
print('Resampled x[:8] :', xnew[:8])
print('Resampled y[:8] :', np.round(ynew[:8], 3))

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, y, 'o-', label='Original (non-uniform)', ms=8)
ax.plot(xnew, ynew, '.-', lw=1, label='Resampled (uniform, step=0.5)', ms=4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('resample_1D_array - linear interpolation to uniform grid')
ax.legend()
plt.tight_layout()
plt.savefig(FIG + '/tools_resample.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved tools_resample.png')

# =====================================================================
# TOOLS: entropy
# =====================================================================
print()
print('='*62)
print('TOOLS: entropy')
print('='*62)
flat = np.ones(100, dtype=np.float64)
peaked = np.zeros(100, dtype=np.float64)
peaked[50] = 100.0
H_flat   = tools.entropy(flat)
H_peaked = tools.entropy(peaked)
print('Flat spectrum entropy  :', round(H_flat, 4), ' (high = broadband)')
print('Peaked spectrum entropy:', round(H_peaked, 4), '(low  = tonal)')

# =====================================================================
# TOOLS: derivative_1d
# =====================================================================
print()
print('='*62)
print('TOOLS: derivative_1d')
print('='*62)
y_d = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
d1 = tools.derivative_1d(y_d, order=1)
d2 = tools.derivative_1d(y_d, order=2)
print('Input    :', y_d)
print('1st deriv:', d1)
print('2nd deriv:', d2)

# =====================================================================
# TOOLS: find_peaks
# =====================================================================
print()
print('='*62)
print('TOOLS: find_peaks')
print('='*62)
x_peaks = np.array([0.0, 1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 0.5])
peak_idx, peak_val = tools.find_peaks(x_peaks)
trough_idx, trough_val = tools.find_peaks(x_peaks, troughs=True)
print('Signal     :', x_peaks)
print('Peak idx   :', peak_idx, '-> values', peak_val)
print('Trough idx :', trough_idx, '-> values', trough_val)

# =====================================================================
# TOOLS: envelope
# =====================================================================
print()
print('='*62)
print('TOOLS: envelope')
print('='*62)
rng2 = np.random.default_rng(0)
t_env = np.linspace(0, 1, 500)
carrier = np.sin(2 * np.pi * 20 * t_env)
modulation = np.abs(np.sin(2 * np.pi * 2 * t_env)) + 0.1
sig_env = carrier * modulation + rng2.normal(0, 0.05, len(t_env))
env_high, env_low = tools.envelope(sig_env)
print('Signal length  :', len(sig_env))
print('env_high range :', round(float(env_high.min()), 3), 'to', round(float(env_high.max()), 3))
print('env_low  range :', round(float(env_low.min()), 3), 'to', round(float(env_low.max()), 3))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_env, sig_env, lw=0.5, color='steelblue', label='Signal')
ax.plot(t_env, env_high, 'r-', lw=2, label='Upper envelope')
ax.plot(t_env, env_low,  'g-', lw=2, label='Lower envelope')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('envelope - upper and lower amplitude envelopes')
ax.legend()
plt.tight_layout()
plt.savefig(FIG + '/tools_envelope.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved tools_envelope.png')

# =====================================================================
# METADATA: DeploymentInfo
# =====================================================================
print()
print('='*62)
print('METADATA: DeploymentInfo - write_template and read')
print('='*62)
tmpdir = tempfile.mkdtemp()
template_path = os.path.join(tmpdir, 'deployment_template.csv')
dep = DeploymentInfo()
dep.write_template(template_path)
print('Template written. Columns:')
df_tmpl = pd.read_csv(template_path)
for col in df_tmpl.columns:
    print('  -', col)

filled_path = os.path.join(tmpdir, 'deployment_filled.csv')
filled = pd.DataFrame({
    'audio_channel_number': [0],
    'UTC_offset': [-8],
    'sampling_frequency': [32000],
    'bit_depth': [24],
    'mooring_platform_name': ['Bottom lander'],
    'recorder_type': ['AMAR'],
    'recorder_SN': ['173'],
    'hydrophone_model': ['HTI-96-MIN'],
    'hydrophone_SN': ['12345'],
    'hydrophone_depth': [40.0],
    'location_name': ['Hornby Island'],
    'location_lat': [49.52],
    'location_lon': [-124.68],
    'location_water_depth': [55.0],
    'deployment_ID': ['HB-2019-001'],
    'deployment_date': ['2019-09-16'],
    'recovery_date': ['2020-03-15'],
})
filled.to_csv(filled_path, index=False)
dep2 = DeploymentInfo()
dep2.read(filled_path)
print()
print('Loaded deployment info (transposed):')
print(dep2.data.T.to_string())

# =====================================================================
# MEASUREMENT: constructor and metadata
# =====================================================================
print()
print('='*62)
print('MEASUREMENT: constructor and metadata attribute')
print('='*62)
meas = Measurement(
    measurer_name='SpectrogramFeatures',
    measurer_version='1.0',
    measurements_name=['SNR_dB', 'peak_freq_Hz', 'duration_s'],
    measurements_parameters={'window_sec': 0.1},
)
print(meas)
print()
print('metadata DataFrame:')
print(meas.metadata.to_string())
print()
print('Extra measurement columns in data:', ['SNR_dB', 'peak_freq_Hz', 'duration_s'])
print('Total columns in data DataFrame   :', len(meas.data.columns))

# MEASUREMENT: from_sqlite
print()
print('='*62)
print('MEASUREMENT: using from_sqlite (Annotation-inherited method)')
print('='*62)
sqlite_file = DATA + '/sqlite_annotations/read/detections1.sqlite'
annot_meas = Annotation()
annot_meas.from_sqlite(sqlite_file, verbose=True)
print()
print('label_class values :', annot_meas.get_labels_class())
print('confidence describe:')
print(annot_meas.data['confidence'].describe().round(3))

print('ALL DONE')
