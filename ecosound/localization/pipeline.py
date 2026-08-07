"""
ecosound.localization.pipeline
==============================
LocalizationPipeline: end-to-end TDOA-based 3-D localization pipeline.

Loads deployment and localization YAML configs, builds the appropriate
localizer (currently GridSearch), then runs the full processing chain
(waveform loading → tightening → upsampling/interpolation → TDOA estimation →
localization → credibility intervals) for every detection in an Annotation.

Usage
-----
Single file, serial (interactive plots work)::

    annot = Annotation()
    annot.from_raven('detections.txt')
    results = pipeline.run(annot, deployment_date=pd.Timestamp('2026-02-10'))

Multiple files, parallel (Dask, per-detection plots disabled)::

    annot = Annotation()
    annot.from_raven(['file1.txt', 'file2.txt', ...])
    results = pipeline.run(annot, parallel=True, save_intermediate=True,
                           deployment_date=...)

Load intermediates for interactive review::

    data = LocalizationPipeline.load_h5('output/AC8_8_localizations.h5')
"""

import hashlib
import json
import os
import pickle
import shutil
import numpy as np
import pandas as pd

import ecosound.core.tools
from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
from ecosound.core.measurement import Measurement

from .deployment import Deployment
from .gridsearch import GridSearch
from .tdoa import compute_tdoas
from .preprocessing import stack_waveforms, tighten_waveforms, upsample_stack
from .visualization import (
    plot_waveform_stack,
    plot_waveforms_overlaid,
    plot_tdoa_pairs,
    plot_localization_3d,
    plot_localization_2d,
    plot_ppd_slices,
    plot_all_localizations_2d,
    plot_all_localizations_3d,
)


# ---------------------------------------------------------------------------
# Grid hash helper
# ---------------------------------------------------------------------------

def _compute_grid_hash(config, array):
    """MD5 hash of all inputs that determine the TDOA grid."""
    gs   = config['gridsearch']
    tdoa = config['tdoa']
    env  = config['environment']
    key  = {
        'sound_speed_mps': env['sound_speed_mps'],
        'ref_channel':     tdoa['ref_channel'],
        'x_limits_m':      gs['x_limits_m'],
        'y_limits_m':      gs['y_limits_m'],
        'z_limits_m':      gs['z_limits_m'],
        'spacing_m':       gs['spacing_m'],
        'hp_x':            array.x.tolist(),
        'hp_y':            array.y.tolist(),
        'hp_z':            array.z.tolist(),
    }
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# HDF5 intermediate helpers
# ---------------------------------------------------------------------------

def _extract_ppd_slices(PPD_xr, m):
    """Extract three 2D PPD slices at the MAP location.

    Returns a dict with keys ``'xy'``, ``'xz'``, ``'yz'``, each containing
    ``'values'`` (float32 2-D array) and the two coordinate arrays
    (``'x'``, ``'y'``, or ``'z'``).  Returns ``None`` when ``PPD_xr`` is
    ``None``.
    """
    if PPD_xr is None:
        return None
    x0, y0, z0 = m
    sl_xy = PPD_xr.PPD.sel(z=z0, method='nearest')
    sl_xz = PPD_xr.PPD.sel(y=y0, method='nearest')
    sl_yz = PPD_xr.PPD.sel(x=x0, method='nearest')
    return {
        'xy': {'values': sl_xy.values.astype(np.float32),
               'x': sl_xy['x'].values, 'y': sl_xy['y'].values},
        'xz': {'values': sl_xz.values.astype(np.float32),
               'x': sl_xz['x'].values, 'z': sl_xz['z'].values},
        'yz': {'values': sl_yz.values.astype(np.float32),
               'y': sl_yz['y'].values, 'z': sl_yz['z'].values},
    }


def _write_detection_to_h5(h5, det_key, data):
    """Write one detection's intermediate data into an open h5py.File.

    Parameters
    ----------
    h5 : h5py.File
        Open (writable) HDF5 file.
    det_key : str
        Group name under ``/intermediates/``, e.g. ``'det_000042'``.
    data : dict
        Keys: ``detec_idx``, ``waveforms``, ``fs_waveforms``,
        ``cc_list``, ``lags_list``, ``fs_cc``,
        ``tdoa_ref_sec``, ``corr_ref``, ``ppd_slices``.
    """
    g = h5.require_group(f'intermediates/{det_key}')
    g.attrs['detection_id'] = int(data['detec_idx'])

    # Tightened waveforms (native fs, before upsampling)
    wg = g.require_group('waveforms')
    wg.attrs['fs'] = float(data['fs_waveforms'])
    for i, w in enumerate(data['waveforms']):
        wg.create_dataset(f'ch_{i}', data=w.astype(np.float32),
                          compression='gzip', compression_opts=4)

    # Cross-correlation functions and lags (at fs_cc)
    cg = g.require_group('cc')
    lg = g.require_group('lags')
    cg.attrs['fs'] = float(data['fs_cc'])
    for i, (cc, lags) in enumerate(zip(data['cc_list'], data['lags_list'])):
        cg.create_dataset(f'pair_{i}', data=cc.astype(np.float32),
                          compression='gzip', compression_opts=4)
        lg.create_dataset(f'pair_{i}', data=lags.astype(np.int32),
                          compression='gzip', compression_opts=4)

    g.create_dataset('tdoa_ref_sec', data=data['tdoa_ref_sec'])
    g.create_dataset('corr_ref',     data=data['corr_ref'])

    # PPD slices (only when localization succeeded)
    if data['ppd_slices'] is not None:
        pg = g.require_group('ppd')
        for plane, sdata in data['ppd_slices'].items():
            sg = pg.require_group(plane)
            sg.create_dataset('values', data=sdata['values'],
                              compression='gzip', compression_opts=4)
            for coord_name, coord_val in sdata.items():
                if coord_name != 'values':
                    sg.create_dataset(coord_name, data=coord_val)


def _init_h5_file(h5, config, hp_x, hp_y, hp_z, hp_channels,
                  x_unique, y_unique, z_unique):
    """Write static metadata into a freshly opened h5py.File."""
    h5.attrs['config'] = json.dumps(config)

    ag = h5.require_group('array')
    ag.create_dataset('hp_x',     data=np.asarray(hp_x,       dtype=np.float64))
    ag.create_dataset('hp_y',     data=np.asarray(hp_y,       dtype=np.float64))
    ag.create_dataset('hp_z',     data=np.asarray(hp_z,       dtype=np.float64))
    ag.create_dataset('channels', data=np.asarray(hp_channels, dtype=np.int32))

    gg = h5.require_group('grid')
    gg.create_dataset('x_unique', data=x_unique.astype(np.float64))
    gg.create_dataset('y_unique', data=y_unique.astype(np.float64))
    gg.create_dataset('z_unique', data=z_unique.astype(np.float64))


def _finalize_h5_file(h5, annotations_df, localizations_df):
    """Append annotation and localization DataFrames to an open h5py.File."""
    import h5py as _h5py
    dt = _h5py.special_dtype(vlen=str)
    ann_json = annotations_df.to_json(date_format='iso')
    loc_json = localizations_df.to_json(date_format='iso')
    h5.create_dataset('annotations',   data=ann_json, dtype=dt)
    h5.create_dataset('localizations', data=loc_json, dtype=dt)
    h5.attrs['n_detections'] = len(annotations_df)


def _merge_temp_files(temp_dir, indices, h5_path, config,
                      hp_x, hp_y, hp_z, hp_channels,
                      x_unique, y_unique, z_unique,
                      annotations_df, localizations_df):
    """Read per-detection temp pickles and assemble one HDF5 file.

    Parameters
    ----------
    temp_dir : str
        Directory containing ``det_{idx:06d}.pkl`` files.
    indices : list of int
        Detection IDs (original annotation row indices) belonging to this
        audio file group.
    h5_path : str
        Destination HDF5 file path.
    """
    import h5py as _h5py
    with _h5py.File(h5_path, 'w') as h5:
        _init_h5_file(h5, config, hp_x, hp_y, hp_z, hp_channels,
                      x_unique, y_unique, z_unique)
        for idx in sorted(indices):
            pkl_path = os.path.join(temp_dir, f'det_{idx:06d}.pkl')
            if not os.path.isfile(pkl_path):
                continue
            with open(pkl_path, 'rb') as fh:
                data = pickle.load(fh)
            _write_detection_to_h5(h5, f'det_{idx:06d}', data)
        _finalize_h5_file(h5, annotations_df, localizations_df)
    print(f'Intermediates saved → {h5_path}')


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class LocalizationPipeline:
    """End-to-end localization pipeline.

    Combines a :class:`~ecosound.localization.deployment.DeployedArray`
    with a localization algorithm (e.g. :class:`~ecosound.localization.gridsearch.GridSearch`)
    and all processing parameters from a ``localization.yaml`` config file.

    Typical usage::

        pipeline = LocalizationPipeline.from_yaml(
            deployment_file   = 'deployment.yaml',
            localization_file = 'localization.yaml',
            tdoa_grid_file    = 'output/tdoa_grid.npz',
        )
        pipeline.out_dir          = 'Figures/'
        pipeline.plot_all_2d      = True
        pipeline.save_nc          = True
        pipeline.save_intermediate = True

        annot = Annotation()
        annot.from_raven(['file1.txt', 'file2.txt'])
        results = pipeline.run(annot, parallel=True,
                               deployment_date=pd.Timestamp('2026-02-10'))

        # Later — load for interactive review
        data = LocalizationPipeline.load_h5('Figures/file1_localizations.h5')

    Parameters
    ----------
    dep_array : DeployedArray
    localizer : object
        Any object implementing ``localizer.localize(tdoa_ref_sec, corr_ref,
        min_corr_val, CI_percentage)``.  Currently only :class:`GridSearch`.
    config : dict
        Full localization config dict.
    """

    def __init__(self, dep_array, localizer, config):
        self.dep_array = dep_array
        self.array     = dep_array.sensor_array
        self.localizer = localizer
        self.config    = config

        # --- environment ------------------------------------------------
        env = config['environment']
        self.sound_speed_mps = env['sound_speed_mps']

        # --- TDOA -------------------------------------------------------
        tdoa = config['tdoa']
        self.ref_channel      = tdoa['ref_channel']
        self.upsample_res_sec = tdoa.get('upsample_res_sec', 1e-7)
        self.min_corr_val     = tdoa['min_corr_val']
        self.tdoa_method      = tdoa['method']
        self.tdoa_precision   = tdoa.get('tdoa_precision', 'upsample')

        # --- waveform tightening ----------------------------------------
        wt = config['waveform_tightening']
        self.tighten_method             = wt['method']
        self.energy_window_perc         = wt['energy_window_perc']
        self.peak_window_half_width_sec = wt['peak_window_half_width_sec']
        self.envelope_threshold_perc    = wt['envelope_threshold_perc']
        self.envelope_pad_sec           = wt['envelope_pad_sec']

        # --- credibility interval ---------------------------------------
        self.CI_percentage = config['credibility_interval']['percentage']

        # --- derived geometry -------------------------------------------
        self.ref_pairs       = self.array.define_pairs(self.ref_channel)
        self.ref_pair_labels = self.array.pair_labels(self.ref_pairs)
        self.hp_x            = self.array.x
        self.hp_y            = self.array.y
        self.hp_z            = self.array.z
        self.hp_channels     = self.array.channels
        self.tdoa_max_sec    = self.array.tdoa_max_sec(self.sound_speed_mps)

        # --- output / display flags (set before calling run()) ----------
        self.out_dir           = None
        self.save_intermediate = False   # write HDF5 with waveforms/CC/PPD slices

        self.plot_waveforms    = False;  self.save_png_waveforms  = False
        self.plot_tightening   = False;  self.save_png_tightening = False
        self.plot_tdoas        = False;  self.save_png_tdoas      = False
        self.plot_loc_3d       = False;  self.save_png_loc_3d     = False
        self.plot_loc_2d       = False;  self.save_png_loc_2d     = False
        self.plot_ppd          = False;  self.save_png_ppd        = False
        self.plot_all_2d       = False;  self.save_png_all_2d     = False
        self.plot_all_3d       = False;  self.save_png_all_3d     = False
        self.save_nc           = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, deployment_file, localization_file, tdoa_grid_file,
                  array_number=0):
        """Build a pipeline from YAML config files.

        The TDOA grid is re-used when all inputs that determine it match a
        stored hash sidecar (``<tdoa_grid_file>.hash``).

        Parameters
        ----------
        deployment_file : str
        localization_file : str
        tdoa_grid_file : str
        array_number : int

        Returns
        -------
        LocalizationPipeline
        """
        config     = ecosound.core.tools.read_yaml(localization_file)
        deployment = Deployment.from_yaml(deployment_file)
        dep_array  = deployment.get_array(array_number)
        array      = dep_array.sensor_array

        method = config.get('method', 'gridsearch')
        if method == 'gridsearch':
            sound_speed_mps = config['environment']['sound_speed_mps']
            ref_channel     = config['tdoa']['ref_channel']
            gs              = config['gridsearch']

            os.makedirs(
                os.path.dirname(os.path.abspath(tdoa_grid_file)), exist_ok=True
            )

            hash_file    = tdoa_grid_file + '.hash'
            new_hash     = _compute_grid_hash(config, array)
            need_rebuild = True

            if os.path.isfile(tdoa_grid_file) and os.path.isfile(hash_file):
                try:
                    with open(hash_file) as fh:
                        if fh.read().strip() == new_hash:
                            need_rebuild = False
                            print('TDOA grid: cache hit — reusing existing grid '
                                  f'({tdoa_grid_file})')
                except OSError:
                    pass

            if need_rebuild:
                print('TDOA grid: building grid …')
                GridSearch.create_tdoa_grid(
                    gs['x_limits_m'], gs['y_limits_m'], gs['z_limits_m'],
                    gs['spacing_m'],
                    array, ref_channel, sound_speed_mps, tdoa_grid_file,
                )
                with open(hash_file, 'w') as fh:
                    fh.write(new_hash)
                print(f'TDOA grid: saved to {tdoa_grid_file}')

            localizer = GridSearch(GridSearch.load_tdoa_grid(tdoa_grid_file))
        else:
            raise ValueError(
                f"Unknown localization method {method!r}. "
                f"Supported: 'gridsearch'."
            )

        return cls(dep_array, localizer, config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, annotations, parallel=False, deployment_date=None):
        """Run the full localization pipeline on an Annotation object.

        Parameters
        ----------
        annotations : Annotation
            Detections to localize.  May span one or many audio files.
            Build with ``Annotation.from_raven()`` which accepts a path,
            list of paths, or folder.
        parallel : bool
            If True, process each detection as an independent Dask task.
            Per-detection plots are suppressed; save flags still work.
        deployment_date : pandas.Timestamp or None
            When provided, absolute timestamps are added to detections.

        Returns
        -------
        pandas.DataFrame
        """
        import h5py as _h5py
        import matplotlib.pyplot as plt

        # --- timestamps -----------------------------------------------
        if deployment_date is not None:
            annotations.data['time_min_date'] = deployment_date + pd.to_timedelta(
                annotations.data['time_min_offset'], unit='s')
            annotations.data['time_max_date'] = deployment_date + pd.to_timedelta(
                annotations.data['time_max_offset'], unit='s')

        # --- resolve audio context once per unique recording ----------
        audio_ctx = {}
        for audio_name, grp in annotations.data.groupby('audio_file_name'):
            row0           = grp.iloc[0]
            audio_filename = audio_name + row0['audio_file_extension']
            audio_files    = self.dep_array.find_audio_files(audio_filename)
            ref_sound      = Sound(audio_files['path'][0])
            ref_sound.read(channel=audio_files['channel'][0],
                           chunk=[0, 1], unit='sec')
            fs_raw = ref_sound.waveform_sampling_frequency
            audio_ctx[audio_name] = (audio_files, fs_raw)
            annotations.data.loc[grp.index, 'audio_sampling_frequency'] = fs_raw

        flat = [
            (idx, row,
             audio_ctx[row['audio_file_name']][0],
             audio_ctx[row['audio_file_name']][1])
            for idx, row in annotations.data.iterrows()
        ]
        n_total = len(flat)

        # --- open HDF5 handles (serial) or temp dir (parallel) --------
        do_h5     = self.save_intermediate and self.out_dir is not None
        h5_handles = {}   # audio_name -> (h5py.File, h5_path)  [serial only]
        temp_dir   = None

        if do_h5:
            os.makedirs(self.out_dir, exist_ok=True)
            if not parallel:
                for audio_name in audio_ctx:
                    h5_path = os.path.join(
                        self.out_dir, audio_name + '_localizations.h5'
                    )
                    h5 = _h5py.File(h5_path, 'w')
                    _init_h5_file(h5, self.config,
                                  self.hp_x, self.hp_y, self.hp_z,
                                  self.hp_channels,
                                  self.localizer.x_unique,
                                  self.localizer.y_unique,
                                  self.localizer.z_unique)
                    h5_handles[audio_name] = (h5, h5_path)
            else:
                temp_dir = os.path.join(self.out_dir, '_tmp_intermediates')
                os.makedirs(temp_dir, exist_ok=True)

        # --- run -------------------------------------------------------
        if parallel:
            localizations_list = self._run_dask(flat, temp_dir=temp_dir)
        else:
            localizations_list = self._run_serial(flat, n_total,
                                                   h5_handles=h5_handles)

        # --- assemble DataFrame ----------------------------------------
        localizations_df = pd.DataFrame(localizations_list).set_index('detection_id')

        print('\n=== Localization summary ===')
        print(localizations_df[[
            'x_m', 'y_m', 'z_m',
            'x_ci_low', 'x_ci_high',
            'y_ci_low', 'y_ci_high',
            'z_ci_low', 'z_ci_high', 'loc_ok',
        ]])

        # --- summary plots (all detections combined) -------------------
        audio_names = list(audio_ctx.keys())
        out_stem    = audio_names[0] if len(audio_names) == 1 else 'localizations'

        if self.plot_all_2d or self.save_png_all_2d:
            fig = plot_all_localizations_2d(
                localizations_df,
                self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                self.localizer.x_unique, self.localizer.y_unique,
                self.localizer.z_unique,
            )
            self._show_save(fig, self.plot_all_2d,
                self._fig_path_summary(out_stem, 'all_localizations_2d')
                if self.save_png_all_2d else None)
        if self.plot_all_3d or self.save_png_all_3d:
            fig = plot_all_localizations_3d(
                localizations_df,
                self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                self.localizer.x_unique, self.localizer.y_unique,
                self.localizer.z_unique,
            )
            self._show_save(fig, self.plot_all_3d,
                self._fig_path_summary(out_stem, 'all_localizations_3d')
                if self.save_png_all_3d else None)

        # --- .nc saving: one file per audio recording -----------------
        if self.save_nc and self.out_dir:
            for audio_name, (audio_files, fs_raw) in audio_ctx.items():
                mask    = annotations.data['audio_file_name'] == audio_name
                indices = annotations.data[mask].index
                grp_ann = Annotation()
                grp_ann.data = annotations.data[mask].copy()
                grp_ldf = localizations_df.loc[
                    localizations_df.index.isin(indices)
                ]
                self._save_netcdf(grp_ldf, grp_ann, fs_raw, audio_name)

        # --- finalize HDF5 files --------------------------------------
        if do_h5:
            if not parallel:
                for audio_name, (h5, h5_path) in h5_handles.items():
                    mask    = annotations.data['audio_file_name'] == audio_name
                    grp_ann = annotations.data[mask]
                    grp_ldf = localizations_df[localizations_df.index.isin(grp_ann.index)]
                    _finalize_h5_file(h5, grp_ann, grp_ldf)
                    h5.close()
                    print(f'Intermediates saved → {h5_path}')
            else:
                for audio_name in audio_ctx:
                    h5_path = os.path.join(
                        self.out_dir, audio_name + '_localizations.h5'
                    )
                    mask    = annotations.data['audio_file_name'] == audio_name
                    grp_ann = annotations.data[mask]
                    grp_ldf = localizations_df[localizations_df.index.isin(grp_ann.index)]
                    _merge_temp_files(
                        temp_dir, grp_ann.index.tolist(), h5_path, self.config,
                        self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                        self.localizer.x_unique, self.localizer.y_unique,
                        self.localizer.z_unique, grp_ann, grp_ldf,
                    )
                shutil.rmtree(temp_dir)

        return localizations_df

    # ------------------------------------------------------------------
    # Serial path — full per-detection plots, optional streaming H5
    # ------------------------------------------------------------------

    def _run_serial(self, flat, n_total, h5_handles=None):
        """Process detections one at a time; all plot/save flags are active.

        Parameters
        ----------
        h5_handles : dict or None
            ``{audio_file_name: (h5py.File, h5_path)}``.  When provided,
            intermediate data for each detection is written to the
            corresponding HDF5 file as it completes.
        """
        localizations_list = []
        use_interpolation  = (self.tdoa_precision == 'interpolate')

        for detec_idx, detec, audio_files, fs_raw in flat:
            fs = fs_raw

            print(f'\n=== Detection {detec_idx + 1}/{n_total} '
                  f'  t=[{detec["time_min_offset"]:.3f}, {detec["time_max_offset"]:.3f}] s'
                  f'  f=[{detec["frequency_min"]:.0f}, {detec["frequency_max"]:.0f}] Hz ===')

            audio_file_stem = detec['audio_file_name']

            # 1. Load waveforms
            waveform_stack = stack_waveforms(audio_files, detec, self.tdoa_max_sec)
            if self.plot_waveforms or self.save_png_waveforms:
                fig = plot_waveform_stack(
                    waveform_stack, fs, self.hp_channels, self.ref_channel,
                    detec_idx, detec, title_suffix='raw (padded)',
                )
                self._show_save(fig, self.plot_waveforms,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'waveforms_stacked')
                    if self.save_png_waveforms else None)
                t_raw_ms = np.arange(len(waveform_stack[0])) / fs * 1000
                fig = plot_waveforms_overlaid(
                    waveform_stack, t_raw_ms, self.hp_channels, self.ref_channel,
                    detec_idx, detec, title_suffix='raw waveforms overlaid',
                )
                self._show_save(fig, self.plot_waveforms,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'waveforms_overlaid')
                    if self.save_png_waveforms else None)

            # 2. Tighten
            waveform_stack, t_wf_ms = tighten_waveforms(
                waveform_stack, fs, self.ref_channel,
                method                  = self.tighten_method,
                energy_window_perc      = self.energy_window_perc,
                peak_half_width_sec     = self.peak_window_half_width_sec,
                envelope_threshold_perc = self.envelope_threshold_perc,
                envelope_pad_sec        = self.envelope_pad_sec,
            )
            if self.plot_tightening or self.save_png_tightening:
                fig = plot_waveforms_overlaid(
                    waveform_stack, t_wf_ms, self.hp_channels, self.ref_channel,
                    detec_idx, detec, title_suffix='tightened waveforms overlaid',
                )
                self._show_save(fig, self.plot_tightening,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'tightening')
                    if self.save_png_tightening else None)

            # Save tightened waveforms at native fs (before upsampling)
            waveforms_to_save = [w.copy() for w in waveform_stack]

            # 3. Sub-sample precision
            if not use_interpolation:
                waveform_stack, fs = upsample_stack(
                    waveform_stack, fs, self.upsample_res_sec
                )

            # 4. TDOAs
            tdoa_ref_sec, corr_ref, cc_list, lags_list = compute_tdoas(
                waveform_stack, self.ref_pairs, fs,
                self.tdoa_max_sec, method=self.tdoa_method,
                subsample_interpolation=use_interpolation,
            )
            if self.plot_tdoas or self.save_png_tdoas:
                fig = plot_tdoa_pairs(
                    waveform_stack, fs, self.ref_pairs, self.ref_pair_labels,
                    self.hp_channels, cc_list, lags_list,
                    tdoa_ref_sec.ravel(), corr_ref.ravel(),
                    self.tdoa_max_sec, detec_idx, detec,
                    method_name=self.tdoa_method.upper(),
                )
                self._show_save(fig, self.plot_tdoas,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'tdoas')
                    if self.save_png_tdoas else None)

            print('  TDOAs (ref pairs):')
            for lbl, tdoa, corr in zip(
                self.ref_pair_labels, tdoa_ref_sec.ravel(), corr_ref.ravel()
            ):
                print(f'    {lbl}: {tdoa * 1000:+.4f} ms   corr={corr:.3f}')

            # 5. Localize
            m, Px_CI, Py_CI, Pz_CI, loc_ok, PPD_xr = self.localizer.localize(
                tdoa_ref_sec, corr_ref, self.min_corr_val, self.CI_percentage,
            )
            if self.plot_loc_3d or self.save_png_loc_3d:
                fig = plot_localization_3d(
                    m, Px_CI, Py_CI, Pz_CI, loc_ok,
                    self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                    self.localizer.x_unique, self.localizer.y_unique,
                    self.localizer.z_unique,
                    corr_ref, self.min_corr_val, detec_idx, detec,
                )
                self._show_save(fig, self.plot_loc_3d,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'loc3d')
                    if self.save_png_loc_3d else None)
            if self.plot_loc_2d or self.save_png_loc_2d:
                fig = plot_localization_2d(
                    m, Px_CI, Py_CI, Pz_CI, loc_ok,
                    self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                    self.localizer.x_unique, self.localizer.y_unique,
                    self.localizer.z_unique, detec_idx, detec,
                )
                self._show_save(fig, self.plot_loc_2d,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'loc2d')
                    if self.save_png_loc_2d else None)
            if (self.plot_ppd or self.save_png_ppd) and loc_ok:
                fig = plot_ppd_slices(
                    PPD_xr, m,
                    self.hp_x, self.hp_y, self.hp_z, self.hp_channels,
                    detec_idx, detec,
                )
                self._show_save(fig, self.plot_ppd,
                    self._fig_path(audio_file_stem, detec_idx, detec, 'ppd')
                    if self.save_png_ppd else None)

            if loc_ok:
                print(f'  Location : x={m[0]:.3f} m  y={m[1]:.3f} m  z={m[2]:.3f} m')
                print(f'  {self.CI_percentage*100:.0f}% CI  : '
                      f'x=[{Px_CI[0]:.3f},{Px_CI[1]:.3f}]  '
                      f'y=[{Py_CI[0]:.3f},{Py_CI[1]:.3f}]  '
                      f'z=[{Pz_CI[0]:.3f},{Pz_CI[1]:.3f}] m')
            else:
                print(f'  REJECTED  '
                      f'(min corr={np.min(corr_ref):.3f} < threshold={self.min_corr_val})')

            # 6. Write intermediates to HDF5 (streaming — one detection at a time)
            if h5_handles is not None:
                h5, _ = h5_handles[detec['audio_file_name']]
                _write_detection_to_h5(h5, f'det_{detec_idx:06d}', {
                    'detec_idx'    : detec_idx,
                    'waveforms'    : waveforms_to_save,
                    'fs_waveforms' : float(fs_raw),
                    'cc_list'      : cc_list,
                    'lags_list'    : lags_list,
                    'fs_cc'        : float(fs),
                    'tdoa_ref_sec' : tdoa_ref_sec,
                    'corr_ref'     : corr_ref,
                    'ppd_slices'   : _extract_ppd_slices(PPD_xr, m) if loc_ok else None,
                })

            # 7. Store result
            localizations_list.append({
                'detection_id'    : detec_idx,
                'time_min_offset' : detec['time_min_offset'],
                'x_m': m[0], 'y_m': m[1], 'z_m': m[2],
                'x_ci_low' : Px_CI[0], 'x_ci_high': Px_CI[1],
                'y_ci_low' : Py_CI[0], 'y_ci_high': Py_CI[1],
                'z_ci_low' : Pz_CI[0], 'z_ci_high': Pz_CI[1],
                'loc_ok'   : loc_ok,
                **{f'tdoa_{lbl}_ms': v * 1000
                   for lbl, v in zip(self.ref_pair_labels, tdoa_ref_sec.ravel())},
                **{f'corr_{lbl}': v
                   for lbl, v in zip(self.ref_pair_labels, corr_ref.ravel())},
            })

        return localizations_list

    # ------------------------------------------------------------------
    # Parallel path — Dask, temp pickle per detection
    # ------------------------------------------------------------------

    def _run_dask(self, flat, temp_dir=None):
        """Process detections in parallel with Dask.

        Each worker writes a temp pickle when ``temp_dir`` is provided.
        Returns only scalar result dicts (the merge to HDF5 happens in
        ``run()`` after all tasks complete).
        """
        import dask
        from dask import delayed

        print(f'Processing {len(flat)} detections in parallel (Dask) …')
        tasks = [
            delayed(self._localize_one)(detec_idx, detec, audio_files, fs_raw,
                                        temp_dir=temp_dir)
            for detec_idx, detec, audio_files, fs_raw in flat
        ]
        return list(dask.compute(*tasks, scheduler='processes'))

    # ------------------------------------------------------------------
    # Headless single-detection (parallel workers + localize_detection)
    # ------------------------------------------------------------------

    def _localize_one(self, detec_idx, detec, audio_files, fs_raw,
                      temp_dir=None):
        """Localize one detection without plotting.

        When ``temp_dir`` is provided, writes a compressed pickle with all
        intermediate data (tightened waveforms, CC functions, PPD slices)
        and returns only the scalar result dict.

        Parameters
        ----------
        temp_dir : str or None
            Directory for temp pickle output.  Filename:
            ``det_{detec_idx:06d}.pkl``.
        """
        fs = fs_raw
        use_interpolation = (self.tdoa_precision == 'interpolate')

        waveform_stack = stack_waveforms(audio_files, detec, self.tdoa_max_sec)
        waveform_stack, _ = tighten_waveforms(
            waveform_stack, fs, self.ref_channel,
            method                  = self.tighten_method,
            energy_window_perc      = self.energy_window_perc,
            peak_half_width_sec     = self.peak_window_half_width_sec,
            envelope_threshold_perc = self.envelope_threshold_perc,
            envelope_pad_sec        = self.envelope_pad_sec,
        )

        # Save tightened waveforms at native fs (before upsampling)
        waveforms_to_save = [w.astype(np.float32) for w in waveform_stack]

        if not use_interpolation:
            waveform_stack, fs = upsample_stack(
                waveform_stack, fs, self.upsample_res_sec
            )
        tdoa_ref_sec, corr_ref, cc_list, lags_list = compute_tdoas(
            waveform_stack, self.ref_pairs, fs,
            self.tdoa_max_sec, method=self.tdoa_method,
            subsample_interpolation=use_interpolation,
        )
        m, Px_CI, Py_CI, Pz_CI, loc_ok, PPD_xr = self.localizer.localize(
            tdoa_ref_sec, corr_ref, self.min_corr_val, self.CI_percentage,
        )

        # Write temp pickle for later H5 merge
        if temp_dir is not None:
            pkl_data = {
                'detec_idx'      : detec_idx,
                'audio_file_name': detec['audio_file_name'],
                'waveforms'      : waveforms_to_save,
                'fs_waveforms'   : float(fs_raw),
                'cc_list'        : [cc.astype(np.float32) for cc in cc_list],
                'lags_list'      : [l.astype(np.int32)   for l  in lags_list],
                'fs_cc'          : float(fs),
                'tdoa_ref_sec'   : tdoa_ref_sec,
                'corr_ref'       : corr_ref,
                'ppd_slices'     : _extract_ppd_slices(PPD_xr, m) if loc_ok else None,
            }
            pkl_path = os.path.join(temp_dir, f'det_{detec_idx:06d}.pkl')
            with open(pkl_path, 'wb') as fh:
                pickle.dump(pkl_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            'detection_id'    : detec_idx,
            'time_min_offset' : detec['time_min_offset'],
            'x_m': m[0], 'y_m': m[1], 'z_m': m[2],
            'x_ci_low' : Px_CI[0], 'x_ci_high': Px_CI[1],
            'y_ci_low' : Py_CI[0], 'y_ci_high': Py_CI[1],
            'z_ci_low' : Pz_CI[0], 'z_ci_high': Pz_CI[1],
            'loc_ok'   : loc_ok,
            **{f'tdoa_{lbl}_ms': v * 1000
               for lbl, v in zip(self.ref_pair_labels, tdoa_ref_sec.ravel())},
            **{f'corr_{lbl}': v
               for lbl, v in zip(self.ref_pair_labels, corr_ref.ravel())},
        }

    # ------------------------------------------------------------------
    # Interactive single-detection entry point
    # ------------------------------------------------------------------

    def localize_detection(self, detec, audio_files, fs_raw):
        """Localize a single detection Series without plotting or saving.

        Parameters
        ----------
        detec : pandas.Series
        audio_files : dict  — ``{'path': [...], 'channel': [...]}``
        fs_raw : int

        Returns
        -------
        m, Px_CI, Py_CI, Pz_CI, loc_ok, PPD_xr
        """
        use_interpolation = (self.tdoa_precision == 'interpolate')
        waveform_stack = stack_waveforms(audio_files, detec, self.tdoa_max_sec)
        waveform_stack, _ = tighten_waveforms(
            waveform_stack, fs_raw, self.ref_channel,
            method                  = self.tighten_method,
            energy_window_perc      = self.energy_window_perc,
            peak_half_width_sec     = self.peak_window_half_width_sec,
            envelope_threshold_perc = self.envelope_threshold_perc,
            envelope_pad_sec        = self.envelope_pad_sec,
        )
        if not use_interpolation:
            waveform_stack, fs = upsample_stack(
                waveform_stack, fs_raw, self.upsample_res_sec
            )
        else:
            fs = fs_raw
        tdoa_ref_sec, corr_ref, _, _ = compute_tdoas(
            waveform_stack, self.ref_pairs, fs,
            self.tdoa_max_sec, method=self.tdoa_method,
            subsample_interpolation=use_interpolation,
        )
        return self.localizer.localize(
            tdoa_ref_sec, corr_ref, self.min_corr_val, self.CI_percentage,
        )

    # ------------------------------------------------------------------
    # HDF5 loader (for the review interface)
    # ------------------------------------------------------------------

    @staticmethod
    def load_h5(h5_path):
        """Load all data from an intermediate HDF5 file.

        Parameters
        ----------
        h5_path : str
            Path to a ``*_localizations.h5`` file produced by the pipeline.

        Returns
        -------
        dict with keys:

        ``'config'``
            Full localization config dict.
        ``'array'``
            Dict with ``hp_x``, ``hp_y``, ``hp_z``, ``channels``.
        ``'grid'``
            Dict with ``x_unique``, ``y_unique``, ``z_unique``.
        ``'annotations'``
            pandas DataFrame of original detections.
        ``'localizations'``
            pandas DataFrame of localization results.
        ``'intermediates'``
            List of dicts (one per detection), each containing:
            ``detection_id``, ``waveforms``, ``fs_waveforms``,
            ``cc_list``, ``lags_list``, ``fs_cc``,
            ``tdoa_ref_sec``, ``corr_ref``, ``ppd_slices``.
            ``ppd_slices`` is a dict with keys ``'xy'``, ``'xz'``, ``'yz'``
            (each a dict with ``'values'`` and coordinate arrays), or
            ``None`` when the detection was rejected.
        """
        import h5py as _h5py

        with _h5py.File(h5_path, 'r') as h5:
            config = json.loads(h5.attrs['config'])

            array = {
                'hp_x':     h5['array/hp_x'][:],
                'hp_y':     h5['array/hp_y'][:],
                'hp_z':     h5['array/hp_z'][:],
                'channels': h5['array/channels'][:],
            }
            grid = {
                'x_unique': h5['grid/x_unique'][:],
                'y_unique': h5['grid/y_unique'][:],
                'z_unique': h5['grid/z_unique'][:],
            }

            annotations_df   = pd.read_json(h5['annotations'][()].decode())
            localizations_df = pd.read_json(h5['localizations'][()].decode())

            intermediates = []
            if 'intermediates' in h5:
                for det_key in sorted(h5['intermediates'].keys()):
                    g  = h5[f'intermediates/{det_key}']
                    wg = g['waveforms']
                    cg = g['cc']
                    lg = g['lags']

                    ppd_slices = None
                    if 'ppd' in g:
                        ppd_slices = {}
                        for plane in ('xy', 'xz', 'yz'):
                            sg = g[f'ppd/{plane}']
                            ppd_slices[plane] = {
                                k: sg[k][:] for k in sg.keys()
                            }

                    intermediates.append({
                        'detection_id' : int(g.attrs['detection_id']),
                        'waveforms'    : [wg[f'ch_{i}'][:] for i in range(len(wg))],
                        'fs_waveforms' : float(wg.attrs['fs']),
                        'cc_list'      : [cg[f'pair_{i}'][:] for i in range(len(cg))],
                        'lags_list'    : [lg[f'pair_{i}'][:] for i in range(len(lg))],
                        'fs_cc'        : float(cg.attrs['fs']),
                        'tdoa_ref_sec' : g['tdoa_ref_sec'][:],
                        'corr_ref'     : g['corr_ref'][:],
                        'ppd_slices'   : ppd_slices,
                    })

        return {
            'config'        : config,
            'array'         : array,
            'grid'          : grid,
            'annotations'   : annotations_df,
            'localizations' : localizations_df,
            'intermediates' : intermediates,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_netcdf(self, localizations_df, detections, fs_raw, audio_file_stem):
        meas_cols = [
            'x_m', 'y_m', 'z_m',
            'x_err_low_m', 'x_err_high_m', 'x_err_span_m',
            'y_err_low_m', 'y_err_high_m', 'y_err_span_m',
            'z_err_low_m', 'z_err_high_m', 'z_err_span_m',
            'loc_ok', 'tdoa_sec', 'corr_val',
        ]
        loc_meas = Measurement(
            measurer_name='LocalizationPipeline',
            measurer_version='0.1',
            measurements_name=meas_cols,
            measurements_parameters=self.config,
        )
        merged = detections.data.reset_index(drop=True).copy()
        merged['audio_channel'] = (
            merged['audio_channel'].fillna(self.ref_channel).astype(int)
        )
        merged['audio_sampling_frequency'] = (
            merged['audio_sampling_frequency'].fillna(int(fs_raw)).astype(int)
        )
        merged['audio_bit_depth'] = merged['audio_bit_depth'].fillna(0).astype(int)

        ldf = localizations_df.reset_index(drop=True)
        tdoa_cols_ms = [c for c in ldf.columns if c.startswith('tdoa_') and c.endswith('_ms')]
        corr_cols    = [c for c in ldf.columns if c.startswith('corr_')]

        merged['x_m']          = ldf['x_m'].values
        merged['y_m']          = ldf['y_m'].values
        merged['z_m']          = ldf['z_m'].values
        merged['x_err_low_m']  = ldf['x_ci_low'].values
        merged['x_err_high_m'] = ldf['x_ci_high'].values
        merged['x_err_span_m'] = (ldf['x_ci_high'] - ldf['x_ci_low']).values
        merged['y_err_low_m']  = ldf['y_ci_low'].values
        merged['y_err_high_m'] = ldf['y_ci_high'].values
        merged['y_err_span_m'] = (ldf['y_ci_high'] - ldf['y_ci_low']).values
        merged['z_err_low_m']  = ldf['z_ci_low'].values
        merged['z_err_high_m'] = ldf['z_ci_high'].values
        merged['z_err_span_m'] = (ldf['z_ci_high'] - ldf['z_ci_low']).values
        merged['loc_ok']       = ldf['loc_ok'].values
        merged['tdoa_sec'] = [
            str(list(row.values / 1000.0)) for _, row in ldf[tdoa_cols_ms].iterrows()
        ]
        merged['corr_val'] = [
            str(list(row.values)) for _, row in ldf[corr_cols].iterrows()
        ]
        loc_meas.data = merged
        os.makedirs(self.out_dir, exist_ok=True)
        nc_path = os.path.join(self.out_dir, audio_file_stem + '_localizations.nc')
        loc_meas.to_netcdf(nc_path)
        print(f'Localizations saved to {nc_path}')

    def _show_save(self, fig, show, save_path):
        import matplotlib.pyplot as plt
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if not show:
            plt.close(fig)

    def _fig_path(self, stem, detec_idx, detec, plot_type):
        if self.out_dir is None:
            return None
        fname = (f'{stem}'
                 f'_det{detec_idx + 1:03d}'
                 f'_t{detec["time_min_offset"]:.3f}s'
                 f'_{plot_type}.png')
        return os.path.join(self.out_dir, fname)

    def _fig_path_summary(self, stem, plot_type):
        if self.out_dir is None:
            return None
        return os.path.join(self.out_dir, f'{stem}_{plot_type}.png')

    def __repr__(self):
        method = self.config.get('method', 'gridsearch')
        return (
            f"LocalizationPipeline("
            f"method={method!r}, "
            f"array={self.array.array_name!r}, "
            f"n_hydrophones={self.array.n}, "
            f"tdoa_precision={self.tdoa_precision!r})"
        )
