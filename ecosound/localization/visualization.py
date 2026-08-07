"""
ecosound.localization.visualization
=====================================
Diagnostic and summary plot functions for the TDOA grid-search localization
pipeline.

All plot functions return the matplotlib Figure object so callers can display,
save, or further customize it.  None of the functions call plt.show() or
plt.close() — that is left to the caller (or to a helper such as
``_show_save`` in the calling script).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for projection='3d'
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.signal import hilbert


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_HP_COLORS = plt.cm.tab10.colors   # 10 RGBA colours (tuple), indexed by channel


def _hpc(ch):
    """Return the tab10 colour for hydrophone channel ch (int)."""
    return _HP_COLORS[int(ch) % 10]


# ---------------------------------------------------------------------------
# Title helper
# ---------------------------------------------------------------------------

def _det_title(detec_idx, detec):
    """One-line detection summary for use as plot title."""
    return (
        f"Detection {detec_idx + 1} | "
        f"t=[{detec['time_min_offset']:.3f}, {detec['time_max_offset']:.3f}] s  "
        f"f=[{detec['frequency_min']:.0f}, {detec['frequency_max']:.0f}] Hz"
    )


# ---------------------------------------------------------------------------
# Waveform plots
# ---------------------------------------------------------------------------

def plot_waveform_stack(waveform_stack, fs, hp_channels, ref_channel,
                        detec_idx, detec, title_suffix="", vlines_ms=None):
    """One subplot per hydrophone, stacked vertically.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    fs : float
    hp_channels : array-like of int
        Physical channel numbers (one per hydrophone).
    ref_channel : int
        Index of the reference hydrophone in waveform_stack (highlighted).
    detec_idx : int
    detec : pandas.Series
    title_suffix : str
    vlines_ms : list of float, optional
        Vertical lines to draw on every sub-panel (e.g. tightening limits).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_ch  = len(waveform_stack)
    t_ms  = np.arange(len(waveform_stack[0])) / fs * 1000
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]
    for ch_idx, (ax, wf) in enumerate(zip(axes, waveform_stack)):
        ax.plot(t_ms, wf, lw=0.6, color=_hpc(hp_channels[ch_idx]))
        if vlines_ms is not None:
            for v in vlines_ms:
                ax.axvline(v, color="red", lw=1.0, ls="--", alpha=0.8)
        ax.set_ylabel(f"CH{hp_channels[ch_idx]}", fontsize=8)
        ax.yaxis.set_tick_params(labelsize=7)
        if ch_idx == ref_channel:
            ax.set_facecolor("#f0f4ff")
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(f"{_det_title(detec_idx, detec)}  —  {title_suffix}")
    fig.tight_layout()
    return fig


def plot_waveforms_overlaid(waveform_stack, t_ms, hp_channels, ref_channel,
                             detec_idx, detec,
                             title_suffix="tightened waveforms"):
    """All channels normalised and overlaid on a single axes.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    t_ms : numpy.ndarray
        Time axis in milliseconds.
    hp_channels : array-like of int
    ref_channel : int
    detec_idx : int
    detec : pandas.Series
    title_suffix : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    for ch_idx, wf in enumerate(waveform_stack):
        ax.plot(
            t_ms,
            wf / (np.max(np.abs(wf)) or 1),
            color=_hpc(hp_channels[ch_idx]),
            lw=1.2 if ch_idx == ref_channel else 0.7,
            alpha=1.0 if ch_idx == ref_channel else 0.75,
            label=f"CH{hp_channels[ch_idx]}",
        )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Norm. amplitude")
    ax.legend(fontsize=7, ncol=len(waveform_stack))
    ax.set_title(f"{_det_title(detec_idx, detec)}  —  {title_suffix}")
    fig.tight_layout()
    return fig


def plot_tightening_comparison(waveform_stack, fs, ref_channel, hp_channels,
                                chunk_m1, chunk_m2, chunk_m3,
                                peak_half_width_sec, envelope_threshold_perc,
                                detec_idx, detec):
    """Reference-channel waveform + envelope with three tightening windows.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    fs : float
    ref_channel : int
    hp_channels : array-like of int
    chunk_m1 : [int, int]
        Sample indices for the cumulative-energy window.
    chunk_m2 : [int, int]
        Sample indices for the peak window.
    chunk_m3 : [int, int]
        Sample indices for the envelope window.
    peak_half_width_sec : float
    envelope_threshold_perc : float
    detec_idx : int
    detec : pandas.Series

    Returns
    -------
    matplotlib.figure.Figure
    """
    ref_wf   = waveform_stack[ref_channel]
    t_ms     = np.arange(len(ref_wf)) / fs * 1000
    envelope = np.abs(hilbert(ref_wf))

    fig, (ax_wf, ax_env) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax_wf.plot(t_ms,  ref_wf,  color="#444", lw=0.6, label="waveform")
    ax_env.plot(t_ms, envelope, color="#444", lw=0.8, label="envelope")
    ax_env.axhline(
        envelope.max() * envelope_threshold_perc / 100,
        color="C2", lw=0.8, ls=":",
        label=f"M3 threshold ({envelope_threshold_perc}%)",
    )

    for label, chunk, color in [
        ("M1 cumul. energy", chunk_m1, "C0"),
        (f"M2 peak ±{peak_half_width_sec * 1000:.0f} ms", chunk_m2, "C1"),
        (f"M3 envelope >{envelope_threshold_perc}%", chunk_m3, "C2"),
    ]:
        dur_ms = (chunk[1] - chunk[0]) / fs * 1000
        for ax in (ax_wf, ax_env):
            ax.axvline(chunk[0] / fs * 1000, color=color, lw=1.2, ls="--")
            ax.axvline(chunk[1] / fs * 1000, color=color, lw=1.2, ls="--",
                       label=f"{label}  ({dur_ms:.1f} ms)")

    ax_wf.set_ylabel("Amplitude")
    ax_env.set_ylabel("Envelope")
    ax_env.set_xlabel("Time (ms)")
    ax_wf.legend(fontsize=7, loc="upper right")
    ax_env.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"{_det_title(detec_idx, detec)}  —  "
        f"tightening comparison (ref CH{hp_channels[ref_channel]})"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TDOA cross-correlation plot
# ---------------------------------------------------------------------------

def plot_tdoa_pairs(waveform_stack, fs, pairs, pair_labels, hp_channels,
                    cc_list, lags_list, tdoa_sec_list, corr_list,
                    tdoa_max_sec, detec_idx, detec, method_name="GCC"):
    """One column per pair: normalised waveform overlay (top) + correlation (bottom).

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    fs : float
    pairs : list of [int, int]
    pair_labels : list of str
    hp_channels : array-like of int
    cc_list : list of numpy.ndarray
    lags_list : list of numpy.ndarray
    tdoa_sec_list : array-like of float
        TDOA in seconds for each pair.
    corr_list : array-like of float
        Peak correlation value for each pair.
    tdoa_max_sec : float
    detec_idx : int
    detec : pandas.Series
    method_name : str
        Label for the y-axis of the correlation panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_pairs = len(pairs)
    fig, axes = plt.subplots(2, n_pairs, figsize=(5 * n_pairs, 5), squeeze=False)

    for col, (pair, label, cc, lags, tdoa, corr) in enumerate(
        zip(pairs, pair_labels, cc_list, lags_list, tdoa_sec_list, corr_list)
    ):
        ch_i, ch_j = pair
        t_ms   = np.arange(len(waveform_stack[ch_i])) / fs * 1000
        lag_ms = lags / fs * 1000

        # row 0 — waveforms
        ax_wv = axes[0, col]
        ax_wv.plot(t_ms,
                   waveform_stack[ch_i] / (np.max(np.abs(waveform_stack[ch_i])) or 1),
                   color=_hpc(hp_channels[ch_i]), lw=0.7, label=f"CH{hp_channels[ch_i]}")
        ax_wv.plot(t_ms,
                   waveform_stack[ch_j] / (np.max(np.abs(waveform_stack[ch_j])) or 1),
                   color=_hpc(hp_channels[ch_j]), lw=0.7, alpha=0.7, label=f"CH{hp_channels[ch_j]}")
        ax_wv.set_xlabel("Time (ms)")
        ax_wv.set_ylabel("Norm. amplitude")
        ax_wv.legend(fontsize=7)
        ax_wv.set_title(label, fontsize=9)

        # row 1 — correlation function
        ax_cc = axes[1, col]
        peak_idx = int(np.argmin(np.abs(lags - round(tdoa * fs))))
        ax_cc.plot(lag_ms, cc, color="black", lw=0.8)
        ax_cc.axvline(lag_ms[peak_idx], color="tab:red", lw=1.2, ls="--")
        ax_cc.scatter([lag_ms[peak_idx]], [cc[peak_idx]], color="tab:red", zorder=5, s=40)
        ax_cc.text(lag_ms[peak_idx], cc[peak_idx],
                   f"  {tdoa * 1000:+.3f} ms\n  corr={corr:.3f}",
                   fontsize=7, va="top", color="tab:red")
        ax_cc.axvspan(-tdoa_max_sec * 1000, tdoa_max_sec * 1000,
                      alpha=0.08, color="green")
        ax_cc.set_xlabel("Lag (ms)")
        ax_cc.set_ylabel(method_name)
        cc_margin = 0.1 * max(cc.max() - cc.min(), 1e-6)
        ax_cc.set_ylim(cc.min() - cc_margin, cc.max() + cc_margin)

    fig.suptitle(f"{_det_title(detec_idx, detec)}  —  {method_name} cross-correlations")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-detection localisation plots
# ---------------------------------------------------------------------------

def plot_localization_3d(m, Px_CI, Py_CI, Pz_CI, loc_ok,
                          hp_x, hp_y, hp_z, hp_channels,
                          x_unique, y_unique, z_unique,
                          corr_ref, min_corr_val, detec_idx, detec):
    """3D scatter: hydrophones, MAP estimate, and 68% CI cross-hair.

    Parameters
    ----------
    m : array-like, shape (3,)
        MAP estimate (x, y, z).
    Px_CI, Py_CI, Pz_CI : [float, float]
        Credibility intervals.
    loc_ok : bool
    hp_x, hp_y, hp_z : array-like
    hp_channels : array-like of int
    x_unique, y_unique, z_unique : numpy.ndarray
        Grid axis values (used for axis limits and aspect ratio).
    corr_ref : array-like
    min_corr_val : float
    detec_idx : int
    detec : pandas.Series

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for i in range(len(hp_x)):
        ax.scatter([hp_x[i]], [hp_y[i]], [hp_z[i]], c=[_hpc(hp_channels[i])],
                   s=40, marker="o", depthshade=False, zorder=5,
                   label=f"CH{hp_channels[i]}")

    if loc_ok:
        ax.scatter([m[0]], [m[1]], [m[2]],
                   c="0.7", marker="*", s=150, label="Location", zorder=6,
                   edgecolors="k", linewidths=0.5)
        ax.plot([Px_CI[0], Px_CI[1]], [m[1], m[1]],    [m[2], m[2]],
                color="0.4", lw=1.0, label="68% CI")
        ax.plot([m[0], m[0]],    [Py_CI[0], Py_CI[1]], [m[2], m[2]],
                color="0.4", lw=1.0)
        ax.plot([m[0], m[0]],    [m[1], m[1]],    [Pz_CI[0], Pz_CI[1]],
                color="0.4", lw=1.0)
        ci_str = (f"68% CI: x±{(Px_CI[1] - Px_CI[0]) / 2:.3f}  "
                  f"y±{(Py_CI[1] - Py_CI[0]) / 2:.3f}  "
                  f"z±{(Pz_CI[1] - Pz_CI[0]) / 2:.3f} m")
        fig.suptitle(
            f"{_det_title(detec_idx, detec)}\n"
            f"x={m[0]:.3f}  y={m[1]:.3f}  z={m[2]:.3f} m  |  {ci_str}",
            fontsize=8,
        )
    else:
        fig.suptitle(
            f"{_det_title(detec_idx, detec)}\n"
            f"REJECTED  (min corr={np.min(corr_ref):.3f} < {min_corr_val})",
            fontsize=8, color="tab:red",
        )

    ax.set_xlim(x_unique[0], x_unique[-1])
    ax.set_ylim(y_unique[0], y_unique[-1])
    ax.set_zlim(z_unique[0], z_unique[-1])
    ax.set_box_aspect([x_unique[-1] - x_unique[0],
                       y_unique[-1] - y_unique[0],
                       z_unique[-1] - z_unique[0]])
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_localization_2d(m, Px_CI, Py_CI, Pz_CI, loc_ok,
                          hp_x, hp_y, hp_z, hp_channels,
                          x_unique, y_unique, z_unique,
                          detec_idx, detec, figsize=(15.1, 7.52)):
    """Orthogonal 2D projections of the MAP estimate and CI cross-hairs.

    Layout: XY (top view) left column; XZ and YZ stacked on the right.
    Panel sizes are solved analytically so that all panels share equal spatial
    scale (metres per inch) and their tops/bottoms align.

    Parameters
    ----------
    m : array-like, shape (3,)
    Px_CI, Py_CI, Pz_CI : [float, float]
    loc_ok : bool
    hp_x, hp_y, hp_z : array-like
    hp_channels : array-like of int
    x_unique, y_unique, z_unique : numpy.ndarray
    detec_idx : int
    detec : pandas.Series
    figsize : (float, float)

    Returns
    -------
    matplotlib.figure.Figure
    """
    x0, y0, z0 = m

    dx = x_unique[-1] - x_unique[0]
    dy = y_unique[-1] - y_unique[0]
    dz = z_unique[-1] - z_unique[0]

    m_left   = 0.60;  m_right  = 0.80
    m_bottom = 0.50;  m_top    = 0.60
    gap_row  = 0.55;  cb_gap   = 0.25;  cb_w = 0.20

    fig_w, fig_h = figsize

    ax_xy_h = fig_h - m_bottom - m_top
    ax_w_l  = ax_xy_h * dx / dy
    ax_w_r  = (ax_xy_h - gap_row) / (dz * (1.0 / dx + 1.0 / dy))
    ax_xz_h = ax_w_r * dz / dx
    ax_yz_h = ax_w_r * dz / dy
    gap_col = fig_w - m_left - ax_w_l - ax_w_r - cb_gap - cb_w - m_right

    fig = plt.figure(figsize=figsize)

    def _pos(l, b, w, h):
        return [l / fig_w, b / fig_h, w / fig_w, h / fig_h]

    col2  = m_left + ax_w_l + gap_col
    ax_xy = fig.add_axes(_pos(m_left, m_bottom,                      ax_w_l, ax_xy_h))
    ax_xz = fig.add_axes(_pos(col2,   m_bottom + ax_yz_h + gap_row,  ax_w_r, ax_xz_h))
    ax_yz = fig.add_axes(_pos(col2,   m_bottom,                      ax_w_r, ax_yz_h))

    px = 0.06 * dx;  xlim = [x_unique[0] - px, x_unique[-1] + px]
    py = 0.06 * dy;  ylim = [y_unique[0] - py, y_unique[-1] + py]
    pz = 0.06 * dz;  zlim = [z_unique[0] - pz, z_unique[-1] + pz]

    hp_kw  = dict(s=55, marker="o", zorder=5, edgecolors="white", linewidths=0.5)
    fs_ax  = 10;  fs_tk = 9;  fs_leg = 8
    ci_kw  = dict(color="0.4", lw=1.0, zorder=4)

    def _style(ax):
        ax.tick_params(labelsize=fs_tk)
        ax.grid(True, color="0.4", lw=0.6, alpha=0.6, zorder=3, linestyle="--")

    # ---- XY ----
    for i in range(len(hp_x)):
        ax_xy.scatter(hp_x[i], hp_y[i], color=_hpc(hp_channels[i]),
                      label=f"CH{hp_channels[i]}", **hp_kw)
    if loc_ok:
        ax_xy.scatter([x0], [y0], c="0.5", marker="*", s=180, zorder=6,
                      edgecolors="k", linewidths=0.5, label="MAP")
        ax_xy.plot([Px_CI[0], Px_CI[1]], [y0, y0], **ci_kw, label="68% CI")
        ax_xy.plot([x0, x0], [Py_CI[0], Py_CI[1]], **ci_kw)
    ax_xy.set_xlim(xlim);  ax_xy.set_ylim(ylim)
    ax_xy.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xy.set_ylabel("Y (m)", fontsize=fs_ax)
    ax_xy.set_title(f"XY  (z = {z0:.2f} m)", fontsize=fs_ax)
    ax_xy.legend(fontsize=fs_leg, framealpha=0.9, edgecolor="0.7", loc="upper left")
    _style(ax_xy)

    # ---- XZ ----
    for i in range(len(hp_x)):
        ax_xz.scatter(hp_x[i], hp_z[i], color=_hpc(hp_channels[i]), **hp_kw)
    if loc_ok:
        ax_xz.scatter([x0], [z0], c="0.5", marker="*", s=180, zorder=6,
                      edgecolors="k", linewidths=0.5)
        ax_xz.plot([Px_CI[0], Px_CI[1]], [z0, z0], **ci_kw)
        ax_xz.plot([x0, x0], [Pz_CI[0], Pz_CI[1]], **ci_kw)
    ax_xz.set_xlim(xlim);  ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_xz.set_title(f"XZ  (y = {y0:.2f} m)", fontsize=fs_ax)
    _style(ax_xz)

    # ---- YZ ----
    for i in range(len(hp_y)):
        ax_yz.scatter(hp_y[i], hp_z[i], color=_hpc(hp_channels[i]), **hp_kw)
    if loc_ok:
        ax_yz.scatter([y0], [z0], c="0.5", marker="*", s=180, zorder=6,
                      edgecolors="k", linewidths=0.5)
        ax_yz.plot([Py_CI[0], Py_CI[1]], [z0, z0], **ci_kw)
        ax_yz.plot([y0, y0], [Pz_CI[0], Pz_CI[1]], **ci_kw)
    ax_yz.set_xlim(ylim);  ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel("Y (m)", fontsize=fs_ax)
    ax_yz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_yz.set_title(f"YZ  (x = {x0:.2f} m)", fontsize=fs_ax)
    _style(ax_yz)

    if loc_ok:
        ci_str = (f"68% CI: x±{(Px_CI[1] - Px_CI[0]) / 2:.3f}  "
                  f"y±{(Py_CI[1] - Py_CI[0]) / 2:.3f}  "
                  f"z±{(Pz_CI[1] - Pz_CI[0]) / 2:.3f} m")
        title = (f"{_det_title(detec_idx, detec)}  —  "
                 f"x={x0:.3f}  y={y0:.3f}  z={z0:.3f} m  |  {ci_str}")
    else:
        title = f"{_det_title(detec_idx, detec)}  —  REJECTED"

    fig.text(0.5, (m_bottom + ax_xy_h + m_top * 0.55) / fig_h,
             title, ha="center", va="bottom", fontsize=10)
    return fig


def plot_ppd_slices(PPD_xr, m, hp_x, hp_y, hp_z, hp_channels,
                    detec_idx, detec, figsize=(15.1, 7.52)):
    """Three orthogonal PPD slices through the MAP estimate.

    Layout: XY left; XZ and YZ stacked on the right.  Panel sizes are solved
    analytically so every panel has equal spatial scale and tops/bottoms align.
    A single shared colorbar is placed on the far right.

    Parameters
    ----------
    PPD_xr : xarray.Dataset
        3-D PPD as returned by :meth:`~GridSearch.localize`.
    m : array-like, shape (3,)
    hp_x, hp_y, hp_z : array-like
    hp_channels : array-like of int
    detec_idx : int
    detec : pandas.Series
    figsize : (float, float)

    Returns
    -------
    matplotlib.figure.Figure
    """
    x0, y0, z0 = m

    if isinstance(PPD_xr, dict):
        # Pre-extracted slices loaded from HDF5
        xy_vals = PPD_xr['xy']['values']
        x_vals  = PPD_xr['xy']['x']
        y_vals  = PPD_xr['xy']['y']
        xz_vals = PPD_xr['xz']['values']
        z_vals  = PPD_xr['xz']['z']
        yz_vals = PPD_xr['yz']['values']
    else:
        # Full xarray PPD Dataset (normal pipeline output)
        sl_xy   = PPD_xr.PPD.sel(z=z0, method="nearest")
        sl_xz   = PPD_xr.PPD.sel(y=y0, method="nearest")
        sl_yz   = PPD_xr.PPD.sel(x=x0, method="nearest")
        xy_vals = sl_xy.values
        x_vals  = sl_xy["x"].values
        y_vals  = sl_xy["y"].values
        xz_vals = sl_xz.values
        z_vals  = sl_xz["z"].values
        yz_vals = sl_yz.values

    vmin = min(xy_vals.min(), xz_vals.min(), yz_vals.min())
    vmax = max(xy_vals.max(), xz_vals.max(), yz_vals.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    dx = x_vals[-1] - x_vals[0]
    dy = y_vals[-1] - y_vals[0]
    dz = z_vals[-1] - z_vals[0]

    m_left   = 0.60;  m_right  = 0.80
    m_bottom = 0.50;  m_top    = 0.60
    gap_row  = 0.55;  cb_gap   = 0.25;  cb_w = 0.20

    fig_w, fig_h = figsize

    ax_xy_h = fig_h - m_bottom - m_top
    ax_w_l  = ax_xy_h * dx / dy
    ax_w_r  = (ax_xy_h - gap_row) / (dz * (1.0 / dx + 1.0 / dy))
    ax_xz_h = ax_w_r * dz / dx
    ax_yz_h = ax_w_r * dz / dy
    gap_col = fig_w - m_left - ax_w_l - ax_w_r - cb_gap - cb_w - m_right

    fig = plt.figure(figsize=figsize)

    def _pos(l, b, w, h):
        return [l / fig_w, b / fig_h, w / fig_w, h / fig_h]

    col2  = m_left + ax_w_l + gap_col
    ax_xy = fig.add_axes(_pos(m_left, m_bottom,                      ax_w_l, ax_xy_h))
    ax_xz = fig.add_axes(_pos(col2,   m_bottom + ax_yz_h + gap_row,  ax_w_r, ax_xz_h))
    ax_yz = fig.add_axes(_pos(col2,   m_bottom,                      ax_w_r, ax_yz_h))
    ax_cb = fig.add_axes(_pos(col2 + ax_w_r + cb_gap, m_bottom,      cb_w,   ax_xy_h))

    px = 0.06 * dx;  xlim = [x_vals[0] - px, x_vals[-1] + px]
    py = 0.06 * dy;  ylim = [y_vals[0] - py, y_vals[-1] + py]
    pz = 0.06 * dz;  zlim = [z_vals[0] - pz, z_vals[-1] + pz]

    hp_kw  = dict(s=55, marker="o", zorder=5, edgecolors="white", linewidths=0.5)
    map_kw = dict(c="white", marker="*", s=80, zorder=6, edgecolors="0.4", linewidths=0.4)
    fs_ax  = 10;  fs_tk = 9;  fs_leg = 8

    def _style(ax):
        ax.tick_params(labelsize=fs_tk)
        ax.grid(True, color="0.4", lw=0.6, alpha=0.6, zorder=4, linestyle="--")

    # ---- XY ----
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    im = ax_xy.pcolormesh(X, Y, xy_vals, cmap="hot_r", shading="auto", norm=norm)
    for i in range(len(hp_x)):
        ax_xy.scatter(hp_x[i], hp_y[i], color=_hpc(hp_channels[i]),
                      label=f"CH{hp_channels[i]}", **hp_kw)
    ax_xy.scatter([x0], [y0], label="Localized source (MAP)", **map_kw)
    ax_xy.set_xlim(xlim);  ax_xy.set_ylim(ylim)
    ax_xy.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xy.set_ylabel("Y (m)", fontsize=fs_ax)
    ax_xy.set_title(f"XY  (z = {z0:.2f} m)", fontsize=fs_ax)
    ax_xy.legend(fontsize=fs_leg, framealpha=0.9, edgecolor="0.7", loc="upper left")
    _style(ax_xy)

    # ---- XZ ----
    X, Z = np.meshgrid(x_vals, z_vals, indexing="ij")
    ax_xz.pcolormesh(X, Z, xz_vals, cmap="hot_r", shading="auto", norm=norm)
    for i in range(len(hp_x)):
        ax_xz.scatter(hp_x[i], hp_z[i], color=_hpc(hp_channels[i]), **hp_kw)
    ax_xz.scatter([x0], [z0], **map_kw)
    ax_xz.set_xlim(xlim);  ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_xz.set_title(f"XZ  (y = {y0:.2f} m)", fontsize=fs_ax)
    _style(ax_xz)

    # ---- YZ ----
    Y, Z = np.meshgrid(y_vals, z_vals, indexing="ij")
    ax_yz.pcolormesh(Y, Z, yz_vals, cmap="hot_r", shading="auto", norm=norm)
    for i in range(len(hp_y)):
        ax_yz.scatter(hp_y[i], hp_z[i], color=_hpc(hp_channels[i]), **hp_kw)
    ax_yz.scatter([y0], [z0], **map_kw)
    ax_yz.set_xlim(ylim);  ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel("Y (m)", fontsize=fs_ax)
    ax_yz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_yz.set_title(f"YZ  (x = {x0:.2f} m)", fontsize=fs_ax)
    _style(ax_yz)

    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("PPD", fontsize=fs_ax)
    cbar.ax.tick_params(labelsize=fs_tk)

    fig.text(0.5, (m_bottom + ax_xy_h + m_top * 0.55) / fig_h,
             f"{_det_title(detec_idx, detec)}  —  PPD slices through MAP estimate",
             ha="center", va="bottom", fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Summary plots (all detections)
# ---------------------------------------------------------------------------

def plot_all_localizations_2d(localizations_df, hp_x, hp_y, hp_z, hp_channels,
                               x_unique, y_unique, z_unique, figsize=(15.1, 7.52)):
    """Orthogonal 2D projections of all accepted localizations.

    Localizations are coloured by time_min_offset.  A time colorbar is placed
    on the far right.  Hydrophones are shown in light grey with a black outline.

    Parameters
    ----------
    localizations_df : pandas.DataFrame
        Must contain columns: loc_ok, x_m, y_m, z_m, time_min_offset,
        x_ci_low/high, y_ci_low/high, z_ci_low/high.
    hp_x, hp_y, hp_z : array-like
    hp_channels : array-like of int
    x_unique, y_unique, z_unique : numpy.ndarray
    figsize : (float, float)

    Returns
    -------
    matplotlib.figure.Figure
    """
    ok = localizations_df[localizations_df["loc_ok"]]

    dx = x_unique[-1] - x_unique[0]
    dy = y_unique[-1] - y_unique[0]
    dz = z_unique[-1] - z_unique[0]

    m_left   = 0.60;  m_right  = 0.80
    m_bottom = 0.50;  m_top    = 0.60
    gap_row  = 0.55;  cb_gap   = 0.25;  cb_w = 0.20

    fig_w, fig_h = figsize

    ax_xy_h = fig_h - m_bottom - m_top
    ax_w_l  = ax_xy_h * dx / dy
    ax_w_r  = (ax_xy_h - gap_row) / (dz * (1.0 / dx + 1.0 / dy))
    ax_xz_h = ax_w_r * dz / dx
    ax_yz_h = ax_w_r * dz / dy
    gap_col = fig_w - m_left - ax_w_l - ax_w_r - cb_gap - cb_w - m_right

    fig = plt.figure(figsize=figsize)

    def _pos(l, b, w, h):
        return [l / fig_w, b / fig_h, w / fig_w, h / fig_h]

    col2  = m_left + ax_w_l + gap_col
    ax_xy = fig.add_axes(_pos(m_left, m_bottom,                      ax_w_l, ax_xy_h))
    ax_xz = fig.add_axes(_pos(col2,   m_bottom + ax_yz_h + gap_row,  ax_w_r, ax_xz_h))
    ax_yz = fig.add_axes(_pos(col2,   m_bottom,                      ax_w_r, ax_yz_h))
    ax_cb = fig.add_axes(_pos(col2 + ax_w_r + cb_gap, m_bottom,      cb_w,   ax_xy_h))

    px = 0.06 * dx;  xlim = [x_unique[0] - px, x_unique[-1] + px]
    py = 0.06 * dy;  ylim = [y_unique[0] - py, y_unique[-1] + py]
    pz = 0.06 * dz;  zlim = [z_unique[0] - pz, z_unique[-1] + pz]

    hp_kw  = dict(s=55, marker="o", zorder=5, color="0.85", edgecolors="k", linewidths=0.6)
    fs_ax  = 10;  fs_tk = 9;  fs_leg = 8

    def _style(ax):
        ax.tick_params(labelsize=fs_tk)
        ax.grid(True, color="0.4", lw=0.6, alpha=0.6, zorder=3, linestyle="--")

    cmap = plt.cm.viridis
    if len(ok):
        t_vals = ok["time_min_offset"].values
        norm_t = plt.Normalize(vmin=t_vals.min(), vmax=t_vals.max())
    else:
        norm_t = plt.Normalize(vmin=0, vmax=1)

    def _tcol(t):
        return cmap(norm_t(t))

    for i in range(len(hp_x)):
        ax_xy.scatter(hp_x[i], hp_y[i], **hp_kw)
        ax_xz.scatter(hp_x[i], hp_z[i], **hp_kw)
        ax_yz.scatter(hp_y[i], hp_z[i], **hp_kw)

    for _, row in ok.iterrows():
        col   = _tcol(row["time_min_offset"])
        ci_kw = dict(color=col, lw=0.8, alpha=0.8, zorder=4)
        mk_kw = dict(s=55, marker="o", zorder=6, color=col, alpha=0.7,
                     edgecolors="k", linewidths=0.6)
        x0, y0, z0 = row["x_m"], row["y_m"], row["z_m"]

        ax_xy.scatter([x0], [y0], **mk_kw)
        ax_xy.plot([row["x_ci_low"], row["x_ci_high"]], [y0, y0], **ci_kw)
        ax_xy.plot([x0, x0], [row["y_ci_low"], row["y_ci_high"]], **ci_kw)

        ax_xz.scatter([x0], [z0], **mk_kw)
        ax_xz.plot([row["x_ci_low"], row["x_ci_high"]], [z0, z0], **ci_kw)
        ax_xz.plot([x0, x0], [row["z_ci_low"], row["z_ci_high"]], **ci_kw)

        ax_yz.scatter([y0], [z0], **mk_kw)
        ax_yz.plot([row["y_ci_low"], row["y_ci_high"]], [z0, z0], **ci_kw)
        ax_yz.plot([y0, y0], [row["z_ci_low"], row["z_ci_high"]], **ci_kw)

    hp_proxy = Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                      mfc="0.85", mec="k", mew=0.6, label="Hydrophones")
    loc_dot  = Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                      mfc=cmap(0.5), mec="k", mew=0.6)
    ci_cross = Line2D([0], [0], marker="+", linestyle="none", markersize=9,
                      color=cmap(0.5), mew=1.2)

    ax_xy.set_xlim(xlim);  ax_xy.set_ylim(ylim)
    ax_xy.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xy.set_ylabel("Y (m)", fontsize=fs_ax)
    ax_xy.set_title("XY", fontsize=fs_ax)
    ax_xy.legend(
        handles=[hp_proxy, (loc_dot, ci_cross)],
        labels=["Hydrophones", "Localizations + CI"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=fs_leg, framealpha=0.9, edgecolor="0.7", loc="upper left",
    )
    _style(ax_xy)

    ax_xz.set_xlim(xlim);  ax_xz.set_ylim(zlim)
    ax_xz.set_xlabel("X (m)", fontsize=fs_ax)
    ax_xz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_xz.set_title("XZ", fontsize=fs_ax)
    _style(ax_xz)

    ax_yz.set_xlim(ylim);  ax_yz.set_ylim(zlim)
    ax_yz.set_xlabel("Y (m)", fontsize=fs_ax)
    ax_yz.set_ylabel("Z (m)", fontsize=fs_ax)
    ax_yz.set_title("YZ", fontsize=fs_ax)
    _style(ax_yz)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_t)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Time (s)", fontsize=fs_ax)
    cbar.ax.tick_params(labelsize=fs_tk)

    fig.text(0.5, (m_bottom + ax_xy_h + m_top * 0.55) / fig_h,
             f"All localizations  —  {len(ok)}/{len(localizations_df)} accepted",
             ha="center", va="bottom", fontsize=11)
    return fig


def plot_all_localizations_3d(localizations_df, hp_x, hp_y, hp_z, hp_channels,
                               x_unique, y_unique, z_unique):
    """3D scatter of all accepted localizations over the hydrophone array.

    Parameters
    ----------
    localizations_df : pandas.DataFrame
    hp_x, hp_y, hp_z : array-like
    hp_channels : array-like of int
    x_unique, y_unique, z_unique : numpy.ndarray

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(hp_x, hp_y, hp_z, c="0.85", s=40, marker="o",
               depthshade=False, zorder=5, label="Hydrophones",
               edgecolors="k", linewidths=0.6)

    ok   = localizations_df[localizations_df["loc_ok"]]
    cmap = plt.cm.viridis
    if len(ok):
        t_vals = ok["time_min_offset"].values
        norm_t = plt.Normalize(vmin=t_vals.min(), vmax=t_vals.max())
    else:
        norm_t = plt.Normalize(vmin=0, vmax=1)

    if len(ok):
        cols = [cmap(norm_t(t)) for t in ok["time_min_offset"].values]
        ax.scatter(ok["x_m"], ok["y_m"], ok["z_m"],
                   c=cols, marker="o", s=55, zorder=6, alpha=0.7,
                   depthshade=False, edgecolors="k", linewidths=0.6,
                   label=f"Localized (n={len(ok)})")
        for (_, row), col in zip(ok.iterrows(), cols):
            x0, y0, z0 = row["x_m"], row["y_m"], row["z_m"]
            ci_kw = dict(color=col, lw=0.8, alpha=0.6, zorder=5)
            ax.plot([row["x_ci_low"], row["x_ci_high"]], [y0, y0], [z0, z0], **ci_kw)
            ax.plot([x0, x0], [row["y_ci_low"], row["y_ci_high"]], [z0, z0], **ci_kw)
            ax.plot([x0, x0], [y0, y0], [row["z_ci_low"], row["z_ci_high"]], **ci_kw)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_t)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Time (s)", shrink=0.6, pad=0.1)

    ax.set_xlim(x_unique[0], x_unique[-1])
    ax.set_ylim(y_unique[0], y_unique[-1])
    ax.set_zlim(z_unique[0], z_unique[-1])
    ax.set_box_aspect([x_unique[-1] - x_unique[0],
                       y_unique[-1] - y_unique[0],
                       z_unique[-1] - z_unique[0]])
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title(f"All localizations  —  {len(ok)}/{len(localizations_df)} accepted",
                 fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    return fig
