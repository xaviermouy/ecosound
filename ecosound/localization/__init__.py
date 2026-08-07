"""
ecosound.localization
=====================
3-D source localization from hydrophone array TDOA measurements.

Public API
----------
Classes:
    SensorArray           — YAML-based hardware array description (hydrophones, recorders, cameras)
    Deployment            — YAML-based per-deployment context (file paths, location, orientation)
    DeployedArray         — one array within a deployment (hardware + deployment context combined)
    GridSearch            — pre-computed TDOA grid and grid-search localization
    LocalizationPipeline  — end-to-end pipeline (config → waveforms → TDOAs → localization)

TDOA estimation:
    gcc             — plain normalised cross-correlation
    gcc_phat        — GCC-PHAT (cross-spectrum whitening)
    compute_tdoas   — batch TDOA estimation for all hydrophone pairs

Preprocessing:
    stack_waveforms    — load and bandpass-filter multi-channel waveforms
    tighten_waveforms  — crop to a tight signal window on the reference channel
    upsample_stack     — upsample all waveforms for sub-sample TDOA precision

Visualization (diagnostic / summary plots):
    plot_waveform_stack
    plot_waveforms_overlaid
    plot_tightening_comparison
    plot_tdoa_pairs
    plot_localization_3d
    plot_localization_2d
    plot_ppd_slices
    plot_all_localizations_2d
    plot_all_localizations_3d
"""

from .array import SensorArray
from .deployment import Deployment, DeployedArray
from .gridsearch import GridSearch
from .pipeline import LocalizationPipeline, _extract_ppd_slices
from .tdoa import gcc, gcc_phat, compute_tdoas
from .preprocessing import stack_waveforms, tighten_waveforms, upsample_stack
from .visualization import (
    plot_waveform_stack,
    plot_waveforms_overlaid,
    plot_tightening_comparison,
    plot_tdoa_pairs,
    plot_localization_3d,
    plot_localization_2d,
    plot_ppd_slices,
    plot_all_localizations_2d,
    plot_all_localizations_3d,
)

__all__ = [
    "SensorArray",
    "Deployment",
    "DeployedArray",
    "GridSearch",
    "LocalizationPipeline",
    "gcc",
    "gcc_phat",
    "compute_tdoas",
    "stack_waveforms",
    "tighten_waveforms",
    "upsample_stack",
    "plot_waveform_stack",
    "plot_waveforms_overlaid",
    "plot_tightening_comparison",
    "plot_tdoa_pairs",
    "plot_localization_3d",
    "plot_localization_2d",
    "plot_ppd_slices",
    "plot_all_localizations_2d",
    "plot_all_localizations_3d",
]
