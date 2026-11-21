# -*- coding: utf-8 -*-
"""
CLI subcommands for HMD QA/QC.

Registered as the ``qaqc`` group under the top-level ``ecosound`` command.

Usage
-----
.. code-block:: bash

   # Run QA/QC on a folder of HMD NetCDF files
   ecosound qaqc run C:\\data\\deployment_01\\nc -o C:\\results\\deploy_01

   # Customise sensitivity and enable tidal detection
   ecosound qaqc run C:\\data\\deploy_01\\nc -s 3 --tidal -o C:\\results

   # Plot a previously-saved Measurement NetCDF
   ecosound qaqc plot C:\\results\\deploy_01\\qc_merged.nc
"""

from __future__ import annotations

import os
import sys

import click


@click.group()
def qaqc():
    """HMD QA/QC -- automated quality control for HMD spectral data."""


@qaqc.command()
@click.argument("hmd_folder", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", "output_dir", default=None,
              type=click.Path(file_okay=False),
              help="Output directory for results. Defaults to HMD_FOLDER.")
@click.option("-s", "--sensitivity", default=2, type=click.IntRange(1, 3),
              show_default=True,
              help="Global sensitivity (1=permissive, 2=balanced, 3=strict).")
@click.option("--level-shift-sensitivity", "ls_sensitivity", default=None,
              type=click.IntRange(1, 3),
              help="Override sensitivity for T5 level-shift only.")
@click.option("--min-duration", "min_duration", default=60.0,
              show_default=True,
              help="Minimum event duration in minutes.")
@click.option("--rolling-window", "rolling_window", default=6.0,
              show_default=True,
              help="Rolling window for T2 statistics (hours).")
@click.option("--no-t0", is_flag=True, help="Disable T0 gross range check.")
@click.option("--no-t1", is_flag=True, help="Disable T1 structural checks.")
@click.option("--no-t2", is_flag=True, help="Disable T2 statistical checks.")
@click.option("--no-t3", is_flag=True, help="Disable T3 PCA spectral check.")
@click.option("--tidal", is_flag=True,
              help="Enable T4 tidal flow-noise detection (off by default).")
@click.option("--no-t5", is_flag=True, help="Disable T5 level-shift check.")
@click.option("--db-range", nargs=2, type=float, default=(30, 115),
              show_default=True,
              help="dB colour range for the overview plot (min max).")
@click.option("--no-plot", is_flag=True,
              help="Skip generating the overview figure.")
@click.option("--csv", "export_csv", is_flag=True,
              help="Also export results as CSV (in addition to NetCDF).")
@click.option("--dask/--no-dask", default=False, show_default=True,
              help="Use Dask for lazy loading of HMD data.")
def run(hmd_folder, output_dir, sensitivity, ls_sensitivity, min_duration,
        rolling_window, no_t0, no_t1, no_t2, no_t3, tidal, no_t5,
        db_range, no_plot, export_csv, dask):
    """Run QA/QC on a folder of HMD NetCDF files.

    HMD_FOLDER is the path to the directory containing the HMD .nc files.

    Outputs are written to OUTPUT_DIR (or HMD_FOLDER if not specified):
      - qc_merged.nc       merged events (Measurement NetCDF)
      - qc_per_test.nc     per-test detections (Measurement NetCDF)
      - qc_overview.png    three-panel overview figure
      - qc_summary.txt     text summary

    \b
    Examples:
      ecosound qaqc run C:\\data\\deploy_01\\nc
      ecosound qaqc run ./nc_files -s 3 --tidal -o ./results
      ecosound qaqc run ./nc_files --no-t3 --no-t5 --csv
    """
    # Imports are deferred so --help is fast even without heavy dependencies
    from ecosound.soundscape.hmd import HMD
    from ecosound.soundscape.hmd_qaqc import HMDQualityChecker

    # Resolve output directory
    if output_dir is None:
        output_dir = hmd_folder
    os.makedirs(output_dir, exist_ok=True)

    # Load HMD data
    click.echo(f"Loading HMD data from: {hmd_folder}")
    hmd = HMD(use_dask=dask)
    hmd.load_nc_files(hmd_folder)

    # Run QA/QC
    qc = HMDQualityChecker(
        hmd,
        sensitivity=sensitivity,
        level_shift_sensitivity=ls_sensitivity,
        min_event_duration_min=min_duration,
        rolling_window_hours=rolling_window,
        run_t0=not no_t0,
        run_t1=not no_t1,
        run_t2=not no_t2,
        run_t3=not no_t3,
        detect_tidal_noise=tidal,
        run_t5=not no_t5,
    )
    qc.run()

    # Summary (to screen and file)
    qc.summary()
    summary_path = os.path.join(output_dir, "qc_summary.txt")
    _save_summary(qc, summary_path)

    # Export merged events
    merged_path = os.path.join(output_dir, "qc_merged.nc")
    meas_merged = qc.to_measurement(merged=True)
    meas_merged.to_netcdf(merged_path)
    click.echo(f"Merged events saved to: {merged_path}")

    # Export per-test detections
    per_test_path = os.path.join(output_dir, "qc_per_test.nc")
    meas_per_test = qc.to_measurement(merged=False)
    meas_per_test.to_netcdf(per_test_path)
    click.echo(f"Per-test detections saved to: {per_test_path}")

    # Optional CSV export
    if export_csv:
        csv_merged = os.path.join(output_dir, "qc_merged.csv")
        csv_per_test = os.path.join(output_dir, "qc_per_test.csv")
        qc.to_csv(csv_merged, merged=True)
        qc.to_csv(csv_per_test, merged=False)

    # Overview plot
    if not no_plot:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for CLI
        plot_path = os.path.join(output_dir, "qc_overview.png")
        qc.plot_overview(db_range=db_range, save_path=plot_path)

    click.echo(f"\nAll results written to: {output_dir}")


def _save_summary(qc, path):
    """Capture qc.summary() output to a text file."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        qc.summary()
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


@qaqc.command()
@click.argument("nc_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--db-range", nargs=2, type=float, default=(30, 115),
              show_default=True,
              help="dB colour range for the LTSA (min max).")
@click.option("-o", "--output", "save_path", default=None,
              type=click.Path(dir_okay=False),
              help="Save figure to this path instead of displaying.")
def plot(nc_file, db_range, save_path):
    """Plot a previously-saved QC Measurement NetCDF file.

    NC_FILE is the path to a qc_merged.nc or qc_per_test.nc file
    produced by ``ecosound qaqc run``.

    \b
    Examples:
      ecosound qaqc plot qc_merged.nc
      ecosound qaqc plot qc_merged.nc -o overview.png
    """
    from ecosound.core.measurement import Measurement

    click.echo(f"Loading QC results from: {nc_file}")
    meas = Measurement()
    meas.from_netcdf(nc_file)

    df = meas.data
    n_events = len(df)
    if n_events == 0:
        click.echo("No events in file.")
        return

    click.echo(f"Loaded {n_events} events")
    click.echo(f"Time range: {df['time_min_date'].min()} -> "
               f"{df['time_max_date'].max()}")

    # Print a brief summary table
    if "test_name" in df.columns:
        click.echo("\nBy test:")
        click.echo(df.groupby("test_name").size().to_string())
    if "issue_type" in df.columns:
        click.echo("\nBy issue type:")
        click.echo(df.groupby("issue_type").size().to_string())

    click.echo(f"\nUse the ecosound Python API for full plotting capabilities.")
    click.echo(f"See: qc.plot_overview() after loading with HMDQualityChecker.")
