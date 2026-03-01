"""
HMD (Hybrid Millidecade) Class for Acoustic Data Analysis

"""

import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from dask.distributed import Client, LocalCluster
import dask
from pathlib import Path


class HMD:
    """
    Hybrid Millidecade acoustic data processor for NetCDF spectral files.

    Parameters
    ----------
    n_workers : int, optional
        Number of Dask workers (default: 4)
    memory_per_worker : str, optional
        Memory limit per worker (default: '4GB')
    temp_directory : str, optional
        Directory for Dask temporary files (default: system temp)
    use_dask : bool, optional
        Enable Dask parallel processing (default: True)
    use_processes : bool, optional
        Use processes instead of threads (default: auto-detect, False on Windows)

    Examples
    --------
    >>> hmd = HMD(n_workers=4)
    >>> hmd.load_nc_files('path/to/deployment_01')
    >>> result = hmd.extract_band_levels([[50, 300]], ['ship'])
    >>> hmd.summary()
    """

    def __init__(self, n_workers=4, memory_per_worker='4GB',
                 temp_directory=None, use_dask=True, use_processes=None):
        """Initialize HMD processor with optional Dask cluster"""
        import platform
        import tempfile

        self.n_workers = n_workers
        self.memory_per_worker = memory_per_worker
        self.temp_directory = temp_directory or tempfile.gettempdir()
        self.use_dask = use_dask
        self.client = None
        self.ds = None

        if use_processes is None:
            use_processes = platform.system() != 'Windows'

        self.use_processes = use_processes

        if self.use_dask:
            self._setup_dask()
        else:
            print("Running without Dask (single-threaded mode)")



    def load_nc_files(self, path, freq_range=None, time_range=None,
                      recursive=False, chunks=None, prefilter_by_date=False,
                      filename_pattern=None):
        """
        Load NetCDF files from a directory.

        Parameters
        ----------
        path : str
            Path to directory containing .nc files
        freq_range : tuple, optional
            Frequency range in Hz (min_freq, max_freq)
        time_range : tuple, optional
            Time range as strings ('2021-06-01', '2021-06-30').
            Start is inclusive, end is exclusive: [start, end)
        recursive : bool, optional
            If True, search subdirectories recursively (default: False)
        chunks : dict, optional
            Custom chunk sizes (default: {'time': 1440, 'frequency': 500})
        prefilter_by_date : bool, optional
            Filter files by date before loading (default: False)
        filename_pattern : str, optional
            Regex pattern to filter files by name. Only files matching this pattern
            will be loaded. Examples:
            - r'_\d{8}\.nc$' : Files ending with _YYYYMMDD.nc (e.g., data_20201121.nc)
            - r'^deployment_.*\.nc$' : Files starting with 'deployment_'
            - r'.*_HMD_.*\.nc$' : Files containing '_HMD_'
            Default: None (load all .nc files)

        Returns
        -------
        self

        Examples
        --------
        >>> # Load all .nc files
        >>> hmd.load_nc_files('data/deployment_01')
        >>>
        >>> # Load only files ending with date pattern _YYYYMMDD.nc
        >>> hmd.load_nc_files('data/',
        ...                   recursive=True,
        ...                   filename_pattern=r'_\d{8}\.nc$')
        >>>
        >>> # Load only files with specific prefix
        >>> hmd.load_nc_files('data/',
        ...                   filename_pattern=r'^HMD_.*\.nc$')
        """
        if chunks is None:
            chunks = {'time': 1440, 'frequency': 500}

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        # Find all .nc files
        if recursive:
            all_files = sorted(list(path.rglob('*.nc')))
        else:
            all_files = sorted(list(path.glob('*.nc')))

        if len(all_files) == 0:
            search_type = "or subdirectories" if recursive else ""
            raise FileNotFoundError(f"No .nc files found in {path} {search_type}")

        print(f"Found {len(all_files)} .nc files")

        # Filter files by filename pattern if requested
        if filename_pattern is not None:
            import re
            initial_count = len(all_files)
            pattern = re.compile(filename_pattern)
            all_files = [f for f in all_files if pattern.search(f.name)]
            filtered_count = initial_count - len(all_files)

            if filtered_count > 0:
                print(f"Filtered by filename pattern '{filename_pattern}': "
                      f"{len(all_files)} files (excluded {filtered_count})")

            if len(all_files) == 0:
                raise FileNotFoundError(f"No files match pattern '{filename_pattern}'")

        # Filter files by date if requested
        if prefilter_by_date and time_range is not None:
            all_files, skipped = self._filter_files_by_date(all_files, time_range)
            print(f"Pre-filtered to {len(all_files)} files (skipped {skipped} outside date range)")

        if len(all_files) == 0:
            raise FileNotFoundError(f"No files found within time range {time_range}")

        # Group files by parent directory
        files_by_dir = {}
        for file in all_files:
            parent = file.parent
            if parent not in files_by_dir:
                files_by_dir[parent] = []
            files_by_dir[parent].append(file)

        # Show what we found
        if len(files_by_dir) > 1:
            print(f"Files organized in {len(files_by_dir)} directories:")
            for dir_path, files in sorted(files_by_dir.items()):
                try:
                    rel_path = dir_path.relative_to(path)
                    dir_label = str(rel_path) if str(rel_path) != '.' else "(base)"
                except ValueError:
                    dir_label = dir_path.name
                print(f"  - {dir_label}: {len(files)} files")

        print(f"\nLoading {len(all_files)} files...")

        # Create mapping of file to deployment name
        file_to_deployment = {}
        for dir_path, files in files_by_dir.items():
            if len(files_by_dir) == 1:
                deployment_name = dir_path.name
            else:
                try:
                    rel_path = dir_path.relative_to(path)
                    deployment_name = str(rel_path).replace('\\', '/') if str(rel_path) != '.' else path.name
                except ValueError:
                    deployment_name = dir_path.name

            for file in files:
                file_to_deployment[str(file)] = deployment_name

        # Preprocessing function
        def preprocess(ds):
            source_file = ds.encoding.get('source', '')
            deployment_name = file_to_deployment.get(source_file, 'unknown')

            if freq_range is not None:
                ds = ds.sel(frequency=slice(freq_range[0], freq_range[1]))

            ds = ds.assign_coords({'deployment': deployment_name})
            return ds

        # Load all files at once
        self.ds = xr.open_mfdataset(
            [str(f) for f in all_files],
            engine='h5netcdf',
            chunks=chunks if self.use_dask else None,
            preprocess=preprocess,
            parallel=True if self.use_dask else False,
            combine='by_coords',
            decode_timedelta=True,
            data_vars='minimal',
            coords='minimal',
            compat='override',
            lock=False if self.use_processes else None,
        )

        # Subset time range if specified (end exclusive)
        if time_range is not None:
            print("Subsetting time range...")
            import pandas as pd
            # Convert end time and subtract one microsecond to make it exclusive
            end_time = pd.Timestamp(time_range[1]) - pd.Timedelta(microseconds=1)
            self.ds = self.ds.sel(time=slice(time_range[0], end_time))

        print(f"✓ Loaded dataset: {len(self.ds.time)} time points, "
              f"{len(self.ds.frequency)} frequency bins")

        if 'deployment' in self.ds.dims:
            print(f"  Deployments: {list(self.ds.deployment.values)}")

        # Provide chunking advice if chunks don't align well
        if self.ds.chunks:
            freq_chunks = self.ds.chunks.get('frequency', [])
            if len(freq_chunks) > 1:
                print("\n  ℹ Note: Data has multiple frequency chunks which may cause warnings")
                print("  For better performance when extracting band levels, consider:")
                print("    hmd.rechunk({'time': 1440, 'frequency': -1})")

        return self

    def extract_band_levels(self, freq_bands, band_names=None, persist=True):
        """
        Extract time series at specific frequencies or frequency bands.

        Parameters
        ----------
        freq_bands : list of lists/tuples
            List of frequency specifications. Each element can be:
            - [freq]: Single frequency point (e.g., [100] for 100 Hz)
            - [freq_min, freq_max]: Frequency band (e.g., [50, 300] for 50-300 Hz)
        band_names : list of str, optional
            Names for each band. If None, auto-generated as 'band_0', 'band_1', etc.
        persist : bool
            If True, keep result in distributed memory

        Returns
        -------
        xarray.Dataset
            Dataset with time series for each band/frequency

        Examples
        --------
        >>> # Single frequencies
        >>> bands = [[100], [500], [1000]]
        >>> result = hmd.extract_band_levels(bands)

        >>> # Frequency bands
        >>> bands = [[50, 300], [500, 2000], [2000, 10000]]
        >>> names = ['ship', 'fish', 'mammal']
        >>> result = hmd.extract_band_levels(bands, band_names=names)

        >>> # Mixed: single frequencies and bands
        >>> bands = [[100], [50, 300], [1000], [2000, 10000]]
        >>> names = ['100Hz', 'ship', '1000Hz', 'mammal']
        >>> result = hmd.extract_band_levels(bands, band_names=names)

        >>> # Access results
        >>> ship_noise = result['ship']
        >>> ts_100hz = result['100Hz']
        """
        self._check_loaded()

        # Auto-generate band names if not provided
        if band_names is None:
            band_names = [f'band_{i}' for i in range(len(freq_bands))]

        if len(band_names) != len(freq_bands):
            raise ValueError(f"Number of band_names ({len(band_names)}) must match "
                             f"number of freq_bands ({len(freq_bands)})")

        results = {}

        for band_name, freq_spec in zip(band_names, freq_bands):
            if len(freq_spec) == 1:
                # Single frequency point
                freq = freq_spec[0]
                timeseries = self.ds.psd.sel(frequency=freq, method='nearest')
                # Drop the frequency coordinate to avoid conflicts when combining
                timeseries = timeseries.drop_vars('frequency', errors='ignore')
                print(f"  {band_name}: {freq} Hz (single frequency)")

            elif len(freq_spec) == 2:
                # Frequency band - integrate properly
                freq_min, freq_max = freq_spec
                band = self.ds.sel(frequency=slice(freq_min, freq_max))
                timeseries = self._integrate_band_db(band.psd)
                print(f"  {band_name}: {freq_min}-{freq_max} Hz (integrated band)")

            else:
                raise ValueError(f"Each freq_spec must have 1 or 2 elements, "
                                 f"got {len(freq_spec)} for {band_name}")

            results[band_name] = timeseries

        # Combine into a dataset
        result_ds = xr.Dataset(results)

        if persist:
            return result_ds.persist()
        else:
            return result_ds

    def compute_timeseries_stats(self, data, percentiles=[10, 50, 90], resolution='1h'):
        """
        Compute statistics (mean and percentiles) on acoustic data in dB.
        Optimized for parallel computation with Dask. Robust to NaN values.
        """
        import xarray as xr
        import dask.array as da
        import numpy as np
        import warnings

        # Determine data type and structure
        if isinstance(data, xr.DataArray):
            process_data = {'data': data}
            is_dataset = False
        elif isinstance(data, xr.Dataset):
            process_data = {var: data[var] for var in data.data_vars}
            is_dataset = True
        else:
            raise ValueError("data must be xarray.DataArray or Dataset")

        # Check dimensions
        first_var = list(process_data.values())[0]
        has_time = 'time' in first_var.dims
        has_freq = 'frequency' in first_var.dims

        if not (has_time and not has_freq):
            raise ValueError("Data must only have a 'time' dimension (time series)")

        print(f"Computing time series statistics at {resolution} resolution...")

        results = {}

        # Process all variables
        for name, arr in process_data.items():
            print(f"  Processing {name}...")

            # Check if data is already computed (not lazy)
            # If so, convert back to Dask array to avoid large graph warning
            if not hasattr(arr.data, 'dask'):
                # Data is already computed (numpy array)
                # Re-chunk it as a Dask array with optimal chunks
                print(f"    Converting computed data back to Dask array...")
                if resolution in ['1h', '1H']:
                    time_chunk = 24 * 60  # 1 day worth of minute data
                elif resolution in ['1D', '1d']:
                    time_chunk = 365  # 1 year worth of daily data
                else:
                    # Try to estimate good chunk size
                    time_chunk = min(10000, len(arr.time))

                arr = arr.chunk({'time': time_chunk})
            else:
                # Data is already lazy (Dask array)
                # Rechunk to optimal size for resampling
                if resolution in ['1h', '1H']:
                    time_chunk = 24 * 60  # 1 day worth of minute data
                elif resolution in ['1D', '1d']:
                    time_chunk = 30 * 1440  # 1 month worth of minute data
                else:
                    time_chunk = 10000  # Default large chunk

                arr = arr.chunk({'time': time_chunk})

            # Create resampler
            resampler = arr.resample(time=resolution,label='left')

            # Compute all statistics in parallel
            stats_dict = {}

            # Count valid observations per bin
            count = resampler.count()

            # For dB mean, convert to linear → resample → convert back
            linear = 10.0 ** (arr / 10.0)
            linear_mean = linear.resample(time=resolution,label='left').mean(skipna=True)

            # Safe log10 that handles zero/negative values and NaNs
            def safe_log10(x):
                """Safely compute log10, returning NaN for invalid values"""
                result = np.where((x > 0) & np.isfinite(x), np.log10(x), np.nan)
                return result

            stats_dict['mean'] = 10 * xr.apply_ufunc(
                safe_log10,
                linear_mean,
                dask='parallelized',
                output_dtypes=[float]
            )

            # Suppress RuntimeWarnings from flox during std calculation with NaNs
            #with warnings.catch_warnings():
            #    warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            #    stats_dict['std'] = resampler.nanstd(skipna=True)

            #stats_dict['min'] = resampler.nanmin(skipna=True)
            #stats_dict['max'] = resampler.nanmax(skipna=True)

            # FIX: Compute all percentiles at once, then separate them
            # This avoids reference issues
            if percentiles:
                # Compute all quantiles together
                all_quantiles = resampler.quantile([p / 100 for p in percentiles], skipna=True)

                # Now extract each percentile separately
                for i, p in enumerate(percentiles):
                    # Select this specific percentile and drop the quantile coordinate
                    percentile_result = all_quantiles.isel(quantile=i).drop_vars('quantile')
                    stats_dict[f'L{p}'] = percentile_result

            # Add count as well for diagnostics
            stats_dict['count'] = count
            
            # Combine into single Dataset
            var_stats = xr.Dataset(stats_dict)

            # Persist immediately to trigger computation and cache results
            if self.use_dask:
                var_stats = var_stats.persist()
                print(f"    ✓ {name} persisted to distributed memory")

            results[name] = var_stats

        print(f"✓ Time series statistics complete ({resolution} resolution)")

        # Return appropriate format
        if is_dataset:
            if len(results) == 1:
                return list(results.values())[0]
            else:
                # Combine multiple variables
                combined = xr.Dataset()
                for var_name, var_stats in results.items():
                    for stat_name in var_stats.data_vars:
                        combined[f'{var_name}_{stat_name}'] = var_stats[stat_name]
                return combined
        else:
            return results['data']

    def compute_frequency_stats(self, data=None, percentiles=[1, 10, 25, 50, 75, 90, 99],
                                include_mean=False, persist=True):
        """
        Compute statistics (percentiles and optionally mean) across the time dimension for each frequency.
        Optimized for parallel computation with Dask. Robust to NaN values.

        This method computes statistics across time for spectral data (PSD), producing
        a frequency-domain summary showing how sound levels are distributed at each frequency.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset, optional
            Input spectral data in dB units with 'frequency' and 'time' dimensions.
            If None, uses self.ds.psd (default: None)
        percentiles : list of int/float, optional
            Percentiles to compute (default: [1, 10, 25, 50, 75, 90, 99]).
            Each percentile will be saved as 'p{value}' (e.g., p50 for median)
        include_mean : bool, optional
            If True, also compute the proper dB mean (convert to linear, average, convert back).
            Default is False since mean is less commonly used for PSD analysis.
        persist : bool, optional
            If True and using Dask, immediately persist results to distributed memory.
            Default is True for better performance.

        Returns
        -------
        xarray.Dataset
            Statistics with 'frequency' dimension. Variables include:
            - 'p{N}': Nth percentile for each frequency (e.g., p50 is median)
            - 'mean': Proper dB mean (only if include_mean=True)
            - 'count': Number of valid observations at each frequency

        Examples
        --------
        >>> # Compute default percentiles for PSD
        >>> hmd = HMD(n_workers=4)
        >>> hmd.load_nc_files('deployment_01/')
        >>> freq_stats = hmd.compute_frequency_stats()
        >>>
        >>> # Custom percentiles without persisting
        >>> freq_stats = hmd.compute_frequency_stats(
        ...     percentiles=[5, 50, 95],
        ...     include_mean=True,
        ...     persist=False
        ... )
        >>>
        >>> # Use with custom data
        >>> freq_stats = hmd.compute_frequency_stats(data=hmd.ds.psd)

        Notes
        -----
        - For dB data, mean is computed properly: convert to linear → average → convert back
        - All percentiles are computed together in one operation for efficiency
        - Results are persisted to Dask distributed memory for faster subsequent access
        - Use this method in plot_psd for consistent statistics computation
        """
        import xarray as xr
        import dask.array as da
        import numpy as np

        # Use PSD by default
        if data is None:
            self._check_loaded()
            data = self.ds.psd

        # Determine data type and structure
        if isinstance(data, xr.DataArray):
            process_data = {'data': data}
            is_dataset = False
        elif isinstance(data, xr.Dataset):
            process_data = {var: data[var] for var in data.data_vars}
            is_dataset = True
        else:
            raise ValueError("data must be xarray.DataArray or Dataset")

        # Check dimensions
        first_var = list(process_data.values())[0]
        has_time = 'time' in first_var.dims
        has_freq = 'frequency' in first_var.dims

        if not (has_time and has_freq):
            raise ValueError("Data must have both 'time' and 'frequency' dimensions (spectral data)")

        print(f"Computing frequency statistics across time dimension...")

        results = {}

        # Process all variables
        for name, arr in process_data.items():
            print(f"  Processing {name}...")

            # Check if data is already computed (not lazy)
            # If so, convert back to Dask array to avoid large graph warning
            if not hasattr(arr.data, 'dask'):
                # Data is already computed (numpy array)
                # Re-chunk it as a Dask array with optimal chunks
                print(f"    Converting computed data back to Dask array...")
                time_chunk = min(10000, len(arr.time))
                arr = arr.chunk({'time': time_chunk, 'frequency': -1})
            else:
                # Data is already lazy (Dask array)
                # Ensure optimal chunking: all frequencies, chunked time
                current_chunks = arr.chunks
                if 'frequency' in arr.dims:
                    # Rechunk to have all frequencies in one chunk
                    arr = arr.chunk({'time': 10000, 'frequency': -1})

            # Compute all statistics in parallel
            stats_dict = {}

            # Count valid observations per frequency
            count = arr.count(dim='time')

            # Compute mean if requested
            if include_mean:
                # For dB mean, convert to linear → average → convert back
                linear = 10.0 ** (arr / 10.0)
                linear_mean = linear.mean(dim='time', skipna=True)

                # Safe log10 that handles zero/negative values and NaNs
                def safe_log10(x):
                    """Safely compute log10, returning NaN for invalid values"""
                    result = np.where((x > 0) & np.isfinite(x), np.log10(x), np.nan)
                    return result

                stats_dict['mean'] = 10 * xr.apply_ufunc(
                    safe_log10,
                    linear_mean,
                    dask='parallelized',
                    output_dtypes=[float]
                )

            # Compute all percentiles at once for efficiency
            if percentiles:
                print(f"    Computing percentiles: {percentiles}...")
                # Compute all quantiles together
                all_quantiles = arr.quantile([p / 100 for p in percentiles], dim='time', skipna=True)

                # Now extract each percentile separately
                for i, p in enumerate(percentiles):
                    # Select this specific percentile and drop the quantile coordinate
                    percentile_result = all_quantiles.isel(quantile=i).drop_vars('quantile')
                    stats_dict[f'p{p}'] = percentile_result

            # Add count for diagnostics
            stats_dict['count'] = count

            # Combine into single Dataset
            var_stats = xr.Dataset(stats_dict)

            # Persist immediately to trigger computation and cache results
            if persist and self.use_dask:
                var_stats = var_stats.persist()
                print(f"    ✓ {name} persisted to distributed memory")

            results[name] = var_stats

        print(f"✓ Frequency statistics complete")

        # Return appropriate format
        if is_dataset:
            if len(results) == 1:
                return list(results.values())[0]
            else:
                # Combine multiple variables
                combined = xr.Dataset()
                for var_name, var_stats in results.items():
                    for stat_name in var_stats.data_vars:
                        combined[f'{var_name}_{stat_name}'] = var_stats[stat_name]
                return combined
        else:
            return results['data']

    def plot_psd(self, style='quantile', percentiles=[1, 10, 25, 50, 75, 90, 99],
                 freq_range=None, db_range=None, scale='log',
                 cmap='bimary', colors=None, linewidth=1.5,
                 alpha=0.7, legend_loc='best', title=None,
                 figsize=(10, 6), dpi=100, save_path=None, show=True,
                 return_data=False):
        """
        Plot Power Spectral Density (PSD) with statistical summaries across frequency.

        Creates visualizations showing the distribution of sound levels across frequency
        using either quantile lines (with confidence bands) or density heatmaps.

        Parameters
        ----------
        style : str, optional
            Visualization style (default: 'quantile'):
            - 'quantile': Line plot with percentile bands
            - 'density': 2D histogram heatmap showing distribution
            - 'both': Overlay density heatmap with quantile lines on top
        percentiles : list of float, optional
            Percentiles to display (default: [1, 10, 25, 50, 75, 90, 99]).
            Each percentile is plotted as a separate line labeled as L1, L10, etc.
        freq_range : tuple, optional
            Frequency range to display (min_freq, max_freq) in Hz.
            If None, uses full frequency range from data.
        db_range : tuple, optional
            dB scale limits (min_db, max_db) for y-axis.
            If None, auto-scales to data range.
        scale : str, optional
            Frequency axis scale: 'log' (default) or 'linear'
        cmap : str, optional
            Matplotlib colormap for density plots (default: 'viridis').
            Good options: 'viridis', 'plasma', 'inferno', 'Blues', 'YlOrRd'
        colors : str, list, or colormap, optional
            Colors for percentile lines. Can be:
            - None: Auto-generated colors
            - String: Colormap name (e.g., 'viridis', 'plasma')
            - List: Explicit list of colors
        linewidth : float, optional
            Line width for percentile curves (default: 1.5)
        alpha : float, optional
            Line transparency (default: 0.7)
        legend_loc : str, optional
            Legend location (default: 'best'). Same options as plot_multiyear_overlay:
            - 'best', 'upper right', 'lower left', etc.
            - 'outside right', 'outside right top', 'outside right bottom'
            - 'outside left', 'outside left top', 'outside left bottom'
        title : str, optional
            Plot title (auto-generated if None)
        figsize : tuple, optional
            Figure size in inches (default: (10, 6))
        dpi : int, optional
            Resolution in dots per inch (default: 100)
        save_path : str, optional
            Path to save figure (PNG, PDF, etc.)
        show : bool, optional
            Whether to display the figure (default: True)
        return_data : bool, optional
            If True, return computed statistics instead of plotting.

        Returns
        -------
        matplotlib.figure.Figure or dict
            Figure object if return_data=False, otherwise dict of computed statistics

        Examples
        --------
        >>> # Load data and plot PSD quantiles with default percentiles
        >>> hmd = HMD(n_workers=4)
        >>> hmd.load_nc_files('deployment_01/', time_range=('2020-01-01', '2020-02-01'))
        >>> hmd.plot_psd(style='quantile')  # Shows L1, L10, L25, L50, L75, L90, L99
        >>>
        >>> # Custom percentiles with legend outside and custom title
        >>> hmd.plot_psd(style='quantile',
        ...              percentiles=[5, 50, 95],
        ...              title='Acoustic Spectral Characteristics',
        ...              legend_loc='outside right top')
        >>>
        >>> # Use colormap for percentile lines
        >>> hmd.plot_psd(style='quantile',
        ...              colors='viridis',
        ...              linewidth=2,
        ...              alpha=0.8,
        ...              title='PSD Analysis')
        >>>
        >>> # Density plot with custom frequency range
        >>> hmd.plot_psd(style='density', freq_range=(20, 2000), scale='log')
        >>>
        >>> # Overlay density and quantile plots
        >>> hmd.plot_psd(style='both', figsize=(12, 6))
        >>>
        >>> # Get statistics for further analysis
        >>> stats = hmd.plot_psd(return_data=True)

        Notes
        -----
        - PSD plots show spectral characteristics of the acoustic environment
        - Quantile plots are useful for understanding typical and extreme levels
        - Density plots reveal the full distribution of sound levels at each frequency
        - Log scale is recommended for frequency axis to better visualize wide ranges
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import numpy as np

        self._check_loaded()

        print(f"Creating PSD plot (style: {style})...")

        # Subset frequency range if specified
        if freq_range is not None:
            print(f"  Subsetting frequency range: {freq_range[0]}-{freq_range[1]} Hz")
            psd = self.ds.psd.sel(frequency=slice(freq_range[0], freq_range[1]))
        else:
            psd = self.ds.psd

        # Use compute_frequency_stats for optimized statistics computation
        stats_ds = self.compute_frequency_stats(
            data=psd,
            percentiles=percentiles,
            include_mean=False,
            persist=True
        )

        # Get frequency coordinates
        freqs = stats_ds.coords['frequency'].values

        # Return data if requested
        if return_data:
            # Convert Dataset to dict format for backward compatibility
            stats = {var: stats_ds[var] for var in stats_ds.data_vars if var.startswith('p')}
            return stats

        # Extract stats for plotting (compute if still lazy)
        stats = {}
        for var in stats_ds.data_vars:
            if var.startswith('p'):
                if hasattr(stats_ds[var], 'compute'):
                    stats[var] = stats_ds[var].compute()
                else:
                    stats[var] = stats_ds[var]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        # Plot density first (as background) if style is 'both' or 'density'
        if style in ['density', 'both']:
            print("  Creating density plot...")

            # Determine dB bins
            if db_range is not None:
                db_min, db_max = db_range
            else:
                # Compute percentiles from the data without loading all into memory
                if hasattr(psd, 'compute'):
                    # Use a sample for percentile estimation to avoid memory issues
                    print("    Estimating dB range from data sample...")
                    sample = psd.isel(time=slice(0, None, max(1, len(psd.time) // 10000))).compute()
                    db_min = np.nanpercentile(sample.values, 1)
                    db_max = np.nanpercentile(sample.values, 99)
                else:
                    db_min = np.nanpercentile(psd.values, 1)
                    db_max = np.nanpercentile(psd.values, 99)

            n_db_bins = 100
            db_bins = np.linspace(db_min, db_max, n_db_bins + 1)
            db_centers = (db_bins[:-1] + db_bins[1:]) / 2

            # Create density matrix (frequency x dB level)
            density_matrix = np.zeros((len(freqs), len(db_centers)))

            # Compute histogram for each frequency separately to save memory
            print(f"    Computing density histograms for {len(freqs)} frequency bins...")
            for i, freq in enumerate(freqs):
                # Extract just this frequency across all time
                freq_data = psd.sel(frequency=freq, method='nearest')

                # Compute this frequency's data
                if hasattr(freq_data, 'compute'):
                    freq_values = freq_data.compute().values
                else:
                    freq_values = freq_data.values

                # Remove NaNs
                freq_data_clean = freq_values[~np.isnan(freq_values)]

                if len(freq_data_clean) > 0:
                    # Compute histogram
                    hist, _ = np.histogram(freq_data_clean, bins=db_bins)
                    # Normalize to probability density
                    density_matrix[i, :] = hist / hist.sum() if hist.sum() > 0 else hist

                # Progress indicator every 100 frequencies
                if (i + 1) % 100 == 0:
                    print(f"      Progress: {i + 1}/{len(freqs)} frequencies")

            # Plot density as pcolormesh
            # Create meshgrid edges
            freq_edges = np.concatenate([
                [freqs[0] * 0.95],  # Extend slightly before first
                (freqs[:-1] + freqs[1:]) / 2,  # Midpoints
                [freqs[-1] * 1.05]  # Extend slightly after last
            ])

            db_edges = db_bins

            im = ax.pcolormesh(
                freq_edges,
                db_edges,
                density_matrix.T,
                cmap=cmap,
                shading='flat',
                rasterized=True,
                zorder=1  # Behind the lines
            )

            # Add colorbar (smaller for overlay)
            if style == 'both':
                cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
            else:
                cbar = plt.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Probability Density', fontsize=10, fontweight='bold')

        # Plot quantile lines on top if style is 'both' or 'quantile'
        if style in ['quantile', 'both']:
            # Sort percentiles
            percentiles_sorted = sorted(percentiles)
            n_percentiles = len(percentiles_sorted)

            # Set up colors for lines
            if colors is None:
                # Use a colormap for better distinction
                cmap_obj = plt.cm.get_cmap('tab10' if n_percentiles <= 10 else 'tab20')
                line_colors = [cmap_obj(i % cmap_obj.N) for i in range(n_percentiles)]
            elif isinstance(colors, str):
                # String provided - assume it's a colormap name
                try:
                    cmap_obj = plt.cm.get_cmap(colors)
                    line_colors = [cmap_obj(i / max(n_percentiles - 1, 1)) for i in range(n_percentiles)]
                except:
                    # If not a valid colormap, treat as single color
                    line_colors = [colors] * n_percentiles
            elif hasattr(colors, 'N'):
                # It's a colormap object
                line_colors = [colors(i / max(n_percentiles - 1, 1)) for i in range(n_percentiles)]
            else:
                # Assume it's a list of colors
                line_colors = colors

            # Plot all percentile lines
            for idx, p in enumerate(percentiles_sorted):
                p_values = stats[f'p{p}'].values
                color = line_colors[idx % len(line_colors)]
                # Higher zorder to plot on top of density
                ax.plot(freqs, p_values,
                       color=color,
                       linewidth=linewidth,
                       alpha=alpha,
                       label=f'L{p}',  # Use L notation
                       zorder=3)

            # Add legend with same options as plot_multiyear_overlay
            if legend_loc == 'outside right':
                ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.5),
                         framealpha=0.9, ncol=1)
            elif legend_loc == 'outside right top':
                ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0),
                         framealpha=0.9, ncol=1)
            elif legend_loc == 'outside right bottom':
                ax.legend(loc='lower left', bbox_to_anchor=(1.15, 0.0),
                         framealpha=0.9, ncol=1)
            elif legend_loc == 'outside left':
                ax.legend(loc='center right', bbox_to_anchor=(-0.02, 0.5),
                         framealpha=0.9, ncol=1)
            elif legend_loc == 'outside left top':
                ax.legend(loc='upper right', bbox_to_anchor=(-0.02, 1.0),
                         framealpha=0.9, ncol=1)
            elif legend_loc == 'outside left bottom':
                ax.legend(loc='lower right', bbox_to_anchor=(-0.02, 0.0),
                         framealpha=0.9, ncol=1)
            else:
                ax.legend(loc=legend_loc, framealpha=0.9)

        # Set scale and labels (common for all styles)
        if scale == 'log':
            ax.set_xscale('log')
        ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power Spectral Density (dB re 1 µPa²/Hz)',
                     fontsize=11, fontweight='bold')

        if db_range is not None:
            ax.set_ylim(db_range)

        # Add grid
        if style == 'quantile':
            # Full grid for quantile only
            ax.grid(True, which='major', alpha=0.3, linestyle='--')
            ax.grid(True, which='minor', alpha=0.15, linestyle=':')
        else:
            # X-axis grid only for density and both (y-axis grid would obscure density)
            ax.grid(True, which='major', alpha=0.3, linestyle='--', axis='x')
            ax.grid(True, which='minor', alpha=0.15, linestyle=':', axis='x')

        # Enable minor ticks for better grid granularity
        ax.minorticks_on()

        # Set title
        if title is not None:
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            # Auto-generate title based on style
            time_start = pd.to_datetime(self.ds.time.min().values).strftime('%Y-%m-%d')
            time_end = pd.to_datetime(self.ds.time.max().values).strftime('%Y-%m-%d')

            if style == 'both':
                subtitle = f'Power Spectral Density Analysis\n{time_start} to {time_end}'
            elif style == 'quantile':
                subtitle = f'Power Spectral Density - Quantile Plot\n{time_start} to {time_end}'
            else:  # density
                subtitle = f'Power Spectral Density - Density Plot\n{time_start} to {time_end}'

            ax.set_title(subtitle, fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved PSD plot to {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_ltsa(self, bin='1H', freq_range=None, db_range=(32, 108),
                  scale='log', cmap='rainbow', statistic='median',
                  plot_date_range=None, title=None, figsize=(14, 6), dpi=100,
                  save_path=None, show=True, return_data=False):
        """
        Plot Long-Term Spectral Average (LTSA) spectrogram.

        Creates a spectrogram visualization with time on x-axis, frequency on y-axis,
        and color representing sound intensity. Data is binned in time and the specified
        statistic (median, mean, etc.) is computed for each time-frequency bin.

        Parameters
        ----------
        bin : str, optional
            Time interval for binning (default: '1H'). Examples:
            - '1H': 1 hour bins
            - '6H': 6 hour bins
            - '1D': 1 day bins
            - '1W': 1 week bins
        freq_range : tuple, optional
            Frequency range to display (min_freq, max_freq) in Hz.
            If None, uses full frequency range from data.
        db_range : tuple, optional
            Fixed dB scale limits (min_db, max_db) for color mapping.
            If None, auto-scales to data range.
        scale : str, optional
            Frequency axis scale: 'log' (default) or 'linear'
        cmap : str, optional
            Matplotlib colormap name (default: 'rainbow').
            Good options: 'rainbow', 'jet', 'viridis', 'plasma', 'inferno'
        statistic : str, optional
            Statistic to compute for each bin (default: 'median').
            Options: 'median', 'mean', 'min', 'max', 'std'
        plot_date_range : tuple, list, or str, optional
            Date range to display on x-axis (default: None, uses full data range).
            Can be:
            - Tuple/list of two dates: (start_date, end_date) as strings or datetime
            - 'fullyear': Automatically set to Jan 1 - Dec 31 of the year being plotted
            - None: Use full range of available data
        title : str, optional
            Plot title (auto-generated if None)
        figsize : tuple, optional
            Figure size in inches (default: (14, 6))
        dpi : int, optional
            Resolution in dots per inch (default: 100)
        save_path : str, optional
            Path to save figure (PNG, PDF, etc.)
        show : bool, optional
            Whether to display the figure (default: True)
        return_data : bool, optional
            If True, return the binned data array instead of plotting.
            Useful for further analysis or custom plotting.

        Returns
        -------
        matplotlib.figure.Figure or xarray.DataArray
            Figure object if return_data=False, otherwise the binned data

        Examples
        --------
        >>> # Load data and plot LTSA with 6-hour bins
        >>> hmd = HMD(n_workers=4)
        >>> hmd.load_nc_files('deployment_01/', time_range=('2020-01-01', '2020-02-01'))
        >>> hmd.plot_ltsa(bin='6H', freq_range=(50, 1000), scale='log')
        >>>
        >>> # Daily bins with custom color range
        >>> hmd.plot_ltsa(bin='1D', db_range=(60, 120), cmap='plasma')
        >>>
        >>> # Display full year (Jan 1 - Dec 31)
        >>> hmd.plot_ltsa(bin='1D', plot_date_range='fullyear')
        >>>
        >>> # Display specific date range
        >>> hmd.plot_ltsa(bin='1H', plot_date_range=('2020-06-01', '2020-06-30'))
        >>>
        >>> # Get binned data for further analysis
        >>> ltsa_data = hmd.plot_ltsa(bin='1H', return_data=True)

        Notes
        -----
        - LTSA is useful for visualizing long-term acoustic patterns
        - Binning reduces data volume while preserving key features
        - Log scale is recommended for frequency axis in most cases
        - Median is robust to outliers compared to mean
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.colors import LogNorm, Normalize

        self._check_loaded()

        print(f"Creating LTSA plot with {bin} bins...")

        # Subset frequency range if specified
        if freq_range is not None:
            print(f"  Subsetting frequency range: {freq_range[0]}-{freq_range[1]} Hz")
            data = self.ds.sel(frequency=slice(freq_range[0], freq_range[1]))
        else:
            data = self.ds

        # Get the PSD data
        psd = data.psd

        # Optimize chunking for Dask performance
        if not hasattr(psd.data, 'dask'):
            # Data is already computed (numpy array) - re-chunk as Dask array
            print(f"  Converting computed data back to Dask array for optimal resampling...")
            # Determine optimal chunk size based on bin resolution
            if bin in ['1h', '1H']:
                time_chunk = 24 * 60  # 1 day worth of minute data
            elif bin in ['6h', '6H']:
                time_chunk = 24 * 10  # ~1 day worth of 6-hour bins
            elif bin in ['1D', '1d']:
                time_chunk = 365  # 1 year worth of daily data
            else:
                # Default: estimate good chunk size
                time_chunk = min(10000, len(psd.time))

            psd = psd.chunk({'time': time_chunk, 'frequency': -1})
        else:
            # Data is already lazy - ensure optimal chunking for resampling
            if bin in ['1h', '1H']:
                time_chunk = 24 * 60  # 1 day worth of minute data
            elif bin in ['6h', '6H']:
                time_chunk = 24 * 10  # ~1 day worth of 6-hour bins
            elif bin in ['1D', '1d']:
                time_chunk = 30 * 1440  # 1 month worth of minute data
            else:
                time_chunk = 10000  # Default large chunk

            psd = psd.chunk({'time': time_chunk, 'frequency': -1})

        # Resample in time using the specified statistic
        print(f"  Resampling time to {bin} intervals using {statistic}...")

        resampler = psd.resample(time=bin, label='left')

        if statistic == 'median':
            ltsa_data = resampler.median(skipna=True)
        elif statistic == 'mean':
            ltsa_data = resampler.mean(skipna=True)
        elif statistic == 'min':
            ltsa_data = resampler.min(skipna=True)
        elif statistic == 'max':
            ltsa_data = resampler.max(skipna=True)
        elif statistic == 'std':
            ltsa_data = resampler.std(skipna=True)
        else:
            raise ValueError(f"Unknown statistic: {statistic}. "
                           f"Choose from: median, mean, min, max, std")

        # Persist to distributed memory if using Dask (speeds up subsequent operations)
        if self.use_dask and hasattr(ltsa_data, 'persist'):
            print("  Persisting LTSA data to distributed memory...")
            ltsa_data = ltsa_data.persist()

        # Compute if using Dask
        if hasattr(ltsa_data, 'compute'):
            print("  Computing LTSA data...")
            ltsa_data = ltsa_data.compute()

        print(f"  LTSA shape: {ltsa_data.shape[0]} time bins × {ltsa_data.shape[1]} frequency bins")

        # Return data if requested
        if return_data:
            return ltsa_data

        # Create plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Get time and frequency coordinates
        times = ltsa_data.coords['time'].values
        freqs = ltsa_data.coords['frequency'].values
        values = ltsa_data.values  # Keep original orientation (time x frequency)

        # Set up color normalization
        if db_range is not None:
            vmin, vmax = db_range
        else:
            vmin, vmax = np.nanpercentile(values, [1, 99])

        # Plot the spectrogram using pcolormesh
        # Following pbp library approach: time on x-axis, frequency on y-axis
        im = ax.pcolormesh(
            times,           # Time coordinates directly (matplotlib handles datetime64)
            freqs,           # Frequency coordinates
            values.T,        # Transpose: (frequency, time) for pcolormesh
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='nearest',  # Nearest neighbor (like pbp)
            rasterized=True     # Faster rendering for large datasets
        )

        # ax0 = fig.add_subplot(spec[2])
        # vmin, vmax = cmlim
        # sg = plt.pcolormesh(
        #     ds.time, ds.frequency, da, shading="nearest", cmap="rainbow", vmin=vmin, vmax=vmax
        # )
        # plt.yscale("log")
        # plt.ylim(list(ylim))
        # plt.ylabel(freqlabl)
        # xl = ax0.get_xlim()
        # ax0.set_xticks([])

        # Set frequency scale
        if scale == 'log':
            ax.set_yscale('log')

        # Set frequency limits if specified
        if freq_range is not None:
            ax.set_ylim(freq_range)

        # Set x-axis (time) limits if specified
        if plot_date_range is not None:
            if plot_date_range == 'fullyear':
                # Automatically set to Jan 1 - Dec 31 of the year in the data
                # Use the first time point to determine the year
                first_time = pd.to_datetime(times[0])
                year = first_time.year
                xlim_start = pd.Timestamp(f'{year}-01-01')
                xlim_end = pd.Timestamp(f'{year}-12-31 23:59:59')
                ax.set_xlim(xlim_start, xlim_end)
                print(f"  Setting x-axis to full year: {year}")
            elif isinstance(plot_date_range, (list, tuple)) and len(plot_date_range) == 2:
                # User-specified date range
                xlim_start = pd.Timestamp(plot_date_range[0])
                xlim_end = pd.Timestamp(plot_date_range[1])
                ax.set_xlim(xlim_start, xlim_end)
                print(f"  Setting x-axis range: {xlim_start.strftime('%Y-%m-%d')} to {xlim_end.strftime('%Y-%m-%d')}")
            else:
                raise ValueError(
                    "plot_date_range must be 'fullyear' or a tuple/list of two dates. "
                    f"Got: {plot_date_range}"
                )

        # Format x-axis (time) using concise date formatter like pbp
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

        # Labels and title
        ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')

        if title is None:
            time_start = pd.to_datetime(times[0]).strftime('%Y-%m-%d')
            time_end = pd.to_datetime(times[-1]).strftime('%Y-%m-%d')
            title = f'Long-Term Spectral Average (LTSA)\n{time_start} to {time_end} | Bin: {bin} | Statistic: {statistic.capitalize()}'

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Power Spectral Density (dB re 1 µPa²/Hz)',
                      fontsize=10, fontweight='bold')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"✓ Saved LTSA plot to {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_multiyear_overlay(self, data, band_names=None,
                                time_axis='dayofyear', smoothing_window=None,
                                title=None, xlabel=None, ylabel='Sound Level (dB)',
                                figsize=None, colors=None, alpha=0.7,
                                linewidth=1.5, grid=True, legend_loc='best',
                                save_path=None, show=True,
                                show_median=False, median_color='black', median_linewidth=2.5,
                                median_linestyle='-', median_alpha=1.0,
                                show_percentile_range=False, percentiles=[10, 90],
                                range_color='lightgray', range_alpha=0.3,
                                **kwargs):
        """
        Plot multi-year overlay of time series data with aligned time axes.

        Creates plots where each year is shown as a separate line, aligned by
        day of year, week of year, etc. Useful for comparing seasonal patterns
        across multiple years.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Time series data spanning multiple years.
            - If DataArray: plots single variable across years
            - If Dataset: plots selected or all data variables as subplots
        band_names : str or list of str, optional
            For Dataset input, specify which variables to plot.
            If None, plots all data variables.
        time_axis : str, optional
            How to align time across years (default: 'dayofyear'):
            - 'dayofyear': Day of year (1-366), best for daily+ resolution
            - 'weekofyear': Week of year (1-53), good for weekly+ data
            - 'month': Month (1-12), for monthly data
            - 'dayofweek': Day of week (0-6), for analyzing weekly patterns
        smoothing_window : int, optional
            Apply rolling mean smoothing with this window size.
            Units match time_axis (e.g., days for dayofyear).
            Default: None (no smoothing)
        title : str, optional
            Plot title (auto-generated if None)
        xlabel : str, optional
            X-axis label. If None, auto-generated based on time_axis
            (e.g., 'Day of Year', 'Week of Year', etc.)
        ylabel : str, optional
            Y-axis label (default: 'Sound Level (dB)')
        figsize : tuple, optional
            Figure size. Default: (14, 6) for single plot, scaled for multiple
        colors : str, list, or colormap, optional
            Colors for each year. Can be:
            - None: Auto-generated using 'tab10' or 'tab20' colormap
            - String: Colormap name (e.g., 'viridis', 'plasma', 'coolwarm')
            - Colormap object: matplotlib.cm colormap
            - List: Explicit list of colors for each year
        alpha : float, optional
            Line transparency (default: 0.7)
        linewidth : float, optional
            Line width (default: 1.5)
        grid : bool, optional
            Show grid (default: True)
        legend_loc : str, optional
            Legend location (default: 'best'). Can be:
            - 'best', 'upper right', 'upper left', 'lower left', 'lower right', etc.
            - 'outside right': Places legend outside plot on the right (centered)
            - 'outside right top': Places legend outside plot on the right (top-aligned)
            - 'outside right bottom': Places legend outside plot on the right (bottom-aligned)
            - 'outside left': Places legend outside plot on the left (centered)
            - 'outside left top': Places legend outside plot on the left (top-aligned)
            - 'outside left bottom': Places legend outside plot on the left (bottom-aligned)
        save_path : str, optional
            Path to save figure
        show : bool, optional
            Whether to display the figure (default: True)
        show_median : bool, optional
            If True, display median line across all years (default: False)
        median_color : str, optional
            Color for median line (default: 'black')
        median_linewidth : float, optional
            Line width for median line (default: 2.5)
        median_linestyle : str, optional
            Line style for median ('-', '--', '-.', ':', default: '-')
        median_alpha : float, optional
            Transparency for median line (default: 1.0)
        show_percentile_range : bool, optional
            If True, display shaded range between percentiles (default: False)
        percentiles : list of two numbers, optional
            Lower and upper percentiles for range (default: [10, 90])
        range_color : str, optional
            Color for percentile range shading (default: 'lightgray')
        range_alpha : float, optional
            Transparency for percentile range (default: 0.3)
        **kwargs : dict
            Additional matplotlib plot arguments

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> # Load multi-year data
        >>> hmd.load_nc_files('deployment/', time_range=('2018-01-01', '2021-01-01'))
        >>> band_levels = hmd.extract_band_levels([[50, 300]], ['ship'])
        >>> stats = hmd.compute_timeseries_stats(band_levels, resolution='1D')
        >>>
        >>> # Plot mean levels across years, aligned by day of year
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            time_axis='dayofyear',
        ...                            smoothing_window=7)  # 7-day smoothing
        >>>
        >>> # Compare multiple bands across years
        >>> hmd.plot_multiyear_overlay(stats,
        ...                            band_names=['ship_mean', 'fish_mean'],
        ...                            time_axis='weekofyear')
        >>>
        >>> # Use a colormap for years
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            colors='viridis',  # Colormap name
        ...                            time_axis='dayofyear')
        >>>
        >>> # Use a diverging colormap
        >>> import matplotlib.pyplot as plt
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            colors='coolwarm',
        ...                            time_axis='dayofyear')
        >>>
        >>> # Use explicit colors
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            colors=['red', 'blue', 'green'],
        ...                            time_axis='dayofyear')
        >>>
        >>> # Place legend outside plot on the right
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            legend_loc='outside right',
        ...                            time_axis='dayofyear')
        >>>
        >>> # Place legend outside on the right, top-aligned
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            legend_loc='outside right top',
        ...                            time_axis='dayofyear')
        >>>
        >>> # Custom axis labels
        >>> hmd.plot_multiyear_overlay(stats['ship_mean'],
        ...                            xlabel='Day of Year (Jan 1 = 1)',
        ...                            ylabel='SPL (dB re 1 µPa)',
        ...                            time_axis='dayofyear')
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Determine what we're plotting
        if isinstance(data, xr.DataArray):
            plot_data = {data.name or 'Time Series': data}
            is_single = True
        elif isinstance(data, xr.Dataset):
            if band_names is not None:
                if isinstance(band_names, str):
                    band_names = [band_names]
                missing = [b for b in band_names if b not in data.data_vars]
                if missing:
                    raise ValueError(f"Band names not found in Dataset: {missing}")
                plot_data = {var: data[var] for var in band_names}
            else:
                plot_data = {var: data[var] for var in data.data_vars}
            is_single = len(plot_data) == 1
        else:
            raise ValueError("data must be xarray.DataArray or xarray.Dataset")

        # Compute data if needed
        for name in plot_data:
            if hasattr(plot_data[name], 'compute'):
                print(f"Computing {name}...")
                plot_data[name] = plot_data[name].compute()

        n_vars = len(plot_data)

        # Set default figure size
        if figsize is None:
            if is_single:
                figsize = (14, 6)
            else:
                figsize = (14, min(4 * n_vars, 12))

        # Create subplots
        if is_single:
            fig, axes = plt.subplots(1, 1, figsize=figsize)
            axes = [axes]
        else:
            fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
            if n_vars == 1:
                axes = [axes]

        # Process each variable
        for var_idx, (var_name, ts) in enumerate(plot_data.items()):
            ax = axes[var_idx]

            # Convert to pandas for easier time manipulation
            df = ts.to_dataframe(name='value').reset_index()
            df['time'] = pd.to_datetime(df['time'])
            df = df.dropna(subset=['value'])

            # Extract year and time alignment column
            df['year'] = df['time'].dt.year

            if time_axis == 'dayofyear':
                df['time_aligned'] = df['time'].dt.dayofyear
                default_xlabel = 'Day of Year'
                xticks_spacing = 30  # ~monthly
                xlim = (1, 366)
            elif time_axis == 'weekofyear':
                df['time_aligned'] = df['time'].dt.isocalendar().week
                default_xlabel = 'Week of Year'
                xticks_spacing = 4  # ~monthly
                xlim = (1, 53)
            elif time_axis == 'month':
                df['time_aligned'] = df['time'].dt.month
                default_xlabel = 'Month'
                xticks_spacing = 1
                xlim = (1, 12)
            elif time_axis == 'dayofweek':
                df['time_aligned'] = df['time'].dt.dayofweek
                default_xlabel = 'Day of Week'
                xticks_spacing = 1
                xlim = (0, 6)
            else:
                raise ValueError(f"Unknown time_axis: {time_axis}. "
                               f"Choose from: dayofyear, weekofyear, month, dayofweek")

            # Get unique years
            years = sorted(df['year'].unique())
            n_years = len(years)

            # Set up colors
            if colors is None:
                # Use a colormap for better year distinction
                cmap = plt.cm.get_cmap('tab10' if n_years <= 10 else 'tab20')
                year_colors = [cmap(i % cmap.N) for i in range(n_years)]
            elif isinstance(colors, str):
                # String provided - assume it's a colormap name
                try:
                    cmap = plt.cm.get_cmap(colors)
                    year_colors = [cmap(i / max(n_years - 1, 1)) for i in range(n_years)]
                except:
                    # If not a valid colormap, treat as single color
                    year_colors = [colors] * n_years
            elif hasattr(colors, 'N'):
                # It's a colormap object
                year_colors = [colors(i / max(n_years - 1, 1)) for i in range(n_years)]
            else:
                # Assume it's a list of colors
                year_colors = colors

            # Plot each year
            for year_idx, year in enumerate(years):
                year_data = df[df['year'] == year].copy()

                # Sort by aligned time
                year_data = year_data.sort_values('time_aligned')

                # Apply smoothing if requested
                if smoothing_window is not None and smoothing_window > 1:
                    year_data['value'] = year_data['value'].rolling(
                        window=smoothing_window,
                        center=True,
                        min_periods=1
                    ).mean()

                # Detect gaps in time sequence and break lines at those points
                # This prevents matplotlib from connecting across missing data periods
                color = year_colors[year_idx % len(year_colors)]

                # Calculate expected time step based on resolution
                time_diffs = year_data['time_aligned'].diff()

                # For most resolutions, the typical diff should be 1
                # But account for edge cases around year boundaries
                if len(time_diffs) > 1:
                    # Use median to get typical step size
                    typical_step = time_diffs[time_diffs > 0].median()
                    # A gap is when the diff is more than 1.5x the typical step
                    gap_threshold = typical_step * 1.5
                else:
                    gap_threshold = 2  # Default threshold

                # Find where gaps occur
                is_gap = time_diffs > gap_threshold

                # Find continuous segments between gaps
                segment_breaks = year_data.index[is_gap].tolist()

                # Create list of segment boundaries
                segment_bounds = [year_data.index[0]] + segment_breaks + [year_data.index[-1]]

                # Plot each continuous segment separately
                first_segment = True
                for i in range(len(segment_bounds) - 1):
                    start_idx = segment_bounds[i]
                    end_idx = segment_bounds[i + 1]

                    # Get the segment
                    if i < len(segment_bounds) - 2:
                        # Not the last segment - exclude the gap point
                        segment = year_data.loc[start_idx:end_idx].iloc[:-1]
                    else:
                        # Last segment - include all points
                        segment = year_data.loc[start_idx:end_idx]

                    # Skip empty segments
                    if len(segment) == 0:
                        continue

                    # Only add label to first segment to avoid duplicate legend entries
                    label_text = str(year) if first_segment else None

                    ax.plot(segment['time_aligned'], segment['value'],
                           label=label_text, color=color, alpha=alpha,
                           linewidth=linewidth, **kwargs)

                    first_segment = False

            # Compute and plot median line across all years if requested
            if show_median:
                # Group all data by time_aligned and compute median
                median_by_aligned = df.groupby('time_aligned')['value'].median()

                # Plot median line
                ax.plot(median_by_aligned.index, median_by_aligned.values,
                       color=median_color, linewidth=median_linewidth,
                       linestyle=median_linestyle, alpha=median_alpha,
                       label='Median', zorder=10)  # High zorder to plot on top

            # Compute and plot percentile range across all years if requested
            if show_percentile_range:
                if len(percentiles) != 2:
                    raise ValueError("percentiles must be a list of exactly 2 values [lower, upper]")

                # Group all data by time_aligned and compute percentiles
                lower_percentile = df.groupby('time_aligned')['value'].quantile(percentiles[0] / 100)
                upper_percentile = df.groupby('time_aligned')['value'].quantile(percentiles[1] / 100)

                # Plot shaded range
                ax.fill_between(lower_percentile.index,
                               lower_percentile.values,
                               upper_percentile.values,
                               color=range_color, alpha=range_alpha,
                               label=f'{percentiles[0]}-{percentiles[1]}% Range',
                               zorder=1)  # Low zorder to plot behind lines

            # Formatting
            # Use custom xlabel if provided, otherwise use default based on time_axis
            final_xlabel = xlabel if xlabel is not None else default_xlabel
            ax.set_xlabel(final_xlabel, fontsize=10)
            ax.set_ylabel(ylabel if is_single else f'{var_name}\n({ylabel})',
                         fontsize=10)
            ax.set_xlim(xlim)

            # Set x-ticks
            if time_axis == 'dayofyear':
                # Show month labels
                month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax.set_xticks(month_starts)
                ax.set_xticklabels(month_names)
            elif time_axis == 'weekofyear':
                ax.set_xticks(np.arange(1, 54, xticks_spacing))
            elif time_axis == 'month':
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(month_names)
            elif time_axis == 'dayofweek':
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                ax.set_xticks(range(7))
                ax.set_xticklabels(day_names)

            if grid:
                ax.grid(True, alpha=0.3, linestyle='--')

            # Add legend
            if legend_loc == 'outside right':
                ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         framealpha=0.9, title='Year', ncol=1)
            elif legend_loc == 'outside right top':
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                         framealpha=0.9, title='Year', ncol=1)
            elif legend_loc == 'outside right bottom':
                ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0),
                         framealpha=0.9, title='Year', ncol=1)
            elif legend_loc == 'outside left':
                ax.legend(loc='center right', bbox_to_anchor=(-0.02, 0.5),
                         framealpha=0.9, title='Year', ncol=1)
            elif legend_loc == 'outside left top':
                ax.legend(loc='upper right', bbox_to_anchor=(-0.02, 1.0),
                         framealpha=0.9, title='Year', ncol=1)
            elif legend_loc == 'outside left bottom':
                ax.legend(loc='lower right', bbox_to_anchor=(-0.02, 0.0),
                         framealpha=0.9, title='Year', ncol=1)
            else:
                ax.legend(loc=legend_loc, framealpha=0.9, title='Year',
                         ncol=min(n_years, 6))

            # Add title for single variable or subplot titles
            if is_single and title is None:
                smooth_text = f" ({smoothing_window}-{time_axis.replace('of', ' ')} smoothing)" if smoothing_window else ""
                title = f'Multi-Year Overlay: {var_name}{smooth_text}'

            if is_single:
                ax.set_title(title, fontsize=12, fontweight='bold')
            elif n_vars > 1:
                ax.set_title(var_name, fontsize=11, fontweight='bold')

        # Overall title for multiple variables
        if not is_single and title:
            fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def plot_timeseries(self, data, band_names=None, overlay=True, title=None,
                        ylabel='Sound Level (dB)', xlabel='Time',
                        figsize=None, colors=None, alpha=0.7,
                        linewidth=1.5, grid=True, legend_loc='best',
                        sharex=True, sharey=False, save_path=None, **kwargs):
        """
        Plot time series of pre-extracted acoustic data.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            Pre-computed time series data.
            - If DataArray: plots single time series
            - If Dataset: plots selected or all data variables
        band_names : str or list of str, optional
            For Dataset input, specify which variables to plot.
            If None, plots all data variables.
            For DataArray input, this is ignored.
        overlay : bool, optional
            If True (default), plot all series on same axes.
            If False, create separate subplot for each series.
        title : str, optional
            Plot title (auto-generated if None)
        figsize : tuple, optional
            Figure size. Defaults based on overlay and number of series:
            - Single: (12, 4)
            - Multiple overlaid: (14, 6)
            - Multiple subplots: (12, 3*n_series)
        colors : str or list, optional
            Colors for plotting. Single color or list of colors.
        alpha : float, optional
            Line transparency (default: 0.7)
        linewidth : float, optional
            Line width (default: 1.5)
        grid : bool, optional
            Show grid (default: True)
        legend_loc : str, optional
            Legend location (default: 'best'), only used when overlay=True
        sharex : bool, optional
            Share x-axis across subplots when overlay=False (default: True)
        sharey : bool, optional
            Share y-axis across subplots when overlay=False (default: False)
        save_path : str, optional
            Path to save figure
        **kwargs : dict
            Additional matplotlib plot arguments

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> # Single time series
        >>> hmd.plot_timeseries(result['ship'])

        >>> # Multiple bands overlaid (default)
        >>> hmd.plot_timeseries(result, band_names=['ship', 'fish'])

        >>> # Multiple bands as separate subplots
        >>> hmd.plot_timeseries(result, band_names=['ship', 'fish'], overlay=False)

        >>> # All bands as subplots with shared y-axis
        >>> hmd.plot_timeseries(result, overlay=False, sharey=True)
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Determine what we're plotting
        if isinstance(data, xr.DataArray):
            # Single time series
            plot_data = {data.name or 'Time Series': data}
            is_single = True

        elif isinstance(data, xr.Dataset):
            # Multiple time series from Dataset
            if band_names is not None:
                # Plot specific bands
                if isinstance(band_names, str):
                    band_names = [band_names]

                # Validate band names
                missing = [b for b in band_names if b not in data.data_vars]
                if missing:
                    raise ValueError(f"Band names not found in Dataset: {missing}")

                plot_data = {var: data[var] for var in band_names}
            else:
                # Plot all data variables
                plot_data = {var: data[var] for var in data.data_vars}

            is_single = len(plot_data) == 1

        else:
            raise ValueError("data must be xarray.DataArray or xarray.Dataset")

        # Ensure data is computed (not lazy)
        for name in plot_data:
            if hasattr(plot_data[name], 'compute'):
                print(f"Computing {name}...")
                plot_data[name] = plot_data[name].compute()

        n_series = len(plot_data)

        # Set default figure size based on single vs multiple and overlay mode
        if figsize is None:
            if is_single:
                figsize = (12, 4)
            elif overlay:
                figsize = (14, 6)
            else:
                # Subplots: scale height with number of series
                figsize = (12, min(3 * n_series, 12))  # Cap at 12 inches height

        # Set up colors
        if colors is None:
            if is_single or overlay:
                colors = ['steelblue', 'darkgreen', 'coral', 'purple', 'brown',
                          'pink', 'gray', 'olive', 'cyan', 'red']
            else:
                # For subplots, can use same color for all
                colors = ['steelblue'] * n_series
        elif isinstance(colors, str):
            colors = [colors] * n_series

        # Create figure and axes
        if is_single or overlay:
            # Single plot or overlaid plots
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax] * n_series  # Same axis for all
        else:
            # Separate subplots
            fig, axes = plt.subplots(n_series, 1, figsize=figsize,
                                     sharex=sharex, sharey=sharey,
                                     squeeze=False)
            axes = axes.flatten()

        # Plot each time series
        for idx, (name, ts) in enumerate(plot_data.items()):
            ax = axes[idx]
            color = colors[idx % len(colors)]

            if overlay and not is_single:
                # Overlaid plot needs labels for legend
                label = name
            else:
                # Subplots don't need legend
                label = None

            ax.plot(ts.time.values, ts.values,
                    label=label, color=color, alpha=alpha,
                    linewidth=linewidth, **kwargs)

            # For subplots, add individual titles and y-labels
            if not overlay and not is_single:
                ax.set_title(name, fontsize=11, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=9)

                # Format x-axis for each subplot
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())

                if grid:
                    ax.grid(True, alpha=0.3, linestyle='--')

        # Set overall title and labels
        if overlay or is_single:
            # Single plot setup
            ax = axes[0]

            if title is None:
                if is_single:
                    name = list(plot_data.keys())[0]
                    title = f'Sound Level Time Series: {name}'
                else:
                    n_bands = len(plot_data)
                    title = f'Acoustic Time Series ({n_bands} bands)'

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            # Add legend if multiple series overlaid
            if overlay and not is_single and n_series > 1:
                ax.legend(loc=legend_loc, framealpha=0.9)

            if grid:
                ax.grid(True, alpha=0.3, linestyle='--')

        else:
            # Subplots setup
            if title is None:
                title = f'Acoustic Time Series ({n_series} bands)'

            fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

            # Only label bottom x-axis
            axes[-1].set_xlabel(xlabel, fontsize=10)

        # Rotate x-labels
        fig.autofmt_xdate(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        else:
            plt.show()

        return fig

    def timeseries_dashboard(self, data, title='Time Series Dashboard',
                                  port=5006, show=True, save_html=None):
        """
        Create advanced interactive dashboard with variable selection and multiple views.

        Parameters
        ----------
        data : xarray.Dataset or list of xarray.Dataset
            Time series data (must be computed). Can be a single Dataset or a list
            of Datasets to overlay on the same plot.
        title : str, optional
            Dashboard title
        port : int, optional
            Port for web server
        show : bool, optional
            Auto-open browser
        save_html : str, optional
            Save to HTML file

        Returns
        -------
        panel.template.Template
            Panel template dashboard

        Examples
        --------
        >>> # Single dataset
        >>> stats = hmd.compute_timeseries_stats(band_levels, resolution='1h')
        >>> hmd.plot_interactive_advanced(stats.compute())

        >>> # Multiple datasets overlaid
        >>> hmd.plot_interactive_advanced([band_levels.compute(), stats.compute()])
        """
        try:
            import holoviews as hv
            import panel as pn
            import pandas as pd
            import numpy as np
            from holoviews import opts
            from bokeh.models import HoverTool
        except ImportError:
            raise ImportError(
                "Interactive plotting requires holoviews and panel. Install with:\n"
                "  pip install holoviews panel bokeh"
            )

        hv.extension('bokeh')
        pn.extension()

        # Handle single dataset or list of datasets
        if not isinstance(data, list):
            data_list = [data]
        else:
            data_list = data

        # Compute all datasets if needed
        for i in range(len(data_list)):
            if hasattr(data_list[i], 'compute'):
                print(f"Computing dataset {i + 1}/{len(data_list)}...")
                data_list[i] = data_list[i].compute()

        # Validate all are Datasets
        for i, ds in enumerate(data_list):
            if not isinstance(ds, xr.Dataset):
                raise ValueError(f"Item {i} must be xarray.Dataset, got {type(ds)}")

        # Collect all variables from all datasets with prefixes
        all_vars = []
        var_to_dataset = {}  # Map variable name to (dataset_index, original_var_name)

        for idx, ds in enumerate(data_list):
            prefix = f"Dataset{idx + 1}_" if len(data_list) > 1 else ""
            for var in ds.data_vars:
                display_name = f"{prefix}{var}"
                all_vars.append(display_name)
                var_to_dataset[display_name] = (idx, var)

        # Create variable selector
        var_selector = pn.widgets.MultiChoice(
            name='Select Variables to Plot',
            options=all_vars,
            value=all_vars[:min(5, len(all_vars))],  # Default to first 5
            width=400
        )

        # Color palette selector
        color_schemes = {
            'Default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'Viridis': ['#440154', '#31688e', '#35b779', '#fde724', '#b5de2b'],
            'Warm': ['#d62728', '#ff7f0e', '#bcbd22', '#e377c2', '#8c564b'],
            'Cool': ['#1f77b4', '#17becf', '#9467bd', '#2ca02c', '#7f7f7f']
        }
        color_selector = pn.widgets.Select(
            name='Color Scheme',
            options=list(color_schemes.keys()),
            value='Default',
            width=200
        )

        # Date range selector (use first dataset for range)
        time_min = pd.Timestamp(data_list[0].time.min().values)
        time_max = pd.Timestamp(data_list[0].time.max().values)

        # Extend range if other datasets have wider ranges
        for ds in data_list[1:]:
            ds_min = pd.Timestamp(ds.time.min().values)
            ds_max = pd.Timestamp(ds.time.max().values)
            if ds_min < time_min:
                time_min = ds_min
            if ds_max > time_max:
                time_max = ds_max

        date_range = pn.widgets.DatetimeRangePicker(
            name='Date Range',
            start=time_min,
            end=time_max,
            value=(time_min, time_max),
            width=400
        )

        # Line width slider
        line_width_slider = pn.widgets.FloatSlider(
            name='Line Width',
            start=0.5,
            end=5,
            step=0.5,
            value=2,
            width=200
        )

        # Alpha slider
        alpha_slider = pn.widgets.FloatSlider(
            name='Transparency',
            start=0.1,
            end=1.0,
            step=0.1,
            value=0.8,
            width=200
        )

        @pn.depends(var_selector.param.value, color_selector.param.value,
                    date_range.param.value, line_width_slider.param.value,
                    alpha_slider.param.value)
        def create_plot(selected_vars, color_scheme, date_range_val, line_width, alpha):
            if not selected_vars:
                return pn.pane.Markdown("### Please select at least one variable to plot")

            colors = color_schemes[color_scheme]
            curves = []

            for idx, display_var in enumerate(selected_vars):
                # Get the dataset and original variable name
                ds_idx, original_var = var_to_dataset[display_var]
                ts = data_list[ds_idx][original_var]

                # Filter by date range
                if date_range_val:
                    start, end = date_range_val
                    ts = ts.sel(time=slice(start, end))

                # Convert to DataFrame and add series name column
                df = ts.to_dataframe(name='value').reset_index()
                df = df.dropna(subset=['value'])

                if len(df) == 0:
                    continue

                # Add series name to dataframe for tooltip
                df['series'] = display_var

                # Create curve with all necessary vdims for tooltips
                curve = hv.Curve(df, kdims=['time'], vdims=['value', 'series'], label=display_var)
                curve = curve.opts(
                    color=colors[idx % len(colors)],
                    line_width=line_width,
                    alpha=alpha
                )
                curves.append(curve)

            if not curves:
                return pn.pane.Markdown("### No valid data in selected range")

            # Create custom HoverTool
            hover = HoverTool(
                tooltips=[
                    ('Series', '@series'),
                    ('Time', '@time{%Y-%m-%d %H:%M:%S}'),
                    ('Value', '@value{0.3f} dB'),
                ],
                formatters={
                    '@time': 'datetime',
                },
                mode='mouse'
            )

            # Overlay all curves - apply layout options to Overlay, not individual Curves
            overlay = hv.Overlay(curves).opts(
                width=1400,
                height=600,
                xlabel='Time',
                ylabel='Sound Level (dB)',
                title='Time Series Comparison',
                legend_position='top_right',
                show_grid=True,
                toolbar='above',
                tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
                active_tools=['pan', 'wheel_zoom']
            )

            return overlay

        # Info panel with dataset information
        info_text = "## Dataset Information\n\n"
        for i, ds in enumerate(data_list):
            prefix = f"Dataset {i + 1}" if len(data_list) > 1 else "Dataset"
            info_text += f"**{prefix}:**\n"
            info_text += f"- Variables: {len(ds.data_vars)}\n"
            info_text += f"- Time points: {len(ds.time)}\n"
            ds_min = pd.Timestamp(ds.time.min().values)
            ds_max = pd.Timestamp(ds.time.max().values)
            info_text += f"- Date range: {ds_min.date()} to {ds_max.date()}\n\n"

        info_text += f"**Total variables available:** {len(all_vars)}\n"

        # Build dashboard using template
        template = pn.template.FastListTemplate(
            title=title,
            sidebar=[
                pn.pane.Markdown("## Controls"),
                var_selector,
                pn.layout.Divider(),
                color_selector,
                line_width_slider,
                alpha_slider,
                pn.layout.Divider(),
                date_range,
                pn.layout.Divider(),
                pn.pane.Markdown(info_text)
            ],
            main=[
                pn.Column(
                    # pn.pane.Markdown(
                    #     "**Interactive Features:**\n"
                    #     "- Hover over lines to see values\n"
                    #     "- Scroll to zoom, drag to pan\n"
                    #     "- Box zoom: drag while holding shift\n"
                    #     "- Reset: click reset button to restore view"
                    # ),
                    create_plot
                )
            ],
            accent_base_color="#2196F3",
            header_background="#2196F3"
        )

        # Save to HTML if requested
        if save_html:
            template.save(save_html, embed=True)
            print(f"✓ Saved interactive dashboard to {save_html}")

        # Show in browser
        if show:
            print(f"✓ Opening advanced dashboard in browser (port {port})...")
            template.show(port=port, threaded=True)

        return template

    def summary(self):
        """Print summary of loaded dataset"""
        if self.ds is None:
            print("No data loaded. Use .load_nc_files() first.")
            return

        print("=" * 70)
        print("HYBRID MILLIDECADE DATASET SUMMARY")
        print("=" * 70)
        print(f"Time range     : {self.ds.time.min().values} to "
              f"{self.ds.time.max().values}")
        print(f"Time points    : {len(self.ds.time)}")
        print(f"Frequency range: {self.ds.frequency.min().values:.1f} - "
              f"{self.ds.frequency.max().values:.1f} Hz")
        print(f"Frequency bins : {len(self.ds.frequency)}")
        print(f"PSD shape      : {self.ds.psd.shape}")
        print(f"Total size     : {self.ds.nbytes / 1e9:.3f} GB")

        if hasattr(self.ds, 'deployment'):
            if 'deployment' in self.ds.dims:
                print(f"Deployments    : {list(self.ds.deployment.values)}")
            else:
                print(f"Deployment     : {self.ds.deployment.values}")

        print(f"Data variables : {list(self.ds.data_vars)}")

        if self.ds.chunks:
            print(f"Chunks         : {dict(self.ds.chunks)}")

        print("=" * 70)

    def rechunk(self, chunks=None):
        """
        Rechunk the dataset for optimal performance.

        Rechunking can improve performance when chunks don't align well with
        the stored data or when you're doing operations along specific dimensions.

        Parameters
        ----------
        chunks : dict, optional
            Chunk sizes for each dimension. If None, uses intelligent defaults
            based on the operation type:
            - For time-series extraction: {'time': -1, 'frequency': 'auto'}
            - For spectral analysis: {'time': 'auto', 'frequency': -1}
            Default: {'time': 1440, 'frequency': -1} (good for band extraction)

        Returns
        -------
        self

        Examples
        --------
        >>> # Rechunk for time-series extraction (extract_band_levels)
        >>> hmd.rechunk({'time': -1, 'frequency': 'auto'})
        >>>
        >>> # Rechunk for spectral statistics
        >>> hmd.rechunk({'time': 'auto', 'frequency': -1})
        >>>
        >>> # Custom chunking
        >>> hmd.rechunk({'time': 10000, 'frequency': 500})

        Notes
        -----
        - Use -1 to load entire dimension into single chunk
        - Use 'auto' to let Dask determine optimal size
        - Rechunking can be slow but improves downstream performance
        - Call this after load_nc_files() and before analysis operations
        """
        self._check_loaded()

        if chunks is None:
            # Default: optimize for band extraction (common use case)
            chunks = {'time': 1440, 'frequency': -1}

        print(f"Rechunking dataset with chunks: {chunks}")
        print("  This may take a moment but will improve downstream performance...")

        # Rechunk the dataset
        self.ds = self.ds.chunk(chunks)

        # Persist to trigger computation and avoid repeated rechunking
        if self.use_dask:
            print("  Persisting rechunked data to distributed memory...")
            self.ds = self.ds.persist()

        print("✓ Rechunking complete")

        return self

    def subset(self, freq_range=None, time_range=None, persist=True):
        """Create a subset of the data"""
        self._check_loaded()

        ds_subset = self.ds

        if freq_range is not None:
            ds_subset = ds_subset.sel(frequency=slice(freq_range[0], freq_range[1]))

        if time_range is not None:
            ds_subset = ds_subset.sel(time=slice(time_range[0], time_range[1]))

        if persist:
            ds_subset = ds_subset.persist()

        return ds_subset

    def analyze_spatial_correlation(self, timeseries, spatial_grid, method='pearson',
                                     min_periods=None, fill_grid_nan=True, absolute=False, compute=True,
                                     use_lag=False, max_lag=None):
        """
        Compute spatial correlation between a time series and gridded spatial data.

        This method correlates a single time series (e.g., acoustic levels at a point)
        with time series at each spatial grid cell (e.g., vessel counts across a region).
        Uses Dask for parallel processing across grid cells.

        Parameters
        ----------
        timeseries : xarray.DataArray
            1D time series with 'time' dimension (e.g., SPL at recorder location)
        spatial_grid : xarray.DataArray
            3D gridded data with dimensions (time, latitude, longitude)
            (e.g., vessel counts on a spatial grid)
        method : str, optional
            Correlation method: 'pearson' (default), 'spearman', or 'kendall'
        min_periods : int, optional
            Minimum number of overlapping time points required.
            Default is None (uses all available overlap)
        fill_grid_nan : bool, optional
            If True (default), fill NaN values in spatial_grid with zeros.
            This is useful for vessel count grids where NaN typically means zero vessels.
            A warning will be displayed if NaNs are found and filled.
        absolute : bool, optional
            If True, return absolute values of correlation coefficients [0, 1].
            If False (default), return full correlation values [-1, 1].
            Use absolute=True when you care about correlation strength regardless of direction.
        compute : bool, optional
            If True (default), compute result immediately.
            If False, return lazy Dask array.
        use_lag : bool, optional
            If True, use cross-correlation with lag to find maximum correlation.
            If False (default), use standard correlation at lag=0.
            This is useful when the spatial signal (e.g., vessel activity) may
            precede or lag the acoustic signal.
        max_lag : int, optional
            Maximum time lag (in time steps) to consider when use_lag=True.
            The correlation will be computed for all lags from -max_lag to +max_lag.
            If None and use_lag=True, defaults to min(50, n_times // 4).
            Positive lags mean the grid leads the timeseries.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            If use_lag=False:
                2D correlation map with dimensions (latitude, longitude)
                containing correlation coefficients:
                - If absolute=False: values in [-1, 1]
                - If absolute=True: values in [0, 1]
            If use_lag=True:
                Dataset with two DataArrays:
                - 'correlation': Maximum correlation coefficient at each grid cell
                - 'lag': Time lag (in time steps) where maximum occurs
                  Positive lag means grid leads timeseries, negative means timeseries leads grid

        Examples
        --------
        >>> # Correlate acoustic levels with vessel grid
        >>> with AISQueryHelper(db_file) as ais:
        ...     vessel_grid = ais.create_gridded_vessel_counts(...)
        >>>
        >>> with HMD(n_workers=8) as hmd:
        ...     hmd.load_nc_files(deployment_dir, time_range=time_range)
        ...     band_levels = hmd.extract_band_levels([[50, 300]], ['ship'])
        ...     stats = hmd.compute_timeseries_stats(band_levels, resolution='1H')
        ...
        ...     # Correlate mean ship noise with vessel counts (fills NaNs with 0)
        ...     corr_map = hmd.analyze_spatial_correlation(
        ...         stats['ship_mean'],
        ...         vessel_grid,
        ...         method='pearson',
        ...         fill_grid_nan=True  # Default, fills NaNs in grid with 0
        ...     )
        ...
        ...     # Get absolute correlation (strength regardless of direction)
        ...     corr_map_abs = hmd.analyze_spatial_correlation(
        ...         stats['ship_mean'],
        ...         vessel_grid,
        ...         absolute=True  # Returns values in [0, 1]
        ...     )
        ...
        ...     # Or keep NaNs in the grid (may result in NaN correlations)
        ...     corr_map = hmd.analyze_spatial_correlation(
        ...         stats['ship_mean'],
        ...         vessel_grid,
        ...         fill_grid_nan=False
        ...     )
        ...
        ...     # Plot correlation map
        ...     corr_map.plot()
        ...
        ...     # Use cross-correlation with lag to account for temporal offset
        ...     result = hmd.analyze_spatial_correlation(
        ...         stats['ship_mean'],
        ...         vessel_grid,
        ...         use_lag=True,
        ...         max_lag=24  # Search up to 24 hours of lag
        ...     )
        ...
        ...     # Access the correlation and lag maps
        ...     corr_map_lag = result['correlation']  # Maximum correlation at each point
        ...     lag_map = result['lag']  # Lag in time steps where max occurs
        ...
        ...     # Plot results
        ...     corr_map_lag.plot()
        ...     lag_map.plot()  # Positive = grid leads, negative = timeseries leads

        Notes
        -----
        - Time coordinates must overlap between timeseries and spatial_grid
        - NaN values are handled automatically (excluded from correlation)
        - Computation is parallelized across grid cells using Dask
        - Grid cells with insufficient valid data return NaN
        - When use_lag=True, only Pearson correlation is supported (method must be 'pearson')
        """
        import xarray as xr
        import numpy as np
        import dask.array as da
        from scipy.stats import pearsonr, spearmanr, kendalltau

        # Validate inputs
        if not isinstance(timeseries, xr.DataArray):
            raise ValueError("timeseries must be an xarray.DataArray")

        if not isinstance(spatial_grid, xr.DataArray):
            raise ValueError("spatial_grid must be an xarray.DataArray")

        if 'time' not in timeseries.dims:
            raise ValueError("timeseries must have 'time' dimension")

        required_dims = {'time', 'latitude', 'longitude'}
        if not required_dims.issubset(spatial_grid.dims):
            raise ValueError(f"spatial_grid must have dimensions: {required_dims}")

        if timeseries.dims != ('time',):
            raise ValueError("timeseries must be 1D with only 'time' dimension")

        # Select correlation function
        corr_funcs = {
            'pearson': pearsonr,
            'spearman': spearmanr,
            'kendall': kendalltau
        }

        if method not in corr_funcs:
            raise ValueError(f"method must be one of {list(corr_funcs.keys())}")

        # Validate use_lag parameter
        if use_lag and method != 'pearson':
            raise ValueError("use_lag=True requires method='pearson' (other correlation methods not supported for lag analysis)")

        print(f"Computing {method} correlation between time series and spatial grid...")
        print(f"  Time series shape: {timeseries.shape}")
        print(f"  Spatial grid shape: {spatial_grid.shape}")

        # Handle NaN values in spatial grid
        if fill_grid_nan:
            # Check if there are NaNs
            if hasattr(spatial_grid, 'compute'):
                # For Dask arrays, check a sample
                sample = spatial_grid.isel(time=0, latitude=0, longitude=0).compute()
                has_nans = True  # Assume yes for Dask arrays to avoid full computation
            else:
                has_nans = np.any(np.isnan(spatial_grid.values))

            if has_nans:
                n_nans = np.sum(np.isnan(spatial_grid.values)) if not hasattr(spatial_grid, 'compute') else "unknown"
                print(f"  ⚠ WARNING: NaN values found in spatial_grid (count: {n_nans})")
                print(f"  → Filling NaN values with 0 (typical for vessel count grids)")
                spatial_grid = spatial_grid.fillna(0)

        # Align time coordinates - find overlapping times
        timeseries_aligned, grid_aligned = xr.align(
            timeseries,
            spatial_grid,
            join='inner',  # Only keep overlapping times
            copy=False
        )

        n_times = len(timeseries_aligned.time)
        print(f"  Overlapping time points: {n_times}")

        if n_times == 0:
            raise ValueError("No overlapping time points between timeseries and spatial_grid")

        # Set default min_periods
        if min_periods is None:
            min_periods = max(3, int(0.5 * n_times))  # At least 3 or 50% of points

        print(f"  Minimum valid points required: {min_periods}")

        # Set default max_lag if using lag-based correlation
        if use_lag and max_lag is None:
            max_lag = min(50, n_times // 4)
            print(f"  Using default max_lag: {max_lag}")

        # Convert to numpy arrays for the timeseries (compute if needed)
        if hasattr(timeseries_aligned, 'compute'):
            ts_values = timeseries_aligned.compute().values
        else:
            ts_values = timeseries_aligned.values

        if use_lag:
            # Define lag-based correlation function for a single grid cell
            def correlate_grid_cell_with_lag(grid_cell_values):
                """
                Compute cross-correlation with lag and return max correlation and corresponding lag.
                Returns a 2-element array: [max_corr, best_lag]
                """
                # Find valid (non-NaN) points in both series
                valid_mask = ~(np.isnan(ts_values) | np.isnan(grid_cell_values))
                n_valid = np.sum(valid_mask)

                # Return NaN for both if insufficient valid points
                if n_valid < min_periods:
                    return np.array([np.nan, np.nan])

                # Extract valid points
                ts_valid = ts_values[valid_mask]
                grid_valid = grid_cell_values[valid_mask]

                # Check for constant values (correlation undefined)
                if np.std(ts_valid) == 0 or np.std(grid_valid) == 0:
                    return np.array([np.nan, np.nan])

                # Normalize the series for correlation
                ts_norm = (ts_valid - np.mean(ts_valid)) / np.std(ts_valid)
                grid_norm = (grid_valid - np.mean(grid_valid)) / np.std(grid_valid)

                max_corr = np.nan
                best_lag = 0

                try:
                    # Compute correlation at each lag
                    for lag in range(-max_lag, max_lag + 1):
                        if lag == 0:
                            # No shift needed
                            corr_value = np.mean(ts_norm * grid_norm)
                        elif lag > 0:
                            # Grid leads: shift grid forward (or ts backward)
                            # Correlate ts[lag:] with grid[:-lag]
                            if lag < len(ts_norm):
                                corr_value = np.mean(ts_norm[lag:] * grid_norm[:-lag])
                            else:
                                continue  # Skip if lag is too large
                        else:  # lag < 0
                            # Timeseries leads: shift ts forward (or grid backward)
                            # Correlate ts[:lag] with grid[-lag:]
                            abs_lag = abs(lag)
                            if abs_lag < len(ts_norm):
                                corr_value = np.mean(ts_norm[:-abs_lag] * grid_norm[abs_lag:])
                            else:
                                continue  # Skip if lag is too large

                        # Keep track of maximum absolute correlation
                        if np.isnan(max_corr) or abs(corr_value) > abs(max_corr):
                            max_corr = corr_value
                            best_lag = lag

                    return np.array([max_corr, best_lag])
                except Exception:
                    return np.array([np.nan, np.nan])

            # Apply lag-based correlation function
            result_array = xr.apply_ufunc(
                correlate_grid_cell_with_lag,
                grid_aligned,
                input_core_dims=[['time']],
                output_core_dims=[['stat']],  # Output has a 'stat' dimension for [corr, lag]
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'stat': 2}}
            )

            # Split the result into correlation and lag DataArrays
            correlation_map = result_array.isel(stat=0)
            lag_map = result_array.isel(stat=1)

        else:
            # Define standard correlation function for a single grid cell (lag=0)
            def correlate_grid_cell(grid_cell_values):
                """
                Correlate timeseries with a single grid cell time series.
                Handles NaN values and returns correlation coefficient.
                """
                # Find valid (non-NaN) points in both series
                valid_mask = ~(np.isnan(ts_values) | np.isnan(grid_cell_values))
                n_valid = np.sum(valid_mask)

                # Return NaN if insufficient valid points
                if n_valid < min_periods:
                    return np.nan

                # Extract valid points
                ts_valid = ts_values[valid_mask]
                grid_valid = grid_cell_values[valid_mask]

                # Check for constant values (correlation undefined)
                if np.std(ts_valid) == 0 or np.std(grid_valid) == 0:
                    return np.nan

                try:
                    # Compute correlation
                    corr, _ = corr_funcs[method](ts_valid, grid_valid)
                    return corr
                except Exception:
                    return np.nan

            # Apply correlation function along time dimension using xarray's apply_ufunc
            # This leverages Dask for parallel processing
            correlation_map = xr.apply_ufunc(
                correlate_grid_cell,
                grid_aligned,
                input_core_dims=[['time']],  # Apply function along time dimension
                vectorize=True,  # Vectorize over latitude/longitude
                dask='parallelized',  # Use Dask for parallel processing
                output_dtypes=[float],
            )

        # Apply absolute value if requested
        if absolute:
            print("  Taking absolute values of correlations...")
            correlation_map = np.abs(correlation_map)

        # Add metadata
        corr_type = 'absolute_' if absolute else ''
        valid_range = [0.0, 1.0] if absolute else [-1.0, 1.0]

        correlation_map.name = f'{corr_type}{method}_correlation'
        correlation_map.attrs = {
            'long_name': f'{method.capitalize()} Correlation Coefficient' + (' (Absolute)' if absolute else ''),
            'description': f'Spatial correlation between time series and gridded data',
            'method': method,
            'absolute': absolute,
            'min_periods': min_periods,
            'n_time_points': n_times,
            'valid_range': valid_range
        }

        # Compute if requested
        if compute:
            print("  Computing correlations...")
            correlation_map = correlation_map.compute()

            if use_lag:
                lag_map = lag_map.compute()

            # Report statistics
            valid_cells = np.sum(~np.isnan(correlation_map.values))
            total_cells = correlation_map.size
            print(f"✓ Correlation analysis complete")
            print(f"  Valid grid cells: {valid_cells}/{total_cells} ({100*valid_cells/total_cells:.1f}%)")

            if valid_cells > 0:
                corr_vals = correlation_map.values[~np.isnan(correlation_map.values)]
                print(f"  Correlation range: [{np.min(corr_vals):.3f}, {np.max(corr_vals):.3f}]")
                print(f"  Mean correlation: {np.mean(corr_vals):.3f}")

                if use_lag:
                    lag_vals = lag_map.values[~np.isnan(lag_map.values)]
                    print(f"  Lag range: [{int(np.min(lag_vals))} to {int(np.max(lag_vals))}] time steps")
                    print(f"  Mean lag: {np.mean(lag_vals):.1f} time steps")

        # Return Dataset with correlation and lag if use_lag=True, otherwise just correlation
        if use_lag:
            # Add metadata to lag map
            lag_map.name = 'lag'
            lag_map.attrs = {
                'long_name': 'Time Lag at Maximum Correlation',
                'description': 'Lag in time steps where maximum correlation occurs',
                'units': 'time steps',
                'positive_lag_means': 'grid leads timeseries',
                'negative_lag_means': 'timeseries leads grid',
                'max_lag_searched': max_lag
            }

            # Create Dataset with both correlation and lag
            result = xr.Dataset({
                'correlation': correlation_map,
                'lag': lag_map
            })
            return result
        else:
            return correlation_map

    def save_to_csv(self, data, output_path, wide_format=True):
        """
        Save xarray Dataset or DataArray to CSV file(s).

        Parameters
        ----------
        data : xarray.Dataset or xarray.DataArray
            Data to save
        output_path : str
            Output file path. For Datasets with multiple variables:
            - If wide_format=True: saves to single file at this path
            - If wide_format=False: saves multiple files with variable names appended
        wide_format : bool, optional
            If True (default), save all variables in one wide CSV.
            If False, save each variable to a separate CSV file.

        Examples
        --------
        >>> # Save statistics to CSV
        >>> stats = hmd.compute_timeseries_stats(band_levels, resolution='1h')
        >>> hmd.save_to_csv(stats, 'statistics.csv')

        >>> # Save each variable separately
        >>> hmd.save_to_csv(stats, 'statistics.csv', wide_format=False)
        """
        from pathlib import Path

        # Compute if lazy
        if hasattr(data, 'compute'):
            print("Computing data before saving...")
            data = data.compute()

        if isinstance(data, xr.DataArray):
            # Single array - convert to DataFrame and save
            df = data.to_dataframe()
            df.to_csv(output_path)
            print(f"✓ Saved to {output_path}")

        elif isinstance(data, xr.Dataset):
            if wide_format:
                # Save all variables in one wide CSV
                df = data.to_dataframe()
                df.to_csv(output_path)
                print(f"✓ Saved {len(data.data_vars)} variables to {output_path}")
            else:
                # Save each variable to separate file
                output_path = Path(output_path)
                stem = output_path.stem
                suffix = output_path.suffix
                parent = output_path.parent

                for var_name in data.data_vars:
                    var_path = parent / f"{stem}_{var_name}{suffix}"
                    df = data[var_name].to_dataframe()
                    df.to_csv(var_path)
                    print(f"✓ Saved {var_name} to {var_path}")
        else:
            raise ValueError("data must be xarray.DataArray or Dataset")

    @staticmethod
    def _integrate_band_db(db_values, freq_axis='frequency'):
        """Properly integrate dB values across a frequency band"""
        linear = 10 ** (db_values / 10)
        integrated_linear = linear.sum(dim=freq_axis)
        return 10 * np.log10(integrated_linear)

    def _check_loaded(self):
        """Check if dataset is loaded"""
        if self.ds is None:
            raise ValueError("No data loaded. Use .load_nc_files() first.")

    def _setup_dask(self):
        """Setup Dask cluster for parallel processing"""
        try:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2 if self.use_processes else 4,
                processes=self.use_processes,
                memory_limit=self.memory_per_worker,
                local_directory=self.temp_directory
            )
            self.client = Client(cluster)

            mode = "processes" if self.use_processes else "threads"
            print(f"✓ Dask cluster started ({mode} mode)")
            print(f"  Dashboard: {self.client.dashboard_link}")

        except RuntimeError as e:
            if "multiprocessing" in str(e) or "bootstrapping" in str(e):
                print("\n" + "=" * 70)
                print("ERROR: Windows multiprocessing issue detected!")
                print("=" * 70)
                print("\nSOLUTION: Use threaded mode")
                print("  hmd = HMD(use_processes=False)")
                print("=" * 70 + "\n")
                print("→ Automatically falling back to threaded mode...")
                self.use_processes = False
                self._setup_dask()
            else:
                raise

    @staticmethod
    def _filter_files_by_date(files, time_range):
        """Filter files based on date in filename (end date exclusive)"""
        import pandas as pd

        if time_range is None:
            return files, 0

        start_date = pd.to_datetime(time_range[0]).to_pydatetime()
        end_date = pd.to_datetime(time_range[1]).to_pydatetime()

        filtered_files = []
        skipped_count = 0
        unparseable_count = 0

        for file in files:
            file_date = HMD._parse_date_from_filename(file.name)

            if file_date is None:
                filtered_files.append(file)
                unparseable_count += 1
            elif start_date <= file_date < end_date:  # End date is exclusive
                filtered_files.append(file)
            else:
                skipped_count += 1

        if unparseable_count > 0:
            print(f"  Note: {unparseable_count} file(s) don't match date pattern")

        return filtered_files, skipped_count

    @staticmethod
    def _parse_date_from_filename(filename):
        """Extract date from filename ending with '_YYYYMMDD.nc'"""
        import re
        from datetime import datetime

        filename_str = str(filename)
        match = re.search(r'_(\d{8})\.nc$', filename_str)

        if match:
            date_str = match.group(1)
            try:
                return datetime.strptime(date_str, '%Y%m%d')
            except ValueError:
                return None
        return None

    def close(self):
        """Close the Dask client and clean up resources"""
        if self.client is not None:
            print("Closing Dask client...")
            self.client.close()
            self.client = None
            print("✓ Dask client closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __repr__(self):
        """String representation"""
        if self.ds is None:
            return "HMD(no data loaded)"
        else:
            time_span = len(self.ds.time)
            freq_bins = len(self.ds.frequency)
            size_gb = self.ds.nbytes / 1e9
            return (f"HMD(time={time_span}, frequency={freq_bins}, "
                    f"size={size_gb:.2f}GB)")

    # def analyze_frequency_bands(self, bands_dict, time_resolution='1H'):
    #     """
    #     Analyze multiple frequency bands efficiently.
    #
    #     Parameters
    #     ----------
    #     bands_dict : dict
    #         Dictionary of band names and frequency ranges (Hz)
    #     time_resolution : str
    #         Temporal resolution ('1T', '1H', '1D', etc.)
    #
    #     Returns
    #     -------
    #     dict
    #
    #     Examples
    #     --------
    #     >>> bands = {'low': (100, 500), 'mid': (500, 2000)}
    #     >>> results = hmd.analyze_frequency_bands(bands, time_resolution='1H')
    #     """
    #     self._check_loaded()
    #
    #     results = {}
    #
    #     for band_name, (f_min, f_max) in bands_dict.items():
    #         print(f"Processing band '{band_name}': {f_min}-{f_max} Hz")
    #
    #         band_data = self.ds.sel(frequency=slice(f_min, f_max))
    #
    #         linear_power = 10 ** (band_data.psd / 10)
    #         integrated_power = linear_power.sum(dim='frequency')
    #         integrated_level = 10 * np.log10(integrated_power)
    #
    #         stats = {
    #             'integrated': integrated_level,
    #             'max': band_data.psd.max(dim='frequency'),
    #             'median': band_data.psd.median(dim='frequency'),
    #             'std': band_data.psd.std(dim='frequency')
    #         }
    #
    #         if time_resolution != '1T':
    #             stats = {k: v.resample(time=time_resolution).mean()
    #                      for k, v in stats.items()}
    #
    #         results[band_name] = stats
    #
    #     print("Computing statistics...")
    #     computed_results = {}
    #     for band_name in results:
    #         computed_results[band_name] = dask.compute(results[band_name])[0]
    #
    #     print("✓ Band analysis complete")
    #     return computed_results


    # def to_zarr(self, output_path, chunks=None):
    #     """Convert dataset to Zarr format"""
    #     self._check_loaded()
    #
    #     if chunks is not None:
    #         ds_rechunked = self.ds.chunk(chunks)
    #     else:
    #         ds_rechunked = self.ds
    #
    #     print(f"Converting to Zarr format: {output_path}")
    #     ds_rechunked.to_zarr(output_path, mode='w', consolidated=True)
    #     print("✓ Conversion complete")

    # def analyze_temporal_patterns(self, freq_band=None):
    #     """Analyze daily and hourly patterns"""
    #     self._check_loaded()
    #
    #     print("Analyzing temporal patterns...")
    #
    #     linear_power = 10 ** (self.ds.psd / 10)
    #
    #     if freq_band is not None:
    #         linear_power = linear_power.sel(
    #             frequency=slice(freq_band[0], freq_band[1])
    #         )
    #
    #     integrated_power = linear_power.sum(dim='frequency')
    #     data_db = 10 * np.log10(integrated_power)
    #
    #     hour_of_day = data_db.groupby('time.hour').mean()
    #     day_of_week = data_db.groupby('time.dayofweek').mean()
    #     daily_mean = data_db.resample(time='1D').mean()
    #     hourly_mean = data_db.resample(time='1H').mean()
    #
    #     result = dask.compute({
    #         'hour_of_day': hour_of_day,
    #         'day_of_week': day_of_week,
    #         'daily': daily_mean,
    #         'hourly': hourly_mean
    #     })[0]
    #
    #     print("✓ Temporal pattern analysis complete")
    #     return result

 # def resample_temporal(self, resolution='1H', freq_band=None):
    #     """Resample data to coarser temporal resolution"""
    #     self._check_loaded()
    #
    #     if freq_band is not None:
    #         band_data = self.ds.sel(frequency=slice(freq_band[0], freq_band[1]))
    #         linear_power = 10 ** (band_data.psd / 10)
    #         integrated_power = linear_power.sum(dim='frequency')
    #         data_db = 10 * np.log10(integrated_power)
    #         resampled = data_db.resample(time=resolution).mean()
    #     else:
    #         resampled = self.ds.psd.resample(time=resolution).mean()
    #
    #     return resampled

if __name__ == "__main__":
    # Example: Extract band levels and use plot_timeseries
    hmd = HMD(n_workers=4)
    hmd.load_nc_files('path/to/deployment_01')

    # Extract bands
    freq_bands = [[100], [50, 300], [1000], [2000, 10000]]
    names = ['100Hz', 'ship', '1kHz', 'mammal']
    result = hmd.extract_band_levels(freq_bands, band_names=names)

    # Plot with new unified method - overlaid (default)
    hmd.plot_timeseries(result.compute())

    # Plot as separate subplots
    hmd.plot_timeseries(result.compute(), overlay=False)

    # Plot specific bands only
    hmd.plot_timeseries(result.compute(), band_names=['ship', 'mammal'])

    # Plot single band
    hmd.plot_timeseries(result['ship'].compute())



    # # In plot_hmd_test.py
    #
    # # Example 1: Single dataset
    # stats_timeseries = hmd.compute_timeseries_stats(band_levels, resolution='1h')
    # hmd.plot_interactive_advanced(stats_timeseries.compute())
    #
    # # Example 2: Plot raw band levels
    # band_levels_computed = band_levels.compute()
    # hmd.plot_interactive_advanced(band_levels_computed, title='Raw Band Levels')
    #
    # # Example 3: Compare raw data with statistics (MULTIPLE DATASETS)
    # band_levels_computed = band_levels.compute()
    # stats_computed = stats_timeseries.compute()
    #
    # hmd.plot_interactive_advanced(
    #     [band_levels_computed, stats_computed],
    #     title='Raw Data vs Hourly Statistics'
    # )
    #
    # # Example 4: Compare multiple time periods or deployments
    # deployment1 = hmd.subset(time_range=('2018-11-01', '2018-11-15')).compute()
    # deployment2 = hmd.subset(time_range=('2018-11-16', '2018-11-30')).compute()
    #
    # hmd.plot_interactive_advanced(
    #     [deployment1, deployment2],
    #     title='Deployment Comparison'
    # )
    #
    # # Example 5: Save to HTML without opening browser
    # hmd.plot_interactive_advanced(
    #     [band_levels_computed, stats_computed],
    #     save_html='comparison_dashboard.html',
    #     show=False
    # )