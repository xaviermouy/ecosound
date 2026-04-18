# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Grid Plotter
Creates maps from gridded xarray data (time, latitude, longitude).
Works with any gridded oceanographic or environmental data.

"""

import folium
from folium import plugins
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, Tuple, List
from pathlib import Path
import tempfile
import webbrowser
import os
import branca.colormap as cm
from datetime import datetime


class GridPlotter:
    """
    Interactive map plotter for gridded xarray data with time, latitude, and longitude dimensions.
    Supports time-slider visualization and static summary maps for any type of gridded data.
    """

    # Available basemap tiles
    BASEMAPS = {
        'OpenStreetMap': 'OpenStreetMap',
        'CartoDB Positron': 'CartoDB positron',
        'CartoDB Dark': 'CartoDB dark_matter',
        'Esri WorldImagery': 'Esri WorldImagery',
        'Esri Ocean': 'Esri.OceanBasemap',
        'OpenTopoMap': 'OpenTopoMap',
    }

    def __init__(self,
                 center_lat: Optional[float] = None,
                 center_lon: Optional[float] = None,
                 zoom_start: int = 8,
                 basemap: str = 'CartoDB Positron'):
        """
        Initialize the grid plotter.

        Args:
            center_lat: Center latitude (auto-calculated if None)
            center_lon: Center longitude (auto-calculated if None)
            zoom_start: Initial zoom level (default: 8)
            basemap: Basemap style (see BASEMAPS)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start
        self.basemap = self.BASEMAPS.get(basemap, 'CartoDB positron')
        self.map = None

    def plot_timeseries_grid(self,
                            data_grid: xr.DataArray,
                            label: str = 'Data',
                            colormap: str = 'YlOrRd',
                            opacity: float = 0.6,
                            show_colorbar: bool = True,
                            playback_speed_ms: int = 1000,
                            time_step: int = 1) -> folium.Map:
        """
        Create interactive map with time slider for gridded data.

        Args:
            data_grid: xarray DataArray with dimensions (time, latitude, longitude)
            label: Label describing the data being displayed (used in plots and popups)
            colormap: Matplotlib colormap name (default: 'YlOrRd')
            opacity: Grid cell opacity (default: 0.6)
            show_colorbar: Show color legend (default: True)
            playback_speed_ms: Playback speed in milliseconds (default: 1000 = 1 second per frame)
                              Lower values = faster playback (e.g., 500 = 0.5 sec, 200 = 0.2 sec)
            time_step: Show every Nth time step (default: 1 = all time steps)
                      Use higher values to reduce file size (e.g., 6 = every 6 hours for hourly data)

        Returns:
            Folium map object with time slider
        """
        if data_grid is None:
            print("Warning: No data provided")
            return None

        # Get dimensions and subsample time if requested
        times = data_grid.coords['time'].values[::time_step]
        lats = data_grid.coords['latitude'].values
        lons = data_grid.coords['longitude'].values

        # Subsample the data grid
        data_grid = data_grid.isel(time=slice(None, None, time_step))

        print(f"Plotting {len(times)} time steps (time_step={time_step})")

        # Calculate map center if not provided
        if self.center_lat is None or self.center_lon is None:
            self.center_lat = float(np.nanmean(lats))
            self.center_lon = float(np.nanmean(lons))

        # Create base map
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=self.basemap
        )

        # Get value range for consistent coloring across time
        vmin = float(np.nanmin(data_grid.values))
        vmax = float(np.nanmax(data_grid.values))

        # Create colormap
        cmap = plt.cm.get_cmap(colormap)
        if vmax > vmin:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

        # Create separate FeatureGroups for each time step
        # This approach shows only one time step at a time without accumulation
        time_layers = []

        for t_idx, time in enumerate(times):
            time_str = pd.Timestamp(time).strftime('%Y-%m-%d %H:%M:%S')

            # Create a feature group for this time step with unique ID
            fg = folium.FeatureGroup(name=f"time_layer_{t_idx}", show=False)

            # Add grid cells for this time step
            lat_spacing = lats[1] - lats[0] if len(lats) > 1 else 0.1
            lon_spacing = lons[1] - lons[0] if len(lons) > 1 else 0.1

            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    value = data_grid.values[t_idx, lat_idx, lon_idx]

                    # Skip NaN values
                    if np.isnan(value):
                        continue

                    # Convert to Python native floats for JSON serialization
                    lat_min = float(lat - lat_spacing/2)
                    lat_max = float(lat + lat_spacing/2)
                    lon_min = float(lon - lon_spacing/2)
                    lon_max = float(lon + lon_spacing/2)

                    # Get color
                    color = mcolors.rgb2hex(cmap(norm(value)))

                    # Create rectangle
                    folium.Rectangle(
                        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=opacity,
                        weight=1,
                        opacity=0.8,
                        popup=f"Time: {time_str}<br>Lat: {float(lat):.3f}<br>Lon: {float(lon):.3f}<br>{label}: {float(value):.2f}"
                    ).add_to(fg)

            # Add feature group to map and store reference
            fg.add_to(self.map)
            time_layers.append((time_str, fg))

        # Add custom time slider control with JavaScript
        if len(time_layers) > 1:
            time_strings = [t[0] for t in time_layers]  # Extract time strings

            # Store layer IDs for JavaScript access
            layer_var_names = [t[1].get_name() for t in time_layers]

            # Create HTML for time slider
            slider_html = f'''
            <div id="time-slider-container" style="
                position: fixed;
                bottom: 50px;
                left: 50%;
                transform: translateX(-50%);
                background-color: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                z-index: 1000;
                min-width: 500px;
            ">
                <div style="margin-bottom: 10px; font-weight: bold; text-align: center;">
                    Time: <span id="current-time">{time_strings[0]}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <button id="prev-button" style="padding: 5px 12px; cursor: pointer; font-size: 14px;" title="Previous frame">◀</button>
                    <button id="play-button" style="padding: 5px 15px; cursor: pointer;">▶ Play</button>
                    <button id="next-button" style="padding: 5px 12px; cursor: pointer; font-size: 14px;" title="Next frame">▶</button>
                    <input type="range" id="time-slider" min="0" max="{len(time_strings)-1}" value="0"
                           style="flex: 1; cursor: pointer;">
                    <span id="time-index">1/{len(time_strings)}</span>
                </div>
            </div>
            '''

            # Create JavaScript for controlling layers
            time_strings_js = str(time_strings)
            layer_names_js = str(layer_var_names)
            map_var_name = self.map.get_name()

            slider_js = f'''
            <script>
            (function() {{
                var timeList = {time_strings_js};
                var layerNames = {layer_names_js};
                var currentIndex = 0;
                var isPlaying = false;
                var playInterval = null;

                function showTimeStep(index) {{
                    // Get map object
                    var theMap = window['{map_var_name}'];
                    if (!theMap) {{
                        console.error('Map not found: {map_var_name}');
                        console.log('Available window objects:', Object.keys(window).filter(k => k.includes('map')));
                        return;
                    }}

                    // Hide all layers
                    for (var i = 0; i < layerNames.length; i++) {{
                        var layer = window[layerNames[i]];
                        if (layer && theMap.hasLayer(layer)) {{
                            theMap.removeLayer(layer);
                        }}
                    }}

                    // Show selected layer
                    var selectedLayer = window[layerNames[index]];
                    if (selectedLayer) {{
                        theMap.addLayer(selectedLayer);
                        console.log('Showing layer ' + index + ': ' + layerNames[index]);
                    }} else {{
                        console.error('Layer not found: ' + layerNames[index]);
                    }}

                    // Update display
                    document.getElementById('current-time').textContent = timeList[index];
                    document.getElementById('time-index').textContent = (index + 1) + '/' + timeList.length;
                    currentIndex = index;
                }}

                function togglePlay() {{
                    isPlaying = !isPlaying;
                    var playButton = document.getElementById('play-button');

                    if (isPlaying) {{
                        playButton.textContent = '⏸ Pause';
                        playInterval = setInterval(function() {{
                            currentIndex = (currentIndex + 1) % timeList.length;
                            document.getElementById('time-slider').value = currentIndex;
                            showTimeStep(currentIndex);
                        }}, {playback_speed_ms});
                    }} else {{
                        playButton.textContent = '▶ Play';
                        if (playInterval) {{
                            clearInterval(playInterval);
                            playInterval = null;
                        }}
                    }}
                }}

                function goToPrevious() {{
                    if (isPlaying) {{
                        togglePlay();  // Stop playing
                    }}
                    currentIndex = (currentIndex - 1 + timeList.length) % timeList.length;
                    document.getElementById('time-slider').value = currentIndex;
                    showTimeStep(currentIndex);
                }}

                function goToNext() {{
                    if (isPlaying) {{
                        togglePlay();  // Stop playing
                    }}
                    currentIndex = (currentIndex + 1) % timeList.length;
                    document.getElementById('time-slider').value = currentIndex;
                    showTimeStep(currentIndex);
                }}

                // Event listeners
                document.getElementById('time-slider').addEventListener('input', function(e) {{
                    if (isPlaying) {{
                        togglePlay();  // Stop playing if user moves slider
                    }}
                    showTimeStep(parseInt(e.target.value));
                }});

                document.getElementById('play-button').addEventListener('click', togglePlay);
                document.getElementById('prev-button').addEventListener('click', goToPrevious);
                document.getElementById('next-button').addEventListener('click', goToNext);

                // Initialize - show first time step after brief delay
                setTimeout(function() {{
                    console.log('Initializing time slider with ' + layerNames.length + ' layers');
                    showTimeStep(0);
                }}, 500);
            }})();
            </script>
            '''

            # Add HTML and JavaScript to map
            self.map.get_root().html.add_child(folium.Element(slider_html))
            self.map.get_root().html.add_child(folium.Element(slider_js))

        # Add colorbar
        if show_colorbar:
            colorbar = cm.LinearColormap(
                colors=[mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 10)],
                vmin=vmin,
                vmax=vmax,
                caption=f"{label}".capitalize()
            )
            colorbar.add_to(self.map)

        # Add title
        start_time = pd.Timestamp(times[0]).strftime('%Y-%m-%d %H:%M')
        end_time = pd.Timestamp(times[-1]).strftime('%Y-%m-%d %H:%M')
        title = f"{label} Grid: {start_time} to {end_time}".capitalize()
        self._add_title(title)

        # Add mouse position display
        self._add_mouse_position()

        # Add scale bar
        self._add_scale_bar()

        # Add layer control
        folium.LayerControl().add_to(self.map)

        return self.map

    def plot_summary_grid(self,
                         data_grid: xr.DataArray,
                         label: str = 'Data',
                         aggregation: str = 'sum',
                         colormap: str = 'YlOrRd',
                         opacity: float = 0.6,
                         show_colorbar: bool = True) -> folium.Map:
        """
        Create static map showing aggregated data over all time periods.

        Args:
            data_grid: xarray DataArray with dimensions (time, latitude, longitude)
            label: Label describing the data being displayed (used in plots and popups)
            aggregation: Aggregation method ('sum', 'mean', 'max') (default: 'sum')
            colormap: Matplotlib colormap name (default: 'YlOrRd')
            opacity: Grid cell opacity (default: 0.6)
            show_colorbar: Show color legend (default: True)

        Returns:
            Folium map object
        """
        if data_grid is None:
            print("Warning: No data provided")
            return None

        # Aggregate over time
        if aggregation == 'sum':
            grid_agg = data_grid.sum(dim='time', skipna=True, min_count=1)
            agg_label = 'Sum of ' + label
        elif aggregation == 'mean':
            grid_agg = data_grid.mean(dim='time', skipna=True)
            agg_label = 'Mean ' + label
        elif aggregation == 'max':
            grid_agg = data_grid.max(dim='time',skipna=True)
            agg_label = 'Max ' + label
        else:
            print(f"Unknown aggregation: {aggregation}, using 'sum'")
            grid_agg = data_grid.sum(dim='time',skipna=True, min_count=1)
            agg_label = 'Sum of ' + label

        # Get dimensions
        lats = grid_agg.coords['latitude'].values
        lons = grid_agg.coords['longitude'].values

        # Calculate map center if not provided
        if self.center_lat is None or self.center_lon is None:
            self.center_lat = float(np.nanmean(lats))
            self.center_lon = float(np.nanmean(lons))

        # Create base map
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=self.basemap
        )

        # Get value range for coloring
        vmin = float(np.nanmin(grid_agg.values))
        vmax = float(np.nanmax(grid_agg.values))

        # Create colormap
        cmap = plt.cm.get_cmap(colormap)
        if vmax > vmin:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

        # Add grid cells
        for lat_idx, lat in enumerate(lats):
            for lon_idx, lon in enumerate(lons):
                value = grid_agg.values[lat_idx, lon_idx]

                # Skip NaN values
                if np.isnan(value):
                    continue

                # Get cell boundaries
                lat_spacing = lats[1] - lats[0] if len(lats) > 1 else 0.1
                lon_spacing = lons[1] - lons[0] if len(lons) > 1 else 0.1

                bounds = [
                    [lat - lat_spacing/2, lon - lon_spacing/2],
                    [lat + lat_spacing/2, lon + lon_spacing/2]
                ]

                # Get color
                color = mcolors.rgb2hex(cmap(norm(value)))

                # Create rectangle
                folium.Rectangle(
                    bounds=bounds,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=opacity,
                    weight=1,
                    opacity=0.8,
                    popup=f"Lat: {lat:.3f}<br>Lon: {lon:.3f}<br>{agg_label}: {value:.2f}"
                ).add_to(self.map)

        # Add colorbar
        if show_colorbar:
            colorbar = cm.LinearColormap(
                colors=[mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 10)],
                vmin=vmin,
                vmax=vmax,
                caption=f'{agg_label}'
            )
            colorbar.add_to(self.map)

        # Add title with time range
        times = data_grid.coords['time'].values
        start_time = pd.Timestamp(times[0]).strftime('%Y-%m-%d %H:%M')
        end_time = pd.Timestamp(times[-1]).strftime('%Y-%m-%d %H:%M')
        title = f"{label} Grid ({agg_label}): {start_time} to {end_time}"
        self._add_title(title)

        # Add mouse position display
        self._add_mouse_position()

        # Add scale bar
        self._add_scale_bar()

        return self.map

    def save(self, filename: str) -> None:
        """
        Save map to HTML file.

        Args:
            filename: Output HTML filename
        """
        if self.map is None:
            raise ValueError("No map to save. Call plot_timeseries_grid() or plot_summary_grid() first.")

        self.map.save(filename)
        print(f"Map saved to: {filename}")

    def display(self) -> None:
        """
        Display the map in the default web browser without saving to a file.
        Creates a temporary HTML file that is deleted after opening.
        """
        if self.map is None:
            raise ValueError("No map to display. Call plot_timeseries_grid() or plot_summary_grid() first.")

        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
            self.map.save(temp_path)

        # Open in default browser
        webbrowser.open('file://' + os.path.abspath(temp_path))
        print(f"Map opened in browser (temporary file: {temp_path})")
        print("Note: Temporary file will remain until you delete it manually or restart your system.")

    def _add_title(self, title: str) -> None:
        """
        Add title to map.

        Args:
            title: Title text
        """
        title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 50px;
                    background-color: white;
                    border: 2px solid grey;
                    z-index: 9999;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            {title}
        </div>
        '''
        self.map.get_root().html.add_child(folium.Element(title_html))

    def _add_mouse_position(self) -> None:
        """
        Add mouse position display (lat/lon) to map.
        Shows coordinates at bottom-left as mouse moves over map.
        """
        # Use Folium's MousePosition plugin
        formatter = "function(num) {return L.Util.formatNum(num, 5) + ' ';};"

        plugins.MousePosition(
            position='bottomleft',
            separator=' | ',
            empty_string='Move mouse over map',
            lng_first=False,  # Show lat first
            num_digits=20,
            prefix='Lat/Lon:',
            lat_formatter=formatter,
            lng_formatter=formatter,
        ).add_to(self.map)

    def _add_scale_bar(self) -> None:
        """
        Add scale bar to map.
        Shows distance scale in both metric and imperial units.
        """
        # Use Folium's MiniMap plugin for scale
        # Actually, use a custom scale control
        folium.plugins.MeasureControl(
            position='bottomright',
            primary_length_unit='kilometers',
            secondary_length_unit='miles',
            primary_area_unit='sqkilometers',
            secondary_area_unit='sqmiles'
        ).add_to(self.map)

    def add_markers(self,
                   locations: List[Tuple[float, float]],
                   labels: Optional[List[str]] = None,
                   colors: Optional[List[str]] = None,
                   icon: str = 'info-sign',
                   popup_text: Optional[List[str]] = None,
                   marker_type: str = 'circle') -> None:
        """
        Add custom markers to the map (e.g., recorder locations).

        Args:
            locations: List of (lat, lon) tuples
            labels: Labels for each marker
            colors: Colors for each marker (default: red)
            icon: Icon name (FontAwesome or Bootstrap icons)
            popup_text: Custom popup text for each marker
            marker_type: 'circle' or 'icon' (default: circle)
        """
        if self.map is None:
            raise ValueError("Must call plot_timeseries_grid() or plot_summary_grid() first to create map")

        if labels is None:
            labels = [f"Location {i+1}" for i in range(len(locations))]

        if colors is None:
            colors = ['red'] * len(locations)

        if popup_text is None:
            popup_text = labels

        for i, (lat, lon) in enumerate(locations):
            if marker_type == 'icon':
                folium.Marker(
                    location=[lat, lon],
                    popup=popup_text[i],
                    tooltip=labels[i],
                    icon=folium.Icon(color=colors[i], icon=icon)
                ).add_to(self.map)
            else:  # circle
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    popup=popup_text[i],
                    tooltip=labels[i],
                    color=colors[i],
                    fill=True,
                    fillColor=colors[i],
                    fillOpacity=0.9,
                    weight=3
                ).add_to(self.map)

    def add_recorder_locations(self,
                              recorder_df: pd.DataFrame,
                              lat_col: str = 'latitude',
                              lon_col: str = 'longitude',
                              name_col: str = 'name',
                              color: str = 'red',
                              icon: str = 'microphone') -> None:
        """
        Add acoustic recorder locations to the map.

        Args:
            recorder_df: DataFrame with recorder locations
            lat_col: Column name for latitude
            lon_col: Column name for longitude
            name_col: Column name for recorder name
            color: Marker color (default: red)
            icon: Icon name (default: microphone)
        """
        locations = list(zip(recorder_df[lat_col], recorder_df[lon_col]))
        labels = recorder_df[name_col].tolist() if name_col in recorder_df.columns else None

        # Create detailed popup text
        popup_text = []
        for idx, row in recorder_df.iterrows():
            html = f"<b>Recorder: {row[name_col]}</b><br>"
            html += f"Lat: {row[lat_col]:.4f}<br>"
            html += f"Lon: {row[lon_col]:.4f}<br>"
            # Add any other columns
            for col in recorder_df.columns:
                if col not in [lat_col, lon_col, name_col]:
                    html += f"{col}: {row[col]}<br>"
            popup_text.append(html)

        self.add_markers(
            locations=locations,
            labels=labels,
            colors=[color] * len(locations),
            icon=icon,
            popup_text=popup_text,
            marker_type='icon'
        )

    def _load_bathymetry_data(self, extent: List[float]) -> Optional[dict]:
        """
        Load bathymetry data for the specified extent using GEBCO or ETOPO data.

        Args:
            extent: [lon_min, lon_max, lat_min, lat_max]

        Returns:
            Dictionary with 'longitude', 'latitude', 'elevation' arrays, or None if failed
        """
        try:
            import pooch
        except ImportError:
            print("Warning: pooch is required for bathymetry data. Install with: pip install pooch")
            return None

        lon_min, lon_max, lat_min, lat_max = extent

        try:
            # Try to load ETOPO1 data (coarser but global)
            print("  Fetching ETOPO1 bathymetry data...")

            # Download ETOPO1 bedrock data (1 arc-minute resolution)
            # Use the actual current hash or None to skip verification
            etopo_file = pooch.retrieve(
                url="https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz",
                known_hash="md5:aff52d6081160379fe5a42067053b35f",  # Updated hash
                progressbar=True,
            )

            # Load with xarray
            bathy_ds = xr.open_dataset(etopo_file)

            # Extract subset for the region of interest
            bathy_subset = bathy_ds.sel(
                x=slice(lon_min, lon_max),
                y=slice(lat_min, lat_max)
            )

            return {
                'longitude': bathy_subset['x'].values,
                'latitude': bathy_subset['y'].values,
                'elevation': bathy_subset['z'].values
            }

        except Exception as e:
            print(f"  Warning: Could not load bathymetry data: {e}")
            return None

    def plot_static_map(self,
                       data_grid: xr.DataArray,
                       timestamp: Union[str, datetime, pd.Timestamp],
                       label: str = 'Data',
                       colormap: str = 'YlOrRd',
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None,
                       cbar_min: Optional[float] = None,
                       cbar_max: Optional[float] = None,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (12, 10),
                       dpi: int = 300,
                       axes_fontsize: int = 10,
                       colorbar_fontsize: int = 12,
                       title_fontsize: int = 14,
                       markers: Optional[List[Tuple[float, float]]] = None,
                       marker_labels: Optional[List[str]] = None,
                       marker_colors: Optional[List[str]] = None,
                       marker_type: str = 'o',
                       marker_size: float = 100,
                       recorder_df: Optional[pd.DataFrame] = None,
                       recorder_lat_col: str = 'latitude',
                       recorder_lon_col: str = 'longitude',
                       recorder_name_col: str = 'name',
                       show_recorder_names: bool = True,
                       recorder_name_fontsize: int = 8,
                       bathymetry_contours: Optional[List[float]] = None,
                       bathymetry_color: str = 'gray',
                       bathymetry_linewidth: float = 1.0,
                       bathymetry_linestyle: str = '--',
                       bathymetry_fontsize: int = 8,
                       save_path: Optional[str] = None,
                       show: bool = True) -> plt.Figure:
        """
        Create publication-quality static map using Cartopy for a specific timestamp.

        Args:
            data_grid: xarray DataArray with dimensions (time, latitude, longitude)
            timestamp: Timestamp to plot (string 'YYYY-MM-DD HH:MM:SS' or datetime object)
            label: Label for the colorbar
            colormap: Matplotlib colormap name (default: 'YlOrRd')
            vmin: Minimum value for colormap (auto-detected if None)
            vmax: Maximum value for colormap (auto-detected if None)
            title: Plot title (auto-generated from timestamp if None)
            figsize: Figure size in inches (default: (12, 10))
            dpi: Resolution in dots per inch (default: 300 for publication quality)
            save_path: Path to save the figure (PNG, PDF, etc.). If None, doesn't save.
            show: Whether to display the figure (default: True)

        Returns:
            matplotlib Figure object

        Example:
            >>> plotter = GridPlotter()
            >>> fig = plotter.plot_static_map(
            ...     data_grid,
            ...     timestamp='2018-01-01 12:00:00',
            ...     label='Sea Surface Temperature (°C)',
            ...     colormap='RdYlBu_r',
            ...     save_path='sst_map.png'
            ... )
        """
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            from matplotlib_scalebar.scalebar import ScaleBar
        except ImportError:
            raise ImportError(
                "Cartopy and matplotlib-scalebar are required for static maps. "
                "Install with: pip install cartopy matplotlib-scalebar"
            )

        # Handle both Dataset and DataArray inputs
        if isinstance(data_grid, xr.Dataset):
            # If Dataset, get the first data variable
            data_vars = list(data_grid.data_vars)
            if len(data_vars) == 0:
                raise ValueError("Dataset has no data variables")
            if len(data_vars) > 1:
                print(f"Warning: Dataset has multiple variables {data_vars}. Using '{data_vars[0]}'")
            data_grid = data_grid[data_vars[0]]

        # Convert timestamp to pandas Timestamp
        ts = pd.Timestamp(timestamp)

        # Check if data has a time dimension
        if 'time' in data_grid.coords:
            # Find the closest time in the data
            times = pd.to_datetime(data_grid.coords['time'].values)
            time_diffs = np.abs(times - ts)
            closest_idx = np.argmin(time_diffs)
            closest_time = times[closest_idx]

            # Get data for this time step
            data_slice = data_grid.isel(time=closest_idx)
            lats = data_grid.coords['latitude'].values
            lons = data_grid.coords['longitude'].values

            # Get the actual data values as numpy array
            data_values = np.array(data_slice)
        else:
            # Data has no time dimension (e.g., already aggregated)
            # Use the data as-is
            closest_time = ts
            lats = data_grid.coords['latitude'].values
            lons = data_grid.coords['longitude'].values
            data_values = np.array(data_grid)

        # Auto-detect vmin/vmax if not provided
        if vmin is None:
            vmin = float(np.nanmin(data_values))
        if vmax is None:
            vmax = float(np.nanmax(data_values))

        # Use cbar_min/cbar_max if provided, otherwise use vmin/vmax
        if cbar_min is None:
            cbar_min = vmin
        if cbar_max is None:
            cbar_max = vmax

        # Create figure with Cartopy projection
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set map extent
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Add ocean background first
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

        # Plot the data
        im = ax.pcolormesh(
            lons, lats, data_values,
            transform=ccrs.PlateCarree(),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            shading='auto'
        )

        # Add coastline and land features ON TOP of data
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10, edgecolor='none')
        ax.coastlines(resolution='10m', color='black', linewidth=0.8, zorder=11)

        # Add bathymetry contours if requested
        if bathymetry_contours is not None and len(bathymetry_contours) > 0:
            try:
                # Try to load GEBCO or ETOPO bathymetry data
                print(f"Loading bathymetry data for contours at {bathymetry_contours} m...")
                bathy = self._load_bathymetry_data(extent)

                if bathy is not None:
                    # Ensure contour levels are in increasing order
                    sorted_contours = sorted(bathymetry_contours)

                    # Plot contours (zorder=14 puts them above land/coast but below markers)
                    contours = ax.contour(
                        bathy['longitude'], bathy['latitude'], bathy['elevation'],
                        levels=sorted_contours,
                        colors=bathymetry_color,
                        linewidths=bathymetry_linewidth,
                        linestyles=bathymetry_linestyle,
                        transform=ccrs.PlateCarree(),
                        zorder=14
                    )
                    # Add contour labels
                    ax.clabel(contours, inline=True, fontsize=bathymetry_fontsize, fmt='%d m')
                    print(f"Added bathymetry contours at {sorted_contours} m")
                else:
                    print("Warning: Could not load bathymetry data")
            except Exception as e:
                print(f"Warning: Could not add bathymetry contours: {e}")

        # Add gridlines with labels
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--',
            zorder=12
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': axes_fontsize}
        gl.ylabel_style = {'size': axes_fontsize}

        # Add colorbar with height matching the axis
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label, fontsize=colorbar_fontsize, weight='bold')
        cbar.ax.tick_params(labelsize=colorbar_fontsize - 2)

        # Add markers if provided
        if markers is not None and len(markers) > 0:
            if marker_colors is None:
                marker_colors = ['red'] * len(markers)
            if marker_labels is None:
                marker_labels = [f"Marker {i+1}" for i in range(len(markers))]

            for i, (lat, lon) in enumerate(markers):
                ax.plot(lon, lat, marker_type,
                       markersize=np.sqrt(marker_size)/2,
                       color=marker_colors[i],
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       transform=ccrs.PlateCarree(),
                       zorder=15,
                       label=marker_labels[i] if i < 10 else None)  # Only label first 10 to avoid clutter

        # Add recorder locations if provided
        if recorder_df is not None and len(recorder_df) > 0:
            lats_rec = recorder_df[recorder_lat_col].values
            lons_rec = recorder_df[recorder_lon_col].values
            names_rec = recorder_df[recorder_name_col].values if recorder_name_col in recorder_df.columns else None

            for i, (lat, lon) in enumerate(zip(lats_rec, lons_rec)):
                label_text = names_rec[i] if names_rec is not None else f"Recorder {i+1}"
                # Plot recorder marker
                ax.plot(lon, lat, marker_type,
                       markersize=marker_size,
                       color='black',
                       markeredgecolor='black',
                       markeredgewidth=1.5,
                       transform=ccrs.PlateCarree(),
                       zorder=16,
                       label=label_text if i < 10 else None)  # Only label first 10

                # Add recorder name as text if requested
                if show_recorder_names:
                    ax.text(lon, lat, f'  {label_text}',
                           fontsize=recorder_name_fontsize,
                           transform=ccrs.PlateCarree(),
                           verticalalignment='center',
                           horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0, edgecolor='none'),
                           zorder=17)

        # Add legend if markers or recorders were added
        if (markers is not None and len(markers) > 0) or (recorder_df is not None and len(recorder_df) > 0):
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=8)

        # Add title
        if title is None:
            title = f"{label}\n{closest_time.strftime('%Y-%m-%d %H:%M UTC')}"
        ax.set_title(title, fontsize=title_fontsize, weight='bold', pad=15)

        # Tight layout
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()

        return fig


# Example usage
if __name__ == '__main__':
    """
    Example usage of AIS Grid Plotter.
    """
    from AIS_query_helper import AISQueryHelper

    # Path to your DuckDB database
    DB_PATH = 'ais_data.duckdb'

    # Create gridded data
    print("Creating gridded vessel counts...")
    with AISQueryHelper(DB_PATH) as ais:
        vessel_grid = ais.create_gridded_vessel_counts(
            start_date='2018-01-01',
            end_date='2018-01-03',
            min_lat=42.0,
            max_lat=44.0,
            min_lon=-70.0,
            max_lon=-68.0,
            width_km=10,
            height_km=10,
            time_resolution_hours=1
        )

    if vessel_grid is not None:
        print("\nExample 1: Time-series grid map with slider")
        print("=" * 60)
        plotter = GridPlotter(basemap='CartoDB Positron', zoom_start=8)
        map1 = plotter.plot_timeseries_grid(
            vessel_grid,
            label='Unique Vessel Count',
            colormap='YlOrRd',
            opacity=0.6
        )
        plotter.save('ais_grid_timeseries.html')
        print("Saved: ais_grid_timeseries.html")

        print("\nExample 2: Summary grid map (total vessels)")
        print("=" * 60)
        plotter2 = GridPlotter(basemap='CartoDB Positron', zoom_start=8)
        map2 = plotter2.plot_summary_grid(
            vessel_grid,
            label='Unique Vessel Count',
            aggregation='sum',
            colormap='YlOrRd',
            opacity=0.6
        )
        plotter2.save('ais_grid_summary_total.html')
        print("Saved: ais_grid_summary_total.html")

        print("\nExample 3: Summary grid map (mean vessels)")
        print("=" * 60)
        plotter3 = GridPlotter(basemap='Esri Ocean', zoom_start=8)
        map3 = plotter3.plot_summary_grid(
            vessel_grid,
            label='Unique Vessel Count',
            aggregation='mean',
            colormap='viridis',
            opacity=0.7
        )
        plotter3.save('ais_grid_summary_mean.html')
        print("Saved: ais_grid_summary_mean.html")

        print("\n✓ All examples completed!")
        print("Open the HTML files in your browser to view the interactive maps.")