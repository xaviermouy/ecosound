# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
AIS Interactive Map Plotter
Creates interactive maps from AIS query results with color-coded markers.
Uses Folium for interactive maps with automatic basemap tiles (no shapefile needed).

Author: Data Processing Script
Date: 2025
"""

import folium
from folium import plugins
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import tempfile
import webbrowser
import os
import duckdb


class AISMapPlotter:
    """
    Interactive map plotter for AIS data using Folium.
    Supports color-coding by categories and adding custom markers.
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

    # Predefined color schemes for categories (kept for backwards compatibility)
    # Now dynamically generated from database when available
    COLOR_SCHEMES = {
        'vessel_category': {
            'Cargo': '#1f77b4',      # Blue
            'Tanker': '#ff7f0e',     # Orange
            'Fishing': '#2ca02c',    # Green
            'Passenger': '#d62728',  # Red
            'Tug': '#9467bd',        # Purple
            'Pilot': '#8c564b',      # Brown
            'SAR': '#e377c2',        # Pink
            'Pleasure': '#7f7f7f',   # Gray
            'HSC': '#bcbd22',        # Yellow-green
            'Military': '#17becf',   # Cyan
            'Other': '#aaaaaa',      # Light gray
            'Unknown': '#000000',    # Black
        }
    }

    def __init__(self,
                 center_lat: Optional[float] = None,
                 center_lon: Optional[float] = None,
                 zoom_start: int = 10,
                 basemap: str = 'OpenStreetMap',
                 db_path: Optional[str] = None):
        """
        Initialize the map plotter.

        Args:
            center_lat: Center latitude (auto-calculated if None)
            center_lon: Center longitude (auto-calculated if None)
            zoom_start: Initial zoom level (default: 10)
            basemap: Basemap style (see BASEMAPS)
            db_path: Path to DuckDB database (optional, for dynamic color schemes)
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start
        self.basemap = self.BASEMAPS.get(basemap, 'OpenStreetMap')
        self.db_path = db_path
        self.map = None

        # Generate color schemes from database if available
        if db_path:
            self._generate_color_schemes_from_db()

    def plot_ais_data(self,
                     gdf: gpd.GeoDataFrame,
                     color_by: Optional[str] = None,
                     color_map: Optional[Dict] = None,
                     continuous_cmap: str = 'viridis',
                     popup_fields: Optional[List[str]] = None,
                     marker_size: int = 5,
                     add_heatmap: bool = False,
                     add_cluster: bool = False,
                     title: Optional[str] = None) -> folium.Map:
        """
        Plot AIS data on an interactive map.

        Args:
            gdf: GeoDataFrame with AIS data
            color_by: Column name to color-code markers (None = blue)
            color_map: Custom color mapping dict (for categorical data)
            continuous_cmap: Colormap for continuous data (matplotlib colormap name)
            popup_fields: Fields to show in popup (default: mmsi, vessel_name, base_datetime)
            marker_size: Size of markers (default: 5)
            add_heatmap: Add heatmap layer (default: False)
            add_cluster: Cluster markers (good for large datasets)
            title: Map title (auto-generated with date range if None)

        Returns:
            Folium map object
        """
        if gdf.empty:
            print("Warning: Empty GeoDataFrame provided")
            return None

        # Calculate map center if not provided
        if self.center_lat is None or self.center_lon is None:
            bounds = gdf.total_bounds
            self.center_lat = (bounds[1] + bounds[3]) / 2
            self.center_lon = (bounds[0] + bounds[2]) / 2

        # Create map
        self.map = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=self.basemap
        )

        # Generate title with date range if not provided
        if title is None and 'base_datetime' in gdf.columns:
            min_date = gdf['base_datetime'].min()
            max_date = gdf['base_datetime'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                title = f"AIS Data: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            else:
                title = "AIS Data"
        elif title is None:
            title = "AIS Data"

        # Add title to map
        self._add_title(title)

        # Add mouse position display
        self._add_mouse_position()

        # Add scale bar
        self._add_scale_bar()

        # Default popup fields
        if popup_fields is None:
            popup_fields = ['mmsi', 'vessel_name', 'base_datetime', 'sog', 'cog']
            # Filter to only existing columns
            popup_fields = [f for f in popup_fields if f in gdf.columns]

        # Determine coloring strategy and check if categorical
        is_categorical = False
        if color_by is None:
            # Single color
            colors = ['blue'] * len(gdf)
            category_groups = None
        elif color_by in gdf.columns:
            if gdf[color_by].dtype == 'object' or gdf[color_by].dtype.name == 'category':
                # Categorical coloring - use feature groups for toggling
                is_categorical = True
                colors = self._get_categorical_colors(gdf[color_by], color_map)
                # Group data by category
                category_groups = {}
                for cat in gdf[color_by].unique():
                    if pd.notna(cat):
                        category_groups[cat] = gdf[gdf[color_by] == cat]
            else:
                # Continuous coloring
                colors = self._get_continuous_colors(gdf[color_by], continuous_cmap)
                category_groups = None
        else:
            print(f"Warning: Column '{color_by}' not found. Using blue.")
            colors = ['blue'] * len(gdf)
            category_groups = None

        # Add markers - use feature groups if categorical for toggleability
        if is_categorical and category_groups is not None and not add_cluster:
            # Create a feature group for each category (toggleable)
            # Sort categories for consistent ordering
            sorted_categories = sorted(category_groups.keys(), key=str)

            for category in sorted_categories:
                cat_gdf = category_groups[category]
                # Create feature group with explicit overlay=True for layer control
                fg = folium.FeatureGroup(name=str(category), overlay=True, control=True, show=True)

                # Get color for this category
                cat_color = color_map.get(category, '#808080') if color_map else '#808080'

                for idx, row in cat_gdf.iterrows():
                    popup_html = self._create_popup_html(row, popup_fields)
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=marker_size,
                        popup=folium.Popup(popup_html, max_width=300),
                        color=cat_color,
                        fill=True,
                        fillColor=cat_color,
                        fillOpacity=0.7,
                        weight=1
                    ).add_to(fg)

                fg.add_to(self.map)
        else:
            # Standard approach - all markers in one layer
            if add_cluster:
                marker_cluster = plugins.MarkerCluster().add_to(self.map)
                parent = marker_cluster
            else:
                parent = self.map

            for idx, row in gdf.iterrows():
                popup_html = self._create_popup_html(row, popup_fields)
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=marker_size,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=colors[idx],
                    fill=True,
                    fillColor=colors[idx],
                    fillOpacity=0.7,
                    weight=1
                ).add_to(parent)

        # Add heatmap layer if requested
        if add_heatmap:
            heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf.iterrows()]
            plugins.HeatMap(heat_data).add_to(self.map)

        # Add legend if color-coding
        if color_by is not None and color_map is not None:
            self._add_legend(color_map, color_by)

        # Add layer control (allows toggling feature groups for categorical data)
        if is_categorical:
            folium.LayerControl(collapsed=False).add_to(self.map)

        return self.map

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
            raise ValueError("Must call plot_ais_data() first to create map")

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

    def save(self, filename: str) -> None:
        """
        Save map to HTML file.

        Args:
            filename: Output HTML filename
        """
        if self.map is None:
            raise ValueError("No map to save. Call plot_ais_data() first.")

        self.map.save(filename)
        print(f"Map saved to: {filename}")

    def display(self) -> None:
        """
        Display the map in the default web browser without saving to a file.
        Creates a temporary HTML file that is deleted after opening.
        """
        if self.map is None:
            raise ValueError("No map to display. Call plot_ais_data() first.")

        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
            self.map.save(temp_path)

        # Open in default browser
        webbrowser.open('file://' + os.path.abspath(temp_path))
        print(f"Map opened in browser (temporary file: {temp_path})")
        print("Note: Temporary file will remain until you delete it manually or restart your system.")

    def _get_categorical_colors(self,
                                series: pd.Series,
                                color_map: Optional[Dict] = None) -> List[str]:
        """
        Get colors for categorical data.

        Args:
            series: Pandas series with categorical data
            color_map: Custom color mapping

        Returns:
            List of color strings
        """
        # Use provided color map or generate one
        if color_map is None:
            unique_vals = series.unique()
            # Use tab10 colormap for up to 10 categories
            if len(unique_vals) <= 10:
                cmap = plt.cm.tab10
                color_map = {val: mcolors.rgb2hex(cmap(i))
                            for i, val in enumerate(unique_vals)}
            else:
                # Use tab20 for more categories
                cmap = plt.cm.tab20
                color_map = {val: mcolors.rgb2hex(cmap(i % 20))
                            for i, val in enumerate(unique_vals)}

        # Map values to colors
        colors = [color_map.get(val, '#808080') for val in series]
        return colors

    def _get_continuous_colors(self,
                              series: pd.Series,
                              cmap_name: str = 'viridis') -> List[str]:
        """
        Get colors for continuous data.

        Args:
            series: Pandas series with continuous data
            cmap_name: Matplotlib colormap name

        Returns:
            List of color strings
        """
        # Normalize values to 0-1
        vmin, vmax = series.min(), series.max()
        if vmax == vmin:
            return ['blue'] * len(series)

        normalized = (series - vmin) / (vmax - vmin)

        # Get colormap
        cmap = plt.cm.get_cmap(cmap_name)

        # Map to colors
        colors = [mcolors.rgb2hex(cmap(val)) for val in normalized]
        return colors

    def _create_popup_html(self, row: pd.Series, fields: List[str]) -> str:
        """
        Create HTML content for popup.

        Args:
            row: DataFrame row
            fields: Fields to include in popup

        Returns:
            HTML string
        """
        html = "<div style='font-family: Arial; font-size: 12px;'>"

        for field in fields:
            if field in row.index and pd.notna(row[field]):
                value = row[field]
                # Format datetime
                if isinstance(value, pd.Timestamp):
                    value = value.strftime('%Y-%m-%d %H:%M:%S')
                # Format floats
                elif isinstance(value, (float, np.floating)):
                    value = f"{value:.2f}"

                html += f"<b>{field}:</b> {value}<br>"

        html += "</div>"
        return html

    def _add_legend(self, color_map: Dict, title: str) -> None:
        """
        Add legend to map (draggable, larger, with single-line entries).

        Args:
            color_map: Dictionary mapping categories to colors
            title: Legend title
        """
        # Add JavaScript for dragging functionality
        drag_script = '''
        <script>
        function dragElement(elmnt) {
            var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            var header = document.getElementById(elmnt.id + "header");
            if (header) {
                header.onmousedown = dragMouseDown;
            } else {
                elmnt.onmousedown = dragMouseDown;
            }

            function dragMouseDown(e) {
                e = e || window.event;
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
            }

            function elementDrag(e) {
                e = e || window.event;
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
                elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
            }

            function closeDragElement() {
                document.onmouseup = null;
                document.onmousemove = null;
            }
        }
        </script>
        '''

        legend_html = f'''
        {drag_script}
        <div id="maplegend" style="position: fixed;
                    top: 10px; right: 10px;
                    min-width: 300px;
                    max-width: 500px;
                    max-height: 80vh;
                    overflow-y: auto;
                    background-color: white;
                    border: 2px solid grey;
                    z-index: 9999;
                    font-size: 13px;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <div id="maplegendheader" style="cursor: move;
                                             padding-bottom: 5px;
                                             margin-bottom: 10px;
                                             border-bottom: 2px solid #ccc;
                                             font-weight: bold;
                                             font-size: 15px;">
                {title}
                <span style="float: right; color: #999; font-size: 12px;">(drag to move)</span>
            </div>
        '''

        for category, color in color_map.items():
            legend_html += f'''
            <div style="margin: 4px 0;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        display: flex;
                        align-items: center;"
                 title="{category}">
                <span style="background-color: {color};
                             width: 18px;
                             height: 18px;
                             display: inline-block;
                             border: 1px solid black;
                             margin-right: 8px;
                             flex-shrink: 0;">
                </span>
                <span style="flex-grow: 1; overflow: hidden; text-overflow: ellipsis;">
                    {category}
                </span>
            </div>
            '''

        legend_html += '''
        </div>
        <script>
        dragElement(document.getElementById("maplegend"));
        </script>
        '''

        self.map.get_root().html.add_child(folium.Element(legend_html))

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
                    font-size: 18px;
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

    def _generate_color_schemes_from_db(self) -> None:
        """
        Generate color schemes dynamically from the vessel_type_lookup table.
        Creates color mappings for both vessel categories and vessel type names.
        """
        try:
            conn = duckdb.connect(self.db_path, read_only=True)

            # Get all unique categories and their vessel types
            query = """
                SELECT DISTINCT category, description
                FROM vessel_type_lookup
                ORDER BY category, description
            """
            df = conn.execute(query).df()
            conn.close()

            # Get unique categories
            unique_categories = df['category'].unique()

            # Generate colors for categories using a consistent colormap
            category_colormap = {}
            if len(unique_categories) <= 10:
                cmap = plt.cm.tab10
            elif len(unique_categories) <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.tab20c

            for i, category in enumerate(sorted(unique_categories)):
                category_colormap[category] = mcolors.rgb2hex(cmap(i % cmap.N))

            # Override with existing COLOR_SCHEMES where available for consistency
            for category, color in self.COLOR_SCHEMES.get('vessel_category', {}).items():
                if category in category_colormap:
                    category_colormap[category] = color

            # Update the class COLOR_SCHEMES
            self.COLOR_SCHEMES['vessel_category'] = category_colormap

            # Generate colors for vessel type names (descriptions)
            # Map each vessel type to its category color
            vessel_type_colormap = {}
            for _, row in df.iterrows():
                vessel_type_colormap[row['description']] = category_colormap.get(row['category'], '#808080')

            self.COLOR_SCHEMES['vessel_type_name'] = vessel_type_colormap

            print(f"Generated color schemes for {len(category_colormap)} categories and {len(vessel_type_colormap)} vessel types from database")

        except Exception as e:
            print(f"Warning: Could not generate color schemes from database: {e}")
            print("Using default color schemes")

    @staticmethod
    def get_color_scheme_from_db(db_path: str, scheme_type: str = 'category') -> Dict[str, str]:
        """
        Static method to get color scheme from database without creating a plotter instance.

        Args:
            db_path: Path to DuckDB database
            scheme_type: 'category' for vessel categories or 'type_name' for vessel type names

        Returns:
            Dictionary mapping vessel types/categories to hex colors
        """
        try:
            conn = duckdb.connect(db_path, read_only=True)

            query = """
                SELECT DISTINCT category, description
                FROM vessel_type_lookup
                ORDER BY category, description
            """
            df = conn.execute(query).df()
            conn.close()

            if scheme_type == 'category':
                unique_values = df['category'].unique()
            else:  # type_name
                unique_values = df['description'].unique()

            # Generate colors
            if len(unique_values) <= 10:
                cmap = plt.cm.tab10
            elif len(unique_values) <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.tab20c

            color_map = {}
            for i, value in enumerate(sorted(unique_values)):
                color_map[value] = mcolors.rgb2hex(cmap(i % cmap.N))

            # For type_name, use category colors
            if scheme_type == 'type_name':
                # First get category colors
                category_colors = AISMapPlotter.get_color_scheme_from_db(db_path, 'category')
                # Then map type names to their category colors
                for _, row in df.iterrows():
                    color_map[row['description']] = category_colors.get(row['category'], '#808080')

            return color_map

        except Exception as e:
            print(f"Warning: Could not get color scheme from database: {e}")
            return {}


# Convenience function
def quick_map(gdf: gpd.GeoDataFrame,
              color_by: Optional[str] = None,
              basemap: str = 'OpenStreetMap',
              save_as: Optional[str] = None,
              **kwargs) -> folium.Map:
    """
    Quick function to create a map from GeoDataFrame.

    Args:
        gdf: GeoDataFrame with AIS data
        color_by: Column to color-code by
        basemap: Basemap style
        save_as: Filename to save (optional)
        **kwargs: Additional arguments passed to plot_ais_data()

    Returns:
        Folium map object
    """
    plotter = AISMapPlotter(basemap=basemap)

    # Determine color map for vessel categories
    color_map = None
    if color_by == 'vessel_category' and 'vessel_category' in gdf.columns:
        color_map = AISMapPlotter.COLOR_SCHEMES['vessel_category']

    map_obj = plotter.plot_ais_data(gdf, color_by=color_by, color_map=color_map, **kwargs)

    if save_as:
        plotter.save(save_as)

    return map_obj


# Example usage
if __name__ == '__main__':
    """
    Example usage of AIS Map Plotter.
    """
    from ecosound.environment import AISQueryHelper

    # Query some AIS data
    with AISQueryHelper('ais_data.duckdb') as ais:
        # Get vessel data
        gdf = ais.query_by_vessel_category(
            start_date='2018-01-01',
            end_date='2018-01-07',
            min_lat=41.65,
            max_lat=46.02,
            min_lon=-71.1,
            max_lon=-65.0,
            categories=['Cargo', 'Tanker', 'Fishing'],
            limit=500
        )

    print(f"Plotting {len(gdf)} AIS positions...")

    # Example 1: Color by vessel category
    print("\nExample 1: Color by vessel category")
    plotter = AISMapPlotter(basemap='CartoDB Positron')
    map1 = plotter.plot_ais_data(
        gdf,
        color_by='vessel_category',
        color_map=AISMapPlotter.COLOR_SCHEMES['vessel_category'],
        popup_fields=['mmsi', 'vessel_name', 'vessel_category', 'sog'],
        marker_size=6
    )

    # Add recorder locations
    recorder_data = pd.DataFrame({
        'name': ['Recorder_A', 'Recorder_B', 'Recorder_C'],
        'latitude': [42.5, 43.0, 44.5],
        'longitude': [-68.0, -68.5, -69.0],
        'depth_m': [100, 150, 200]
    })

    plotter.add_recorder_locations(recorder_data)
    plotter.save('ais_map_by_category.html')

    # Example 2: Color by speed (continuous)
    print("\nExample 2: Color by speed (SOG)")
    plotter2 = AISMapPlotter(basemap='Esri Ocean')
    map2 = plotter2.plot_ais_data(
        gdf,
        color_by='sog',
        continuous_cmap='RdYlGn',  # Red-Yellow-Green
        popup_fields=['mmsi', 'vessel_name', 'sog', 'cog'],
        marker_size=5
    )
    plotter2.save('ais_map_by_speed.html')

    # Example 3: Clustered markers (good for large datasets)
    print("\nExample 3: Clustered markers")
    plotter3 = AISMapPlotter(basemap='OpenStreetMap')
    map3 = plotter3.plot_ais_data(
        gdf,
        color_by='vessel_category',
        color_map=AISMapPlotter.COLOR_SCHEMES['vessel_category'],
        add_cluster=True
    )
    plotter3.save('ais_map_clustered.html')

    # Example 4: With heatmap
    print("\nExample 4: With heatmap overlay")
    plotter4 = AISMapPlotter(basemap='CartoDB Dark')
    map4 = plotter4.plot_ais_data(
        gdf,
        add_heatmap=True,
        marker_size=3
    )
    plotter4.save('ais_map_heatmap.html')

    # Example 5: Quick map (one-liner)
    print("\nExample 5: Quick map")
    quick_map(gdf, color_by='vessel_category', save_as='ais_quick_map.html')

    print("\n✓ All examples completed!")
    print("Open the HTML files in your browser to view the interactive maps.")