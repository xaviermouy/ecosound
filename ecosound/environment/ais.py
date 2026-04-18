# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
AIS DuckDB Query Helper
Provides optimized query methods for the AIS DuckDB + Parquet database.
Designed for maximum query performance using partitioning and columnar storage.

All query methods return GeoDataFrames by default for easy spatial operations and visualization.

Author: Data Processing Script
Date: 2025
"""

import duckdb
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
from typing import Optional, List, Union
from pathlib import Path
import xarray as xr


class AISQueryHelper:
    """
    Helper class for efficiently querying AIS data stored in DuckDB + Parquet.
    Optimized for date, geographic, and vessel-based queries.

    All query methods return GeoDataFrames with Point geometries (WGS84/EPSG:4326).
    """

    def __init__(self, db_path: str):
        """
        Initialize the query helper.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path, read_only=True)

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _add_vessel_type_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add vessel_type_name and vessel_category columns by joining with lookup table.

        Args:
            df: DataFrame with vessel_type column

        Returns:
            DataFrame with added vessel_type_name and vessel_category columns
        """
        if df.empty or 'vessel_type' not in df.columns:
            return df

        # Get vessel type lookup
        lookup_query = """
            SELECT vessel_type_code, description, category
            FROM vessel_type_lookup
        """
        try:
            lookup_df = self.conn.execute(lookup_query).df()

            # Merge with main dataframe
            df = df.merge(
                lookup_df,
                left_on='vessel_type',
                right_on='vessel_type_code',
                how='left'
            )

            # Rename columns for clarity
            df = df.rename(columns={
                'description': 'vessel_type_name',
                'category': 'vessel_category'
            })

            # Drop the redundant vessel_type_code column
            if 'vessel_type_code' in df.columns:
                df = df.drop(columns=['vessel_type_code'])

            # Fill NaN values for vessels not in lookup table
            df['vessel_type_name'] = df['vessel_type_name'].fillna('Unknown')
            df['vessel_category'] = df['vessel_category'].fillna('Unknown')

        except Exception as e:
            # If lookup table doesn't exist or error occurs, just return original df
            print(f"Warning: Could not add vessel type names: {e}")
            pass

        return df

    @staticmethod
    def _to_geodataframe(df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Convert DataFrame to GeoDataFrame.

        Args:
            df: DataFrame with latitude and longitude columns
            crs: Coordinate reference system (default: WGS84)

        Returns:
            GeoDataFrame with Point geometries
        """
        if df.empty:
            return gpd.GeoDataFrame(df, geometry=[], crs=crs)

        # Create Point geometries from lon, lat (note: lon first for Point)
        geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        return gdf

    def query_rectangle(self,
                       start_date: str,
                       end_date: str,
                       min_lat: float,
                       max_lat: float,
                       min_lon: float,
                       max_lon: float,
                       mmsi: Optional[Union[int, List[int]]] = None,
                       vessel_type: Optional[Union[int, List[int]]] = None,
                       vessel_name: Optional[str] = None,
                       limit: Optional[int] = None,
                       return_gdf: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Query AIS data within a geographic rectangle and date range.
        OPTIMIZED: Uses Hive partitioning (date) and Parquet column statistics (lat/lon).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            mmsi: Single MMSI or list of MMSIs (optional)
            vessel_type: Single vessel type code or list of codes (optional)
            vessel_name: Vessel name pattern for LIKE search (optional)
            limit: Maximum number of records to return (optional)
            return_gdf: Return GeoDataFrame (default: True) or DataFrame

        Returns:
            GeoDataFrame with query results (or DataFrame if return_gdf=False)
        """
        # Build query with optimal filter order
        # Note: transceiver_class column removed due to schema inconsistency across years
        # (transceiver_class vs TranscieverClass in different parquet files)
        query = f"""
            SELECT
                mmsi,
                base_datetime,
                latitude,
                longitude,
                sog,
                cog,
                heading,
                vessel_name,
                imo,
                call_sign,
                vessel_type,
                status,
                length,
                width,
                draft,
                cargo
            FROM ais_data
            WHERE base_datetime BETWEEN '{start_date}' AND '{end_date}'
              AND latitude BETWEEN {min_lat} AND {max_lat}
              AND longitude BETWEEN {min_lon} AND {max_lon}
        """

        # Add MMSI filter if specified
        if mmsi is not None:
            if isinstance(mmsi, list):
                mmsi_str = ','.join(str(m) for m in mmsi)
                query += f"\n              AND mmsi IN ({mmsi_str})"
            else:
                query += f"\n              AND mmsi = {mmsi}"

        # Add vessel type filter if specified
        if vessel_type is not None:
            if isinstance(vessel_type, list):
                vtype_str = ','.join(str(v) for v in vessel_type)
                query += f"\n              AND vessel_type IN ({vtype_str})"
            else:
                query += f"\n              AND vessel_type = {vessel_type}"

        # Add vessel name filter if specified (slower - use sparingly)
        if vessel_name is not None:
            query += f"\n              AND vessel_name LIKE '%{vessel_name}%'"

        # Add ORDER BY for consistent results
        query += "\n            ORDER BY base_datetime, mmsi"

        # Add limit if specified
        if limit is not None:
            query += f"\n            LIMIT {limit}"

        df = self.conn.execute(query).df()

        # Add vessel type names
        df = self._add_vessel_type_names(df)

        if return_gdf:
            return self._to_geodataframe(df)
        return df

    def query_radius(self,
                    start_date: str,
                    end_date: str,
                    center_lat: float,
                    center_lon: float,
                    radius_km: float,
                    mmsi: Optional[Union[int, List[int]]] = None,
                    vessel_type: Optional[Union[int, List[int]]] = None,
                    vessel_name: Optional[str] = None,
                    limit: Optional[int] = None,
                    return_gdf: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Query AIS data within a radius from a center point.
        OPTIMIZED: Uses bounding box first, then calculates exact distance.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            center_lat: Center latitude (decimal degrees)
            center_lon: Center longitude (decimal degrees)
            radius_km: Radius in kilometers
            mmsi: Single MMSI or list of MMSIs (optional)
            vessel_type: Single vessel type code or list of codes (optional)
            vessel_name: Vessel name pattern for LIKE search (optional)
            limit: Maximum number of records to return (optional)
            return_gdf: Return GeoDataFrame (default: True) or DataFrame

        Returns:
            GeoDataFrame with query results including 'distance_km' column
        """
        # Calculate bounding box (approximate, faster filtering)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        min_lat = center_lat - lat_delta
        max_lat = center_lat + lat_delta
        min_lon = center_lon - lon_delta
        max_lon = center_lon + lon_delta

        # Query with bounding box first (fast) - get DataFrame first
        df = self.query_rectangle(
            start_date, end_date,
            min_lat, max_lat, min_lon, max_lon,
            mmsi, vessel_type, vessel_name,
            limit=None,  # Don't limit yet - need to filter by radius first
            return_gdf=False  # Get DataFrame for distance calculation
        )

        if df.empty:
            return self._to_geodataframe(df) if return_gdf else df

        # Calculate exact distances using Haversine formula
        df['distance_km'] = self._haversine_distance(
            center_lat, center_lon,
            df['latitude'].values, df['longitude'].values
        )

        # Filter by exact radius
        df = df[df['distance_km'] <= radius_km].copy()

        # Sort by distance
        df = df.sort_values('distance_km')

        # Apply limit if specified
        if limit is not None:
            df = df.head(limit)

        if return_gdf:
            return self._to_geodataframe(df)
        return df

    def query_vessel_track(self,
                          mmsi: int,
                          start_date: str,
                          end_date: str,
                          min_lat: Optional[float] = None,
                          max_lat: Optional[float] = None,
                          min_lon: Optional[float] = None,
                          max_lon: Optional[float] = None,
                          return_gdf: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Query all positions for a specific vessel (track).
        OPTIMIZED: MMSI filter is highly selective.

        Args:
            mmsi: Vessel MMSI
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_lat, max_lat: Optional latitude bounds
            min_lon, max_lon: Optional longitude bounds
            return_gdf: Return GeoDataFrame (default: True) or DataFrame

        Returns:
            GeoDataFrame with vessel track, sorted by time
        """
        query = f"""
            SELECT
                mmsi,
                base_datetime,
                latitude,
                longitude,
                sog,
                cog,
                heading,
                vessel_name,
                vessel_type,
                status
            FROM ais_data
            WHERE mmsi = {mmsi}
              AND base_datetime BETWEEN '{start_date}' AND '{end_date}'
        """

        # Add geographic bounds if specified
        if min_lat is not None:
            query += f"""
              AND latitude BETWEEN {min_lat} AND {max_lat}
              AND longitude BETWEEN {min_lon} AND {max_lon}
            """

        query += "\n            ORDER BY base_datetime"

        df = self.conn.execute(query).df()

        # Add vessel type names
        df = self._add_vessel_type_names(df)

        if return_gdf:
            return self._to_geodataframe(df)
        return df

    def query_by_vessel_category(self,
                                start_date: str,
                                end_date: str,
                                min_lat: float,
                                max_lat: float,
                                min_lon: float,
                                max_lon: float,
                                categories: Union[str, List[str]],
                                limit: Optional[int] = None,
                                return_gdf: bool = True) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """
        Query AIS data by vessel category (Cargo, Tanker, Fishing, etc.).
        Uses the vessel_type_lookup table for categorization.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            categories: Single category or list of categories
                       (e.g., 'Cargo', ['Cargo', 'Tanker'])
            limit: Maximum number of records to return (optional)
            return_gdf: Return GeoDataFrame (default: True) or DataFrame

        Returns:
            GeoDataFrame with query results including vessel category
        """
        if isinstance(categories, str):
            categories = [categories]

        categories_str = "', '".join(categories)

        query = f"""
            SELECT
                a.mmsi,
                a.base_datetime,
                a.latitude,
                a.longitude,
                a.sog,
                a.cog,
                a.heading,
                a.vessel_name,
                a.imo,
                a.vessel_type,
                v.category as vessel_category,
                v.description as vessel_type_description,
                a.length,
                a.width,
                a.draft
            FROM ais_data a
            JOIN vessel_type_lookup v ON a.vessel_type = v.vessel_type_code
            WHERE a.base_datetime BETWEEN '{start_date}' AND '{end_date}'
              AND a.latitude BETWEEN {min_lat} AND {max_lat}
              AND a.longitude BETWEEN {min_lon} AND {max_lon}
              AND v.category IN ('{categories_str}')
            ORDER BY a.base_datetime, a.mmsi
        """

        if limit is not None:
            query += f"\n            LIMIT {limit}"

        df = self.conn.execute(query).df()

        if return_gdf:
            return self._to_geodataframe(df)
        return df

    def get_unique_vessels(self,
                          start_date: str,
                          end_date: str,
                          min_lat: float,
                          max_lat: float,
                          min_lon: float,
                          max_lon: float) -> pd.DataFrame:
        """
        Get unique vessels in a region during a time period.
        Note: Returns DataFrame (not GeoDataFrame) as each vessel has multiple positions.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude

        Returns:
            DataFrame with unique vessels and their details
        """
        query = f"""
            SELECT DISTINCT
                mmsi,
                FIRST(vessel_name ORDER BY base_datetime) as vessel_name,
                FIRST(imo ORDER BY base_datetime) as imo,
                FIRST(call_sign ORDER BY base_datetime) as call_sign,
                FIRST(vessel_type ORDER BY base_datetime) as vessel_type,
                FIRST(length ORDER BY base_datetime) as length,
                FIRST(width ORDER BY base_datetime) as width,
                COUNT(*) as num_positions,
                MIN(base_datetime) as first_seen,
                MAX(base_datetime) as last_seen
            FROM ais_data
            WHERE base_datetime BETWEEN '{start_date}' AND '{end_date}'
              AND latitude BETWEEN {min_lat} AND {max_lat}
              AND longitude BETWEEN {min_lon} AND {max_lon}
            GROUP BY mmsi
            ORDER BY num_positions DESC
        """

        df = self.conn.execute(query).df()

        # Add vessel type names
        df = self._add_vessel_type_names(df)

        return df

    def get_statistics(self,
                      start_date: str,
                      end_date: str,
                      min_lat: Optional[float] = None,
                      max_lat: Optional[float] = None,
                      min_lon: Optional[float] = None,
                      max_lon: Optional[float] = None) -> dict:
        """
        Get statistics about AIS data for a given region and time period.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_lat, max_lat: Optional latitude bounds
            min_lon, max_lon: Optional longitude bounds

        Returns:
            Dictionary with statistics
        """
        where_clause = f"WHERE base_datetime BETWEEN '{start_date}' AND '{end_date}'"

        if min_lat is not None:
            where_clause += f"""
                AND latitude BETWEEN {min_lat} AND {max_lat}
                AND longitude BETWEEN {min_lon} AND {max_lon}
            """

        query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT mmsi) as unique_vessels,
                MIN(base_datetime) as start_time,
                MAX(base_datetime) as end_time,
                AVG(sog) as avg_speed,
                MAX(sog) as max_speed,
                MIN(latitude) as min_lat,
                MAX(latitude) as max_lat,
                MIN(longitude) as min_lon,
                MAX(longitude) as max_lon
            FROM ais_data
            {where_clause}
        """

        result = self.conn.execute(query).df()
        return result.iloc[0].to_dict()

    def create_gridded_vessel_counts(self,
                                     start_date: str,
                                     end_date: str,
                                     min_lat: float,
                                     max_lat: float,
                                     min_lon: float,
                                     max_lon: float,
                                     width_km: float,
                                     height_km: float,
                                     time_resolution_hours: int = 1) -> xr.DataArray:
        """
        Create a gridded xarray DataArray with counts of unique vessels per hour.

        Args:
            start_date: Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
            end_date: End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            width_km: Grid cell width in kilometers
            height_km: Grid cell height in kilometers
            time_resolution_hours: Time resolution in hours (default: 1)

        Returns:
            xarray.DataArray with dimensions (time, lat, lon) containing unique vessel counts (NaN for zero counts)
        """
        # Query all data for the region and time period
        print(f"Querying AIS data from {start_date} to {end_date}...")
        df = self.query_rectangle(
            start_date=start_date,
            end_date=end_date,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            return_gdf=False
        )

        if df.empty:
            print("Warning: No data found for the specified region and time period")
            return None

        print(f"Found {len(df)} AIS records")

        # Convert base_datetime to pandas datetime if not already
        df['base_datetime'] = pd.to_datetime(df['base_datetime'])

        # Create time bins (hourly)
        time_start = pd.to_datetime(start_date)
        time_end = pd.to_datetime(end_date)
        time_bins = pd.date_range(time_start, time_end, freq=f'{time_resolution_hours}h')

        if len(time_bins) < 2:
            print("Warning: Need at least 2 time bins. Adjusting end_date.")
            time_end = time_start + pd.Timedelta(hours=time_resolution_hours)
            time_bins = pd.date_range(time_start, time_end, freq=f'{time_resolution_hours}h')

        # Convert km to degrees (approximate)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude) - use mean latitude
        mean_lat = (min_lat + max_lat) / 2
        lat_spacing = height_km / 111.0
        lon_spacing = width_km / (111.0 * np.cos(np.radians(mean_lat)))

        # Create spatial grid
        lat_bins = np.arange(min_lat, max_lat + lat_spacing/2, lat_spacing)
        lon_bins = np.arange(min_lon, max_lon + lon_spacing/2, lon_spacing)

        print(f"Creating grid: {len(lat_bins)-1} x {len(lon_bins)-1} cells")
        print(f"Time bins: {len(time_bins)-1} periods")

        # Assign each record to bins using digitize
        # For time bins, convert datetime to numeric (timestamps in nanoseconds)
        df['time_idx'] = np.digitize(df['base_datetime'].values.astype('int64'),
                                     time_bins.values.astype('int64')) - 1
        df['lat_idx'] = np.digitize(df['latitude'].values, lat_bins) - 1
        df['lon_idx'] = np.digitize(df['longitude'].values, lon_bins) - 1

        # Filter to keep only records within valid bin indices
        df = df[
            (df['time_idx'] >= 0) & (df['time_idx'] < len(time_bins) - 1) &
            (df['lat_idx'] >= 0) & (df['lat_idx'] < len(lat_bins) - 1) &
            (df['lon_idx'] >= 0) & (df['lon_idx'] < len(lon_bins) - 1)
        ].copy()

        print(f"Records within grid bounds: {len(df)}")

        # Group by indices and count unique MMSIs
        print("Counting unique vessels per grid cell and time bin...")
        grouped = df.groupby(['time_idx', 'lat_idx', 'lon_idx'])['mmsi'].nunique().reset_index()
        grouped.columns = ['time_idx', 'lat_idx', 'lon_idx', 'vessel_count']

        print(f"Non-empty cells: {len(grouped)}")

        # Create full grid with NaN for empty cells
        # Use bin centers for coordinates
        lat_centers = lat_bins[:-1] + lat_spacing / 2
        lon_centers = lon_bins[:-1] + lon_spacing / 2
        time_centers = time_bins[:-1]

        # Create DataArray initialized with NaN (using IOOS-compliant coordinate names)
        vessel_counts = xr.DataArray(
            np.full((len(time_centers), len(lat_centers), len(lon_centers)), np.nan),
            coords={
                'time': time_centers,
                'latitude': lat_centers,
                'longitude': lon_centers
            },
            dims=['time', 'latitude', 'longitude'],
            name='unique_vessel_count'
        )

        # Fill in the actual counts
        for _, row in grouped.iterrows():
            time_idx = int(row['time_idx'])
            lat_idx = int(row['lat_idx'])
            lon_idx = int(row['lon_idx'])
            vessel_counts.values[time_idx, lat_idx, lon_idx] = row['vessel_count']

        # Add metadata
        vessel_counts.attrs = {
            'description': 'Unique vessel count per grid cell per time period (NaN = no vessels)',
            'start_date': str(start_date),
            'end_date': str(end_date),
            'grid_width_km': width_km,
            'grid_height_km': height_km,
            'time_resolution_hours': time_resolution_hours,
            'units': 'number of unique vessels (MMSI)',
            'crs': 'EPSG:4326'
        }

        vessel_counts['time'].attrs = {'long_name': 'Time', 'standard_name': 'time'}
        vessel_counts['latitude'].attrs = {'long_name': 'Latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}
        vessel_counts['longitude'].attrs = {'long_name': 'Longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}

        print(f"Created xarray DataArray with shape: {vessel_counts.shape}")
        non_nan_count = np.sum(~np.isnan(vessel_counts.values))
        print(f"Non-empty cells: {non_nan_count}")
        print(f"Total vessel-cell-hour observations: {int(np.nansum(vessel_counts.values))}")

        return vessel_counts

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float,
                           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Calculate Haversine distance between point and array of points.
        Vectorized for performance.

        Args:
            lat1, lon1: Center point coordinates
            lat2, lon2: Arrays of coordinates

        Returns:
            Array of distances in kilometers
        """
        # Convert to radians
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in kilometers
        r = 6371.0

        return c * r

    @staticmethod
    def calculate_bounding_box(center_lat: float, center_lon: float,
                              radius_km: float) -> dict:
        """
        Calculate bounding box for a given center point and radius.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Radius in kilometers

        Returns:
            Dictionary with min_lat, max_lat, min_lon, max_lon
        """
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * np.cos(np.radians(center_lat)))

        return {
            'min_lat': center_lat - lat_delta,
            'max_lat': center_lat + lat_delta,
            'min_lon': center_lon - lon_delta,
            'max_lon': center_lon + lon_delta
        }


# Example usage
if __name__ == '__main__':
    """
    Example usage of the AIS Query Helper with GeoDataFrames.
    """

    # Path to your DuckDB database
    DB_PATH = 'ais_data.duckdb'

    # Using context manager (recommended)
    with AISQueryHelper(DB_PATH) as ais:

        print("Example 1: Query by rectangle (returns GeoDataFrame)")
        print("=" * 60)
        gdf = ais.query_rectangle(
            start_date='2018-01-01',
            end_date='2018-01-07',
            min_lat=41.65,
            max_lat=46.02,
            min_lon=-71.1,
            max_lon=-65.0,
            limit=1000
        )
        print(f"Found {len(gdf)} records")
        print(f"Type: {type(gdf)}")
        print(f"CRS: {gdf.crs}")
        print(gdf.head())
        print()

        # Plot on map (if matplotlib/contextily installed)
        # gdf.plot()

        print("Example 2: Query by radius (returns GeoDataFrame)")
        print("=" * 60)
        gdf = ais.query_radius(
            start_date='2018-01-01',
            end_date='2018-01-07',
            center_lat=42.5,
            center_lon=-68.0,
            radius_km=50,  # 50 km radius
            limit=1000
        )
        print(f"Found {len(gdf)} records within 50km")
        print(f"Has 'distance_km' column: {'distance_km' in gdf.columns}")
        print(gdf[['mmsi', 'vessel_name', 'distance_km', 'geometry']].head())
        print()

        print("Example 3: Track a specific vessel (returns GeoDataFrame)")
        print("=" * 60)
        gdf = ais.query_vessel_track(
            mmsi=367123456,  # Replace with actual MMSI
            start_date='2018-01-01',
            end_date='2018-01-07'
        )
        print(f"Found {len(gdf)} positions for vessel")
        print(gdf.head())

        # Can convert to LineString for track visualization
        # from shapely.geometry import LineString
        # track_line = LineString(gdf.geometry.tolist())
        print()

        print("Example 4: Spatial operations with GeoDataFrame")
        print("=" * 60)
        gdf = ais.query_by_vessel_category(
            start_date='2018-01-01',
            end_date='2018-01-07',
            min_lat=41.65,
            max_lat=46.02,
            min_lon=-71.1,
            max_lon=-65.0,
            categories=['Cargo', 'Tanker'],
            limit=1000
        )

        # Spatial operations
        print(f"Bounding box: {gdf.total_bounds}")
        print(f"Centroid: {gdf.geometry.centroid.iloc[0]}")

        # Can buffer points, intersect with polygons, etc.
        # buffered = gdf.buffer(0.1)  # Buffer by 0.1 degrees
        print()

        print("Example 5: Export to various formats")
        print("=" * 60)
        # gdf.to_file("ais_data.geojson", driver="GeoJSON")
        # gdf.to_file("ais_data.shp")  # Shapefile
        # gdf.to_parquet("ais_data_geo.parquet")  # Geoparquet
        print("GeoDataFrames can be exported to: GeoJSON, Shapefile, Geoparquet, etc.")