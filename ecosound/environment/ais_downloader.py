# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Marine Cadastre AIS Data Downloader with DuckDB + Parquet Storage
Downloads AIS (Automatic Identification System) data from Marine Cadastre
and stores it efficiently in DuckDB with Parquet files for ultra-fast queries.

Key Features:
- DuckDB for OLAP-style queries (100x+ faster than SQLite on analytical queries)
- Parquet columnar storage with compression
- Hive partitioning by year/month/day for automatic query pruning
- Optimized for date and geographic boundary queries
- Each day stored as a separate Parquet file for maximum query efficiency

Author: Data Processing Script
Date: 2025
"""

import duckdb
import csv
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import tempfile
import shutil
from typing import Optional, Tuple, List
import argparse
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

class AISDataDownloaderDuckDB:
    """
    Downloads and processes AIS data from Marine Cadastre with DuckDB + Parquet storage.
    Optimized for fast temporal and geographic queries.
    """

    BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

    def __init__(self, db_path: str, parquet_dir: str, temp_dir: Optional[str] = None):
        """
        Initialize the AIS data downloader with DuckDB.

        Args:
            db_path: Path to DuckDB database file
            parquet_dir: Directory to store Parquet files (partitioned by date)
            temp_dir: Temporary directory for downloads (default: system temp)
        """
        self.db_path = db_path
        self.parquet_dir = Path(parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "ais_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.setup_database()

    def setup_database(self):
        """Initialize DuckDB database with optimized settings and schema."""
        conn = duckdb.connect(self.db_path)

        # Configure DuckDB for optimal performance
        conn.execute("SET threads TO 8")  # Use multiple threads
        conn.execute("SET memory_limit = '8GB'")  # Adjust based on your system
        conn.execute("SET temp_directory = 'temp_duckdb'")

        # Create metadata table to track downloads
        conn.execute('''
            CREATE TABLE IF NOT EXISTS download_metadata (
                date DATE PRIMARY KEY,
                download_timestamp TIMESTAMP,
                records_inserted BIGINT,
                parquet_file_path VARCHAR
            )
        ''')

        # Create vessel type lookup table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS vessel_type_lookup (
                vessel_type_code INTEGER PRIMARY KEY,
                description VARCHAR NOT NULL,
                category VARCHAR NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        print(f"✓ DuckDB database initialized at {self.db_path}")
        print(f"✓ Parquet storage directory: {self.parquet_dir}")

    def create_ais_view(self):
        """
        Create a view that unions all Parquet files for easy querying.
        This view enables querying all AIS data as a single table.
        """
        conn = duckdb.connect(self.db_path)

        # Create view that reads all parquet files with Hive partitioning
        parquet_pattern = str(self.parquet_dir / "**/*.parquet")

        conn.execute(f'''
            CREATE OR REPLACE VIEW ais_data AS
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
                cargo,
                transceiver_class,
                year,
                month,
                day
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        ''')

        conn.commit()
        conn.close()
        print("✓ Created ais_data view for querying all Parquet files")

    def generate_date_urls(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """
        Generate download URLs for date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of (url, date_string) tuples
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        urls = []
        current = start
        while current <= end:
            year = current.strftime('%Y')
            month = current.strftime('%m')
            day = current.strftime('%d')
            date_str = current.strftime('%Y-%m-%d')

            url = f"{self.BASE_URL}/{year}/AIS_{year}_{month}_{day}.zip"
            urls.append((url, date_str))

            current += timedelta(days=1)

        return urls

    def is_date_in_database(self, date_str: str) -> bool:
        """
        Check if a date has already been processed and is in the database.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            True if date exists in database, False otherwise
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            conn = duckdb.connect(self.db_path)
            result = conn.execute(
                'SELECT COUNT(*) FROM download_metadata WHERE date = ?',
                [date_obj.date()]
            ).fetchone()
            conn.close()
            return result[0] > 0
        except Exception:
            return False

    async def download_file(self, session: aiohttp.ClientSession, url: str,
                            date_str: str, force_download: bool = False) -> Optional[Path]:
        """
        Download a single AIS data file.

        Args:
            session: aiohttp client session
            url: Download URL
            date_str: Date string for filename
            force_download: If True, download even if file already exists

        Returns:
            Path to downloaded file or None if failed
        """
        filename = self.temp_dir / f"AIS_{date_str}.zip"

        # Skip if already downloaded (unless force_download is True)
        if filename.exists() and not force_download:
            print(f"✓ Already downloaded: {date_str}")
            return filename

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                            f.write(chunk)
                    print(f"✓ Downloaded: {date_str} ({filename.stat().st_size / 1024 / 1024:.1f} MB)")
                    return filename
                elif response.status == 404:
                    print(f"✗ Not available: {date_str}")
                    return None
                else:
                    print(f"✗ Error downloading {date_str}: HTTP {response.status}")
                    return None
        except Exception as e:
            print(f"✗ Exception downloading {date_str}: {str(e)}")
            return None

    async def download_files(self, start_date: str, end_date: str,
                             max_concurrent: int = 5,
                             force_download: bool = False) -> List[Path]:
        """
        Download multiple AIS data files concurrently.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_concurrent: Maximum concurrent downloads
            force_download: If True, download files even if already in database

        Returns:
            List of downloaded file paths
        """
        urls = self.generate_date_urls(start_date, end_date)

        # Filter out dates already in database (unless force_download is True)
        if not force_download:
            urls_to_download = []
            skipped_count = 0
            for url, date_str in urls:
                if self.is_date_in_database(date_str):
                    skipped_count += 1
                else:
                    urls_to_download.append((url, date_str))

            if skipped_count > 0:
                print(f"\n✓ Skipping {skipped_count} dates already in database")
                print(f"  (use --force-download to re-download them)")

            urls = urls_to_download

        if not urls:
            print("\n✓ All requested dates are already in the database")
            return []

        print(f"\nDownloading {len(urls)} files from {start_date} to {end_date}...")

        downloaded_files = []

        # Use semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_download(session, url, date_str):
            async with semaphore:
                return await self.download_file(session, url, date_str, force_download)

        async with aiohttp.ClientSession() as session:
            tasks = [bounded_download(session, url, date_str)
                     for url, date_str in urls]
            results = await asyncio.gather(*tasks)
            downloaded_files = [f for f in results if f is not None]

        print(f"\n✓ Downloaded {len(downloaded_files)} files successfully")
        return downloaded_files

    def extract_and_process_file(self, zip_path: Path,
                                 min_lat: Optional[float] = None,
                                 max_lat: Optional[float] = None,
                                 min_lon: Optional[float] = None,
                                 max_lon: Optional[float] = None,
                                 force_process: bool = False) -> int:
        """
        Extract ZIP file and process CSV data into Parquet with geographic filtering.

        Args:
            zip_path: Path to ZIP file
            min_lat, max_lat: Latitude boundaries (optional)
            min_lon, max_lon: Longitude boundaries (optional)
            force_process: If True, reprocess even if already in database

        Returns:
            Number of records inserted
        """
        date_str = zip_path.stem.replace('AIS_', '')
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Check if already processed (unless force_process is True)
        conn = duckdb.connect(self.db_path)
        result = conn.execute(
            'SELECT records_inserted FROM download_metadata WHERE date = ?',
            [date_obj.date()]
        ).fetchone()

        if result and not force_process:
            print(f"✓ Already processed: {date_str} ({result[0]} records)")
            conn.close()
            return result[0]
        conn.close()

        if result and force_process:
            print(f"Processing (forced): {date_str}...", end=' ', flush=True)
        else:
            print(f"Processing: {date_str}...", end=' ', flush=True)

        try:
            # Extract CSV from ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_filename = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
                csv_path = self.temp_dir / csv_filename
                zip_ref.extract(csv_filename, self.temp_dir)

            # Read CSV with pandas and apply filtering
            df = pd.read_csv(csv_path, low_memory=False)

            # Rename columns to match our schema
            column_mapping = {
                'MMSI': 'mmsi',
                'BaseDateTime': 'base_datetime',
                'LAT': 'latitude',
                'LON': 'longitude',
                'SOG': 'sog',
                'COG': 'cog',
                'Heading': 'heading',
                'VesselName': 'vessel_name',
                'IMO': 'imo',
                'CallSign': 'call_sign',
                'VesselType': 'vessel_type',
                'Status': 'status',
                'Length': 'length',
                'Width': 'width',
                'Draft': 'draft',
                'Cargo': 'cargo',
                'TransceiverClass': 'transceiver_class'
            }
            df = df.rename(columns=column_mapping)

            # Apply geographic filtering if specified
            initial_count = len(df)
            if min_lat is not None:
                df = df[(df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                       (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)]

            # Add partition columns
            df['year'] = date_obj.year
            df['month'] = date_obj.month
            df['day'] = date_obj.day

            # Convert datetime column to proper format
            df['base_datetime'] = pd.to_datetime(df['base_datetime'])

            # Define output path with Hive partitioning (year/month/day)
            year_str = f"{date_obj.year:04d}"
            month_str = f"{date_obj.month:02d}"
            day_str = f"{date_obj.day:02d}"
            partition_dir = self.parquet_dir / f"year={year_str}" / f"month={month_str}" / f"day={day_str}"
            partition_dir.mkdir(parents=True, exist_ok=True)

            parquet_file = partition_dir / f"ais_{date_str}.parquet"

            # Write to Parquet with optimized settings
            df.to_parquet(
                parquet_file,
                engine='pyarrow',
                compression='snappy',  # Good balance of speed and compression
                index=False,
                # Row group size for optimal query performance
                row_group_size=100000
            )

            # Update metadata
            conn = duckdb.connect(self.db_path)
            conn.execute('''
                INSERT OR REPLACE INTO download_metadata VALUES (?, ?, ?, ?)
            ''', [date_obj.date(), datetime.utcnow(), len(df), str(parquet_file)])
            conn.commit()
            conn.close()

            # Clean up extracted CSV
            csv_path.unlink()

            records_inserted = len(df)
            print(f"✓ {records_inserted:,} records")
            return records_inserted

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return 0

    def process_all_files(self, downloaded_files: List[Path],
                          min_lat: Optional[float] = None,
                          max_lat: Optional[float] = None,
                          min_lon: Optional[float] = None,
                          max_lon: Optional[float] = None,
                          max_workers: int = 4,
                          force_process: bool = False):
        """
        Process all downloaded files in parallel using ThreadPoolExecutor.

        Args:
            downloaded_files: List of downloaded ZIP file paths
            min_lat, max_lat: Latitude boundaries
            min_lon, max_lon: Longitude boundaries
            max_workers: Maximum number of parallel workers (default: 4)
            force_process: If True, reprocess even if already in database
        """
        print(f"\n\nProcessing {len(downloaded_files)} files with {max_workers} workers...")
        total_records = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(
                    self.extract_and_process_file,
                    zip_path, min_lat, max_lat, min_lon, max_lon, force_process
                ): zip_path
                for zip_path in downloaded_files
            }

            # Process results as they complete
            for future in as_completed(future_to_file):
                zip_path = future_to_file[future]
                try:
                    records = future.result()
                    total_records += records
                    completed += 1
                    # Show progress
                    if completed % 10 == 0 or completed == len(downloaded_files):
                        print(f"Progress: {completed}/{len(downloaded_files)} files processed")
                except Exception as e:
                    print(f"✗ Error processing {zip_path}: {str(e)}")

        print(f"\n✓ Total records inserted: {total_records:,}")

    def optimize_database(self):
        """
        Optimize the database and create the unified view.
        """
        print("Optimizing database and creating unified view...")
        self.create_ais_view()

        # Note: We don't need to run ANALYZE on the view since DuckDB
        # automatically optimizes queries on Parquet files with statistics
        # embedded in the Parquet file metadata

        print("✓ Database optimized")

    def cleanup_temp_files(self):
        """Remove temporary download directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"✓ Cleaned up temporary files")

    def get_stats(self):
        """Print database statistics."""
        conn = duckdb.connect(self.db_path)

        # Total records
        total = conn.execute('SELECT COUNT(*) FROM ais_data').fetchone()[0]

        # Date range
        date_range = conn.execute(
            'SELECT MIN(base_datetime), MAX(base_datetime) FROM ais_data'
        ).fetchone()

        # Unique vessels
        vessels = conn.execute('SELECT COUNT(DISTINCT mmsi) FROM ais_data').fetchone()[0]

        # Geographic bounds
        bounds = conn.execute('''
            SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude)
            FROM ais_data
        ''').fetchone()

        # Storage stats
        parquet_files = list(self.parquet_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files) / (1024 ** 3)  # GB

        conn.close()

        print("\n" + "=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)
        print(f"Total records:      {total:,}")
        print(f"Unique vessels:     {vessels:,}")
        print(f"Date range:         {date_range[0]} to {date_range[1]}")
        print(f"Latitude range:     {bounds[0]:.4f} to {bounds[1]:.4f}")
        print(f"Longitude range:    {bounds[2]:.4f} to {bounds[3]:.4f}")
        print(f"Parquet files:      {len(parquet_files)}")
        print(f"Total storage:      {total_size:.2f} GB")
        print("=" * 60)

    def create_vessel_type_lookup_table(self):
        """
        Create and populate a vessel_type_lookup table in the DuckDB database.
        Based on the AIS specification standard codes.
        """

        # AIS Vessel Type Codes (standardized)
        vessel_types = [
            # Not available / default
            (0, 'Not available (default)', 'Unknown'),

            # Reserved for future use (1-19)
            (1, 'Reserved for future use', 'Reserved'),
            (2, 'Reserved for future use', 'Reserved'),

            # Wing in ground (WIG) (20-29)
            (20, 'Wing in ground (WIG), all ships of this type', 'WIG'),
            (21, 'Wing in ground (WIG), Hazardous category A', 'WIG'),
            (22, 'Wing in ground (WIG), Hazardous category B', 'WIG'),
            (23, 'Wing in ground (WIG), Hazardous category C', 'WIG'),
            (24, 'Wing in ground (WIG), Hazardous category D', 'WIG'),
            (25, 'Wing in ground (WIG), Reserved for future use', 'WIG'),
            (26, 'Wing in ground (WIG), Reserved for future use', 'WIG'),
            (27, 'Wing in ground (WIG), Reserved for future use', 'WIG'),
            (28, 'Wing in ground (WIG), Reserved for future use', 'WIG'),
            (29, 'Wing in ground (WIG), Reserved for future use', 'WIG'),

            # Fishing (30)
            (30, 'Fishing', 'Fishing'),

            # Towing (31-32)
            (31, 'Towing', 'Towing'),
            (32, 'Towing: length exceeds 200m or breadth exceeds 25m', 'Towing'),

            # Dredging or underwater ops (33)
            (33, 'Dredging or underwater ops', 'Dredging'),

            # Diving ops (34)
            (34, 'Diving ops', 'Diving'),

            # Military ops (35)
            (35, 'Military ops', 'Military'),

            # Sailing (36)
            (36, 'Sailing', 'Sailing'),

            # Pleasure Craft (37)
            (37, 'Pleasure Craft', 'Pleasure'),

            # Reserved (38-39)
            (38, 'Reserved', 'Reserved'),
            (39, 'Reserved', 'Reserved'),

            # High speed craft (HSC) (40-49)
            (40, 'High speed craft (HSC), all ships of this type', 'HSC'),
            (41, 'High speed craft (HSC), Hazardous category A', 'HSC'),
            (42, 'High speed craft (HSC), Hazardous category B', 'HSC'),
            (43, 'High speed craft (HSC), Hazardous category C', 'HSC'),
            (44, 'High speed craft (HSC), Hazardous category D', 'HSC'),
            (45, 'High speed craft (HSC), Reserved for future use', 'HSC'),
            (46, 'High speed craft (HSC), Reserved for future use', 'HSC'),
            (47, 'High speed craft (HSC), Reserved for future use', 'HSC'),
            (48, 'High speed craft (HSC), Reserved for future use', 'HSC'),
            (49, 'High speed craft (HSC), No additional information', 'HSC'),

            # Pilot Vessel (50)
            (50, 'Pilot Vessel', 'Pilot'),

            # Search and Rescue vessel (51)
            (51, 'Search and Rescue vessel', 'SAR'),

            # Tug (52)
            (52, 'Tug', 'Tug'),

            # Port Tender (53)
            (53, 'Port Tender', 'Port Tender'),

            # Anti-pollution equipment (54)
            (54, 'Anti-pollution equipment', 'Anti-pollution'),

            # Law Enforcement (55)
            (55, 'Law Enforcement', 'Law Enforcement'),

            # Spare - Local Vessels (56-57)
            (56, 'Spare - Local Vessel', 'Local'),
            (57, 'Spare - Local Vessel', 'Local'),

            # Medical Transport (58)
            (58, 'Medical Transport', 'Medical'),

            # Noncombatant ship (59)
            (59, 'Noncombatant ship according to RR Resolution No. 18', 'Noncombatant'),

            # Passenger (60-69)
            (60, 'Passenger, all ships of this type', 'Passenger'),
            (61, 'Passenger, Hazardous category A', 'Passenger'),
            (62, 'Passenger, Hazardous category B', 'Passenger'),
            (63, 'Passenger, Hazardous category C', 'Passenger'),
            (64, 'Passenger, Hazardous category D', 'Passenger'),
            (65, 'Passenger, Reserved for future use', 'Passenger'),
            (66, 'Passenger, Reserved for future use', 'Passenger'),
            (67, 'Passenger, Reserved for future use', 'Passenger'),
            (68, 'Passenger, Reserved for future use', 'Passenger'),
            (69, 'Passenger, No additional information', 'Passenger'),

            # Cargo (70-79)
            (70, 'Cargo, all ships of this type', 'Cargo'),
            (71, 'Cargo, Hazardous category A', 'Cargo'),
            (72, 'Cargo, Hazardous category B', 'Cargo'),
            (73, 'Cargo, Hazardous category C', 'Cargo'),
            (74, 'Cargo, Hazardous category D', 'Cargo'),
            (75, 'Cargo, Reserved for future use', 'Cargo'),
            (76, 'Cargo, Reserved for future use', 'Cargo'),
            (77, 'Cargo, Reserved for future use', 'Cargo'),
            (78, 'Cargo, Reserved for future use', 'Cargo'),
            (79, 'Cargo, No additional information', 'Cargo'),

            # Tanker (80-89)
            (80, 'Tanker, all ships of this type', 'Tanker'),
            (81, 'Tanker, Hazardous category A', 'Tanker'),
            (82, 'Tanker, Hazardous category B', 'Tanker'),
            (83, 'Tanker, Hazardous category C', 'Tanker'),
            (84, 'Tanker, Hazardous category D', 'Tanker'),
            (85, 'Tanker, Reserved for future use', 'Tanker'),
            (86, 'Tanker, Reserved for future use', 'Tanker'),
            (87, 'Tanker, Reserved for future use', 'Tanker'),
            (88, 'Tanker, Reserved for future use', 'Tanker'),
            (89, 'Tanker, No additional information', 'Tanker'),

            # Other Types (90-99)
            (90, 'Other Type, all ships of this type', 'Other'),
            (91, 'Other Type, Hazardous category A', 'Other'),
            (92, 'Other Type, Hazardous category B', 'Other'),
            (93, 'Other Type, Hazardous category C', 'Other'),
            (94, 'Other Type, Hazardous category D', 'Other'),
            (95, 'Other Type, Reserved for future use', 'Other'),
            (96, 'Other Type, Reserved for future use', 'Other'),
            (97, 'Other Type, Reserved for future use', 'Other'),
            (98, 'Other Type, Reserved for future use', 'Other'),
            (99, 'Other Type, no additional information', 'Other'),
        ]

        # Create DataFrame
        df = pd.DataFrame(vessel_types, columns=['vessel_type_code', 'description', 'category'])

        # Connect to database
        conn = duckdb.connect(self.db_path)

        # Drop existing table if it exists and insert data
        conn.execute('DROP TABLE IF EXISTS vessel_type_lookup')
        conn.execute('''
            CREATE TABLE vessel_type_lookup (
                vessel_type_code INTEGER PRIMARY KEY,
                description VARCHAR NOT NULL,
                category VARCHAR NOT NULL
            )
        ''')

        # Insert data using DuckDB
        conn.register('vessel_type_df', df)
        conn.execute('INSERT INTO vessel_type_lookup SELECT * FROM vessel_type_df')
        conn.unregister('vessel_type_df')

        conn.commit()
        conn.close()

        print(f"✓ Created vessel_type_lookup table with {len(df)} vessel types")
        return df

    def example_queries(self):
        """
        Print example queries to demonstrate the fast querying capabilities.
        """
        print("\n" + "=" * 60)
        print("EXAMPLE QUERIES FOR FAST DATA ACCESS")
        print("=" * 60)
        print(f"\nDatabase location: {self.db_path}")
        print(f"Parquet files location: {self.parquet_dir}")
        print("\nPython examples:")
        print(f"""
import duckdb

# Connect to database
conn = duckdb.connect('{self.db_path}')

# Example 1: Query by date range
df = conn.execute('''
    SELECT * FROM ais_data
    WHERE base_datetime BETWEEN '2024-01-01' AND '2024-01-31'
''').df()

# Example 2: Query by geographic bounds (fast!)
df = conn.execute('''
    SELECT * FROM ais_data
    WHERE latitude BETWEEN 40.0 AND 41.0
      AND longitude BETWEEN -74.5 AND -73.5
      AND base_datetime >= '2024-01-01'
''').df()

# Example 3: Query by vessel type
df = conn.execute('''
    SELECT a.*, v.category
    FROM ais_data a
    JOIN vessel_type_lookup v ON a.vessel_type = v.vessel_type_code
    WHERE v.category = 'Cargo'
      AND base_datetime >= '2024-01-01'
''').df()

# Example 4: Aggregate statistics
df = conn.execute('''
    SELECT
        DATE_TRUNC('day', base_datetime) as day,
        COUNT(*) as total_records,
        COUNT(DISTINCT mmsi) as unique_vessels,
        AVG(sog) as avg_speed
    FROM ais_data
    WHERE base_datetime BETWEEN '2024-01-01' AND '2024-01-31'
    GROUP BY day
    ORDER BY day
''').df()

conn.close()
        """)
        print("=" * 60)


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Download and store AIS data with DuckDB + Parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download data for January 2024 in New York area
  python ais_downloader_duckdb.py --start 2024-01-01 --end 2024-01-31 \\
      --min-lat 40.0 --max-lat 41.0 --min-lon -74.5 --max-lon -73.5

  # Download data without geographic filtering
  python ais_downloader_duckdb.py --start 2024-06-01 --end 2024-06-07

  # Get statistics from existing database
  python ais_downloader_duckdb.py --stats-only

  # Show example queries
  python ais_downloader_duckdb.py --examples
        '''
    )

    parser.add_argument('--start', type=str,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--min-lat', type=float,
                        help='Minimum latitude (decimal degrees)')
    parser.add_argument('--max-lat', type=float,
                        help='Maximum latitude (decimal degrees)')
    parser.add_argument('--min-lon', type=float,
                        help='Minimum longitude (decimal degrees)')
    parser.add_argument('--max-lon', type=float,
                        help='Maximum longitude (decimal degrees)')
    parser.add_argument('--db', type=str, default='ais_data.duckdb',
                        help='DuckDB database path (default: ais_data.duckdb)')
    parser.add_argument('--parquet-dir', type=str, default='default',
                        help='Parquet files directory (default: default)')
    parser.add_argument('--temp-dir', type=str,
                        help='Temporary directory for downloads')
    parser.add_argument('--max-concurrent', type=int, default=5,
                        help='Maximum concurrent downloads (default: 5)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for processing files (default: 4)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force re-download of files even if already in database')
    parser.add_argument('--force-process', action='store_true',
                        help='Force re-processing of files even if already in database (overwrites existing data)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary downloaded files')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show database statistics')
    parser.add_argument('--examples', action='store_true',
                        help='Show example queries')

    args = parser.parse_args()


    if args.parquet_dir == 'default':
        args.parquet_dir = os.path.join(os.path.dirname(args.db),os.path.splitext(os.path.basename(args.db))[0])+'_parquet_files'

    # Initialize downloader
    downloader = AISDataDownloaderDuckDB(args.db, args.parquet_dir, args.temp_dir)

    # Examples mode
    if args.examples:
        downloader.example_queries()
        return

    # Stats only mode
    if args.stats_only:
        downloader.get_stats()
        return

    # Validate required arguments
    if not args.start or not args.end:
        parser.error("--start and --end dates are required (unless using --stats-only or --examples)")

    # Validate geographic bounds
    if any([args.min_lat, args.max_lat, args.min_lon, args.max_lon]):
        if not all([args.min_lat, args.max_lat, args.min_lon, args.max_lon]):
            parser.error("All geographic bounds must be specified together")

        if args.min_lat >= args.max_lat:
            parser.error("min-lat must be less than max-lat")
        if args.min_lon >= args.max_lon:
            parser.error("min-lon must be less than max-lon")

    print("=" * 60)
    print("MARINE CADASTRE AIS DATA DOWNLOADER (DuckDB + Parquet)")
    print("=" * 60)
    print(f"Date range:     {args.start} to {args.end}")
    if args.min_lat:
        print(f"Latitude:       {args.min_lat} to {args.max_lat}")
        print(f"Longitude:      {args.min_lon} to {args.max_lon}")
    else:
        print("Geographic:     No filtering (all U.S. coastal waters)")
    print(f"Database:       {args.db}")
    print(f"Parquet dir:    {args.parquet_dir}")
    print("=" * 60)

    try:
        # Step 1: Download files
        downloaded_files = asyncio.run(
            downloader.download_files(
                args.start,
                args.end,
                args.max_concurrent,
                force_download=args.force_download
            )
        )

        if not downloaded_files:
            print("\n✗ No files were downloaded successfully")
            return

        # Step 2: Process files in parallel
        downloader.process_all_files(
            downloaded_files,
            args.min_lat, args.max_lat,
            args.min_lon, args.max_lon,
            max_workers=args.workers,
            force_process=args.force_process
        )

        # Step 3: Optimize database and create views
        downloader.optimize_database()

        # Step 4: Create vessel type lookup table
        downloader.create_vessel_type_lookup_table()

        # Step 5: Show statistics
        downloader.get_stats()

        # Step 6: Show example queries
        #downloader.example_queries()

        # Step 7: Cleanup
        if not args.keep_temp:
            downloader.cleanup_temp_files()
        else:
            print(f"\nℹ Temporary files kept in: {downloader.temp_dir}")

        print("\n✓ All operations completed successfully!")

    except KeyboardInterrupt:
        print("\n\n✗ Operation cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()