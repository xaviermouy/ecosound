"""
ecosound.localization.deployment
=================================
Deployment: per-deployment field context for one or more sensor arrays.
DeployedArray: one array within a deployment (hardware + deployment context).
"""

import os
import numpy as np
import yaml

from .array import SensorArray


class DeployedArray:
    """One sensor array within a deployment.

    Combines hardware description (:class:`SensorArray`) with
    deployment-specific information: position, orientation, and file paths.

    Do not instantiate directly — obtain via :meth:`Deployment.get_array`.

    Parameters
    ----------
    sensor_array : SensorArray
    array_name : str
    array_number : int
    array_config_path : str
        Original path written in the deployment YAML (may be relative).
    location_x : float or None
        Array origin X-coordinate in the deployment's local Cartesian frame
        (metres).  Typically East in a local ENU system.
    location_y : float or None
        Array origin Y-coordinate (metres).  Typically North.
    location_z : float or None
        Array origin Z-coordinate (metres).  Typically Up / depth.
    water_depth_m : float or None
    array_depth_m : float or None
    orientation_yaw_deg : float
    orientation_pitch_deg : float
    orientation_roll_deg : float
    hydrophone_data : list of dict
        Per-hydrophone file paths and roots.  Each dict must have keys:
        ``hydrophone_number``, ``recorder_number``, ``data_path``,
        ``file_name_root``.
    video_data : list of dict
        Per-camera file paths and roots (optional).
    """

    def __init__(
        self,
        sensor_array,
        array_name,
        array_number,
        array_config_path,
        location_x,
        location_y,
        location_z,
        water_depth_m,
        array_depth_m,
        orientation_yaw_deg,
        orientation_pitch_deg,
        orientation_roll_deg,
        hydrophone_data,
        video_data,
    ):
        self.sensor_array          = sensor_array
        self.array_name            = array_name
        self.array_number          = array_number
        self.array_config_path     = array_config_path
        self.location_x            = location_x
        self.location_y            = location_y
        self.location_z            = location_z
        self.water_depth_m         = water_depth_m
        self.array_depth_m         = array_depth_m
        self.orientation_yaw_deg   = orientation_yaw_deg
        self.orientation_pitch_deg = orientation_pitch_deg
        self.orientation_roll_deg  = orientation_roll_deg

        import pandas as pd
        self._hydrophone_data = (
            pd.DataFrame(hydrophone_data) if hydrophone_data else pd.DataFrame()
        )
        self._video_data = (
            pd.DataFrame(video_data) if video_data else pd.DataFrame()
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hydrophone_data(self):
        """Deployment hydrophone data as a DataFrame."""
        return self._hydrophone_data

    @property
    def video_data(self):
        """Deployment video data as a DataFrame (empty if none defined)."""
        return self._video_data

    # ------------------------------------------------------------------
    # Audio-file discovery
    # ------------------------------------------------------------------

    def find_audio_files(self, filename):
        """Find audio file paths and channel indices for all hydrophones.

        Merges :attr:`sensor_array.hydrophones` (hardware info) with
        :attr:`hydrophone_data` (deployment paths) on ``hydrophone_number``,
        then constructs file paths using the common suffix of *filename*.

        Parameters
        ----------
        filename : str
            Path or basename of any one audio file belonging to this array.

        Returns
        -------
        dict
            ``{'path': [str, ...], 'channel': [int, ...]}``

        Raises
        ------
        ValueError
            If no ``file_name_root`` entry matches *filename*.
        RuntimeError
            If the merge produces no rows (hydrophone_number mismatch).
        """
        import pandas as pd

        hp_df   = self.sensor_array.hydrophones
        data_df = self._hydrophone_data

        if data_df.empty:
            raise RuntimeError(
                "No hydrophone_data defined for this DeployedArray."
            )

        merged = hp_df.merge(data_df, on='hydrophone_number', how='inner')
        if merged.empty:
            raise RuntimeError(
                "Merge of sensor_array.hydrophones with hydrophone_data "
                "produced no rows — check hydrophone_number values."
            )

        basename  = os.path.basename(filename)
        file_tail = None
        for file_root in merged['file_name_root']:
            idx = basename.find(str(file_root))
            if idx >= 0:
                file_tail = basename[len(str(file_root)):]
                break
        if file_tail is None:
            raise ValueError(
                f"Could not match any file_name_root to '{basename}'."
            )

        audio_files = {'path': [], 'channel': []}
        for _, row in merged.iterrows():
            fpath = os.path.join(
                str(row['data_path']),
                str(row['file_name_root']) + file_tail,
            )
            audio_files['path'].append(fpath)
            audio_files['channel'].append(int(row['audio_file_channel']))
        return audio_files


class Deployment:
    """Per-deployment field context for one or more sensor arrays.

    Stores deployment metadata (dates, GPS position of the site) and a list
    of :class:`DeployedArray` objects, each combining a :class:`SensorArray`
    with deployment-specific file paths, Cartesian position, and orientation.

    Instantiate via :meth:`from_yaml` or :meth:`write_template` + edit.

    Parameters
    ----------
    data : dict
        Dictionary as returned by ``yaml.safe_load`` of a Deployment YAML.
    yaml_dir : str or None
        Directory of the source YAML file, used to resolve relative
        ``array_config`` paths.  Pass ``None`` to use the current directory.
    """

    def __init__(self, data, yaml_dir=None):
        self.deployment_id        = data.get('deployment_id', '')
        self.description          = data.get('description', '')
        self.location_name        = data.get('location_name', '')
        self.location_lat         = data.get('location_lat', None)
        self.location_lon         = data.get('location_lon', None)
        self.deployed_datetime    = data.get('deployed_datetime', None)
        self.recovered_datetime   = data.get('recovered_datetime', None)

        if yaml_dir is None:
            yaml_dir = os.getcwd()

        self.arrays = []
        for arr_data in data.get('arrays', []):
            config_path_raw = arr_data['array_config']
            if os.path.isabs(config_path_raw):
                config_path = config_path_raw
            else:
                config_path = os.path.normpath(
                    os.path.join(yaml_dir, config_path_raw)
                )
            sensor_array = SensorArray.from_yaml(config_path)

            dep_array = DeployedArray(
                sensor_array          = sensor_array,
                array_name            = arr_data.get('array_name', ''),
                array_number          = arr_data.get('array_number', 0),
                array_config_path     = config_path_raw,
                location_x            = arr_data.get('location_x', None),
                location_y            = arr_data.get('location_y', None),
                location_z            = arr_data.get('location_z', None),
                water_depth_m         = arr_data.get('water_depth_m', None),
                array_depth_m         = arr_data.get('array_depth_m', None),
                orientation_yaw_deg   = arr_data.get('orientation_yaw_deg', 0.0),
                orientation_pitch_deg = arr_data.get('orientation_pitch_deg', 0.0),
                orientation_roll_deg  = arr_data.get('orientation_roll_deg', 0.0),
                hydrophone_data       = arr_data.get('hydrophone_data', []),
                video_data            = arr_data.get('video_data', []),
            )
            self.arrays.append(dep_array)

    # ------------------------------------------------------------------
    # Construction / serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path):
        """Load a Deployment from a YAML file.

        Parameters
        ----------
        path : str
            Path to the Deployment YAML file.

        Returns
        -------
        Deployment
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        yaml_dir = os.path.dirname(os.path.abspath(path))
        return cls(data, yaml_dir=yaml_dir)

    def to_yaml(self, path):
        """Save this Deployment to a YAML file.

        Note: ``array_config`` paths are written as they were originally
        specified (relative paths are preserved as-is).

        Parameters
        ----------
        path : str
            Output file path.
        """
        arrays_out = []
        for dep_arr in self.arrays:
            arr_entry = {
                'array_name':            dep_arr.array_name,
                'array_number':          int(dep_arr.array_number),
                'array_config':          dep_arr.array_config_path,
                'location_x':            dep_arr.location_x,
                'location_y':            dep_arr.location_y,
                'location_z':            dep_arr.location_z,
                'water_depth_m':         dep_arr.water_depth_m,
                'array_depth_m':         dep_arr.array_depth_m,
                'orientation_yaw_deg':   dep_arr.orientation_yaw_deg,
                'orientation_pitch_deg': dep_arr.orientation_pitch_deg,
                'orientation_roll_deg':  dep_arr.orientation_roll_deg,
                'hydrophone_data': (
                    dep_arr.hydrophone_data.to_dict('records')
                    if not dep_arr.hydrophone_data.empty else []
                ),
                'video_data': (
                    dep_arr.video_data.to_dict('records')
                    if not dep_arr.video_data.empty else []
                ),
            }
            arrays_out.append(arr_entry)

        data = {
            'deployment_id':      self.deployment_id,
            'description':        self.description,
            'location_name':      self.location_name,
            'location_lat':       self.location_lat,
            'location_lon':       self.location_lon,
            'deployed_datetime':  str(self.deployed_datetime) if self.deployed_datetime else None,
            'recovered_datetime': str(self.recovered_datetime) if self.recovered_datetime else None,
            'arrays':             arrays_out,
        }

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def write_template(path):
        """Write a commented YAML template file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        template = """\
# Deployment configuration file
# Per-deployment field context — references one or more SensorArray YAML files.

deployment_id: "PR_2026_02_10"
description: "Puerto Rico red hind survey, mobile array"
location_name: "Mona Island, Puerto Rico"
location_lat: 18.09      # GPS latitude of the deployment site (decimal degrees)
location_lon: -67.89     # GPS longitude of the deployment site (decimal degrees)
deployed_datetime: "2026-02-10T14:00:00"
recovered_datetime: "2026-02-10T18:00:00"

arrays:
  - array_name: "Array 1"
    array_number: 0
    array_config: "sensor_array_tetrahedral_v1.yaml"   # relative to this file, or absolute
    location_x: 0.0     # array origin in the deployment local Cartesian frame (metres)
    location_y: 0.0     # e.g. East/North/Up (ENU) relative to a reference point
    location_z: 0.0
    water_depth_m: 15.0
    array_depth_m: 5.0
    orientation_yaw_deg: 0.0     # compass bearing of the array's +x axis (degrees, 0 = North)
    orientation_pitch_deg: 0.0
    orientation_roll_deg: 0.0

    hydrophone_data:
      - hydrophone_number: 0     # must match hydrophone_number in the SensorArray YAML
        recorder_number: 0       # must match recorder_number in the SensorArray YAML
        audio_file_channel: 0    # channel index in the audio file (0-based)
        data_path: "E:/audio/REC1/"
        file_name_root: "AC8_1"
      - hydrophone_number: 1
        recorder_number: 0
        audio_file_channel: 1
        data_path: "E:/audio/REC1/"
        file_name_root: "AC8_2"
      - hydrophone_number: 2
        recorder_number: 0
        audio_file_channel: 2
        data_path: "E:/audio/REC1/"
        file_name_root: "AC8_3"
      - hydrophone_number: 3
        recorder_number: 0
        audio_file_channel: 3
        data_path: "E:/audio/REC1/"
        file_name_root: "AC8_4"
      # Add one entry per hydrophone

    video_data:                  # (optional — remove if no cameras)
      - camera_number: 0
        data_path: "E:/video/CAM1/"
        file_name_root: "VID_CAM1_"
      # Add one entry per camera

  # Add more arrays as needed
  # - array_name: "Array 2"
  #   array_number: 1
  #   array_config: "sensor_array_linear_v1.yaml"
  #   location_x: 5.0
  #   location_y: 0.0
  #   location_z: 0.0
  #   ...
"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(template)

    # ------------------------------------------------------------------
    # Array access
    # ------------------------------------------------------------------

    def get_array(self, identifier):
        """Return a :class:`DeployedArray` by array_number or array_name.

        Parameters
        ----------
        identifier : int or str
            ``array_number`` (int) or ``array_name`` (str).

        Returns
        -------
        DeployedArray

        Raises
        ------
        KeyError
            If no array matches *identifier*.
        """
        if isinstance(identifier, int):
            for arr in self.arrays:
                if arr.array_number == identifier:
                    return arr
        else:
            for arr in self.arrays:
                if arr.array_name == identifier:
                    return arr
        raise KeyError(
            f"No array found with identifier {identifier!r}. "
            f"Available: {[a.array_number for a in self.arrays]}"
        )

    def __len__(self):
        return len(self.arrays)

    def __repr__(self):
        return (
            f"Deployment(id={self.deployment_id!r}, "
            f"arrays={[a.array_name for a in self.arrays]})"
        )
