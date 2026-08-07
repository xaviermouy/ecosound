"""
ecosound.localization.array
===========================
SensorArray: YAML-based hardware description of a sensor array
(hydrophones, acoustic recorders, optional cameras, optional frame geometry).
"""

import os
import numpy as np
import pandas as pd
import yaml

# Okabe–Ito colorblind-safe palette (Wong 2011, Nature Methods).
# Suitable for scientific publications and colour-vision deficiencies.
_PALETTE = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#F0E442',  # yellow
]


def _palette_color(index):
    """Return a palette colour by *index*, cycling if necessary."""
    return _PALETTE[int(index) % len(_PALETTE)]


def _fov_visible(show_fov, cam_num):
    """Return True if FOV should be drawn for camera number *cam_num*.

    Parameters
    ----------
    show_fov : bool or list of int
        ``True``/``False`` applies to all cameras.  A list of camera numbers
        selects which cameras to draw.
    cam_num : int
    """
    if isinstance(show_fov, (list, tuple, set)):
        return int(cam_num) in [int(c) for c in show_fov]
    return bool(show_fov)


class SensorArray:
    """Hardware description of a sensor array.

    Stores hydrophone positions and identifiers, acoustic recorder metadata,
    optional camera configurations, and optional frame geometry.  Intended to
    be reused across multiple deployments.

    Instantiate via :meth:`from_yaml` or :meth:`write_template` + edit.

    Parameters
    ----------
    data : dict
        Dictionary as returned by ``yaml.safe_load`` of a SensorArray YAML.
    """

    def __init__(self, data):
        self.array_name   = data.get('array_name', '')
        self.array_number = data.get('array_number', 0)
        self.description  = data.get('description', '')
        self.created_by   = data.get('created_by', '')
        self.created_date = str(data.get('created_date', ''))
        self._recorders   = data.get('acoustic_recorders', [])

        hp_list = data.get('hydrophones', [])
        self._hydrophones = pd.DataFrame(hp_list) if hp_list else pd.DataFrame()

        cam_list = data.get('cameras', [])
        self._cameras = pd.DataFrame(cam_list) if cam_list else pd.DataFrame()

        self.frame = data.get('frame', None)

    # ------------------------------------------------------------------
    # Construction / serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path):
        """Load a SensorArray from a YAML file.

        Parameters
        ----------
        path : str
            Path to the SensorArray YAML file.

        Returns
        -------
        SensorArray
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)

    def to_yaml(self, path):
        """Save this SensorArray to a YAML file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        data = {
            'array_name':         self.array_name,
            'array_number':       int(self.array_number),
            'description':        self.description,
            'created_by':         self.created_by,
            'created_date':       self.created_date,
            'acoustic_recorders': self._recorders,
            'hydrophones': (
                self._hydrophones.to_dict('records')
                if not self._hydrophones.empty else []
            ),
        }
        if not self._cameras.empty:
            data['cameras'] = self._cameras.to_dict('records')
        if self.frame is not None:
            data['frame'] = self.frame

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
# SensorArray configuration file
# Hardware description of the sensor array — reusable across deployments.
# Edit this file to match your array and save under a meaningful name,
# e.g. sensor_array_tetrahedral_v1.yaml

array_name: "Tetrahedral array v1"
array_number: 0
description: "4-element tetrahedral hydrophone frame"
created_by: "Your Name"
created_date: "2026-01-01"

acoustic_recorders:
  - recorder_number: 0
    brand: "JASCO"
    model: "AMAR G4"
    serial_number: "AMAR-G4-001"
  # Add more recorders as needed

hydrophones:
  # Note: audio_file_channel is defined in deployment.yaml (hydrophone_data),
  # not here, because channel assignments depend on how the recorder was
  # connected for each specific deployment.
  - name: "H1"
    hydrophone_number: 0
    recorder_number: 0          # links to acoustic_recorders above
    brand: "DolphinEar"
    model: "DE200"
    serial_number: "DE200-001"
    x: 0.0                      # position in array frame (metres)
    y: 0.0
    z: 0.0
  - name: "H2"
    hydrophone_number: 1
    recorder_number: 0
    brand: "DolphinEar"
    model: "DE200"
    serial_number: "DE200-002"
    x: 1.0
    y: 0.0
    z: 0.0
  - name: "H3"
    hydrophone_number: 2
    recorder_number: 0
    brand: "DolphinEar"
    model: "DE200"
    serial_number: "DE200-003"
    x: 0.5
    y: 1.0
    z: 0.0
  - name: "H4"
    hydrophone_number: 3
    recorder_number: 0
    brand: "DolphinEar"
    model: "DE200"
    serial_number: "DE200-004"
    x: 0.5
    y: 0.5
    z: 1.0
  # Add more hydrophones as needed

# cameras: (optional — remove this section if no cameras are on the array)
cameras:
  - name: "CAM1"
    camera_number: 0
    brand: "GoPro"
    model: "Hero 11"
    serial_number: "GP11-001"
    x: 0.10
    y: 0.00
    z: 0.50
    horizontal_angle_deg: 0.0
    vertical_angle_deg: -15.0
    fov_horizontal_deg: 120.0
    fov_vertical_deg: 90.0
  # Add more cameras as needed

# frame: (optional — for 3D/2D plotting only)
# Vertices are named (any string key).  Positions can be:
#   {hydrophone: N}  — copied from hydrophones[N]
#   {camera: N}      — copied from cameras[N]
#   [x, y, z]        — explicit coordinates (for structural/junction points)
# Edges reference vertex names, so no integer indexing needed.
frame:
  vertices:
    H1:        {hydrophone: 0}   # position copied from hydrophones[0]
    H2:        {hydrophone: 1}
    H3:        {hydrophone: 2}
    H4:        {hydrophone: 3}
    CAM0:      {camera: 0}       # position copied from cameras[0]
    junction:  [0.5, 0.0, 0.0]  # explicit coords for structural points
  edges:
    - [H1, junction]
    - [H2, junction]
    - [junction, H3]
    - [H4, H3]
    - [CAM0, junction]
"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            f.write(template)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hydrophones(self):
        """Hydrophone configuration as a DataFrame."""
        return self._hydrophones

    @property
    def config(self):
        """Backward-compatible alias for `hydrophones` (required by GridSearch)."""
        return self._hydrophones

    @property
    def recorders(self):
        """List of acoustic recorder metadata dicts."""
        return self._recorders

    @property
    def cameras(self):
        """Camera configuration as a DataFrame (empty if none defined)."""
        return self._cameras

    @property
    def n(self):
        """Number of hydrophones."""
        return len(self._hydrophones)

    @property
    def x(self):
        """X-coordinates of all hydrophones (metres)."""
        return self._hydrophones['x'].values

    @property
    def y(self):
        """Y-coordinates of all hydrophones (metres)."""
        return self._hydrophones['y'].values

    @property
    def z(self):
        """Z-coordinates of all hydrophones (metres)."""
        return self._hydrophones['z'].values

    @property
    def channels(self):
        """Hydrophone numbers as integers (from ``hydrophone_number`` column)."""
        return self._hydrophones['hydrophone_number'].astype(int).values

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def distances(self):
        """Pairwise Euclidean distance matrix (metres), shape (n, n).

        Returns
        -------
        numpy.ndarray, shape (n, n)
        """
        coords = self._hydrophones[["x", "y", "z"]].to_numpy()
        n = len(coords)
        D = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = np.linalg.norm(coords[i] - coords[j])
        return D

    def tdoa_max_sec(self, sound_speed_mps):
        """Maximum possible TDOA given array aperture and sound speed.

        Parameters
        ----------
        sound_speed_mps : float

        Returns
        -------
        float
        """
        return float(np.max(self.distances()) / sound_speed_mps)

    # ------------------------------------------------------------------
    # Pair definitions
    # ------------------------------------------------------------------

    def define_pairs(self, ref_channel):
        """Return [ref_channel, i] pairs for all i != ref_channel.

        Parameters
        ----------
        ref_channel : int
            0-based row index of the reference hydrophone.

        Returns
        -------
        list of [int, int]
        """
        return [[ref_channel, i] for i in range(self.n) if i != ref_channel]

    def pair_labels(self, pairs):
        """Human-readable 'CHa-CHb' labels for a list of hydrophone pairs.

        Parameters
        ----------
        pairs : list of [int, int]
            As returned by :meth:`define_pairs`.

        Returns
        -------
        list of str
        """
        return [
            f"CH{self.channels[i]}-CH{self.channels[j]}"
            for i, j in pairs
        ]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _resolve_frame_vertices(self):
        """Resolve frame vertices into a name → (x, y, z) dict.

        Vertices may be explicit coordinates ``[x, y, z]`` or hydrophone
        references ``{hydrophone: N}``, which are looked up in
        :attr:`hydrophones`.

        Returns
        -------
        dict of str → numpy.ndarray, shape (3,)

        Raises
        ------
        ValueError
            If a hydrophone reference cannot be found.
        """
        resolved = {}
        for name, val in self.frame['vertices'].items():
            if isinstance(val, dict) and 'hydrophone' in val:
                hp_num = val['hydrophone']
                mask = self._hydrophones['hydrophone_number'] == hp_num
                if not mask.any():
                    raise ValueError(
                        f"Frame vertex '{name}' references unknown "
                        f"hydrophone_number {hp_num}."
                    )
                row = self._hydrophones[mask].iloc[0]
                resolved[name] = np.array([row['x'], row['y'], row['z']])
            elif isinstance(val, dict) and 'camera' in val:
                cam_num = val['camera']
                mask = self._cameras['camera_number'] == cam_num
                if not mask.any():
                    raise ValueError(
                        f"Frame vertex '{name}' references unknown "
                        f"camera_number {cam_num}."
                    )
                row = self._cameras[mask].iloc[0]
                resolved[name] = np.array([row['x'], row['y'], row['z']])
            else:
                resolved[name] = np.array(val, dtype=float)
        return resolved

    def plot_3d(self, fov_distance=None, show_fov=True, fov_alpha=0.15,
                show_cameras=True, show_frame=True,
                frame_lw=1.0, frame_alpha=1.0):
        """3D plot of the array geometry: hydrophones, frame, and cameras with FOV.

        Parameters
        ----------
        fov_distance : float or None
            Length of the FOV volume in metres.  Defaults to 30% of the
            maximum inter-hydrophone distance.
        show_fov : bool or list of int
            ``True`` draws FOV for all cameras; ``False`` hides all.  Pass a
            list of camera numbers (e.g. ``[0, 2]``) to show FOV only for
            those cameras.
        fov_alpha : float
            Opacity of the FOV fill (0–1).
        show_cameras : bool
            Draw camera markers.
        show_frame : bool
            Draw frame edges.
        frame_lw : float
            Frame edge line width.
        frame_alpha : float
            Frame edge opacity (0–1).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if fov_distance is None:
            fov_distance = 0.3 * float(np.max(self.distances())) if self.n > 1 else 1.0

        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection='3d')

        # Hydrophones
        for _, row in self._hydrophones.iterrows():
            ch    = int(row['hydrophone_number'])
            color = _palette_color(ch)
            name  = str(row.get('name', f'H{ch}'))
            ax.scatter([row['x']], [row['y']], [row['z']],
                       c=[color], s=80, marker='o', depthshade=False, zorder=5,
                       label=name)

        # Frame
        if show_frame and self.frame is not None:
            verts = self._resolve_frame_vertices()
            for n0, n1 in self.frame['edges']:
                p0, p1 = verts[n0], verts[n1]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color='0.5', lw=frame_lw, alpha=frame_alpha, zorder=3)

        # Cameras
        if show_cameras and not self._cameras.empty:
            for ci, cam in self._cameras.iterrows():
                cam_num = int(cam['camera_number'])
                color   = _palette_color(cam_num)
                name    = str(cam.get('name', f'CAM{ci}'))
                origin  = np.array([cam['x'], cam['y'], cam['z']])
                ax.scatter([cam['x']], [cam['y']], [cam['z']], label=name,
                           c=[color], s=60, marker='s', depthshade=False,
                           zorder=6, edgecolors='k', linewidths=0.8)

                if _fov_visible(show_fov, cam_num):
                    corners = _camera_fov_rays(
                        cam['x'], cam['y'], cam['z'],
                        cam.get('horizontal_angle_deg', 0) or 0,
                        cam.get('vertical_angle_deg',   0) or 0,
                        cam.get('fov_horizontal_deg',  90) or 90,
                        cam.get('fov_vertical_deg',    90) or 90,
                        fov_distance,
                    )
                    bl, tl, br, tr = corners
                    faces = [
                        [origin, bl, br],   # bottom face
                        [origin, tl, tr],   # top face
                        [origin, bl, tl],   # left face
                        [origin, br, tr],   # right face
                        [bl, br, tr, tl],   # back face
                    ]
                    poly = Poly3DCollection(faces, facecolor=color,
                                           edgecolor='none', alpha=fov_alpha,
                                           zorder=2)
                    ax.add_collection3d(poly)

        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(self.array_name, fontsize=10)
        ax.grid(True)

        # Axis limits — equal aspect with padding
        all_x = list(self.x)
        all_y = list(self.y)
        all_z = list(self.z)
        if not self._cameras.empty:
            all_x += list(self._cameras['x'])
            all_y += list(self._cameras['y'])
            all_z += list(self._cameras['z'])
        if self.frame is not None:
            rv = self._resolve_frame_vertices()
            all_x += [v[0] for v in rv.values()]
            all_y += [v[1] for v in rv.values()]
            all_z += [v[2] for v in rv.values()]
        span = max(max(all_x) - min(all_x),
                   max(all_y) - min(all_y),
                   max(all_z) - min(all_z), 0.1)
        pad  = 0.15 * span
        cx_  = (max(all_x) + min(all_x)) / 2
        cy_  = (max(all_y) + min(all_y)) / 2
        cz_  = (max(all_z) + min(all_z)) / 2
        half = span / 2 + pad
        ax.set_xlim(cx_ - half, cx_ + half)
        ax.set_ylim(cy_ - half, cy_ + half)
        ax.set_zlim(cz_ - half, cz_ + half)
        ax.set_box_aspect([1, 1, 1])

        fig.tight_layout()
        return fig

    def plot_2d(self, fov_distance=None, show_fov=True, fov_alpha=0.15,
                show_cameras=True, show_frame=True,
                frame_lw=1.0, frame_alpha=1.0):
        """2D orthogonal projections of the array: hydrophones, frame, and cameras with FOV.

        Three panels: XY (top view), XZ (front view), YZ (side view).
        All three subplots use the same spatial scale so they are the same size.

        Parameters
        ----------
        fov_distance : float or None
            Length of the FOV projection in metres.  Defaults to 30% of the
            maximum inter-hydrophone distance.
        show_fov : bool or list of int
            ``True`` draws FOV for all cameras; ``False`` hides all.  Pass a
            list of camera numbers (e.g. ``[0, 2]``) to show FOV only for
            those cameras.
        fov_alpha : float
            Opacity of the FOV fill (0–1).
        show_cameras : bool
            Draw camera markers.
        show_frame : bool
            Draw frame edges.
        frame_lw : float
            Frame edge line width.
        frame_alpha : float
            Frame edge opacity (0–1).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if fov_distance is None:
            fov_distance = 0.3 * float(np.max(self.distances())) if self.n > 1 else 1.0

        fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(15, 5))

        hp_kw_base = dict(s=80, marker='o', zorder=5,
                          edgecolors='white', linewidths=0.5)

        # Hydrophones
        for _, row in self._hydrophones.iterrows():
            ch    = int(row['hydrophone_number'])
            color = _palette_color(ch)
            name  = str(row.get('name', f'H{ch}'))
            kw    = dict(**hp_kw_base, color=color)
            ax_xy.scatter(row['x'], row['y'], label=name, **kw)
            ax_xz.scatter(row['x'], row['z'], **kw)
            ax_yz.scatter(row['y'], row['z'], **kw)

        # Frame
        if show_frame and self.frame is not None:
            verts = self._resolve_frame_vertices()
            for n0, n1 in self.frame['edges']:
                p0, p1 = verts[n0], verts[n1]
                fr_kw = dict(color='0.5', lw=frame_lw, alpha=frame_alpha, zorder=3)
                ax_xy.plot([p0[0], p1[0]], [p0[1], p1[1]], **fr_kw)
                ax_xz.plot([p0[0], p1[0]], [p0[2], p1[2]], **fr_kw)
                ax_yz.plot([p0[1], p1[1]], [p0[2], p1[2]], **fr_kw)

        # Cameras
        if show_cameras and not self._cameras.empty:
            for ci, cam in self._cameras.iterrows():
                cam_num = int(cam['camera_number'])
                color   = _palette_color(cam_num)
                name    = str(cam.get('name', f'CAM{ci}'))
                origin  = np.array([cam['x'], cam['y'], cam['z']])
                cam_kw  = dict(s=60, marker='s', color=color, zorder=6,
                               edgecolors='k', linewidths=0.8)
                ax_xy.scatter(cam['x'], cam['y'], label=name, **cam_kw)
                ax_xz.scatter(cam['x'], cam['z'], **cam_kw)
                ax_yz.scatter(cam['y'], cam['z'], **cam_kw)

                if _fov_visible(show_fov, cam_num):
                    corners = _camera_fov_rays(
                        cam['x'], cam['y'], cam['z'],
                        cam.get('horizontal_angle_deg', 0) or 0,
                        cam.get('vertical_angle_deg',   0) or 0,
                        cam.get('fov_horizontal_deg',  90) or 90,
                        cam.get('fov_vertical_deg',    90) or 90,
                        fov_distance,
                    )
                    fov_kw = dict(color=color, alpha=fov_alpha, zorder=2)
                    for (a0, a1), ax in (((0, 1), ax_xy), ((0, 2), ax_xz), ((1, 2), ax_yz)):
                        poly = _fov_polygon_2d(origin, corners, (a0, a1))
                        ax.fill(poly[:, 0], poly[:, 1], **fov_kw)

        # ------------------------------------------------------------------
        # Equal axis limits: same spatial span on all three subplots so they
        # render at the same physical size regardless of the data range.
        # ------------------------------------------------------------------
        all_x = list(self.x)
        all_y = list(self.y)
        all_z = list(self.z)
        if not self._cameras.empty:
            all_x += list(self._cameras['x'])
            all_y += list(self._cameras['y'])
            all_z += list(self._cameras['z'])
        if self.frame is not None:
            rv = self._resolve_frame_vertices()
            all_x += [v[0] for v in rv.values()]
            all_y += [v[1] for v in rv.values()]
            all_z += [v[2] for v in rv.values()]

        span = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z),
            0.1,
        )
        half  = span / 2 * 1.25   # 25 % padding
        x_ctr = (max(all_x) + min(all_x)) / 2
        y_ctr = (max(all_y) + min(all_y)) / 2
        z_ctr = (max(all_z) + min(all_z)) / 2

        for ax, xlabel, ylabel, title in [
            (ax_xy, 'X (m)', 'Y (m)', 'XY  (top view)'),
            (ax_xz, 'X (m)', 'Z (m)', 'XZ  (front view)'),
            (ax_yz, 'Y (m)', 'Z (m)', 'YZ  (side view)'),
        ]:
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(True, color='0.4', lw=0.6, alpha=0.6, linestyle='--')
            ax.set_aspect('equal')

        ax_xy.set_xlim(x_ctr - half, x_ctr + half)
        ax_xy.set_ylim(y_ctr - half, y_ctr + half)
        ax_xz.set_xlim(x_ctr - half, x_ctr + half)
        ax_xz.set_ylim(z_ctr - half, z_ctr + half)
        ax_yz.set_xlim(y_ctr - half, y_ctr + half)
        ax_yz.set_ylim(z_ctr - half, z_ctr + half)

        ax_xy.legend(fontsize=8, framealpha=0.9, edgecolor='0.7')
        fig.suptitle(self.array_name, fontsize=11)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _gift_wrap_2d(pts):
    """2D convex hull of *pts* using gift-wrapping (Jarvis march).

    Returns hull vertices in counter-clockwise order.  Works for any small
    point cloud; handles collinear points (they are excluded from the hull).

    Parameters
    ----------
    pts : numpy.ndarray, shape (n, 2)

    Returns
    -------
    numpy.ndarray, shape (h, 2)  where h <= n
    """
    n = len(pts)
    # Start from lex-minimum (bottom-left) — guaranteed on the hull
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    hull  = [start]
    while True:
        cur = hull[-1]
        nxt = (cur + 1) % n          # initial candidate (will be updated)
        for i in range(n):
            if i == cur:
                continue
            v_cn = pts[nxt] - pts[cur]
            v_ci = pts[i]   - pts[cur]
            # 2D cross product: negative → i is more CW than nxt → nxt is better
            cross = v_cn[0] * v_ci[1] - v_cn[1] * v_ci[0]
            if cross < 0:
                nxt = i
        if nxt == hull[0]:
            break
        hull.append(nxt)
        if len(hull) > n:            # safety guard against degenerate loops
            break
    return pts[np.array(hull)]


def _fov_polygon_2d(origin, corners, axes):
    """Project FOV frustum corners onto a 2D plane and return a filled polygon.

    Computes the convex hull of (camera origin + 4 frustum corner projections).

    * When the origin is at the tip of the wedge (e.g. top or side view of a
      forward-facing camera), it IS on the hull and is kept → triangle.
    * When the origin projects to the interior of the corner rectangle (e.g.
      front view of a camera pointing along the missing axis), it is correctly
      excluded → clean rectangle.

    Parameters
    ----------
    origin : numpy.ndarray, shape (3,)
        Camera position.
    corners : list of numpy.ndarray, length 4
        FOV corner endpoints as returned by :func:`_camera_fov_rays`.
    axes : (int, int)
        Two coordinate indices to project onto, e.g. ``(0, 1)`` for XY.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Convex-hull polygon vertices in CCW order, ready for ``ax.fill``.
    """
    a0, a1 = axes
    pts = np.array(
        [[origin[a0], origin[a1]]] + [[c[a0], c[a1]] for c in corners]
    )
    # Deduplicate (corners can collapse when camera points along a projected axis)
    _, unique_idx = np.unique(pts.round(10), axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]
    if len(pts) < 3:
        return pts      # degenerate; ax.fill will silently skip
    return _gift_wrap_2d(pts)


def _camera_fov_rays(cx, cy, cz, h_deg, v_deg, fov_h_deg, fov_v_deg, length):
    """Return the 4 corner endpoints of a camera FOV frustum at a given distance.

    Parameters
    ----------
    cx, cy, cz : float
        Camera position (metres).
    h_deg : float
        Horizontal pointing angle in degrees (azimuth from the +X axis).
    v_deg : float
        Vertical pointing angle in degrees (elevation above horizontal).
    fov_h_deg, fov_v_deg : float
        Full horizontal / vertical field of view in degrees.
    length : float
        Distance from the camera to the FOV plane (metres).

    Returns
    -------
    list of numpy.ndarray, length 4
        Corner endpoints ordered [bottom-left, top-left, bottom-right, top-right].
    """
    h   = np.radians(h_deg)
    v   = np.radians(v_deg)
    fh  = np.radians(fov_h_deg / 2)
    fv  = np.radians(fov_v_deg / 2)
    # Orthonormal basis: forward / right / up
    fwd = np.array([ np.cos(v) * np.cos(h),  np.cos(v) * np.sin(h),  np.sin(v)])
    rgt = np.array([-np.sin(h),               np.cos(h),              0.0      ])
    up  = np.array([-np.sin(v) * np.cos(h),  -np.sin(v) * np.sin(h), np.cos(v) ])
    origin = np.array([cx, cy, cz])
    corners = []
    for sh in (-1, 1):          # left / right
        for sv in (-1, 1):      # bottom / top
            d = fwd + sh * np.tan(fh) * rgt + sv * np.tan(fv) * up
            corners.append(origin + d / np.linalg.norm(d) * length)
    return corners  # BL, TL, BR, TR

