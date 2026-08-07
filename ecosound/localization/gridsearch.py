"""
ecosound.localization.gridsearch
================================
GridSearch localization: pre-computed TDOA lookup table, grid-search inversion,
and PPD-based credibility intervals.
"""

import numpy as np
import pandas as pd


class GridSearch:
    """Grid-search 3D localization with a pre-computed TDOA table.

    Instantiate with the output of :meth:`load_tdoa_grid` to get an object
    whose :meth:`localize` method can be called once per detection.

    Parameters
    ----------
    tdoa_grid : dict
        As returned by :meth:`load_tdoa_grid`.
    """

    def __init__(self, tdoa_grid):
        self._grid = tdoa_grid
        self.modelled_tdoa = tdoa_grid["tdoa"]      # (n_pairs, n_pts)
        self.grid_coords   = tdoa_grid["grid_coord"]  # (n_pts, 3)
        self.x_unique = np.unique(self.grid_coords[:, 0])
        self.y_unique = np.unique(self.grid_coords[:, 1])
        self.z_unique = np.unique(self.grid_coords[:, 2])

    # ------------------------------------------------------------------
    # Per-detection localization
    # ------------------------------------------------------------------

    def localize(self, tdoa_ref_sec, corr_ref, min_corr_val, CI_percentage):
        """Run grid-search localization for a single detection.

        Parameters
        ----------
        tdoa_ref_sec : numpy.ndarray, shape (n_pairs, 1)
            Measured TDOAs in seconds.
        corr_ref : numpy.ndarray, shape (n_pairs, 1)
            Peak correlation values per pair.
        min_corr_val : float
            Minimum acceptable correlation; *all* pairs must meet this threshold
            for the detection to be localised.
        CI_percentage : float
            Credibility-interval coverage (e.g. 0.68 for 68%).

        Returns
        -------
        m : numpy.ndarray, shape (3,)
            MAP estimate (x, y, z) in metres.
        Px_CI : [float, float]
        Py_CI : [float, float]
        Pz_CI : [float, float]
            [low, high] credibility intervals.
        loc_ok : bool
        PPD_xr : xarray.Dataset or None
            3-D PPD as an xarray Dataset; ``None`` if ``loc_ok`` is ``False``.
        """
        loc_ok = bool(np.min(corr_ref) >= min_corr_val)

        delta_tdoa      = self.modelled_tdoa - tdoa_ref_sec   # (n_pairs, n_pts)
        delta_tdoa_norm = np.linalg.norm(delta_tdoa, axis=0)
        min_idx         = int(np.argmin(delta_tdoa_norm))
        m               = self.grid_coords[min_idx]

        if not loc_ok:
            nan2 = [np.nan, np.nan]
            return m, nan2, nan2, nan2, loc_ok, None

        tdoa_err_std = np.sqrt(
            np.sum(delta_tdoa[:, min_idx] ** 2)
            / (delta_tdoa[:, min_idx].size - 1)
        )
        L   = np.exp(-delta_tdoa_norm ** 2 / (2 * tdoa_err_std ** 2))
        PPD = L / L.sum()

        PPD_xr = pd.DataFrame({
            "x": self.grid_coords[:, 0],
            "y": self.grid_coords[:, 1],
            "z": self.grid_coords[:, 2],
            "PPD": PPD,
        }).set_index(["x", "y", "z"]).to_xarray()

        Px = PPD_xr.PPD.sum("z").sum("y")
        Py = PPD_xr.PPD.sum("x").sum("z")
        Pz = PPD_xr.PPD.sum("x").sum("y")

        Px_CI = GridSearch.calc_credibility_interval(
            Px["x"].values, Px.to_numpy(), m[0], CI_percentage
        )
        Py_CI = GridSearch.calc_credibility_interval(
            Py["y"].values, Py.to_numpy(), m[1], CI_percentage
        )
        Pz_CI = GridSearch.calc_credibility_interval(
            Pz["z"].values, Pz.to_numpy(), m[2], CI_percentage
        )

        return m, Px_CI, Py_CI, Pz_CI, loc_ok, PPD_xr

    # ------------------------------------------------------------------
    # Grid creation and loading
    # ------------------------------------------------------------------

    @staticmethod
    def create_tdoa_grid(x_limits, y_limits, z_limits, spacing,
                         array_or_config, ref_channel, sound_speed_mps, outfile):
        """Pre-compute and save the modelled-TDOA grid.

        Parameters
        ----------
        x_limits, y_limits, z_limits : [float, float]
            Search-volume extent in each dimension (metres).
        spacing : float
            Grid spacing in metres.
        array_or_config : SensorArray or pandas.DataFrame
            Hydrophone configuration.  A DataFrame (with x, y, z columns) is
            accepted for backward compatibility.
        ref_channel : int
            Reference hydrophone index (0-based position in the config).
        sound_speed_mps : float
        outfile : str
            Output path for the .npz file.
        """
        # Accept either SensorArray or raw DataFrame
        try:
            config_df = array_or_config.config   # HydrophoneArray
        except AttributeError:
            config_df = array_or_config          # DataFrame

        n_hp   = len(config_df)
        pairs  = np.array([[ref_channel, i] for i in range(n_hp) if i != ref_channel])
        hydrophones_coord = config_df[["x", "y", "z"]].to_numpy()

        # Build Cartesian grid
        vec_x = np.arange(min(x_limits), max(x_limits), spacing)
        vec_y = np.arange(min(y_limits), max(y_limits), spacing)
        vec_z = np.arange(min(z_limits), max(z_limits), spacing)
        X, Y, Z = np.meshgrid(vec_x, vec_y, vec_z, indexing="ij")
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        tdoas = _predict_tdoa(points, sound_speed_mps, hydrophones_coord, pairs)

        np.savez(
            outfile,
            tdoa=tdoas,
            grid_coord=points,
            x_limits=x_limits,
            y_limits=y_limits,
            z_limits=z_limits,
            spacing=spacing,
            ref_channel=ref_channel,
            sound_speed_mps=sound_speed_mps,
            hydrophones_coord=hydrophones_coord,
        )

    @staticmethod
    def load_tdoa_grid(grid_file):
        """Load a pre-computed TDOA grid from a .npz file.

        Parameters
        ----------
        grid_file : str
            Path to the .npz file created by :meth:`create_tdoa_grid`.

        Returns
        -------
        dict
        """
        npz = np.load(grid_file)
        return {
            "tdoa":             npz["tdoa"],
            "grid_coord":       npz["grid_coord"],
            "x_limits":         npz["x_limits"],
            "y_limits":         npz["y_limits"],
            "z_limits":         npz["z_limits"],
            "spacing":          npz["spacing"],
            "ref_channel":      npz["ref_channel"],
            "sound_speed_mps":  npz["sound_speed_mps"],
            "hydrophones_coord": npz["hydrophones_coord"],
        }

    # ------------------------------------------------------------------
    # Credibility interval
    # ------------------------------------------------------------------

    @staticmethod
    def calc_credibility_interval(axis, values, start_value, percentage):
        """Compute a centred credibility interval from a 1-D marginal PPD.

        Parameters
        ----------
        axis : numpy.ndarray
            Coordinate values (e.g. x-grid points).
        values : numpy.ndarray
            Marginal probability values (same length as axis).
        start_value : float
            The MAP coordinate; the interval is centred here.
        percentage : float
            Desired coverage (e.g. 0.68 for 68%).

        Returns
        -------
        [float, float]
            [lower_bound, upper_bound]
        """
        values = values / values.sum()
        start_index = int(np.where(axis == start_value)[0])
        IC_low_idx,  remainder = _find_half_ci(values, start_index,     percentage / 2,             -1)
        IC_high_idx, remainder = _find_half_ci(values, start_index + 1, percentage / 2 + remainder,  1)
        if remainder > 0:
            IC_low_idx, remainder = _find_half_ci(values, IC_low_idx, remainder, -1)
        return [axis[IC_low_idx], axis[IC_high_idx]]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _find_half_ci(values, start_index, stop_val, step):
    """Accumulate probability from start_index in direction step until stop_val."""
    min_val     = 0.001 * max(values)
    current_idx = start_index
    current_sum = 0.0
    max_idx     = len(values) - 1
    while (current_sum < stop_val) and (0 <= current_idx <= max_idx):
        current_sum += values[current_idx]
        if values[current_idx] < min_val:
            break
        current_idx += step
    remainder = stop_val - current_sum
    return current_idx - step, remainder


def _predict_tdoa(m, V, hydrophones_coords, hydrophones_pairs):
    """Vectorised forward model: TDOAs for an array of source locations.

    Parameters
    ----------
    m : numpy.ndarray, shape (n_pts, 3)
        Source location candidates.
    V : float
        Sound speed (m/s).
    hydrophones_coords : numpy.ndarray, shape (n_hp, 3)
    hydrophones_pairs : numpy.ndarray, shape (n_pairs, 2)

    Returns
    -------
    numpy.ndarray, shape (n_pairs, n_pts)
    """
    n_pairs = hydrophones_pairs.shape[0]
    n_pts   = m.shape[0]
    dt      = np.empty((n_pairs, n_pts))
    for k in range(n_pairs):
        p1 = hydrophones_pairs[k, 0]
        p2 = hydrophones_pairs[k, 1]
        d1 = np.linalg.norm(m - hydrophones_coords[p1], axis=1)
        d2 = np.linalg.norm(m - hydrophones_coords[p2], axis=1)
        dt[k] = (d2 - d1) / V
    return dt
