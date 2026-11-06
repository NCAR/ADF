"""
Collection of python unit tests
for plotting whatever data there is.

When only the test case or only the baseline has a variable, the zonal and
meridional mean scripts plot that one on its own (plot_single_case_and_save)
instead of skipping the variable, and the scripts reading time series straight
from 'cam_ts_loc' leave out the cases without them
(AdfData.timeseries_available with any_case=True).

NOTE: the plot tests import matplotlib, cartopy and the rest of the plotting
stack, so they are skipped in CI, which installs only PyYAML and pytest.
"""

import os.path
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    import matplotlib

    matplotlib.use("Agg")
    import plotting_functions

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False

try:
    from adf_dataset import AdfData

    _HAS_DATASET = True
except ImportError:
    _HAS_DATASET = False


class _Log:  # pylint: disable=too-few-public-methods
    """Collects debug messages in place of an ADF object."""

    def __init__(self):
        self.msgs = []

    def debug_log(self, msg):
        """Keep the message."""
        self.msgs.append(msg)


def _field(has_lev):
    """A (lev,) lat, lon field with units, as a seasonal mean would be."""
    lat = np.linspace(-89, 89, 90)
    lon = np.arange(0, 360, 4.0)
    if has_lev:
        lev = np.array([10.0, 100.0, 500.0, 1000.0])
        data = (
            200
            + lev[:, None, None] / 10
            + np.cos(np.deg2rad(lat))[None, :, None] * np.ones(lon.size)
        )
        return xr.DataArray(
            data,
            dims=("lev", "lat", "lon"),
            coords={"lev": lev, "lat": lat, "lon": lon},
            attrs={"units": "K"},
        )
    data = (
        250 + 30 * np.cos(np.deg2rad(lat))[:, None] + np.sin(np.deg2rad(lon))[None, :]
    )
    return xr.DataArray(
        data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs={"units": "K"}
    )


@unittest.skipUnless(_HAS_PLOTTING, "plotting_functions dependencies not available")
class SingleCasePlotTestRoutine(unittest.TestCase):
    """
    Unit tests for plot_single_case_and_save.
    """

    def _plot(self, kind, role, has_lev, log_p=False):
        with tempfile.TemporaryDirectory() as tmp:
            wks = Path(tmp) / f"{kind}_{role}.png"
            plotting_functions.plot_single_case_and_save(
                _Log(),
                wks,
                kind,
                role,
                "case",
                [1, 5],
                _field(has_lev),
                has_lev,
                log_p=log_p,
                var_name="T",
                season="ANN",
            )
            self.assertTrue(wks.is_file())
            self.assertGreater(wks.stat().st_size, 0)

    def test_zonal_line_and_pressure_latitude(self):
        """A 2-d field is a line, a 3-d one a pressure-latitude panel."""
        self._plot("zonal", "Test", has_lev=False)
        self._plot("zonal", "Baseline", has_lev=True)
        self._plot("zonal", "Test", has_lev=True, log_p=True)

    def test_meridional_line_and_pressure_longitude(self):
        """The same for the 5S-5N meridional mean."""
        self._plot("meridional", "Baseline", has_lev=False)
        self._plot("meridional", "Test", has_lev=True)


@unittest.skipUnless(_HAS_DATASET, "adf_dataset dependencies not available")
class TimeseriesAvailableTestRoutine(unittest.TestCase):
    """
    Unit tests for AdfData.timeseries_available.
    """

    def _data(self, tmp, with_files):
        """Two test cases and a baseline; `with_files` says which have time series."""
        locs = {}
        for case in ("a", "b", "base"):
            loc = Path(tmp) / case
            loc.mkdir()
            if case in with_files:
                (loc / f"{case}.cam.h0a.T.000101-000512.nc").touch()
            locs[case] = str(loc)
        info = {"cam_ts_loc": [locs["a"], locs["b"]]}
        adf = SimpleNamespace(
            compare_obs=False,
            get_cam_info=info.get,
            get_baseline_info=lambda key: locs["base"],
            debug_log=lambda msg: None,
        )
        return SimpleNamespace(case_names=["a", "b"], ref_case_label="base", adf=adf)

    def _available(self, with_files, any_case):
        with tempfile.TemporaryDirectory() as tmp:
            obj = self._data(tmp, with_files)
            return AdfData.timeseries_available(obj, "test", any_case=any_case)

    def test_every_case_needed_by_default(self):
        """By default one case without time series skips the script."""
        self.assertTrue(self._available({"a", "b", "base"}, any_case=False))
        self.assertFalse(self._available({"a", "b"}, any_case=False))

    def test_any_case_is_enough_when_asked(self):
        """With any_case, the script runs while one case has time series."""
        self.assertTrue(self._available({"a"}, any_case=True))
        self.assertFalse(self._available(set(), any_case=True))


if __name__ == "__main__":
    unittest.main()
