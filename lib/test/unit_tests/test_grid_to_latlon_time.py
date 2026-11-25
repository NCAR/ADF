"""
Collection of python unit tests
for the time coordinate of "adf_utils.grid_to_latlon" output.

The ADF grids native-grid (CAM "ncol") time series to lat/lon and writes them
to a "gridded" directory, which the diagnostics then read with the time
coordinate moved to the midpoint of its bounds.  The regridders return a new
time coordinate, and when it lost the original's encoding the file was written
with time in hours since its first step but the carried-over time bounds in
days since 0001-01-01: read back, five years of months fell into 75 days and
the tape recorder failed (QBO and global-mean time series were quietly wrong).

NOTE: these tests import adf_utils and xesmf, so they are skipped in CI,
which installs only PyYAML and pytest.  They run in a full ADF environment.
"""

import os
import os.path
import sys
import tempfile
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    import xesmf  # noqa: F401  (grid_to_latlon needs it)
    from adf_utils import grid_to_latlon, use_time_bounds_midpoint
    from adf_diag import save_to_nc  # the writer AdfDiag.grid_native_time_series uses

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _identity_weights(path):
    """
    Write an ESMF-style weights file mapping 4 native columns one-to-one onto a
    2 x 2 lat/lon grid.
    """

    lat, lon = np.array([-45.0, 45.0]), np.array([90.0, 270.0])
    yc_b, xc_b = np.meshgrid(lat, lon, indexing="ij")
    xr.Dataset(
        {
            "src_grid_dims": (("src_grid_rank",), np.array([4], dtype="i4")),
            "dst_grid_dims": (("dst_grid_rank",), np.array([2, 2], dtype="i4")),
            "yc_b": (("n_b",), yc_b.ravel()),
            "xc_b": (("n_b",), xc_b.ravel()),
            "col": (("n_s",), np.arange(1, 5, dtype="i4")),
            "row": (("n_s",), np.arange(1, 5, dtype="i4")),
            "S": (("n_s",), np.ones(4)),
        }
    ).to_netcdf(path)


def _native_time_series(path):
    """
    Write five years of monthly native-grid data the way CAM time series are
    written: time and its bounds both in days since 0001-01-01.
    """

    nyears = 5
    month_len = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    edges = np.concatenate([[0], np.cumsum(np.tile(month_len, nyears))]).astype("f8")
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    ds = xr.Dataset(
        {
            "Q": (("time", "ncol"), np.ones((12 * nyears, 4), dtype="f4")),
            "time_bounds": (("time", "nbnd"), bounds),
        },
        coords={"time": ("time", bounds.mean(axis=1))},
    )
    ds["time"].attrs = {
        "long_name": "time",
        "units": "days since 0001-01-01 00:00:00",
        "calendar": "noleap",
        "bounds": "time_bounds",
    }
    ds["time_bounds"].attrs = {
        "units": "days since 0001-01-01 00:00:00",
        "calendar": "noleap",
    }
    ds.to_netcdf(path)


@unittest.skipUnless(_HAS_DEPS, "adf_utils/xesmf dependencies not available")
class GridToLatlonTimeTestRoutine(unittest.TestCase):
    """
    Unit tests for the time coordinate of gridded native-grid time series.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.wgt = os.path.join(self._tmp.name, "wgts.nc")
        self.native = os.path.join(self._tmp.name, "native.nc")
        self.gridded = os.path.join(self._tmp.name, "gridded.nc")
        _identity_weights(self.wgt)
        _native_time_series(self.native)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_gridded(self):
        with xr.open_dataset(self.native) as ts_ds:
            gridded = grid_to_latlon(ts_ds, "Q", wgt_file=self.wgt, method="bilinear")
            save_to_nc(gridded, self.gridded)

    def test_time_keeps_its_units_and_bounds(self):
        """
        The gridded file's time is encoded like the native file's, and still
        names its bounds.
        """

        self._write_gridded()

        with xr.open_dataset(self.gridded, decode_times=False) as ds:
            # (xarray may drop the " 00:00:00", which means the same)
            self.assertTrue(
                ds["time"].attrs.get("units", "").startswith("days since 0001-01-01")
            )
            self.assertEqual(ds["time"].attrs.get("bounds"), "time_bounds")

    def test_read_back_covers_every_month(self):
        """
        Read the way the time series loader reads it, the gridded file still
        holds five years of months, January to December.
        """

        self._write_gridded()

        with xr.open_dataset(self.gridded, decode_times=False) as ds:
            fixed = xr.decode_cf(use_time_bounds_midpoint(ds))
            months = [t.month for t in fixed["time"].values]
            years = sorted({t.year for t in fixed["time"].values})

        self.assertEqual(months, list(range(1, 13)) * 5)
        self.assertEqual(years, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
