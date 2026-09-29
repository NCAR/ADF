"""
Collection of python unit tests
for "adf_utils.domain_stats".

The region mean has to be the same area-weighted mean whichever longitude
convention the data uses and whichever way latitude runs.  The polar plots ask
for [-180, 180, 45, 90] while regridded CAM output is on 0 to 360, and a
selection that did not allow for that averaged half of the cap.

NOTE: these tests import adf_utils, which imports xarray, so they are skipped
in CI, which installs only PyYAML and pytest.  They run in a full ADF
environment.
"""

import os
import os.path
import sys
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    from adf_utils import domain_stats, spatial_average

    _HAS_ADF_UTILS = True
except ImportError:
    _HAS_ADF_UTILS = False


def _field(lon, lat):
    """A field that varies with both longitude and latitude."""
    lon = np.asarray(lon, dtype="f8")
    lat = np.asarray(lat, dtype="f8")
    vals = np.sin(np.deg2rad(lon))[None, :] * 10 + lat[:, None]
    return xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def _brute_force(da, lon_ok, lat_ok):
    """cos(lat)-weighted mean over the points picked by two boolean tests."""
    lat = da.lat.values
    lon = da.lon.values
    sel = lat_ok(lat)[:, None] & lon_ok(lon % 360)[None, :]
    wgt = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], da.shape)
    return float((da.values * wgt)[sel].sum() / wgt[sel].sum())


@unittest.skipUnless(_HAS_ADF_UTILS, "adf_utils dependencies not available")
class DomainStatsTestRoutine(unittest.TestCase):
    """Unit tests for the regional mean, maximum and minimum."""

    lat = np.arange(-89.0, 90.0, 2.0)

    def test_same_answer_for_either_longitude_convention(self):
        """The polar cap must not depend on how longitude is numbered."""
        east = _field(np.arange(0.0, 360.0, 2.5), self.lat)
        west = east.assign_coords(lon=((east.lon + 180) % 360) - 180).sortby("lon")
        domain = [-180, 180, 45, 90]
        got_east = domain_stats(east, domain)
        got_west = domain_stats(west, domain)
        expected = _brute_force(east, lambda x: x >= 0, lambda x: x >= 45)
        self.assertAlmostEqual(got_east[0], expected, places=10)
        np.testing.assert_allclose(got_east, got_west, rtol=1e-12)

    def test_latitude_can_descend(self):
        """A north-to-south latitude axis gives the same answer."""
        up = _field(np.arange(0.0, 360.0, 5.0), self.lat)
        down = up.isel(lat=slice(None, None, -1))
        np.testing.assert_allclose(
            domain_stats(up, [0, 360, -90, -45]),
            domain_stats(down, [0, 360, -90, -45]),
            rtol=1e-12,
        )

    def test_box_that_crosses_the_zero_line(self):
        """West greater than east means the region wraps through 0/360."""
        da = _field(np.arange(0.0, 360.0, 2.5), self.lat)
        mean, vmax, vmin = domain_stats(da, [350, 10, -30, 30])
        expected = _brute_force(
            da, lambda x: (x >= 350) | (x <= 10), lambda x: abs(x) <= 30
        )
        self.assertAlmostEqual(mean, expected, places=10)
        self.assertLessEqual(vmax, da.max().item())
        self.assertGreaterEqual(vmin, da.min().item())

    def test_whole_globe_is_spatial_average(self):
        """A global box reproduces the shared weighted mean."""
        da = _field(np.arange(0.0, 360.0, 2.5), self.lat)
        mean, vmax, vmin = domain_stats(da, [-180, 180, -90, 90])
        self.assertAlmostEqual(mean, spatial_average(da).item(), places=10)
        self.assertEqual(vmax, da.max().item())
        self.assertEqual(vmin, da.min().item())

    def test_missing_values_are_left_out(self):
        """NaN points drop out of the mean rather than counting as zero."""
        da = _field(np.arange(0.0, 360.0, 5.0), self.lat)
        da[-3:, :10] = np.nan
        mean = domain_stats(da, [0, 360, 45, 90])[0]
        good = da.notnull().values
        wgt = np.broadcast_to(np.cos(np.deg2rad(da.lat.values))[:, None], da.shape)
        sel = good & (da.lat.values >= 45)[:, None]
        expected = float((da.values * wgt)[sel].sum() / wgt[sel].sum())
        self.assertAlmostEqual(mean, expected, places=10)

    def test_edges_of_the_longitude_range(self):
        """Boxes ending on 0, 180 or 360 pick the right points."""
        da = _field(np.array([0.0, 90.0, 180.0, 270.0, 360.0]), self.lat)
        whole = domain_stats(da, [-180, 180, -90, 90])
        for domain in ([0, 360, -90, 90], [180, 540, -90, 90]):
            np.testing.assert_allclose(domain_stats(da, domain), whole, rtol=1e-12)
        mean = domain_stats(da, [0, 180, -90, 90])[0]
        expected = _brute_force(da, lambda x: (x >= 0) & (x <= 180), lambda x: x > -91)
        self.assertAlmostEqual(mean, expected, places=10)


if __name__ == "__main__":
    unittest.main()
