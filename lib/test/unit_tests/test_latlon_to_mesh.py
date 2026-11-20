"""
Collection of python unit tests
for putting lat/lon fields onto a native mesh: "adf_utils.latlon_to_mesh",
"write_latlon_scrip", "make_mesh_weights" and "apply_mesh_weights".

The destination mesh here is a coarse lat/lon grid read as an unstructured
grid, so the expected values are known: a constant field stays constant, a
field that varies stays within its range, and missing source cells do not
pull the result towards 0.

NOTE: these tests import adf_utils, which imports xarray and uxarray, so they
are skipped in CI, which installs only PyYAML and pytest.  The ESMF test also
needs ESMF_RegridWeightGen on the PATH.
"""

import os
import os.path
import shutil
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
    import uxarray as ux
    from adf_utils import (
        latlon_to_mesh,
        write_latlon_scrip,
        make_mesh_weights,
        apply_mesh_weights,
        _nearest_index,
    )

    _HAS_ADF_UTILS = True
except ImportError:
    _HAS_ADF_UTILS = False


def _latlon(nlat, nlon, values=None):
    """A field on a regular lat/lon grid (cell centres, 0-360 longitudes)."""
    lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)
    lon = np.arange(nlon) * 360.0 / nlon + 180.0 / nlon
    if values is None:
        values = (
            np.cos(np.deg2rad(lat))[:, None] * (2 + np.sin(np.deg2rad(lon)))[None, :]
        )
    return xr.DataArray(
        np.broadcast_to(values, (nlat, nlon)).astype("f8"),
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        name="T",
        attrs={"units": "K"},
    )


@unittest.skipUnless(_HAS_ADF_UTILS, "needs xarray, uxarray and adf_utils")
class TestNearestIndex(unittest.TestCase):
    """The face-to-cell matching in latlon_to_mesh."""

    def test_unsorted_centres(self):
        """Centres need not be sorted."""
        centres = np.array([30.0, 10.0, 20.0])
        np.testing.assert_array_equal(
            _nearest_index(centres, np.array([11.0, 29.0, 19.0])), [1, 0, 2]
        )

    def test_periodic_seam(self):
        """Longitudes match across the 0/360 seam."""
        centres = np.array([5.0, 15.0, 355.0])
        # 359 is nearest 355, 1 is nearest 5 (not 355 - 360 apart):
        np.testing.assert_array_equal(
            _nearest_index(centres, np.array([359.0, 1.0]), period=360), [2, 0]
        )
        # 0.5 is nearer 355 across the seam than 15:
        self.assertEqual(
            _nearest_index(np.array([15.0, 355.0]), np.array([0.5]), period=360)[0], 1
        )


@unittest.skipUnless(_HAS_ADF_UTILS, "needs xarray, uxarray and adf_utils")
class TestLatlonToMesh(unittest.TestCase):
    """uxarray remap onto a mesh."""

    def setUp(self):
        dst = _latlon(18, 36)
        self.mesh = ux.Grid.from_structured(
            lon=dst["lon"].values, lat=dst["lat"].values
        )

    def test_constant_stays_constant(self):
        """A constant field stays constant, with its units kept."""
        out = latlon_to_mesh(_latlon(45, 90, values=7.0), self.mesh)
        self.assertEqual(out.dims, ("n_face",))
        np.testing.assert_allclose(out.values, 7.0, rtol=1e-12)
        self.assertEqual(out.attrs["units"], "K")
        self.assertIn("bilinear", out.attrs["remap_method"])

    def test_values_follow_the_field(self):
        """The remapped field stays in the source's range and shape."""
        src = _latlon(45, 90)
        out = latlon_to_mesh(src, self.mesh)
        self.assertLessEqual(float(out.max()), float(src.max()) + 1e-9)
        self.assertGreaterEqual(float(out.min()), float(src.min()) - 1e-9)
        # Larger near the equator than near the poles, as the source:
        flat = out.uxgrid.face_lat.values
        self.assertGreater(
            float(out.values[np.abs(flat) < 10].mean()),
            float(out.values[np.abs(flat) > 70].mean()),
        )

    def test_extra_dims_kept(self):
        """Dimensions other than lat/lon (e.g. time) are kept."""
        src = _latlon(45, 90).expand_dims(time=3).copy()
        src[1] = src[1] * 2
        out = latlon_to_mesh(src, self.mesh)
        self.assertEqual(out.dims, ("time", "n_face"))
        np.testing.assert_allclose(
            out.isel(time=1).values, 2 * out.isel(time=0).values, rtol=1e-12
        )


@unittest.skipUnless(_HAS_ADF_UTILS, "needs xarray, uxarray and adf_utils")
class TestApplyMeshWeights(unittest.TestCase):
    """Applying a weights file, with a hand-made map so no ESMF is needed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        dst = _latlon(2, 2)  # four destination faces (uxarray needs 2 rows)
        self.mesh = ux.Grid.from_structured(
            lon=dst["lon"].values, lat=dst["lat"].values
        )
        # Source 1 x 4; face 0 averages source cells 0 and 1 (weights 0.25,
        # 0.75), face 1 cells 2 and 3 (0.5 each); faces 2 and 3 get nothing:
        self.wfile = os.path.join(self.tmp, "w.nc")
        xr.Dataset(
            {
                "S": ("n_s", [0.25, 0.75, 0.5, 0.5]),
                "row": ("n_s", np.array([1, 1, 2, 2], dtype="i4")),
                "col": ("n_s", np.array([1, 2, 3, 4], dtype="i4")),
                "frac_a": ("n_a", np.ones(4)),
                "frac_b": ("n_b", np.ones(4)),
            }
        ).to_netcdf(self.wfile)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_weighted_sum(self):
        """Each face is the weighted sum of its source cells."""
        src = _latlon(1, 4, values=np.array([[1.0, 5.0, 2.0, 4.0]]))
        out = apply_mesh_weights(src, self.wfile, self.mesh)
        np.testing.assert_allclose(out.values[:2], [0.25 + 3.75, 3.0])
        self.assertTrue(np.isnan(out.values[2:]).all())

    def test_missing_cells_renormalized(self):
        """Missing source cells are left out, not counted as 0."""
        src = _latlon(1, 4, values=np.array([[np.nan, 5.0, np.nan, np.nan]]))
        out = apply_mesh_weights(src, self.wfile, self.mesh)
        self.assertEqual(out.values[0], 5.0)  # not 0.75 * 5
        self.assertTrue(np.isnan(out.values[1]))  # no valid source


@unittest.skipUnless(
    _HAS_ADF_UTILS and shutil.which("ESMF_RegridWeightGen"),
    "needs adf_utils and ESMF_RegridWeightGen",
)
class TestEsmfWeights(unittest.TestCase):
    """SCRIP file and conservative weights through ESMF, lat/lon to lat/lon."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_conserves_global_mean(self):
        """Conservative ESMF weights keep the area-weighted mean."""
        src = _latlon(36, 72)
        dst = _latlon(18, 36)
        s_file = write_latlon_scrip(
            src["lat"], src["lon"], os.path.join(self.tmp, "s.nc")
        )
        d_file = write_latlon_scrip(
            dst["lat"], dst["lon"], os.path.join(self.tmp, "d.nc")
        )
        wfile = os.path.join(self.tmp, "w.nc")
        # Both SCRIP, so tell ESMF the destination type:
        import subprocess  # pylint: disable=import-outside-toplevel

        subprocess.run(
            [
                "ESMF_RegridWeightGen",
                "-s",
                str(s_file),
                "-d",
                str(d_file),
                "-w",
                wfile,
                "-m",
                "conserve",
                "--no_log",
                "-i",
            ],
            check=True,
            capture_output=True,
        )
        mesh = ux.Grid.from_structured(lon=dst["lon"].values, lat=dst["lat"].values)
        out = apply_mesh_weights(src, wfile, mesh)
        coslat = np.cos(np.deg2rad(src["lat"]))
        src_mean = float((src * coslat).sum() / (coslat.sum() * src.sizes["lon"]))
        area = mesh.face_areas.values
        out_mean = float((out.values * area).sum() / area.sum())
        self.assertAlmostEqual(out_mean, src_mean, places=3)
        self.assertTrue(callable(make_mesh_weights))


if __name__ == "__main__":
    unittest.main()
