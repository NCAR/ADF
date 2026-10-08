"""
Collection of python unit tests
for "adf_utils.seasonal_diffs".

The lat/lon and polar maps both use this for their seasonal means and
differences.  It has to work for fields written with no units (e.g. AODVISdn),
and has to leave the percent difference missing where it cannot be computed,
so masked points plot as blank rather than as 0%.

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
    from adf_utils import seasonal_diffs

    _HAS_ADF_UTILS = True
except ImportError:
    _HAS_ADF_UTILS = False


def _climo(vals, **attrs):
    """A 12-month climatology on a 2-point lat grid, constant in time."""
    data = np.broadcast_to(np.asarray(vals, dtype="f8"), (12, len(vals)))
    return xr.DataArray(
        data.copy(),
        dims=("time", "lat"),
        coords={"time": np.arange(1, 13), "lat": [10.0, 20.0]},
        attrs=attrs,
    )


@unittest.skipUnless(_HAS_ADF_UTILS, "adf_utils dependencies not available")
class SeasonalDiffsTestRoutine(unittest.TestCase):
    """Unit tests for the seasonal means, difference and percent difference."""

    def test_values_and_units(self):
        """Difference and percent difference, with the test case's units."""
        _, _, dif, pct = seasonal_diffs(
            _climo([3.0, 6.0], units="K"), _climo([2.0, 4.0], units="K"), "DJF"
        )
        np.testing.assert_allclose(dif.values, [1.0, 2.0])
        np.testing.assert_allclose(pct.values, [50.0, 50.0])
        self.assertEqual(dif.attrs["units"], "K")
        self.assertEqual(pct.attrs["units"], "%")

    def test_no_units(self):
        """A field with no units attribute (e.g. AODVISdn) does not fail."""
        _, _, dif, pct = seasonal_diffs(_climo([3.0, 6.0]), _climo([2.0, 4.0]), "JJA")
        self.assertNotIn("units", dif.attrs)
        self.assertEqual(pct.attrs["units"], "%")

    def test_missing_stays_missing(self):
        """Masked points and a zero reference give NaN, not 0%."""
        _, _, _, pct = seasonal_diffs(_climo([1.0, 1.0]), _climo([np.nan, 0.0]), "ANN")
        self.assertTrue(np.isnan(pct.values).all())


if __name__ == "__main__":
    unittest.main()
