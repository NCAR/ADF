"""
Collection of python unit tests
for the contour level and percent difference options in plotting_utils.

A variable's defaults give contour levels as an explicit list, a range or a
linspace, either at the top level or in a section for one plot type. These
check each form is used as written, that settings outside a plot type's
section still apply, and that a bad setting falls back rather than raising.

NOTE: these import matplotlib, cartopy and the rest of the plotting stack, so
they need the full ADF environment, which CI runs them in (see
ADF_unit_tests.yaml).
"""

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
    import matplotlib

    matplotlib.use("Agg")
    import plotting_utils

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


class _Log:  # pylint: disable=too-few-public-methods
    """Collects debug messages in place of an ADF object."""

    def __init__(self):
        self.msgs = []

    def debug_log(self, msg):
        """Keep the message."""
        self.msgs.append(msg)


_POLAR_NAMES = {"NHPolar": "nh", "SHPolar": "sh"}


def _levels(plot_type_dict, kwargs=None):
    """Resolve the case contour levels."""
    return plotting_utils.resolve_levels(
        _Log(), "case", plot_type_dict, kwargs or {}, _POLAR_NAMES
    )


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class ResolveLevelsTestRoutine(unittest.TestCase):
    """
    Unit tests for resolve_levels.
    """

    def test_three_explicit_levels_are_used(self):
        """A three-value contour_levels list is used as is, as on main."""
        self.assertEqual(_levels({"contour_levels": [-1, 0, 1]}), [-1, 0, 1])

    def test_one_explicit_level_is_not_enough(self):
        """One level cannot make a contour plot: use the data range instead."""
        self.assertIsNone(_levels({"contour_levels": [5]}))

    def test_range_and_linspace(self):
        """A range is numpy.arange, a linspace is numpy.linspace."""
        np.testing.assert_allclose(
            _levels({"contour_levels_range": [0, 10, 5]}), [0, 5]
        )
        np.testing.assert_allclose(
            _levels({"contour_levels_linspace": [0, 10, 3]}), [0, 5, 10]
        )

    def test_bad_range_falls_through(self):
        """A range without exactly three values moves on to the next setting."""
        levels = _levels(
            {"contour_levels_range": [0, 10], "contour_levels_linspace": [0, 1, 3]}
        )
        np.testing.assert_allclose(levels, [0, 0.5, 1])

    def test_level_and_hemisphere_specific(self):
        """Settings can be split by hemisphere and pressure level."""
        entry = {"contour_levels_range": {"sh": {850: [0, 4, 2]}, "nh": [0, 9, 3]}}
        np.testing.assert_allclose(
            _levels(entry, {"hemi": "SHPolar", "lev": 850}), [0, 2]
        )
        np.testing.assert_allclose(_levels(entry, {"hemi": "NHPolar"}), [0, 3, 6])

    def test_top_level_setting_used_without_one_in_section(self):
        """A plot type section without levels uses the top-level ones."""
        self.assertEqual(_levels({}, {"contour_levels": [1, 2, 3, 4]}), [1, 2, 3, 4])


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class PercentDifferenceOptionsTestRoutine(unittest.TestCase):
    """
    Unit tests for the percent difference options of prep_contour_plot.
    """

    def setUp(self):
        self.data = xr.DataArray(
            np.linspace(-1, 1, 16).reshape(4, 4), dims=("lat", "lon")
        )

    def _prep(self, **kwargs):
        data = self.data
        return plotting_utils.prep_contour_plot(
            data, data, data - data, data - data, **kwargs
        )

    def test_top_level_settings_with_a_plot_type_section(self):
        """Top-level pct_diff settings still apply when a section exists."""
        cp_info = self._prep(
            plot_type="global_latlon_map",
            global_latlon_map={"colormap": "viridis"},
            pct_diff_colormap="RdBu",
            pct_diff_contour_levels=[-10, 0, 10],
        )
        self.assertEqual(cp_info["cmap_pctdiff"], "RdBu")
        self.assertEqual(list(cp_info["levels_pctdiff"]), [-10, 0, 10])

    def test_section_settings_win(self):
        """Settings in the plot type's section win over top-level ones."""
        cp_info = self._prep(
            plot_type="global_latlon_map",
            global_latlon_map={"pct_diff_colormap": "BrBG"},
            pct_diff_colormap="RdBu",
        )
        self.assertEqual(cp_info["cmap_pctdiff"], "BrBG")

    def test_bad_range_uses_default_levels(self):
        """A pct_diff_contour_range without three values does not stop the run."""
        cp_info = self._prep(pct_diff_contour_range=[0, 10])
        self.assertEqual(cp_info["levels_pctdiff"][0], -100)
        self.assertEqual(cp_info["cmap_pctdiff"], "PuOr_r")


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
