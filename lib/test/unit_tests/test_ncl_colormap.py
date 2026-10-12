"""
Collection of python unit tests
for loading NCL colormaps in plotting_utils.

A variable's defaults can name an NCL colormap rather than a matplotlib one.
These check the loader reads a local "<name>.rgb" file, skips lines it cannot
parse, and falls back to a matplotlib colormap when the NCL one cannot be had,
rather than raising part way through a plot.

NOTE: these import matplotlib, cartopy and the rest of the plotting stack, so
they are skipped in CI, which installs only PyYAML and pytest.
"""

import os
import os.path
import sys
import tempfile
import unittest
from unittest import mock

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


class _Log:
    """Collects debug messages in place of an ADF object."""

    def __init__(self):
        self.msgs = []

    def debug_log(self, msg):
        """Keep the message."""
        self.msgs.append(msg)


_RGB = """ncolors=4
# r g b
0 0 255
not a color line
1.2.3 4 5
0 255 0
255 0 0 ; trailing comment
255 255 255
"""


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class NclColormapTestRoutine(unittest.TestCase):
    """
    Unit tests for the NCL colormap helpers.
    """

    def setUp(self):
        """Run each test in an empty directory, where local .rgb files go."""
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _write(self, name, text=_RGB):
        with open(f"{name}.rgb", "w", encoding="utf-8") as fil:
            fil.write(text)

    def test_read_skips_lines_it_cannot_parse(self):
        """Each color appears once: a bad line must not repeat the one before."""
        self._write("mymap")
        table = plotting_utils.read_ncl_colormap(_Log(), "mymap.rgb")
        self.assertEqual(table.shape, (4, 3))
        self.assertEqual(table[0].tolist(), [0, 0, 255])
        self.assertEqual(table[2].tolist(), [255, 0, 0])

    def test_local_file_is_loaded(self):
        """A local <name>.rgb file is used, without going to NCL's website."""
        self._write("mymap")
        with mock.patch.object(plotting_utils, "urlretrieve") as fetch:
            cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "mymap")
        fetch.assert_not_called()
        self.assertEqual(cmap.name, "mymap")
        self.assertEqual(cmap.N, 4)

    def test_load_colormap_with_a_local_file(self):
        """This used to raise TypeError when the local file existed."""
        self._write("mymap")
        cmap = plotting_utils.load_colormap(_Log(), "mymap")
        self.assertEqual(cmap.name, "mymap")

    def test_unavailable_colormap_falls_back(self):
        """No local file and no download: use the fallback, do not raise."""
        with mock.patch.object(
            plotting_utils, "urlretrieve", side_effect=OSError("offline")
        ):
            cmap, msg = plotting_utils.try_load_ncl_cmap(
                _Log(), "nosuchmap", fallback="PuOr_r"
            )
        self.assertEqual(cmap, "PuOr_r")
        self.assertIn("offline", msg)

    def test_empty_file_falls_back(self):
        """A file with no colors in it is a failed load, not a crash."""
        self._write("emptymap", "ncolors=0\n")
        cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "emptymap")
        self.assertEqual(cmap, "viridis")

    def test_percent_difference_ncl_colormap(self):
        """
        An NCL colormap for the percent difference panel used to raise
        TypeError ('"." / str') before any plot was drawn.
        """
        self._write("mymap")
        data = xr.DataArray(np.linspace(-1, 1, 16).reshape(4, 4), dims=("lat", "lon"))
        cp_info = plotting_utils.prep_contour_plot(
            data, data, data - data, data - data, pct_diff_colormap="mymap"
        )
        self.assertEqual(cp_info["cmap_pctdiff"].name, "mymap")


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
