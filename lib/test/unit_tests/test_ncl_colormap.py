"""
Collection of python unit tests
for loading NCL colormaps in plotting_utils.

A variable's defaults can name an NCL colormap rather than a matplotlib one.
These check the loader reads the copies shipped in lib/plotting/colormaps/NCL,
downloads (once) only colormaps that are not there, skips lines it cannot
parse, and falls back to a matplotlib colormap when the NCL one cannot be had,
rather than raising part way through a plot.

NOTE: these import matplotlib, cartopy and the rest of the plotting stack, so
they are skipped in CI, which installs only PyYAML and pytest.
"""

# The tests reset the module's colormap cache:
# pylint: disable=protected-access

import os.path
import sys
import tempfile
import unittest
from pathlib import Path
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
255 0 10# comment with no space before it
255 255 255 ; trailing comment
"""


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class NclColormapTestRoutine(unittest.TestCase):
    """
    Unit tests for the NCL colormap helpers.
    """

    def setUp(self):
        """Point the NCL colormap directory at an empty one, with no cache."""
        self._tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.ncl_dir = Path(self._tmp.name)
        patches = [
            mock.patch.object(plotting_utils, "NCL_COLORMAP_DIR", self.ncl_dir),
            mock.patch.dict(plotting_utils._ncl_cmap_cache, clear=True),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, text=_RGB):
        (self.ncl_dir / f"{name}.rgb").write_text(text, encoding="utf-8")

    def test_read_skips_lines_it_cannot_parse(self):
        """Each color appears once, and comments do not eat digits."""
        self._write("mymap")
        table = plotting_utils.read_ncl_colormap(_Log(), self.ncl_dir / "mymap.rgb")
        self.assertEqual(table.shape, (4, 3))
        self.assertEqual(table[0].tolist(), [0, 0, 255])
        self.assertEqual(table[2].tolist(), [255, 0, 10])

    def test_single_color_file(self):
        """A one-color file is a (1, 3) table, not a crash."""
        self._write("onecolor", "255 0 0\n")
        cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "onecolor")
        self.assertEqual(cmap.N, 1)

    def test_local_file_is_loaded(self):
        """A colormap in the NCL directory is used, without going to NCL's website."""
        self._write("mymap")
        with mock.patch.object(plotting_utils, "urlopen") as fetch:
            cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "mymap")
        fetch.assert_not_called()
        self.assertEqual(cmap.name, "mymap")
        self.assertEqual(cmap.N, 4)

    def test_reversed_colormap(self):
        """ "<name>_r" is the reverse of a local colormap."""
        self._write("mymap")
        cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "mymap_r")
        self.assertEqual(cmap.name, "mymap_r")

    def test_missing_colormap_is_downloaded_and_saved(self):
        """A colormap that is not shipped is downloaded once and kept."""
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = _RGB.encode()
        with mock.patch.object(
            plotting_utils, "urlopen", return_value=response
        ) as fetch:
            cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "newmap")
            plotting_utils.try_load_ncl_cmap(_Log(), "newmap")
        fetch.assert_called_once()
        self.assertIn("timeout", fetch.call_args.kwargs)
        self.assertEqual(cmap.N, 4)
        self.assertTrue((self.ncl_dir / "newmap.rgb").is_file())

    def test_unavailable_colormap_falls_back_once(self):
        """No local file and no download: use the fallback, and do not retry."""
        with mock.patch.object(
            plotting_utils, "urlopen", side_effect=OSError("offline")
        ) as fetch:
            cmap, msg = plotting_utils.try_load_ncl_cmap(
                _Log(), "nosuchmap", fallback="PuOr_r"
            )
            cmap2, _ = plotting_utils.try_load_ncl_cmap(_Log(), "nosuchmap")
        self.assertEqual(cmap, "PuOr_r")
        self.assertEqual(cmap2, "viridis")
        self.assertIn("offline", msg)
        fetch.assert_called_once()

    def test_empty_file_falls_back(self):
        """A file with no colors in it is a failed load, not a crash."""
        self._write("emptymap", "ncolors=0\n")
        cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "emptymap")
        self.assertEqual(cmap, "viridis")

    def test_get_cmap_accepts_any_ncl_name(self):
        """Not just "ncl_default": any NCL colormap name works."""
        self._write("mymap")
        cmap = plotting_utils.get_cmap(_Log(), "case", {"colormap": "mymap"}, {}, {})
        self.assertEqual(cmap.name, "mymap")

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


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class ShippedNclColormapTestRoutine(unittest.TestCase):
    """
    The NCL colormaps shipped with the ADF all load.
    """

    def test_all_shipped_colormaps_load(self):
        """Every shipped file is an (N, 3) table of 0-255 (or 0-1) values."""
        files = sorted(plotting_utils.NCL_COLORMAP_DIR.glob("*.rgb"))
        self.assertGreater(len(files), 300)
        for fil in files:
            with self.subTest(colormap=fil.stem):
                table = plotting_utils.read_ncl_colormap(_Log(), fil)
                self.assertEqual(table.ndim, 2)
                self.assertEqual(table.shape[1], 3)
                self.assertGreaterEqual(table.min(), 0)
                self.assertLessEqual(table.max(), 255)

    def test_ncl_default_needs_no_download(self):
        """ncl_default, used by the shipped defaults, is read locally."""
        with mock.patch.object(plotting_utils, "urlopen") as fetch, mock.patch.dict(
            plotting_utils._ncl_cmap_cache,
            clear=True,
        ):
            cmap, _ = plotting_utils.try_load_ncl_cmap(_Log(), "ncl_default")
        fetch.assert_not_called()
        self.assertEqual(cmap.name, "ncl_default")


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
