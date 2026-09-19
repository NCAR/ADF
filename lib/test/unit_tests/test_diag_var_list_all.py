"""
Collection of python unit tests
for the "diag_var_list: all" variable search.

Instead of listing every variable to process, a user can write "all" in
'diag_var_list' and have the ADF read the variable names out of the input
files.  Two things have to hold for that to be useful: history files must
yield the fields and not the coefficients and counters sitting alongside them
(hyam, date, nsteph ...), and a pre-made time series directory must yield the
same names without opening hundreds of files.

NOTE: these tests import adf_info, which pulls in xarray and (through
adf_utils) the rest of the scientific stack.  The ADF unit test workflow
installs only PyYAML and pytest, so they are skipped there and only run in a
full ADF environment.  The file-name half of the search lives in
adf_file_utils and is tested in CI by test_adf_file_utils.
"""

# +++++++++++++++++++++++
# Import required modules
# +++++++++++++++++++++++

import unittest
import sys
import os
import os.path
import tempfile
from pathlib import Path

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    from adf_info import AdfInfo, variables_in_case_files

    _HAS_ADF_INFO = True
except ImportError:
    _HAS_ADF_INFO = False


def _write_hist(path):
    """Write a small history file holding fields alongside the usual clutter."""
    time = xr.date_range(
        "0001-01-01", periods=1, freq="MS", calendar="noleap", use_cftime=True
    )
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    lev = np.array([500.0, 850.0])
    coords = {"time": time, "lat": lat, "lon": lon, "lev": lev}
    ds = xr.Dataset(
        {
            "TS": (("time", "lat", "lon"), np.ones((1, 2, 2), dtype="f4")),
            "T": (("time", "lev", "lat", "lon"), np.ones((1, 2, 2, 2), dtype="f4")),
            # Not diagnostics: no horizontal dimension, or no time dimension.
            "hyam": (("lev",), np.ones(2, dtype="f8")),
            "date": (("time",), np.array([10101], dtype="i4")),
            "gw": (("lat",), np.ones(2, dtype="f8")),
        },
        coords=coords,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class _StubInfo:
    """
    Minimal stand-in for AdfInfo.

    'expand_var_list_all' reads the test case config entries and logs, so the
    method can be exercised unbound without building a real AdfInfo (which
    needs a full config file and existing case directories).
    """

    def __init__(self, cam_info):
        self.cam_info = cam_info
        self.messages = []

    def get_cam_info(self, var_str, required=False):
        """Return the stubbed 'diag_cam_climo' entry."""
        return self.cam_info.get(var_str)

    def debug_log(self, msg):
        """Collect rather than write, so tests can assert on the log."""
        self.messages.append(msg)

    def end_diag_fail(self, msg):
        """Raise rather than exit, so a failure can be asserted on."""
        raise RuntimeError(msg)


def _expand(stub, var_list):
    """Call the method unbound with the stub standing in for self."""
    return AdfInfo.expand_var_list_all(stub, var_list)


@unittest.skipUnless(_HAS_ADF_INFO, "adf_info dependencies not available")
class DiagVarListAllTestRoutine(unittest.TestCase):
    """
    Unit tests for the "all" variable search.
    """

    def test_history_files_yield_fields_only(self):
        """Fields come back; coefficients, counters and weights do not."""

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_hist(Path(tmpdir) / "case.cam.h0a.0001-01.nc")

            found = variables_in_case_files(
                "case", tmpdir, "cam.h0a", None, ts_done=False
            )

            self.assertEqual(found, {"TS", "T"})

    def test_history_stream_is_respected(self):
        """A file from another stream must not contribute its variables."""

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_hist(Path(tmpdir) / "case.cam.h1.0001-01.nc")

            found = variables_in_case_files(
                "case", tmpdir, "cam.h0a", None, ts_done=False
            )

            self.assertEqual(found, set())

    def test_premade_time_series_read_from_file_names(self):
        """Pre-made time series give up their variables without being opened."""

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in (
                "case.cam.h0a.TS.000101-002012.nc",
                "case.cam.h0a.PRECT.000101-002012.nc",
                "othercase.cam.h0a.RELHUM.000101-002012.nc",
            ):
                # Empty files: the names alone carry the variable, so nothing
                # here should need to be a readable dataset.
                (Path(tmpdir) / name).touch()

            found = variables_in_case_files(
                "case", None, "cam.h0a", tmpdir, ts_done=True
            )

            self.assertEqual(found, {"TS", "PRECT"})

    def test_all_expands_and_keeps_listed_variables(self):
        """
        "all" is replaced by what the cases hold, from every test case, and a
        variable listed alongside it (a derived one, say) is kept.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "case_a.cam.h0a.TS.000101-002012.nc").touch()
            (Path(tmpdir) / "case_b.cam.h0a.SWCF.000101-002012.nc").touch()

            stub = _StubInfo(
                {
                    "cam_case_name": ["case_a", "case_b"],
                    "cam_hist_loc": [None, None],
                    "hist_str": [["cam.h0a"], ["cam.h0a"]],
                    "cam_ts_loc": [tmpdir, tmpdir],
                    "cam_ts_done": [True, True],
                }
            )

            self.assertEqual(_expand(stub, ["all", "PRECT"]), ["PRECT", "SWCF", "TS"])

    def test_list_without_all_is_untouched(self):
        """An ordinary variable list must not be searched for or reordered."""

        stub = _StubInfo({})

        self.assertEqual(_expand(stub, ["TS", "SWCF"]), ["TS", "SWCF"])

    def test_nothing_found_is_an_error(self):
        """ "all" with no variables behind it must stop the run, not run empty."""

        with tempfile.TemporaryDirectory() as tmpdir:
            stub = _StubInfo(
                {
                    "cam_case_name": ["case_a"],
                    "cam_hist_loc": [tmpdir],
                    "hist_str": [["cam.h0a"]],
                    "cam_ts_loc": [None],
                    "cam_ts_done": [False],
                }
            )

            with self.assertRaises(RuntimeError):
                _expand(stub, ["all"])


# +++++++++++++++++++++++++++++++++++++++++
# Run unit tests if this script is directly
# called from the command line:
# +++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    unittest.main()
