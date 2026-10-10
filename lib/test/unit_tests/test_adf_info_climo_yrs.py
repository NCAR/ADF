"""
Collection of python unit tests
for "AdfInfo.get_climo_yrs_from_ts".

This is the method that works out the climatology year range when a user runs
ADF against pre-made time series files and gives no years in the config file.
It searches for the first variable in 'diag_var_list' it can find a file for,
trying the configured history stream(s) before the older, looser "any h0
stream" pattern, and looking in the directory itself before recursing into a
nested (GenTS-style) layout.

NOTE: unlike test_adf_file_utils, these tests import adf_info, which pulls in
xarray and (through adf_utils) the rest of the scientific stack.  The ADF unit
test workflow installs only PyYAML and pytest, so they are skipped there and
only run in a full ADF environment.  Moving the search itself into
adf_file_utils would make it testable in CI; that is a bigger change than the
one these tests were written to cover.
"""

#+++++++++++++++++++++++
#Import required modules
#+++++++++++++++++++++++

import unittest
import sys
import os
import os.path
import tempfile
from pathlib import Path

#Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

#Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    from adf_base import AdfError
    from adf_info import AdfInfo
    _HAS_ADF_INFO = True
except ImportError:
    _HAS_ADF_INFO = False


class _StubInfo:
    """
    Minimal stand-in for AdfInfo.

    get_climo_yrs_from_ts reads only 'diag_var_list' and calls 'debug_log', so
    the method can be exercised unbound without building a real AdfInfo (which
    needs a full config file and existing case directories).
    """

    def __init__(self, diag_var_list):
        self.diag_var_list = diag_var_list
        self.messages = []

    def debug_log(self, msg):
        """Collect rather than write, so tests can assert on the log."""
        self.messages.append(msg)


def _write_ts(path, var, start_year, nyears):
    """Write a small time series file holding `var` over `nyears` years."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    time = xr.date_range(f"{start_year:04d}-01-01", periods=nyears * 12,
                         freq="MS", calendar="noleap", use_cftime=True)
    ds = xr.Dataset({var: (("time",), np.ones(len(time), dtype="f4"))},
                    coords={"time": time})
    ds.to_netcdf(path)


def _call(stub, ts_loc, case_name, hist_str=None):
    """Call the method unbound with the stub standing in for self."""
    return AdfInfo.get_climo_yrs_from_ts(stub, ts_loc, case_name,
                                         hist_str=hist_str)


@unittest.skipUnless(_HAS_ADF_INFO, "adf_info dependencies not available")
class AdfInfoClimoYrsTestRoutine(unittest.TestCase):

    """
    Unit tests for the pre-made time series year search.
    """

    def test_flat_layout_configured_stream(self):

        """
        The ordinary case: files sitting directly in 'cam_ts_loc', found using
        the history stream the config file asked for.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))

    def test_nested_layout_found_by_recursive_sweep(self):

        """
        A GenTS-style <component>/proc/tseries/<frequency>/ tree must still be
        found, which is what the second, recursive sweep is for.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "atm" / "proc" / "tseries" / "month_1"
                      / "case.cam.h0a.T.000101-002012.nc", "T", 1, 20)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))

    def test_missing_first_variable_falls_through(self):

        """
        A variable absent from the run is normal; the search moves on to the
        next one in 'diag_var_list' and logs why.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.PRECT.000101-001012.nc",
                      "PRECT", 1, 10)

            stub = _StubInfo(["T", "PRECT"])
            syr, eyr = _call(stub, tmpdir, "case", hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 10))
            self.assertTrue(any("'T' not in dataset" in m
                                for m in stub.messages))

    def test_no_hist_str_uses_loose_h0_pattern(self):

        """
        With no history stream configured, the older 'any h0 stream' pattern
        has to keep working, since that is what pre-existing directories and
        pre-existing configs rely on.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case", hist_str=None)

            self.assertEqual((syr, eyr), (1, 20))

    def test_hist_str_list_tries_each_stream(self):

        """
        'hist_str' arrives as a list when several streams are configured; a
        file belonging to the second one must still be found.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h1a.T.000101-000512.nc",
                      "T", 1, 5)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str=["cam.h0a", "cam.h1a"])

            self.assertEqual((syr, eyr), (1, 5))

    def test_configured_stream_wins_over_loose_pattern(self):

        """
        When both the configured stream and some other h0 stream are present,
        the configured one must win -- the loose pattern is a fallback, not a
        competitor.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)
            _write_ts(Path(tmpdir) / "case.cam.h0b.T.010001-010412.nc",
                      "T", 100, 5)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))

    def test_flat_files_win_over_nested_files(self):

        """
        A file in the directory itself beats one buried in a sub-directory, so
        a stray nested archive cannot shadow the expected location.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)
            _write_ts(Path(tmpdir) / "atm" / "proc" / "tseries" / "month_1"
                      / "case.cam.h0a.T.010001-010412.nc", "T", 100, 5)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))

    def test_flat_sweep_precedes_recursive_sweep_across_variables(self):

        """
        Pins down a deliberate consequence of sweeping flat across every
        variable before recursing at all.

        Here the first variable in 'diag_var_list' exists only nested, while a
        later variable sits flat in the directory.  The flat sweep reaches the
        later variable first, so the years come from that file.  Searching one
        variable at a time -- recursing before moving on -- would instead have
        returned the first variable's nested file.

        Both are legitimate: the method documents that it assumes every
        variable covers the same dates, and picks whichever it finds first.
        The test exists so that the choice is a recorded decision rather than
        an accident, and so a future change to the search order is noticed.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "atm" / "proc" / "tseries" / "month_1"
                      / "case.cam.h0a.T.000101-002012.nc", "T", 1, 20)
            _write_ts(Path(tmpdir) / "case.cam.h0a.PRECT.010001-010412.nc",
                      "PRECT", 100, 5)

            syr, eyr = _call(_StubInfo(["T", "PRECT"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (100, 104))

    def test_configured_stream_beats_flat_file_from_another_stream(self):

        """
        The configured stream outranks the search order.

        A stray flat file from some other h0 stream must not beat the
        configured stream's file just because the configured one happens to
        sit in a sub-directory: 'hist_str' is what the user asked for, and the
        loose "any h0 stream" pattern is only a fallback.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0b.T.010001-010412.nc",
                      "T", 100, 5)
            _write_ts(Path(tmpdir) / "atm" / "proc" / "tseries" / "month_1"
                      / "case.cam.h0a.T.020001-020912.nc", "T", 200, 10)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (200, 209))

    def test_overlapping_sets_report_full_span(self):

        """
        Years 1-20 alongside years 1-40, which is what a re-post-processed run
        leaves behind, must report the whole available span.

        These files cannot be opened together -- the combined time axis would
        have duplicate times -- and this method runs during ADF setup, so
        reading the period from the file names rather than the data is what
        keeps the run alive long enough for the year range in the config file
        to be honored.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-004012.nc",
                      "T", 1, 40)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 40))

    def test_partially_overlapping_sets_report_full_span(self):

        """
        The same, for sets that only partly overlap (years 1-20 alongside
        years 10-40), which is the layout that used to raise
        'Resulting object does not have monotonic global indexes'.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.001001-004012.nc",
                      "T", 10, 31)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 40))

    def test_unreadable_dates_fall_back_to_the_data(self):

        """
        A directory whose names do not carry the period still has to work,
        which is what the fallback that opens the files is for.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.someperiod.nc", "T", 7, 3)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (7, 9))

    def test_nested_layout_does_not_flood_the_debug_log(self):

        """
        A nested archive is a supported layout, not a problem.  Reporting
        every variable in 'diag_var_list' as missing on the way to finding it
        would bury the real messages in a long run.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "atm" / "proc" / "tseries" / "month_1"
                      / "case.cam.h0a.T.000101-002012.nc", "T", 1, 20)

            stub = _StubInfo(["T", "PRECT", "PS", "U", "V"])
            syr, eyr = _call(stub, tmpdir, "case", hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))
            self.assertEqual(stub.messages, [])

    def test_chunked_files_span_all_chunks(self):

        """
        A variable split into consecutive chunks reports the whole range, not
        just the first chunk's.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.000101-001012.nc",
                      "T", 1, 10)
            _write_ts(Path(tmpdir) / "case.cam.h0a.T.001101-002012.nc",
                      "T", 11, 10)

            syr, eyr = _call(_StubInfo(["T"]), tmpdir, "case",
                             hist_str="cam.h0a")

            self.assertEqual((syr, eyr), (1, 20))

    def test_nothing_found_raises(self):

        """
        No files for any variable is a configuration error, and has to say so
        rather than failing later on an unbound variable.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(Path(tmpdir) / "othercase.cam.h0a.T.000101-002012.nc",
                      "T", 1, 20)

            with self.assertRaises(AdfError):
                _call(_StubInfo(["T"]), tmpdir, "case", hist_str="cam.h0a")

    def test_missing_directory_raises(self):

        """
        'cam_ts_done' says the files already exist, so a missing directory
        means the configuration is wrong and should be reported plainly.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(AdfError):
                _call(_StubInfo(["T"]), os.path.join(tmpdir, "nope"),
                      "case", hist_str="cam.h0a")

#++++++++++++++++++

#Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
#End of file
