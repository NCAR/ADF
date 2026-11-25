"""
Collection of python unit tests
for the 'calc_cam_ts' setting.

'calc_cam_ts' says whether the ADF makes a case's time series from its history
files, mirroring 'calc_cam_climo'.  It replaces 'cam_ts_done', which meant the
opposite and must keep working for existing config files.

NOTE: these tests import adf_info, which pulls in xarray and (through
adf_utils) the rest of the scientific stack.  The ADF unit test workflow
installs only PyYAML and pytest, so they are skipped there and only run in a
full ADF environment.
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
    from adf_info import AdfInfo, calc_ts_setting, _climo_loc_years
    from adf_diag import AdfDiag

    _HAS_ADF_INFO = True
except ImportError:
    _HAS_ADF_INFO = False


class _StubInfo:
    """
    Minimal stand-in for AdfInfo.

    The methods under test read the test case config entries and report
    problems, so they can be exercised unbound without building a real
    AdfInfo (which needs a full config file and existing case directories).
    """

    def __init__(self, cam_info):
        self.cam_info = cam_info

    def get_cam_info(self, var_str, required=False):
        """Return the stubbed 'diag_cam_climo' entry."""
        return self.cam_info.get(var_str)

    def debug_log(self, msg):
        """Ignore debug messages."""

    def end_diag_fail(self, msg):
        """Raise rather than exit, so a failure can be asserted on."""
        raise RuntimeError(msg)


def _premade(case_name, inputs, syear=None, eyear=None):
    """Call the (name-mangled) private pre-made-case method unbound."""
    # pylint: disable=protected-access
    return AdfInfo._AdfInfo__premade_case_info(
        _StubInfo({}), case_name, inputs, syear, eyear
    )


def _to_make(ts_dir, hist_names, years, overwrite=False, var="TS"):
    """Call AdfDiag._ts_files_to_make unbound for case 'c', stream cam.h0a."""
    # pylint: disable=protected-access
    hist_files = sorted(Path(n) for n in hist_names)
    names = {"case": "c", "hist_str": "cam.h0a", "var": var, "ts_dir": ts_dir}
    result = AdfDiag._ts_files_to_make(
        _StubInfo({}), names, hist_files, years, overwrite
    )
    return [(len(files), Path(out).name) for files, out in result]


def _hist(years):
    """Monthly history file names for the given years."""
    return [f"c.cam.h0a.{y:04d}-{m:02d}.nc" for y in years for m in range(1, 13)]


def _resolve(calc_ts, ts_done, case_names):
    """Call the (name-mangled) private resolver unbound with a stub as self."""
    # pylint: disable=protected-access
    return AdfInfo._AdfInfo__resolve_calc_ts(
        _StubInfo({}), calc_ts, ts_done, case_names
    )


@unittest.skipUnless(_HAS_ADF_INFO, "adf_info dependencies not available")
class CalcCamTsTestRoutine(unittest.TestCase):
    """
    Unit tests for 'calc_cam_ts' and its older name 'cam_ts_done'.
    """

    def test_default_makes_time_series(self):
        """With neither entry given the ADF makes the time series, as before."""
        self.assertTrue(calc_ts_setting(None, None))

    def test_calc_cam_ts_is_used_as_given(self):
        """'calc_cam_ts' alone is the answer."""
        self.assertTrue(calc_ts_setting(True, None))
        self.assertFalse(calc_ts_setting(False, None))

    def test_cam_ts_done_is_inverted(self):
        """'cam_ts_done' alone still works, with its opposite meaning."""
        self.assertFalse(calc_ts_setting(None, True))
        self.assertTrue(calc_ts_setting(None, False))

    def test_agreeing_entries(self):
        """Both entries given, and saying the same thing, is fine."""
        self.assertFalse(calc_ts_setting(False, True))
        self.assertTrue(calc_ts_setting(True, False))

    def test_contradicting_entries(self):
        """Both entries given but contradicting each other is reported as None."""
        self.assertIsNone(calc_ts_setting(True, True))
        self.assertIsNone(calc_ts_setting(False, False))

    def test_resolver_fills_missing_entries(self):
        """Missing lists mean "not given" for every case."""
        self.assertEqual(_resolve(None, None, ["a", "b"]), [True, True])

    def test_resolver_mixes_cases(self):
        """Each case is resolved on its own."""
        self.assertEqual(
            _resolve([False, None], [None, True], ["a", "b"]), [False, False]
        )

    def test_resolver_fails_on_contradiction(self):
        """A contradictory pair stops the run and names both entries."""
        with self.assertRaises(RuntimeError) as err:
            _resolve([True], [True], ["a"])
        # End with
        self.assertIn("calc_cam_ts", str(err.exception))
        self.assertIn("cam_ts_done", str(err.exception))

    def test_var_list_all_reads_premade_time_series(self):
        """
        "diag_var_list: all" searches 'cam_ts_loc' when 'calc_cam_ts' is false,
        exactly as it did for 'cam_ts_done: true'.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "case_a.cam.h0a.TS.000101-002012.nc").touch()
            for entries in ({"calc_cam_ts": [False]}, {"cam_ts_done": [True]}):
                stub = _StubInfo(
                    {
                        "cam_case_name": ["case_a"],
                        "cam_hist_loc": [None],
                        "hist_str": [["cam.h0a"]],
                        "cam_ts_loc": [tmpdir],
                        **entries,
                    }
                )
                self.assertEqual(AdfInfo._expand_var_list_all(stub, ["all"]), ["TS"])
            # End for


@unittest.skipUnless(_HAS_ADF_INFO, "adf_info dependencies not available")
class PremadeInputsTestRoutine(unittest.TestCase):
    """
    Unit tests for cases that start from pre-made files.
    """

    def test_climo_folder_years(self):
        """The innermost 'yrs_<start>_<end>' folder gives the years."""
        self.assertEqual(_climo_loc_years("/a/climo/case/yrs_1_5/"), (1, 5))
        self.assertEqual(_climo_loc_years("/a/yrs_1_100/b/yrs_10_20"), (10, 20))
        self.assertIsNone(_climo_loc_years("/a/climo/case/"))
        self.assertIsNone(_climo_loc_years(None))

    def test_climo_only_years_from_folder(self):
        """No time series and no years given: read them from the climo folder."""
        inputs = {
            "ts_loc": None,
            "hist_str": ["cam.h0a"],
            "climo_loc": "/nowhere/yrs_3_7",
            "calc_climo": False,
            "section": "diag_cam_climo",
        }
        self.assertEqual(_premade("c", inputs), (3, 7, False))

    def test_climo_only_configured_years_win(self):
        """Configured years are used before the folder name."""
        inputs = {
            "ts_loc": None,
            "hist_str": ["cam.h0a"],
            "climo_loc": "/nowhere/yrs_3_7",
            "calc_climo": False,
            "section": "diag_cam_climo",
        }
        self.assertEqual(_premade("c", inputs, 4, 6), (4, 6, False))

    def test_climo_only_without_years_fails(self):
        """No time series, no years, no 'yrs_' folder: say what to add."""
        inputs = {
            "ts_loc": None,
            "hist_str": ["cam.h0a"],
            "climo_loc": "/nowhere/climo",
            "calc_climo": False,
            "section": "diag_cam_climo",
        }
        with self.assertRaises(RuntimeError) as err:
            _premade("c", inputs)
        # End with
        self.assertIn("start_year", str(err.exception))

    def test_no_time_series_but_climo_wanted_fails(self):
        """Climatologies cannot be made without time series."""
        inputs = {
            "ts_loc": None,
            "hist_str": ["cam.h0a"],
            "climo_loc": "/nowhere/yrs_1_5",
            "calc_climo": True,
            "section": "diag_cam_climo",
        }
        with self.assertRaises(RuntimeError) as err:
            _premade("c", inputs, 1, 5)
        # End with
        self.assertIn("calc_cam_climo", str(err.exception))

    def test_no_existing_files_makes_everything(self):
        """A variable with no time series is made from all the history files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                _to_make(tmpdir, _hist(range(1, 6)), (1, 5)),
                [(60, "c.cam.h0a.TS.000101-000512.nc")],
            )

    def test_covered_variable_is_skipped(self):
        """Existing files covering every year mean nothing to make."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "c.cam.h0a.TS.000101-001012.nc").touch()
            self.assertEqual(_to_make(tmpdir, _hist(range(1, 6)), (1, 5)), [])

    def test_extended_run_makes_only_new_years(self):
        """Years 1-5 exist and 1-10 are wanted: make 6-10, in a file of its own."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "c.cam.h0a.TS.000101-000512.nc").touch()
            self.assertEqual(
                _to_make(tmpdir, _hist(range(1, 11)), (1, 10)),
                [(60, "c.cam.h0a.TS.000601-001012.nc")],
            )

    def test_gap_in_history_is_not_spanned(self):
        """Missing years with a hole in the history files give two files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(
                _to_make(tmpdir, _hist([1, 2, 5]), (1, 5)),
                [
                    (24, "c.cam.h0a.TS.000101-000212.nc"),
                    (12, "c.cam.h0a.TS.000501-000512.nc"),
                ],
            )

    def test_overwrite_remakes_everything(self):
        """'cam_overwrite_ts' remakes the whole range, as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "c.cam.h0a.TS.000101-000512.nc").touch()
            self.assertEqual(
                _to_make(tmpdir, _hist(range(1, 6)), (1, 5), overwrite=True),
                [(60, "c.cam.h0a.TS.000101-000512.nc")],
            )


# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()
