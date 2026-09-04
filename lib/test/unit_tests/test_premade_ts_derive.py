"""
Collection of python unit tests for deriving variables from pre-made time
series, i.e. the 'cam_ts_done: true' path through AdfDiag.

These exercise AdfDiag.derive_from_premade_ts against a stub object rather
than a configured run, so no config file or history files are needed.  They
import adf_diag, which needs the scientific stack, so they do not run in the
ADF unit test workflow (see adf_file_utils for what does).
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

import numpy as np
import xarray as xr

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

from adf_diag import AdfDiag

_CASE = "mycase"
_STREAM = "cam.h0a"
_SPAN = "000101-000212"


def _write_ts(ts_dir, var, value, stream=_STREAM):
    """Write a minimal ADF-named time series file holding one variable."""
    time = xr.date_range(
        "0001-01-01", periods=24, freq="MS", calendar="noleap", use_cftime=True
    )
    data = np.full((24, 2, 2), value, dtype="float32")
    ds = xr.Dataset(
        {var: (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 180.0]},
    )
    ds[var].attrs = {"units": "W/m2", "long_name": f"{var} for testing"}
    fname = Path(ts_dir) / f"{_CASE}.{stream}.{var}.{_SPAN}.nc"
    ds.to_netcdf(fname)
    return fname


class _StubData:
    """Stands in for AdfData, whose only use here is opening the files."""

    @staticmethod
    def load_dataset(fils):
        """Open the constituent files the way AdfData does."""
        if not fils:
            return None
        return xr.open_mfdataset([str(f) for f in fils], decode_times=True)


class _StubAdf:
    """The parts of AdfDiag that derive_from_premade_ts actually touches."""

    def __init__(self, diag_var_list):
        self.diag_var_list = diag_var_list
        self.data = _StubData()
        self.debug_msgs = []

    def debug_log(self, msg):
        """Record instead of writing a log file."""
        self.debug_msgs.append(msg)

    # Borrow the real implementations under test:
    derive_from_premade_ts = AdfDiag.derive_from_premade_ts
    _premade_constits = AdfDiag._premade_constits


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main pre-made time series derivation testing routine
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class PremadeTsDeriveTestRoutine(unittest.TestCase):
    """
    Unit tests for deriving variables from time series the ADF did not make,
    which is the case a 'cam_ts_done: true' run is in.
    """

    def test_derives_restom(self):
        """
        Check that RESTOM is derived from pre-made FSNT and FLNT.
        """

        res = {"RESTOM": {"derivable_from": ["FSNT", "FLNT"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(tmpdir, "FSNT", 240.0)
            _write_ts(tmpdir, "FLNT", 235.0)
            adf = _StubAdf(["RESTOM"])

            adf.derive_from_premade_ts(_CASE, tmpdir, res, [_STREAM], syr=1, eyr=2)

            out = Path(tmpdir) / f"{_CASE}.{_STREAM}.RESTOM.{_SPAN}.nc"
            self.assertTrue(out.is_file())
            with xr.open_dataset(out) as ds:
                self.assertEqual(float(ds["RESTOM"].mean()), 5.0)

    def test_derives_with_no_stream_configured(self):
        """
        Check that derivation still happens when no history stream is set.

        With pre-made time series there are no history files to name a stream
        for, so 'hist_str' is legitimately empty and the ADF records it as "".
        Iterating that yields nothing, so a stream loop alone would skip the
        derivation without saying a word.
        """

        res = {"RESTOM": {"derivable_from": ["FSNT", "FLNT"]}}
        for hist_strs in ("", [], None):
            with self.subTest(hist_strs=hist_strs):
                with tempfile.TemporaryDirectory() as tmpdir:
                    _write_ts(tmpdir, "FSNT", 240.0)
                    _write_ts(tmpdir, "FLNT", 235.0)
                    adf = _StubAdf(["RESTOM"])

                    adf.derive_from_premade_ts(
                        _CASE, tmpdir, res, hist_strs, syr=1, eyr=2
                    )

                    out = Path(tmpdir) / f"{_CASE}.{_STREAM}.RESTOM.{_SPAN}.nc"
                    self.assertTrue(
                        out.is_file(), f"nothing derived for hist_strs={hist_strs!r}"
                    )

    def test_plain_cam_constituents_preferred_when_cam_chem_incomplete(self):
        """
        Check that a variable carrying both constituent lists uses the plain
        CAM one when the CAM-CHEM constituents are not all present.

        SO4 and SOA declare both.  Preferring 'derivable_from_cam_chem'
        outright would ask an ordinary CAM run for constituents it never
        wrote, so nothing would be derived.
        """

        res = {
            "SO4": {
                "derivable_from": ["so4_a1", "so4_a2"],
                "derivable_from_cam_chem": ["so4_a1", "so4_a2", "so4_a5"],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(tmpdir, "so4_a1", 1.0)
            _write_ts(tmpdir, "so4_a2", 2.0)
            adf = _StubAdf(["SO4"])

            chosen = adf._premade_constits(
                res["SO4"], tmpdir, _CASE, _STREAM, syr=1, eyr=2
            )

            self.assertEqual(chosen, ["so4_a1", "so4_a2"])

    def test_cam_chem_constituents_used_when_all_present(self):
        """
        Check that the CAM-CHEM list wins once all of its constituents are
        there, which is the choice check_derive makes from a history file.
        """

        res = {
            "SO4": {
                "derivable_from": ["so4_a1", "so4_a2"],
                "derivable_from_cam_chem": ["so4_a1", "so4_a2", "so4_a5"],
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for constit in ("so4_a1", "so4_a2", "so4_a5"):
                _write_ts(tmpdir, constit, 1.0)
            adf = _StubAdf(["SO4"])

            chosen = adf._premade_constits(
                res["SO4"], tmpdir, _CASE, _STREAM, syr=1, eyr=2
            )

            self.assertEqual(chosen, ["so4_a1", "so4_a2", "so4_a5"])

    def test_nothing_said_when_derived_file_already_there(self):
        """
        Check that an unwritable directory is not complained about when it
        already holds the derived variable.

        Reading someone else's finished time series is the whole point, so a
        directory that needs nothing written to it must not be reported.
        """

        res = {"RESTOM": {"derivable_from": ["FSNT", "FLNT"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly = Path(tmpdir) / "readonly"
            readonly.mkdir()
            _write_ts(readonly, "FSNT", 240.0)
            _write_ts(readonly, "FLNT", 235.0)
            _write_ts(readonly, "RESTOM", 5.0)
            readonly.chmod(0o555)
            try:
                adf = _StubAdf(["RESTOM"])

                adf.derive_from_premade_ts(
                    _CASE, readonly, res, [_STREAM], syr=1, eyr=2
                )

                self.assertEqual(adf.debug_msgs, [])
            finally:
                readonly.chmod(0o755)

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_unwritable_directory_reported_once(self):
        """
        Check that a directory that cannot gain the derived file says so,
        naming the variable, rather than raising PermissionError.
        """

        res = {"RESTOM": {"derivable_from": ["FSNT", "FLNT"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            readonly = Path(tmpdir) / "readonly"
            readonly.mkdir()
            _write_ts(readonly, "FSNT", 240.0)
            _write_ts(readonly, "FLNT", 235.0)
            readonly.chmod(0o555)
            try:
                adf = _StubAdf(["RESTOM"])

                adf.derive_from_premade_ts(
                    _CASE, readonly, res, [_STREAM], syr=1, eyr=2
                )

                self.assertEqual(len(adf.debug_msgs), 1)
                self.assertIn("RESTOM", adf.debug_msgs[0])
                self.assertIn("permission to write", adf.debug_msgs[0])
            finally:
                readonly.chmod(0o755)

    def test_non_derivable_variables_ignored(self):
        """
        Check that a variable with no 'derivable_from' is left alone.
        """

        res = {"TS": {"colormap": "Reds"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(tmpdir, "TS", 288.0)
            before = sorted(p.name for p in Path(tmpdir).glob("*.nc"))
            adf = _StubAdf(["TS"])

            adf.derive_from_premade_ts(_CASE, tmpdir, res, [_STREAM], syr=1, eyr=2)

            self.assertEqual(sorted(p.name for p in Path(tmpdir).glob("*.nc")), before)


# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
