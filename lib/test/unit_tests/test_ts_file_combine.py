"""
Collection of python unit tests
for combining time series files into one file.

The MDTF file move writes one file per variable, so a variable split into
consecutive files has to be combined.  Whether that is safe is decided by
"adf_file_utils.ts_files_need_combining", which reads the periods out of the
file names.  These tests check that decision against what xarray actually does
with the files, so the two cannot drift apart: a set the decision calls
combinable must open, and a set it rejects must be one that really would fail.

Getting that wrong ended the whole ADF run rather than one variable, because
the exception was raised where nothing catches it.

NOTE: these tests import xarray, so they are skipped in CI, which installs only
PyYAML and pytest.  The decision itself is tested without xarray in
test_adf_file_utils.py.
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

from adf_file_utils import ts_files_need_combining

try:
    import numpy as np
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False


#Every file of a case carries the same reference date, because ncrcat and
#GenTS both take it from the history files.  That matters here: files with
#different reference dates hold numbers that do not overlap even when the
#periods do, which hides the failure this module is about.
_UNITS = "days since 0001-01-01 00:00:00"


def _write_ts(path, first_month, nmonths):
    """Write a time series file covering `nmonths` months, monthly."""
    time = np.arange(first_month, first_month + nmonths) * 30.0
    ds = xr.Dataset({"T": (("time",), np.arange(nmonths, dtype="f4"))},
                    coords={"time": time})
    ds.time.attrs = {"units": _UNITS, "calendar": "noleap"}
    ds.to_netcdf(path)


@unittest.skipUnless(_HAS_XARRAY, "xarray not available")
class TsFileCombineTestRoutine(unittest.TestCase):

    """
    Unit tests pairing the combine decision with xarray's own behavior.
    """

    def test_consecutive_files_combine(self):

        """
        Consecutive files -- what GenTS writes when 'gents_slice_years' is set
        -- are the case the combine exists for: they must be combinable, and
        combining must give one continuous record.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fils = [str(Path(tmpdir) / "case.cam.h0a.T.000101-001012.nc"),
                    str(Path(tmpdir) / "case.cam.h0a.T.001101-002012.nc")]
            _write_ts(fils[0], 0, 120)
            _write_ts(fils[1], 120, 120)

            self.assertTrue(ts_files_need_combining(fils))

            with xr.open_mfdataset(fils, decode_times=False,
                                   combine="by_coords") as ds:
                self.assertEqual(len(ds.time), 240)
                self.assertTrue(bool(np.all(np.diff(ds.time.values) > 0)))
            #End with

    def test_overlapping_files_are_not_combined(self):

        """
        Two sets covering years 1-20 and 10-40, which is what extending a run
        and re-processing the time series leaves behind.

        The decision has to reject these, and the test also shows why: opening
        them together raises, and the caller raised where nothing catches it,
        so the whole run ended.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fils = [str(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc"),
                    str(Path(tmpdir) / "case.cam.h0a.T.001001-004012.nc")]
            _write_ts(fils[0], 0, 240)
            _write_ts(fils[1], 108, 372)

            self.assertFalse(ts_files_need_combining(fils))

            with self.assertRaises(ValueError):
                xr.open_mfdataset(fils, decode_times=False,
                                  combine="by_coords")
            #End with

    def test_single_file_needs_no_combining(self):

        """
        One file is the ordinary case, and is copied rather than rewritten.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fil = str(Path(tmpdir) / "case.cam.h0a.T.000101-002012.nc")
            _write_ts(fil, 0, 240)

            self.assertFalse(ts_files_need_combining([fil]))

#++++++++++++++++++

#Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
#End of file
#############
