"""
Collection of python unit tests
for choosing a case's history stream.

The tape recorder plot builds a list of history streams and then indexes it by
case number, alongside the case names, time series locations and years.  So the
list has to hold exactly one entry per case: an earlier version appended only
when a case had a "cam.h0" or "cam.h0a" stream, which made the list shorter
than the cases whenever one did not, and every case after the missing one was
read off the wrong entry.  A baseline running on pre-made time series had no
stream recorded at all, so the plot raised IndexError and could not be used
against pre-made time series (NCAR/ADF issue #471).

Both the tape recorder plot and the aerosol and gas tables build such a list,
so the choice lives in adf_file_utils, which imports nothing but pathlib and
can therefore be tested in CI.
"""

# +++++++++++++++++++++++
# Import required modules
# +++++++++++++++++++++++

import unittest
import sys
import os
import os.path

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

from adf_file_utils import as_hist_str_list, pick_hist_str

# The streams the tape recorder can use, as that script defines them:
_SUBSTRINGS = {"cam.h0", "cam.h0a"}


class HistStrSelectionTestRoutine(unittest.TestCase):
    """
    Unit tests for choosing one history stream per case.
    """

    def test_as_hist_str_list_accepts_what_the_adf_stores(self):
        """
        The ADF holds one stream as a plain string, several as a list, and an
        empty string when none was configured.  All three have to be accepted:
        iterating a plain string would walk its characters instead.
        """

        self.assertEqual(as_hist_str_list("cam.h0"), ["cam.h0"])
        self.assertEqual(as_hist_str_list(["cam.h0", "cam.h1"]), ["cam.h0", "cam.h1"])
        self.assertEqual(as_hist_str_list(""), [])
        self.assertEqual(as_hist_str_list(None), [])
        self.assertEqual(as_hist_str_list([]), [])

    def test_pick_from_a_list_of_streams(self):
        """
        A case with several streams contributes the one the plot can use.
        """

        self.assertEqual(pick_hist_str(["cam.h1", "cam.h0a"], _SUBSTRINGS), "cam.h0a")
        self.assertEqual(pick_hist_str(["cam.h0"], _SUBSTRINGS), "cam.h0")

    def test_pick_from_a_plain_string(self):
        """
        One configured stream arrives as a string, not a list of one.
        """

        self.assertEqual(pick_hist_str("cam.h0", _SUBSTRINGS), "cam.h0")

    def test_no_usable_stream_still_answers(self):
        """
        The case this exists for: an unset stream, which is what a baseline on
        pre-made time series used to have, and a stream the plot cannot use.

        Both give an empty string rather than nothing at all.  An empty string
        leaves the file search matching whichever stream the files are in,
        which is what the search did before any of this was filtered.
        """

        self.assertEqual(pick_hist_str("", _SUBSTRINGS), "")
        self.assertEqual(pick_hist_str([], _SUBSTRINGS), "")
        self.assertEqual(pick_hist_str("cam.h3", _SUBSTRINGS), "")
        self.assertEqual(pick_hist_str(["cam.h1", "cam.h2"], _SUBSTRINGS), "")

    def test_one_entry_per_case(self):
        """
        The property the plot depends on: as many entries as there are cases,
        in the same order, whatever each case holds.
        """

        cases = [["cam.h0a"], ["cam.h3"], "cam.h0", "", ["cam.h1", "cam.h0a"]]

        picked = [pick_hist_str(case, _SUBSTRINGS) for case in cases]

        self.assertEqual(len(picked), len(cases))
        self.assertEqual(picked, ["cam.h0a", "", "cam.h0", "", "cam.h0a"])


# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
