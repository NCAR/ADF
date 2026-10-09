"""
Collection of python unit tests
for "create_climo_files.find_ts_for_variable".

Climatologies are built per history stream, so the stream configured for a case
decides which time series files are used.  That stream does not have to match
the names of pre-made time series files: the example config carries
``hist_str: cam.h0a`` in both case blocks, while a CESM2-era run's files are
named ``cam.h0``.  Before the stream was recorded for runs on pre-made time
series, those runs searched without one and found the files anyway; once it is
recorded, a mismatched stream would find nothing and the variable would be
skipped, leaving no baseline to compare against.  So a search that comes back
empty for every configured stream falls back to searching without one.

NOTE: these tests import create_climo_files, which imports xarray, so they are
skipped in CI, which installs only PyYAML and pytest.  They run in a full ADF
environment.
"""

import os
import os.path
import sys
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)
_ADF_AVERAGING_DIR = os.path.join(
    _CURRDIR, os.pardir, os.pardir, os.pardir, "scripts", "averaging"
)

# Add ADF "lib" and averaging script directories to python path:
sys.path.append(_ADF_LIB_DIR)
sys.path.append(_ADF_AVERAGING_DIR)

try:
    from create_climo_files import find_ts_for_variable

    _HAS_CREATE_CLIMO = True
except ImportError:
    _HAS_CREATE_CLIMO = False


class _StubData:
    """Stand-in for AdfData: answers file searches from a stream -> files map."""

    def __init__(self, available):
        self.available = available

    def get_ref_timeseries_file(self, var, hist_str=None):
        """Files for one stream, or for any stream when hist_str is None."""
        if hist_str is None:
            return [fil for fils in self.available.values() for fil in fils]
        return self.available.get(hist_str, [])

    def get_timeseries_file(self, case, var, hist_str=None):
        """Test cases are searched the same way, with the case name added."""
        return self.get_ref_timeseries_file(var, hist_str=hist_str)


class _StubAdf:
    """Minimal stand-in for the ADF object: only its data member is used."""

    def __init__(self, available):
        self.data = _StubData(available)


@unittest.skipUnless(_HAS_CREATE_CLIMO, "create_climo_files dependencies not available")
class ClimoStreamFallbackTestRoutine(unittest.TestCase):
    """
    Unit tests for finding a variable's time series across history streams.
    """

    # A case whose files are named for one stream:
    FILES = {"cam.h0": ["case.cam.h0.Q.200001-200412.nc"]}

    def test_configured_stream_is_used(self):
        """The ordinary case: the stream matches, and is used and reported."""

        found = find_ts_for_variable(
            _StubAdf(self.FILES), "case", "Q", ["cam.h0"], True
        )

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0][0], "cam.h0")

    def test_mismatched_stream_falls_back(self):
        """
        The case this exists for: a stream is configured but does not match the
        file names, which is what the example config gives a CESM2-era baseline
        on pre-made time series.

        The files are found, and reported against no stream, so the climatology
        keeps the stream-agnostic name rather than claiming a stream its data
        did not come from.
        """

        found = find_ts_for_variable(
            _StubAdf(self.FILES), "case", "Q", ["cam.h0a"], True
        )

        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0][0])
        self.assertEqual(len(found[0][1]), 1)

    def test_matching_stream_does_not_also_fall_back(self):
        """
        The fallback is a last resort.  If it ran anyway, a variable would be
        written twice: once under its stream and once without.
        """

        found = find_ts_for_variable(
            _StubAdf(self.FILES), "case", "Q", ["cam.h0a", "cam.h0"], True
        )

        self.assertEqual([hist_str for hist_str, _ in found], ["cam.h0"])

    def test_several_streams_hold_the_variable(self):
        """A variable can be in more than one stream, and each is kept."""

        files = {
            "cam.h0": ["case.cam.h0.Q.200001-200412.nc"],
            "cam.h1": ["case.cam.h1.Q.200001-200412.nc"],
        }

        found = find_ts_for_variable(
            _StubAdf(files), "case", "Q", ["cam.h0", "cam.h1"], True
        )

        self.assertEqual([hist_str for hist_str, _ in found], ["cam.h0", "cam.h1"])

    def test_no_stream_known_searches_without_one(self):
        """
        A case with no stream recorded searches without one, and must not then
        search a second time and report the files twice.
        """

        found = find_ts_for_variable(_StubAdf(self.FILES), "case", "Q", [None], True)

        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0][0])

    def test_variable_genuinely_absent(self):
        """
        A variable that is not in the run at all is normal, and has to come
        back empty so the caller can warn and move on.
        """

        found = find_ts_for_variable(_StubAdf({}), "case", "Q", ["cam.h0a"], True)

        self.assertEqual(found, [])

    def test_test_case_search_matches_baseline_search(self):
        """Test cases take the same path, with the case name passed along."""

        found = find_ts_for_variable(
            _StubAdf(self.FILES), "case", "Q", ["cam.h0a"], False
        )

        self.assertEqual(len(found), 1)
        self.assertIsNone(found[0][0])


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
