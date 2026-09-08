"""
Collection of python unit tests
for the "adf_file_utils" time series file discovery helpers.
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

# adf_file_utils imports nothing but pathlib, so these run in CI, where only
# PyYAML and pytest are installed:
from adf_file_utils import (
    describe_dir_problem,
    find_ts_files,
    select_ts_files,
    ts_files_overlap,
    ts_files_need_combining,
    ts_file_span,
)


class AdfFileUtilsTestRoutine(unittest.TestCase):
    """
    Unit tests for the time series file search helper, which has to cope with
    both ADF's flat time series directory and GenTS's nested
    <component>/proc/tseries/<frequency>/ layout.
    """

    def test_find_ts_files_flat(self):
        """
        Check that a file sitting directly in the search directory is found.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = "case.cam.h0a.T.000101-001112.nc"
            open(os.path.join(tmpdir, fname), "w").close()

            found = find_ts_files(tmpdir, "case.cam.h0a.T.*nc")

            self.assertEqual([f.name for f in found], [fname])

    def test_find_ts_files_nested(self):
        """
        Check that a file buried in a GenTS-style sub-directory is found when
        nothing matches at the top level.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "atm", "proc", "tseries", "month_1")
            os.makedirs(subdir)
            fname = "case.cam.h0a.T.000101-001112.nc"
            open(os.path.join(subdir, fname), "w").close()

            found = find_ts_files(tmpdir, "case.cam.h0a.T.*nc")

            self.assertEqual([f.name for f in found], [fname])

    def test_find_ts_files_prefers_flat(self):
        """
        Check that the flat match wins, so that a nested directory of files
        from some other run can't shadow the expected location.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "atm", "proc", "tseries", "month_1")
            os.makedirs(subdir)
            fname = "case.cam.h0a.T.000101-001112.nc"
            open(os.path.join(tmpdir, fname), "w").close()
            open(os.path.join(subdir, fname), "w").close()

            found = find_ts_files(tmpdir, "case.cam.h0a.T.*nc")

            self.assertEqual(len(found), 1)
            self.assertEqual(os.path.dirname(str(found[0])), tmpdir)

    def test_find_ts_files_no_match(self):
        """
        Check that a search with no matches returns an empty list, rather
        than raising, so callers can warn and move on.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "case.cam.h0a.T.000101-001112.nc"), "w").close()

            self.assertEqual(find_ts_files(tmpdir, "case.cam.h0a.PRECT.*nc"), [])

    def test_find_ts_files_no_recurse(self):
        """
        With recursive=False a nested file is not found, so a caller sweeping
        many patterns can avoid walking the whole tree for each one.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "atm", "proc", "tseries", "month_1")
            os.makedirs(subdir)
            fname = "case.cam.h0a.T.000101-001112.nc"
            open(os.path.join(subdir, fname), "w").close()

            self.assertEqual(
                find_ts_files(tmpdir, "case.cam.h0a.T.*nc", recursive=False), []
            )
            # ...but the default still finds it:
            self.assertEqual(
                [f.name for f in find_ts_files(tmpdir, "case.cam.h0a.T.*nc")], [fname]
            )

    def test_find_ts_files_no_recurse_still_finds_flat(self):
        """
        recursive=False must not change the flat case, which is what ADF's own
        time series step and a flat GenTS run both produce.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = "case.cam.h0a.T.000101-001112.nc"
            open(os.path.join(tmpdir, fname), "w").close()

            found = find_ts_files(tmpdir, "case.cam.h0a.T.*nc", recursive=False)

            self.assertEqual([f.name for f in found], [fname])

    def test_ts_files_overlap_consecutive(self):
        """
        Consecutive chunks (what GenTS writes when 'slice_years' is set, and
        how CMIP-style archives are laid out) can safely be opened together.
        """

        fils = ["case.cam.h0a.T.001001-001912.nc", "case.cam.h0a.T.002001-002912.nc"]

        self.assertFalse(ts_files_overlap(fils))

    def test_ts_files_overlap_overlapping(self):
        """
        Two sets covering overlapping periods (years 1-20 alongside years
        1-40, from a run that was extended and re-processed) must be refused.
        """

        fils = ["case.cam.h0a.T.000101-002012.nc", "case.cam.h0a.T.000101-004012.nc"]

        self.assertTrue(ts_files_overlap(fils))

    def test_ts_files_overlap_single_file(self):
        """
        A single file never overlaps anything.
        """

        self.assertFalse(ts_files_overlap(["case.cam.h0a.T.001001-001112.nc"]))

    def test_ts_files_overlap_unparsable(self):
        """
        Names the date range cannot be read from are reported as overlapping,
        so that callers do not blindly combine them.
        """

        fils = ["case.cam.h0a.T.somethingelse.nc", "case.cam.h0a.T.001001-001912.nc"]

        self.assertTrue(ts_files_overlap(fils))

    def test_ts_files_overlap_mixed_date_widths(self):
        """
        Mixed date widths do not sort chronologically as strings, so they are
        refused rather than compared incorrectly.
        """

        fils = ["case.cam.h0a.T.0010-0019.nc", "case.cam.h0a.T.002001-002912.nc"]

        self.assertTrue(ts_files_overlap(fils))

    def test_ts_files_overlap_accepts_paths(self):
        """
        Callers pass Path objects from find_ts_files, not just strings.
        """

        fils = [
            Path("/some/dir/case.cam.h0a.T.001001-001912.nc"),
            Path("/some/dir/case.cam.h0a.T.002001-002912.nc"),
        ]

        self.assertFalse(ts_files_overlap(fils))

    def test_need_combining(self):
        """
        The question a caller wanting one file per variable has to ask.

        Used by the MDTF file move, which writes one file per variable: one
        file needs no combining, consecutive files do, and overlapping files
        cannot be combined at all -- attempting it raises and would end the
        run.
        """

        one = ["case.cam.h0a.T.000101-002012.nc"]
        chunks = ["case.cam.h0a.T.000101-001012.nc", "case.cam.h0a.T.001101-002012.nc"]
        overlapping = [
            "case.cam.h0a.T.000101-002012.nc",
            "case.cam.h0a.T.001001-004012.nc",
        ]
        unreadable = ["case.cam.h0a.T.first.nc", "case.cam.h0a.T.second.nc"]

        self.assertFalse(ts_files_need_combining([]))
        self.assertFalse(ts_files_need_combining(one))
        self.assertTrue(ts_files_need_combining(chunks))
        self.assertFalse(ts_files_need_combining(overlapping))
        self.assertFalse(ts_files_need_combining(unreadable))

    def test_need_combining_agrees_with_selection(self):
        """
        The two functions have to agree, because the caller applies them in
        turn: whatever selection could not resolve must not then be combined.
        """

        overlapping = [
            "case.cam.h0a.T.000101-002012.nc",
            "case.cam.h0a.T.001001-004012.nc",
        ]

        # Years 1-40 cannot be covered without holding both files, so selection
        # passes them through -- and they must not be combined:
        chosen = select_ts_files(overlapping, 1, 40)
        self.assertEqual(chosen, overlapping)
        self.assertFalse(ts_files_need_combining(chosen))

    def test_ts_file_span_single_file(self):
        """
        A single file spans its own dates, so a derived variable built from
        one constituent file keeps exactly the name ADF has always written.
        """

        span = ts_file_span(["case.cam.h0a.FSNT.000101-002012.nc"])

        self.assertEqual(span, ("000101", "002012"))

    def test_ts_file_span_chunked(self):
        """
        Consecutive chunks span from the first start to the last end, which is
        what the derived file's name has to advertise.
        """

        fils = [
            "case.cam.h0a.FSNT.000101-001012.nc",
            "case.cam.h0a.FSNT.001101-002012.nc",
        ]

        self.assertEqual(ts_file_span(fils), ("000101", "002012"))

    def test_ts_file_span_unordered(self):
        """
        The span must not depend on the order the files arrive in.
        """

        fils = [
            "case.cam.h0a.FSNT.001101-002012.nc",
            "case.cam.h0a.FSNT.000101-001012.nc",
        ]

        self.assertEqual(ts_file_span(fils), ("000101", "002012"))

    def test_ts_file_span_unparsable(self):
        """
        Names the dates cannot be read from give None, so the caller falls back
        to leaving the name alone rather than inventing a span.
        """

        self.assertIsNone(ts_file_span(["case.cam.h0a.FSNT.somethingelse.nc"]))
        self.assertIsNone(ts_file_span([]))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # select_ts_files: choosing between sets that cover the same years
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def test_select_nested_sets_full_range(self):
        """
        Years 1-20 alongside years 1-40, which is what a re-post-processed run
        leaves behind, must resolve to the one set covering the years asked
        for.  These cannot be opened together, so choosing is the whole point.
        """

        short = "case.cam.h0a.T.000101-002012.nc"
        long = "case.cam.h0a.T.000101-004012.nc"

        self.assertEqual(select_ts_files([short, long], 1, 40), [long])
        self.assertFalse(ts_files_overlap(select_ts_files([short, long], 1, 40)))

    def test_select_nested_sets_short_range(self):
        """
        Asking for only the first 20 years of the same directory must give a
        single self-consistent set, and the smaller of the two: reading the
        40-year file to average 20 years of it is twice the work for the same
        answer.
        """

        short = "case.cam.h0a.T.000101-002012.nc"
        long = "case.cam.h0a.T.000101-004012.nc"

        self.assertEqual(select_ts_files([short, long], 1, 20), [short])

    def test_select_sub_window_of_one_file(self):
        """
        A range sitting inside a file is the ordinary case for ADF (climo over
        part of a time series), so it must not be treated as "not covered".
        """

        short = "case.cam.h0a.T.000101-002012.nc"
        long = "case.cam.h0a.T.000101-004012.nc"

        self.assertEqual(select_ts_files([short, long], 5, 15), [short])

    def test_select_leaves_consecutive_chunks_alone(self):
        """
        A variable split into consecutive chunks -- what GenTS writes with
        'slice_years' -- opens as it stands, so there is nothing to choose and
        every file must come back.

        Narrowing these would break the plots that show a whole record: they
        ask for all of a case's files, not only the climatology years.
        """

        fils = [
            "case.cam.h0a.T.000101-001012.nc",
            "case.cam.h0a.T.001101-002012.nc",
            "case.cam.h0a.T.002101-003012.nc",
        ]

        self.assertEqual(select_ts_files(fils, 1, 20), fils)
        self.assertEqual(select_ts_files(fils, 15, 25), fils)

    def test_select_keeps_chunks_needed_against_a_duplicate(self):
        """
        The same chunks, but with a whole-period file of the same run beside
        them, which is a set that cannot be opened.  Now a choice is needed,
        and it has to be one self-consistent set.
        """

        chunks = ["case.cam.h0a.T.000101-001012.nc", "case.cam.h0a.T.001101-002012.nc"]
        whole = "case.cam.h0a.T.000101-002012.nc"

        chosen = select_ts_files(chunks + [whole], 1, 20)

        self.assertFalse(ts_files_overlap(chosen))
        self.assertTrue(chosen == [whole] or chosen == chunks)

    def test_select_prefers_whole_set_over_chunks(self):
        """
        A chunked set and a single whole-period set in the same directory
        overlap, so exactly one of them must be chosen.
        """

        chunks = ["case.cam.h0a.T.000101-001012.nc", "case.cam.h0a.T.001101-002012.nc"]
        whole = "case.cam.h0a.T.000101-002012.nc"

        chosen = select_ts_files(chunks + [whole], 1, 20)

        self.assertEqual(chosen, [whole])

    def test_select_drops_files_outside_range(self):
        """
        When a choice has to be made, a file holding none of the requested
        years is of no use and must not be handed on.
        """

        short = "case.cam.h0a.T.000101-002012.nc"
        long = "case.cam.h0a.T.000101-004012.nc"

        self.assertEqual(select_ts_files([short, long], 21, 40), [long])

    def test_select_leaves_sub_year_files_alone(self):
        """
        Only years are compared, so halves of a year must never be weighed
        against one another: dropping one would quietly lose half the data.
        """

        halves = [
            "case.cam.h1a.PRECT.00010101-00010630.nc",
            "case.cam.h1a.PRECT.00010701-00011231.nc",
        ]

        # These combine cleanly, so they are left alone on that count:
        self.assertEqual(select_ts_files(halves, 1, 1), halves)

        # And still left alone when a whole-year file of the same run makes the
        # set one that cannot be opened:
        whole = "case.cam.h1a.PRECT.00010101-00011231.nc"
        self.assertEqual(select_ts_files(halves + [whole], 1, 1), halves + [whole])

    def test_select_resolves_partly_overlapping_sets(self):
        """
        Years 1-20 beside years 10-40 is the layout that raised "Resulting
        object does not have monotonic global indexes".  A file that runs on
        past the requested years is dropped safely, because the file kept
        holds all of the years asked for.
        """

        early = "case.cam.h0a.T.000101-002012.nc"
        late = "case.cam.h0a.T.001001-004012.nc"

        self.assertEqual(select_ts_files([early, late], 1, 20), [early])
        self.assertEqual(select_ts_files([early, late], 10, 40), [late])

    def test_select_leaves_sub_year_boundaries_alone(self):
        """
        Two sets whose boundary falls inside a year: keeping only the first
        would drop the second half of year 5 from a five-year request, which
        comparing years alone cannot see.  So the set comes back whole.
        """

        fils = [
            "case.cam.h0a.T.00010101-00050630.nc",
            "case.cam.h0a.T.00050701-00101231.nc",
        ]

        self.assertEqual(select_ts_files(fils, 1, 5), fils)

    def test_select_refuses_a_cover_it_cannot_open(self):
        """
        A 25-year set beside 10-year chunks of the same run: the greedy walk
        would hold the 25-year file and the last chunk, which overlap.  Handing
        that back is no better than not choosing, so the set comes back whole.
        """

        fils = [
            "case.cam.h0a.T.000101-002512.nc",
            "case.cam.h0a.T.000101-001012.nc",
            "case.cam.h0a.T.001101-002012.nc",
            "case.cam.h0a.T.002101-003012.nc",
        ]

        self.assertEqual(select_ts_files(fils, 1, 30), fils)

    def test_select_passes_through_when_undecidable(self):
        """
        With no years given, unreadable names, or a gap in the requested
        range, there is nothing to choose, so the caller must be left with
        exactly the behavior it had before.
        """

        # Two sets of the same run with a gap after them, so a choice is
        # wanted but the requested range cannot be covered:
        fils = ["case.cam.h0a.T.000101-001012.nc", "case.cam.h0a.T.000101-000512.nc"]

        # No years given:
        self.assertEqual(select_ts_files(fils, None, None), fils)
        self.assertEqual(select_ts_files(fils, "", ""), fils)
        # Years past what the files hold:
        self.assertEqual(select_ts_files(fils, 1, 30), fils)
        # Unreadable names:
        unreadable = ["case.cam.h0a.T.first.nc", "case.cam.h0a.T.second.nc"]
        self.assertEqual(select_ts_files(unreadable, 1, 20), unreadable)
        # Nothing to choose between:
        self.assertEqual(select_ts_files(fils[:1], 1, 10), fils[:1])
        # Files that open together as they stand:
        clean = ["case.cam.h0a.T.000101-001012.nc", "case.cam.h0a.T.001101-002012.nc"]
        self.assertEqual(select_ts_files(clean, 1, 10), clean)
        # A backwards range, which would otherwise cover nothing at all:
        self.assertEqual(select_ts_files(fils, 20, 1), fils)

    def test_select_accepts_paths_and_year_strings(self):
        """
        Callers pass Path objects, and years arrive from the config file as
        either integers or strings.
        """

        short = Path("/ts/case.cam.h0a.T.000101-002012.nc")
        long = Path("/ts/case.cam.h0a.T.000101-004012.nc")

        self.assertEqual(select_ts_files([short, long], "1", "40"), [long])

    def test_select_annual_dates(self):
        """
        Time series dates can be YYYY rather than YYYYMM.
        """

        short = "case.cam.h0a.T.0001-0020.nc"
        long = "case.cam.h0a.T.0001-0040.nc"

        self.assertEqual(select_ts_files([short, long], 1, 40), [long])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # describe_dir_problem: reading another user's output makes an
    # unreadable directory a likely failure rather than a rare one
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def test_dir_problem_usable(self):
        """
        Check that a readable directory reports no problem.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(describe_dir_problem(tmpdir))
            self.assertIsNone(describe_dir_problem(tmpdir, need_write=True))

    def test_dir_problem_missing(self):
        """
        Check that a path that is not there is reported as such.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "not_there")

            problem = describe_dir_problem(missing)

            self.assertIsNotNone(problem)
            self.assertIn("does not exist", problem)

    def test_dir_problem_is_a_file(self):
        """
        Check that a file given where a directory is wanted is reported.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "case.cam.h0a.T.000101-001112.nc")
            open(fname, "w").close()

            self.assertIsNotNone(describe_dir_problem(fname))

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_dir_problem_unreadable(self):
        """
        Check that an unreadable directory is reported as unreadable.

        This is the case that motivated the helper: such a directory reports
        ``is_dir() == True`` and then globs to nothing, so a caller that only
        checks existence tells the user their history files are missing.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            unreadable = Path(tmpdir) / "unreadable"
            unreadable.mkdir()
            (unreadable / "case.cam.h0a.T.000101-001112.nc").touch()
            unreadable.chmod(0o000)
            try:
                # The misleading behavior this exists to catch:
                self.assertTrue(unreadable.is_dir())
                self.assertEqual(find_ts_files(unreadable, "*.nc"), [])

                problem = describe_dir_problem(unreadable)

                self.assertIsNotNone(problem)
                self.assertIn("permission to read", problem)
            finally:
                # Restore, or the temporary directory cannot be cleaned up:
                unreadable.chmod(0o755)

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_dir_problem_unwritable(self):
        """
        Check that a readable but unwritable directory is reported only when
        the caller says it needs to write, which is what lets ADF read someone
        else's time series while refusing to write into them.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            readonly = Path(tmpdir) / "readonly"
            readonly.mkdir()
            readonly.chmod(0o555)
            try:
                self.assertIsNone(describe_dir_problem(readonly))

                problem = describe_dir_problem(readonly, need_write=True)

                self.assertIsNotNone(problem)
                self.assertIn("permission to write", problem)
            finally:
                readonly.chmod(0o755)


# ++++++++++++++++++

# ++++++++++++++++++

# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
