"""The baseline pass has to cope with a variable the ADF added to its own list.

AdfData reads the variable list once, when it is built, so anything added later
-- the pressure field the regridder prefers -- is missing from its reference
bookkeeping.  load_reference_climo_ds indexes that bookkeeping as soon as it
finds a climo file, and finding one is a filename glob, so a model-vs-baseline
run whose baseline also writes PMID died with KeyError: 'PMID' before this was
guarded.  Model-vs-obs runs never reach the code at all, which is why it went
unnoticed.
"""

import sys
from pathlib import Path

import pytest

# Skip rather than fail collection when the science stack is absent, as in a
# local run outside the ADF environment.  CI runs these in the full ADF
# environment (.github/workflows/ADF_unit_tests.yaml), where a skip is a failure.
# regrid_and_vert_interp imports xesmf, and adf_utils imports geocat.comp.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")
pytest.importorskip("xesmf")

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))

# pylint: disable-next=import-error
import regrid_and_vert_interp as rvi  # noqa: E402

# These tests exercise the private baseline-writing helper directly:
# pylint: disable=protected-access


class FakeData:
    """The reference bookkeeping AdfData builds once, at start-up."""

    def __init__(self, ref_vars):
        self.ref_case_label = "baseline"
        self.ref_var_nam = {v: v for v in ref_vars}
        self.ref_labels = {v: "baseline" for v in ref_vars}

    # pylint: disable-next=unused-argument
    def load_reference_climo_ds(self, case, variablename):
        """Return a stand-in dataset, indexing ref_var_nam like the real one."""
        # The real one globs for a file first and only then indexes
        # ref_var_nam, so a missing entry raises rather than returning None.
        return {"dummy": self.ref_var_nam[variablename]}


class FakeAdf:
    """Just enough AdfDiag for the baseline loop's variable filter."""

    def __init__(self, ref_vars, pressure_field_names=None):
        self.data = FakeData(ref_vars)
        self.user = "tester"
        self.climo_yrs = {"syear_baseline": 1, "eyear_baseline": 5}
        self._names = pressure_field_names
        self.logged = []

    # pylint: disable-next=unused-argument
    def get_basic_info(self, var_str, required=False):
        """Return the configured pressure field names, if asked for them."""
        return self._names if var_str == "pressure_field_names" else None

    def debug_log(self, msg):
        """Record a debug message."""
        self.logged.append(msg)


def test_the_pressure_field_is_skipped_for_the_baseline(tmp_path):
    """PMID is not a reference variable, and asking for it used to crash."""
    adf = FakeAdf(["T", "Q"])
    rvi._write_reference_files(adf, ["PMID"], {}, tmp_path, True)
    assert adf.logged, "the skip should say so in the debug log"
    assert "PMID" in adf.logged[0]
    assert not list(tmp_path.glob("*.nc"))


def test_a_configured_pressure_field_name_is_skipped_too(tmp_path):
    """A pressure field named via pressure_field_names is skipped as well."""
    adf = FakeAdf(["T"], pressure_field_names={"lev": "pfull"})
    rvi._write_reference_files(adf, ["pfull"], {}, tmp_path, True)
    assert adf.logged and "pfull" in adf.logged[0]


def test_an_ordinary_missing_variable_is_not_swallowed(tmp_path):
    """Only the pressure field gets the quiet path.

    A normal variable missing from the reference bookkeeping is a real problem,
    and hiding it here would hide it everywhere.
    """
    adf = FakeAdf(["T"])
    with pytest.raises(KeyError):
        rvi._write_reference_files(adf, ["RESTOM"], {}, tmp_path, True)
    assert not adf.logged


def test_a_user_listed_pressure_field_is_written_as_usual(tmp_path):
    """A user who asks for PMID has it in the reference bookkeeping.

    The guard only covers the case where the ADF added it after the fact.
    """
    adf = FakeAdf(["T", "PMID"])
    with pytest.raises(KeyError):
        # Gets past the filter and into the real work, where this stub's
        # dataset stands in for a climo file it does not have -- that it gets
        # that far is the point.
        rvi._write_reference_files(adf, ["PMID"], {}, tmp_path, True)
    assert not adf.logged
