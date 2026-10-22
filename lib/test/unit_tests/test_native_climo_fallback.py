"""Native plotting with no regridding stage reads the native climo.

A run with 'unstructured_plotting' reads the files the regridding stage leaves
on the native grid.  The CLM uxarray configs leave that stage out, and then
every map script found no files and skipped every variable without an error.
The loaders fall back to the native climo in that case, and only then: a native
climo is no stand-in for a lat/lon regridded file.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

from adf_dataset import AdfData  # noqa: E402

CLIMO = ["case_TBOT_climo.nc"]
BASE_CLIMO = ["base_TBOT_climo.nc"]


def _data(unstructured, regrid_fils=(), ref_regrid_fils=()):
    """An AdfData stand-in recording which files the loaders open."""
    obj = SimpleNamespace(
        adf=SimpleNamespace(
            unstructured_plotting=unstructured, debug_log=lambda msg: None
        ),
        get_regrid_file=lambda case, field: list(regrid_fils),
        get_ref_regrid_file=lambda case, field: list(ref_regrid_fils),
        get_climo_file=lambda case, field: list(CLIMO),
        get_reference_climo_file=lambda field: list(BASE_CLIMO),
        opened=[],
    )
    obj._native_climo_fallback = (
        lambda case, field, fils: AdfData._native_climo_fallback(obj, case, field, fils)
    )
    obj.load_dataset = lambda fils, **kwargs: obj.opened.append(list(fils)) or "ds"
    return obj


def test_native_run_without_regrid_files_reads_the_climo():
    obj = _data(unstructured=True)
    assert AdfData.load_regrid_dataset(obj, "case", "TBOT") == "ds"
    assert AdfData.load_reference_regrid_dataset(obj, "base", "TBOT") == "ds"
    assert obj.opened == [CLIMO, BASE_CLIMO]


def test_regrid_files_win_over_the_climo():
    obj = _data(
        unstructured=True, regrid_fils=["rg.nc"], ref_regrid_fils=["base_rg.nc"]
    )
    AdfData.load_regrid_dataset(obj, "case", "TBOT")
    AdfData.load_reference_regrid_dataset(obj, "base", "TBOT")
    assert obj.opened == [["rg.nc"], ["base_rg.nc"]]


def test_variable_without_a_reference_has_no_regridded_file():
    """Against observations, a variable with no obs was never regridded.

    Asking for it gives no files (and the plots then skip it) rather than a
    KeyError from the reference labels, which ended a native-vs-obs run.
    """
    obj = SimpleNamespace(model_rgrid_loc="/nonexistent", ref_labels={"T": "ERA5"})
    assert AdfData.get_regrid_file(obj, "case", "RESTOM") == []


def test_lat_lon_run_does_not_fall_back():
    obj = _data(unstructured=False)
    with pytest.warns(UserWarning, match="Did not find regrid"):
        assert AdfData.load_regrid_dataset(obj, "case", "TBOT") is None
    with pytest.warns(UserWarning, match="Did not find regridded"):
        assert AdfData.load_reference_regrid_dataset(obj, "base", "TBOT") is None
    assert obj.opened == []
