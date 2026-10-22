"""Climo and regridded files are found by their climatology years.

Climo and regridded files are named after the climatology years they cover,
{stem}_{YYYY-YYYY}.nc, so that a run over other years (an extended run) never
reads a file made for the old ones.  Files written before the years were in
the names ({stem}.nc) are still found, as a fallback.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

from adf_dataset import AdfData  # noqa: E402


def _data(climo_loc, rgrid_loc):
    """An AdfData stand-in: test case "case" (years 3-5), baseline "base" (1-5)."""
    adf = SimpleNamespace(
        climo_yrs={
            "syears": [3],
            "eyears": [5],
            "syear_baseline": 1,
            "eyear_baseline": 5,
        },
        compare_obs=False,
        get_cam_info=lambda name, required=False: [str(climo_loc)],
        get_baseline_info=lambda name, required=False: str(climo_loc),
    )
    obj = SimpleNamespace(
        adf=adf,
        case_names=["case"],
        ref_case_label="base",
        model_rgrid_loc=str(rgrid_loc),
        ref_labels={"T": "base"},
        _hist_strs_for_case=lambda case: ["cam.h0a"],
        _hist_strs_for_reference=lambda: ["cam.h0a"],
        _first_file=AdfData._first_file,
    )
    obj._case_climo_years = lambda case: AdfData._case_climo_years(obj, case)
    obj._ref_climo_years = lambda: AdfData._ref_climo_years(obj)
    return obj


def _touch(loc, *names):
    for name in names:
        (Path(loc) / name).write_text("")


def test_files_with_the_climo_years_are_read_first():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(
            tmp,
            "case_cam.h0a_T_climo.nc",
            "case_cam.h0a_T_climo_0003-0005.nc",
            "base_cam.h0a_T_climo_0001-0005.nc",
            "base_case_T_regridded_0003-0005.nc",
            "base_T_baseline_0001-0005.nc",
            "base_case_T_regridded.nc",
        )
        obj = _data(tmp, tmp)
        assert (
            AdfData.get_climo_file(obj, "case", "T")[0].name
            == "case_cam.h0a_T_climo_0003-0005.nc"
        )
        assert (
            AdfData.get_reference_climo_file(obj, "T")[0].name
            == "base_cam.h0a_T_climo_0001-0005.nc"
        )
        assert (
            AdfData.get_regrid_file(obj, "case", "T")[0].name
            == "base_case_T_regridded_0003-0005.nc"
        )
        assert (
            AdfData.get_ref_regrid_file(obj, "base", "T")[0].name
            == "base_T_baseline_0001-0005.nc"
        )


def test_files_for_other_years_are_not_read():
    """A climo made for years 1-5 is not this case's (years 3-5)."""
    with tempfile.TemporaryDirectory() as tmp:
        _touch(tmp, "case_cam.h0a_T_climo_0001-0005.nc")
        assert AdfData.get_climo_file(_data(tmp, tmp), "case", "T") == []


def test_older_names_without_years_are_still_read():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(
            tmp,
            "case_T_climo.nc",
            "base_cam.h0a_T_climo.nc",
            "base_case_T_regridded.nc",
        )
        obj = _data(tmp, tmp)
        assert AdfData.get_climo_file(obj, "case", "T")[0].name == "case_T_climo.nc"
        assert (
            AdfData.get_reference_climo_file(obj, "T")[0].name
            == "base_cam.h0a_T_climo.nc"
        )
        assert (
            AdfData.get_regrid_file(obj, "case", "T")[0].name
            == "base_case_T_regridded.nc"
        )
