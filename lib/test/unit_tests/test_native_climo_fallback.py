"""Native plotting with no regridding stage reads the native climo.

A run with 'unstructured_plotting' reads the files the regridding stage leaves
on the native grid.  The CLM uxarray configs leave that stage out, and then
every map script found no files and skipped every variable without an error.
The loaders fall back to the native climo in that case, and only then: a native
climo is no stand-in for a lat/lon regridded file.
"""

# The tests drive AdfData's private helpers through a stand-in object:
# pylint: disable=protected-access

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

from adf_dataset import AdfData  # noqa: E402

CLIMO = ["case_TBOT_climo.nc"]
BASE_CLIMO = ["base_TBOT_climo.nc"]


def _data(
    unstructured, regrid_fils=(), ref_regrid_fils=(), on_mesh_fils=(), native_file=False
):
    """An AdfData stand-in recording which files (and meshes) the loaders open."""
    obj = SimpleNamespace(
        adf=SimpleNamespace(
            unstructured_plotting=unstructured,
            debug_log=lambda msg: None,
            mesh_files={
                "test_mesh_file": ["case_mesh.nc"],
                "baseline_mesh_file": "base_mesh.nc",
            },
        ),
        case_names=["case"],
        get_regrid_file=lambda case, field: list(regrid_fils),
        get_ref_regrid_file=lambda case, field: list(ref_regrid_fils),
        get_ref_on_mesh_file=lambda test_case, field: (
            list(on_mesh_fils) if test_case else []
        ),
        get_climo_file=lambda case, field: list(CLIMO),
        get_reference_climo_file=lambda field: list(BASE_CLIMO),
        opened=[],
        meshes=[],
    )
    obj._native_climo_fallback = (
        lambda case, field, fils: AdfData._native_climo_fallback(obj, case, field, fils)
    )
    obj._reference_files = lambda case, field, kwargs: AdfData._reference_files(
        obj, case, field, kwargs
    )
    obj._regrid_files = lambda case, field, kwargs: AdfData._regrid_files(
        obj, case, field, kwargs
    )
    obj._on_native_dim = lambda fil: native_file

    def load_dataset(fils, **kwargs):
        obj.opened.append(list(fils))
        obj.meshes.append(kwargs.get("mesh_file"))
        return "ds"

    obj.load_dataset = load_dataset
    return obj


def test_native_run_without_regrid_files_reads_the_climo():
    """No regridded files: a native run plots the native climo (test case and baseline)."""
    obj = _data(unstructured=True)
    assert AdfData.load_regrid_dataset(obj, "case", "TBOT") == "ds"
    assert AdfData.load_reference_regrid_dataset(obj, "base", "TBOT") == "ds"
    assert obj.opened == [CLIMO, BASE_CLIMO]


def test_regrid_files_win_over_the_climo():
    """Regridded files, when there are any, are read instead of the climo."""
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
    """A lat/lon run never plots a native climo in place of a regridded file."""
    obj = _data(unstructured=False)
    with pytest.warns(UserWarning, match="Did not find regrid"):
        assert AdfData.load_regrid_dataset(obj, "case", "TBOT") is None
    with pytest.warns(UserWarning, match="Did not find regridded"):
        assert AdfData.load_reference_regrid_dataset(obj, "base", "TBOT") is None
    assert obj.opened == []


def test_reference_on_the_case_mesh_is_preferred():
    """With a test case given, its on-mesh reference is read, on its own mesh.

    The regridding stage writes a lat/lon reference onto each native test
    case's mesh; the plots then compare the two on that mesh instead of
    skipping the maps.
    """
    obj = _data(
        unstructured=True, ref_regrid_fils=["base_rg.nc"], on_mesh_fils=["on_mesh.nc"]
    )
    AdfData.load_reference_regrid_dataset(
        obj, "base", "TBOT", test_case="case", mesh_file="base_mesh.nc"
    )
    assert obj.opened == [["on_mesh.nc"]]
    assert obj.meshes == ["case_mesh.nc"]


def test_without_an_on_mesh_file_the_usual_reference_is_read():
    """No on-mesh file (or no test case): the reference regridded file, as before."""
    obj = _data(unstructured=True, ref_regrid_fils=["base_rg.nc"])
    AdfData.load_reference_regrid_dataset(
        obj, "base", "TBOT", test_case="case", mesh_file="base_mesh.nc"
    )
    obj2 = _data(
        unstructured=True, ref_regrid_fils=["base_rg.nc"], on_mesh_fils=["on_mesh.nc"]
    )
    AdfData.load_reference_regrid_dataset(
        obj2, "base", "TBOT", mesh_file="base_mesh.nc"
    )
    assert obj.opened == obj2.opened == [["base_rg.nc"]]
    assert obj.meshes == obj2.meshes == ["base_mesh.nc"]


def test_lat_lon_run_ignores_on_mesh_files():
    """A lat/lon run never reads a reference that was put onto a mesh."""
    obj = _data(
        unstructured=False, ref_regrid_fils=["base_rg.nc"], on_mesh_fils=["on_mesh.nc"]
    )
    AdfData.load_reference_regrid_dataset(obj, "base", "TBOT", test_case="case")
    assert obj.opened == [["base_rg.nc"]]


def test_lat_lon_case_on_the_reference_mesh():
    """A lat/lon case put onto a native reference's mesh opens on that mesh.

    The case has no 'mesh_file' of its own, so the plots pass none; its
    regridded file is on the reference's grid, and is opened on its mesh.
    """
    obj = _data(unstructured=True, regrid_fils=["rg.nc"], native_file=True)
    AdfData.load_regrid_dataset(obj, "case", "TBOT", mesh_file=None)
    assert obj.meshes == ["base_mesh.nc"]


def test_lat_lon_regrid_file_keeps_no_mesh():
    """A regridded file on lat/lon is not opened on a mesh."""
    obj = _data(unstructured=True, regrid_fils=["rg.nc"], native_file=False)
    AdfData.load_regrid_dataset(obj, "case", "TBOT", mesh_file=None)
    assert obj.meshes == [None]
