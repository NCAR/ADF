"""Resolving the model's 3-D pressure field, and asking the ADF to produce it.

Vertical interpolation prefers the pressure a model wrote to reconstructing it
from PS and the hybrid coefficients, but "the pressure a model wrote" is not
always called PMID: the lookup has to cope with another name given in the
config, with a model that only labels the field through CF metadata, and with a
model that writes no pressure at all (where the PS fallback has to stay).
"""

import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
# geocat.comp is pulled in by adf_utils.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))

import adf_utils as utils  # noqa: E402


class FakeAdf:
    """Just enough AdfDiag for request_pressure_field."""

    def __init__(self, diag_var_list, pressure_field_names=None):
        self._vars = list(diag_var_list)
        self._names = pressure_field_names

    @property
    def diag_var_list(self):
        return list(self._vars)

    def add_diag_var(self, var_str):
        if var_str not in self._vars:
            self._vars.append(var_str)

    def get_basic_info(self, var_str, required=False):
        return self._names if var_str == "pressure_field_names" else None


def make_dataset(pressure_name=None, standard_name=None, level_dim="lev"):
    """A history-file-like dataset, optionally carrying a 3-D pressure field."""
    shape = (2, 3, 4)
    coords = {level_dim: np.array([850.0, 500.0]), "lat": np.arange(3.0)}
    data_vars = {
        "T": xr.DataArray(np.zeros(shape), dims=(level_dim, "lat", "lon")),
        "PS": xr.DataArray(np.zeros((3, 4)), dims=("lat", "lon")),
    }
    if pressure_name is not None:
        attrs = {"standard_name": standard_name} if standard_name else {}
        data_vars[pressure_name] = xr.DataArray(
            np.zeros(shape), dims=(level_dim, "lat", "lon"), attrs=attrs
        )
    return xr.Dataset(data_vars, coords=coords)


def test_cam_names_are_the_default():
    assert utils.pressure_field_name("lev") == "PMID"
    assert utils.pressure_field_name("ilev") == "PINT"


def test_config_overrides_the_name():
    names = {"lev": "pfull", "ilev": "phalf"}
    assert utils.pressure_field_name("lev", names) == "pfull"
    assert utils.pressure_field_name("ilev", names) == "phalf"
    # A partial mapping leaves the other dimension at CAM's name:
    assert utils.pressure_field_name("ilev", {"lev": "pfull"}) == "PINT"


def test_false_disables_the_pressure_field():
    """'pressure_field_names: False' keeps the PS + hybrid path."""
    assert utils.pressure_field_name("lev", False) is None
    assert utils.find_pressure_field(make_dataset("PMID"), "lev", False) is None
    adf = FakeAdf(["T"], pressure_field_names=False)
    assert utils.request_pressure_field(adf, make_dataset("PMID")) == []
    assert adf.diag_var_list == ["T"]


def test_found_by_name():
    assert utils.find_pressure_field(make_dataset("PMID"), "lev") == "PMID"
    assert (
        utils.find_pressure_field(make_dataset("pfull"), "lev", {"lev": "pfull"})
        == "pfull"
    )


def test_found_by_cf_metadata():
    """A model the ADF has not met, with no config entry, but with CF metadata."""
    ds = make_dataset("pfull", standard_name="air_pressure")
    assert utils.find_pressure_field(ds, "lev") == "pfull"


def test_cf_metadata_must_be_on_the_right_dimension():
    """Interface pressure is not midpoint pressure, whatever its standard_name."""
    ds = make_dataset("phalf", standard_name="air_pressure", level_dim="ilev")
    assert utils.find_pressure_field(ds, "ilev") == "phalf"
    assert utils.find_pressure_field(ds, "lev") is None


def test_missing_pressure_field_is_not_an_error():
    """No pressure field: the caller falls back to PS + hybrid coefficients."""
    ds = make_dataset()
    assert utils.find_pressure_field(ds, "lev") is None
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field(adf, ds) == []
    assert adf.diag_var_list == ["T"]


def test_request_adds_the_field_once():
    adf = FakeAdf(["T", "Q"])
    assert utils.request_pressure_field(adf, make_dataset("PMID")) == ["PMID"]
    assert adf.diag_var_list == ["T", "Q", "PMID"]
    # Asking again (the next history stream, or the next case) adds nothing:
    assert utils.request_pressure_field(adf, make_dataset("PMID")) == []
    assert adf.diag_var_list == ["T", "Q", "PMID"]


def test_request_covers_both_vertical_dimensions():
    """A run using midpoint and interface fields needs both pressures."""
    ds = make_dataset("PMID")
    ds["PINT"] = xr.DataArray(np.zeros((2, 3, 4)), dims=("ilev", "lat", "lon"))
    ds["Uzm"] = xr.DataArray(np.zeros((2, 3, 4)), dims=("ilev", "lat", "lon"))
    adf = FakeAdf(["T", "Uzm"])
    assert utils.request_pressure_field(adf, ds) == ["PMID", "PINT"]


def write_ts(directory, name, level_dim="lev"):
    """Write a tiny time series file, 3-D when `level_dim` is given."""
    dims = ("time", level_dim, "lat") if level_dim else ("time", "lat")
    shape = (1, 2, 3) if level_dim else (1, 3)
    var = name.split(".")[-3]
    xr.Dataset({var: (dims, np.zeros(shape))}).to_netcdf(directory / name)


def test_cf_scan_ignores_the_vertical_coordinate_itself():
    """CF labels a pressure-level coordinate 'air_pressure' too.

    Returning the 1-D coordinate as "the pressure field" would interpolate a
    column against itself, so only a field with horizontal structure counts.
    """
    ds = make_dataset()
    ds["lev"].attrs["standard_name"] = "air_pressure"
    assert utils.find_pressure_field(ds, "lev") is None
    # ... but a real 3-D field in the same file is still found:
    ds["pfull"] = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("lev", "lat", "lon"),
        attrs={"standard_name": "air_pressure"},
    )
    assert utils.find_pressure_field(ds, "lev") == "pfull"


def test_request_from_time_series(tmp_path):
    """'cam_ts_done: true' skips the history files; the time series remain."""
    write_ts(tmp_path, "case.cam.h0.T.000101-000512.nc")
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == []
    write_ts(tmp_path, "case.cam.h0.PMID.000101-000512.nc")
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == ["PMID"]
    assert adf.diag_var_list == ["T", "PMID"]
    # Asking again (the next history stream) adds nothing:
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == []


def test_request_from_time_series_skips_a_two_dimensional_run(tmp_path):
    """A 3-D pressure field is a whole climatology; do not make one for nothing.

    The time series tree may hold a pressure field left by another run even
    though nothing in this one sits on model levels.
    """
    write_ts(tmp_path, "case.cam.h0.TS.000101-000512.nc", level_dim=None)
    write_ts(tmp_path, "case.cam.h0.PMID.000101-000512.nc")
    adf = FakeAdf(["TS"])
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == []
    assert adf.diag_var_list == ["TS"]


def test_request_from_a_nested_time_series_tree(tmp_path):
    """A GenTS archive: 'cam_ts_loc' points at the top of the tree.

    See find_ts_files -- the layout is <component>/proc/tseries/<frequency>/,
    and a flat glob of the top directory finds nothing at all.
    """
    nested = tmp_path / "atm" / "proc" / "tseries" / "month_1"
    nested.mkdir(parents=True)
    write_ts(nested, "case.cam.h0a.T.000101-000512.nc")
    write_ts(nested, "case.cam.h0a.PMID.000101-000512.nc")
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == ["PMID"]


def test_request_from_time_series_is_anchored_on_the_case(tmp_path):
    """Several cases can share one time series tree.

    Another case's pressure field is not this case's, and acting on it would
    send the ADF looking for a time series this case does not have.
    """
    write_ts(tmp_path, "caseA.cam.h0.T.000101-000512.nc")
    write_ts(tmp_path, "caseA.cam.h0.PMID.000101-000512.nc")
    write_ts(tmp_path, "caseB.cam.h0.T.000101-000512.nc")
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "caseB") == []
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "caseA") == ["PMID"]


def test_request_from_time_series_is_anchored_on_the_stream(tmp_path):
    """A pressure field on a stream this case does not use is not its own."""
    write_ts(tmp_path, "case.cam.h0.T.000101-000512.nc")
    write_ts(tmp_path, "case.cam.h1.PMID.000101-000512.nc")
    assert (
        utils.request_pressure_field_from_ts(FakeAdf(["T"]), tmp_path, "case", "cam.h0")
        == []
    )
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field_from_ts(
        adf, tmp_path, "case", ["cam.h0", "cam.h1"]
    ) == ["PMID"]


def test_request_from_time_series_honors_the_config(tmp_path):
    write_ts(tmp_path, "case.cam.h0.T.000101-000512.nc")
    write_ts(tmp_path, "case.cam.h0.pfull.000101-000512.nc")
    assert utils.request_pressure_field_from_ts(FakeAdf(["T"]), tmp_path, "case") == []
    adf = FakeAdf(["T"], pressure_field_names={"lev": "pfull"})
    assert utils.request_pressure_field_from_ts(adf, tmp_path, "case") == ["pfull"]
    off = FakeAdf(["T"], pressure_field_names=False)
    assert utils.request_pressure_field_from_ts(off, tmp_path, "case") == []


def test_request_skips_an_unused_vertical_dimension():
    """A file can carry 'ilev' while every requested variable is on midpoints.

    Interface pressure is another full 3-D field to write, so it is only
    requested when something in the run is actually on interfaces.
    """
    ds = make_dataset("PMID")
    ds["PINT"] = xr.DataArray(np.zeros((2, 3, 4)), dims=("ilev", "lat", "lon"))
    adf = FakeAdf(["T"])
    assert utils.request_pressure_field(adf, ds) == ["PMID"]
