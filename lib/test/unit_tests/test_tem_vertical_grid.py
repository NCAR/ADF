"""TEM has to cope with either CAM vertical grid, and with a mix of the two.

CAM writes the zonal-mean fields either on layer midpoints ('lev') or on layer
interfaces ('ilev'), and a history stream can carry both. calc_tem sizes every
array from one vertical coordinate, so a stream it disagrees with fails with a
bare broadcasting error:

    ValueError: operands could not be broadcast together with shapes (71,96) (70,96)

Which grid the output lands on also decides which pressure field the plotting
script must ask for -- PMID on midpoints, PINT on interfaces -- so getting this
wrong silently pairs the fields with the wrong pressure.
"""
import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
# geocat.comp is pulled in by adf_utils, which create_TEM_files imports.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3] / "scripts" / "averaging"))

import adf_utils as utils  # noqa: E402
from create_TEM_files import (  # noqa: E402
    TEM_INPUT_VARS,
    calc_tem,
    harmonize_tem_levels,
)

NLAT, NMID, NINT = 12, 5, 6


def _dataset(on_interfaces=()):
    """TEM inputs, with the named fields put on 'ilev' and the rest on 'lev'."""
    lev = np.linspace(1.0, 1000.0, NMID)
    ilev = np.linspace(0.5, 1000.0, NINT)
    lat = np.linspace(-85.0, 85.0, NLAT)
    rng = np.random.default_rng(0)

    ds = xr.Dataset(coords={"lev": lev, "ilev": ilev, "zalat": lat})
    for name in TEM_INPUT_VARS:
        dim, size = ("ilev", NINT) if name in on_interfaces else ("lev", NMID)
        values = rng.normal(size=(size, NLAT))
        if name == "THzm":
            values = 250.0 + values
        ds[name] = ((dim, "zalat"), values)
    ds["date"], ds["datesec"] = 101, 0
    return ds


def test_all_on_midpoints_stays_on_midpoints():
    ds, lev_name = harmonize_tem_levels(_dataset())
    assert lev_name == "lev"
    assert {utils.vertical_dim(ds[v]) for v in TEM_INPUT_VARS} == {"lev"}


def test_all_on_interfaces_stays_on_interfaces():
    """An interface-only stream is left alone; it is self-consistent already."""
    ds, lev_name = harmonize_tem_levels(_dataset(on_interfaces=TEM_INPUT_VARS))
    assert lev_name == "ilev"
    assert {utils.vertical_dim(ds[v]) for v in TEM_INPUT_VARS} == {"ilev"}


def test_mixed_grids_collapse_onto_midpoints():
    """Midpoints are the common grid, because PMID is the pressure we prefer."""
    ds, lev_name = harmonize_tem_levels(_dataset(on_interfaces=("UVzm", "UWzm")))
    assert lev_name == "lev"
    assert {utils.vertical_dim(ds[v]) for v in TEM_INPUT_VARS} == {"lev"}
    assert ds["UVzm"].sizes["lev"] == NMID


@pytest.mark.parametrize(
    "on_interfaces,expected_dim,expected_size",
    [
        ((), "lev", NMID),
        (TEM_INPUT_VARS, "ilev", NINT),
        (("UVzm", "UWzm", "VTHzm"), "lev", NMID),
    ],
)
def test_calc_tem_runs_on_whatever_harmonize_returns(on_interfaces, expected_dim,
                                                     expected_size):
    """The point of harmonizing: calc_tem must not hit a broadcasting error."""
    ds, lev_name = harmonize_tem_levels(_dataset(on_interfaces=on_interfaces))
    out = calc_tem(ds, lev_name)
    assert utils.vertical_dim(out["UZM"]) == expected_dim
    assert out["UZM"].sizes[expected_dim] == expected_size


def test_pressure_field_follows_the_grid():
    """PMID pairs with midpoints, PINT with interfaces."""
    assert utils.pressure_field_name("lev") == "PMID"
    assert utils.pressure_field_name("ilev") == "PINT"


def test_vertical_dim_ignores_other_dimensions():
    da = xr.DataArray(np.zeros((2, 3)), dims=("time", "zalat"))
    assert utils.vertical_dim(da) is None


def test_unreconcilable_grids_are_reported_not_raised(capsys):
    """The ADF reports what it cannot do and carries on with the rest.

    A mix of grids needs the 'lev' coordinate values to interpolate onto. A file
    can have the dimension without the coordinate, and that case has to come back
    as a skip signal rather than an exception, or one bad case takes down every
    other case's TEM output with it.
    """
    ds = _dataset(on_interfaces=("UVzm",)).drop_vars("lev")
    assert "lev" in ds.dims and "lev" not in ds.coords

    ds_out, lev_name = harmonize_tem_levels(ds)

    assert lev_name is None, "caller needs a skip signal, not an exception"
    assert "WARNING" in capsys.readouterr().out
    assert ds_out is ds


def test_no_vertical_dimension_is_reported_not_guessed(capsys):
    """Fields on neither 'lev' nor 'ilev' cannot be used, so say so and skip.

    Guessing a grid here would send calc_tem into a broadcasting error rather
    than a message naming the case that could not be processed.
    """
    ds = _dataset()
    for name in TEM_INPUT_VARS:
        ds[name] = ds[name].rename({"lev": "something_else"})

    assert harmonize_tem_levels(ds)[1] is None
    assert "WARNING" in capsys.readouterr().out


def test_tem_variable_without_observations_is_skipped(capsys):
    """Not every TEM variable has an observational counterpart.

    THZM does not: the ERA5 TEM file carries no potential temperature, so its
    variable_defaults entry has no 'obs_file'. Looking that key up directly
    raises KeyError and takes the whole observational TEM file with it.
    """
    from create_TEM_files import _write_obs_tem_file

    class _Adf:
        def get_basic_info(self, key, required=False):
            return None

    # a variable with no 'obs_file', and one absent from variable_defaults
    _write_obs_tem_file(_Adf(), ["THZM", "NOT_A_VARIABLE"], {"THZM": {}}, Path("."))

    out = capsys.readouterr().out
    assert "WARNING" in out and "no observation files" in out
