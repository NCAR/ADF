"""Fields on a pressure surface: made from the model levels, compared level to level.

CAM writes U200 and friends only when asked, so a run that has the 3-D field
and not the surface can have the surface made from it ('derive_level').  The
reanalysis side is distributed as whole columns, so the level is taken out of
the 3-D observation rather than a file being staged per level ('obs_lev').
"""

import sys
import warnings
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))

import adf_dataset  # noqa: E402
import adf_derive  # noqa: E402
import adf_utils as utils  # noqa: E402

# ---------------------------------------------------------------- observations


class FakeAdf:
    """Just enough AdfDiag for the level selection."""

    def __init__(self, defaults):
        self.variable_defaults = defaults
        self.compare_obs = True

    def debug_log(self, msg):
        pass


def obs_data(levels, level_dim="lev"):
    """A 3-D observation whose values are the level, so a slice is checkable."""
    lat = np.arange(3.0)
    values = np.broadcast_to(np.asarray(levels)[:, None], (len(levels), 3))
    return xr.DataArray(
        values.astype(float),
        dims=(level_dim, "lat"),
        coords={level_dim: np.asarray(levels, dtype=float), "lat": lat},
        name="U",
    )


def selector(defaults):
    """The bound _at_obs_level method, with no real AdfData built."""
    data = adf_dataset.AdfData.__new__(adf_dataset.AdfData)
    data.adf = FakeAdf(defaults)
    return data._at_obs_level


def test_the_named_level_is_taken():
    at = selector({"U200": {"obs_lev": 200}})
    out = at(obs_data([100.0, 200.0, 850.0]), "U200")
    assert "lev" not in out.dims
    assert float(out.mean()) == 200.0


def test_a_level_coordinate_in_pascals_is_understood():
    """'obs_lev' is hPa; the file may not be."""
    at = selector({"U200": {"obs_lev": 200}})
    out = at(obs_data([10000.0, 20000.0, 85000.0]), "U200")
    assert float(out.mean()) == 20000.0


@pytest.mark.parametrize("dim", ["lev", "level", "plev", "pressure"])
def test_the_vertical_dimension_may_be_named_anything_usual(dim):
    at = selector({"U200": {"obs_lev": 200}})
    out = at(obs_data([100.0, 200.0, 850.0], level_dim=dim), "U200")
    assert dim not in out.dims
    assert float(out.mean()) == 200.0


def test_nothing_happens_without_the_key():
    at = selector({"U": {}})
    data = obs_data([100.0, 200.0])
    assert at(data, "U").equals(data)


def test_an_observation_already_on_one_surface_is_left_alone():
    """A staged 2-D file and a 3-D file can both serve the same entry."""
    at = selector({"U200": {"obs_lev": 200}})
    flat = xr.DataArray(np.zeros(3), dims=("lat",), name="U200")
    assert at(flat, "U200").equals(flat)


def test_a_distant_level_warns_but_is_used():
    at = selector({"U200": {"obs_lev": 200}})
    with pytest.warns(UserWarning, match="nearest level"):
        out = at(obs_data([100.0, 300.0]), "U200")
    assert float(out.mean()) == 300.0


def test_a_level_within_a_percent_is_quiet():
    at = selector({"U200": {"obs_lev": 200}})
    with _no_warning():
        out = at(obs_data([175.0, 201.0]), "U200")
    assert float(out.mean()) == 201.0


class _no_warning:
    """Assert no warning is raised (pytest.warns(None) was removed in pytest 8)."""

    def __enter__(self):
        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        warnings.simplefilter("always")
        return self._log

    def __exit__(self, *exc):
        result = self._ctx.__exit__(*exc)
        assert not [m for m in self._log if issubclass(m.category, UserWarning)]
        return result


def test_missing_level_is_not_an_error():
    """A file with no vertical dimension at all still loads."""
    at = selector({"U200": {"obs_lev": 200}})
    assert at(None, "U200") is None


# ----------------------------------------------------------------------- model


class FakeDeriveAdf:
    """Just enough AdfDiag for interpolate_to_level's fallback path."""

    def __init__(self):
        self.logged = []

    def get_basic_info(self, var_str, required=False):
        return None

    def debug_log(self, msg):
        self.logged.append(msg)


def hybrid_dataset():
    """A tiny hybrid-coordinate time series, with U increasing upward."""
    lev = np.array([850.0, 500.0, 200.0])
    hyam = np.array([0.0, 0.2, 0.2])
    hybm = np.array([0.85, 0.3, 0.0])
    # PS = 100000 Pa gives pressures of 85000, 50000, 20000 Pa
    time = np.arange(2.0)
    u = np.tile(np.array([10.0, 20.0, 30.0])[None, :, None], (2, 1, 3))
    return xr.Dataset(
        {
            "U200": (("time", "lev", "lat"), u),
            "PS": (("time", "lat"), np.full((2, 3), 100000.0)),
            "hyam": (("lev",), hyam),
            "hybm": (("lev",), hybm),
        },
        coords={"time": time, "lev": lev, "lat": np.arange(3.0)},
    )


def test_interpolation_falls_back_to_ps_and_hybrid(tmp_path):
    """No pressure time series in the directory: PS + hyam/hybm are in the file."""
    ds = hybrid_dataset()
    out = adf_derive.interpolate_to_level(
        FakeDeriveAdf(), ds, "U200", 200, tmp_path, "case"
    )
    assert out is not None
    assert "lev" not in out.dims and set(out.dims) == {"time", "lat"}
    # The 200 hPa level of this column is exactly the top value:
    assert np.allclose(out.values, 30.0)
    assert out.attrs["pressure_level"] == "200 hPa"
    assert "hybrid" in out.attrs["interpolated_with"]


def test_interpolation_between_levels(tmp_path):
    """A surface the model has no level at is interpolated to."""
    out = adf_derive.interpolate_to_level(
        FakeDeriveAdf(), hybrid_dataset(), "U200", 500, tmp_path, "case"
    )
    assert np.allclose(out.values, 20.0)


def test_a_two_dimensional_field_is_refused(tmp_path):
    """Nothing to interpolate: say so rather than writing something wrong."""
    ds = xr.Dataset({"TS": (("time", "lat"), np.zeros((2, 3)))})
    assert (
        adf_derive.interpolate_to_level(
            FakeDeriveAdf(), ds, "TS", 200, tmp_path, "case"
        )
        is None
    )


def test_no_pressure_at_all_is_refused(tmp_path):
    """Neither a pressure time series nor PS in the file."""
    ds = hybrid_dataset().drop_vars("PS")
    assert (
        adf_derive.interpolate_to_level(
            FakeDeriveAdf(), ds, "U200", 200, tmp_path, "case"
        )
        is None
    )


def test_pressure_in_pa_converts_only_when_needed():
    hpa = xr.DataArray(np.array([850.0]), attrs={"units": "hPa"})
    pa = xr.DataArray(np.array([85000.0]), attrs={"units": "Pa"})
    assert utils.pressure_in_pa(hpa).values[0] == 85000.0
    assert utils.pressure_in_pa(pa).values[0] == 85000.0
    assert utils.pressure_in_pa(hpa).attrs["units"] == "Pa"
    # No units attribute: decided by magnitude.
    bare = xr.DataArray(np.array([85000.0]))
    assert utils.pressure_in_pa(bare).values[0] == 85000.0


# ------------------------------------------------- asking for the pressure field


class ListAdf:
    """An ADF whose variable list can be added to, as AdfInfo's can."""

    def __init__(self, variables):
        self._vars = list(variables)

    @property
    def diag_var_list(self):
        return list(self._vars)

    def add_diag_var(self, var_str):
        if var_str not in self._vars:
            self._vars.append(var_str)

    def get_basic_info(self, var_str, required=False):
        return None


def history_like():
    """A history file with a model-level U and no U200."""
    return xr.Dataset(
        {
            "U": (("time", "lev", "lat"), np.zeros((1, 2, 3))),
            "PMID": (("time", "lev", "lat"), np.zeros((1, 2, 3))),
            "TS": (("time", "lat"), np.zeros((1, 3))),
        },
        coords={"lev": [850.0, 200.0]},
    )


def test_the_constituent_decides_whether_pressure_is_needed():
    """A run asking only for U200 has nothing on model levels -- yet.

    The pressure field has to be judged by the constituent that is about to be
    added, or a derived pressure-surface field silently falls back to PS and
    the hybrid coefficients.
    """
    adf = ListAdf(["U200"])
    assert utils.request_pressure_field(adf, history_like()) == []
    assert utils.request_pressure_field(adf, history_like(), variables=["U"]) == [
        "PMID"
    ]
    assert "PMID" in adf.diag_var_list


def test_a_two_dimensional_run_still_asks_for_nothing():
    adf = ListAdf(["TS"])
    assert utils.request_pressure_field(adf, history_like(), variables=["TS"]) == []
