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
    lev = np.array([200.0, 500.0, 850.0])
    hyam = np.array([0.2, 0.2, 0.0])
    hybm = np.array([0.0, 0.3, 0.85])
    # PS = 100000 Pa gives pressures of 20000, 50000, 85000 Pa
    time = np.arange(2.0)
    u = np.tile(np.array([30.0, 20.0, 10.0])[None, :, None], (2, 1, 3))
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


# ------------------------------------------------ the wiring, not the arithmetic


class HookAdf(ListAdf):
    """An ADF that can stand in for the one check_derive is given."""

    def __init__(self, variables, defaults):
        super().__init__(variables)
        self.variable_defaults = defaults
        self.logged = []

    def debug_log(self, msg):
        self.logged.append(msg)


DEFAULTS = {
    "U200": {"derivable_from": ["U"], "derive_level": 200},
    "U850": {"derivable_from": ["U"], "derive_level": 850},
    "PRECT": {"derivable_from": ["PRECC", "PRECL"]},
}


def run_check_derive(requested, defaults=None, history=None):
    """Walk the variable list the way create_time_series does.

    The loop appends to the list it is iterating, which is the part that has
    gone wrong twice: the returned list has to keep the constituents *and* gain
    the pressure field.
    """
    defaults = DEFAULTS if defaults is None else defaults
    history = history_like() if history is None else history
    adf = HookAdf(requested, defaults)
    diag_var_list = list(requested)
    constit_dict = {}
    for var in diag_var_list:
        if var in history.data_vars:
            continue
        diag_var_list, constit_dict = adf_derive.check_derive(
            adf, defaults, var, "case", diag_var_list, constit_dict, history, "hist0"
        )
    # End for
    return adf, diag_var_list, constit_dict


def test_the_constituent_and_the_pressure_field_both_survive_the_loop():
    """The regression that dropped V200/V850: the list must not be replaced."""
    adf, diag_var_list, constit_dict = run_check_derive(["U200", "U850"])
    assert constit_dict == {"U200": ["U"], "U850": ["U"]}
    for name in ("U200", "U850", "U", "PMID"):
        assert name in diag_var_list, f"{name} fell out of the variable list"
    # PMID is asked for once, however many surfaces want it:
    assert diag_var_list.count("PMID") == 1
    assert "PMID" in adf.diag_var_list


def test_the_pressure_field_is_requested_for_a_derived_surface():
    """The regression that fell back to PS + hybrid: nothing in the run is on
    model levels until the constituent is added, so the constituent has to be
    what the request is judged by."""
    _, diag_var_list, _ = run_check_derive(["U200"])
    assert "PMID" in diag_var_list


def test_an_ordinary_derived_variable_asks_for_no_pressure():
    """PRECT has no 'derive_level', so nothing changes for it."""
    history = xr.Dataset(
        {
            "PRECC": (("time", "lat"), np.zeros((1, 3))),
            "PRECL": (("time", "lat"), np.zeros((1, 3))),
            "U": (("time", "lev", "lat"), np.zeros((1, 2, 3))),
            "PMID": (("time", "lev", "lat"), np.zeros((1, 2, 3))),
        },
        coords={"lev": [850.0, 200.0]},
    )
    _, diag_var_list, constit_dict = run_check_derive(["PRECT"], history=history)
    assert constit_dict == {"PRECT": ["PRECC", "PRECL"]}
    assert "PMID" not in diag_var_list


def test_a_model_without_a_pressure_field_still_derives():
    """No PMID in the history files: the constituent is still requested."""
    history = xr.Dataset(
        {"U": (("time", "lev", "lat"), np.zeros((1, 2, 3)))},
        coords={"lev": [850.0, 200.0]},
    )
    _, diag_var_list, constit_dict = run_check_derive(["U200"], history=history)
    assert constit_dict == {"U200": ["U"]}
    assert "U" in diag_var_list and "PMID" not in diag_var_list


# ------------------------------------------- the model's own pressure field path


class DataStub:
    """The one AdfData method interpolate_to_level uses."""

    @staticmethod
    def load_dataset(fils):
        if len(fils) == 1:
            return xr.open_dataset(fils[0])
        return xr.open_mfdataset(fils, combine="by_coords")


class PresAdf(FakeDeriveAdf):
    """A derive-stage ADF that can read a pressure time series."""

    def __init__(self):
        super().__init__()
        self.data = DataStub()


def write_pressure_ts(directory, times, surface_pa=100000.0, top_down=True):
    """A PMID time series whose surface pressure is `surface_pa`.

    CAM writes the column top down; `top_down=False` writes it the other way
    round, which the interpolation has to cope with rather than quietly
    returning the wrong end.
    """
    lev = (
        np.array([200.0, 500.0, 850.0]) if top_down else np.array([850.0, 500.0, 200.0])
    )
    # Pressures scale with the surface, so a low surface puts 850 hPa below ground
    fractions = np.array([0.2, 0.5, 0.85]) if top_down else np.array([0.85, 0.5, 0.2])
    column = fractions * surface_pa
    pmid = np.tile(column[None, :, None], (len(times), 1, 3))
    xr.Dataset(
        {"PMID": (("time", "lev", "lat"), pmid)},
        coords={
            "time": np.asarray(times, dtype=float),
            "lev": lev,
            "lat": np.arange(3.0),
        },
    ).to_netcdf(directory / f"case.cam.h0a.PMID.000101-000212.nc")


def field_dataset(times, top_down=True):
    """The constituent, with values that identify the level they came from."""
    lev = (
        np.array([200.0, 500.0, 850.0]) if top_down else np.array([850.0, 500.0, 200.0])
    )
    values = np.array([30.0, 20.0, 5.0]) if top_down else np.array([5.0, 20.0, 30.0])
    u = np.tile(values[None, :, None], (len(times), 1, 3))
    return xr.Dataset(
        {"U200": (("time", "lev", "lat"), u)},
        coords={
            "time": np.asarray(times, dtype=float),
            "lev": lev,
            "lat": np.arange(3.0),
        },
    )


def test_the_model_pressure_field_is_used_when_there_is_one(tmp_path):
    write_pressure_ts(tmp_path, [0.0, 1.0])
    out = adf_derive.interpolate_to_level(
        PresAdf(),
        field_dataset([0.0, 1.0]),
        "U200",
        200,
        tmp_path,
        "case",
        hist_str="cam.h0a",
    )
    assert out is not None
    assert out.attrs["interpolated_with"] == "PMID"
    assert np.allclose(out.values, 30.0)


def test_below_ground_is_missing_not_clamped(tmp_path):
    """850 hPa under a 700 hPa surface is not the bottom model level.

    np.interp, which the stacked interpolation uses, clamps to the end value
    instead of returning NaN -- so without the mask the boundary-layer wind is
    reported as the 850 hPa wind over Tibet, Greenland and Antarctica, and the
    hybrid path reports the same point as missing.
    """
    write_pressure_ts(tmp_path, [0.0, 1.0], surface_pa=70000.0)
    out = adf_derive.interpolate_to_level(
        PresAdf(),
        field_dataset([0.0, 1.0]),
        "U200",
        850,
        tmp_path,
        "case",
        hist_str="cam.h0a",
    )
    assert out is not None
    assert np.all(np.isnan(out.values)), "below-ground points must be missing"


def test_a_longer_pressure_series_is_cut_to_the_field(tmp_path):
    """A pressure field covering more times than the constituent.

    It passes the coverage check, and broadcasting joins on the union, so
    without the selection the field meets the wrong times' pressure.  The
    surface pressure changes partway through here, so pairing the field with
    the wrong end of the series gives a different answer rather than the same
    one.
    """
    lev = np.array([200.0, 500.0, 850.0])
    times = np.array([-2.0, -1.0, 0.0, 1.0])
    # The first two times have a much lower surface pressure than the last two:
    surfaces = np.array([70000.0, 70000.0, 100000.0, 100000.0])
    column = np.array([0.2, 0.5, 0.85])[None, :] * surfaces[:, None]
    xr.Dataset(
        {"PMID": (("time", "lev", "lat"), np.repeat(column[:, :, None], 3, axis=2))},
        coords={"time": times, "lev": lev, "lat": np.arange(3.0)},
    ).to_netcdf(tmp_path / "case.cam.h0a.PMID.000101-000212.nc")

    out = adf_derive.interpolate_to_level(
        PresAdf(),
        field_dataset([0.0, 1.0]),
        "U200",
        200,
        tmp_path,
        "case",
        hist_str="cam.h0a",
    )
    assert out is not None
    assert out.sizes["time"] == 2
    # 200 hPa is exactly the top level of the *later* columns:
    assert np.allclose(out.values, 30.0)


def test_a_shorter_pressure_series_falls_back(tmp_path):
    """Genuinely missing times: use PS and the hybrid coefficients instead."""
    write_pressure_ts(tmp_path, [0.0])
    ds = field_dataset([0.0, 1.0])
    ds["PS"] = (("time", "lat"), np.full((2, 3), 100000.0))
    ds["hyam"] = (("lev",), np.array([0.2, 0.2, 0.0]))
    ds["hybm"] = (("lev",), np.array([0.0, 0.3, 0.85]))
    adf = PresAdf()
    out = adf_derive.interpolate_to_level(
        adf, ds, "U200", 200, tmp_path, "case", hist_str="cam.h0a"
    )
    assert out is not None
    assert "hybrid" in out.attrs["interpolated_with"]
    assert adf.logged, "the fallback should say why in the debug log"


def test_surface_pressure_may_be_in_its_own_time_series(tmp_path):
    """GenTS writes PS to its own file, so the fallback has to look for it."""
    times = [0.0, 1.0]
    xr.Dataset(
        {"PS": (("time", "lat"), np.full((2, 3), 100000.0))},
        coords={"time": np.asarray(times), "lat": np.arange(3.0)},
    ).to_netcdf(tmp_path / "case.cam.h0a.PS.000101-000212.nc")
    ds = field_dataset(times)
    ds["hyam"] = (("lev",), np.array([0.2, 0.2, 0.0]))
    ds["hybm"] = (("lev",), np.array([0.0, 0.3, 0.85]))
    out = adf_derive.interpolate_to_level(
        PresAdf(), ds, "U200", 200, tmp_path, "case", hist_str="cam.h0a"
    )
    assert out is not None
    assert np.allclose(out.values, 30.0)


def test_a_column_stored_bottom_up_gives_the_same_answer(tmp_path):
    """np.interp needs increasing pressures and does not check for them."""
    write_pressure_ts(tmp_path, [0.0, 1.0], top_down=False)
    out = adf_derive.interpolate_to_level(
        PresAdf(),
        field_dataset([0.0, 1.0], top_down=False),
        "U200",
        200,
        tmp_path,
        "case",
        hist_str="cam.h0a",
    )
    assert out is not None
    assert np.allclose(out.values, 30.0)


def test_a_baseline_field_is_never_sliced():
    """'obs_lev' is about observations; a baseline's own U200 is already 2-D.

    Making that structural rather than incidental means a 3-D baseline field
    cannot be sliced on the reference side only.
    """
    data = adf_dataset.AdfData.__new__(adf_dataset.AdfData)
    data.adf = FakeAdf({"U200": {"obs_lev": 200}})
    data.adf.compare_obs = False
    three_d = obs_data([100.0, 200.0, 850.0])
    assert data._at_obs_level(three_d, "U200").equals(three_d)


def test_an_unknown_vertical_dimension_is_reported():
    """A 3-D observation the slicer cannot read must not pass silently."""
    data = adf_dataset.AdfData.__new__(adf_dataset.AdfData)
    data.adf = FakeAdf({"U200": {"obs_lev": 200}})
    odd = xr.DataArray(
        np.zeros((2, 2, 4, 5)),
        dims=("time", "isobaric", "lat", "lon"),
        coords={"isobaric": [200.0, 850.0]},
    )
    with pytest.warns(UserWarning, match="vertical"):
        out = data._at_obs_level(odd, "U200")
    assert out.equals(odd)


# --------------------------------------------------------- the derived file


class DeriveAdf(PresAdf):
    """Enough AdfDiag for derive_variable end to end."""

    def __init__(self):
        super().__init__()
        self.variable_defaults = {}

    @property
    def diag_var_list(self):
        return []


def write_constituent(directory, times, name="U"):
    """A 3-D constituent time series, as the time series stage would leave it."""
    lev = np.array([200.0, 500.0, 850.0])
    values = np.array([30.0, 20.0, 5.0])
    ds = xr.Dataset(
        {
            name: (
                ("time", "lev", "lat"),
                np.tile(values[None, :, None], (len(times), 1, 3)),
            )
        },
        coords={
            "time": np.asarray(times, dtype=float),
            "lev": lev,
            "lat": np.arange(3.0),
        },
    )
    ds[name].attrs = {"units": "m/s", "long_name": "Zonal wind", "mdims": 1}
    ds.to_netcdf(directory / f"case.cam.h0a.{name}.000101-000212.nc")


@pytest.mark.parametrize("keep_attrs", [True, False])
def test_the_derived_file_keeps_the_units(tmp_path, keep_attrs):
    """The units come from the constituent, not from the arithmetic on it.

    Whether `0 + da` keeps attributes depends on the xarray version -- it does
    not in the one env/conda_environment.yaml pins -- so reading them off the
    summed field loses the units in some environments and not others, and the
    AMWG table then prints '--' for every one of these variables.
    """
    write_constituent(tmp_path, [0.0, 1.0])
    write_pressure_ts(tmp_path, [0.0, 1.0])
    res = {"U200": {"derivable_from": ["U"], "derive_level": 200}}
    with xr.set_options(keep_attrs=keep_attrs):
        adf_derive.derive_variable(
            DeriveAdf(),
            "case",
            "U200",
            res=res,
            ts_dir=tmp_path,
            constit_list=["U"],
            overwrite=True,
            hist_str="cam.h0a",
        )
    written = list(tmp_path.glob("case.cam.h0a.U200.*.nc"))
    assert written, "no derived file was written"
    with xr.open_dataset(written[0]) as out:
        assert "U200" in out
        assert set(out["U200"].dims) == {"time", "lat"}
        assert out["U200"].attrs.get("units") == "m/s"
        assert out["U200"].attrs["long_name"] == "Zonal wind at 200 hPa"
        assert "mdims" not in out["U200"].attrs
        assert np.allclose(out["U200"].values, 30.0)


def test_a_level_above_twenty_hectopascals(tmp_path):
    """The mask has to know what units the interpolated levels are in.

    Guessing from their magnitude reads a 10 hPa target (1000 Pa) as hPa and
    masks the entire field; 5 hPa passes the range test instead and turns the
    mask off.  QBO levels are exactly where someone would use this.
    """
    lev = np.array([5.0, 50.0, 500.0])
    pmid = np.tile((np.array([0.005, 0.05, 0.5]) * 100000.0)[None, :, None], (2, 1, 3))
    xr.Dataset(
        {"PMID": (("time", "lev", "lat"), pmid)},
        coords={"time": np.arange(2.0), "lev": lev, "lat": np.arange(3.0)},
    ).to_netcdf(tmp_path / "case.cam.h0a.PMID.000101-000212.nc")
    ds = xr.Dataset(
        {
            "U10hpa": (
                ("time", "lev", "lat"),
                np.tile(np.array([40.0, 25.0, 10.0])[None, :, None], (2, 1, 3)),
            )
        },
        coords={"time": np.arange(2.0), "lev": lev, "lat": np.arange(3.0)},
    )
    out = adf_derive.interpolate_to_level(
        PresAdf(), ds, "U10hpa", 10, tmp_path, "case", hist_str="cam.h0a"
    )
    assert out is not None
    assert not np.any(np.isnan(out.values)), "10 hPa is inside this column"
    # Between the 5 hPa (40) and 50 hPa (25) levels, nearer the 5 hPa end:
    assert 25.0 < float(out.values.mean()) < 40.0
