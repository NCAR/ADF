"""The time series dataset and history loaders in AdfData.

Each kind of file the ADF reads has loaders that find the files for a case
(``get_*``), return a Dataset (``load_*_dataset``) and return one converted
variable (``load_*_da``), for the test cases and for the baseline (issue #424).
These pin down the ones that read whole time series Datasets and raw history
files: that they find the right files, and that they convert with the variable
defaults the way the other loaders do.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

from adf_dataset import AdfData  # noqa: E402

CASE = "test_case"
BASE = "base_case"
DEFAULTS = {
    "U": {"scale_factor": 2, "add_offset": 0, "new_unit": "cm/s"},
    "UZM": {"scale_factor": 2, "new_unit": "cm/s", "obs_scale_factor": 3},
}


def _write(path, months, extra=None):
    """A monthly file of U (all ones, in m/s) for the given (year, month) pairs."""
    # noleap days since 0001-01-01 of the start and end of each month:
    starts = np.array([365 * (y - 1) + 31 * (m - 1) for y, m in months], float)
    bnds = np.stack([starts, starts + 31], axis=1)
    ds = xr.Dataset(
        {
            "U": (("time", "lat"), np.ones((len(months), 3)), {"units": "m/s"}),
            "time_bnds": (("time", "nbnd"), bnds),
        },
        coords={
            "time": (
                "time",
                bnds[:, 1],
                {
                    "units": "days since 0001-01-01",
                    "calendar": "noleap",
                    "bounds": "time_bnds",
                },
            ),
            "lat": [-5.0, 0.0, 5.0],
        },
    )
    if extra:
        ds = ds.assign(extra)
    ds.to_netcdf(path)


def _data(tmp_path, compare_obs=False):
    """An AdfData with one test case and a baseline, files under `tmp_path`."""
    for case in (CASE, BASE):
        hist = tmp_path / case / "hist"
        hist.mkdir(parents=True)
        for year in (1, 2, 3):
            _write(hist / f"{case}.cam.h4.{year:04d}-01.nc", [(year, 1)])
        ts_dir = tmp_path / case / "ts"
        ts_dir.mkdir()
        _write(
            ts_dir / f"{case}.cam.h0a.U.000101-000102.nc",
            [(1, 1), (1, 2)],
            extra={"area": ("lat", [1.0, 2.0, 1.0])},
        )
        # TEM files, named for the climatology years (test 2-3, baseline 1-2):
        syr, eyr = (2, 3) if case == CASE else (1, 2)
        tem_dir = tmp_path / case / "tem"
        tem_dir.mkdir()
        _write(
            tem_dir / AdfData.tem_file_name(case, syr, eyr),
            [(syr, 1), (syr, 2)],
            extra={"UZM": (("time", "lat"), np.ones((2, 3)), {"units": "m/s"})},
        )
    _write(
        tmp_path / CASE / "tem" / "Obs.TEMdiag.nc",
        [(1, 1)],
        extra={"UZM": (("time", "lat"), np.ones((1, 3)), {"units": "m/s"})},
    )

    cam = {
        "cam_hist_loc": [str(tmp_path / CASE / "hist")],
        "cam_ts_loc": [str(tmp_path / CASE / "ts")],
        "cam_tem_loc": [str(tmp_path / CASE / "tem")],
    }
    base = {
        "cam_hist_loc": str(tmp_path / BASE / "hist"),
        "cam_ts_loc": str(tmp_path / BASE / "ts"),
        "cam_tem_loc": str(tmp_path / BASE / "tem"),
    }
    adf = SimpleNamespace(
        compare_obs=compare_obs,
        variable_defaults=DEFAULTS,
        get_cam_info=lambda key, required=False: cam.get(key),
        get_baseline_info=lambda key, required=False: base.get(key),
        hist_string={"test_hist_str": [["cam.h4"]], "base_hist_str": ["cam.h4"]},
        climo_yrs={
            "syears": [2],
            "eyears": [3],
            "syear_baseline": 1,
            "eyear_baseline": 2,
        },
        ts_yrs={
            "syears": [1],
            "eyears": [1],
            "syear_baseline": 1,
            "eyear_baseline": 1,
        },
        native_grid={},
        unstructured_plotting=False,
        debug_log=lambda msg: None,
    )
    data = AdfData.__new__(AdfData)
    data.adf = adf
    data.case_names = [CASE]
    data.ref_case_label = BASE
    data.ref_labels = {"U": BASE}
    return data


def _years(fils):
    return [int(Path(f).name.split(".")[-2][:4]) for f in fils]


# History files
# ------------------
def test_history_files_default_to_configured_stream_and_climo_years(tmp_path):
    data = _data(tmp_path)
    assert _years(data.get_history_files(CASE)) == [2, 3]
    assert _years(data.get_ref_history_files()) == [1, 2]


def test_history_files_years_and_stream_can_be_given(tmp_path):
    data = _data(tmp_path)
    assert _years(data.get_history_files(CASE, "cam.h4", 1, 1)) == [1]
    assert data.get_history_files(CASE, "cam.h1") == []


def test_history_dataset_spans_the_files(tmp_path):
    data = _data(tmp_path)
    ds = data.load_history_dataset(CASE)
    assert ds.sizes["time"] == 2
    assert "U" in ds and "time_bnds" in ds


def test_history_da_is_converted(tmp_path):
    data = _data(tmp_path)
    da = data.load_history_da(CASE, "U")
    assert float(da.max()) == 2.0
    assert da.attrs["units"] == "cm/s"
    raw = data.load_reference_history_da("U", apply_scaling=False)
    assert float(raw.max()) == 1.0
    assert raw.attrs["units"] == "m/s"


def test_no_history_files_for_observations(tmp_path):
    data = _data(tmp_path, compare_obs=True)
    with pytest.warns(UserWarning):
        assert data.get_ref_history_files() is None
    with pytest.warns(UserWarning):
        assert data.load_reference_history_dataset() is None


# Time series Datasets
# ------------------
def test_timeseries_dataset_converts_field_and_keeps_the_rest(tmp_path):
    data = _data(tmp_path)
    for ds in (
        data.load_timeseries_dataset(CASE, "U"),
        data.load_reference_timeseries_dataset("U"),
    ):
        assert float(ds["U"].max()) == 2.0
        assert ds["U"].attrs["units"] == "cm/s"
        assert ds["U"].attrs["transformed"] == 1
        # the rest of the Dataset is there for scripts that need it:
        assert "area" in ds
        # time moved to the middle of each month:
        assert ds.time.dt.month.values.tolist() == [1, 2]


def test_timeseries_dataset_without_scaling(tmp_path):
    data = _data(tmp_path)
    ds = data.load_timeseries_dataset(CASE, "U", apply_scaling=False)
    assert float(ds["U"].max()) == 1.0


def test_timeseries_dataset_missing_variable(tmp_path):
    data = _data(tmp_path)
    with pytest.warns(UserWarning):
        assert data.load_timeseries_dataset(CASE, "V") is None


# TEM files
# ------------------
def test_tem_files_are_named_for_the_climo_years(tmp_path):
    data = _data(tmp_path)
    assert [f.name for f in data.get_tem_file(CASE)] == [f"{CASE}.TEMdiag_2-3.nc"]
    assert [f.name for f in data.get_ref_tem_file()] == [f"{BASE}.TEMdiag_1-2.nc"]


def test_tem_da_is_converted_and_time_centred(tmp_path):
    data = _data(tmp_path)
    for da in (data.load_tem_da(CASE, "UZM"), data.load_reference_tem_da("UZM")):
        assert float(da.max()) == 2.0
        assert da.attrs["units"] == "cm/s"
        assert da.time.dt.month.values.tolist() == [1, 2]
    raw = data.load_tem_da(CASE, "UZM", apply_scaling=False)
    assert float(raw.max()) == 1.0


def test_observation_tem_file_uses_obs_scaling(tmp_path):
    data = _data(tmp_path, compare_obs=True)
    assert [f.name for f in data.get_ref_tem_file()] == ["Obs.TEMdiag.nc"]
    da = data.load_reference_tem_da("UZM")
    assert float(da.max()) == 3.0
    # observations keep the units they arrived with:
    assert da.attrs["units"] == "m/s"


def test_tem_field_not_in_file(tmp_path):
    data = _data(tmp_path)
    with pytest.warns(UserWarning):
        assert data.load_tem_da(CASE, "PZM") is None


def test_missing_tem_file(tmp_path):
    data = _data(tmp_path)
    data.case_names = [CASE, "other"]
    assert data.get_tem_file("other") == []
