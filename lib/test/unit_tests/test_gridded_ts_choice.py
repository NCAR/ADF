"""Which time series a native-grid case is read from: native or "gridded".

When native-grid data is plotted on lat/lon, the ADF grids the time series it
makes into a "gridded" sub-directory of 'cam_ts_loc'.  A gridded copy is read
only when it is up to date for the variable: every native file needed for the
case's years has a copy that is not older than it and records the gridding
settings this run would use.  Otherwise the native files are read and nothing
is written (with pre-made time series, 'cam_ts_loc' may be someone else's
folder).  These pin down that choice (AdfData.timeseries_dir), the helpers
behind it in adf_utils, and that the recursive file search never picks up the
gridded copies by itself.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position
import adf_utils as utils  # noqa: E402
from adf_dataset import AdfData  # noqa: E402

# pylint: enable=import-error,wrong-import-position

CASE = "native_case"
BASE = "native_base"
SETTINGS = ("map_ne30pg3_to_f09_bilin.nc", "bilinear", None, False)
SIGNATURE = utils.gridding_signature(*SETTINGS)


def _native(path):
    """A tiny native-grid time series file."""
    xr.Dataset({"T": (("time", "ncol"), np.ones((1, 4)))}).to_netcdf(path)


def _gridded(native_file, signature=SIGNATURE, age=0):
    """The gridded copy of `native_file`, `age` seconds older than it."""
    gridded_file = utils.gridded_ts_name(native_file)
    gridded_file.parent.mkdir(exist_ok=True)
    ds = xr.Dataset({"T": (("time", "lat", "lon"), np.ones((1, 2, 2)))})
    ds.attrs.update(signature or {})
    ds.to_netcdf(gridded_file)
    mtime = Path(native_file).stat().st_mtime + (10 - age)
    os.utime(gridded_file, (mtime, mtime))
    return gridded_file


def _data(settings=SETTINGS, unstructured_plotting=False):
    """An AdfData whose test case and baseline are both on a native grid."""
    messages = []
    adf = SimpleNamespace(
        native_grid={CASE: True, BASE: True},
        unstructured_plotting=unstructured_plotting,
        ts_yrs={"syears": [1], "eyears": [2], "syear_baseline": 1, "eyear_baseline": 2},
        gridding_settings=lambda case_idx, var, baseline=False: settings,
        debug_log=messages.append,
    )
    data = AdfData.__new__(AdfData)
    data.adf = adf
    data.case_names = [CASE]
    data.ref_case_label = BASE
    data._gridded_ts_choice = {}  # pylint: disable=protected-access
    return data, messages


def _ts_dir(tmp_path, case=CASE, var="T"):
    """A time series directory with one native file for years 1-2."""
    ts_dir = tmp_path / case
    ts_dir.mkdir(exist_ok=True)
    native_file = ts_dir / f"{case}.cam.h0a.{var}.000101-000212.nc"
    _native(native_file)
    return ts_dir, native_file


# The helpers
# -----------
def test_signature_is_strings_for_file_attributes():
    """Every value is a string, so it can be written as a netCDF attribute."""
    sig = utils.gridding_signature(Path("w.nc"), None, None, True)
    assert sig == {
        "adf_gridding_weights_file": "w.nc",
        "adf_gridding_method": "",
        "adf_gridding_latlon_file": "",
        "adf_gridding_keep_precision": "true",
    }


def test_up_to_date_copy_has_no_problem(tmp_path):
    """A newer copy with the same settings can stand in for its native file."""
    _, native_file = _ts_dir(tmp_path)
    gridded_file = _gridded(native_file)
    assert utils.gridded_copy_problem(native_file, gridded_file, SIGNATURE) is None


@pytest.mark.parametrize(
    "make_copy, expected",
    [
        (utils.gridded_ts_name, "no gridded copy"),
        (lambda native: _gridded(native, age=20), "older than the native file"),
        (lambda native: _gridded(native, signature=None), "does not record"),
        (
            lambda native: _gridded(
                native,
                signature=utils.gridding_signature("other.nc", "bilinear", None, False),
            ),
            "another weights file ('other.nc', now 'map_ne30pg3_to_f09_bilin.nc')",
        ),
    ],
    ids=["missing", "older", "no-record", "other-weights"],
)
def test_stale_copies_say_why(tmp_path, make_copy, expected):
    """Missing, older, unrecorded or differently gridded copies are refused."""
    _, native_file = _ts_dir(tmp_path)
    gridded_file = make_copy(native_file)
    problem = utils.gridded_copy_problem(native_file, gridded_file, SIGNATURE)
    assert problem is not None and expected in problem


def test_recursive_search_skips_gridded_copies(tmp_path):
    """With no native file at the top, the gridded copies are not found instead."""
    ts_dir = tmp_path / "ts"
    (ts_dir / "gridded").mkdir(parents=True)
    _native(ts_dir / "gridded" / f"{CASE}.cam.h0a.T.000101-000212_gridded.nc")
    (ts_dir / "atm" / "proc").mkdir(parents=True)
    nested = ts_dir / "atm" / "proc" / f"{CASE}.cam.h0a.T.000101-000212.nc"
    _native(nested)
    assert utils.find_ts_files(ts_dir, f"{CASE}.*.T.*nc") == [nested]


# The choice in AdfData
# ---------------------
def test_up_to_date_copy_is_read(tmp_path):
    """Up-to-date copies for every needed native file: read 'gridded'."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(native_file)
    data, messages = _data()
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir / "gridded"
    assert not messages


def test_other_weights_fall_back_to_native(tmp_path):
    """Copies made with another weights file: read the native files, say why."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(native_file)
    data, messages = _data(settings=("map_conserve.nc", "conservative", None, False))
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir
    assert len(messages) == 1 and "another weights file" in messages[0]
    # Asked again (e.g. once per history stream), it neither re-checks nor repeats:
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir
    assert len(messages) == 1


def test_missing_variable_falls_back_per_variable(tmp_path):
    """A variable with no copy is read natively; others still use 'gridded'."""
    ts_dir, native_t = _ts_dir(tmp_path, var="T")
    _, _native_u = _ts_dir(tmp_path, var="U")
    _gridded(native_t)
    data, messages = _data()
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir / "gridded"
    assert data.timeseries_dir(CASE, ts_dir, "U") == ts_dir
    assert (
        len(messages) == 1 and "'U'" in messages[0] and "no gridded copy" in messages[0]
    )


def test_other_years_fall_back_to_native(tmp_path):
    """A copy only for other years: the native files for these years are read."""
    ts_dir, native_file = _ts_dir(tmp_path)  # years 1-2, as the case wants
    old = ts_dir / f"{CASE}.cam.h0a.T.000301-000412.nc"
    _native(old)
    _gridded(old)  # a copy exists, but for years 3-4
    native_file.touch()
    data, _ = _data()
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir


def test_baseline_uses_baseline_years(tmp_path):
    """The baseline is checked with its own years and settings."""
    ts_dir, native_file = _ts_dir(tmp_path, case=BASE)
    _gridded(native_file)
    data, _ = _data()
    assert data.timeseries_dir(BASE, ts_dir, "T", baseline=True) == ts_dir / "gridded"


def test_unstructured_plotting_reads_native(tmp_path):
    """Plotting on the mesh always reads the native files."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(native_file)
    data, _ = _data(unstructured_plotting=True)
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir


# Making the copies (AdfDiag.grid_native_time_series)
# ---------------------------------------------------
def _grid(monkeypatch, ts_dir, settings=SETTINGS):
    """Run grid_native_time_series on `ts_dir` with a stand-in for the gridding.

    Returns the variables that were gridded.
    """
    adf_diag = pytest.importorskip("adf_diag")
    gridded_vars = []

    def fake_grid(_ts_ds, var, **_kwargs):
        gridded_vars.append(var)
        return xr.Dataset({var: (("time", "lat", "lon"), np.ones((1, 2, 2)))})

    monkeypatch.setattr(adf_diag.utils, "grid_to_latlon", fake_grid)
    monkeypatch.setattr(adf_diag.utils, "chunk_native_in_time", lambda ds, var: ds)
    monkeypatch.setattr(adf_diag.utils, "coarser_grid", lambda a, b: "same")
    diag = SimpleNamespace(
        native_grid={CASE: True},
        unstructured_plotting=False,
        latlon_wgt_files={"test_wgts_file": [settings[0]]},
        gridding_settings=lambda case_idx, var, baseline=False: settings,
        model_component="atm",
    )
    adf_diag.AdfDiag.grid_native_time_series(diag, CASE, 0, ts_dir, ["cam.h0a"], False)
    return gridded_vars


def test_stale_copy_is_gridded_again_and_recorded(tmp_path, monkeypatch):
    """With overwrite off, a copy made with other settings is still remade."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(
        native_file, signature=utils.gridding_signature("old.nc", None, None, False)
    )
    assert _grid(monkeypatch, ts_dir) == ["T"]
    with xr.open_dataset(utils.gridded_ts_name(native_file)) as ds:
        assert {key: ds.attrs[key] for key in SIGNATURE} == SIGNATURE


def test_up_to_date_copy_is_kept(tmp_path, monkeypatch):
    """With overwrite off, an up-to-date copy is not gridded again."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(native_file)
    assert not _grid(monkeypatch, ts_dir)


# Statistics read the native files (skip_gridded_copy=True)
# ----------------------------------------------
def test_native_reads_native_even_with_an_up_to_date_copy(tmp_path):
    """Tables and global means average on the native grid, so skip the copy."""
    ts_dir, native_file = _ts_dir(tmp_path)
    _gridded(native_file)
    data, messages = _data()
    assert data.timeseries_dir(CASE, ts_dir, "T", skip_gridded_copy=True) == ts_dir
    assert not messages
    # The plots still read the copy, and the two choices do not mix:
    assert data.timeseries_dir(CASE, ts_dir, "T") == ts_dir / "gridded"
    assert data.timeseries_dir(CASE, ts_dir, "T", skip_gridded_copy=True) == ts_dir
