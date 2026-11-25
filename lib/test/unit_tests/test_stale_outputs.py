"""Reusing ADF outputs only when their inputs have not changed since.

A climatology, regridded file or baseline file is reused on a later run when
it already exists and overwrite is off.  That is only right if the files it was
made from are not newer than it: otherwise (time series gridded again with
other weights, say) the plots would come from the old data while the tables,
which read the time series, come from the new.  These pin down
"newer_inputs"/"stale_note" in adf_file_utils and the regrid step's
"_reuse_existing".
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position
from adf_file_utils import newer_inputs, stale_note  # noqa: E402

# pylint: enable=import-error,wrong-import-position


def _file(path, age):
    """Write `path` and date it `age` seconds before a fixed time."""
    path.write_text("x")
    when = 1_800_000_000 - age
    os.utime(path, (when, when))
    return path


def test_up_to_date_output_has_no_newer_inputs(tmp_path):
    """Inputs older than the output: nothing to report."""
    out = _file(tmp_path / "climo.nc", age=10)
    ins = [_file(tmp_path / "ts1.nc", age=100), _file(tmp_path / "ts2.nc", age=20)]
    assert not newer_inputs(out, ins)


def test_newer_inputs_are_listed(tmp_path):
    """Only the inputs written after the output are returned."""
    out = _file(tmp_path / "climo.nc", age=10)
    old = _file(tmp_path / "ts1.nc", age=100)
    new = _file(tmp_path / "ts2.nc", age=1)
    assert newer_inputs(out, [old, new]) == [new]


def test_missing_inputs_are_ignored(tmp_path):
    """An input that is gone cannot make the output stale."""
    out = _file(tmp_path / "climo.nc", age=10)
    assert not newer_inputs(out, [tmp_path / "gone.nc"])
    assert not newer_inputs(out, None)


def test_stale_note_names_first_and_counts_rest():
    """The message names one file and how many more."""
    assert stale_note([Path("/a/ts1.nc")]) == "ts1.nc"
    assert stale_note([Path("/a/ts1.nc"), Path("b"), Path("c")]) == "ts1.nc and 2 more"


# The regrid step's reuse check (needs the science stack to import)
# -----------------------------------------------------------------
@pytest.fixture(name="reuse")
def _reuse():
    pytest.importorskip("xarray")
    sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))
    rvi = pytest.importorskip("regrid_and_vert_interp")
    return rvi._reuse_existing  # pylint: disable=protected-access


def test_reuse_when_up_to_date(tmp_path, reuse, capsys):
    """An up-to-date file with overwrite off is kept and skipped."""
    out = _file(tmp_path / "regridded.nc", age=10)
    assert reuse(
        out, lambda: [_file(tmp_path / "climo.nc", age=100)], False, "Regridded file"
    )
    assert out.is_file() and "already exists, skipping" in capsys.readouterr().out


def test_remade_when_input_is_newer(tmp_path, reuse, capsys):
    """A file older than its climo is removed so that it is made again."""
    out = _file(tmp_path / "regridded.nc", age=100)
    assert not reuse(
        out, lambda: [_file(tmp_path / "climo.nc", age=10)], False, "Regridded file"
    )
    assert not out.is_file()
    assert "older than its input (climo.nc)" in capsys.readouterr().out


def test_overwrite_always_remakes(tmp_path, reuse):
    """With overwrite on, even an up-to-date file is removed."""
    out = _file(tmp_path / "regridded.nc", age=10)
    assert not reuse(
        out, lambda: [_file(tmp_path / "climo.nc", age=100)], True, "Regridded file"
    )
    assert not out.is_file()


def test_missing_output_is_made(tmp_path, reuse):
    """No file yet: make it."""
    assert not reuse(tmp_path / "regridded.nc", list, False, "Regridded file")


# Native climo gridded in the regrid step: the gridding settings are recorded
# in the regridded file, and a change of them makes it out of date
# ----------------------------------------------------------------------------
SIG_A = {"adf_gridding_weights_file": "a.nc", "adf_gridding_method": "bilinear"}
SIG_B = {"adf_gridding_weights_file": "b.nc", "adf_gridding_method": "bilinear"}


def _nc(path, dims, attrs=None, age=None):
    """A small netCDF file with one field on `dims`, dated `age` seconds back."""
    xr = pytest.importorskip("xarray")
    np = pytest.importorskip("numpy")
    shape = tuple(2 for _ in dims)
    ds = xr.Dataset({"T": (dims, np.ones(shape))}, attrs=attrs or {})
    ds.to_netcdf(path)
    if age is not None:
        when = 1_800_000_000 - age
        os.utime(path, (when, when))
    return path


def test_signature_problem_names_the_setting():
    """Matching settings: no problem; another weights file or none recorded: say so."""
    utils = pytest.importorskip("adf_utils")
    assert utils.gridding_signature_problem(dict(SIG_A), SIG_A) is None
    problem = utils.gridding_signature_problem(dict(SIG_A), SIG_B)
    assert problem == "was made with another weights file ('a.nc', now 'b.nc')"
    assert utils.gridding_signature_problem({}, SIG_A) == (
        "does not record how it was gridded"
    )


def test_reuse_when_gridding_settings_match(tmp_path, reuse):
    """Same recorded settings and no newer input: reused."""
    out = _nc(tmp_path / "regridded.nc", ("lat", "lon"), SIG_A, age=10)
    assert reuse(out, list, False, "Regridded file", find_signature=lambda: SIG_A)


def test_remade_when_weights_file_changes(tmp_path, reuse, capsys):
    """Another weights file now: removed and made again, even with no newer input."""
    out = _nc(tmp_path / "regridded.nc", ("lat", "lon"), SIG_A, age=10)
    assert not reuse(out, list, False, "Regridded file", find_signature=lambda: SIG_B)
    assert not out.is_file()
    assert "another weights file ('a.nc', now 'b.nc')" in capsys.readouterr().out


def test_no_signature_check_when_nothing_is_gridded(tmp_path, reuse):
    """find_signature returning None (climo already lat/lon): only the age counts."""
    out = _nc(tmp_path / "regridded.nc", ("lat", "lon"), age=10)
    assert reuse(out, list, False, "Regridded file", find_signature=lambda: None)


@pytest.fixture(name="climo_signature")
def _climo_signature():
    pytest.importorskip("xarray")
    sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))
    rvi = pytest.importorskip("regrid_and_vert_interp")
    return rvi._climo_gridding_signature  # pylint: disable=protected-access


def _adf(native=True, unstructured_plotting=False):
    """A stand-in for the AdfDiag settings _climo_gridding_signature reads."""
    return SimpleNamespace(
        native_grid={"case": native, "base": native},
        unstructured_plotting=unstructured_plotting,
        latlon_wgt_files={"test_wgts_file": ["t.nc"], "baseline_wgts_file": "b.nc"},
        latlon_regrid_method={
            "test_regrid_method": ["bilinear"],
            "baseline_regrid_method": "conservative",
        },
        latlon_files={"test_latlon_file": [None], "baseline_latlon_file": "mask.nc"},
    )


def test_native_climo_has_a_signature(tmp_path, climo_signature):
    """Native climo of a native case plotted on lat/lon: its settings."""
    climo = [_nc(tmp_path / "climo.nc", ("time", "ncol"))]
    sig = climo_signature(_adf(), "case", 0, climo)
    assert sig["adf_gridding_weights_file"] == "t.nc"
    assert sig["adf_gridding_method"] == "bilinear"
    base = climo_signature(_adf(), "base", None, climo)
    assert base["adf_gridding_weights_file"] == "b.nc"
    assert base["adf_gridding_latlon_file"] == "mask.nc"


@pytest.mark.parametrize(
    "adf_kwargs, dims",
    [
        ({}, ("lat", "lon")),  # climo made from gridded time series
        ({"native": False}, ("time", "ncol")),  # not a native case
        ({"unstructured_plotting": True}, ("time", "ncol")),  # plotted on the mesh
    ],
    ids=["latlon-climo", "not-native", "on-mesh"],
)
def test_nothing_gridded_has_no_signature(tmp_path, climo_signature, adf_kwargs, dims):
    """No gridding in the regrid step: nothing to record or compare."""
    climo = [_nc(tmp_path / "climo.nc", dims)]
    assert climo_signature(_adf(**adf_kwargs), "case", 0, climo) is None


# The climo step
# --------------
@pytest.fixture(name="climo_check")
def _climo_check():
    pytest.importorskip("xarray")
    sys.path.append(str(Path(__file__).parents[3] / "scripts" / "averaging"))
    return pytest.importorskip("create_climo_files").climo_out_of_date


def test_climo_reused_when_made_from_these_files(tmp_path, climo_check):
    """Older time series and the same recorded files: reused."""
    ts_file = _file(tmp_path / "case.cam.h0a.T.000101-000512.nc", age=100)
    climo = _nc(
        tmp_path / "climo.nc",
        ("lat", "lon"),
        {"time_series_files": str(ts_file)},
        age=10,
    )
    assert climo_check(climo, [ts_file]) is None


def test_climo_remade_when_time_series_newer(tmp_path, climo_check):
    """A time series made again since: out of date."""
    climo = _nc(tmp_path / "climo.nc", ("lat", "lon"), age=100)
    ts_file = _file(tmp_path / "case.cam.h0a.T.000101-000512.nc", age=10)
    assert "older than its time series" in climo_check(climo, [ts_file])


def test_climo_remade_when_made_from_other_files(tmp_path, climo_check):
    """Made from the gridded copy, now read from the native file: out of date."""
    native = _file(tmp_path / "case.cam.h0a.T.000101-000512.nc", age=200)
    gridded = tmp_path / "gridded" / "case.cam.h0a.T.000101-000512_gridded.nc"
    climo = _nc(
        tmp_path / "climo.nc",
        ("lat", "lon"),
        {"time_series_files": str(gridded)},
        age=10,
    )
    assert "made from other time series files" in climo_check(climo, [native])


def test_climo_without_record_is_judged_by_age_only(tmp_path, climo_check):
    """A climo file from before the files were recorded is not remade for that."""
    ts_file = _file(tmp_path / "case.cam.h0a.T.000101-000512.nc", age=100)
    climo = _nc(tmp_path / "climo.nc", ("lat", "lon"), age=10)
    assert climo_check(climo, [ts_file]) is None
