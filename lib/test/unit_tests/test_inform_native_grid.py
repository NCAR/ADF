"""
Tests for lib/inform_native_grid.py, the step that grids native-grid CAM
history files to lat/lon for the INFORM diagnostics.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

import cam_era5_functions as cfun  # noqa: E402
import inform_native_grid as ingrid  # noqa: E402


class _Opened(Exception):
    """Raised in place of opening the files, carrying the list asked for."""


def _files_load_cam_files_opens(monkeypatch, cam_dir, file_sel, tmin, tmax):
    def fake_open(paths, **_kwargs):
        raise _Opened(list(paths))

    monkeypatch.setattr(cfun.xr, "open_mfdataset", fake_open)
    try:
        cfun.load_cam_files(cam_dir, file_sel, tmin, tmax, campaign="SOCRATES")
    except _Opened as opened:
        return opened.args[0]
    except RuntimeError:
        return []
    raise AssertionError("load_cam_files did not open any files")


@pytest.mark.parametrize(
    "tmin, tmax",
    [
        (datetime(2018, 1, 15, 22), datetime(2018, 1, 16, 5)),  # inside the files
        (datetime(2018, 1, 15, 0), datetime(2018, 1, 15, 1)),  # at the first file
        (datetime(2018, 1, 17, 18), datetime(2018, 1, 18, 6)),  # past the last file
        (datetime(2018, 2, 1), datetime(2018, 2, 2)),  # no files at all
    ],
)
def test_window_files_match_load_cam_files(tmp_path, monkeypatch, tmin, tmax):
    """window_files picks the same history files load_cam_files reads."""
    # Six-hourly files over three days, another stream, and a file without
    # a date, which both must ignore:
    for day in (15, 16, 17):
        for sec in (0, 21600, 43200, 64800):
            (tmp_path / f"case.cam.h1i.2018-01-{day}-{sec:05d}.nc").touch()
            (tmp_path / f"case.cam.h0a.2018-01-{day}-{sec:05d}.nc").touch()
    (tmp_path / "case.cam.h1i.rest.nc").touch()

    expected = _files_load_cam_files_opens(monkeypatch, tmp_path, "cam.h1i", tmin, tmax)
    assert ingrid.window_files(tmp_path, "cam.h1i", tmin, tmax) == expected


def test_lat_lon_case_is_read_in_place(tmp_path):
    """A case already on lat/lon is not gridded: its own directory is used."""
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {"T": (("time", "lat", "lon"), np.zeros((1, 2, 3)))},
        coords={"time": [0.0], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
    ).to_netcdf(tmp_path / "case.cam.h1i.2018-01-15-00000.nc")

    out = ingrid.grid_case_for_window(
        tmp_path, "cam.h1i", datetime(2018, 1, 15), datetime(2018, 1, 15, 1),
        wgts_file=None, out_dir=tmp_path / "gridded",
    )
    assert out == tmp_path
    assert not (tmp_path / "gridded").exists()


def test_native_case_without_weights_file_fails_clearly(tmp_path):
    """A native-grid case with no weights file stops with a message saying so."""
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {"T": (("time", "ncol"), np.zeros((1, 4)))}, coords={"time": [0.0]}
    ).to_netcdf(tmp_path / "case.cam.h1i.2018-01-15-00000.nc")

    with pytest.raises(ValueError, match="weights_file"):
        ingrid.grid_case_for_window(
            tmp_path, "cam.h1i", datetime(2018, 1, 15), datetime(2018, 1, 15, 1),
            wgts_file=None, out_dir=tmp_path / "gridded",
        )


def test_crop_keeps_the_region_plus_margin():
    """Cropping keeps the campaign box plus REGION_MARGIN, in 0..360 longitude."""
    import numpy as np
    import xarray as xr

    lat = np.arange(-90.0, 90.1, 1.0)
    lon = np.arange(0.0, 360.0, 1.25)
    ds = xr.Dataset({"T": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
                    coords={"lat": lat, "lon": lon})

    out = ingrid.crop_to_region(ds, ingrid.CAMPAIGN_REGIONS["SOCRATES"])
    m = ingrid.REGION_MARGIN
    assert float(out.lat.min()) >= -62 - m and float(out.lat.max()) <= -42 + m
    assert float(out.lat.min()) <= -62 and float(out.lat.max()) >= -42
    assert float(out.lon.min()) <= 135 and float(out.lon.max()) >= 165
    assert float(out.lon.max()) <= 165 + m


def test_gridded_dir_changes_with_region_weights_and_method(tmp_path):
    """Gridded files made another way are never reused."""
    region = ingrid.CAMPAIGN_REGIONS["SOCRATES"]
    base = ingrid.gridded_dir(tmp_path, "SOCRATES", "case", "cam.h0i", region,
                              "/w/map_conserve.nc", "conservative")
    assert base != ingrid.gridded_dir(tmp_path, "SOCRATES", "case", "cam.h0i",
                                      ingrid.CAMPAIGN_REGIONS["CSET"],
                                      "/w/map_conserve.nc", "conservative")
    assert base != ingrid.gridded_dir(tmp_path, "SOCRATES", "case", "cam.h0i", region,
                                      "/w/map_bilin.nc", "conservative")
    assert base != ingrid.gridded_dir(tmp_path, "SOCRATES", "case", "cam.h0i", region,
                                      "/w/map_conserve.nc", "bilinear")


def _inform_model_analysis():
    sys.path.insert(0, str(Path(__file__).parents[3] / "scripts" / "plotting"))
    try:
        import inform_model_analysis
    except ImportError as err:
        pytest.skip(f"cannot import inform_model_analysis: {err}")
    return inform_model_analysis


def _hist_file(tmp_path, names):
    import numpy as np
    import xarray as xr

    path = tmp_path / "case.cam.h0i.2018-01-15-00000.nc"
    xr.Dataset({n: ("x", np.zeros(2, dtype="float32")) for n in names}).to_netcdf(path)
    return path


def test_missing_cam_vars_lists_what_a_case_lacks(tmp_path):
    """A case without e.g. the CLUBB fields is reported, so it can be skipped."""
    ima = _inform_model_analysis()
    have = [v for v in ima.REQUIRED_CAM_VARS if v != "WP2_ZT_CLUBB"]
    assert ima.missing_cam_vars(_hist_file(tmp_path, have)) == ["WP2_ZT_CLUBB"]


def test_missing_cam_vars_ignores_regional_suffixes(tmp_path):
    """Regionally subset output (T_130e_to_170e_35s_to_65s) has what it needs."""
    ima = _inform_model_analysis()
    names = [v if v in ("hyam", "hybm") else f"{v}_130e_to_170e_35s_to_65s"
             for v in ima.REQUIRED_CAM_VARS]
    assert ima.missing_cam_vars(_hist_file(tmp_path, names)) == []
