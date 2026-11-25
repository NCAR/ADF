"""Fields are compared on the coarser of the two grids.

The finer grid is regridded onto the coarser one, so no field is interpolated
to a resolution it does not have.  The regridding stage used to put every test
case onto the reference's grid, however fine that was.  Now, when a test case's
lat/lon grid is clearly coarser than the reference's, the reference is
regridded onto the case's grid instead (conservatively, skipping missing
cells), and the case keeps its own grid.  On the same grid nothing is
regridded at all.
"""

# The tests drive the regridding script's private helpers directly:
# pylint: disable=protected-access

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

# Skip rather than fail collection when the science stack is absent, as in a
# local run outside the ADF environment.  CI runs these in the full ADF
# environment (.github/workflows/ADF_unit_tests.yaml), where a skip is a failure.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")
pytest.importorskip("xesmf")

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))

# pylint: disable-next=import-error,wrong-import-position,wrong-import-order
import regrid_and_vert_interp as rvi  # noqa: E402

# pylint: disable-next=import-error,wrong-import-position
import adf_utils as utils  # noqa: E402


def _grid(nlat, nlon, values=None, name="T"):
    """A global lat/lon field with cell centres half a cell from the poles."""
    dlat, dlon = 180.0 / nlat, 360.0 / nlon
    lat = -90.0 + dlat * (np.arange(nlat) + 0.5)
    lon = dlon * (np.arange(nlon) + 0.5)
    if values is None:
        values = np.arange(nlat * nlon, dtype=float).reshape(nlat, nlon)
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name=name,
        attrs={"units": "K"},
    )


def _area_mean(da):
    """Area-weighted mean of a global lat/lon field (cos(lat) weights)."""
    wgt = np.cos(np.deg2rad(da["lat"])) * xr.ones_like(da)
    wgt = wgt.where(da.notnull())
    return float((da * wgt).sum() / wgt.sum())


def test_mean_cell_area_of_global_grids():
    """A global lat/lon grid's mean cell is the sphere over its cell count."""
    assert utils.mean_cell_area(_grid(192, 288)) == pytest.approx(
        4 * np.pi / (192 * 288), rel=1e-6
    )
    native = xr.Dataset({"T": ("ncol", np.zeros(21600))})
    assert utils.mean_cell_area(native) == pytest.approx(4 * np.pi / 21600)


def test_mean_cell_area_of_a_regional_grid():
    """A regional grid gets its own cell size, not the sphere's share."""
    regional = xr.DataArray(
        np.zeros((10, 10)),
        dims=("lat", "lon"),
        coords={"lat": np.arange(10) + 0.5, "lon": np.arange(10) + 0.5},
    )
    one_degree = np.deg2rad(1.0) * np.diff(np.sin(np.deg2rad([0.0, 10.0])))[0] / 10
    assert utils.mean_cell_area(regional) == pytest.approx(one_degree, rel=1e-6)


def test_coarser_grid():
    """f19 is coarser than f09; ne30pg3 (21,600 cells) is coarser than f09."""
    f09, f19 = _grid(192, 288), _grid(96, 144)
    ne30pg3 = xr.Dataset({"T": ("ncol", np.zeros(21600))})
    assert utils.coarser_grid(f19, f09) == "test"
    assert utils.coarser_grid(f09, f19) == "ref"
    assert utils.coarser_grid(ne30pg3, f09) == "test"


def test_close_resolutions_are_a_tie():
    """Mean cell areas within 10% count as the same resolution."""
    assert utils.coarser_grid(_grid(192, 288), _grid(192, 288)) is None
    assert utils.coarser_grid(_grid(180, 360), _grid(172, 360)) is None
    assert utils.coarser_grid(_grid(4, 8), xr.DataArray(np.zeros(3), dims="x")) is None


def test_same_latlon_grid():
    """Same centres make the same grid; another resolution does not."""
    assert utils.same_latlon_grid(_grid(4, 8), _grid(4, 8, values=np.ones((4, 8))))
    assert not utils.same_latlon_grid(_grid(4, 8), _grid(2, 4))


def test_same_grid_is_not_regridded(tmp_path):
    """A field already on the target grid comes back unchanged.

    Conservative regridding of a grid onto itself still altered the polar rows
    (f09 to f09: 288 cells), so nothing is regridded at all.
    """
    src = _grid(4, 8)
    out = rvi._handle_horizontal_regridding(src, src.to_dataset(), tmp_path)
    assert out is src


def _adf(ref_file, compare_obs=False):
    """An AdfDiag stand-in that knows only the reference's regridded file."""
    return SimpleNamespace(
        user="tester",
        compare_obs=compare_obs,
        data=SimpleNamespace(
            get_ref_regrid_file=lambda field: [str(ref_file)],
            ref_var_nam={"T": "T"},
        ),
    )


def test_reference_goes_onto_a_coarser_case_grid(tmp_path):
    """The finer reference is regridded onto the case's grid, keeping its mean.

    The file is named for the case and its years, keeps the variable's name
    and attributes, and AdfData picks it for that case.
    """
    fine = _grid(8, 16)
    ref_file = tmp_path / "base_T_baseline_0001-0005.nc"
    fine.to_dataset().to_netcdf(ref_file)
    coarse = _grid(4, 8)

    assert rvi._write_reference_on_case_grid(
        _adf(ref_file), "case", "T", "base", coarse, tmp_path, False, (1, 5)
    )
    out_file = tmp_path / "base_case_T_reference_on_case_grid_0001-0005.nc"
    out = xr.open_dataset(out_file)["T"]
    assert out.sizes == {"lat": 4, "lon": 8}
    assert out.attrs["units"] == "K"
    assert _area_mean(out) == pytest.approx(_area_mean(fine), rel=1e-3)


def test_missing_fine_cells_are_left_out(tmp_path):
    """A coarse cell partly over missing cells is the mean of the cells with data.

    Below-ground cells on a pressure level, or masked cells, must not pull the
    coarse cell towards zero.
    """
    values = np.full((8, 16), 5.0)
    values[0, 0] = np.nan
    ref_file = tmp_path / "ref.nc"
    _grid(8, 16, values=values).to_dataset().to_netcdf(ref_file)

    rvi._write_reference_on_case_grid(
        _adf(ref_file), "case", "T", "base", _grid(4, 8), tmp_path, False, (1, 5)
    )
    out = xr.open_dataset(tmp_path / "base_case_T_reference_on_case_grid_0001-0005.nc")
    np.testing.assert_allclose(out["T"].values, 5.0)


def test_mostly_missing_coarse_cell_stays_missing(tmp_path):
    """A coarse cell with data in under half of it is left missing.

    In a masked ocean field (SST, sea ice), a coastal cell with only a sliver
    of ocean would otherwise take that sliver's value for the whole cell.
    """
    values = np.full((8, 16), 5.0)
    # Three of the four fine cells under coarse cell (2, 3) are missing:
    values[4, 6] = values[4, 7] = values[5, 6] = np.nan
    ref_file = tmp_path / "ref.nc"
    _grid(8, 16, values=values).to_dataset().to_netcdf(ref_file)

    rvi._write_reference_on_case_grid(
        _adf(ref_file), "case", "T", "base", _grid(4, 8), tmp_path, False, (1, 5)
    )
    out = xr.open_dataset(tmp_path / "base_case_T_reference_on_case_grid_0001-0005.nc")
    assert np.isnan(out["T"].values[2, 3])
    assert np.isfinite(np.delete(out["T"].values.ravel(), 2 * 8 + 3)).all()


def test_existing_file_is_kept(tmp_path):
    """Without overwrite, an existing file is reused, as the other outputs are."""
    out_file = tmp_path / "base_case_T_reference_on_case_grid_0001-0005.nc"
    out_file.write_text("old")
    assert rvi._write_reference_on_case_grid(
        _adf(tmp_path / "missing.nc"),
        "case",
        "T",
        "base",
        _grid(4, 8),
        tmp_path,
        False,
        (1, 5),
    )
    assert out_file.read_text() == "old"
