"""Interpolation driven by the model's own 3-D pressure field (PMID / PINT).

PMID takes precedence over rebuilding pressure from PS and the hybrid
coefficients. vert_remap interpolates linearly in log(p), so a field that is
linear in log(p) must come back exactly -- that is what these checks use.
"""
import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))
sys.path.append(str(Path(__file__).parents[2]))

from regrid_and_vert_interp import (  # noqa: E402
    DEFAULT_PLEVS_Pa,
    _interp_with_pressure_field,
)

A, B = 3.0, -1.5  # data = A + B*log(p)


def _column(nlev=20, p_bot=96000.0, p_top=200.0):
    """Ascending pressures, top first, as CAM stores them."""
    return np.logspace(np.log10(p_top), np.log10(p_bot), nlev)


def _build(level_dim="lev", nlat=4, nlon=6, ntime=2, scale=None):
    p_col = _column()
    nlev = p_col.size
    #Optionally scale each column's pressure, so columns end at different depths:
    if scale is None:
        scale = np.ones((nlat, nlon))
    pres = p_col[None, :, None, None] * scale[None, None, :, :]
    pres = np.broadcast_to(pres, (ntime, nlev, nlat, nlon)).copy()
    data = A + B * np.log(pres)
    dims = ["time", level_dim, "lat", "lon"]
    coords = {"time": np.arange(ntime, dtype=float),
              level_dim: np.arange(nlev, dtype=float),
              "lat": np.linspace(-45.0, 45.0, nlat),
              "lon": np.arange(nlon, dtype=float)}
    return (xr.DataArray(data, dims=dims, coords=coords, name="X"),
            xr.DataArray(pres, dims=dims, coords=coords, name="PMID"))


def test_log_linear_field_is_reproduced_exactly():
    da, pres = _build()
    out = _interp_with_pressure_field(da, pres)

    assert "lev" in out.dims
    #Output levels are reported in hPa:
    assert np.allclose(np.sort(out["lev"].values), np.sort(DEFAULT_PLEVS_Pa / 100.0))

    lev_pa = out["lev"].values * 100.0
    p_col = _column()
    inside = (lev_pa >= p_col.min()) & (lev_pa <= p_col.max())

    got = out.isel(time=0, lat=0, lon=0).values
    expected = A + B * np.log(lev_pa)
    assert np.allclose(got[inside], expected[inside]), "log-linear field not recovered"


def test_levels_outside_the_column_are_nan_not_clamped():
    """np.interp clamps; the hybrid path returns NaN. Keep the NaN semantics."""
    da, pres = _build()
    out = _interp_with_pressure_field(da, pres)
    lev_pa = out["lev"].values * 100.0
    p_col = _column()
    outside = (lev_pa < p_col.min()) | (lev_pa > p_col.max())

    assert outside.sum() > 0, "test needs at least one out-of-range level"
    got = out.isel(time=0, lat=0, lon=0).values
    assert np.all(np.isnan(got[outside])), "out-of-range levels were clamped, not masked"


def test_masking_is_per_column():
    """A shallower column must lose more levels than a deep one."""
    nlat, nlon = 3, 2
    scale = np.ones((nlat, nlon))
    scale[0, 0] = 0.25   # this column stops well short of the surface
    da, pres = _build(nlat=nlat, nlon=nlon, scale=scale)
    out = _interp_with_pressure_field(da, pres)

    shallow = int(out.isel(time=0, lat=0, lon=0).isnull().sum())
    deep = int(out.isel(time=0, lat=1, lon=1).isnull().sum())
    assert shallow > deep, f"expected more NaN in the shallow column ({shallow} vs {deep})"


def test_interface_data_on_ilev_works():
    da, pres = _build(level_dim="ilev")
    out = _interp_with_pressure_field(da, pres)
    #Renamed on the way in; the result is on pressure levels called "lev":
    assert "lev" in out.dims and "ilev" not in out.dims

    lev_pa = out["lev"].values * 100.0
    p_col = _column()
    inside = (lev_pa >= p_col.min()) & (lev_pa <= p_col.max())
    got = out.isel(time=0, lat=0, lon=0).values
    expected = A + B * np.log(lev_pa)
    assert np.allclose(got[inside], expected[inside])


if __name__ == "__main__":
    test_log_linear_field_is_reproduced_exactly()
    test_levels_outside_the_column_are_nan_not_clamped()
    test_masking_is_per_column()
    test_interface_data_on_ilev_works()
    print("all pressure-field interpolation checks passed")
