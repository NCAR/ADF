"""Percent differences: blank where the baseline is ~zero, and a robust headline mean.

A baseline at or near zero makes (test - base) / |base| arbitrarily large.  In
C01 (f09 against OAFlux) the % Diff panel printed "Mean: inf" for LHFLX (a zero
baseline) and a mean of -1e11 % for SHFLX (a baseline of 7e-15 W/m2), where the
area means differ by 43%.  So every percent difference goes through
utils.pct_change, which blanks cells whose |baseline| is below 1% of the
field's area-mean |baseline|, and the panel's Mean is the percent change of
the area means (utils.pct_change_of_means).
"""

import sys
from pathlib import Path

import pytest  # pylint: disable=import-error

# Skip rather than fail collection when the science stack is absent, as in a
# local run outside the ADF environment.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable-next=import-error,wrong-import-position,wrong-import-order
import adf_utils as utils


def _pair():
    """Test and baseline profiles with a zero, a 0/0 and a tiny baseline value."""
    lat = [-10.0, 0.0, 10.0, 20.0]
    kw = {"dims": "lat", "coords": {"lat": lat}, "name": "LHFLX"}
    test = xr.DataArray([2.0, 1.0, 0.0, 3.0], **kw)
    base = xr.DataArray([1.0, 0.0, 0.0, 1e-9], **kw)
    return test, base


def _map(test_vals, base_vals):
    """2 x 2 lat/lon test and baseline maps."""
    coords = {"lat": [-30.0, 30.0], "lon": [0.0, 180.0]}
    test = xr.DataArray(
        np.array(test_vals, float), dims=("lat", "lon"), coords=coords, name="X"
    )
    base = xr.DataArray(
        np.array(base_vals, float), dims=("lat", "lon"), coords=coords, name="X"
    )
    return test, base


def test_zero_and_tiny_baseline_are_blank():
    """A zero baseline (±inf, 0/0) and one far below the field's mean are NaN."""
    test, base = _pair()
    pct = utils.array_diff(test, base, percent=True)
    assert not np.isinf(pct.values).any()
    assert pct.values[0] == pytest.approx(100.0)
    assert np.isnan(pct.values[1:]).all()
    assert pct.attrs["units"] == "%"


def test_min_frac_zero_keeps_tiny_baseline():
    """min_frac=0 blanks only the zero baseline."""
    test, base = _pair()
    pct = utils.array_diff(test, base, percent=True, min_frac=0.0)
    assert np.isnan(pct.values[1:3]).all()
    assert np.isfinite(pct.values[3]) and pct.values[3] > 1e9


def test_threshold_is_relative_to_area_mean():
    """The cut-off is min_frac times the cos(lat)-weighted mean |base|, in any units."""
    # mean |base| = 1 (equal cos weights at ±30), so the cut-off at 1% is 0.01:
    test, base = _map([[2.0, 2.0], [2.0, 2.0]], [[1.98, 0.011], [0.009, 2.0]])
    for scale in (1.0, 1e-6, 1e6):
        pct = utils.pct_change(test * scale, base * scale)
        assert np.isfinite(pct.values[0, 0]) and np.isfinite(pct.values[0, 1])
        assert np.isnan(pct.values[1, 0])
        assert pct.values[1, 1] == pytest.approx(0.0)


def test_negative_baseline_uses_its_magnitude():
    """A negative baseline keeps the sign of the change (test above base is positive)."""
    test, base = _map([[-1.0, 1.0], [1.0, 1.0]], [[-2.0, 1.0], [1.0, 1.0]])
    assert utils.pct_change(test, base).values[0, 0] == pytest.approx(50.0)


def test_lat_lon_fields_are_aligned():
    """Like `case - base`, the percent change is on the common coordinates."""
    test, base = _pair()
    pct = utils.pct_change(test, base.isel(lat=slice(0, 2)))
    assert list(pct["lat"].values) == [-10.0, 0.0]


def test_fill_nan_replaces_the_blanked_points():
    """fill_nan still fills the blanked points."""
    test, base = _pair()
    pct = utils.array_diff(test, base, percent=True, fill_nan=0.0)
    assert list(pct.values) == pytest.approx([100.0, 0.0, 0.0, 0.0])


def test_plain_difference_unchanged():
    """The plain difference is not masked."""
    test, base = _pair()
    dif = utils.array_diff(test, base)
    assert list(dif.values) == pytest.approx([1.0, 1.0, 0.0, 3.0 - 1e-9])


def test_pct_change_of_means():
    """Percent change of the cos(lat)-weighted means over cells where both have data."""
    # a tiny baseline cell no longer dominates: the mean of the per-cell % would be ~1e16
    test, base = _map([[2.0, 3.0], [1.0, np.nan]], [[1.0, 1e-15], [1.0, 5.0]])
    expected = (
        100.0 * ((2.0 + 3.0 + 1.0) / 3 - (1.0 + 1e-15 + 1.0) / 3) / ((2.0 + 1e-15) / 3)
    )
    cos_lat = np.cos(np.deg2rad(test["lat"]))
    for wgt in (None, cos_lat):
        assert utils.pct_change_of_means(test, base, weights=wgt) == pytest.approx(
            expected
        )


def test_pct_change_of_means_uses_the_weights():
    """Weights given only along lat broadcast over lon."""
    test, base = _map([[2.0, 2.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]])
    wgt = xr.DataArray([3.0, 1.0], dims="lat", coords={"lat": test["lat"]})
    assert utils.pct_change_of_means(test, base, weights=wgt) == pytest.approx(75.0)


def test_pct_change_of_means_zero_baseline_mean():
    """A zero baseline mean has no percent change."""
    test, base = _map([[1.0, 1.0], [1.0, 1.0]], [[1.0, -1.0], [1.0, -1.0]])
    assert np.isnan(utils.pct_change_of_means(test, base))


def test_native_mesh_keeps_its_grid():
    """On a uxarray mesh the result stays on the mesh, weighted by face areas."""
    ux = pytest.importorskip("uxarray")
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lat = np.array([-45.0, 45.0])
    grid = ux.Grid.from_structured(lon=lon, lat=lat)
    n = grid.n_face
    base_vals = np.ones(n)
    base_vals[0] = 1e-6
    test = ux.UxDataArray(
        xr.DataArray(np.full(n, 2.0), dims="n_face", name="X"), uxgrid=grid
    )
    base = ux.UxDataArray(xr.DataArray(base_vals, dims="n_face", name="X"), uxgrid=grid)
    pct = utils.pct_change(test, base)
    assert isinstance(pct, ux.UxDataArray) and pct.uxgrid.n_face == n
    assert np.isnan(pct.values[0])
    assert np.allclose(pct.values[1:], 100.0)
    assert np.isfinite(utils.pct_change_of_means(test, base))


def test_polar_header_is_pct_of_region_means():
    """The polar % panel's Mean is the % change of the regional means, not swamped."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    # pylint: disable-next=import-error,import-outside-toplevel
    import plotting_functions as pf

    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(0.0, 360.0, 5.0)
    shape = (lat.size, lon.size)
    coords = {"lat": lat, "lon": lon}
    base = xr.DataArray(np.ones(shape), dims=("lat", "lon"), coords=coords)
    base[-1, 0] = 1e-12  # a near-zero baseline cell inside the Arctic cap
    test = base * 0 + 1.5
    domain = [-180, 180, 45, 90]
    # pylint: disable-next=protected-access
    mean = pf._pct_of_region_means(test, base, domain)
    t_mean = utils.domain_stats(test, domain)[0]
    b_mean = utils.domain_stats(base, domain)[0]
    assert mean == pytest.approx(100.0 * (t_mean - b_mean) / b_mean)
    assert 49.0 < mean < 51.0
