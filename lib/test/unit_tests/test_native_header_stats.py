"""Map-header numbers: over the cells both fields cover, from the native grid.

The Mean/Max/Min in the header of a map's test and reference panels are taken
over the cells where both plotted fields have data, so that the two means are
over the same area (observations with gaps, below-ground levels, an ocean
mask).  A case on a native grid that is drawn on lat/lon has its numbers taken
from its native climatology with its own cell areas, not from the
interpolated field.  These pin down adf_utils.nearest_cells, overlap_mask and
overlap_stats, AdfData.header_stats, and the plots' use of the numbers.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position,protected-access
import adf_utils as utils  # noqa: E402
from adf_dataset import AdfData  # noqa: E402

# pylint: enable=import-error,wrong-import-position

LAT = np.array([-60.0, -20.0, 20.0, 60.0])
LON = np.array([0.0, 90.0, 180.0, 270.0])


def _latlon(values):
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("lat", "lon"),
        coords={"lat": LAT, "lon": LON},
    )


# Helpers in adf_utils
# --------------------
def test_nearest_cells_wraps_longitude():
    """Points go to the nearest grid cell, across the 0/360 line too."""
    ilat, ilon = utils.nearest_cells(
        [-59.0, 19.0, 61.0], [359.0, -95.0, 181.0], LAT, LON
    )
    assert list(ilat) == [0, 2, 3]
    assert list(ilon) == [0, 3, 2]


def test_overlap_stats_weights_and_keeps():
    """Weighted mean, max and min over the kept cells with data only."""
    vals = np.array([1.0, 3.0, np.nan, 100.0])
    wgt = np.array([1.0, 3.0, 1.0, 1.0])
    keep = np.array([True, True, True, False])
    mean, vmax, vmin = utils.overlap_stats(vals, wgt, keep)
    assert mean == pytest.approx((1 + 9) / 4)
    assert (vmax, vmin) == (3.0, 1.0)
    assert np.isnan(utils.overlap_stats(vals, wgt, np.zeros(4, bool))[0])


def test_overlap_mask_needs_both():
    """A cell counts only where the test and the reference both have data."""
    test = _latlon(np.ones((4, 4)))
    ref = test.copy()
    ref[0, :] = np.nan  # reference missing the southern row
    test[3, 0] = np.nan  # test missing one cell
    keep = utils.overlap_mask(test, ref)
    assert int(keep.sum()) == 16 - 4 - 1


# AdfData.header_stats
# --------------------
def _data(native_cases=(), compare_obs=True):
    adf = SimpleNamespace(
        unstructured_plotting=False,
        native_grid={case: True for case in native_cases},
        compare_obs=compare_obs,
        model_component="atm",
    )
    data = AdfData.__new__(AdfData)
    data.adf = adf
    data._native_fields = {}
    data.ref_case_label = "base"
    return data


def test_latlon_sides_use_the_overlap():
    """Lat/lon test vs obs with gaps: both means over the cells both cover."""
    test = _latlon(np.tile([[10.0], [20.0], [30.0], [40.0]], (1, 4)))
    ref = test - 1.0
    ref[0, :] = np.nan  # obs missing 60S
    stats = _data().header_stats("case", "T", "ANN", test, ref)
    wgt = np.cos(np.deg2rad(LAT[1:]))
    expected = float((np.array([20.0, 30.0, 40.0]) * wgt).sum() / wgt.sum())
    assert stats[0][0] == pytest.approx(expected)
    assert stats[1][0] == pytest.approx(expected - 1.0)
    assert stats[0][1:] == (40.0, 20.0, False)  # max, min over the overlap
    assert stats[1][3] is False


def test_native_case_uses_its_own_cells(monkeypatch):
    """A native case's numbers come from its cells, weighted by their areas."""
    data = _data(native_cases=("case",))
    cells = {
        "values": np.array([5.0, 7.0, 1000.0]),
        "weights": np.array([1.0, 3.0, 1.0]),
        "lat": np.array([19.0, 21.0, -61.0]),  # the last is in the obs gap
        "lon": np.array([91.0, 179.0, 0.0]),
    }
    monkeypatch.setattr(data, "native_header_field", lambda *a, **k: cells)
    test = _latlon(np.full((4, 4), 1.0))  # the interpolated field: not used
    ref = test.copy()
    ref[0, :] = np.nan
    stats = data.header_stats("case", "T", "ANN", test, ref)
    assert stats[0] == (pytest.approx((5 + 21) / 4), 7.0, 5.0, True)
    # Observations are never "native":
    assert stats[1][3] is False


def test_polar_domain_keeps_only_the_cap():
    """With a domain, only its latitudes count (the polar maps' caps)."""
    test = _latlon(np.tile([[1.0], [2.0], [3.0], [4.0]], (1, 4)))
    stats = _data().header_stats(
        "case", "T", "ANN", test, test, domain=[-180, 180, 45, 90]
    )
    assert stats[0][:3] == (4.0, 4.0, 4.0)


def test_unstructured_plotting_has_no_native_header_field():
    """Plots on the mesh weight by cell area already: no native field here."""
    data = _data(native_cases=("case",))
    data.adf.unstructured_plotting = True
    assert data.native_header_field("case", "T", "ANN") is None


def test_native_monthly_on_a_pressure_level(monkeypatch, tmp_path):
    """A 3-D native climo is put onto a pressure level with PS and hybrid coefficients."""
    pytest.importorskip("geocat.comp")
    nt, nlev, ncol = 12, 4, 3
    hyam = xr.DataArray([0.0, 0.0, 0.0, 0.0], dims="lev")
    hybm = xr.DataArray([0.25, 0.5, 0.75, 1.0], dims="lev")
    field = xr.DataArray(
        np.broadcast_to(np.array([250.0, 260.0, 270.0, 280.0])[None, :, None],
                        (nt, nlev, ncol)).copy(),
        dims=("time", "lev", "ncol"),
        coords={"lev": [250.0, 500.0, 750.0, 1000.0]},
    )  # fmt: skip
    dset = xr.Dataset(
        {
            "T": field,
            "PS": (("time", "ncol"), np.full((nt, ncol), 100000.0)),
            "hyam": hyam,
            "hybm": hybm,
            "area": (("time", "ncol"), np.ones((nt, ncol))),
            "lat": (("time", "ncol"), np.zeros((nt, ncol))),
            "lon": (("time", "ncol"), np.zeros((nt, ncol))),
        }
    )
    # As CAM writes it, so that geocat finds the vertical coordinate:
    dset["lev"].attrs.update(
        positive="down", standard_name="atmosphere_hybrid_sigma_pressure_coordinate"
    )
    data = _data(native_cases=("case",))
    data.adf.get_basic_info = lambda *a, **k: None
    monkeypatch.setattr(data, "get_climo_file", lambda *a: [tmp_path / "x.nc"])
    monkeypatch.setattr(data, "load_climo_dataset", lambda *a, **k: dset)
    out = data._native_monthly("case", "T", lev=500)
    np.testing.assert_allclose(np.asarray(out["field"]), 260.0)
    np.testing.assert_allclose(out["weights"].sum(), 1.0)


# The plots
# ---------
def test_stats_text_marks_native_numbers():
    """Native-grid numbers say so in the header."""
    pf = pytest.importorskip("plotting_functions")
    assert pf._stats_text(1.0, 2.0, 0.5, True).endswith("(native grid)")
    assert "native" not in pf._stats_text(1.0, 2.0, 0.5)


def test_polar_domain():
    """The polar caps, as the polar maps draw them."""
    pf = pytest.importorskip("plotting_functions")
    assert pf.polar_domain("NH") == [-180, 180, 45, 90]
    assert pf.polar_domain("Arctic") == [-180, 180, 50, 90]
    assert pf.polar_domain("SH") == [-180, 180, -90, -45]
    assert pf.polar_domain("SH", [0, 1, 2, 3]) == [0, 1, 2, 3]
