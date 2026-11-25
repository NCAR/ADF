"""Native-grid (raster) panels are colored like the lat/lon (contourf) panels.

The lat/lon maps color each band between contour levels with
``cmap(norm(midpoint))``; the native-grid maps used to spread the colormap
evenly over the range instead, ignoring the configured levels.  For the
percent panel, whose default levels are packed near zero, that made a 5-10%
change almost white on the mesh.  These pin down
plotting_functions._raster_colors against what contourf actually draws.
"""

import sys
from pathlib import Path

import pytest  # pylint: disable=import-error

np = pytest.importorskip("numpy")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position,protected-access
import plotting_functions as pf  # noqa: E402

# pylint: enable=import-error,wrong-import-position

PCT_LEVELS = np.array(
    [-100, -75, -50, -40, -30, -20, -10, -8, -6, -4, -2, 0,
     2, 4, 6, 8, 10, 20, 30, 40, 50, 75, 100], dtype=float
)  # fmt: skip

CASES = {
    # field panels: a linear norm over the levels (SWCF's [-150, 50, 10])
    "linear": ("Blues", mpl.colors.Normalize(-150, 50), np.arange(-150, 51, 10.0)),
    # difference panels: white at zero
    "centred": (
        "BrBG",
        mpl.colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
        np.arange(-20, 21, 2.0),
    ),
    # % panels: the default levels with a BoundaryNorm, as prep_contour_plot sets
    "pct": ("PuOr_r", mpl.colors.BoundaryNorm(PCT_LEVELS, 256), PCT_LEVELS),
}


def _contourf_colors(cmap, norm, levels, extend):
    """The face colors contourf gives each band (and extension), in order."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 50)
    z = np.tile(np.linspace(levels[0] - 5, levels[-1] + 5, 50), (50, 1))
    cs = ax.contourf(x, x, z, levels=levels, cmap=cmap, norm=norm, extend=extend)
    colors = np.asarray(cs.get_facecolor())
    plt.close(fig)
    return colors


@pytest.mark.parametrize("extend", ["neither", "min", "max", "both"])
@pytest.mark.parametrize("case", list(CASES))
def test_matches_contourf(case, extend):
    """Each band, and each extension, gets contourf's color."""
    cmap, norm, levels = CASES[case]
    rcmap, rnorm = pf._raster_colors(cmap, norm, levels, extend)
    expected = _contourf_colors(cmap, norm, levels, extend)
    lower = 1 if extend in ("min", "both") else 0
    got = []
    if lower:
        got.append(rcmap.get_under())
    got.extend(rcmap(np.arange(rcmap.N)))
    if extend in ("max", "both"):
        got.append(rcmap.get_over())
    # one face color per band (and per extension), each contourf's:
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-6)
    # and a value in each band is drawn with that band's color:
    mids = 0.5 * (levels[:-1] + levels[1:])
    band_colors = expected[lower : lower + mids.size]
    np.testing.assert_allclose(rcmap(rnorm(mids)), band_colors, atol=1e-6)


def test_small_percent_change_is_colored():
    """A 5% change gets the 4-6% band's color, not a near-white one."""
    cmap, norm, levels = CASES["pct"]
    rcmap, rnorm = pf._raster_colors(cmap, norm, levels)
    five = np.asarray(rcmap(rnorm(np.array([5.0])))[0])
    zero_band = np.asarray(rcmap(rnorm(np.array([1.0])))[0])
    # the old linear scaling over -100..100 put 5% next to 0%:
    old = np.asarray(mpl.colormaps[cmap](mpl.colors.Normalize(-100, 100)(5.0)))
    assert np.abs(five - zero_band).sum() > 0.1
    assert np.abs(five - old).sum() > 0.05


def test_missing_cells_stay_blank():
    """NaN cells (ocean of a land field, below-ground levels) are not colored."""
    cmap, norm, levels = CASES["linear"]
    rcmap, rnorm = pf._raster_colors(cmap, norm, levels, "both")
    assert rcmap(rnorm(np.ma.masked_invalid([np.nan])))[0][3] == 0.0


def test_no_norm_maps_levels_linearly():
    """Without a norm the levels are mapped linearly, as contourf does."""
    levels = np.arange(0, 11, 1.0)
    rcmap, _ = pf._raster_colors("viridis", None, levels)
    expected = mpl.colormaps["viridis"](
        mpl.colors.Normalize(0, 10)(0.5 * (levels[:-1] + levels[1:]))
    )
    np.testing.assert_allclose(rcmap(np.arange(rcmap.N)), expected)
