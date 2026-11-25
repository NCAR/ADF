"""Where the panels go when the percent-difference panel is, or is not, drawn.

The percent panel is off unless the variable defaults turn it on
(``pct_diff_panel``).  When it is not drawn the plots leave it out of the
layout instead of leaving an empty slot: the lat/lon and polar maps put the
difference panel alone on the bottom row, centred under the test and baseline
maps, and the zonal and meridional means drop that row.
"""

import sys
from pathlib import Path

import pytest  # pylint: disable=import-error

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("cartopy")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position
import plotting_functions as pf  # noqa: E402

# pylint: enable=import-error,wrong-import-position

LAT = np.linspace(-88.5, 88.5, 60)
LON = np.arange(0, 360, 6.0)
LEV = np.array([100.0, 200.0, 500.0, 850.0, 1000.0])
YRS = [1, 10]


class _Adf:  # pylint: disable=too-few-public-methods
    """The bits of AdfDiag the plotting functions use."""

    def debug_log(self, msg):
        """Ignore debug messages."""


def _field(base, lev=False):
    """A one-signed lat/lon (or lev/lat/lon) field, and one 3% larger."""
    lat2d = np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)
    vals = base + 10 * lat2d
    dims, coords = ("lat", "lon"), {"lat": LAT, "lon": LON}
    if lev:
        vals = vals[None] * np.ones(LEV.size)[:, None, None]
        dims, coords = ("lev",) + dims, {"lev": LEV, **coords}
    test = xr.DataArray(vals * 1.03, dims=dims, coords=coords, attrs={"units": "mm/d"})
    ref = xr.DataArray(vals, dims=dims, coords=coords, attrs={"units": "mm/d"})
    return test, ref


@pytest.fixture(name="figs")
def _figs(monkeypatch):
    """Keep each figure the plotting functions close, to look at afterwards."""
    kept = []

    def _keep(fig=None):
        kept.append(fig if isinstance(fig, mpl.figure.Figure) else plt.gcf())

    monkeypatch.setattr(plt, "close", _keep)
    yield kept
    for fig in kept:
        mpl.pyplot.figure(fig.number)
    plt.close("all")


def _kwargs(setting):
    kwargs = {"var_name": "PRECT", "season": "ANN"}
    if setting is not None:
        kwargs["pct_diff_panel"] = setting
    return kwargs


def _maps(fig):
    """The map panels (not the colorbar insets), top to bottom, left to right."""
    axes = [a for a in fig.axes if hasattr(a, "projection")]
    # rows by rounded height, as aspect-adjusted (polar) panels differ slightly:
    return sorted(
        axes, key=lambda a: (-round(a.get_position().y0, 1), a.get_position().x0)
    )


def _centre(ax):
    pos = ax.get_position()
    return 0.5 * (pos.x0 + pos.x1)


def _check_maps(fig, show_pct):
    """Two maps on top; the bottom row holds two maps, or one centred under them."""
    maps = _maps(fig)
    top, bottom = maps[:2], maps[2:]
    assert top[0].get_position().y0 == pytest.approx(top[1].get_position().y0)
    if show_pct:
        assert len(bottom) == 2
        titles = [a.get_title(loc="left") for a in bottom]
        assert titles[0] == "Test % Diff Baseline"
    else:
        assert len(bottom) == 1
        # centred under the two maps above it, and the same size as they are:
        mid = 0.5 * (_centre(top[0]) + _centre(top[1]))
        assert _centre(bottom[0]) == pytest.approx(mid, abs=0.01)
        assert bottom[0].get_position().width == pytest.approx(
            top[0].get_position().width, rel=0.01
        )
        assert "Test % Diff Baseline" not in [a.get_title(loc="left") for a in fig.axes]
    assert "Baseline" in bottom[-1].get_title(loc="left")  # Test - Baseline


@pytest.mark.parametrize("setting", [None, False, True])
def test_latlon_layout(tmp_path, figs, setting):
    """Lat/lon map: the difference map is centred when there is no percent panel."""
    test, ref = _field(5.0)
    kwargs = _kwargs(setting)
    kwargs["wgt"] = np.cos(np.deg2rad(LAT))
    pf.plot_map_and_save(
        _Adf(),
        tmp_path / "map.png",
        "test",
        "base",
        YRS,
        YRS,
        test,
        ref,
        test - ref,
        pf.utils.pct_change(test, ref),
        **kwargs,
    )
    _check_maps(figs[-1], show_pct=setting is True)


@pytest.mark.parametrize("setting", [None, True])
def test_polar_layout(tmp_path, figs, setting):
    """Polar map: the same layout as the lat/lon map."""
    test, ref = _field(5.0)
    pf.make_polar_plot(
        _Adf(),
        tmp_path / "polar.png",
        "test",
        "base",
        YRS,
        YRS,
        test,
        ref,
        hemisphere="NH",
        **_kwargs(setting),
    )
    _check_maps(figs[-1], show_pct=setting is True)


def test_blown_up_panel_gets_a_note(tmp_path, figs):
    """A percent panel that was turned on but blows up is left out, with a note."""
    test, ref = _field(5.0)
    ref = ref - 7.0  # crosses zero
    kwargs = _kwargs(True)
    kwargs["wgt"] = np.cos(np.deg2rad(LAT))
    pf.plot_map_and_save(
        _Adf(),
        tmp_path / "map.png",
        "test",
        "base",
        YRS,
        YRS,
        test,
        ref,
        test - ref,
        pf.utils.pct_change(test, ref),
        **kwargs,
    )
    fig = figs[-1]
    _check_maps(fig, show_pct=False)
    notes = [t.get_text() for a in fig.axes for t in a.texts]
    assert pf.PCT_HIDDEN_MESSAGE in notes


def test_off_panel_has_no_note(tmp_path, figs):
    """A percent panel that is just off (the default) leaves no note."""
    test, ref = _field(5.0)
    kwargs = _kwargs(None)
    kwargs["wgt"] = np.cos(np.deg2rad(LAT))
    pf.plot_map_and_save(
        _Adf(),
        tmp_path / "map.png",
        "test",
        "base",
        YRS,
        YRS,
        test,
        ref,
        test - ref,
        pf.utils.pct_change(test, ref),
        **kwargs,
    )
    texts = [t.get_text() for a in figs[-1].axes for t in a.texts]
    texts += [t.get_text() for t in figs[-1].texts]
    assert not [t for t in texts if "not shown" in t]


def _rows(fig):
    """The data panels (not colorbars), top to bottom."""
    # colorbar insets have no subplot spec:
    axes = [a for a in fig.axes if a.get_subplotspec() is not None]
    return sorted(axes, key=lambda a: -a.get_position().y0)


@pytest.mark.parametrize("has_lev", [True, False])
@pytest.mark.parametrize("setting", [None, True])
@pytest.mark.parametrize("which", ["zonal", "meridional"])
def test_mean_plots_drop_the_row(tmp_path, figs, which, setting, has_lev):
    """Zonal and meridional means: one row fewer without the percent panel."""
    test, ref = _field(5.0, lev=has_lev)
    if which == "zonal":
        test, ref = test.mean("lon"), ref.mean("lon")
        func = pf.plot_zonal_mean_and_save
        args = (has_lev, False)
    else:
        func = pf.plot_meridional_mean_and_save
        args = (has_lev,)
    func(_Adf(), tmp_path / "mean.png", "test", "base", YRS, YRS, test, ref,
         *args, **_kwargs(setting))  # fmt: skip
    rows = _rows(figs[-1])
    expected = (4 if has_lev else 3) - (setting is not True)
    assert len(rows) == expected
    titles = [a.get_title(loc="left") for a in rows]
    assert ("Test % Diff Baseline" in titles) is (setting is True)
