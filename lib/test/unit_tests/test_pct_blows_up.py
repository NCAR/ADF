"""When a percent-difference panel is left out of a plot.

A baseline that changes sign (TAUX, TAUY, U, V) is small along every zero
line, and there the percent change runs to thousands of percent: the panel
then shows where the baseline crosses zero, not how the test case differs.
The panel is off unless the variable defaults turn it on.  These pin down
the variable-defaults setting pct_diff_panel (false, true or always) and
adf_utils.pct_blows_up, which leaves out a panel that is on (true) anyway.
"""

import sys
from pathlib import Path

import pytest  # pylint: disable=import-error

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position
import adf_utils as utils  # noqa: E402

# pylint: enable=import-error,wrong-import-position

LAT = np.linspace(-89, 89, 90)
LON = np.arange(0, 360, 4.0)


def _field(vals):
    return xr.DataArray(vals, coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"))


def _pct(case, base):
    return utils.pct_change(_field(case), _field(base))


def test_sign_changing_baseline_blows_up():
    """A baseline crossing zero near the equator, and a test case offset from it."""
    base = ((LAT - 0.1) / 90)[:, None] * np.ones(LON.size)
    assert utils.pct_blows_up(_pct(base + 0.2, base))


def test_positive_field_does_not_blow_up():
    """Temperature-like: positive everywhere, a few percent apart."""
    base = 250 + 40 * np.cos(np.deg2rad(LAT))[:, None] * np.ones(LON.size)
    assert not utils.pct_blows_up(_pct(base * 1.03, base))


def test_isolated_large_cells_do_not_blow_up():
    """A few cells beyond the limit (SWCF near the poles) keep the panel."""
    base = np.ones((LAT.size, LON.size))
    case = base.copy()
    case[-1, :3] = 50.0  # 4900 %, three cells at the highest latitude
    assert not utils.pct_blows_up(_pct(case, base))


def test_area_is_weighted_by_latitude():
    """The same number of large cells counts for more near the equator."""
    base = np.ones((LAT.size, LON.size))
    polar, equatorial = base.copy(), base.copy()
    polar[-1, :] = 50.0
    equatorial[LAT.size // 2, :] = 50.0
    frac = 0.5 / LAT.size  # between the polar and the equatorial row's area
    assert not utils.pct_blows_up(_pct(polar, base), area_frac=frac)
    assert utils.pct_blows_up(_pct(equatorial, base), area_frac=frac)


def test_limit_and_area_fraction_are_settable():
    """The limit and the area fraction are arguments, with module defaults."""
    base = np.ones((LAT.size, LON.size))
    pct = _pct(base * 3.0, base)  # 200 % everywhere
    assert not utils.pct_blows_up(pct)
    assert utils.pct_blows_up(pct, limit=100)
    assert not utils.pct_blows_up(pct, limit=100, area_frac=1.0)


def test_no_data_does_not_blow_up():
    """An all-missing panel is not hidden here (it says "No Valid Data")."""
    pct = _field(np.full((LAT.size, LON.size), np.nan))
    assert not utils.pct_blows_up(pct)


def test_profile_without_latitude():
    """A meridional-mean profile (lon only) is weighted equally."""
    pct = xr.DataArray(np.r_[np.full(10, 5000.0), np.ones(80)], dims="lon")
    assert utils.pct_blows_up(pct)
    assert not utils.pct_blows_up(pct, area_frac=0.2)


@pytest.fixture(name="pf")
def _pf():
    return pytest.importorskip("plotting_functions")


@pytest.fixture(name="blown")
def _blown():
    """A percent change that blows up (a baseline crossing zero)."""
    base = ((LAT - 0.1) / 90)[:, None] * np.ones(LON.size)
    return _pct(base + 0.2, base)


@pytest.fixture(name="calm")
def _calm():
    """A percent change that does not."""
    base = np.full((LAT.size, LON.size), 280.0)
    return _pct(base * 1.01, base)


# pylint: disable=protected-access


@pytest.mark.parametrize("setting", [None, False, "absent"])
def test_off_by_default(pf, blown, calm, setting):
    """Unset or false: left out, with the "off" note, whatever the data."""
    kwargs = {} if setting == "absent" else {"pct_diff_panel": setting}
    assert pf._pct_panel_note(blown, kwargs) == pf.PCT_OFF_MESSAGE
    assert pf._pct_panel_note(calm, kwargs) == pf.PCT_OFF_MESSAGE


def test_true_follows_the_check(pf, blown, calm):
    """true: drawn, unless the percent change blows up."""
    assert pf._pct_panel_note(blown, {"pct_diff_panel": True}) == pf.PCT_HIDDEN_MESSAGE
    assert pf._pct_panel_note(calm, {"pct_diff_panel": True}) is None


def test_always_draws(pf, blown):
    """always: drawn even when the percent change blows up."""
    assert pf._pct_panel_note(blown, {"pct_diff_panel": "always"}) is None


def test_plot_section_wins(pf, calm):
    """A setting in the plot type's section wins over the top-level one."""
    kwargs = {
        "pct_diff_panel": True,
        "plot_type": "zonal_mean",
        "zonal_mean": {"pct_diff_panel": False},
    }
    assert pf._pct_panel_note(calm, kwargs) == pf.PCT_OFF_MESSAGE
    # another plot type takes the top-level setting:
    kwargs["plot_type"] = "polar_map"
    assert pf._pct_panel_note(calm, kwargs) is None


def test_section_can_turn_it_on(pf, calm):
    """A section can turn the panel on for one plot type only."""
    kwargs = {"plot_type": "polar_map", "polar_map": {"pct_diff_panel": True}}
    assert pf._pct_panel_note(calm, kwargs) is None
    kwargs["plot_type"] = "zonal_mean"
    assert pf._pct_panel_note(calm, kwargs) == pf.PCT_OFF_MESSAGE


@pytest.mark.parametrize("setting", ["auto", "yes", 1.5])
def test_unknown_setting_warns_and_leaves_it_out(pf, calm, setting):
    """A value other than true, false or always warns and leaves the panel out."""
    with pytest.warns(UserWarning, match="pct_diff_panel"):
        note = pf._pct_panel_note(calm, {"pct_diff_panel": setting})
    assert note == pf.PCT_OFF_MESSAGE


@pytest.mark.parametrize(
    "path", ["adf_variable_defaults.yaml", "adf_variable_defaults_era5-1deg.yaml"]
)
def test_defaults_turn_it_on_only_for_one_signed_fields(path):
    """The shipped defaults turn the panel on for PRECT but not for U, T or TAUX."""
    yaml = pytest.importorskip("yaml")
    defaults = yaml.safe_load((Path(__file__).parents[2] / path).read_text())
    for var in ("PRECT", "Q", "SWCF", "CLDTOT"):
        assert defaults[var]["pct_diff_panel"] is True, var
    for var in ("U", "V", "T", "TAUX", "TAUY", "PSL", "RESTOM", "SHFLX"):
        assert "pct_diff_panel" not in (defaults[var] or {}), var
