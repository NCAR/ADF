"""Which variable-defaults conversion a load applies: the model's or the obs'.

The conversion follows from the case the data belongs to.  Every model case
(test cases and a baseline simulation) uses 'scale_factor'/'add_offset';
the observation reference uses 'obs_scale_factor'/'obs_add_offset'.  It used
to follow from whether the variable had an observation file, which, when
comparing against observations, left every variable without one in raw model
units (CLOUD, the aerosol burdens, NPP, ...).  These pin down
AdfData.get_value_converters, and that the fields the regrid step needs in raw
model units (PS for vertical interpolation) are still read unconverted.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

# pylint: disable=import-error,wrong-import-position
from adf_dataset import AdfData  # noqa: E402

# pylint: enable=import-error,wrong-import-position

CASE = "test_case"
BASE = "base_case"
DEFAULTS = {
    # has observations, with obs factors of its own
    "Q": {
        "scale_factor": 1000,
        "add_offset": 0,
        "obs_scale_factor": 10,
        "obs_add_offset": 0,
    },
    # has no observations
    "CLOUD": {"scale_factor": 100, "add_offset": 0},
    # converted for plotting, needed raw for vertical interpolation
    "PS": {"scale_factor": 0.01, "add_offset": 0, "new_unit": "hPa"},
    "T": {},
}


def _data(compare_obs, ref_labels=None):
    """An AdfData comparing against observations or the baseline `BASE`."""
    data = AdfData.__new__(AdfData)
    data.adf = SimpleNamespace(
        variable_defaults=DEFAULTS,
        compare_obs=compare_obs,
        unstructured_plotting=False,
        debug_log=lambda msg: None,
    )
    data.case_names = [CASE]
    data.ref_case_label = "Obs" if compare_obs else BASE
    data.ref_labels = ref_labels if ref_labels is not None else {}
    return data


@pytest.mark.parametrize(
    "case, field, expected",
    [
        (CASE, "Q", (0, 1000)),  # a model case: model factors
        ("ERA5", "Q", (0, 10)),  # the variable's observation source
        (CASE, "CLOUD", (0, 100)),  # no observations: still converted
        ("Obs", "CLOUD", (0, 1)),  # the generic obs label: obs factors (none)
        (None, "Q", (0, 1000)),  # no case named: not observations
        (CASE, "T", (0, 1)),  # an entry with no conversion
        (CASE, "NOT_IN_DEFAULTS", (0, 1)),
    ],
    ids=["model", "obs", "no-obs-model", "no-obs-obs", "no-case", "none", "absent"],
)
def test_against_observations(case, field, expected):
    """Obs factors for the observation source only; model factors otherwise."""
    data = _data(compare_obs=True, ref_labels={"Q": "ERA5"})
    assert data.get_value_converters(case, field) == expected


@pytest.mark.parametrize("case", [CASE, BASE])
def test_against_a_baseline(case):
    """Test case and baseline are both models, listed in diag_var_list or not."""
    data = _data(compare_obs=False, ref_labels={"Q": BASE})
    assert data.get_value_converters(case, "Q") == (0, 1000)
    assert data.get_value_converters(case, "CLOUD") == (0, 100)


def test_variable_without_obs_is_converted_on_load(tmp_path):
    """A test case's time series of a variable with no obs file is converted."""
    fil = tmp_path / f"{CASE}.cam.h0a.CLOUD.000101-000112.nc"
    xr.Dataset({"CLOUD": (("time", "lev"), np.full((1, 2), 0.5))}).to_netcdf(fil)
    data = _data(compare_obs=True, ref_labels={"Q": "ERA5"})
    data.get_timeseries_file = lambda case, var, **_kwargs: [fil]
    da = data.load_timeseries_da(CASE, "CLOUD")
    assert np.allclose(da.values, 50.0)


@pytest.fixture(name="rvi")
def _rvi():
    sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))
    return pytest.importorskip("regrid_and_vert_interp")


@pytest.mark.parametrize("compare_obs", [True, False], ids=["obs", "baseline"])
def test_surface_pressure_stays_in_pa(tmp_path, rvi, compare_obs):
    """PS read for vertical interpolation is in Pa, though PS plots in hPa."""
    ps_file = tmp_path / f"{CASE}_PS_climo.nc"
    ps = xr.Dataset({"PS": (("lat", "lon"), np.full((2, 2), 1.0e5))})
    ps.assign_coords(lat=[0.0, 1.0], lon=[0.0, 1.0]).to_netcdf(ps_file)
    data = _data(compare_obs, ref_labels={"PS": "ERA5" if compare_obs else BASE})
    data.get_climo_file = lambda case, var: [ps_file] if var == "PS" else []
    adf = SimpleNamespace(data=data, unstructured_plotting=False)
    no_ps = xr.Dataset({"T": (("lat", "lon"), np.ones((2, 2)))})
    # pylint: disable-next=protected-access
    found = rvi._find_surface_pressure(no_ps, adf, case=CASE)
    assert np.allclose(found.values, 1.0e5)
