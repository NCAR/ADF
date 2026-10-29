"""The model pressure create_TEM_files stores in the TEM file ('PZM').

tem.py turns the TEM potential temperature (THZM) into temperature with the
model's own pressure.  The TEM history stream carries none, so it comes from the
PMID (or PINT) time series, and it has to cover the same time steps as the TEM
fields: a pressure averaged over other years would convert THZM with the wrong
pressure and give a plausible-looking, wrong temperature.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
# geocat.comp is pulled in by adf_utils, which create_TEM_files imports.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("cftime")
pytest.importorskip("geocat.comp")

import cftime  # noqa: E402  # pylint: disable=import-error
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3] / "scripts" / "averaging"))

from create_TEM_files import tem_pressure  # noqa: E402  # pylint: disable=import-error

LEV = np.array([10.0, 100.0, 500.0])
ZALAT = np.array([-45.0, 0.0, 45.0])


def _date(year, month, day=1):
    return cftime.DatetimeNoLeap(year, month, day)


def _tem(intervals):
    """A TEM Dataset with one step per (start, end) interval, stamped at the end."""
    ends = [end for _, end in intervals]
    return xr.Dataset(
        {
            "THZM": (("time", "lev", "zalat"), np.full((len(ends), 3, 3), 300.0)),
            "time_bnds": (("time", "nbnd"), np.array(intervals, dtype=object)),
        },
        coords={"time": ends, "lev": LEV, "zalat": ZALAT},
    )


def _pressure(months, units="Pa", field="PMID"):
    """Monthly PMID time series, the value in each month being yyyymm, mid-month."""
    times = [_date(y, m, 15) for y, m in months]
    values = np.array([100 * y + m for y, m in months], float)
    data = np.broadcast_to(values[:, None, None, None], (len(months), 3, 4, 2)).copy()
    return xr.Dataset(
        {field: (("time", "lev", "lat", "lon"), data, {"units": units})},
        coords={
            "time": times,
            "lev": LEV,
            "lat": [-60.0, -20.0, 20.0, 60.0],
            "lon": [0.0, 180.0],
        },
    )


def _adf(ds_pres):
    """An ADF stand-in whose data layer hands back `ds_pres` for case and baseline."""
    return SimpleNamespace(
        data=SimpleNamespace(
            load_timeseries_dataset=lambda case, field: ds_pres,
            load_reference_timeseries_dataset=lambda field: ds_pres,
        )
    )


MONTHLY = [(_date(2, 1), _date(2, 2)), (_date(2, 2), _date(2, 3))]


def test_each_step_gets_its_own_month():
    """January's step is stamped February 1st; it still gets January's pressure."""
    ds_pres = _pressure([(1, 12), (2, 1), (2, 2), (2, 3)])
    pzm = tem_pressure(_adf(ds_pres), "case", _tem(MONTHLY), "lev")
    assert pzm.dims == ("time", "lev", "zalat")
    assert pzm.isel(lev=0, zalat=0).values.tolist() == [201.0, 202.0]
    assert pzm.attrs["units"] == "Pa"


def test_baseline_reads_the_reference_time_series():
    """The baseline (case_name None) takes its pressure from the reference time series."""
    ds_pres = _pressure([(2, 1), (2, 2)])
    adf = _adf(None)
    adf.data.load_reference_timeseries_dataset = lambda field: ds_pres
    pzm = tem_pressure(adf, None, _tem(MONTHLY), "lev")
    assert pzm.isel(lev=0, zalat=0).values.tolist() == [201.0, 202.0]


def test_month_missing_from_the_time_series_is_left_missing():
    """A TEM month with no pressure is left missing, not filled from another month."""
    ds_pres = _pressure([(2, 1)])
    pzm = tem_pressure(_adf(ds_pres), "case", _tem(MONTHLY), "lev")
    values = pzm.isel(lev=0, zalat=0).values
    assert values[0] == 201.0 and np.isnan(values[1])


def test_daily_steps_share_their_months_pressure():
    """Daily TEM steps each take the mean pressure of their month."""
    daily = [(_date(2, 1, 1), _date(2, 1, 2)), (_date(2, 1, 2), _date(2, 1, 3))]
    pzm = tem_pressure(_adf(_pressure([(2, 1)])), "case", _tem(daily), "lev")
    assert pzm.isel(lev=0, zalat=0).values.tolist() == [201.0, 201.0]


def test_hpa_is_converted_to_pa():
    """A pressure time series in hPa is stored in Pa."""
    ds_pres = _pressure([(2, 1), (2, 2)], units="hPa")
    pzm = tem_pressure(_adf(ds_pres), "case", _tem(MONTHLY), "lev")
    assert pzm.isel(lev=0, zalat=0).values.tolist() == [20100.0, 20200.0]


@pytest.mark.parametrize(
    "ds_pres",
    [
        None,  # no PMID time series
        _pressure([(5, 1)]),  # none for the TEM years
        _pressure([(2, 1)]).rename({"lat": "ncol"}).drop_vars("ncol"),  # native grid
    ],
)
def test_no_usable_pressure_gives_none(ds_pres):
    """No time series, none for the TEM years, or a native grid: no PZM."""
    assert tem_pressure(_adf(ds_pres), "case", _tem(MONTHLY), "lev") is None
