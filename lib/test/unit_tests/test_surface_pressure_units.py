"""Surface pressure handed to the hybrid-to-pressure interpolation must be in Pa.

A PS read back from a '*_PS_regridded.nc' file has the variable defaults applied
and so is in hPa; using it unconverted makes every tropospheric target level
come out NaN with no error raised anywhere.
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

from regrid_and_vert_interp import _pressure_in_pa  # noqa: E402


def _ps(values, units=None):
    da = xr.DataArray(np.array(values, dtype=float), dims=["lat"])
    if units is not None:
        da.attrs["units"] = units
    return da


def test_hpa_is_converted():
    out = _pressure_in_pa(_ps([968.0, 1013.0], units="hPa"))
    assert np.allclose(out.values, [96800.0, 101300.0])
    assert out.attrs["units"] == "Pa"


def test_pa_is_left_alone():
    inp = _ps([96800.0, 101300.0], units="Pa")
    out = _pressure_in_pa(inp)
    assert np.allclose(out.values, inp.values)


def test_missing_units_inferred_from_magnitude():
    #Looks like Pa, so leave it:
    assert np.allclose(_pressure_in_pa(_ps([96800.0])).values, [96800.0])
    #Looks like hPa, so scale it:
    assert np.allclose(_pressure_in_pa(_ps([968.0])).values, [96800.0])


def test_hpa_ps_would_lose_the_troposphere():
    """The failure this guards against, stated in terms of pressure levels."""
    hyam = np.array([0.00364, 0.17823, 0.0])
    hybm = np.array([0.0, 0.01968, 0.99256])
    #850 hPa rather than 1000: with a 968 hPa surface the 1000 hPa level is
    #genuinely below ground, which is a NaN for honest reasons.
    plevs_pa = np.array([850, 500, 200, 100, 50, 10, 5], dtype=float) * 100.0

    def covered(ps_pa):
        p = hyam * 100000.0 + hybm * ps_pa
        return int(((plevs_pa >= p.min()) & (plevs_pa <= p.max())).sum())

    #Correct units cover the whole requested column; hPa covers almost none of it:
    assert covered(96800.0) == len(plevs_pa)
    assert covered(968.0) < len(plevs_pa)
    #And the fix turns the broken input into the good one:
    fixed = _pressure_in_pa(_ps([968.0], units="hPa"))
    assert covered(float(fixed.max())) == len(plevs_pa)


if __name__ == "__main__":
    test_hpa_is_converted()
    test_pa_is_left_alone()
    test_missing_units_inferred_from_magnitude()
    test_hpa_ps_would_lose_the_troposphere()
    print("all surface-pressure unit checks passed")
