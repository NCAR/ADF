"""Taylor statistics must be computed over a single, common set of valid points.

Masked variables (land/ocean subsets) do not carry identical masks in the model
and the observations, and computing each moment over its own population lets the
correlation leave [-1, 1].
"""
import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[3] / "scripts" / "plotting"))
sys.path.append(str(Path(__file__).parents[2]))

from cam_taylor_diagram import taylor_stats_single  # noqa: E402


def _field(values):
    values = np.asarray(values, dtype=float)
    nlat, nlon = values.shape
    return xr.DataArray(
        values,
        dims=["lat", "lon"],
        coords={"lat": np.linspace(-60.0, 60.0, nlat), "lon": np.arange(nlon, dtype=float)},
    )


def test_identical_fields_give_unit_correlation_and_ratio():
    rng = np.random.default_rng(0)
    a = _field(rng.normal(size=(12, 24)))
    corr, ratio, bias = taylor_stats_single(a, a)
    assert np.isclose(corr, 1.0)
    assert np.isclose(ratio, 1.0)
    assert np.isclose(bias, 0.0)


def test_correlation_stays_within_bounds_for_mismatched_masks():
    """The Land2mTemperature failure: masks that disagree along a coastline."""
    rng = np.random.default_rng(1)
    base = 280.0 + 15.0 * rng.normal(size=(24, 48))
    case = _field(base.copy())
    ref = _field(base + 0.5 * rng.normal(size=(24, 48)))

    #Two land masks that overlap but disagree, as a model and obs mask do:
    case_mask = np.zeros((24, 48), dtype=bool)
    case_mask[5:20, 5:30] = True
    ref_mask = np.zeros((24, 48), dtype=bool)
    ref_mask[7:22, 8:33] = True

    case = case.where(xr.DataArray(case_mask, dims=["lat", "lon"], coords=case.coords))
    ref = ref.where(xr.DataArray(ref_mask, dims=["lat", "lon"], coords=ref.coords))

    corr, ratio, bias = taylor_stats_single(case, ref)
    assert -1.0 <= corr <= 1.0, f"correlation out of bounds: {corr}"
    assert ratio > 0.0
    assert np.isfinite(bias)


def test_masked_points_do_not_influence_the_result():
    """Whatever sits under the other field's mask must not change the answer."""
    rng = np.random.default_rng(2)
    vals = rng.normal(size=(16, 32))
    keep = np.ones((16, 32), dtype=bool)
    keep[:4, :] = False  # reference has no data in this band

    ref_vals = vals + 0.25 * rng.normal(size=(16, 32))
    ref = _field(np.where(keep, ref_vals, np.nan))

    case_a = _field(vals)
    #Same field, but absurd values where the reference is masked out:
    poisoned = vals.copy()
    poisoned[:4, :] = 1.0e6
    case_b = _field(poisoned)

    assert np.allclose(taylor_stats_single(case_a, ref),
                       taylor_stats_single(case_b, ref))


if __name__ == "__main__":
    test_identical_fields_give_unit_correlation_and_ratio()
    test_correlation_stays_within_bounds_for_mismatched_masks()
    test_masked_points_do_not_influence_the_result()
    print("all Taylor statistic checks passed")
