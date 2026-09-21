"""The pressure levels 3-D fields are interpolated onto.

'interp_press_levels' replaces the ADF's standard set, for a model whose top is
above 1 hPa or a run that wants something else.  A level that is zero, negative
or not a number interpolates to nothing but NaN, so a bad entry has to be
rejected rather than carried into every 3-D field of the run.
"""

import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
# geocat.comp is pulled in by adf_utils.
pytest.importorskip("numpy")
pytest.importorskip("xarray")
pytest.importorskip("geocat.comp")

import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))

import adf_utils as utils  # noqa: E402

DEFAULT = [1000, 850, 500, 200, 100, 10, 1]


def test_empty_falls_back_to_the_default():
    for empty in (None, [], ()):
        assert np.array_equal(
            utils.pressure_levels_pa(empty, DEFAULT),
            np.array(DEFAULT, dtype=float) * 100.0,
        )


def test_levels_are_converted_to_pascals():
    assert np.array_equal(
        utils.pressure_levels_pa([500, 100], DEFAULT), np.array([50000.0, 10000.0])
    )


def test_levels_come_back_largest_first():
    """Whatever order they are given in: the ADF's own set runs 1000 hPa down."""
    out = utils.pressure_levels_pa([1, 500, 0.1, 100], DEFAULT)
    assert np.array_equal(out, np.array([50000.0, 10000.0, 100.0, 10.0]))


def test_repeated_levels_are_dropped():
    """A level given twice would be interpolated and written twice."""
    assert np.array_equal(
        utils.pressure_levels_pa([500, 100, 500], DEFAULT), np.array([50000.0, 10000.0])
    )


def test_a_top_above_one_hpa_is_allowed():
    """The whole point: a high-top model needs levels finer than 1 hPa."""
    out = utils.pressure_levels_pa([1, 0.5, 0.1, 0.01], DEFAULT)
    assert np.allclose(out, np.array([100.0, 50.0, 10.0, 1.0]))


@pytest.mark.parametrize(
    "bad",
    [
        0,  # not a sequence
        "1000,500",  # a string iterates into characters
        [1000, 0],  # zero pressure
        [1000, -500],  # negative pressure
        [1000, float("nan")],
        [1000, "surface"],
        [1000, None],
    ],
)
def test_bad_levels_are_rejected(bad):
    with pytest.raises(ValueError):
        utils.pressure_levels_pa(bad, DEFAULT)


def test_the_default_itself_is_not_trusted_blindly():
    """A bad default is still a bad set of levels."""
    with pytest.raises(ValueError):
        utils.pressure_levels_pa(None, [1000, -1])
