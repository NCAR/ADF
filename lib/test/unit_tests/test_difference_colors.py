"""Colors for a difference panel, including panels with nothing in them.

Asking for a pressure level above the model top is legitimate -- a high-top run
interpolates onto levels a low-top reference never reaches -- and the field
there is entirely NaN.  get_difference_colors took its range with np.min/np.max,
so a single missing point was enough to send every branch to False and leave the
colormap unassigned, ending the whole run in an UnboundLocalError.
"""

import sys
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")

import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).parents[2]))

import plotting_utils as pu  # noqa: E402


def test_all_missing_gives_usable_colors():
    norm, cmap = pu.get_difference_colors(np.full(6, np.nan))
    assert cmap is not None
    assert norm(0.0) == pytest.approx(0.5)


def test_a_missing_point_does_not_break_the_range():
    """The field is positive and negative; one NaN must not hide that."""
    norm, cmap = pu.get_difference_colors(np.array([-2.0, np.nan, 4.0]))
    assert cmap.name == "RdBu_r"
    assert norm(0.0) == pytest.approx(0.5)  # centered on zero


def test_positive_definite():
    norm, cmap = pu.get_difference_colors(np.array([1.0, 5.0]))
    assert cmap.name == "OrRd"
    assert norm(1.0) == pytest.approx(0.0)
    assert norm(5.0) == pytest.approx(1.0)


def test_negative_definite():
    _, cmap = pu.get_difference_colors(np.array([-5.0, -1.0]))
    assert cmap.name == "BuPu_r"


def test_straddling_zero_is_centered():
    norm, cmap = pu.get_difference_colors(np.array([-1.0, 3.0]))
    assert cmap.name == "RdBu_r"
    assert norm(0.0) == pytest.approx(0.5)
