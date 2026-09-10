"""Regridded files must be converted exactly once.

The regridding stage applies the variable-defaults conversion when it writes and
stamps ``transformed`` on the result. Converting again at load time double-scales
(TAUX/TAUY, scaled by -1, come back with the wrong sign); never converting plots
files written by an older ADF -- which did the conversion at plot time -- in raw
units. Neither failure raises anything, so this pins the choice down.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
pytest.importorskip("numpy")
pytest.importorskip("xarray")

sys.path.append(str(Path(__file__).parents[2]))

from adf_dataset import AdfData  # noqa: E402

CONVERTED = (0, -1)  # (add_offset, scale_factor), as for TAUX
NO_CONVERSION = (0, 1)


def _data(stamped, converters=CONVERTED):
    """An AdfData stand-in holding one variable, with or without the stamp."""
    attrs = {"transformed": 1} if stamped else {}
    obj = SimpleNamespace(
        get_value_converters=lambda case, field: converters,
    )
    obj.reads = 0

    def load_dataset(fils, _obj=obj):
        _obj.reads += 1
        return {"TAUX": SimpleNamespace(attrs=attrs)}

    obj.load_dataset = load_dataset
    return obj


def _call(obj, apply_scaling=None):
    return AdfData._regrid_converters(
        obj, ["f.nc"], "TAUX", "case", "TAUX", apply_scaling
    )


def test_stamped_file_is_not_converted_again():
    assert _call(_data(stamped=True)) == NO_CONVERSION


def test_unstamped_legacy_file_is_converted():
    assert _call(_data(stamped=False)) == CONVERTED


def test_variable_without_a_conversion_never_opens_the_file():
    obj = _data(stamped=False, converters=NO_CONVERSION)
    assert _call(obj) == NO_CONVERSION
    assert obj.reads == 0


def test_explicit_override_wins_over_the_stamp():
    assert _call(_data(stamped=False), apply_scaling=False) == NO_CONVERSION
    assert _call(_data(stamped=True), apply_scaling=True) == CONVERTED
