"""
Collection of python unit tests for "choose_regrid_weights" in adf_diag.py,
which picks the weights file a native-grid variable is gridded to lat/lon
with, and for the "regrid_conservative" entries in the variable defaults
files.

NOTE: the choose_regrid_weights tests import adf_diag (xarray, xesmf, ...),
so they are skipped in CI, which installs only PyYAML and pytest.  They run
in a full ADF environment.  The variable defaults tests need only PyYAML.
"""

import os
import os.path
import sys
import unittest

import yaml

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    from adf_diag import choose_regrid_weights

    _HAS_DEPS = True
except (ImportError, SystemExit):
    # adf_diag exits if a required module is missing
    _HAS_DEPS = False

_DEFAULT = ("map_blin.nc", "bilinear")


@unittest.skipUnless(_HAS_DEPS, "adf_diag dependencies not available")
class ChooseRegridWeightsTestRoutine(unittest.TestCase):
    """
    Unit tests for choose_regrid_weights.
    """

    def test_marked_variable_uses_conservative_map(self):
        """A variable marked regrid_conservative uses the conservative map."""

        self.assertEqual(
            choose_regrid_weights(
                {"regrid_conservative": True}, _DEFAULT, "map_aave.nc"
            ),
            ("map_aave.nc", "conservative"),
        )

    def test_unmarked_variable_uses_default(self):
        """Unmarked, explicitly False, or unknown variables use the defaults."""

        for var_default in (
            {"colormap": "Blues"},
            {"regrid_conservative": False},
            None,
        ):
            self.assertEqual(
                choose_regrid_weights(var_default, _DEFAULT, "map_aave.nc"), _DEFAULT
            )

    def test_no_conservative_map(self):
        """Without a conservative map, a marked variable uses the defaults."""

        for cons in (None, ""):
            self.assertEqual(
                choose_regrid_weights({"regrid_conservative": True}, _DEFAULT, cons),
                _DEFAULT,
            )


class VariableDefaultsRegridConservativeTestRoutine(unittest.TestCase):
    """
    The "regrid_conservative" entries in the variable defaults files are
    booleans, and the main fluxes are marked.
    """

    def _defaults(self, name):
        with open(os.path.join(_ADF_LIB_DIR, name), encoding="utf-8") as dfil:
            return yaml.safe_load(dfil)

    def test_values_are_booleans(self):
        """Every regrid_conservative value is True or False."""

        for name in (
            "adf_variable_defaults.yaml",
            "adf_variable_defaults_era5-1deg.yaml",
        ):
            for var, entry in self._defaults(name).items():
                if isinstance(entry, dict) and "regrid_conservative" in entry:
                    with self.subTest(file=name, var=var):
                        self.assertIsInstance(entry["regrid_conservative"], bool)

    def test_fluxes_are_marked(self):
        """Precipitation, the top-of-model and surface fluxes, and the wind stress
        are marked."""

        defaults = self._defaults("adf_variable_defaults.yaml")
        for var in (
            "PRECT",
            "PRECC",
            "PRECL",
            "FLUT",
            "FSNT",
            "LHFLX",
            "SHFLX",
            "TAUX",
            "TAUY",
        ):
            with self.subTest(var=var):
                self.assertTrue(defaults[var].get("regrid_conservative"))


if __name__ == "__main__":
    unittest.main()
