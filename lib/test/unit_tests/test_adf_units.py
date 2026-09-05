"""
Collection of python unit tests for the "adf_units" unit string helpers.
"""

# +++++++++++++++++++++++
# Import required modules
# +++++++++++++++++++++++

import unittest
import sys
import os
import os.path

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

# adf_units imports nothing but re, so these run in CI, where only PyYAML and
# pytest are installed:
from adf_units import normalize_units, units_equivalent

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main adf_units testing routine, used when script is run directly
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class AdfUnitsTestRoutine(unittest.TestCase):
    """
    Unit tests for comparing unit strings, which the ADF does to decide
    whether a conversion has already been applied to a file.
    """

    def test_one_unit_written_many_ways(self):
        """
        Check that the spellings of watts per square metre all agree.

        These are not hypothetical: CAM writes 'W/m2', and
        adf_variable_defaults.yaml contains both 'Wm$^{-2}$' and
        'W m$^{-2}$'.
        """

        spellings = [
            "W/m2",
            "W m-2",
            "Wm-2",
            "Wm^-2",
            "W m^-2",
            "W/m^2",
            "W m**-2",
            "W m$^{-2}$",
            "Wm$^{-2}$",
            "W m⁻²",
        ]

        canonical = {normalize_units(u) for u in spellings}

        self.assertEqual(
            len(canonical), 1, f"expected one canonical form, got {canonical}"
        )
        for unit in spellings:
            self.assertTrue(units_equivalent(unit, "W/m2"), unit)

    def test_defaults_spellings_match_file_spellings(self):
        """
        Check the pairs that actually occur: a unit as CAM writes it against
        the same unit as the variable defaults write it.
        """

        pairs = [
            ("W/m2", "Wm$^{-2}$"),
            ("m/s", "ms$^{-1}$"),
            ("mm/day", "mm d$^{-1}$"),
            ("ug/m3", "$\\mu$g/m3"),
            ("mol/mol", "mol mol$^{-1}$"),
            ("fraction", "Fraction"),
            ("%", "Percent"),
        ]

        for from_file, from_defaults in pairs:
            with self.subTest(units=from_file):
                self.assertTrue(units_equivalent(from_file, from_defaults))

    def test_different_units_stay_different(self):
        """
        Check that units which differ are not reported as equivalent.

        This is the direction that matters for correctness: a false match
        means a conversion is skipped and the data is plotted in the wrong
        units.
        """

        pairs = [
            ("W/m2", "W/m3"),
            ("m/s", "s/m"),
            ("K", "W/m2"),
            ("mm/day", "mm/s"),
            ("Pa", "hPa"),
            ("kg/m2", "kg/m3"),
            ("ppbv", "ppmv"),
            ("degrees_east", "degrees_north"),
        ]

        for first, second in pairs:
            with self.subTest(units=(first, second)):
                self.assertFalse(units_equivalent(first, second))

    def test_dimensionless_forms_agree(self):
        """
        Check that a unit whose factors cancel is dimensionless.
        """

        for unit in [
            "kg/kg",
            "kg kg-1",
            "fraction",
            "Fraction",
            "1",
            "unitless",
            "none",
        ]:
            with self.subTest(units=unit):
                self.assertEqual(normalize_units(unit), "1")

    def test_missing_units_match_nothing(self):
        """
        Check that an absent unit is not equivalent to anything.

        Nothing is known about a variable that does not say what its units
        are, so the ADF must not conclude a conversion was already applied.
        """

        self.assertEqual(normalize_units(None), "")
        for missing in [None, "", "   "]:
            with self.subTest(units=missing):
                self.assertFalse(units_equivalent(missing, "K"))
                self.assertFalse(units_equivalent("K", missing))
                self.assertFalse(units_equivalent(missing, missing))

    def test_unfamiliar_unit_matches_itself(self):
        """
        Check that a unit this does not understand still compares equal to
        itself, and unequal to something else.
        """

        self.assertTrue(units_equivalent("ppbv", "ppbv"))
        self.assertTrue(units_equivalent("kg m-2 s-1", "kg/m2/s"))
        self.assertFalse(units_equivalent("frobnicate", "ppbv"))

    def test_hpa_is_not_hecto_pascal_split(self):
        """
        Check that a known symbol is not split into smaller ones.

        'hPa' would become hour-pascal if the run-together splitting were
        applied to a symbol that stands on its own.
        """

        self.assertEqual(normalize_units("hPa"), "hpa^1")
        self.assertFalse(units_equivalent("hPa", "Pa"))


# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
