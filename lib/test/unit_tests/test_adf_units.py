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
        Check that the words for "no units" all mean the same thing.

        A ratio that cancels is dimensionless too, but keeps what cancelled --
        see test_dimensionless_ratios_keep_what_cancelled.
        """

        for unit in ["fraction", "Fraction", "1", "unitless", "none"]:
            with self.subTest(units=unit):
                self.assertEqual(normalize_units(unit), "1")

    def test_case_decides_a_run_together_factor(self):
        """
        Check that "Nm" and "nm" are not read as the same unit.

        Newton-metre and nanometre differ only in case, so the run-together
        factors have to be split before anything lower-cases the string.
        Reporting these two as equal would let a conversion be skipped on a
        file whose units are nothing like the ones being converted to.
        """

        self.assertTrue(units_equivalent("N/m2", "Nm-2"))
        self.assertFalse(units_equivalent("Nm-2", "nm-2"))
        self.assertNotEqual(normalize_units("nm"), normalize_units("N m"))

    def test_unit_names_are_not_chopped_into_letters(self):
        """
        Check that a unit name is left whole rather than read as a product of
        whatever symbols happen to spell it.

        "Sv" is sieverts, not siemens-volt; "cal" is calories.
        """

        for unit in ["Sv", "cal", "dam", "DU", "molec"]:
            with self.subTest(units=unit):
                self.assertEqual(normalize_units(unit), unit.lower() + "^1")
        self.assertFalse(units_equivalent("Sv", "vs"))

    def test_dimensionless_ratios_keep_what_cancelled(self):
        """
        Check that two dimensionless ratios of different things differ.

        A mass mixing ratio and a volume mixing ratio are both dimensionless
        and are not the same number.
        """

        self.assertTrue(units_equivalent("kg/kg", "kg kg-1"))
        self.assertFalse(units_equivalent("kg/kg", "mol/mol"))
        self.assertFalse(units_equivalent("kg/kg", "m3/m3"))

    def test_units_with_several_factors(self):
        """
        Check units written with more than one solidus or exponent.
        """

        self.assertTrue(units_equivalent("kg/m2/s", "kg m-2 s-1"))
        self.assertTrue(units_equivalent("W/m2/K", "W m-2 K-1"))
        self.assertTrue(units_equivalent("1/s", "s-1"))
        self.assertFalse(units_equivalent("kg/m2/s", "kg/m2"))

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
