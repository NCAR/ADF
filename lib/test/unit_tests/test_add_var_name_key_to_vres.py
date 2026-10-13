"""
Collection of python unit tests
for plotting_utils.add_var_name_key_to_vres.

The variable defaults are one dictionary that every plotting script shares,
so the plot settings a script adds to a variable's entry (season, plot_type,
lev, ...) must not end up in it, where the next script would pick them up.

NOTE: plotting_utils imports matplotlib, cartopy and the rest of the plotting
stack, so these are skipped in CI, which installs only PyYAML and pytest.
"""

import os
import os.path
import sys
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import matplotlib

    matplotlib.use("Agg")
    from plotting_utils import add_var_name_key_to_vres

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


class _Adf:  # pylint: disable=too-few-public-methods
    """Just enough of an ADF object: the variable defaults."""

    def __init__(self, defaults):
        self.variable_defaults = defaults


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class AddVarNameKeyToVresTestRoutine(unittest.TestCase):
    """
    Unit tests for add_var_name_key_to_vres.
    """

    def test_shared_defaults_are_not_changed(self):
        """Settings added for one plot stay out of the shared defaults."""
        defaults = {"T": {"colormap": "RdBu_r"}}
        vres = add_var_name_key_to_vres(_Adf(defaults), "T", defaults["T"])
        vres["season"] = "DJF"
        self.assertEqual(defaults["T"], {"colormap": "RdBu_r"})
        self.assertEqual(vres["var_name"], "T")

    def test_nickname_is_used(self):
        """A nickname is what the plots show."""
        defaults = {"TAUX": {"nickname": "Zonal Stress"}}
        vres = add_var_name_key_to_vres(_Adf(defaults), "TAUX", defaults["TAUX"])
        self.assertEqual(vres["var_name"], "Zonal Stress")

    def test_vector_uses_the_vector_nickname(self):
        """On a vector plot, a vector component is named after its vector."""
        defaults = {
            "TAUX": {"vector_name": "Surface_Wind_Stress"},
            "Surface_Wind_Stress": {"nickname": "Surface Wind Stress"},
        }
        vres = add_var_name_key_to_vres(
            _Adf(defaults), "TAUX", defaults["TAUX"], vector=True
        )
        self.assertEqual(vres["var_name"], "Surface Wind Stress")

    def test_scalar_plot_of_a_vector_component(self):
        """On its own plots, U is still titled "U", not "Wind"."""
        defaults = {"U": {"vector_name": "Wind"}}
        vres = add_var_name_key_to_vres(_Adf(defaults), "U", defaults["U"])
        self.assertEqual(vres["var_name"], "U")


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
