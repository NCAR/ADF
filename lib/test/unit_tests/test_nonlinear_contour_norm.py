"""
Collection of python unit tests
for the non-linear contour normalization.

Some variables are drawn with contour levels that are deliberately not evenly
spaced -- the TEM fluxes run from -5e7 to 5e7 but are dense near zero -- and
they ask for a colour mapping that follows those levels rather than the range.
Two things stopped that working (NCAR/ADF issue #478): the lookup used to turn
a colormap name into a colormap was removed in matplotlib 3.9, and the
variable defaults asked for the mapping under a name nothing read.

The key check here is the last one, which reads the defaults file and fails on
any spelling of the option other than the one the code looks for.  That is the
mistake that hid the first bug for so long: the defaults looked correct.

NOTE: the tests that exercise plotting_utils import matplotlib, cartopy and
the rest of the plotting stack, so they are skipped in CI, which installs only
PyYAML and pytest.  The defaults check needs only PyYAML and runs everywhere.
"""

import os
import os.path
import sys
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)
_DEFAULTS_FILE = os.path.join(_ADF_LIB_DIR, "adf_variable_defaults.yaml")

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

import yaml

try:
    import numpy as np
    import xarray as xr
    import matplotlib

    matplotlib.use("Agg")
    from plotting_utils import colormap_object, prep_contour_plot

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


# The option the code reads, and the near misses that silently do nothing:
_OPTION = "non_linear"


def _load_defaults():
    """Read the variable defaults file."""
    with open(_DEFAULTS_FILE, encoding="utf-8") as fil:
        return yaml.safe_load(fil)


class NonLinearOptionNameTestRoutine(unittest.TestCase):
    """
    The defaults have to ask for the option by the name the code reads.
    """

    def test_no_variable_uses_another_spelling(self):
        """
        A variable asking for the non-linear mapping under any other name gets
        a linear one instead, with nothing to say so: the plot comes out, and
        only its colours are wrong.  This is what happened to the TEM fluxes,
        which asked for "non_linear_levels".
        """

        wrong = {}
        for variable, settings in _load_defaults().items():
            if not isinstance(settings, dict):
                continue
            # End if
            for key in settings:
                if key != _OPTION and _OPTION in str(key):
                    wrong.setdefault(str(variable), []).append(key)
                # End if
            # End for
        # End for

        self.assertEqual(
            wrong,
            {},
            msg=f"these variables ask for '{_OPTION}' under another name: {wrong}",
        )

    def test_the_option_is_used_by_some_variable(self):
        """
        Guards the check above against quietly passing because the option has
        been dropped from the defaults altogether.
        """

        users = [
            variable
            for variable, settings in _load_defaults().items()
            if isinstance(settings, dict) and settings.get(_OPTION)
        ]

        self.assertTrue(users, msg=f"no variable sets '{_OPTION}'")


@unittest.skipUnless(_HAS_PLOTTING, "plotting_utils dependencies not available")
class NonLinearNormTestRoutine(unittest.TestCase):
    """
    Unit tests for the colormap lookup and the norm it feeds.
    """

    def _data(self):
        """A small field spanning the range the TEM fluxes cover."""
        return xr.DataArray(
            np.linspace(-5e7, 5e7, 100).reshape(10, 10), dims=("lev", "lat")
        )

    def test_colormap_from_a_name(self):
        """The defaults give colormaps by name."""

        cmap = colormap_object("viridis")

        self.assertTrue(hasattr(cmap, "N"))

    def test_colormap_from_a_colormap(self):
        """A colormap that has already been looked up is passed through."""

        cmap = colormap_object("viridis")

        self.assertIs(colormap_object(cmap), cmap)

    def test_non_linear_gives_a_boundary_norm(self):
        """
        With the option set, the colours follow the contour levels.  Each of
        the three ways of specifying levels has its own branch, so each is
        checked, with a colormap named as every variable that uses the option
        does.
        """

        data = self._data()
        levels = [-5e7, -1e7, -1e6, 0, 1e6, 1e7, 5e7]

        for kwargs in (
            {"contour_levels": levels},
            {"contour_levels_range": [-5e7, 5e7, 1e7]},
            {},  # levels worked out from the data
        ):
            cp_info = prep_contour_plot(
                data,
                data,
                data - data,
                data - data,
                non_linear=True,
                colormap="RdYlBu_r",
                **kwargs,
            )

            self.assertEqual(
                type(cp_info["norm1"]).__name__, "BoundaryNorm", msg=str(kwargs)
            )

    def test_a_centred_norm_still_wins_without_a_colormap(self):
        """
        Records an interaction that is easy to trip over: with neither a
        colormap nor explicit levels given, a later block replaces the norm
        with one centred on zero, so asking for the non-linear mapping has no
        effect.  Every variable that asks for it names a colormap, so this
        does not arise in practice, but a new one that did not would be
        puzzling.
        """

        data = self._data()

        cp_info = prep_contour_plot(
            data, data, data - data, data - data, non_linear=True
        )

        self.assertEqual(type(cp_info["norm1"]).__name__, "TwoSlopeNorm")

    def test_without_the_option_the_norm_is_linear(self):
        """Variables that do not ask for it are unaffected."""

        data = self._data()

        cp_info = prep_contour_plot(
            data, data, data - data, data - data, contour_levels=[-1, 0, 1]
        )

        self.assertEqual(type(cp_info["norm1"]).__name__, "Normalize")

    def test_every_variable_that_asks_for_it_works(self):
        """
        Run the real defaults of every variable that sets the option: before
        this was fixed, each of them raised AttributeError from the removed
        colormap lookup.
        """

        data = self._data()
        defaults = _load_defaults()
        checked = 0

        for variable, settings in defaults.items():
            if not isinstance(settings, dict) or not settings.get(_OPTION):
                continue
            # End if
            cp_info = prep_contour_plot(
                data, data, data - data, data - data, **settings
            )
            self.assertEqual(
                type(cp_info["norm1"]).__name__, "BoundaryNorm", msg=str(variable)
            )
            checked += 1
        # End for

        self.assertGreater(checked, 0)


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
