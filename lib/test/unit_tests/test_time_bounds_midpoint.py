"""
Collection of python unit tests
for "adf_utils.use_time_bounds_midpoint".

CAM stamps a monthly average with one end of the interval it covers, and which
end depends on the model version: an older CAM h0 file stamps January with
February 1st.  Anything that then asks which year a step belongs to puts it in
the wrong one, which drops a year from an annual mean and shifts a seasonal
cycle by a month (NCAR/ADF issue #423).

The interval is not in doubt, because the file records it, so the time
coordinate is replaced by the midpoint of the recorded interval wherever files
are opened.  These tests cover which variable is believed, and that files
recording nothing are left alone.

NOTE: these tests import adf_utils, which imports xarray, so they are skipped
in CI, which installs only PyYAML and pytest.  They run in a full ADF
environment.
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
    import numpy as np
    import xarray as xr
    from adf_utils import plain_text_units, use_time_bounds_midpoint

    _HAS_ADF_UTILS = True
except ImportError:
    _HAS_ADF_UTILS = False


def _monthly_dataset(
    bounds_name, stamp="end", attrs_name=None, bounds_dim="nbnd", calendar="noleap"
):
    """
    Build a year of monthly data stamped at one end of each interval.

    `stamp` is the end the time coordinate carries, as CAM would write it:
    "end" is the older CAM h0 convention that puts January at February 1st.
    `attrs_name` is what the time coordinate's "bounds" attribute claims, which
    is not always the name of a variable that exists.
    """
    edges = xr.date_range(
        "0001-01-01", periods=13, freq="MS", calendar=calendar, use_cftime=True
    )
    lower, upper = edges[:-1], edges[1:]
    time = upper if stamp == "end" else lower
    bounds = np.stack([np.array(lower), np.array(upper)], axis=1)

    ds = xr.Dataset(
        {
            "T": (("time",), np.arange(12, dtype="f4")),
            bounds_name: (("time", bounds_dim), bounds),
        },
        coords={"time": np.array(time)},
    )
    ds["time"].attrs = {"long_name": "time", "bounds": attrs_name or bounds_name}
    return ds


@unittest.skipUnless(_HAS_ADF_UTILS, "adf_utils dependencies not available")
class TimeBoundsMidpointTestRoutine(unittest.TestCase):
    """
    Unit tests for moving the time coordinate to its interval midpoint.
    """

    def test_end_stamped_month_moves_into_its_own_month(self):
        """
        The bug itself: January stamped February 1st is counted as February,
        and in the last month of a year it is counted as the next year.
        """

        ds = _monthly_dataset("time_bnds")

        self.assertEqual(ds["time"].values[0].month, 2)  # before: wrong month
        fixed = use_time_bounds_midpoint(ds)
        self.assertEqual(fixed["time"].values[0].month, 1)
        self.assertEqual(fixed["time"].values[0].day, 16)
        # A year of data covers exactly one year once it is stamped correctly:
        self.assertEqual({t.year for t in fixed["time"].values}, {1})

    def test_bounds_attribute_is_believed_first(self):
        """
        What the file says its bounds are wins.  Here a second, wrong bounds
        variable carries the conventional name, so trusting the name rather
        than the attribute would shift every step by a month.
        """

        ds = _monthly_dataset("real_bnds", attrs_name="real_bnds")
        # A decoy under the conventional name, covering the following month:
        decoy_edges = xr.date_range(
            "0001-02-01", periods=13, freq="MS", calendar="noleap", use_cftime=True
        )
        ds["time_bnds"] = (
            ("time", "nbnd"),
            np.stack([np.array(decoy_edges[:-1]), np.array(decoy_edges[1:])], axis=1),
        )

        fixed = use_time_bounds_midpoint(ds)

        self.assertEqual(fixed["time"].values[0].month, 1)

    def test_conventional_names_when_no_attribute(self):
        """
        Files that record bounds without pointing at them still have to work,
        under either spelling the ADF has met.
        """

        for name in ("time_bnds", "time_bounds"):
            ds = _monthly_dataset(name)
            del ds["time"].attrs["bounds"]

            fixed = use_time_bounds_midpoint(ds)

            self.assertEqual(fixed["time"].values[0].month, 1, msg=name)

    def test_attribute_naming_something_absent_falls_back(self):
        """
        A "bounds" attribute pointing at a variable the file does not contain
        must not stop the conventional names being tried.
        """

        ds = _monthly_dataset("time_bnds", attrs_name="not_here")

        fixed = use_time_bounds_midpoint(ds)

        self.assertEqual(fixed["time"].values[0].month, 1)

    def test_already_at_the_midpoint_is_unchanged(self):
        """
        Newer streams stamp the middle of the interval already, so applying
        this must leave them exactly as they are.
        """

        edges = xr.date_range(
            "0001-01-01", periods=13, freq="MS", calendar="noleap", use_cftime=True
        )
        ds = _monthly_dataset("time_bnds", stamp="start")
        ds = ds.assign_coords(
            time=np.array([lo + (up - lo) / 2 for lo, up in zip(edges[:-1], edges[1:])])
        )
        before = ds["time"].values.copy()

        fixed = use_time_bounds_midpoint(ds)

        self.assertTrue(all(a == b for a, b in zip(before, fixed["time"].values)))

    def test_no_bounds_returns_what_it_was_given(self):
        """
        With nothing recorded there is nothing better than the stamp, so the
        dataset comes back untouched -- callers rely on that to know whether
        anything was done.
        """

        ds = _monthly_dataset("time_bnds")
        ds = ds.drop_vars("time_bnds")
        del ds["time"].attrs["bounds"]

        self.assertIs(use_time_bounds_midpoint(ds), ds)

    def test_no_time_variable_returns_what_it_was_given(self):
        """A dataset without a time coordinate is not this function's business."""

        ds = xr.Dataset({"area": (("ncol",), np.arange(4, dtype="f4"))})

        self.assertIs(use_time_bounds_midpoint(ds), ds)

    def test_unexpected_bounds_shape_returns_what_it_was_given(self):
        """
        Bounds that are not (time, 2) cannot be averaged into midpoints, so
        the file is left alone rather than guessed at.
        """

        ds = _monthly_dataset("time_bnds")
        ds["time_bnds"] = ds["time_bnds"].isel(nbnd=0)  # now one dimensional

        self.assertIs(use_time_bounds_midpoint(ds), ds)

    def test_bounds_dimension_may_be_called_anything(self):
        """
        CAM calls the second dimension 'nbnd'; other models use 'bnds' or 'nv'.
        Whichever it is, it is the one that is not time.
        """

        ds = _monthly_dataset("time_bnds", bounds_dim="bnds")

        fixed = use_time_bounds_midpoint(ds)

        self.assertEqual(fixed["time"].values[0].month, 1)

    def test_time_attributes_are_kept(self):
        """
        The coordinate keeps its metadata, which downstream code and any file
        written from it still need.
        """

        ds = _monthly_dataset("time_bnds")

        fixed = use_time_bounds_midpoint(ds)

        self.assertEqual(fixed["time"].attrs.get("long_name"), "time")
        self.assertEqual(fixed["time"].attrs.get("bounds"), "time_bnds")

    def test_real_calendar_dates_are_handled(self):
        """
        Cases run on a standard calendar as well as noleap, and the two are
        different date types in xarray.
        """

        ds = _monthly_dataset("time_bnds", calendar="standard")

        fixed = use_time_bounds_midpoint(ds)

        self.assertEqual(fixed["time"].values[0].month, 1)


@unittest.skipUnless(_HAS_ADF_UTILS, "adf_utils dependencies not available")
class PlainTextUnitsTestRoutine(unittest.TestCase):
    """
    Unit tests for rendering a unit string as plain text.

    The units in the variable defaults are written for plot labels, where
    matplotlib renders the LaTeX in them.  Written into a table -- a csv file
    or a web page -- the markup appears exactly as typed, which is what the
    AMWG tables started showing once they began using these units.
    """

    def test_every_markup_in_the_defaults(self):
        """
        Each form that appears in lib/adf_variable_defaults.yaml today.
        """

        cases = {
            "mm d$^{-1}$": "mm d^-1",
            "W m$^{-2}$": "W m^-2",
            "Wm$^{-2}$": "Wm^-2",
            "kg m$^{-2}$": "kg m^-2",
            "g m$^{-2}$": "g m^-2",
            "hPa d$^{-1}$": "hPa d^-1",
            "mol mol$^{-1}$": "mol mol^-1",
            "ms$^{-1}$": "ms^-1",
            r"$\mu$g/m3": "ug/m3",
        }

        for markup, plain in cases.items():
            self.assertEqual(plain_text_units(markup), plain, msg=markup)

    def test_plain_units_are_untouched(self):
        """
        Most units carry no markup, and those must come through unchanged --
        including after the defaults are rewritten in plain text, which is
        what pull request #427 proposes.
        """

        for unit in ("K", "Pa", "hPa", "m/s", "g/m2", "W/m2", "fraction", "--"):
            self.assertEqual(plain_text_units(unit), unit)

    def test_non_strings_are_returned_as_they_are(self):
        """
        A variable with no units at all should not become the string "None".
        """

        self.assertIsNone(plain_text_units(None))
        self.assertEqual(plain_text_units(1), 1)


# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
#############
