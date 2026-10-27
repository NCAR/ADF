"""
Use time series files to produce regional mean time series plots for LDF web site.

The settings, loading and region helpers are shared with regional_climatology.
"""
import warnings  # use to warn user about missing files.

import adf_utils as utils
from regional_climatology import (
    load_with_weights,
    make_regional_plots,
    plot_cases,
    regional_setup,
    report_missing,
)

warnings.formatwarning = utils.my_formatwarning


def regional_timeseries(adfobj):
    """
    Load time series files, subset for each region and each variable, and
    make a combined plot of the land-weighted, annual mean regional time series
    (baseline and test case) for each region.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information.

    Notes
    -----
    Config entries as for regional_climatology: 'region_list',
    'regional_var_list' and 'regions_file'.

    Uses ``adfobj.data`` (``get_timeseries_file``, ``get_ref_timeseries_file``,
    ``load_dataset``, ``apply_conversion``, ``ref_case_label``), the
    regional_climatology helpers, and ``utils.annual_mean``.

    TODO
    - make sure that TS's are being plotted with the preferred units
    - reinstate obs plotting once obs TS files are available
    """

    #Notify user that script has started:
    print("\n  --- Generating regional time series plots... ---")

    setup = regional_setup(adfobj)
    if setup is None:
        return
    case_name = setup["case_name"]
    ref_name = adfobj.data.ref_case_label

    base_data = {}
    case_data = {}
    for field in setup["var_list"]:
        # Load the global time series (and land weights) for this variable;
        # each case is read on its own grid (see `regional_setup`):
        base_data[field] = load_with_weights(
            adfobj, _load_ts(adfobj, adfobj.data.get_ref_timeseries_file(field),
                             setup["base_kwargs"]),
            field, ref_name)
        case_data[field] = load_with_weights(
            adfobj, _load_ts(adfobj, adfobj.data.get_timeseries_file(case_name, field),
                             setup["case_kwargs"]),
            field, case_name)
        report_missing(field, "time series", {ref_name: base_data[field],
                                              case_name: case_data[field]})
    # End for

    def annual(var):
        """Annual means, against the years (WW set time axis to years)."""
        var_ann = utils.annual_mean(var, whole_years=True, time_name="time")
        return var_ann.year, var_ann
    # End def

    def plot_series(ax, field, region):
        """Plot the annual mean regional time series of `field` on `ax`."""
        return plot_cases(ax, field, region, setup, base_data, case_data, annual)
    # End def

    make_regional_plots(adfobj, setup, plot_series,
                        plot_kind="RegionalTimeSeries", title="time series")

    print("\n  --- Regional time series plots generated successfully! ---")


def _load_ts(adfobj, fils, kwargs):
    """Time series dataset, with time at the middle of each interval."""
    if not fils:
        return None
    return adfobj.data.load_dataset(fils, use_time_bounds=True, **kwargs)


##############
#END OF SCRIPT
