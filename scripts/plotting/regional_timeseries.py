"""
Use time series files to produce regional mean time series plots for LDF web site.

The settings, loading and region helpers are shared with regional_climatology.
"""
import warnings  # use to warn user about missing files.

import adf_utils as utils
from regional_climatology import (
    load_with_weights,
    make_regional_plots,
    region_sum,
    regional_setup,
    subset_region,
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
    kwargs = setup["kwargs"]
    ref_name = adfobj.data.ref_case_label

    base_data = {}
    case_data = {}
    for field in setup["var_list"]:
        # Load the global time series (and land weights) for this variable
        base_data[field] = load_with_weights(
            adfobj, _load_ts(adfobj, adfobj.data.get_ref_timeseries_file(field), kwargs),
            field, ref_name)
        case_data[field] = load_with_weights(
            adfobj, _load_ts(adfobj, adfobj.data.get_timeseries_file(case_name, field), kwargs),
            field, case_name)
        if base_data[field] is None or case_data[field] is None:
            print(f"\t    WARNING: Missing time series file for {field}, skipping it.")
            base_data[field] = None
        # End if
    # End for

    def plot_series(ax, field, region):
        """Plot the annual mean regional time series of `field` on `ax`."""
        base_var, base_wgt = subset_region(base_data[field], region, kwargs)
        case_var, case_wgt = subset_region(case_data[field], region, kwargs)

        # annually averaged
        base_var_ann = utils.annual_mean(region_sum(base_var, base_wgt),
                                         whole_years=True, time_name="time")
        case_var_ann = utils.annual_mean(region_sum(case_var, case_wgt),
                                         whole_years=True, time_name="time")

        # WW set time axis to years
        ax.plot(base_var_ann.year, base_var_ann, label=setup["base_nickname"], linewidth=2)
        ax.plot(case_var_ann.year, case_var_ann, label=setup["case_nickname"], linewidth=2)
        ax.legend()
        return base_var
    # End def

    make_regional_plots(adfobj, setup, base_data, plot_series,
                        plot_kind="RegionalTimeSeries", title="time series")

    print("\n  --- Regional time series plots generated successfully! ---")


def _load_ts(adfobj, fils, kwargs):
    """Time series dataset, with time at the middle of each interval."""
    if not fils:
        return None
    return adfobj.data.load_dataset(fils, use_time_bounds=True, **kwargs)


##############
#END OF SCRIPT
