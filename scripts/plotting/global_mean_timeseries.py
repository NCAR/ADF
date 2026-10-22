"""Use time series files to produce global mean time series plots for ADF web site.

Includes a minimal Class for bringing in CESM2 LENS2 global means, from the
file each variable's defaults name ('lens2_file', found in the config's
'lens2_data_loc').

"""

from pathlib import Path
from types import NoneType

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotting_utils as plot_utils

import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning


def global_mean_timeseries(adfobj):
    """
    load time series file, calculate global mean, annual mean
    for each case
    Make a combined plot, save it, add it to website.
    Include the CESM2 LENS2 range and mean when the variable defaults name a
    'lens2_file' for the variable (a path, or a file name in the config's
    'lens2_data_loc') and it can be read.

    Land (CLM) cases, where ``adfobj.model_component`` is "lnd", are averaged
    the same way except that:
    - values are weighted by the land area of each cell (area * landfrac);
    - the variable defaults' 'avg_method' gives a weighted mean ('mean'), a
      global total ('sum'), or a running total of it ('cumsum', e.g. for
      NBP), scaled by 'scale_factor_table' and labelled with 'ts_unit' (or
      'table_unit');
    - the plot goes under the variable's website category.

    Whatever data there is gets plotted: a baseline (or test case) that lacks
    the variable, or has it in a form this script can't average, is left off
    the plot, and the variable is only skipped when no case is left.
    """

    is_lnd = adfobj.model_component == "lnd"

    #Notify user that script has started:
    comp_str = "land " if is_lnd else ""
    msg = f"\n  Generating {comp_str}global mean time series plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Gather ADF configurations
    plot_loc = get_plot_loc(adfobj)
    plot_type = adfobj.read_config_var("diag_basic_info").get("plot_type", "png")
    redo_plot = adfobj.get_basic_info("redo_plot")
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    # Loop over variables
    for field in adfobj.plot_var_list:
        #Notify user of variable being plotted:
        print(f"\t - time series plot for {field}")

        # This field's plot is named the same way whatever the data turns out
        # to hold, so a re-run can settle it here -- before opening the
        # reference time series and every case's, and before taking the
        # global means of them, which is the whole cost of this script:
        plot_name = plot_loc / f"{field}_GlobalMean_ANN_TimeSeries_Mean.{plot_type}"
        if (not redo_plot) and plot_name.is_file():
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(
                plot_name,
                f"{field}_GlobalMean",
                None,
                season="ANN",
                multi_case=True,
                plot_type="TimeSeries",
                category=res.get(field, {}).get("category") if is_lnd else None,
            )
            continue
        # End if

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if field in res:
            vres = res[field]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_mean_timeseries: Found variable defaults for {field}")
        else:
            vres = {}
        #End if
        vres = plot_utils.add_var_name_key_to_vres(adfobj, field, vres)
        vres["plot_type"] = __name__

        # reference time series (DataArray)
        ref_ts_da = adfobj.data.load_reference_timeseries_da(field)

        base_name = adfobj.data.ref_case_label

        # Check to see if this field is available; without it the test cases
        # are plotted on their own:
        if ref_ts_da is None:
            if not adfobj.compare_obs:
                print(f"\t    WARNING: No baseline time series for {field}, skipping baseline.")
            # End if
        else:
            # check data dimensions:
            has_lat_ref, has_lev_ref = utils.zm_validate_dims(ref_ts_da)

            # check if this is a "2-d" varaible:
            if has_lev_ref:
                print(
                    f"\t    WARNING: Variable {field} has a lev dimension for '{base_name}', which does not work with this script; skipping baseline."
                )
                ref_ts_da = None
            # End if

            # check if there is a lat dimension (native-grid data, ncol or lndgrid, has none):
            elif not has_lat_ref and not _is_native(ref_ts_da):
                print(
                    f"\t    WARNING: Variable {field} is missing a lat dimension for '{base_name}', skipping baseline."
                )
                ref_ts_da = None
            # End if

        if ref_ts_da is not None:
            # reference time series global average
            if is_lnd:
                ref_ts_da_ga = _land_global(
                    ref_ts_da, adfobj.data.get_ref_timeseries_file(field), vres
                )
            else:
                ref_ts_da_ga = utils.spatial_average(
                    ref_ts_da, weights=_native_weights(adfobj, field, ref_ts_da), spatial_dims=None
                )
            # End if
            # annually averaged (the test cases are plotted on their own when
            # the baseline can't be averaged)
            if ref_ts_da_ga is None:
                ref_ts_da = None
            else:
                ref_ts_da = utils.annual_mean(ref_ts_da_ga, whole_years=True, time_name="time")
            # End if
        # End if

        # Loop over model cases:
        case_ts = {}  # dictionary of annual mean, global mean time series

        # use case nicknames instead of full case names if supplied:
        labels = {
            case_name: nickname if nickname else case_name
            for nickname, case_name in zip(
                adfobj.data.test_nicknames, adfobj.data.case_names
            )
        }
        ref_label = (
            adfobj.data.ref_nickname
            if adfobj.data.ref_nickname
            else base_name
        )

        # A case without usable data is left off the plot; the others still
        # get plotted:
        unit = None  # the units of the last case plotted (for land)
        for case_name in adfobj.data.case_names:

            c_ts_da = adfobj.data.load_timeseries_da(case_name, field)

            if c_ts_da is None:
                print(
                    f"\t    WARNING: No time series of {field} for case '{case_name}', leaving it off the plot."
                )
                continue
            # End if

            # If no reference, we still need to check if this is a "2-d" varaible:
            # check data dimensions:
            has_lat_case, has_lev_case = utils.zm_validate_dims(c_ts_da)

            # If 3-d variable, notify user, flag and move to next test case
            if has_lev_case:
                print(
                    f"\t    WARNING: Variable {field} has a lev dimension for '{case_name}', which does not work with this script."
                )
                continue
            # End if

            # check if there is a lat dimension (native-grid data, ncol or lndgrid, has none):
            if not has_lat_case and not _is_native(c_ts_da):
                print(
                    f"\t    WARNING: Variable {field} is missing a lat dimension for '{case_name}', leaving it off the plot."
                )
                continue
            # End if

            # Gather spatial avg for test case
            if is_lnd:
                c_ts_da_ga = _land_global(
                    c_ts_da, adfobj.data.get_timeseries_file(case_name, field), vres
                )
                if c_ts_da_ga is None:
                    continue
                # End if
            else:
                c_ts_da_ga = utils.spatial_average(
                    c_ts_da, weights=_native_weights(adfobj, field, c_ts_da, case_name)
                )
            # End if
            case_ts[labels[case_name]] = utils.annual_mean(c_ts_da_ga, whole_years=is_lnd)
            unit = c_ts_da.attrs.get("units", "none")

        # Nothing at all to plot for this variable:
        if (not case_ts) and (ref_ts_da is None):
            print(f"\t    WARNING: No usable time series of {field} for any case, skipping it.")
            continue
        # End if

        # Running totals, e.g. of land carbon uptake (NBP):
        if is_lnd and vres.get("avg_method", "mean") == "cumsum":
            case_ts = {label: cdata.cumsum() for label, cdata in case_ts.items()}
            if ref_ts_da is not None:
                ref_ts_da = ref_ts_da.cumsum()
            # End if
        # End if

        # The variable's LENS2 file, when its defaults name one (class below):
        lens2_data = Lens2Data(field, vres, adfobj.get_basic_info("lens2_data_loc"))

        ## SPECIAL SECTION -- CESM2 LENS DATA:
        # Plot the timeseries
        fig, ax = make_plot(
            case_ts,
            lens2_data,
            label=ref_label if is_lnd else adfobj.data.ref_nickname,
            ref_ts_da=ref_ts_da,
        )

        if is_lnd:
            # the last of these given wins:
            if unit is None:
                unit = ref_ts_da.attrs.get("units", "none")
            # End if
            for unit_key in ("new_unit", "table_unit", "ts_unit"):
                unit = vres.get(unit_key, unit)
            # End for
        else:
            unit = getattr(ref_ts_da, "unit", vres.get("new_unit", "[-]"))
        # End if
        ax.set_ylabel(unit) # add units

        conditional_save(adfobj, plot_name, fig)

        adfobj.add_website_data(
            plot_name,
            f"{field}_GlobalMean",
            None,
            season="ANN",
            multi_case=True,
            plot_type="TimeSeries",
            category=vres.get("category") if is_lnd else None,
        )

    #Notify user that script has ended:
    print(f"  ... {comp_str}global mean time series plots have been generated successfully.")


# Helper/plotting functions
###########################

def conditional_save(adfobj, plot_name, fig, verbose=None):
    """Determines whether to save figure"""
    # double check this
    if adfobj.get_basic_info("redo_plot") and plot_name.is_file():
        # Case 1: Delete old plot, save new plot
        plot_name.unlink()
        fig.savefig(plot_name)
    elif (adfobj.get_basic_info("redo_plot") and not plot_name.is_file()) or (
        not adfobj.get_basic_info("redo_plot") and not plot_name.is_file()
    ):
        # Save new plot
        fig.savefig(plot_name)
    elif not adfobj.get_basic_info("redo_plot") and plot_name.is_file():
        # Case 2: Keep old plot, do not save new plot
        if verbose:
            print("\t - plot file detected, redo is false, so keep existing file.")
    else:
        warnings.warn(
            f"Conditional save found unknown condition. File will not be written: {plot_name}"
        )
    plt.close(fig)
######


def get_plot_loc(adfobj, verbose=None):
    """Return the path for plot files.
    Contains side-effect: will make the directory and parents if needed.
    """
    plot_location = adfobj.plot_location
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            # Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                if verbose:
                    print(f"\t    {pl} not found, making new directory")
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            if verbose:
                print(
                    f"\t Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}"
                )
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)
    print(f"\t Determined plot location: {plot_loc}")
    return plot_loc
######


def _native_weights(adfobj, field, data, case_name=None):
    """Cell-area weights for native-grid (ncol) data, else None.

    None leaves the weighting to utils.spatial_average, which uses cos(lat)
    for lat/lon data.  The areas are read from the case's time series files,
    as the loaded data array does not carry them.
    """
    if "ncol" not in data.dims:
        return None
    if case_name is None:
        fils = adfobj.data.get_ref_timeseries_file(field)
    else:
        fils = adfobj.data.get_timeseries_file(case_name, field)
    return utils.native_grid_weights(fils[0])


def _is_native(data):
    """True for data on a native (unstructured) grid: CAM's ncol or CLM's lndgrid."""
    return ("ncol" in data.dims) or ("lndgrid" in data.dims)


def _land_global(data, fils, vres):
    """Land-area weighted global mean, or total, of land (CLM) data.

    Parameters
    ----------
    data : xarray.DataArray
        The variable, as loaded through ``adfobj.data``.
    fils : list
        The variable's time series files, which hold 'area' and 'landfrac'.
    vres : dict
        The variable's defaults: 'avg_method' ('mean' for a weighted mean,
        otherwise a global total) and 'scale_factor_table'.

    Returns
    -------
    xarray.DataArray or None
        The global series, or None (with a warning) when the files have no
        'area' or 'landfrac'.
    """
    weights = utils.land_weights(fils[0], normalize=(vres.get("avg_method", "mean") == "mean"))
    if weights is None:
        print(f"\t    WARNING: No 'area' or 'landfrac' with {data.name}, so it can't be averaged over land.")
        return None
    # End if
    data = data * vres.get("scale_factor_table", 1)
    return utils.spatial_average_lnd(data, weights=weights)


class Lens2Data:
    """CESM2 LENS2 global, annual means of one variable, if it has any.

    The file is the variable defaults' 'lens2_file': a path, or a file name
    in the config's 'lens2_data_loc' (as 'obs_file' and 'obs_data_loc').  A
    variable without one gets no LENS2 band.
    """

    def __init__(self, field, vres, lens2_data_loc=None):
        self.field = field
        self.has_lens, self.lens2 = self._include_lens(vres.get("lens2_file"), lens2_data_loc)

    def _include_lens(self, lens2_file, lens2_data_loc):
        if not lens2_file:
            return False, None
        lens2_path = Path(lens2_file)
        if not lens2_path.is_file() and lens2_data_loc:
            lens2_path = Path(lens2_data_loc) / lens2_path
        if not lens2_path.is_file():
            warnings.warn(f"\t    INFO: Did not find LENS2 file '{lens2_file}' for {self.field}.")
            return False, None
        try:
            return True, xr.open_mfdataset([str(lens2_path)])
        except OSError as err:
            # e.g. a file this user may not read: plot without LENS2
            warnings.warn(f"\t    INFO: Could not read LENS2 file for {self.field}: {err}")
            return False, None
######


def make_plot(case_ts, lens2, label=None, ref_ts_da=None):
    """plot yearly values of ref_ts_da"""
    fig, ax = plt.subplots()

    # Plot reference/baseline if available (the test cases, or the baseline,
    # may also be plotted on their own)
    if type(ref_ts_da) != NoneType:
        ax.plot(ref_ts_da.year, ref_ts_da, label=label)
        syr = min(ref_ts_da.year)
        eyr = max(ref_ts_da.year)
    for idx, (c, cdata) in enumerate(case_ts.items()):
        ax.plot(cdata.year, cdata, label=c)
        # Force the plot axis to always plot the test case years
        if idx == 0:
            syr = min(cdata.year)
            eyr = max(cdata.year)

    field = lens2.field  # this will be defined even if no LENS2 data
    if lens2.has_lens:
        lensmin = lens2.lens2[field].min("M")  # note: "M" is the member dimension
        lensmax = lens2.lens2[field].max("M")
        ax.fill_between(lensmin.year, lensmin, lensmax, color="lightgray", alpha=0.5)
        ax.plot(
            lens2.lens2[field].year,
            lens2.lens2[field].mean("M"),
            color="darkgray",
            linewidth=2,
            label="LENS2",
        )
    # Get the current y-axis limits
    ymin, ymax = ax.get_ylim()
    # Check if the y-axis crosses zero
    if ymin < 0 < ymax:
        ax.axhline(y=0, color="lightgray", linestyle="-", linewidth=1)
    ax.set_title(field, loc="left")

    # Set the x-axis limits to the first test case climo years
    ax.set_xlim(syr, eyr)
    # Force x-axis to use only integer labels
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xlabel("YEAR")
    # Place the legend
    ax.legend(
        bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=max(min(len(case_ts), 3), 1)
    )
    plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)

    return fig, ax
######


##############
#END OF SCRIPT
