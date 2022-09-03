"""
Module: regional_map_multicase

Provides a plot with regional maps of specified variables for all cases (up to 10) in a row.

Since this is a specialty plot, it looks looks for several custom options to be provided in the YAML configuration file. For example, one might include this block in the YAML:

region_multicase:
    region_spec: [slat, nlat, wlon, elon]
    region_time_option: <calendar | zeroanchor>  If calendar, will look for specified years. If zeroanchor will use a nyears starting from year_offset from the beginning of timeseries
    region_start_year: 
    region_end_year:
    region_nyear:
    region_year_offset:
    region_month: <NULL means look for season>
    region_season: <NULL means use annual mean>
    region_variables: <list of variables to try to use; allows for a subset of the total diag variables>

"""
#
# --- imports and configuration ---
#
from pathlib import Path
import warnings  # use to warn user about missing files.

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from plotting_functions import pres_from_hybrid, prep_contour_plot


def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = my_formatwarning
#
# --- Main Function Shares Name with Module: regional_map_multicase ---
#
def regional_map_multicase(adfobj):
    """
    regional_map_multicase
    input -> ADF object

    Sketch of workflow:
    - check for regional options (see module docstring),
    - get case names/locations (time series are used),
    - determine plot location and type
    - detect per-variable plot options
    - Loop through variables: make one multi-panel plot per variable
    """

    # Notify user that script has started:
    print("\n  Generating regional plots ...")

    # We need to know:
    # - Variable to plot
    # - Region to plot
    # - Years to include in average
    # - Months/Season to include in average
    # - Cases ... reference on left, test(s) to the right --> IMPOSE UPPER LIMIT OF 10 PANELS

    #
    # Check if regional options were specified... can not proceed without them
    #
    regional_opts = adfobj.read_config_var("region_multicase", required=True)
    if not regional_opts:
        print(
            "Regional options were not specified, so regional_map_multicase can not run. See documentation for options to add to configuration file."
        )
        return

    # case information
    case_names = adfobj.get_cam_info("cam_case_name", required=True)
    if len(case_names) > 10:
        print("ERROR: regional_map_multicase is limited to <= 10 cases.")
        return
    case_loc = adfobj.get_cam_info("cam_ts_loc", required=True)
    #
    # Determine input "reference case"
    #
    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):
        data_name = "obs"  # does not get used, is just here as a placemarker
        data_list = adfobj.read_config_var("obs_type_list")  # Double caution!
        data_loc = adfobj.get_basic_info(
            "obs_ts_loc", required=True
        )  ## Triple caution -- we might not have time series of observations!
    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True)
        data_loc = adfobj.get_baseline_info("cam_ts_loc", required=True)

    #
    # Set plot options
    #
    res = adfobj.variable_defaults  # dict of variable-specific plot preferences
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get("plot_type", "png")
    redo_plot = adfobj.get_basic_info("redo_plot")
    plot_loc = _get_plot_location(adfobj)
    sampling_str = _get_sampling_string(regional_opts)
    #
    # Determine the list of variables to use (defaults to ADF's diag_var_list)
    #
    var_list = regional_opts.get("region_variables", adfobj.diag_var_list)

    # < around here is where we add the plot to the website and delete the old version if redo >

    #
    # LOOP OVER VARIABLES
    #
    for v in var_list:
        # the reference case
        data_to_plot = {}
        data_to_plot[data_name] = _retrieve(
            regional_opts, v, data_name, data_loc
        )  # get the baseline field
        # each of the other cases
        for casenumber, case in enumerate(case_names):
            data_to_plot[case] = _retrieve(regional_opts, v, case, case_loc[casenumber])
        plot = regional_map_multicase_plot(
            adfobj, data_to_plot, opt=res.get(v)
        )  # res will be None if v not in the defaults
        # save the output
        plot_name = _construct_outfile_name(
            plot_loc, plot_type, v, sampling_str
        )  # this has to be consistent with web site generator

        adfobj.add_website_data(plot_name, "Regional Maps", None, multi_case=True)

        plot.savefig(plot_name)


def _retrieve(ropt, variable, casename, location, return_dataset=False):
    """Custom function that retrieves a variable. Returns the variable as a DataArray.

    Applies the regional and time interval selections based in values in ropt.

    ropt -> regional options specified via YAML

    variable -> the variable to get from files

    casename -> case name string for constructing file search

    location -> path to case

    kwarg:
    return_dataset -> if true, return the dataset object, otherwise return the DataArray
                      with `variable`
    """
    # note: this function assumes monthly data

    fils = sorted(Path(location).glob(f"{casename}*_{variable}_*.nc"))
    if len(fils) == 0:
        raise ValueError(f"something went wrong for variable: {variable}")
    elif len(fils) > 1:
        ds = xr.open_mfdataset(fils)  # allows climo files split into pieces
    else:
        ds = xr.open_dataset(fils[0])
    if return_dataset:
        da = ds
    else:
        da = ds[variable]

    # Apply regional selection logic
    # -- assume lat/lon coords
    da = da.sel(
        lat=slice(ropt["region_spec"][0], ropt["region_spec"][1]),
        lon=slice(ropt["region_spec"][2], ropt["region_spec"][3]),
    )

    # -- select time interval
    if "region_time_option" in ropt:
        if ropt["region_time_option"] == "calendar":
            if ("region_start_year" in ropt) and ("region_end_year" in ropt):
                da = da.loc[
                    {
                        "time": da.time[
                            (da.time.dt.year >= ropt["region_start_year"])
                            & (da.time.dt.year <= ropt["region_end_year"])
                        ]
                    }
                ]
        elif ropt["region_time_option"] == "zeroanchor":
            # This option is just in case one wants a specific length of sample after a given initial time,
            # for example, a 20-year (nyear) sample after a 2-year (year_offset) spin up time.
            # So if year_offset = 0, just will take the first nyear years.
            # note: this is designed to be working with years, and does not care if years are complete in the data set or not.
            min_year = da.time.dt.year.min()
            da = da.loc[
                {
                    "time": da.time[
                        (da.time.dt.year >= min_year + ropt["region_year_offset"])
                        & (
                            da.time.dt.year
                            <= min_year
                            + ropt["region_year_offset"]
                            + ropt["region_nyear"]
                        )
                    ]
                }
            ]
        else:
            print("ERROR: region_time_option must be calendar or zeroanchor")

    # -- select month, season, or annual average (precedence in that order)
    if "region_month" in ropt:
        if ropt["region_month"] is not None:
            month_number = _get_month_number(ropt["region_month"])
            if month_number is not None:
                da = da.loc[{"time": da.time[da.time.dt.month == month_number]}]
        else:
            month_number = None
    else:
        month_number = None

    if (month_number is None) and ("region_season" in ropt):
        month_length = da.time.dt.days_in_month
        da = (da * month_length).resample(time="QS-DEC").sum(
            dim="time"
        ) / month_length.resample(time="QS-DEC").sum(dim="time")
    else:
        print("Regional map is defaulting to annual mean.")
        month_length = da.time.dt.days_in_month
        da = (da * month_length).groupby("time.year").sum(
            dim="time"
        ) / month_length.groupby("time.year").sum(dim="time")
    return da


def _get_month_number(m):
    from time import strptime

    if isinstance(m, str):
        if len(m) == 3:
            if m in ["DJF", "MAM", "JJA", "SON"]:
                print(f"Error: it looks like a season was supplied as a month: {m}")
                return None  # fails if season was specified accidentally
            n = strptime(m, "%b").tm_mon  # guessing abbreviation (may is ok)
        elif len(m) > 3:
            n = strptime(m, "%B").tm_mon  # guessing full name
        elif len(m) == 1:
            n = strptime(m, "%m").tm_mon  # guessing supplied the number
        else:
            print(f"ERROR: confused about what month is meant by {m}")
            return None
    elif isinstance(m, int):
        return m
    else:
        print(f"Error: month supplied is neither string nor integer: {m}, {type(m)}")
        return None


def _get_sampling_string(ropt):
    """
    Determines a string to use for the output file.
    The month that is specified as `region_month`,
    the season specified as `region_season`,
    or `ANN` otherwise,
    in that order of precedence.

    Note: if month is specified as an integer, that is what will be returned.
    """
    result = ropt.get("region_month")
    if result is None:
        result = ropt.get("region_season")
        if result is None:
            result = "ANN"
    return result


def regional_map_multicase_plot(adf, datadict, opt=None):
    ncols = len(datadict)
    fig, ax = plt.subplots(
        figsize=(4 * ncols, 3),
        ncols=ncols,
        constrained_layout=True,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    [a.coastlines() for a in ax]
    cntrplts = []
    if opt is not None:
        tiString = opt.pop("tiString", None)
        tiFontSize = opt.pop("tiFontSize", 8)

    for i, xreg in enumerate(datadict):
        lons, lats = np.meshgrid(xreg.lon, xreg.lat)
        #
        # to do: check opt for all the things: cmap, contour levels, etc.
        #
        popts = prep_contour_plot(xreg, xreg, xreg - xreg, **opt)
        popts2 = {
            **popts["contourf_opt"],
            "cmap": popts["cmap1"],
            "norm": popts["norm1"],
            "levels": popts["levels1"],
        }
        cb_opt = popts.get("colorbar_opt")
        cntrplts.append(
            ax[i].contourf(lons, lats, xreg, transform=ccrs.PlateCarree(), **popts2)
        )  # can use ax[i] b/c it is only 1-d array of axes
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=popts2["norm"], cmap=popts2["cmap"]),
        ax=ax,
        orientation="horizontal",
        aspect=50,
        **cb_opt,
    )
    return fig


def _get_plot_location(adfobj):
    """
    Determine plot location based on ADF object.
    Create location if it is not a directory.
    Return a Path object to the directory.
    """
    plot_location = adfobj.plot_location  # ADF output path for plots and tables
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            # Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                print(f"\t    {pl} not found, making new directory")
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            print(
                f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}"
            )
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)
    return plot_loc


def _construct_outfile_name(ploc, ptype, *args):
    """
    Construct a complete Path object for the output plot.
    ploc -> plot location directory (expects Path object)
    ptype -> file extension for output (e.g., pdf, png)
    args -> arbitrary arguments to be joined by underscores

    Example:
    >>> _construct_outfile_name("/scratch/plots", "png", "cam6.120", "PRECT", "March", "LabSea")
    [returns] /scratch/plots/cam6.120_PRECT_March_LabSea_MultiCaseRegion_Mean.png
    """
    cat = "_".join(args)
    return ploc / f"{cat}_MultiCaseRegion_Mean.{ptype}"
