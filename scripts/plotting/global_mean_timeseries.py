"""Use time series files to produce global mean time series plots for ADF web site.

Includes a minimal Class for bringing CESM2 LENS data 
from I. Simpson's directory (to be generalized).

"""

from pathlib import Path
from types import NoneType

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning


def global_mean_timeseries(adfobj):
    """
    load time series file, calculate global mean, annual mean
    for each case
    Make a combined plot, save it, add it to website.
    Include the CESM2 LENS result if it can be found.
    """

    emislist = ["SFmonoterp","SFisoprene","SFSS","SFDUST", "SFSOA", "SFSO4", "SFSO2_net", "SFOM", "SFBC", "SFDMS", "SFH2O2","SFH2SO4"]
    cblist=["cb_SULFATE","cb_isoprene","cb_monoterp","cb_DUST","cb_DMS","cb_BC","cb_OM","cb_H2O2","cb_H2SO4","cb_SALT", "cb_SO2"]

    plot_loc = get_plot_loc(adfobj)

    plot_type = adfobj.read_config_var("diag_basic_info").get("plot_type", "png")
   
    res = adfobj.variable_defaults

    for var in adfobj.diag_var_list:
        ts_files = adfobj.data.get_ref_timeseries_file(var)#
        # If no files exist, try to move to next variable. --> Means we can not proceed with this variable, and it'll be problematic later.
        if not ts_files:
            errmsg = f"Time series files for variable '{var}' not found.  Script will continue to next variable."
            warnings.warn(errmsg)
            continue
        #End if

        #TEMPORARY:  For now, make sure only one file exists:
        if len(ts_files) != 1:
            errmsg =  "Currently the AMWG table script can only handle one time series file per variable."
            errmsg += f" Multiple files were found for the variable '{var}', so it will be skipped."
            warnings.warn(errmsg)
            continue
        #End if

        #Load model variable data from file:
        ds = utils.load_dataset(ts_files)
        ref_ts_da = ds[var]
        
        if var in emislist:
            ref_ts_da = surface_emission(ref_ts_da)
        elif var in cblist:
            ref_ts_da_ga = column_burden(ref_ts_da)
        # reference time series global average
        else:
            # check data dimensions:
            has_lat_ref, has_lev_ref = utils.zm_validate_dims(ref_ts_da)

            # check if this is a "2-d" varaible:
            if has_lev_ref:
                warnings.warn(
                    f"\t Variable named {var} has a lev dimension, which does not work with this script."
                )
                continue
            # End if
            
            # check if there is a lat dimension:
            if not has_lat_ref:
                warnings.warn(
                    f"\t Variable named {var} is missing a lat dimension, cannot continue to plot."
                )                
                continue
            # End if

            # reference time series global average
            ref_ts_da_ga = utils.spatial_average(ref_ts_da, weights=None, spatial_dims=None)

        # annually averaged
        if var not in emislist:
            ref_ts_da = utils.annual_mean(ref_ts_da_ga, whole_years=True, time_name="time")
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
            else adfobj.data.ref_case_label
        )
        for case_name in adfobj.data.case_names:
            c_ts_files = adfobj.data.get_timeseries_file(case_name, var)
            # If no files exist, try to move to next variable. --> Means we can not proceed with this variable, and it'll be problematic later.
            if not c_ts_files:
                errmsg = f"Time series files for case: {case_name} and variable '{var}' not found.  Script will continue to next variable."
                warnings.warn(errmsg)
                continue
            #End if

            #TEMPORARY:  For now, make sure only one file exists:
            if len(c_ts_files) != 1:
                errmsg =  "Currently the AMWG table script can only handle one time series file per variable."
                errmsg += f" Multiple files were found for case: {case_name} and the variable '{var}', so it will be skipped."
                print(errmsg)

            # Load model variable data from file:
            _ds = utils.load_dataset(c_ts_files)
            c_ts_da = _ds[var]

            # If no reference, we still need to check if this is a "2-d" varaible:
            # check data dimensions:
            has_lat_case, has_lev_case = utils.zm_validate_dims(c_ts_da)

            # If 3-d variable, notify user, flag and move to next test case
            if has_lev_case:
                warnings.warn(
                    f"\t    WARNING: Variable {var} has a lev dimension for '{case_name}', which does not work with this script."
                )
                skip_var = True
                continue
            #End if

            # check if there is a lat dimension:
            if not has_lat_case:
                warnings.warn(
                    f"\t    WARNING: Variable {var} is missing a lat dimension for '{case_name}', cannot continue to plot."
                )
                skip_var = True
                continue
            # End if

            # Case global average (keep noresm behavior)
            if var in emislist:
                c_ts_da_ga = surface_emission(c_ts_da)
            elif var in cblist:
                c_ts_da_ga = column_burden(c_ts_da)
            else:
                c_ts_da_ga = utils.spatial_average(c_ts_da, weights=None, spatial_dims=None)

            # annually averaged (keep noresm behavior)
            if var not in emislist:
                c_ts_da_ga = utils.annual_mean(c_ts_da_ga, whole_years=True, time_name="time")

            case_ts[labels[case_name]] = c_ts_da_ga


        fig, ax = make_plot(
            ref_ts_da, case_ts, var, label=adfobj.data.ref_nickname
        )
        ax.set_ylabel(getattr(ref_ts_da,"units", "[-]")) # add units
        plot_name = plot_loc / f"{var}_GlobalMean_ANN_TimeSeries_Mean.{plot_type}"

        conditional_save(adfobj, plot_name, fig)
        
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_latlon_map: Found variable defaults for {var}")

            #Extract category (if available):
            web_category = vres.get("category", None)

        else:
            vres = {}
            web_category = None

        adfobj.add_website_data(
            plot_name,
            f"{var}_GlobalMean",
            None,
            category=web_category,
            season="ANN",
            multi_case=True,
            plot_type="TimeSeries",
        )

def column_burden(_data):
    _area =  _get_area(_data)
    _data = (_data*_area).sum(dim={"lon","lat"})
    _data = 1e-9* _data
    _data.attrs['units'] = " Tg"
    return _data

def surface_emission(_data):
    _area =  _get_area(_data)
    _data = (_data*_area).sum(dim={"lon","lat"})
    # this provides the seconds in months in each year
    # -- do it for each year to allow for non-standard calendars (360-day)
    # -- and also to provision for data with leap years
    _days_gb = _data.time.dt.daysinmonth
    # weighted average with normalized weights: <x> = SUM x_i * w_i  (implied division by SUM w_i)
    _data=  (_data * _days_gb).groupby('time.year').sum(dim='time')
    _data = 1e-9*86400 * _data
    _data.attrs['units'] = " Tg/yr"
    return _data

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


class Lens2Data:
    """Access Isla's LENS2 data to get annual means."""

    def __init__(self, field):
        self.field = field
        self.has_lens, self.lens2 = self._include_lens()

    def _include_lens(self):
        lens2_path = Path(
            f"/glade/campaign/cgd/cas/islas/CESM_DATA/LENS2/global_means/annualmeans/"
        )

        lens2_fil = sorted(lens2_path.glob(f"{self.field}_*LENS2*first50*nc"))
        if lens2_fil:
            lens2_fil = lens2_fil[0]
            lens2 = xr.open_mfdataset(lens2_fil)
            has_lens = True
        else:
            warnings.warn(f"\t    INFO: Did not find LENS2 file for {self.field}.")
            has_lens = False
            lens2 = None
        return has_lens, lens2
######


def make_plot(case_ts, lens2, label=None, ref_ts_da=None):
    """plot yearly values of ref_ts_da"""
    fig, ax = plt.subplots()

    # Plot reference/baseline if available
    if type(ref_ts_da) != NoneType:
        ax.plot(ref_ts_da.year, ref_ts_da, label=label)
    else:
        return fig, ax
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
    print(f"Determined plot location: {plot_loc}")
    return plot_loc


def make_plot(ref_ts_da, case_ts, var, label=None):
    """plot yearly values of ref_ts_da"""
    fig, ax = plt.subplots()
    ax.plot(ref_ts_da.year, ref_ts_da, label=label)
    for c, cdata in case_ts.items():
        ax.plot(cdata.year, cdata, label=c)
    # Get the current y-axis limits
    ymin, ymax = ax.get_ylim()
    # Check if the y-axis crosses zero
    if ymin < 0 < ymax:
        ax.axhline(y=0, color="lightgray", linestyle="-", linewidth=1)
    ax.set_title(var, loc="left")
    ax.set_xlabel("YEAR")
    # Place the legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(
            bbox_to_anchor=(0.5, -0.15),loc="upper center", ncol=min(len(handles), 3),
        )
    plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)

    return fig, ax

##################
# Helper functions
##################
def _get_area(tmp_file):
    """
    This function retrieves the files, latitude, and longitude information
    in all the directories within the chosen dates.
    """
    Earth_rad=6.371e6 # Earth Radius in meters

    lon = tmp_file['lon'].data
    lon[lon > 180.] -= 360 # shift longitude from 0-360˚ to -180-180˚
    lat = tmp_file['lat'].data

   
    dlon = np.abs(lon[1]-lon[0])
    dlat = np.abs(lat[1]-lat[0])

    lon2d,lat2d = np.meshgrid(lon,lat)
            
    dy = Earth_rad*dlat*np.pi/180
    dx = Earth_rad*np.cos(lat2d*np.pi/180)*dlon*np.pi/180

    _area = dx*dy

    # Variables to return
    return _area
#END OF SCRIPT
##############

