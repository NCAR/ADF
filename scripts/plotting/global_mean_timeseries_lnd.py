"""Use time series files to produce global mean time series plots for ADF web site.

Includes a minimal Class for bringing CESM2 LENS data 
from I. Simpson's directory (to be generalized).

"""

from pathlib import Path
from types import NoneType
import warnings  # use to warn user about missing files.
import xarray as xr
import matplotlib.pyplot as plt
import plotting_functions as pf


def my_formatwarning(msg, *args, **kwargs):
    """custom warning"""
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = my_formatwarning


def global_mean_timeseries_lnd(adfobj):
    """
    load time series file, calculate global mean, annual mean
    for each case
    Make a combined plot, save it, add it to website.
    Include the CESM2 LENS result if it can be found.
    """

    model_component = "lnd"

    #Notify user that script has started:
    msg = "\n  Generating global mean time series plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Gather ADF configurations
    plot_loc = get_plot_loc(adfobj)
    plot_type = adfobj.read_config_var("diag_basic_info").get("plot_type", "png")
    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    # Loop over variables
    for field in adfobj.diag_var_list:
        #Notify user of variable being plotted:
        print(f"\t - time series plot for {field}")

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if field in res:
            vres = res[field]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_mean_timeseries: Found variable defaults for {field}")
        else:
            vres = {}
        #End if

        if model_component == "lnd":
            # Extract variables, get defaults, and scale for plotting
            weights, ref_ts_da, c_weights, c_ts_da, units, avg_method = _extract_and_scale_vars(adfobj, field, vres)
        else:
            # reference time series (DataArray)
            ref_ts_da = adfobj.data.load_reference_timeseries_da(field)
            weights = None
            units = vres.get("new_unit","[-]")

        base_name = adfobj.data.ref_case_label

        # Check to see if this field is available
        validate_dims = False
        if ref_ts_da is None:
            if model_component == "lnd":
                print(
                    f"\t    WARNING: Variable {field} for case '{base_name}' provides Nonetype. Skipping this variable"
                )
                validate_dims = True
            else:
                if not adfobj.compare_obs:
                    print(
                        f"\t    WARNING: Variable {field} for case '{base_name}' provides Nonetype. Skipping this variable"
                    )
                continue
        elif model_component == "lnd":
            validate_dims = False
            # reference time series global average
            # TODO, make this more general for land?
            ref_ts_da_ga = pf.spatial_average_or_sum(ref_ts_da, weights=weights, model_component=adfobj.model_component, method=avg_method)
            if model_component == "lnd":
                c_ts_da_ga = pf.spatial_average_or_sum(c_ts_da, weights=c_weights, model_component=adfobj.model_component, method=avg_method)

            # annually averaged
            ref_ts_da = pf.annual_mean(ref_ts_da_ga, whole_years=True, time_name="time")
            if model_component == "lnd":
                c_ts_da = pf.annual_mean(c_ts_da_ga, whole_years=True, time_name="time")

            # check if variable has a lev dimension
            has_lev_case = pf.zm_validate_dims(ref_ts_da)[1]
            if has_lev_case:
                print(
                    f"Variable named {field} has a lev dimension, which does not work with this script."
                )
                continue
        else:
            # check data dimensions:
            has_lat_ref, has_lev_ref = pf.zm_validate_dims(ref_ts_da)

            # check if this is a "2-d" varaible:
            if has_lev_ref:
                print(
                    f"\t    WARNING: Variable {field} has a lev dimension for '{base_name}', which does not work with this script."
                )
                continue
            # End if
            
            # check if there is a lat dimension:
            if not has_lat_ref:
                print(
                    f"\t    WARNING: Variable {field} is missing a lat dimension for '{base_name}', cannot continue to plot."
                )
                continue
            # End if

            # reference time series global average
            ref_ts_da_ga = pf.spatial_average(ref_ts_da, weights=None, spatial_dims=None, model_component=adfobj.model_component)

            # annually averaged
            ref_ts_da = pf.annual_mean(ref_ts_da_ga, whole_years=True, time_name="time")

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

        skip_var = False
        for case_name in adfobj.data.case_names:
            if model_component != "lnd":
                c_ts_da = adfobj.data.load_timeseries_da(case_name, field)

            if c_ts_da is None:
                print(
                    f"\t    WARNING: Variable {field} for case '{case_name}' provides Nonetype. Skipping this variable"

                )
                skip_var = True
                continue

            # If no reference, we still need to check if this is a "2-d" varaible:
            # check data dimensions:
            if validate_dims or model_component != "lnd":
                has_lat_case, has_lev_case = pf.zm_validate_dims(c_ts_da)
            # End if

            if model_component != "lnd":
                has_lat_case, has_lev_case = pf.zm_validate_dims(c_ts_da)

            # If 3-d variable, notify user, flag and move to next test case
            if has_lev_case:
                print(
                    f"\t    WARNING: Variable {field} has a lev dimension for '{case_name}', which does not work with this script."
                )

                skip_var = True
                continue
            # End if

            # check if there is a lat dimension:
            if model_component != "lnd" and not has_lat_case:
                print(
                    f"\t    WARNING: Variable {field} is missing a lat dimension for '{case_name}', cannot continue to plot."
                )
                skip_var = True
                continue
            # End if

            # Gather spatial avg for test case
            if model_component != "lnd":
                c_ts_da_ga = pf.spatial_average(c_ts_da, model_component=model_component)
                case_ts[labels[case_name]] = pf.annual_mean(c_ts_da_ga)
            else:
                case_ts[labels[case_name]] = pf.annual_mean(c_ts_da_ga, whole_years=True, time_name="time")
            

        # If this case is 3-d or missing variable, then break the loop and go to next variable
        if skip_var:
            continue

        lens2_data = Lens2Data(
                field
            )  # Provides access to LENS2 dataset when available (class defined below)

        ## SPECIAL SECTION -- CESM2 LENS DATA:
        # Plot the timeseries
        fig, ax = make_plot(
            case_ts, lens2_data, label=adfobj.data.ref_nickname, ref_ts_da=ref_ts_da
        )

        ax.set_ylabel(getattr(ref_ts_da,"unit", units)) # add units
        plot_name = plot_loc / f"{field}_GlobalMean_ANN_TimeSeries_Mean.{plot_type}"

        conditional_save(adfobj, plot_name, fig)

        adfobj.add_website_data(
            plot_name,
            f"{field}_GlobalMean",
            None,
            season="ANN",
            multi_case=True,
            plot_type="TimeSeries",
        )

    #Notify user that script has ended:
    print("  ... global mean time series plots have been generated successfully.")

def _extract_and_scale_vars(adfobj, field, vres):
    """
    Extract variables, get defaults, and scale for plotting
    """
    # Extract variables:
    # including a simpler way to get a dataset timeseries
    baseline_name     = adfobj.get_baseline_info("cam_case_name", required=True)
    input_ts_baseline = Path(adfobj.get_baseline_info("cam_ts_loc", required=True))
    # TODO hard wired for single case name:
    case_name = adfobj.get_cam_info("cam_case_name", required=True)[0]
    input_ts_case = Path(adfobj.get_cam_info("cam_ts_loc", required=True)[0])

    #Create list of time series files present for variable:
    baseline_ts_filenames = f'{baseline_name}.*.{field}.*nc'
    baseline_ts_files = sorted(input_ts_baseline.glob(baseline_ts_filenames))
    case_ts_filenames = f'{case_name}.*.{field}.*nc'
    case_ts_files = sorted(input_ts_case.glob(case_ts_filenames))

    ref_ts_ds = pf.load_dataset(baseline_ts_files)
    weights = ref_ts_ds.landfrac * ref_ts_ds.area
    ref_ts_da= ref_ts_ds[field]

    c_ts_ds = pf.load_dataset(case_ts_files)
    c_weights = c_ts_ds.landfrac * c_ts_ds.area
    c_ts_da= c_ts_ds[field]

    #Extract category (if available):
    web_category = vres.get("category", None)

    # get variable defaults  
    scale_factor = vres.get('scale_factor', 1)
    scale_factor_table = vres.get('scale_factor_table', 1)
    add_offset = vres.get('add_offset', 0)
    avg_method = vres.get('avg_method', 'mean')
    # get units for variable 
    ref_ts_da.attrs['units'] = vres.get("new_unit", ref_ts_da.attrs.get('units', 'none'))
    ref_ts_da.attrs['units'] = vres.get("table_unit", ref_ts_da.attrs.get('units', 'none'))
    units = ref_ts_da.attrs['units']

    # scale for plotting, if needed
    ref_ts_da = ref_ts_da * scale_factor * scale_factor_table
    ref_ts_da.attrs['units'] = units
    c_ts_da = c_ts_da * scale_factor * scale_factor_table
    c_ts_da.attrs['units'] = units

    return weights,ref_ts_da,c_weights,c_ts_da,units,avg_method


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
            print("plot file detected, redo is false, so keep existing file.")
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
                    f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}"
                )
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)
    print(f"Determined plot location: {plot_loc}")
    return plot_loc
######


class Lens2Data:
    """Access Isla's LENS2 data to get annual means."""

    def __init__(self, field):
        self.field = field
        self.has_lens, self.lens2 = self._include_lens()

    def _include_lens(self):
        lens2_fil = Path(
            f"/glade/campaign/cgd/cas/islas/CESM_DATA/LENS2/global_means/annualmeans/{self.field}_am_LENS2_first50.nc"
        )
        if lens2_fil.is_file():
            lens2 = xr.open_mfdataset(lens2_fil)
            has_lens = True
        else:
            warnings.warn(f"Time Series: Did not find LENS2 file for {self.field}.")
            has_lens = False
            lens2 = None
        return has_lens, lens2
######


def make_plot(case_ts, lens2, label=None, ref_ts_da=None):
    """plot yearly values of ref_ts_da"""
    field = lens2.field  # this will be defined even if no LENS2 data
    fig, ax = plt.subplots()
    
    # Plot reference/baseline if available
    if type(ref_ts_da) != NoneType:
        ax.plot(ref_ts_da.year, ref_ts_da, label=label)
    for c, cdata in case_ts.items():
        ax.plot(cdata.year, cdata, label=c)
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
    ax.set_xlabel("YEAR")
    # Place the legend
    ax.legend(
        bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=min(len(case_ts), 3)
    )
    plt.tight_layout(pad=2, w_pad=1.0, h_pad=1.0)

    return fig, ax
######


##############
#END OF SCRIPT
