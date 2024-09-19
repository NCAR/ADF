from pathlib import Path
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import plotting_functions as pf

import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning


INCLUDE_LENS2 = True


def global_mean_timeseries(adfobj):
    """
    load time series file, calculate global mean, annual mean
    for each case
    Make a combined plot
    Optionally show the CESM2 LENS result.
    """
    var_list = adfobj.diag_var_list

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_locations = adfobj.plot_location

    # ADF variable which contains the output path for plots and tables:
    plot_location = adfobj.plot_location
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            #Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                print(f"\t    {pl} not found, making new directory")
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            print(f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}")
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)
    print(f"YOUR PLOT LOCATION: {plot_loc}")

    plot_type = basic_info_dict.get('plot_type', 'png')

    print(f"CHECK ON PLOT LOCATION: {plot_locations}")


    for field in var_list:
        # reference time series
        # files
        ref_ts_fil = adfobj.data.get_ref_timeseries_file(field)
        # dataset
        ref_ts_ds = adfobj.data.load_timeseries_dataset(ref_ts_fil)
        # check if this is a "2-d" varaible:
        has_lat_ref, has_lev_ref = pf.zm_validate_dims(ref_ts_ds[field])
        if has_lev_ref:
            print(f"Variable named {field} has a lev dimension, which does not work with this script.")
            continue

        # dataarray (global average)
        ref_ts_da_ga = pf.spatial_average(ref_ts_ds[field], weights=None, spatial_dims=None)
        # dataarray (annually averaged)
        ref_ts_da = pf.annual_mean(ref_ts_da_ga, whole_years=True, time_name='time')

        ## SPECIAL SECTION -- CESM2 LENS DATA:
        if INCLUDE_LENS2:
            lens2_fil = Path(f"/glade/campaign/cgd/cas/islas/CESM_DATA/LENS2/global_means/annualmeans/{field}_am_LENS2_first50.nc")
        if lens2_fil.is_file():
            print(f"Found LENS2 file! w00t!")
            lens2 = xr.open_mfdataset(lens2_fil)
            have_lens = True
        else:
            print("DID NOT FIND LENS2 FILE.")
            have_lens = False



        #Loop over model cases:
        case_ts = {} # dictionary of annual mean, global mean time series
        # same steps as in ref
        for case_idx, case_name in enumerate(adfobj.data.case_names):
            cfil = adfobj.data.get_timeseries_file(case_name, field)
            c_ts_ds = adfobj.data.load_timeseries_dataset(cfil)
            c_ts_da_ga = pf.spatial_average(c_ts_ds[field])
            case_ts[case_name] = pf.annual_mean(c_ts_da_ga)
        # now have to plot the timeseries
        fig, ax = plt.subplots()
        ax.plot(ref_ts_da.year, ref_ts_da, label=adfobj.data.ref_nickname)
        for c, cdata in case_ts.items():
            ax.plot(cdata.year, cdata, label=c)
        if have_lens:
            lensmin = lens2[field].min('M')
            lensmax = lens2[field].max('M')
            ax.fill_between(lensmin.year, lensmin, lensmax, color='lightgray', alpha=0.5)
            ax.plot(lens2.year, lens2[field].mean('M'), color='darkgray', linewidth=2, label='LENS2')

        ax.set_title(field, loc='left')
        ax.set_xlabel('YEAR')
        fig.legend()
        ofilename = plot_loc / f"{field}_annual_GlobalMean_TimeSeries.{plot_type}"
        print(f"GLOBAL MEAN TIMESERIES: writing {ofilename.name}")
        fig.savefig(ofilename)
        plt.close(fig)
