
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import xarray as xr
from pathlib import Path
import glob
from itertools import chain

#CAM diagnostic plotting functions:
import plotting_functions as pf


#Set seasonal ranges:
seasons = {"DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]}

#Set monthly codes:
month_dict = {1:'JAN',
    		  2:'FEB',
    		  3:'MAR',
    		  4:'APR',
    		  5:'MAY',
    		  6:'JUN',
    		  7:'JUL',
    		  8:'AUG',
    		  9:'SEP',
    		  10:'OCT',
    		  11:'NOV',
    		  12:'DEC'}

delta_symbol = r'$\Delta$'

#
# --- Main Function Shares Name with Module: regional_map_multicase ---
#
def seasonal_cycle(adfobj):
    """
    Chemistry Map main function
        * Initially start with  Zonal maps
            - This probably can be expanded to LatLon if given single pressure levels?

    """

    # Notify user that script has started:
    print("\n  Generating zonal vertical seasonal cycle plots...")

    # Special ADF variable which contains the output paths for all generated plots and tables:
    plot_locations = adfobj.plot_location
    plot_loc = Path(plot_locations[0])

    # Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # Check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    # CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)
    # Extract cam history files location:
    cam_hist_locs = adfobj.get_cam_info('cam_hist_loc')

    # Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    nicknames = adfobj.case_nicknames["test_nicknames"]

    # Grab history strings:
    cam_hist_strs = adfobj.hist_string["test_hist_str"]

    # Filter the list to include only strings that are exactly in the possible h0 strings
    # - Search for either h0 or h0a
    substrings = {"cam.h0","cam.h0a"}
    case_hist_strs = []
    for cam_case_str in cam_hist_strs:
        # Check each possible h0 string
        for string in cam_case_str:
            if string in substrings:
               case_hist_strs.append(string)
               break

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    if 'waccm_seasonal_cycle' in res:
        seas_cyc_res = res['waccm_seasonal_cycle']
    else:
        errmsg = "Missing 'waccm_seasonal_cycle' options in variable defaults yaml file.\n"
        errmsg += "Please make sure to include these for the seasonal cycle plots!"
        print(errmsg)
        logmsg = "WACCM seasonal cycle:"
        logmsg = "Missing 'waccm_seasonal_cycle' argument in variable defaults yaml file."
        adfobj.debug_log(logmsg)
        return

    merra2_vars = seas_cyc_res['merra2_vars']
    saber_vars = seas_cyc_res['saber_vars']
    obs_cam_vars = seas_cyc_res['obs_cam_vars']

    calc_var_list = seas_cyc_res['calc_var_list']

    # Grab location of ADF default obs files
    adf_obs_loc = Path(adfobj.get_basic_info("obs_data_loc"))
    saber_filename = "SABER_monthly_2002-2014.nc"
    saber_file = adf_obs_loc / saber_filename
    merra_filename = seas_cyc_res['merra2_file']
    merra_file = adf_obs_loc / merra_filename


    if not adfobj.get_basic_info("compare_obs"):
        obs = False
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
    
        # Grab baseline years and add to years lists
        syear_baseline = adfobj.climo_yrs["syear_baseline"]
        syear_cases = syear_cases + [syear_baseline]
        eyear_baseline = adfobj.climo_yrs["eyear_baseline"]
        eyear_cases = eyear_cases + [eyear_baseline]

        # Grab all case nickname(s) and add to nicknames lists
        base_nickname = adfobj.case_nicknames["base_nickname"]
        nicknames = nicknames + [base_nickname]

        # Grab baaseline cae name and add to case names list
        case_names = case_names + data_list

        # Get baeline case history location and add to hist loc list
        baseline_hist_locs = adfobj.get_baseline_info('cam_hist_loc')
        cam_hist_locs = cam_hist_locs + [baseline_hist_locs]

        # Grab history string:
        baseline_hist_strs = adfobj.hist_string["base_hist_str"]
        # Filter the list to include only strings that are exactly in the substrings list
        base_hist_strs = [string for string in baseline_hist_strs if string in substrings]
        hist_strs = case_hist_strs + base_hist_strs
    else:
        syear_cases = syear_cases + [""]
        eyear_cases = eyear_cases + [""]
        hist_strs = case_hist_strs
    # End if

    climo_yrs = [syear_cases, eyear_cases]

    #var_list = adfobj.diag_var_list
    var_list = calc_var_list + ['lat','lev','time']

    # Set up creation of all CAM data dictionaries
    cases_coords = {}
    cases_seasonal = {}
    cases_monthly = {}
    for idx,case_name in enumerate(case_names):

        hist_loc = cam_hist_locs[idx]
        hist_str = hist_strs[idx]

        syr = syear_cases[idx]
        eyr = eyear_cases[idx]

        # Make or access the WACCM zonal mean file
        ncfile = make_zm_files(adfobj,hist_loc,hist_str,case_name,calc_var_list,syr,eyr,return_ds=True)

        # Set up creation of individual CAM data dictionaries
        case_coords = {}
        case_seasonal = {}
        case_monthly = {}
        for var in var_list:

            if var not in case_seasonal:
                case_seasonal[var] = {}
            if var not in case_monthly:
                case_monthly[var] = {}
            case_coords[var] =  ncfile[var]

            # TODO: clean this up,
            if var in calc_var_list:
                for season in seasons:
                    if season not in case_seasonal[var]:
                        case_seasonal[var][season] = time_mean(ncfile, case_coords[var],
                                                               time_avg="season",
                                                               interval=season,
                                                               is_climo=None)

                # Set months number to reflect actual month num, ie 1:Jan, 2:Feb, etc
                for month in np.arange(1,13,1):
                    case_monthly[var][month_dict[month]] = time_mean(ncfile, case_coords[var],
                                                                    time_avg="month",
                                                                    interval=month,
                                                                    is_climo=None)

        cases_coords[case_name] = case_coords
        cases_monthly[case_name] = case_monthly
        cases_seasonal[case_name] = case_seasonal
    
    # Make nested dictionary of all case data
    case_ds_dict = {"coords":cases_coords,
                    "monthly":cases_monthly,
                    "seasonal":cases_seasonal}


    # Get Obs and seasonal and monthly averages
    saber, saber_monthly, saber_seasonal = saber_data(saber_file, saber_vars)
    merra2, merra2_monthly, merra2_seasonal = merra_data(merra_file, merra2_vars)
    #swoosh, swoosh_monthly, swoosh_seasonal = swoosh_data(filename = "/glade/work/richling/ADF/ADF_dev/notebooks/chem-diags/MERRA2_met.nc")

    obs_seas_dict = {"saber":saber_seasonal, "merra":merra2_seasonal}
    obs_month_dict = {"saber":saber_monthly, "merra":merra2_monthly}
    obs_ds_dict = {"monthly":obs_month_dict,
                   "seasonal":obs_seas_dict}
    
    #End gather data
    
    #Seasonal Cycle Plotting
    ########################

    #Zonal Mean Wind and Temp vs MERRA2 and SABER
    #--------------------------------------------

    # Notify user that script has started:
    print("\n\t Making Zonal Mean Wind and Temp vs MERRA2 and SABER...")

    # Comparison plot defaults    
    comp_plots_dict = res['comparison_plots']

    for cam_var in comp_plots_dict['cam_vars']:
        #Notify user of variable being plotted:
        print(f"\t - zonal mean maps for {var}")

        for interval in comp_plots_dict['interval']:
            #Check if interval is integer (month number) or string (season)
            if isinstance(interval, int):
                interval = month_dict[interval]
                season = "month"
            else:
                season = "season"
            #End if

            plot_name = plot_loc / f"{cam_var}_zm_{interval}_WACCM_SeasonalCycle_Mean.{plot_type}"
            if (not redo_plot) and plot_name.is_file():
                adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                adfobj.add_website_data(plot_name, f"{cam_var}_zm", case_name, season=interval,
                                        plot_type="WACCM", category="Seasonal Cycle",
                                        ext="SeasonalCycle_Mean",non_season=True)
            
            elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
                #If redo plot, delete the file
                if plot_name.is_file():
                    plot_name.unlink()
            
                pf.comparison_plots(plot_name, cam_var, case_names, nicknames, case_ds_dict,
                                    obs_ds_dict, season, interval, comp_plots_dict, obs_cam_vars)
                adfobj.add_website_data(plot_name, f"{cam_var}_zm", case_name, season=interval,
                                        plot_type="WACCM", category="Seasonal Cycle",
                                        ext="SeasonalCycle_Mean",non_season=True)
            #End if
        #End for
    #End for
        

    #Polar Cap Temps
    #---------------
    # Notify user that script has started:
    print("\n\t Making Polar Cap Temps...")

    pcap_dict = res['pcap_plots']
    for hemi in ["s","n"]:
        #Notify user of variable being plotted:
        print(f"\t - polar temp maps for {hemi}")

        plot_name = plot_loc / f"{hemi.upper()}PolarCapT_ANN_WACCM_SeasonalCycle_Mean.{plot_type}"

        # Check redo_plot. If set to True: remove old plot, if it already exists:
        redo_plot = adfobj.get_basic_info('redo_plot')

        if (not redo_plot) and plot_name.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(plot_name, f"{hemi.upper()}PolarCapT", case_name, season="ANN",
                                    plot_type="WACCM", category="Seasonal Cycle",
                                    ext="SeasonalCycle_Mean")
        elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
            #If redo plot, delete the file
            if plot_name.is_file():
                plot_name.unlink()

            pf.polar_cap_temp(plot_name, hemi, case_names, cases_coords, cases_monthly, merra2_monthly, pcap_dict)
            adfobj.add_website_data(plot_name, f"{hemi.upper()}PolarCapT", case_name, season="ANN",
                                    plot_type="WACCM", category="Seasonal Cycle",
                                    ext="SeasonalCycle_Mean")
        #End if
    #End for

    # Latitude vs Month Plots
    #########################

    var_dict = res['lat_vs_month']

    #Cold Point Temp/Tropopause @ 90hPa
    #----------------------------------
    # Notify user that script has started:
    print("\n\t Making Cold Point Temp/Tropopause @ 90hPa...")

    var = "T"
    try:
        vert_levs = var_dict[var]["plot_vert_levs"]
    except:
        errmsg = f"Missing 'plot_vert_levs' in variable defaults file for '{var}'\n"
        errmsg += "Please add it to the yaml file under 'lat_vs_month'"
        print(errmsg)

    for vert_lev in vert_levs:
        #Notify user of variable being plotted:
        print(f"\t - cold point temp maps for {vert_lev}hPa")
        plot_name = plot_loc / f"CPT_ANN_WACCM_SeasonalCycle_Mean.{plot_type}"

        if (not redo_plot) and plot_name.is_file():
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(plot_name, "CPT", case_name, season="ANN",
                                        plot_type="WACCM",
                                        ext="SeasonalCycle_Mean",
                                        category="Seasonal Cycle",
                                        )
        
        elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
            #If redo plot, delete the file
            if plot_name.is_file():
                plot_name.unlink()

            pf.month_vs_lat_plot(var, var_dict, plot_name, case_names, nicknames, climo_yrs, cases_coords, cases_monthly, vert_lev)
            adfobj.add_website_data(plot_name, "CPT", case_name, season="ANN",
                                        plot_type="WACCM",
                                        ext="SeasonalCycle_Mean",
                                        category="Seasonal Cycle",
                                        )
        #End if
    #End for


    #H20 Mixing Ratio @ 90 and 100hPa
    #----------------------------------
    # Notify user that script has started:
    print("\n\t Making H20 Mixing Ratio @ 90 and 100hPa...")
    var = "Q"
    try:
        vert_levs = var_dict[var]["plot_vert_levs"]
    except:
        errmsg = f"Missing 'plot_vert_levs' in variable defaults file for '{var}'\n"
        errmsg += "Please add it to the yaml file under 'lat_vs_month'"
        print(errmsg)

    for vert_lev in vert_levs:
        #Notify user of variable being plotted:
        print(f"\t - mixing ratio maps for {vert_lev}hPa")
        plot_name = plot_loc / f"MixRatio_{vert_lev}hPa_ANN_WACCM_SeasonalCycle_Mean.{plot_type}"

        if (not redo_plot) and plot_name.is_file():
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(plot_name, f"MixRatio_{vert_lev}hPa", case_name, season="ANN",
                                        plot_type="WACCM",
                                        ext="SeasonalCycle_Mean",
                                        category="Seasonal Cycle",
                                        )
        
        elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
            #If redo plot, delete the file
            if plot_name.is_file():
                plot_name.unlink()

            pf.month_vs_lat_plot(var, var_dict, plot_name, case_names, nicknames, climo_yrs, cases_coords, cases_monthly, vert_lev)
            adfobj.add_website_data(plot_name, f"MixRatio_{vert_lev}hPa", case_name, season="ANN",
                                        plot_type="WACCM",
                                        ext="SeasonalCycle_Mean",
                                        category="Seasonal Cycle",
                                        )
        #End if
    #End for



    #WACCM QBO
    #---------
    # Notify user that script has started:
    print("\n\t Making WACCM QBO...")
    plot_name = plot_loc / f"QBO_ANN_WACCM_SeasonalCycle_Mean.{plot_type}"
    if (not redo_plot) and plot_name.is_file():
        adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
        adfobj.add_website_data(plot_name, "QBO", case_name, season="ANN",
                                    plot_type="WACCM",
                                    ext="SeasonalCycle_Mean",
                                    category="Seasonal Cycle",
                                    )
    
    elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
        #If redo plot, delete the file
        if plot_name.is_file():
            plot_name.unlink()

        pf.waccm_qbo(plot_name, case_names, nicknames, cases_coords, merra2, syear_cases, eyear_cases)
        adfobj.add_website_data(plot_name, "QBO", case_name, season="ANN",
                                    plot_type="WACCM",
                                    ext="SeasonalCycle_Mean",
                                    category="Seasonal Cycle",
                                    )
    #End if

    #End plotting scripts




# Helper functions
##################

def make_zm_files(adfobj,hist_loc,hist_str,case_name,calc_var_list,syr,eyr,return_ds=True):
    """
    Make zonal mean files from history monthly files

    args:
    -----
       * hist_loc: Path object
          - place to find history files
       * case_name: str
          - name fo current case
       * calc_var_list: list
          - list of variables to compute and save zonal means
       * syr, eyr
          - start and end desired climo years
       * return_ds: boolean
          - return the dataset to xarray DataSet object   

    output: netcdf file
    ------
       - case specific file name with case data, saved to diagnostics plot location
    """

    save_path = adfobj.get_basic_info('cam_diag_plot_loc', required=True)
    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    save_path = adfobj.plot_location[0]

    #Convert output location string to a Path object:
    output_location = Path(save_path)
    #Check if analysis directory exists, and if not, then create it:
    if not output_location.is_dir():
        print(f"\t    {save_path} not found, making new directory")
        output_location.mkdir(parents=True)

    plot_locations = adfobj.plot_location[0]
    save_path = Path(plot_locations)
    zm_file = save_path / f"waccm_zm_{case_name}.nc"

    #Check if file exists. If so, open the file or make it if not
    if zm_file.exists():
        waccm_zm = xr.open_mfdataset(zm_file)
        
    else:
        print(f"\t  ...Making zonal mean average file from history files for {case_name}")
        h0_lists = []

        for yr in np.arange(int(syr),int(eyr)+1):
            h0_lists.append(sorted(glob.glob(f'{hist_loc}*{hist_str}.{yr}-*.nc')))

        h0_list = list(chain(*h0_lists))

        waccm_zm = xr.open_mfdataset(h0_list, use_cftime=True, data_vars=calc_var_list)
        waccm_zm = waccm_zm[calc_var_list].mean(dim='lon')


        attrs_dict = {"Description":"Zonal averaged mean of history files",
            "adf_user": adfobj.user,
            "climo_yrs": f"{syr}-{eyr}",
            "hist_loc":hist_loc,
        }
        waccm_zm = waccm_zm.assign_attrs(attrs_dict)

        #Output variable climatology to NetCDF-4 file:
        waccm_zm.to_netcdf(zm_file)

    if return_ds:
        return waccm_zm
########

def saber_data(filename, saber_vars):
    """

    """
    saber = {}
    saber_seasonal = {}
    saber_monthly = {}

    saber_ncfile = xr.open_dataset(filename, decode_times=True, use_cftime=True)
    saber_ncfile = saber_ncfile.rename({"latitude":"lat"})
    saber_ncfile = saber_ncfile.rename({"pressure":"lev"})

    #WARNING: there is no actual time information in the `time` coordinate!
    # - !! The assigned times are strictly from the file name !!
    start_date = datetime(2002, 1, 1)

    #Grab number of months
    num_months = len(saber_ncfile.time)

    # List to store datetime objects
    datetime_list = []

    # Generate datetime objects incrementally by month
    for i in range(num_months):
        new_date = start_date + relativedelta(months=i)

        # Set the day to the first day of the month
        datetime_list.append(new_date.replace(day=1))

    for index, var in enumerate(saber_vars):
        if var not in saber_seasonal:
            saber_seasonal[var] = {}
        if var not in saber_monthly:
            saber_monthly[var] = {}
        saber[var] = saber_ncfile[var]
        if index < len(saber_vars)-2:
            saber_ncfile[var] = saber_ncfile[var].assign_coords({"time": datetime_list})
            saber[var] = saber_ncfile[var]
            for season in seasons:
                saber_seasonal[var][season] = time_mean(saber_ncfile, saber_ncfile[var],
                                                        time_avg="season", interval=season,
                                                        is_climo=None, obs=True)
            for month in np.arange(1,13,1):
                saber_monthly[var][month_dict[month]] = time_mean(saber_ncfile, saber_ncfile[var],
                                                                time_avg="month", interval=month,
                                                                is_climo=None, obs=True)
    return saber, saber_monthly, saber_seasonal

########

def merra_data(filename, merra2_vars):
    """
    """

    merra2 = {}
    merra2_seasonal = {}
    merra2_monthly = {}

    merra_ncfile = xr.open_dataset(filename, decode_times=True, use_cftime=True)
    merra_ncfile = merra_ncfile.sel(time=merra_ncfile.time.values[0])
    merra_ncfile = merra_ncfile.rename({"time":"first-time"})
    merra_ncfile = merra_ncfile.rename({"record":"time"})

    for index, var in enumerate(merra2_vars):

        merra2[var] = merra_ncfile[var]
        if index < len(merra2_vars)-2:

            start_date = datetime(1980, 1, 1)

            # Number of months to generate
            num_months = len(merra_ncfile[var].time)

            # List to store datetime objects
            datetime_list = []

            # Generate datetime objects incrementally by month
            for i in range(num_months):
                new_date = start_date + relativedelta(months=i)
                datetime_list.append(new_date.replace(day=1))  # Set the day to the first day of the month

            merra_ncfile[var] = merra_ncfile[var].assign_coords({"time": datetime_list})
            if var not in merra2_seasonal:
                merra2_seasonal[var] = {}
            if var not in merra2_monthly:
                merra2_monthly[var] = {}
            merra2[var] = merra_ncfile[var]

            for season in seasons:
                merra2_seasonal[var][season] = time_mean(merra_ncfile, merra2[var], time_avg="season", interval=season, is_climo=None, obs=True)
            for month in np.arange(1,13,1):
                merra2_monthly[var][month_dict[month]] = time_mean(merra_ncfile, merra2[var], time_avg="month", interval=month, is_climo=None, obs=True)

    return merra2, merra2_monthly, merra2_seasonal
########

def time_mean(ncfile, data, time_avg, interval, is_climo=None, obs=False):
    """Calculates the time-weighted seasonal average (or average over all time).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        data to be averaged
    season : str, optional
        the season to extract from `data`
        If season is `ANN` or None, average all available time.
    is_climo : bool, optional
        If True, expects data to have time or month dimenion of size 12.
        If False, then 'time' must be a coordinate,
        and the `time.dt.days_in_month` attribute must be available.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        the average of `data` in season `season`

    Notes
    -----
    If the data is a climatology, the code will make an attempt to understand the time or month
    dimension, but will assume that it is ordered from January to December.
    If the data is a climatology and is just a numpy array with one dimension that is size 12,
    it will assume that dimension is time running from January to December.
    """
    if time_avg == "season":
        if interval is not None:
            assert interval in seasons, f"Unrecognized season string provided: '{interval}'"
        elif interval is None:
            interval = "ANN"

    try:
        month_length = data.time.dt.days_in_month
    except (AttributeError, TypeError):
        print("Nah, nope workingn nope")
        # do our best to determine the temporal dimension and assign weights
        if not is_climo:
            raise ValueError("Non-climo file provided, but without a decoded time dimension.")
        else:
            # CLIMO file: try to determine which dimension is month

            has_time = False
            if isinstance(data, xr.DataArray):
                has_time = 'time' in data.dims
                if not has_time:
                    if "month" in data.dims:
                        data = data.rename({"month":"time"})
                        has_time = True
            if not has_time:
                # this might happen if a pure numpy array gets passed in
                # --> assumes ordered January to December.
                assert ((12 in data.shape) and (data.shape.count(12) == 1)), f"Sorry, {data.shape.count(12)} dimensions have size 12, making determination of which dimension is month ambiguous. Please provide a `time` or `month` dimension."
                time_dim_num = data.shape.index(12)
                fakedims = [f"dim{n}" for n in range(len(data.shape))]
                fakedims[time_dim_num] = "time"
                data = xr.DataArray(data, dims=fakedims, attrs=data.attrs)

            timefix = pd.date_range(start='1/1/1999', end='12/1/1999', freq='MS') # generic time coordinate from a non-leap-year
            data = data.assign_coords({"time":timefix})
        month_length = data.time.dt.days_in_month
    #End try/except
    
    if not obs:
        syr = ncfile.time.dt.year.values[0]
        eyr = ncfile.time.dt.year.values[-2]

        data.attrs["year_range"] = f"{syr}-{eyr}"
        timefix = pd.date_range(start=f'1/1/{syr}', end=f'12/1/{eyr}', freq='MS')
        data['time']=timefix
    data.attrs[time_avg] = interval
    if time_avg == "season":
        data = data.sel(time=data.time.dt.month.isin(seasons[interval])) # directly take the months we want based on season kwarg
    if time_avg == "month":

        data = data.sel(time=data.time.dt.month.isin(interval)) # directly take the months we want


    return data.weighted(data.time.dt.daysinmonth).mean(dim='time', keep_attrs=True)
########