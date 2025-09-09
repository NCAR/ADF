
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

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
    
    saber_filename = seas_cyc_res['saber_file']
    #saber_filename = "SABER_monthly_2002-2014.nc"
    saber_file = adf_obs_loc / saber_filename
    
    merra_filename = seas_cyc_res['merra2_file']
    #merra_filename = "MERRA2_met_FAKESIES.nc"
    merra_file = adf_obs_loc / merra_filename

    if not adfobj.get_basic_info("compare_obs"):
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
        print("\t ** The seasonal cycle plots currently don't work when comparing against obs. Exiting script.")
        return
    # End if

    climo_yrs = [syear_cases, eyear_cases]

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
    #cont_ranges = comp_plots_dict['cont_ranges']

    #for cam_var in ["U","T"]:
    for cam_var in comp_plots_dict['cam_vars']:
        #Notify user of variable being plotted:
        print(f"\t - zonal mean maps for {var}")

        #for interval in [6,12,"DJF", "JJA"]:
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
            
                comparison_plots(plot_name, cam_var, case_names, nicknames, case_ds_dict,
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

            polar_cap_temp(plot_name, hemi, case_names, cases_coords, cases_monthly, merra2_monthly, pcap_dict)
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
    #vert_lev = 90
    try:
        vert_levs = var_dict[var]["plot_vert_levs"]
    except:
        #errmsg = f"Missing 'plot_vert_levs' in variable defaults file for '{var}'\n"
        #errmsg += "Please add it to the yaml file under 'lat_vs_month'"
        #print(errmsg)
        vert_levs = [90]

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

            #pf.cold_point_temp(plot_name, case_names, cases_coords, cases_monthly)
            month_vs_lat_plot(var, var_dict, plot_name, case_names, nicknames, climo_yrs, cases_coords, cases_monthly, vert_lev)
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
        #errmsg = f"Missing 'plot_vert_levs' in variable defaults file for '{var}'\n"
        #errmsg += "Please add it to the yaml file under 'lat_vs_month'"
        #print(errmsg)
        vert_levs = [90, 100]

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

            month_vs_lat_plot(var, var_dict, plot_name, case_names, nicknames, climo_yrs, cases_coords, cases_monthly, vert_lev)
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

        waccm_qbo(plot_name, case_names, nicknames, cases_coords, merra2, syear_cases, eyear_cases)
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
            h0_lists.append(sorted(glob.glob(f'{hist_loc}/*{hist_str}.{yr}-*.nc')))

        h0_list = list(chain(*h0_lists))

        waccm_zm = xr.open_mfdataset(h0_list, use_cftime=True, data_vars=calc_var_list)
        waccm_zm = waccm_zm[calc_var_list].mean(dim='lon')


        attrs_dict = {"Description":"Zonal averaged mean of history files",
            "adf_user": adfobj.user,
            "climo_yrs": f"{syr}-{eyr}",
            "hist_loc":hist_loc,
        }
        waccm_zm = waccm_zm.assign_attrs(attrs_dict)

        #Output zonal mean climatology to NetCDF-4 file:
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
        #data.attrs[time_avg] = interval
        data = data.sel(time=data.time.dt.month.isin(seasons[interval])) # directly take the months we want based on season kwarg
    if time_avg == "month":

        data = data.sel(time=data.time.dt.month.isin(interval)) # directly take the months we want


    return data.weighted(data.time.dt.daysinmonth).mean(dim='time', keep_attrs=True)
########

def comparison_plots(plot_name, cam_var, case_names, case_nicknames, case_ds_dict, obs_ds_dict, time_avg, interval, comp_plots_dict, obs_cam_vars):
    """

    """

    #Get plotting details for variable
    levs = np.arange(*comp_plots_dict[cam_var]["levs"])
    diff_levs = np.arange(*comp_plots_dict[cam_var]["diff_levs"])
    units = comp_plots_dict[cam_var]["units"]

    #Grab obs variable corresponding to CAM variable
    saber_var = obs_cam_vars['saber'][cam_var]
    merra_var = obs_cam_vars['merra'][cam_var]

    font_size = 6

    #Get number of test cases (number of columns)
    casenum = len(case_names)

    #Number of obs to compare
    #Currently, just compared to MERRA2 and SABER
    obsnum = 2
    nrows = obsnum+1

    #Set up plot
    fig = plt.figure(figsize=(casenum*4,nrows*5))

    for idx,case_name in enumerate(case_names):

        data_coords = case_ds_dict["coords"][case_name]

        data_lev = data_coords['lev']
        data_lat = data_coords['lat']

        #Set lat/lev grid for plotting
        [lat_grid, lev_grid] = np.meshgrid(data_lev,data_lat)

        if time_avg == "season":

            data_array = case_ds_dict["seasonal"][case_name][cam_var][interval]

            #Make Obs interpolated field from case
            merra_ds = obs_ds_dict["seasonal"]["merra"][merra_var][interval]
            merra_rfield = merra_ds.interp(lat=data_lat, lev=data_lev, method='linear')

            saber_ds = obs_ds_dict["seasonal"]["saber"][saber_var][interval]
            saber_rfield = saber_ds.interp(lat=data_lat, lev=data_lev, method='linear')

        if time_avg == "month":
            case_ds_dict["monthly"]
            str_month = interval
            data_array = case_ds_dict["monthly"][case_name][cam_var][str_month]

            #Make Obs interpolated fields from case
            merra_ds = obs_ds_dict["monthly"]["merra"][merra_var][str_month]
            merra_rfield = merra_ds.interp(lat=data_lat, lev=data_lev, method='linear')

            saber_ds = obs_ds_dict["monthly"]["saber"][saber_var][str_month]
            saber_rfield = saber_ds.interp(lat=data_lat, lev=data_lev, method='linear')

        #Case plots (contours and contour fill)
        #######################################

        #Set up set of axes for first row
        ax = fig.add_subplot(nrows, casenum, idx+1)

        #Plot case contour fill
        cf=plt.contourf(lev_grid, lat_grid,
                        data_array.transpose(transpose_coords=True),
                        levels=levs, cmap='RdYlBu_r')

        #Plot case contours (for highlighting)
        contour = plt.contour(lev_grid, lat_grid,
                        data_array.transpose(transpose_coords=True),
                    colors="black",linewidths=0.5,levels=levs,zorder=100)
        fmt = {lev: '{:.0f}'.format(lev) for lev in contour.levels}
        ax.clabel(contour, contour.levels[::2], inline=True, fmt=fmt, fontsize=8)

        #Format axes
        plt.yscale("log")


        # Find the next value below highest vertical level
        prev_major_tick = 10 ** (np.floor(np.log10(np.nanmin(data_lev))))
        y_lims = [float(lim) for lim in [1e3,prev_major_tick]]
        ax.set_ylim(y_lims)

        plt.xticks(np.arange(-90,91,45),rotation=40)
        ax.tick_params(axis='x', labelsize=8)
        if idx > 0:
            plt.yticks([])
        else:
            plt.ylabel('hPa',fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

        #Set individual plot title
        plt.title(case_nicknames[idx], fontsize=font_size)

        #Make colorbar on last plot only
        if idx == casenum-1:
            axins = inset_axes(ax,
                        width="5%",
                        height="80%",
                        loc='center right',
                        borderpad=-1.5
                       )
            cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label=units)
            cbar.ax.tick_params(axis='y', labelsize=8)
            cbar.set_label(units, fontsize=10, labelpad=1)

        #Difference with MERRA2 and MERRA2 contours
        ###########################################

        #Set up new set of axes for second row
        ax = fig.add_subplot(nrows, casenum, casenum+idx+1)

        #Plot interpolated contour
        contour = plt.contour(lev_grid, lat_grid, merra_rfield.transpose(transpose_coords=True),
                    colors='black', levels=levs,
                    negative_linestyles='dashed', linewidths=.5, alpha=0.5)

        #Add a legend for the contour lines for first plot only
        legend_elements = [Line2D([0], [0],
                               color=contour.collections[0].get_edgecolor(),
                               label=f'MERRA2 interp {cam_var}')]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=5, bbox_to_anchor=(1., 1.))
        #End if

        #Plot difference contour fill
        cf=plt.contourf(lev_grid, lat_grid,
                        (data_array-merra_rfield).transpose(transpose_coords=True),
                        levels=diff_levs, cmap='RdYlBu_r')

        #Plot case contours (for highlighting)
        contour = plt.contour(lev_grid, lat_grid,
                        (data_array-merra_rfield).transpose(transpose_coords=True),
                    colors="black",linewidths=0.5,levels=diff_levs[::2],zorder=100)
        fmt = {lev: '{:.0f}'.format(lev) for lev in contour.levels}
        ax.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=8)

        #Format axes
        plt.yscale("log")

        # Find the next value below highest vertical level
        prev_major_tick = 10 ** (np.floor(np.log10(np.nanmin(data_lev))))
        y_lims = [float(lim) for lim in [1e3,prev_major_tick]]
        ax.set_ylim(y_lims)


        plt.xticks(np.arange(-90,91,45),rotation=40)
        ax.tick_params(axis='x', labelsize=8)
        if idx > 0:
            plt.yticks([])
        else:
            plt.ylabel('hPa',fontsize=10)

        ax.tick_params(axis='y', labelsize=8)

        #Set individual plot title
        local_title = f'{case_nicknames[idx]}\n {delta_symbol} from MERRA2'
        plt.title(local_title, fontsize=font_size)

        #Make colorbar on last plot only
        if idx == casenum-1:
            axins = inset_axes(ax,
                        width="5%",
                        height="80%",
                        loc='center right',
                        borderpad=-1.5
                       )
            cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label=units)
            cbar.ax.tick_params(axis='y', labelsize=8)
            cbar.set_label(units, fontsize=10, labelpad=1)

        #Difference with SABER and SABER contours (Temp only)
        #####################################################
        if cam_var == "T":
            #Set up new set of axes for third row
            ax = fig.add_subplot(nrows, casenum, (casenum*2)+idx+1)

            #Plot interpolated contour
            contour = plt.contour(lev_grid, lat_grid, saber_rfield.transpose(transpose_coords=True),
                        colors='black', levels=levs,
                        negative_linestyles='dashed', linewidths=.5, alpha=0.5)
            #if idx == 0:
            #Add a legend for the contour lines for first plot only
            legend_elements = [Line2D([0], [0],
                                color=contour.collections[0].get_edgecolor(),
                                label='SABER interp T')]

            ax.legend(handles=legend_elements, loc='upper right', fontsize=5, bbox_to_anchor=(1., 1.))
            #End if

            #Plot difference contour fill
            cf=plt.contourf(lev_grid, lat_grid,
                            (data_array-saber_rfield).transpose(transpose_coords=True),
                            levels=diff_levs, cmap='RdYlBu_r')

            #Plot case contours (for highlighting)
            contour = plt.contour(lev_grid, lat_grid,
                            (data_array-saber_rfield).transpose(transpose_coords=True),
                        colors="black",linewidths=0.5,levels=diff_levs,zorder=100)
            fmt = {lev: '{:.0f}'.format(lev) for lev in contour.levels}
            ax.clabel(contour, contour.levels[::2], inline=True, fmt=fmt, fontsize=8)

            #Format axes
            plt.yscale("log")

            # Find the next value below highest vertical level
            prev_major_tick = 10 ** (np.floor(np.log10(np.nanmin(data_lev))))
            y_lims = [float(lim) for lim in [1e3,prev_major_tick]]

            ax.set_ylim(y_lims)

            plt.xticks(np.arange(-90,91,45),rotation=40)
            ax.tick_params(axis='x', labelsize=8)
            if idx > 0:
                plt.yticks([])
            else:
                plt.ylabel('hPa',fontsize=10)
            ax.tick_params(axis='y', labelsize=8)

            #Set individual plot title
            local_title = f'{case_nicknames[idx]}\n {delta_symbol} from SABER'
            plt.title(local_title, fontsize=font_size)

            #Make colorbar on last plot only
            if idx == casenum-1:
                axins = inset_axes(ax,
                            width="5%",
                            height="80%",
                            loc='center right',
                            borderpad=-1.5
                        )
                cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label=units)
                cbar.ax.tick_params(axis='y', labelsize=8)
                # Set the font size for the colorbar label
                cbar.set_label(units, fontsize=10, labelpad=1)

    #Set up main plot title
    plt.suptitle(f"Zonal Mean {cam_var} - {interval}",fontsize=12,y=0.91)
    
    fig.savefig(plot_name, bbox_inches='tight', dpi=300)

    plt.close()



def polar_cap_temp(plot_name, hemi, case_names, cases_coords, cases_monthly, merra2_monthly, pcap_dict):
    """
    """
    if "levs" in pcap_dict["T"][f"{hemi}h"]:
        levs = np.arange(*pcap_dict["T"][f"{hemi}h"]["levs"])
    else:
        levs = np.arange(140,301,10)
    
    if "diff_levs" in pcap_dict["T"][f"{hemi}h"]:
        diff_levs = np.arange(*pcap_dict["T"][f"{hemi}h"]["diff_levs"])
    else:
        diff_levs = np.arange(-10,11,1)

    #Get number of test cases (number of columns)
    casenum = len(case_names)

    font_size = 6
    if hemi == "s":
        slat = -90
        nlat = -60
        title_ext = f"{np.abs(nlat)}-{np.abs(slat)}\u00b0S"

    if hemi == "n":
        slat = 60
        nlat = 90
        title_ext = f"{slat}-{nlat}\u00b0N"

    nrows = 2
    fig = plt.figure(figsize=(casenum*4,nrows*5))

    for idx,case_name in enumerate(case_names):
        ds = cases_coords[case_name]
        ds_month = cases_monthly[case_name]

        rfield_seas = np.zeros((12,len(ds['lev']),len(ds['lat'])))
        rfield_seas = xr.DataArray(rfield_seas, dims=['month','lev', 'lat'],
                                            coords={'month': np.arange(1,13,1),
                                                    'lev': ds['lev'],
                                                    'lat': ds['lat']})

        case_seas = np.zeros((12,len(ds['lev']),len(ds['lat'])))
        case_seas = xr.DataArray(case_seas, dims=['month','lev', 'lat'],
                                 coords={'month': np.arange(1,13,1),
                                         'lev': ds['lev'],
                                         'lat': ds['lat']})
        #Make array of monthly temp data
        for m in range(0,12):
            rfield_seas[m] = merra2_monthly['T'][month_dict[m+1]].interp(lat=ds['lat'], lev=ds['lev'],
                                                                method='linear')
            case_seas[m] = ds_month['T'][month_dict[m+1]]

        #Average over set of latitudes
        merra2_pcap = pf.coslat_average(rfield_seas,slat,nlat)
        case_pcap = pf.coslat_average(case_seas,slat,nlat)

        #
        [time_grid, lev_grid] = np.meshgrid(ds['lev'],np.arange(0,12))

        #Set up first row - Temps
        ax = fig.add_subplot(nrows, casenum, idx+1)
        cf=plt.contourf(lev_grid, time_grid, case_pcap,
                        levels=levs,cmap='RdYlBu_r'
                       ) #np.arange(-10,11,1)
        c0=plt.contour(lev_grid, time_grid, case_pcap, colors='grey',
                           levels=levs[::2],
                           negative_linestyles='dashed',
                           linewidths=.5, alpha=1)
        fmt = {lev: '{:.0f}'.format(lev) for lev in c0.levels}
        ax.clabel(c0, c0.levels, inline=True, fmt=fmt, fontsize=8)

        #Format the axes
        plt.yscale("log")
        ax.set_ylim(300,1)
        ax.set_yticks([300,100,30,10,1])
        ax.set_xticks(np.arange(0,12,2))#,rotation=40
        ax.set_xticklabels(('Jan','Mar','May','Jul','Sep','Nov'),rotation=40,fontsize=8)
        if idx > 0:
            plt.yticks([])
        else:
            ax.set_yticklabels(["","$10^{2}$","","$10^{1}$",""],fontsize=10)
            plt.ylabel('hPa',fontsize=10)
        
        #Set title
        local_title=f"{case_names[idx]}"
        plt.title(local_title, fontsize=font_size)

        #Make colorbar on last plot only
        if idx == casenum-1:
            axins = inset_axes(ax,
                        width="5%",
                        height="80%",
                        loc='center right',
                        borderpad=-1.5
                       )
            cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label="K")
            cbar.ax.tick_params(axis='y', labelsize=8)
            cbar.set_label("K", fontsize=10, labelpad=1)


        #Set up second row - Temp anomlies and Merra2 contours
        ax = fig.add_subplot(nrows, casenum, casenum+idx+1)
        clevs = np.arange(-10,11,1)
        cf=plt.contourf(lev_grid, time_grid, (case_pcap-merra2_pcap),
                        levels=diff_levs,cmap='RdYlBu_r'
                       ) 
        c0=plt.contour(lev_grid, time_grid, (case_pcap-merra2_pcap), colors='grey',
                           levels=clevs[::3],
                           negative_linestyles='dashed',
                           linewidths=.5, alpha=1)
        fmt = {lev: '{:.0f}'.format(lev) for lev in c0.levels}
        ax.clabel(c0, c0.levels, inline=True, fmt=fmt, fontsize=8)

        c=plt.contour(lev_grid, time_grid, merra2_pcap, colors='black',
                           levels=levs,
                           negative_linestyles='dashed',
                           linewidths=.5, alpha=0.5)

        #Add a legend for the contour lines for first plot only
        legend_elements = [Line2D([0], [0],
                               color=c.collections[0].get_edgecolor(),
                               label='MERRA2 interp T')]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=5, bbox_to_anchor=(1., 1.))
        #Format the axes
        plt.yscale("log")
        ax.set_ylim(300,1)
        ax.set_yticks([300,100,30,10,1])
        ax.set_xticks(np.arange(0,12,2))#,rotation=40
        ax.set_xticklabels(('Jan','Mar','May','Jul','Sep','Nov'),rotation=40,fontsize=8)
        if idx > 0:
            plt.yticks([])
        else:
            ax.set_yticklabels(["","$10^{2}$","","$10^{1}$",""],fontsize=10)
            plt.ylabel('hPa',fontsize=10)

        #Set title
        local_title=f"{case_names[idx]}\n {delta_symbol} from MERRA2"
        plt.title(local_title, fontsize=font_size)

        #Make colorbar on last plot only
        if idx == casenum-1:
            axins = inset_axes(ax,
                                width="5%",
                                height="80%",
                                loc='center right',
                                borderpad=-1.5
                               )

            space = abs(diff_levs[-2]-diff_levs[-1])
            cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label='K', ticks=np.arange(diff_levs[1],diff_levs[-2]+space,3))
            cbar.ax.tick_params(axis='y', labelsize=8)
            # Set the font size for the colorbar label
            cbar.set_label("K", fontsize=10, labelpad=1)

    fig.suptitle(f"{hemi.upper()}H Polar Cap Temp Anomolies - {title_ext}",fontsize=12,y=0.93) #,horizontalalignment="center"
 
    fig.savefig(plot_name, bbox_inches='tight', dpi=300)
    
    #Close plots:
    plt.close()
########


def month_vs_lat_plot(var, var_dict, plot_name, case_names, case_nicknames, climo_yrs, case_runs, cases_monthly, vert_lev):
    """
    """

    ahh = []
    for i in list(month_dict.keys())[::3]:
        ahh.append(month_dict[i].lower().capitalize())

    #Grab values for the month vs lat plot in variable defaults yaml file
    slat = var_dict[var]["slat"]
    nlat = var_dict[var]["nlat"]
    cmap = var_dict[var]["cmap"]
    diff_cmap = var_dict[var]["diff_cmap"]
    levs = np.arange(*var_dict[var]["levels"])
    
    units = var_dict[var]["units"]
    title = var_dict[var]["title"]
    y_labels = var_dict[var]["y_labels"]
    tick_inter = var_dict[var]["tick_inter"]

    # create figure:
    fig = plt.figure(figsize=(18,10))

    # LAYOUT WITH GRIDSPEC
    gs = mpl.gridspec.GridSpec(4, 8, wspace=4,hspace=0.5)
    ax1 = plt.subplot(gs[0:2, :4])#, **cp_info['subplots_opt'])
    ax2 = plt.subplot(gs[0:2, 4:])#, **cp_info['subplots_opt'])
    ax3 = plt.subplot(gs[2:, 2:6])#, **cp_info['subplots_opt'])
    ax = [ax1,ax2,ax3]

    #
    pcap_vals = {}

    #for run in range(len(runs)):
    for idx,case_name in enumerate(case_names):
        ds = case_runs[case_name]
        ds_month = cases_monthly[case_name]

        #Make 24 months so we can have Jan-Dec repeated twice
        case_seas = np.zeros((25,len(ds['lev']),len(ds['lat'])))
        case_seas = xr.DataArray(case_seas, dims=['month','lev', 'lat'],
                                 coords={'month': np.arange(1,26,1),
                                         'lev': ds['lev'],
                                         'lat': ds['lat']})
        #Make array of monthly temp data
        for m in range(0,25):
            month = m
            if m > 11:
                month = m-12
            if month == 12:
                month = 0

            case_seas[m] = ds_month[var][month_dict[month+1]]

        #Average over set of latitudes
        case_pcap = pf.coslat_average(case_seas,slat,nlat)
        case_pcap = case_seas.sel(lev=vert_lev,method="nearest").sel(lat=slice(slat, nlat))
        if var == "Q":
            case_pcap = case_pcap*1e6

        pcap_vals[case_name] = case_pcap

        #
        [time_grid, lat_grid] = np.meshgrid(ds['lat'].sel(lat=slice(slat, nlat)),
                                            np.arange(0,25))
        #Set up plot
        cf=ax[idx].contourf(lat_grid, time_grid, (case_pcap),
                        levels=levs,
                        cmap=cmap,#zorder=100
                      )
        c=ax[idx].contour(lat_grid, time_grid, (case_pcap),
                        levels=levs,
                        colors='k',linewidths=0.5,alpha=0.5
                      )

        # Format contour labels
        if var == "T":
            fmt = {lev: '{:.0f}'.format(lev) for lev in c.levels}
        else:
            fmt = {lev: '{:.1f}'.format(lev) for lev in c.levels}
        ax[idx].clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)

        #Add a horizontal line at 0 degrees latitude
        ax[idx].axhline(0, color='grey', linestyle='-',zorder=200,alpha=0.7)

        #Format the x-axis
        ax[idx].set_xticks(np.arange(0,25,3))#,rotation=40
        ax[idx].set_xticklabels(ahh+ahh+["Jan"],rotation=40)

        #Set title
        if idx == 0:
            plot_title = "$\mathbf{Test}:$"+f"{case_nicknames[0]}\nyears: {climo_yrs[0][0]}-{climo_yrs[1][0]}"
        if idx == 1:
            plot_title = "$\mathbf{Baseline}:$"+f"{case_nicknames[1]}\nyears: {climo_yrs[0][1]}-{climo_yrs[1][1]}"
        ax[idx].set_title(plot_title, loc='left', fontsize=10)

        ax[idx].set_ylabel('Latitude',fontsize=10)

        #Format the y-axis
        ax[idx].set_yticks(np.arange(slat,nlat+1,tick_inter))
        ax[idx].set_yticklabels(y_labels,fontsize=10)

        axins = inset_axes(ax[idx],
                                width="3%",
                                height="80%",
                                loc='center right',
                                borderpad=-2
                               )
        cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label=units,
                                    #ticks=levs
                                   )
        cbar.add_lines(c)
        cbar.ax.tick_params(axis='y', labelsize=8)
        # Set the font size for the colorbar label
        cbar.set_label(units, fontsize=10, labelpad=1)
    #End cases


    #Difference Plots
    #----------------
    idx = 2
    diff_pcap = pcap_vals[case_names[0]] - pcap_vals[case_names[1]]

    #
    [time_grid, lat_grid] = np.meshgrid(ds['lat'].sel(lat=slice(slat, nlat)),
                                            np.arange(0,25))
    #Set up plot
    #ax = fig.add_subplot(nrows, ncols, idx+1)

    if "diff_levels" in var_dict[var]:
        diff_levs = np.arange(*var_dict[var]["diff_levels"])
    else:
        # set a symmetric color bar for diff:
        absmaxdif = np.max(np.abs(diff_pcap))
        # set levels for difference plot:
        diff_levs = np.linspace(-1*absmaxdif, absmaxdif, 21)

    cf=ax[idx].contourf(lat_grid, time_grid, (diff_pcap),
                        levels=diff_levs,
                        cmap=diff_cmap,#zorder=100
                      )
    c=ax[idx].contour(lat_grid, time_grid, (diff_pcap),
                        levels=diff_levs[::2],
                        colors='k',linewidths=0.5,alpha=0.5
                      )

    # Format contour labels
    if var == "T":
        fmt = {lev: '{:.0f}'.format(lev) for lev in c.levels}
    else:
        fmt = {lev: '{:.1f}'.format(lev) for lev in c.levels}
    ax[idx].clabel(c, c.levels, inline=True, fmt=fmt, fontsize=8)

    #Add a horizontal line at 0 degrees latitude
    #plt.axhline(0, color='grey', linestyle='-',zorder=200,alpha=0.7)
    ax[idx].axhline(0, color='grey', linestyle='-',zorder=200,alpha=0.7)

    #Format the x-axis
    ax[idx].set_xticks(np.arange(0,25,3))#,rotation=40
    ax[idx].set_xticklabels(ahh+ahh+["Jan"],rotation=40)

    #Set title
    local_title="$\mathbf{Test} - \mathbf{Baseline}$" #"Test - Baseline"#case_names[idx]

    ax[idx].set_title(local_title, fontsize=10)

    ax[idx].set_ylabel('Latitude',fontsize=10)

    #Format the y-axis
    ax[idx].set_yticks(np.arange(slat,nlat+1,tick_inter))
    ax[idx].set_yticklabels(y_labels,fontsize=10)

       

    axins = inset_axes(ax[idx],
                                width="3%",
                                height="80%",
                                loc='center right',
                                borderpad=-2
                               )
    cbar = fig.colorbar(cf, cax=axins, orientation="vertical", label=units,
                                    #ticks=levs
                                   )
    cbar.add_lines(c)
    cbar.ax.tick_params(axis='y', labelsize=8)
    # Set the font size for the colorbar label
    cbar.set_label(units, fontsize=10, labelpad=1)

    plt.suptitle(f"{title} - {vert_lev}hPa",fontsize=20,y=0.97)

    fig.savefig(plot_name, bbox_inches='tight', dpi=300)


########

#WACCM QBO
import numpy as np

def qbo_amplitude(data):
    """
    Calculate the QBO amplitude
    """
    
    from scipy.signal import convolve
    
    boxcar = np.ones((6, 1)) / 6
    filtered_data = convolve(data, boxcar, mode='valid')
    amplitude=np.std(filtered_data, axis=0)
    
    return amplitude


def qbo_frequency(data):
    """
    Calculate the QBO frequency
    """
    
    [dt,dx]=data.shape
    dt2=int(dt/2)
    f=1*np.arange(0,dt2+1,1)/dt
    f=f[1:]
    f = np.tile(f, (dx, 1)).swapaxes(0,1)
    
    fft_data = np.fft.fft(data, axis=0)
    fft_data = fft_data[1:dt2+1,:]
    
    power_spectrum = np.abs(fft_data)**2
    
    period=np.sum(power_spectrum*(1/f),axis=0)/np.sum(power_spectrum,axis=0)
    
    return period
  

def waccm_qbo(plot_name, case_names, nicknames, case_runs, merra2, syear_cases, eyear_cases):
    """
    
    """

    def format_side_axes(axes, side_axis, x, merra_data, data=None, case_lev=None, merra=False):
        """
        Format the period and amplitiude side axes
        """
        axes[side_axis].plot(merra_data,merra2['lev'],color='k')# s=1
        axes[side_axis].set_ylim(y_lims[0],y_lims[1])
        axes[side_axis].set_yscale("log")

        if merra==False:
            axes[side_axis].plot(data,case_lev)#linewidths=1
        if x == "period":
            axes[side_axis].set_xlim(0,40)
            axes[side_axis].set_xticks(np.arange(0,41,10))
            axes[side_axis].set_xticklabels(np.arange(0,41,10),fontsize=8)
            axes[side_axis].set_xlabel('months',fontsize=10)
        if x == "amplitude":
            axes[side_axis].set_xlim(0,20)
            axes[side_axis].set_xticks(np.arange(0,21,5))
            axes[side_axis].set_xticklabels(np.arange(0,21,5),fontsize=8)
            axes[side_axis].set_xlabel('m/s',fontsize=10)
        axes[side_axis].set_yticks([])
        return axes

    def advance_string(input_string):
        advanced_chars = [chr(ord(char) + 3) if char.isalpha() else char for char in input_string]
        advanced_string = ''.join(advanced_chars)
        return advanced_string


    #Build subplot mosiac based off number of CAM cases
    input_string0 = 'AAAABC'
    ahh = []
    ahh.append(input_string0)
    for i in range(len(case_names)):
        if i ==0:
            input_string = advance_string(input_string0)
        else:
            input_string = advance_string(input_string)
            input_string = f"{input_string}"
        ahh.append(input_string)

    main_key = []
    side1_key = []
    side2_key = []
    mos_str = input_string0
    for idx,i in enumerate(ahh):
        if idx != 0:
            mos_str += f";{i}"
        main_key.append(i[0])
        side1_key.append(i[-2])
        side2_key.append(i[-1])

    fig, axes = plt.subplot_mosaic(mos_str,figsize=(12,5*len(case_names)))

    y = 1.00
    y_lims = [100,0.1]

    contour_levels = np.arange(-35, 36, 2.5)

    #Plot MERRA2 last; this will be based on number of CAM cases
    merra_idx = len(case_names)

    #nt = 108
    nt = 120
    plotdata = pf.coslat_average(merra2['U'],-10,10)

    plotdata_clip = np.clip(np.abs(plotdata), None, 35)
    plotdata=np.sign(plotdata)*plotdata_clip
    [time_grid, lev_grid] = np.meshgrid(merra2['lev'],np.arange(1,nt+1,1))

    start_ind=240
    end_ind=start_ind+nt

    data = plotdata[start_ind:end_ind,:]
    cf = axes[main_key[merra_idx]].contourf(lev_grid, time_grid, data,
                                        levels=contour_levels, cmap='RdBu_r')

    c = axes[main_key[merra_idx]].contour(lev_grid, time_grid, data, alpha=0.75,linewidths=0.3,
                                        levels=contour_levels[::5], colors='k',linestyles=['dashed' if val < 0 else 'solid' for val in np.unique(data)])
    # add contour labels
    lb = plt.clabel(c, fontsize=6, inline=True, fmt='%r')

    axins = inset_axes(axes[main_key[merra_idx]], width="100%", height="5%", loc='lower center', borderpad = -3.5)
    cbar = fig.colorbar(cf, cax=axins, orientation="horizontal", label="m/s",
                                        ticks=contour_levels[::2])
    cbar.ax.tick_params(axis='x', labelsize=8)
    # Set the font size for the colorbar label
    cbar.set_label("m/s", fontsize=10, labelpad=1)

    axes[main_key[merra_idx]].set_ylim(y_lims[0],y_lims[1])
    axes[main_key[merra_idx]].set_yscale("log")
    axes[main_key[merra_idx]].set_ylabel('hPa',fontsize=10)
    axes[main_key[merra_idx]].tick_params(axis='y', labelsize=8)
    axes[main_key[merra_idx]].set_title("MERRA2",y=y,fontsize=10)
    axes[main_key[merra_idx]].set_xticks(np.arange(1,nt+1,12))#,rotation=40
    axes[main_key[merra_idx]].set_xticklabels(np.arange(1,nt+1,12),rotation=40)

    start_year = int(str(plotdata[start_ind].time.values)[0:4])
    axes[main_key[merra_idx]].set_xticklabels(np.arange(start_year,start_year+(nt/12),1).astype(int),fontsize=8)

    #MERRA QBO Amplitude side axis
    amp_m = qbo_amplitude(plotdata)
    axes = format_side_axes(axes,side1_key[merra_idx],"amplitude",amp_m,merra=True)

    #MERRA QBO Period side axis
    period_m = qbo_frequency(plotdata)
    axes = format_side_axes(axes,side2_key[merra_idx],"period",period_m,merra=True)

    #Loop over CAM case data
    for idx,case_name in enumerate(case_names):
        case_data = case_runs[case_name]
        nickname = nicknames[idx]
        yrs = syear_cases[idx]
        last_yr = eyear_cases[idx]
        
        #Get number of time steps
        nt = len(case_data['time'])
        #If the number is greater than 10 years, clip it to 10 years?
        if nt > 120:
            nt_sub = 120
        else:
            nt_sub = nt

        [time_grid, lev_grid] = np.meshgrid(case_data['lev'],np.arange(0,nt_sub+12,1))

        contour_levels = np.arange(-35, 35, 2.5)

        plotdata = pf.coslat_average(case_data['U'],-10,10)
        plotdata_clip = np.clip(np.abs(plotdata), None, 35)
        plotdata=np.sign(plotdata)*plotdata_clip

        #TODO: this will need to be adjusted??
        #Curently this is finding (start_idx)th month and then going out 9 years
        #QUESTION: what if the data doesn't have 9 years? - we will need to clip this...
        start_idx = 0 #119-24
        end_idx = start_idx+nt+1

        yr0 = int(yrs+int(start_idx/12))

        cf = axes[main_key[idx]].contourf(lev_grid[start_idx:end_idx-1,:], time_grid[start_idx:end_idx-1,:], plotdata[start_idx:end_idx-1,:],
                                    levels=contour_levels, cmap='RdBu_r')

        c = axes[main_key[idx]].contour(lev_grid[start_idx:end_idx-1,:], time_grid[start_idx:end_idx-1,:], plotdata[start_idx:end_idx,:],
                                    levels=contour_levels[::5], colors='k',alpha=0.75,linewidths=0.5)
        # add contour labels
        lb = plt.clabel(c, fontsize=6, inline=True, fmt='%r')

        axes[main_key[idx]].set_ylim(y_lims[0],y_lims[1])
        axes[main_key[idx]].set_yscale("log")
        axes[main_key[idx]].set_ylabel('hPa',fontsize=10)
        axes[main_key[idx]].tick_params(axis='y', labelsize=8)
        axes[main_key[idx]].set_title(nickname,y=y,fontsize=10)

        # Set the x-axis limits
        axes[main_key[idx]].set_xticks(range(0, 12*11, 12))
        axes[main_key[idx]].set_xlim(0, 12*10)

        need_yrs = []
        for i in range((yr0+10)-last_yr):
            need_yrs.append(yr0+i)

        axes[main_key[idx]].set_xticklabels(np.arange(yr0, yr0+11, 1), fontsize=8)

        #Case QBO Amplitude side axis
        amp = qbo_amplitude(plotdata)
        axes = format_side_axes(axes, side1_key[idx], "amplitude",amp_m,amp,case_data['lev'])

        #Case QBO Period side axis
        period = qbo_frequency(plotdata)
        axes = format_side_axes(axes, side2_key[idx], "period",period_m,period,case_data['lev'])

        #Label first row of side axes only
        if idx==0:
            axes[side1_key[idx]].set_title('Amplitude',y=y,fontsize=12)
            axes[side2_key[idx]].set_title('Period',y=y,fontsize=12)

    # Adjust the vertical spacing (hspace)
    plt.subplots_adjust(hspace=0.35)

    fig.suptitle(f"QBO Diagnostics",fontsize=16,y=0.93,horizontalalignment="center")

    fig.savefig(plot_name, bbox_inches='tight', dpi=300)

########