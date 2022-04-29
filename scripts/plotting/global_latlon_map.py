from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

def global_latlon_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model fields with continental overlays.

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
    var_list         -> List of CAM output variables provided by "diag_var_list"
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".

    opt              -> optional,
                        if dict : that has keys that are variable names and values that are plotting preferences/defaults.
                        if str  : path to a YAML file that conforms to the dict option.
    """

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("  Generating lat/lon maps...")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    plot_locations = adfobj.plot_location

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):

        #Extract variable-obs dictionary:
        var_obs_dict = adfobj.var_obs_dict

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("No observations found to plot against, so no lat/lon maps will be generated.")
            return

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = adfobj.get_baseline_info("cam_climo_loc", required=True)

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"NOTE: Plot type is set to {plot_type}")
    #-----------------------------------------

    #Set data path variables:
    #-----------------------
    mclimo_rg_loc = Path(model_rgrid_loc)
    if not adfobj.compare_obs:
        dclimo_loc  = Path(data_loc)
    #-----------------------

    #Determine if user wants to plot 3-D variables on
    #pressure levels:
    pres_levs = adfobj.get_basic_info("plot_press_levels")

    #Determine if user wants monthly weights to be applied
    #to the seasonal averages:
    weight_season = adfobj.get_basic_info("weight_season")

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    # probably want to do this one variable at a time:
    for var in var_list:

        if adfobj.compare_obs:
            #Check if obs exist for the variable:
            if var in var_obs_dict:
                #Note: In the future these may all be lists, but for
                #now just convert the target_list.
                #Extract target file:
                dclimo_loc = var_obs_dict[var]["obs_file"]
                #Extract target list (eventually will be a list, for now need to convert):
                data_list = [var_obs_dict[var]["obs_name"]]
                #Extract target variable name:
                data_var = var_obs_dict[var]["obs_var"]
            else:
                dmsg = f"No obs found for variable `{var}`, lat/lon map plotting skipped."
                adfobj.debug_log(dmsg)
                continue
        else:
            #Set "data_var" for consistent use below:
            data_var = var
        #End if

        #Notify user of variable being plotted:
        print("\t - lat/lon maps for {}".format(var))

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_latlon_map: Found variable defaults for {var}")

        else:
            vres = {}

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #For now, only grab one file (but convert to list for use below)
                oclim_fils = [dclimo_loc]
            else:
                oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))

            if len(oclim_fils) > 1:
                oclim_ds = xr.open_mfdataset(oclim_fils, combine='by_coords')
            elif len(oclim_fils) == 1:
                sfil = str(oclim_fils[0])
                oclim_ds = xr.open_dataset(sfil)
            else:
                print("WARNING: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{var}_*.nc")
                continue

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                #Set output plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print("    {} not found, making new directory".format(plot_loc))
                    plot_loc.mkdir(parents=True)

                # load re-gridded model files:
                mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))

                if len(mclim_fils) > 1:
                    mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
                else:
                    mclim_ds = xr.open_dataset(mclim_fils[0])

                #Extract variable of interest
                odata = oclim_ds[data_var].squeeze()  # squeeze in case of degenerate dimensions
                mdata = mclim_ds[var].squeeze()

                # APPLY UNITS TRANSFORMATION IF SPECIFIED:
                # NOTE: looks like our climo files don't have all their metadata
                mdata = mdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                # update units
                mdata.attrs['units'] = vres.get("new_unit", mdata.attrs.get('units', 'none'))

                # Do the same for the baseline case if need be:
                if not adfobj.compare_obs:
                    odata = odata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                    # update units
                    odata.attrs['units'] = vres.get("new_unit", odata.attrs.get('units', 'none'))
                # Or for observations:
                else:
                    odata = odata * vres.get("obs_scale_factor",1) + vres.get("obs_add_offset", 0)
                   # Note: we are going to assume that the specification ensures the conversion makes the units the same. Doesn't make sense to add a different unit.

                #Determine dimensions of variable:
                has_dims = pf.lat_lon_validate_dims(odata)
                if has_dims:
                    #If observations/baseline CAM have the correct
                    #dimensions, does the input CAM run have correct
                    #dimensions as well?
                    has_dims_cam = pf.lat_lon_validate_dims(mdata)

                    #If both fields have the required dimensions, then
                    #proceed with plotting:
                    if has_dims_cam:

                        #
                        # Seasonal Averages
                        # Note: xarray can do seasonal averaging,
                        # but depends on having time accessor,
                        # which these prototype climo files do not have.
                        #

                        #Create new dictionaries:
                        mseasons = {}
                        oseasons = {}
                        dseasons = {} # hold the differences

                        #Loop over season dictionary:
                        for s in seasons:

                            if weight_season:
                                #Add date-stamp to time dimension:
                                #Note: For now using made-up dates, but in the future
                                #it might be good to extract this info from the files
                                #themselves.
                                timefix = pd.date_range(start='1/1/1980', end='12/1/1980', freq='MS')
                                mdata['time']=timefix
                                odata['time']=timefix

                                #Calculate monthly weights based on number of days:
                                month_length = mdata.time.dt.days_in_month
                                weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())

                                #Calculate monthly-weighted seasonal averages:
                                if s == 'ANN':

                                    #Calculate annual weights (i.e. don't group by season):
                                    weights_ann = month_length / month_length.sum()

                                    mseasons[s] = (mdata * weights_ann).sum(dim='time')
                                    oseasons[s] = (odata * weights_ann).sum(dim='time')
                                    # difference: each entry should be (lat, lon)
                                    dseasons[s] = mseasons[s] - oseasons[s]
                                else:
                                    #this is inefficient because we do same calc over and over
                                    mseasons[s] =(mdata * weights).groupby("time.season").sum(dim="time").sel(season=s)
                                    oseasons[s] =(odata * weights).groupby("time.season").sum(dim="time").sel(season=s)
                                    # difference: each entry should be (lat, lon)
                                    dseasons[s] = mseasons[s] - oseasons[s]
                                #End if

                            else:
                                #Just average months as-is:
                                mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                                oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
                                # difference: each entry should be (lat, lon)
                                dseasons[s] = mseasons[s] - oseasons[s]
                            #End if

                            # time to make plot; here we'd probably loop over whatever plots we want for this variable
                            # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                            plot_name = plot_loc / "{}_{}_LatLon_Mean.{}".format(var, s, plot_type)

                            #Remove old plot, if it already exists:
                            if plot_name.is_file():
                                plot_name.unlink()

                            #Create new plot:
                            # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                            # This relies on `plot_map_and_save` knowing how to deal with the options
                            # currently knows how to handle:
                            #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                            #   *Any other entries will be ignored.
                            # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.

                            pf.plot_map_and_save(plot_name, mseasons[s], oseasons[s], dseasons[s], **vres)

                    else: #mdata dimensions check
                        print(f"\t - skipping lat/lon map for {var} as it doesn't have only lat/lon dims.")
                    #End if (dimensions check)

                elif pres_levs: #Is the user wanting to interpolate to a specific pressure level?

                    #For now, there is no easy way to use observations with specified pressure levels, so
                    #bail out here:
                    if adfobj.compare_obs:
                        print(f"\t - plot_press_levels currently doesn't work with observations, so skipping variable '{var}'.")

                    else:

                        #Check that case inputs have the correct dimensions (including "lev"):
                        _, has_lev = pf.zm_validate_dims(mdata)

                        if has_lev:

                            #Next check if the data set contains CAM hybrid level info:
                            if 'hyam' in mclim_ds:
                                mhya = mclim_ds['hyam']
                                mhyb = mclim_ds['hybm']
                                if 'time' in mhya.dims:
                                    mhya = mhya.isel(time=0).squeeze()
                                if 'time' in mhyb.dims:
                                    mhyb = mhyb.isel(time=0).squeeze()
                                if 'P0' in mclim_ds:
                                    P0 = mclim_ds['P0']
                                else:
                                    P0 = 100000.0  # Pa
                                #End if
                                if 'PS' in mclim_ds:
                                    mps = mclim_ds['PS']
                                else:
                                    # look for the file (this isn't great b/c we'd have to constantly re-load)
                                    mps_files = sorted(mclimo_rg_loc.glob(f"{data_src}_{case_name}_PS_*.nc"))
                                    if len(mps_files) > 0:
                                        mps_ds = _load_dataset(mps_files)
                                        mps = mps_ds['PS']
                                    else:
                                        continue  # what else could we do?
                                    #End if
                                #End if

                                # We need a way to check whether 'obs' or 'baseline' here:
                                # (and/or a way to specify pressure levels or hybrid levels)
                                ohya = oclim_ds['hyam']
                                ohyb = oclim_ds['hybm']
                                if 'time' in ohya.dims:
                                    ohya = ohya.isel(time=0).squeeze()
                                if 'time' in ohyb.dims:
                                    ohyb = ohyb.isel(time=0).squeeze()
                                if 'PS' in oclim_ds:
                                    ops = oclim_ds['PS']
                                else:
                                    # look for the file (this isn't great b/c we'd have to constantly re-load)
                                    ops_files = sorted(dclimo_loc.glob(f"{data_src}_PS_*.nc"))
                                    if len(ops_files) > 0:
                                        ops_ds = _load_dataset(ops_files)
                                        ops = ops_ds['PS']
                                    else:
                                        continue  # what else could we do?
                                    #End if
                                #End if

                                #Interpolate variable to provided pressure levels:
                                mdata_i = pf.lev_to_plev(mdata, mps, mhya, mhyb, P0=P0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'), \
                                                         convert_to_mb=True)

                                odata_i = pf.lev_to_plev(odata, ops, ohya, ohyb, P0=P0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'), \
                                                         convert_to_mb=True)

                                #Calculate monthly weights (if applicable):
                                if weight_season:
                                    #Add date-stamp to time dimension:
                                    #Note: For now using made-up dates, but in the future
                                    #it might be good to extract this info from the files
                                    #themselves.
                                    timefix = pd.date_range(start='1/1/1980', end='12/1/1980', freq='MS')
                                    mdata_i['time']=timefix
                                    odata_i['time']=timefix

                                    #Calculate monthly weights based on number of days:
                                    month_length = mdata_i.time.dt.days_in_month
                                    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
                                #End if

                                #Loop over pressure levels:
                                for pres in pres_levs:

                                    #Create new dictionaries:
                                    mseasons = {}
                                    oseasons = {}
                                    dseasons = {}

                                    #Loop over seasons:
                                    for s in seasons:

                                        #If requested, then calculate the monthly-weighted seasona averages:
                                        if weight_season:
                                            if s == 'ANN':
                                                mseasons[s] = (mdata_i * weights).sum(dim='time').sel(lev=pres)
                                                oseasons[s] = (odata_i * weights).sum(dim='time').sel(lev=pres)
                                                # difference: each entry should be (lat, lon)
                                                dseasons[s] = mseasons[s] - oseasons[s]
                                            else:
                                                #this is inefficient because we do same calc over and over
                                                mseasons[s] =(mdata_i * weights).groupby("time.season").sum(dim="time").sel(season=s,lev=pres)
                                                oseasons[s] =(odata_i * weights).groupby("time.season").sum(dim="time").sel(season=s,lev=pres)
                                                # difference: each entry should be (lat, lon)
                                                dseasons[s] = mseasons[s] - oseasons[s]
                                            #End if
                                        else:
                                            #Just average months as-is:
                                            mseasons[s] = mdata.sel(time=seasons[s], lev=pres).mean(dim='time')
                                            oseasons[s] = odata.sel(time=seasons[s], lev=pres).mean(dim='time')
                                            # difference: each entry should be (lat, lon)
                                            dseasons[s] = mseasons[s] - oseasons[s]
                                        #End if

                                        # time to make plot; here we'd probably loop over whatever plots we want for this variable
                                        # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                                        plot_name = plot_loc / f"{var}_{s}_Lev_{pres}hpa_LatLon_Mean.{plot_type}"

                                        #Remove old plot, if it already exists:
                                        if plot_name.is_file():
                                            plot_name.unlink()

                                        #Create new plot:
                                        # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                        # This relies on `plot_map_and_save` knowing how to deal with the options
                                        # currently knows how to handle:
                                        #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                        #   *Any other entries will be ignored.
                                        # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                                        pf.plot_map_and_save(plot_name, mseasons[s], oseasons[s], dseasons[s], **vres)

                                    #End for (seasons)
                                #End for (pressure levels)

                            else:
                                print(f"\t - plot_pres_levels currently only works with hybrid coordinates, so skipping '{var}'.")
                            #End if (hyam)

                        else:
                            print(f"\t - variable '{var}' has no vertical dimension but is not just time/lat/lon, so skipping.")
                        #End if (has_lev)

                else: #odata dimensions check
                     print(f"\t - skipping lat/lon map for {var} as it doesn't have only lat/lon dims.")

                #End if (dimensions check)
            #End for (case loop)
        #End for (obs/baseline loop)
    #End for (variable loop)

    #Notify user that script has ended:
    print("  ...lat/lon maps have been generated successfully.")

#
# Helpers
#
def _load_dataset(fils):
    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)

##############
#END OF SCRIPT
