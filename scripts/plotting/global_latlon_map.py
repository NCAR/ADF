#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.

#Format warning messages:
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
    data_loc         -> Location of comparison data, which is either the location listed
                        in each variable's ""obs_file", or the same as "model_rgrid_loc",
                        depending on whether "compare_obs" is true or false.
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

    #Import necessary modules:
    #------------------------
    import pandas as pd

    #CAM diagnostic plotting functions:
    import plotting_functions as pf
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("\n  Generating lat/lon maps...")

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

    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    #Grab test case nickname(s)
    test_nicknames = adfobj.get_cam_info('case_nickname')
    if test_nicknames == None:
        test_nicknames = case_names

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):

        #Extract variable-obs dictionary:
        var_obs_dict = adfobj.var_obs_dict
        base_nickname = "Obs"

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("No observations found to plot against, so no lat/lon maps will be generated.")
            return

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = model_rgrid_loc #Just use the re-gridded model data path

        syear_baseline = adfobj.climo_yrs["syear_baseline"]
        eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

        #Grab baseline case nickname
        base_nickname = adfobj.get_baseline_info('case_nickname')
        if base_nickname == None:
            base_nickname = data_name

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
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

            #Extract category (if available):
            web_category = vres.get("category", None)

        else:
            vres = {}
            web_category = None
        #End if

        # For global maps, also set the central longitude:
        # can be specified in adfobj basic info as 'central_longitude' or supplied as a number,
        # otherwise defaults to 180
        vres['central_longitude'] = pf.get_central_longitude(adfobj)

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #For now, only grab one file (but convert to list for use below)
                oclim_fils = [dclimo_loc]
            else:
                oclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{var}_baseline.nc"))

            oclim_ds = _load_dataset(oclim_fils)
            if oclim_ds is None:
                print("WARNING: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{var}_*.nc")
                continue

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                #Set case nickname:
                case_nickname = test_nicknames[case_idx]

                #Set output plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print("    {} not found, making new directory".format(plot_loc))
                    plot_loc.mkdir(parents=True)

                # load re-gridded model files:
                mclim_fils = sorted(mclimo_rg_loc.glob(f"{data_src}_{case_name}_{var}_*.nc"))
                mclim_ds = _load_dataset(mclim_fils)

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
                            plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"

                            # Check redo_plot. If set to True: remove old plot, if it already exists:
                            if (not redo_plot) and plot_name.is_file():
                                #Add already-existing plot to website (if enabled):
                                adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                                        season=s, plot_type="LatLon")

                                #Continue to next iteration:
                                continue
                            elif (redo_plot) and plot_name.is_file():
                                plot_name.unlink()

                            #Create new plot:
                            # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                            # This relies on `plot_map_and_save` knowing how to deal with the options
                            # currently knows how to handle:
                            #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                            #   *Any other entries will be ignored.
                            # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.

                            pf.plot_map_and_save(plot_name, case_nickname, base_nickname,
                                                 [syear_cases[case_idx],eyear_cases[case_idx]],
                                                 [syear_baseline,eyear_baseline],
                                                 mseasons[s], oseasons[s], dseasons[s],
                                                 **vres)

                            #Add plot to website (if enabled):
                            adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                                    season=s, plot_type="LatLon")

                    else: #mdata dimensions check
                        print(f"\t - skipping lat/lon map for {var} as it doesn't have only lat/lon dims.")
                    #End if (dimensions check)

                elif pres_levs: #Is the user wanting to interpolate to a specific pressure level?

                    #Check that case inputs have the correct dimensions (including "lev"):
                    _, has_lev = pf.zm_validate_dims(mdata)

                    if has_lev:

                        #Calculate monthly weights (if applicable):
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
                        #End if

                        #Loop over pressure levels:
                        for pres in pres_levs:

                            #Check that the user-requested pressure level
                            #exists in the model data, which should already
                            #have been interpolated to the standard reference
                            #pressure levels:
                            if not (pres in mclim_ds['lev']):
                                #Move on to the next pressure level:
                                print(f"plot_press_levels value '{pres}' not a standard reference pressure, so skipping.")
                                continue
                            #End if

                            #Create new dictionaries:
                            mseasons = {}
                            oseasons = {}
                            dseasons = {}

                            #Loop over seasons:
                            for s in seasons:

                                #If requested, then calculate the monthly-weighted seasonal averages:
                                if weight_season:
                                    if s == 'ANN':
                                        #Calculate annual weights (i.e. don't group by season):
                                        weights_ann = month_length / month_length.sum()

                                        mseasons[s] = (mdata * weights_ann).sum(dim='time').sel(lev=pres)
                                        oseasons[s] = (odata * weights_ann).sum(dim='time').sel(lev=pres)
                                        # difference: each entry should be (lat, lon)
                                        dseasons[s] = mseasons[s] - oseasons[s]
                                    else:
                                        #this is inefficient because we do same calc over and over
                                        mseasons[s] =(mdata * weights).groupby("time.season").sum(dim="time").sel(season=s,lev=pres)
                                        oseasons[s] =(odata * weights).groupby("time.season").sum(dim="time").sel(season=s,lev=pres)
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
                                plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"

                                # Check redo_plot. If set to True: remove old plot, if it already exists:
                                redo_plot = adfobj.get_basic_info('redo_plot')
                                if (not redo_plot) and plot_name.is_file():
                                    #Add already-existing plot to website (if enabled):
                                    adfobj.add_website_data(plot_name, f"{var}_{pres}hpa", case_name, category=web_category,
                                                            season=s, plot_type="LatLon")

                                    #Continue to next iteration:
                                    continue
                                elif (redo_plot) and plot_name.is_file():
                                    plot_name.unlink()

                                #Create new plot:
                                # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                # This relies on `plot_map_and_save` knowing how to deal with the options
                                # currently knows how to handle:
                                #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                #   *Any other entries will be ignored.
                                # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                                pf.plot_map_and_save(plot_name, case_nickname, base_nickname,
                                                     [syear_cases[case_idx],eyear_cases[case_idx]],
                                                     [syear_baseline,eyear_baseline],
                                                     mseasons[s], oseasons[s], dseasons[s],
                                                     **vres)

                                #Add plot to website (if enabled):
                                adfobj.add_website_data(plot_name, f"{var}_{pres}hpa", case_name, category=web_category,
                                                        season=s, plot_type="LatLon")

                            #End for (seasons)
                        #End for (pressure levels)

                    else:
                        print(f"\t - variable '{var}' has no vertical dimension but is not just time/lat/lon, so skipping.")
                    #End if (has_lev)
                else:
                    print(f"\t - skipping polar map for {var} as it has more than lat/lon dims, but no pressure levels were provided")
                #End if (dimensions check and plotting pressure levels)
            #End for (case loop)
        #End for (obs/baseline loop)
    #End for (variable loop)

    #Notify user that script has ended:
    print("  ...lat/lon maps have been generated successfully.")

#########
# Helpers
#########

def _load_dataset(fils):
    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)
    #End if
#End def

##############
#END OF SCRIPT
