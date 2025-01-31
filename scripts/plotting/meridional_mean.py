from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

def meridional_mean(adfobj):

    """
    This script plots meridional averages.
    Follows the old AMWG convention of plotting 5S to 5N.
    **Note:** the constraint of 5S to 5N is easily changed;
    the function that calculates the average can take any range of latitudes.
    Compare CAM climatologies against
    other climatological data (observations or baseline runs).
    """

    #Notify user that script has started:
    print("\n  Generating meridional mean plots...")

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    #Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):
        #Set obs call for observation details for plot titles
        obs = True

        #Extract variable-obs dictionary:
        var_obs_dict = adfobj.var_obs_dict

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("\t No observations found to plot against, so no meridional-mean maps will be generated.")
            return
    else:
        obs = False
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = model_rgrid_loc #Just use the re-gridded model data path
    #End if

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

    #Grab all case nickname(s)
    test_nicknames = adfobj.case_nicknames["test_nicknames"]
    base_nickname = adfobj.case_nicknames["base_nickname"]

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in the config YAML file.

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

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Loop over variables:
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
                dmsg = f"No obs found for variable `{var}`, meridional mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue
            #End if
        else:
            #Set "data_var" for consistent use below:
            data_var = var
        #End if

        #Notify user of variable being plotted:
        print(f"\t - meridional mean plots for {var}")

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"meridional_mean: Found variable defaults for {var}")

        else:
            vres = {}
        #End if

        #loop over different data sets to plot model against:
        for data_src in data_list:
            # load data (observational) comparison files
            # (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #For now, only grab one file (but convert to list for use below)
                oclim_fils = [dclimo_loc]
            else:
                oclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{var}_baseline.nc"))
            #End if
            oclim_ds = pf.load_dataset(oclim_fils)

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                #Set case nickname:
                case_nickname = test_nicknames[case_idx]

                #Set output plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print(f"    {plot_loc} not found, making new directory")
                    plot_loc.mkdir(parents=True)

                # load re-gridded model files:
                mclim_fils = sorted(mclimo_rg_loc.glob(f"{data_src}_{case_name}_{var}_*.nc"))
                mclim_ds = pf.load_dataset(mclim_fils)

                # stop if data is invalid:
                if (oclim_ds is None) or (mclim_ds is None):
                    warnings.warn(f"invalid data, skipping meridional mean plot of {var}")
                    continue

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
                # Or for observations
                else:
                    odata = odata * vres.get("obs_scale_factor",1) + vres.get("obs_add_offset", 0)
                    # Note: we are going to assume that the specification ensures the conversion makes the units the same.
                    #       Doesn't make sense to add a different unit.

                # determine whether it's 2D or 3D
                # 3D triggers search for surface pressure
                validate_lat_lev = pf.validate_dims(mdata, ['lat', 'lev']) # keys=> 'has_lat', 'has_lev', with T/F values

                #Notify user of level dimension:
                if validate_lat_lev['has_lev']:
                    print(f"\t    INFO: {var} has lev dimension.")
                    has_lev = True
                else:
                    has_lev = False

                #
                # Seasonal Averages
                # Note: xarray can do seasonal averaging, but depends on having time accessor,
                # which these prototype climo files don't.
                #

                #Create new dictionaries:
                mseasons = {}
                oseasons = {}

                #Loop over season dictionary:
                for s in seasons:
                    plot_name = plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}"

                    # Check redo_plot. If set to True: remove old plot, if it already exists:
                    if (not redo_plot) and plot_name.is_file():
                        #Add already-existing plot to website (if enabled):
                        adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                        adfobj.add_website_data(plot_name, var, case_name, season=s,
                                                plot_type="Meridional")
                        #Continue to next iteration:
                        continue
                    elif (redo_plot) and plot_name.is_file():
                        plot_name.unlink()

                    mseasons[s] = pf.seasonal_mean(mdata, season=s, is_climo=True)
                    oseasons[s] = pf.seasonal_mean(odata, season=s, is_climo=True)


                    #Create new plot:
                    pf.plot_meridional_mean_and_save(plot_name, case_nickname, base_nickname,
                                                [syear_cases[case_idx],eyear_cases[case_idx]],
                                                [syear_baseline,eyear_baseline],
                                                mseasons[s], oseasons[s], has_lev, latbounds=slice(-5,5), obs=obs, **vres)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s,
                                            plot_type="Meridional")

                #End for (seasons loop)
            #End for (case names loop)
        #End for (obs/baseline loop)
    #End for (variables loop)

    #Notify user that script has ended:
    print("  ...Meridional mean plots have been generated successfully.")


#########
# Helpers
#########


##############
#END OF SCRIPT