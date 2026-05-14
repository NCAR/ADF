from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import plotting_utils as plot_utils

import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

def meridional_mean(adfobj):

    """
    This script plots meridional averages.
    Follows the old AMWG convention of plotting 5S to 5N.
    **Note:** the constraint of 5S to 5N is easily changed;
    the function that calculates the average can take any range of latitudes.
    Compare CAM climatologies against
    other climatological data (observations or baseline runs).
    Compare CAM climatologies against
    other climatological data (observations or baseline runs).

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    None
        Does not return value, produces files.

    Notes
    -----
    Uses AdfData for loading data described by adfobj. 

    Directly uses adfobj for the following:
    diag_var_list, climo_yrs, variable_defaults, read_config_var, 
    get_basic_info, add_website_data, debug_log

    Determines whether `lev` dimension is present. If not, makes
    a line plot, but if so it makes a contour plot.
    TODO: There's a flag to plot linear vs log pressure, but no
          method to infer what the user wants. 
    """

    #Notify user that script has started:
    msg = "\n  Generating meridional mean plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    var_list = adfobj.diag_var_list

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

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


    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Check if plots already exist and redo_plot boolean
    #If redo_plot is false and file exists, keep track and attempt to skip calcs to
    #speed up preformance a bit if re-running the ADF
    meridional_skip = []
    logp_meridional_skip = []

    #Loop over model cases:
    for case_idx, case_name in enumerate(adfobj.data.case_names):
        #Set output plot location:
        plot_loc = Path(plot_locations[case_idx])

        #Check if plot output directory exists, and if not, then create it:
        if not plot_loc.is_dir():
            print(f"    {plot_loc} not found, making new directory")
            plot_loc.mkdir(parents=True)
        #End if

        #Loop over the variables for each season
        for var in var_list:
            for s in seasons:
                #Check meridional log-p:
                plot_name_log = plot_loc / f"{var}_{s}_Meridional_logp_Mean.{plot_type}"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name_log.is_file():
                    logp_meridional_skip.append(plot_name_log)
                    #Continue to next iteration:
                    adfobj.add_website_data(plot_name_log, f"{var}_logp", case_name, season=s,
                                            plot_type="Meridional", category=web_category)
                    pass

                elif (redo_plot) and plot_name_log.is_file():
                    plot_name_log.unlink()
                #End if
                
                #Check regular meridional
                plot_name = plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}"
                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    meridional_skip.append(plot_name)
                    #Add already-existing plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s,
                                                        plot_type="Meridional")

                    continue
                elif (redo_plot) and plot_name.is_file():
                    plot_name.unlink()
                #End if
            #End for (seasons)
        #End for (variables)
    #End for (cases)
    #
    # End redo plots check
    #

    #
    # Setup Plotting
    #
    #Loop over variables:
    for var in var_list:
        #Notify user of variable being plotted:
        print(f"\t - meridional mean plots for {var}")

        if var not in adfobj.data.ref_var_nam:
            dmsg = f"\t    WARNING: No reference data found for variable `{var}`, meridional mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            continue

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"\t    INFO: meridional_mean: Found variable defaults for {var}")

        else:
            vres = {}
        #End if

        vres = plot_utils.add_var_to_vres(adfobj, var, vres)
        vres["plot_type"] = __name__
        #Extract category (if available):
        web_category = vres.get("category", None)

        # load reference data (observational or baseline)
        if not adfobj.compare_obs:
            base_name = adfobj.data.ref_case_label
        else:
            base_name = adfobj.data.ref_labels[var]

        # Gather reference variable data
        odata = adfobj.data.load_reference_regrid_da(base_name, var)

        #Check if regridded file exists, if not skip zonal plot for this var
        if odata is None:
            dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{var}`, zonal mean plotting skipped."
            adfobj.debug_log(dmsg)
            continue

        #Check meridional mean dimensions
        lat_lev_ref = utils.validate_dims(odata, ['lat', 'lev'])
        #Check if reference file has vertical levels
        #Notify user of level dimension:
        if lat_lev_ref['has_lev']:
            print(f"\t    INFO: {var} has lev dimension.")
            has_lev_ref = True
        else:
            has_lev_ref = False

        #Loop over model cases:
        for case_idx, case_name in enumerate(adfobj.data.case_names):

            #Set case nickname:
            case_nickname = adfobj.data.test_nicknames[case_idx]

            #Set output plot location:
            plot_loc = Path(plot_locations[case_idx])

            # load re-gridded model files:
            mdata = adfobj.data.load_regrid_da(case_name, var)

            if mdata is None:
                dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{var}`, zonal mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue

            # determine whether it's 2D or 3D
            # 3D triggers search for surface pressure
            # check data dimensions:
            #has_lat, has_lev = utils.zm_validate_dims(mdata)
            lat_lev = utils.validate_dims(mdata, ['lat', 'lev'])

            #Check if reference file has vertical levels
            #Notify user of level dimension:
            if lat_lev['has_lev']:
                print(f"\t    INFO: {var} has lev dimension.")
                has_lev = True
            else:
                has_lev = False

            #Check to make sure each case has vertical levels if one of the cases does
            if (has_lev) and (not has_lev_ref):
                print(f"\t    WARNING: expecting lev boolean for both case: {has_lev} and ref: {has_lev_ref}")
                continue
            if (has_lev_ref) and (not has_lev):
                print(f"\t    WARNING: expecting lev boolean for both case: {has_lev} and ref: {has_lev_ref}")
                continue

            #
            # Seasonal Averages
            #

            #Create new dictionaries:
            mseasons = {}
            oseasons = {}

            #Loop over season dictionary:
            for s in seasons:
                vres["season"] = s

                # time to make plot; here we'd probably loop over whatever plots we want for this variable
                # I'll just call this one "Zonal_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                # NOTE: Up to this point, nothing really differs from global_latlon_map,
                #       so we could have made one script instead of two.
                #       Merging would make overall timing better because looping twice will double I/O steps.
                #

                # difference: each entry should be (lat, lon) or (plev, lat, lon)
                # dseasons[s] = mseasons[s]    oseasons[s]
                # difference will be calculated in plot_zonal_mean_and_save;
                # because we can let any pressure-level interpolation happen there
                # This could be re-visited for efficiency or improved code structure.

                #Seasonal Averages
                mseasons[s] = utils.seasonal_mean(mdata, season=s, is_climo=True)
                oseasons[s] = utils.seasonal_mean(odata, season=s, is_climo=True)

                #Set the file name
                plot_name = plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}"
                plot_name_log = None

                if has_lev:
                    #Set the file name for log-pressure plots
                    plot_name_log = plot_loc / f"{var}_logp_{s}_Meridional_Mean.{plot_type}"
                #End if

                #Create plots
                if plot_name not in meridional_skip:

                    #Create new plot:
                    pf.plot_meridional_mean_and_save(adfobj, plot_name, case_nickname, adfobj.data.ref_nickname,
                                                    [syear_cases[case_idx],eyear_cases[case_idx]],
                                                    [syear_baseline,eyear_baseline],
                                                    mseasons[s], oseasons[s], has_lev, log_p=False, obs=adfobj.compare_obs, **vres)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s, plot_type="Meridional")
                #End if

                #Create log-pressure plots as well (if applicable)
                if (plot_name_log) and (plot_name_log not in logp_meridional_skip):

                    pf.plot_meridional_mean_and_save(adfobj, plot_name_log, case_nickname, adfobj.data.ref_nickname,
                                                        [syear_cases[case_idx],eyear_cases[case_idx]],
                                                        [syear_baseline,eyear_baseline],
                                                        mseasons[s], oseasons[s], has_lev, log_p=True, obs=adfobj.compare_obs, **vres)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name_log, f"{var}_logp", case_name, season=s, plot_type="Meridional", category=web_category)
                #End if

            #End for (seasons loop)
        #End for (case names loop)
    #End for (variables loop)

    #Notify user that script has ended:
    print("  ...Meridional mean plots have been generated successfully.")


##############
#END OF SCRIPT