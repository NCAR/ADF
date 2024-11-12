"""
Generate global maps of 2-D fields

Functions
---------
global_latlon_map(adfobj)
    use ADF object to make maps
my_formatwarning(msg, *args, **kwargs)
    format warning messages
    (private method)
plot_file_op
    Check on status of output plot file.
"""
#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.

import plotting_functions as pf

#Format warning messages:
def my_formatwarning(msg, *args, **kwargs):
    """Issue `msg` as warning."""
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

def global_latlon_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model fields with continental overlays.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    Does not return a value; produces plots and saves files.

    Notes
    -----

    It uses the AdfDiag object's methods to get necessary information.
    Makes use of AdfDiag's data sub-class.
    Explicitly accesses:
    adfobj.diag_var_list
        List of variables
    adfobj.plot_location
        output plot path
    adfobj.climo_yrs
        start and end climo years of the case(s), `syears` & `eyears`
        start and end climo years of the reference, `syear_baseline` & `eyear_baseline`
    adfobj.variable_defaults 
        dict of variable-specific plot preferences
    adfobj.read_config_var
        dict of basic info, `diag_basic_info`
        Then use to check `plot_type`
    adfobj.debug_log
        Issues debug message
    adfobj.add_website_data
        Communicates information to the website generator
    adfobj.compare_obs
        Logical to determine if comparing to observations

        
    The `plotting_functions` module is needed for:
    pf.get_central_longitude()
        determine central longitude for global plots
    pf.lat_lon_validate_dims()
        makes sure latitude and longitude are valid
    pf.seasonal_mean()
        calculate seasonal mean
    pf.plot_map_and_save()
        send information to make the plot and save the file
    pf.zm_validate_dims()
        Checks on pressure level dimension
    """

    #Notify user that script has started:
    print("\n  Generating lat/lon maps...")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
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

    #Determine if user wants to plot 3-D variables on
    #pressure levels:
    pres_levs = adfobj.get_basic_info("plot_press_levels")

    weight_season = True  #always do seasonal weighting

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    # probably want to do this one variable at a time:
    for var in var_list:
        if var not in adfobj.data.ref_var_nam:
            dmsg = f"No reference data found for variable `{var}`, global lat/lon mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            continue        

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

        # load reference data (observational or baseline)
        if not adfobj.compare_obs:
            base_name = adfobj.data.ref_case_label
        else:
            base_name = adfobj.data.ref_labels[var]

        # Gather reference variable data
        odata = adfobj.data.load_reference_regrid_da(base_name, var)

        if odata is None:
            dmsg = f"No regridded test file for {base_name} for variable `{var}`, global lat/lon mean plotting skipped."
            adfobj.debug_log(dmsg)
            continue

        o_has_dims = pf.validate_dims(odata, ["lat", "lon", "lev"]) # T iff dims are (lat,lon) -- can't plot unless we have both
        if (not o_has_dims['has_lat']) or (not o_has_dims['has_lon']):
            print(f"\t = skipping global map for {var} as REFERENCE does not have both lat and lon")
            continue

        #Loop over model cases:
        for case_idx, case_name in enumerate(adfobj.data.case_names):

            #Set case nickname:
            case_nickname = adfobj.data.test_nicknames[case_idx]

            #Set output plot location:
            plot_loc = Path(plot_locations[case_idx])

            #Check if plot output directory exists, and if not, then create it:
            if not plot_loc.is_dir():
                print("    {} not found, making new directory".format(plot_loc))
                plot_loc.mkdir(parents=True)

            #Load re-gridded model files:
            mdata = adfobj.data.load_regrid_da(case_name, var)

            #Skip this variable/case if the regridded climo file doesn't exist:
            if mdata is None:
                dmsg = f"No regridded test file for {case_name} for variable `{var}`, global lat/lon mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue

            #Determine dimensions of variable:
            has_dims = pf.validate_dims(mdata, ["lat", "lon", "lev"])
            if (not has_dims['has_lat']) or (not has_dims['has_lon']):
                print(f"\t = skipping global map for {var} for case {case_name} as it does not have both lat and lon")
                continue
            else: # i.e., has lat&lon
                if (has_dims['has_lev']) and (not pres_levs):
                    print(f"\t - skipping global map for {var} as it has more than lev dimension, but no pressure levels were provided")
                    continue

            # Check output file. If file does not exist, proceed.
            # If file exists:
            #   if redo_plot is true: delete it now and make plot
            #   if redo_plot is false: add to website and move on
            doplot = {}

            if not has_dims['has_lev']:
                for s in seasons:
                    plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"
                    doplot[plot_name] = plot_file_op(adfobj, plot_name, var, case_name, s, web_category, redo_plot, "LatLon")
            else:
                for pres in pres_levs:
                    for s in seasons:
                        plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"
                        doplot[plot_name] = plot_file_op(adfobj, plot_name, f"{var}_{pres}hpa", case_name, s, web_category, redo_plot, "LatLon")
            if all(value is None for value in doplot.values()):
                print(f"All plots exist for {var}. Redo is {redo_plot}. Existing plots added to website data. Continue.")
                continue

            #Create new dictionaries:
            mseasons = {}
            oseasons = {}
            dseasons = {} # hold the differences
            pseasons = {} # hold percent change

            if not has_dims['has_lev']:  # strictly 2-d data          

                #Loop over season dictionary:
                for s in seasons:
                    plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"
                    if doplot[plot_name] is None:
                        continue

                    if weight_season:
                        mseasons[s] = pf.seasonal_mean(mdata, season=s, is_climo=True)
                        oseasons[s] = pf.seasonal_mean(odata, season=s, is_climo=True)
                    else:
                        #Just average months as-is:
                        mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                        oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
                    #End if

                    # difference: each entry should be (lat, lon)
                    dseasons[s] = mseasons[s] - oseasons[s]
                    
                    # percent change
                    pseasons[s] = (mseasons[s] - oseasons[s]) / np.abs(oseasons[s]) * 100.0 #relative change

                    pf.plot_map_and_save(plot_name, case_nickname, adfobj.data.ref_nickname,
                                            [syear_cases[case_idx],eyear_cases[case_idx]],
                                            [syear_baseline,eyear_baseline],
                                            mseasons[s], oseasons[s], dseasons[s], pseasons[s],
                                            obs=adfobj.compare_obs, **vres)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                            season=s, plot_type="LatLon")

            else: # => pres_levs has values, & we already checked that lev is in mdata (has_lev)

                for pres in pres_levs:

                    #Check that the user-requested pressure level
                    #exists in the model data, which should already
                    #have been interpolated to the standard reference
                    #pressure levels:
                    if (not (pres in mdata['lev'])) or (not (pres in odata['lev'])):
                        print(f"plot_press_levels value '{pres}' not present in {var} [test: {(pres in mdata['lev'])}, ref: {pres in odata['lev']}], so skipping.")
                        continue

                    #Loop over seasons:
                    for s in seasons:
                        plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"
                        if doplot[plot_name] is None:
                            continue

                        if weight_season:
                            mseasons[s] = pf.seasonal_mean(mdata, season=s, is_climo=True)
                            oseasons[s] = pf.seasonal_mean(odata, season=s, is_climo=True)
                        else:
                            #Just average months as-is:
                            mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                            oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
                        #End if

                        # difference: each entry should be (lat, lon)
                        dseasons[s] = mseasons[s] - oseasons[s]
                        
                        # percent change
                        pseasons[s] = (mseasons[s] - oseasons[s]) / np.abs(oseasons[s]) * 100.0 #relative change

                        pf.plot_map_and_save(plot_name, case_nickname, adfobj.data.ref_nickname,
                                                [syear_cases[case_idx],eyear_cases[case_idx]],
                                                [syear_baseline,eyear_baseline],
                                                mseasons[s].sel(lev=pres), oseasons[s].sel(lev=pres), dseasons[s].sel(lev=pres),
                                                pseasons[s].sel(lev=pres),
                                                obs=adfobj.compare_obs, **vres)

                        #Add plot to website (if enabled):
                        adfobj.add_website_data(plot_name, f"{var}_{pres}hpa", case_name, category=web_category,
                                                season=s, plot_type="LatLon")
                    #End for (seasons)
                #End for (pressure levels)
            #End if (plotting pressure levels)
        #End for (case loop)
    #End for (variable loop)

    #Notify user that script has ended:
    print("  ...lat/lon maps have been generated successfully.")


def plot_file_op(adfobj, plot_name, var, case_name, season, web_category, redo_plot, plot_type):
    """Check if output plot needs to be made or remade.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    plot_name : Path
        path of the output plot

    var : str
        name of variable

    case_name : str
        case name
    
    season : str
        season being plotted

    web_category : str
        the category for this variable

    redo_plot : bool
        whether to overwrite existing plot with this file name

    plot_type : str
        the file type for the output plot

    Returns
    -------
    int, None
        Returns 1 if existing file is removed or no existing file.
        Returns None if file exists and redo_plot is False

    Notes
    -----
    The long list of parameters is because add_website_data is called
    when the file exists and will not be overwritten.
    
    """
    # Check redo_plot. If set to True: remove old plot, if it already exists:
    if plot_name.is_file():
        if redo_plot:
            plot_name.unlink()
            return True
        else:
            #Add already-existing plot to website (if enabled):
            adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                    season=season, plot_type=plot_type)
            return False  # False tells caller that file exists and not to overwrite
    else:
        return True

##############
#END OF SCRIPT
