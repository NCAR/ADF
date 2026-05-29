#Import necessary modules:
#------------------------
from pathlib import Path  # python standard library

def global_latlon_vect_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model vector fields with continental
    overlays.
    Description of needed inputs from ADF:
    case_name         -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc   -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name         -> Name of data set CAM case is being compared against,
                         which is always either "obs" or the baseline CAM case name,
                         depending on whether "compare_obs" is true or false.
    data_loc          -> Location of comparison data, which is either "obs_data_loc"
                         or "cam_baseline_climo_loc", depending on whether
                         "compare_obs" is true or false.
    var_list          -> List of CAM output variables provided by "diag_var_list"
    data_list         -> List of data sets CAM will be compared against, which
                         is simply the baseline case name in situations when
                         "compare_obs" is false.
    plot_location     -> Location where plot files will be written to, which is
                         specified by "cam_diag_plot_loc".
    climo_yrs         -> Dictionary containing the start and end years of the test
                         and baseline model data (if applicable).
    variable_defaults -> optional,
                        Dict that has keys that are variable names and values that are plotting preferences/defaults.
    """

    #Import necessary modules:
    #------------------------

    # data loading / analysis
    import xarray as xr
    import numpy as np

    #ADF utility functions:
    import adf_utils as utils
    import plotting_utils as plot_utils
    import plotting_functions as pf

    # Warnings
    import warnings  # use to warn user about missing files.
    warnings.formatwarning = utils.my_formatwarning
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal vector fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    msg = "\n  Generating lat/lon vector maps..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output path for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    kwargs = {}
    # Check for unstructured simulations
    unstruct_plotting = adfobj.unstructured_plotting
    if unstruct_plotting:
        kwargs["unstructured_plotting"] = unstruct_plotting
    else:
        unstructured=False

    #CAM simulation variables:
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
            print("\t No observations found to plot against, so no vector maps will be generated.")
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
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    adfobj.debug_log(f"Vector plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    #print(f"\t NOTE: redo_plot is set to {redo_plot}")

    comp = adfobj.model_component
    unstructured = False

    #-----------------------------------------

    #Set input/output data path variables:
    #------------------------------------
    mclimo_rg_loc = Path(model_rgrid_loc)
    if not adfobj.compare_obs:
        dclimo_loc = Path(data_loc)
    #End if
    #-----------------------------------
    #Determine if user wants to plot 3-D variables on
    #pressure levels:
    pres_levs = adfobj.get_basic_info("plot_press_levels")

    #Set seasonal ranges:

    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    #Initialize skipped variables set:
    skip_vars = set()

    # probably want to do this one variable at a time:
    for var in var_list:

        #Don't process variable if already used in vector:
        if var in skip_vars:
            continue
        #End if

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            vres["unstructured_plotting"] = unstruct_plotting
            vres = plot_utils.add_var_to_vres(adfobj, var, vres)
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_latlon_vect_map: Found variable defaults for {var}")

            #Extract category (if available):
            web_category = vres.get("category", None)

        else:
            vres = {}
            web_category = None
        #End if

        vres["plot_type"] = __name__

        #Make sure that variable is part of a vector pair:
        if "vector_pair" in vres:
            var_pair = vres["vector_pair"]
            var_name = vres["vector_name"]
        else:
            adfobj.debug_log(f"variable '{var}' not a vector pair")
            continue
        #End if

        #Notify user of variable being plotted:
        print(f"\t - lat/lon vector maps for {var},{var_pair}")

        #Add variables to "skipped vars" set:
        skip_vars.add(var)
        skip_vars.add(var_pair)

        # For global maps, also set the central longitude:
        # can be specified in adfobj basic info as 'central_longitude' or supplied as a number,
        # otherwise defaults to 180
        vres['central_longitude'] = plot_utils.get_central_longitude(adfobj)

        #Determine observations to compare against:
        if adfobj.compare_obs:
            if var not in adfobj.data.ref_var_nam:
                dmsg = f"\t    WARNING: No reference data found for variable `{var}`, lat/lon vector map skipped."
                adfobj.debug_log(dmsg)
                print(dmsg)
                continue
            #Check if obs exist for the variable:
            if var in var_obs_dict:
                #Note: In the future these may all be lists, but for
                #now just convert the target_list.
                #Extract target file:
                udclimo_loc = var_obs_dict[var]["obs_file"]
                #Extract target list (eventually will be a list, for now need to convert):
                data_list = [var_obs_dict[var]["obs_name"]]
                #Extract target variable name:
                data_var = [var_obs_dict[var]["obs_var"]]
            else:
                dmsg = f"\t    WARNING: No reference data found for variable `{var}`, lat/lon vector map skipped."
                adfobj.debug_log(dmsg)
                print(dmsg)
                continue
            #End if
            #Check if obs exist for vector pair variable:
            if var_pair in var_obs_dict:
                #Note: In the future these may all be lists, but for
                #now just convert the target_list.
                #Extract target file:
                vdclimo_loc = var_obs_dict[var_pair]["obs_file"]
                #Extract target variable name:
                data_var.append(var_obs_dict[var_pair]["obs_var"])
            else:
                dmsg = f"\t    WARNING: No reference data found for variable `{var}`, lat/lon vector map skipped."
                adfobj.debug_log(dmsg)
                print(dmsg)
                continue
            #End if

        else:
            #Set "data_var" for consistent use below:
            data_var = [var, var_pair]

            # reference (baseline) name
            base_name = adfobj.data.ref_case_label
        #End if

        #loop over different data sets to plot model against:
        for data_src in data_list:
            if unstruct_plotting:
                mesh_file = adfobj.mesh_files["baseline_mesh_file"]
                kwargs["mesh_file"] = mesh_file
                uodata = adfobj.data.load_reference_regrid_da(base_name, data_var[0], **kwargs)
                vodata = adfobj.data.load_reference_regrid_da(base_name, data_var[1], **kwargs)
                unstruct_base = True
                odataset = adfobj.data.load_reference_regrid_dataset(base_name, data_var[0], **kwargs)
                o_has_dims = utils.validate_dims(uodata, [ "lev"])
                if comp == "lnd": 
                    area = odataset.area.isel(time=0)
                    landfrac = odataset.landfrac.isel(time=0)
                    # calculate weights
                    wgt_base = area * landfrac / (area * landfrac).sum()
                if comp == "atm":
                    wgt_base = odataset.isel(time=0)[var]
            else:
                uodata = adfobj.data.load_reference_regrid_da(base_name, data_var[0], **kwargs)
                vodata = adfobj.data.load_reference_regrid_da(base_name, data_var[1], **kwargs)
                if (uodata is None) or (vodata is None):
                    
                    dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
                    adfobj.debug_log(dmsg)
                    continue
                o_has_dims = utils.validate_dims(uodata    , ["lat", "lon", "lev"]) # T iff dims are (lat,lon) -- can't plot unless we have both
                if (not o_has_dims['has_lat']) or (not o_has_dims['has_lon']):
                    print(f"\t    WARNING: skipping global map for {var} as REFERENCE does not have both lat and lon")
                    continue

            if uodata is None:
                dmsg = f"\t    WARNING: No baseline file for {base_name} for variable `{data_var[0]}`, global lat/lon mean plotting skipped."
                #dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{var}`, will"
                adfobj.debug_log(dmsg)
                print(dmsg)
                continue
            if vodata is None:
                dmsg = f"\t    WARNING: No baseline file for {base_name} for variable `{data_var[1]}`, global lat/lon mean plotting skipped."
                #dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{var}`, will"
                adfobj.debug_log(dmsg)
                print(dmsg)
                continue

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                #Set case nickname:
                case_nickname = test_nicknames[case_idx]

                #Set plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print(f"\t {plot_loc} not found, making new directory")
                    plot_loc.mkdir(parents=True)
                #End if

                if unstruct_plotting:
                    mesh_file = adfobj.mesh_files["test_mesh_file"][case_idx]
                    kwargs["mesh_file"] = mesh_file
                    vres["mesh_file"] = mesh_file
                    umdata = adfobj.data.load_regrid_da(case_name, data_var[0], **kwargs)
                    vmdata = adfobj.data.load_regrid_da(case_name, data_var[1], **kwargs)

                    unstruct_case = True
                    mdataset = adfobj.data.load_regrid_dataset(case_name, data_var[0], **kwargs)
                    #Determine dimensions of variable:
                    m_has_dims = utils.validate_dims(umdata, [ "lev"])
                    if comp == "lnd": 
                        area = mdataset.area.isel(time=0)
                        landfrac = mdataset.landfrac.isel(time=0)
                        # calculate weights
                        wgt = area * landfrac / (area * landfrac).sum()
                    if comp == "atm":
                        wgt = mdataset.isel(time=0)[var]
                        #print("LATLON FUNC wgt",wgt,"\n")
                else:
                    umdata = adfobj.data.load_regrid_da(case_name, data_var[0])
                    vmdata = adfobj.data.load_regrid_da(case_name, data_var[1])
                    #Skip this variable/case if the regridded climo file doesn't exist:
                    if (umdata is None) or (vmdata is None):
                        dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
                        adfobj.debug_log(dmsg)
                        continue
                    #Determine dimensions of variable:
                    m_has_dims = utils.validate_dims(umdata, ["lat", "lon", "lev"])
                    if (not m_has_dims['has_lat']) or (not m_has_dims['has_lon']):
                        print(f"\t    WARNING: skipping global map for {var} for case {case_name} as it does not have both lat and lon")
                        continue
                    else: # i.e., has lat&lon
                        if (m_has_dims['has_lev']) and (not pres_levs):
                            print(f"\t    WARNING: skipping global map for {var} as it has more than lev dimension, but no pressure levels were provided")
                            continue
                #Skip this variable/case if the regridded climo file doesn't exist:
                if (umdata is None) or (vmdata is None):
                    dmsg = f"\t    WARNING: No test file for {case_name} for variable `{var}`, global lat/lon mean plotting skipped."
                    adfobj.debug_log(dmsg)
                    continue



                #Determine dimensions of variable:
                if unstruct_plotting:
                    #has_dims = {}
                    if len(wgt.n_face) == len(wgt_base.n_face):
                        vres["wgt"] = wgt
                        vres["indataset"] = mdataset
                        #has_dims = {}
                        #has_dims['has_lev'] = False
                    else:
                        print("The weights are different between test and baseline. Won't continue, eh.")
                        return

                    if (not unstruct_case) and (unstruct_base):
                        print("Base is unstructured but Test is lat/lon. Can't continue?")
                        return
                    if (unstruct_case) and (not unstruct_base):
                        print("Base is lat/lon but Test is unstructured. Can't continue?")
                        return
                    if (unstruct_case) and (unstruct_base):
                        unstructured=True
                    if (not unstruct_case) and (not unstruct_base):
                        unstructured=False
                # Check output file. If file does not exist, proceed.
                # If file exists:
                #   if redo_plot is true: delete it now and make plot
                #   if redo_plot is false: add to website and move on
                """doplot = {}

                if (not m_has_dims['has_lev']) or (not o_has_dims['has_lev']):
                    for s in seasons:
                        plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"
                        doplot[plot_name] = plot_file_op(adfobj, plot_name, var, case_name, s, web_category, redo_plot, "LatLon")
                else:
                    for pres in pres_levs:
                        for s in seasons:
                            plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"
                            doplot[plot_name] = plot_file_op(adfobj, plot_name, f"{var}_{pres}hpa", case_name, s, web_category, redo_plot, "LatLon")
                if all(value is None for value in doplot.values()):
                    print(f"\t    INFO: All plots exist for {var}. Redo is {redo_plot}. Existing plots added to website data. Continue.")
                    continue"""

                #If both fields have the required dimensions, then
                #proceed with plotting:
                #if has_dims_cam:
                if 1==1:

                    #
                    # Seasonal Averages
                    # Note: xarray can do seasonal averaging,
                    # but depends on having time accessor,
                    # which these prototype climo files do not have.
                    #

                    #Create new dictionaries:
                    umseasons = {}
                    vmseasons = {}
                    uoseasons = {}
                    voseasons = {}
                    udseasons = {} # hold the differences
                    vdseasons = {}
                    upseasons = {} # hold the percent differences
                    vpseasons = {}

                    #if has_lev:
                    if m_has_dims['has_lev']:

                        # Loop over levels
                        for lv in pres_levs:

                            #Check that the user-requested pressure level
                            #exists in the model data, which should already
                            #have been interpolated to the standard reference
                            #pressure levels:
                            if (not (lv in umdata['lev'])) or (not (lv in uodata['lev'])):
                                print(f"\t    WARNING: plot_press_levels value '{lv}' not present in {var} [test: {(lv in umdata['lev'])}, ref: {lv in uodata['lev']}], so skipping.")
                                continue

                            vres['lev'] = int(lv)
                            vres["vector"] = True

                            #Loop over season dictionary:
                            for s in seasons:
                                umseasons[s] = (utils.seasonal_mean(umdata, season=s, is_climo=True)).sel(lev=lv)
                                vmseasons[s] = (utils.seasonal_mean(vmdata, season=s, is_climo=True)).sel(lev=lv)
                                uoseasons[s] = (utils.seasonal_mean(uodata, season=s, is_climo=True)).sel(lev=lv)
                                voseasons[s] = (utils.seasonal_mean(vodata, season=s, is_climo=True)).sel(lev=lv)
                                # difference: each entry should be (lat, lon)
                                udseasons[s] = umseasons[s] - uoseasons[s]
                                vdseasons[s] = vmseasons[s] - voseasons[s]

                                upseasons[s] = (umseasons[s] - uoseasons[s]) / np.abs(uoseasons[s]) * 100.0
                                upseasons[s] = upseasons[s].where(np.isfinite(upseasons[s]), np.nan)
                                vpseasons[s] = (vmseasons[s] - voseasons[s]) / np.abs(voseasons[s]) * 100.0
                                vpseasons[s] = vpseasons[s].where(np.isfinite(vpseasons[s]), np.nan)

                                vres["umdlfld_nowrap"] = umseasons[s]
                                vres["vmdlfld_nowrap"] = vmseasons[s]
                                vres["uobsfld_nowrap"] = uoseasons[s]
                                vres["vobsfld_nowrap"] = voseasons[s]
                                vres["udiffld_nowrap"] = udseasons[s]
                                vres["vdiffld_nowrap"] = vdseasons[s]
                                vres["upctdiffld_nowrap"] = upseasons[s]
                                vres["vpctdiffld_nowrap"] = vpseasons[s]

                                vres["season"] = s

                                # time to make plot; here we'd probably loop over whatever plots we want for this variable
                                # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                                plot_name = plot_loc / f"{var_name}_{lv}hpa_{s}_LatLon_Vector_Mean.{plot_type}"


                                # Check redo_plot. If set to True: remove old plot, if it already exists:
                                if (not redo_plot) and plot_name.is_file():
                                    #Add already-existing plot to website (if enabled):
                                    adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                                    adfobj.add_website_data(plot_name, f"{var_name}_{lv}hpa", case_name, category=web_category,
                                                            season=s, plot_type="LatLon_Vector")

                                    #Continue to next iteration:
                                    continue
                                elif (redo_plot) and plot_name.is_file():
                                    plot_name.unlink()

                                # pass in casenames
                                vres["case_name"] = case_name
                                vres["baseline"] = data_src

                                #Create new plot:
                                # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                # This relies on `plot_map_and_save` knowing how to deal with the options
                                # currently knows how to handle:
                                #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                #   *Any other entries will be ignored.
                                # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                                pf.plot_map_and_save(adfobj, plot_name, case_nickname, base_nickname,
                                                    [syear_cases[case_idx],eyear_cases[case_idx]],
                                                    [syear_baseline,eyear_baseline],
                                                    umseasons[s], uoseasons[s],
                                                    udseasons[s], upseasons[s],
                                                    obs=obs, unstructured=unstructured, **vres)

                                #Add plot to website (if enabled):
                                adfobj.add_website_data(plot_name, f"{var_name}_{lv}hpa", case_name, category=web_category,
                                                        season=s, plot_type="LatLon_Vector")

                            #End for (seasons)
                        #End for (pressure levels)
                    else:

                        #Loop over season dictionary:
                        for s in seasons:
                            umseasons[s] = utils.seasonal_mean(umdata, season=s, is_climo=True)
                            vmseasons[s] = utils.seasonal_mean(vmdata, season=s, is_climo=True)
                            uoseasons[s] = utils.seasonal_mean(uodata, season=s, is_climo=True)
                            voseasons[s] = utils.seasonal_mean(vodata, season=s, is_climo=True)
                            # difference: each entry should be (lat, lon)
                            udseasons[s] = umseasons[s] - uoseasons[s]
                            vdseasons[s] = vmseasons[s] - voseasons[s]

                            upseasons[s] = (umseasons[s] - uoseasons[s]) / np.abs(uoseasons[s]) * 100.0
                            upseasons[s] = upseasons[s].where(np.isfinite(upseasons[s]), np.nan)
                            vpseasons[s] = (vmseasons[s] - voseasons[s]) / np.abs(voseasons[s]) * 100.0
                            vpseasons[s] = vpseasons[s].where(np.isfinite(vpseasons[s]), np.nan)

                            vres["umdlfld_nowrap"] = umseasons[s]
                            vres["vmdlfld_nowrap"] = vmseasons[s]
                            vres["uobsfld_nowrap"] = uoseasons[s]
                            vres["vobsfld_nowrap"] = voseasons[s]
                            vres["udiffld_nowrap"] = udseasons[s]
                            vres["vdiffld_nowrap"] = vdseasons[s]
                            vres["upctdiffld_nowrap"] = upseasons[s]
                            vres["vpctdiffld_nowrap"] = vpseasons[s]

                            vres["season"] = s
                            vres["vector"] = True

                            # time to make plot; here we'd probably loop over whatever plots we want for this variable
                            # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                            plot_name = plot_loc / f"{var_name}_{s}_LatLon_Vector_Mean.{plot_type}"

                            # Check redo_plot. If set to True: remove old plot, if it already exists:
                            redo_plot = adfobj.get_basic_info('redo_plot')
                            if (not redo_plot) and plot_name.is_file():
                                #Add already-existing plot to website (if enabled):
                                adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                                adfobj.add_website_data(plot_name, var_name, case_name, category=web_category,
                                                        season=s, plot_type="LatLon_Vector")

                                #Continue to next iteration:
                                continue
                            elif (redo_plot) and plot_name.is_file():
                                plot_name.unlink()

                            # pass in casenames
                            vres["case_name"] = case_name
                            vres["baseline"] = data_src
                            vres["var_name"] = var_name

                            #Create new plot:
                            # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                            # This relies on `plot_map_and_save` knowing how to deal with the options
                            # currently knows how to handle:
                            #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                            #   *Any other entries will be ignored.
                            # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                            pf.plot_map_and_save(adfobj, plot_name, case_nickname, base_nickname,
                                                    [syear_cases[case_idx],eyear_cases[case_idx]],
                                                    [syear_baseline,eyear_baseline],
                                                    umseasons[s], uoseasons[s],
                                                    udseasons[s], upseasons[s],
                                                    obs=obs, unstructured=unstructured, **vres)

                            #Add plot to website (if enabled):
                            adfobj.add_website_data(plot_name, var_name, case_name, category=web_category,
                                                    season=s, plot_type="LatLon_Vector")

                        #End for
                    #End if (has_lev)
                #End if (has_dims)
            #End for (cases)
        #End for (data sources)
    #End for (variables)

    #Notify user that script has ended:
    print("  ...lat/lon vector maps have been generated successfully.")

##############
#END OF SCRIPT