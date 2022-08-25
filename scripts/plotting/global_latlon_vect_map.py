def global_latlon_vect_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model vector fields with continental
    overlays.
    Description of function inputs:
    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_data_loc"
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

    #Import necessary modules:
    #------------------------
    from pathlib import Path  # python standard library

    # data loading / analysis
    import xarray as xr
    import numpy as np

    #CAM diagnostic plotting functions:
    import plotting_functions as pf
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal vector fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("\n  Generating lat/lon vector maps...")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output path for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #CAM simulation variables:
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    #Attempt to grab case start_years (not currently required):
    syear_cases = adfobj.get_cam_info('start_year')
    eyear_cases = adfobj.get_cam_info('end_year')

    if (syear_cases and eyear_cases) == None:
        syear_cases = [None]*len(case_names)
        eyear_cases = [None]*len(case_names)

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

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("\t No observations found to plot against, so no vector maps will be generated.")
            return

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = model_rgrid_loc #Just use the re-gridded model data path

        #Attempt to grab baseline start_years (not currently required):
        syear_baseline = adfobj.get_baseline_info('start_year')
        eyear_baseline = adfobj.get_baseline_info('end_year')

        if (syear_baseline and eyear_baseline) == "None":
            print("No given climo years for baseline, gathering from time series files.")
            #Time series files (to be used for climo years):
            baseline_ts_locs = adfobj.get_baseline_info('cam_ts_loc', required=True)
            starting_location = Path(baseline_ts_locs)
            files_list = sorted(starting_location.glob('*.nc'))
            syear_baseline = int(files_list[0].stem[-13:-9])
            eyear_baseline = int(files_list[0].stem[-6:-2])
    

        #Grab baseline case nickname
        base_nickname = adfobj.get_baseline_info('case_nickname')
        if base_nickname == None:
            base_nickname = data_name
    #End if


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
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

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
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_latlon_vect_map: Found variable defaults for {var}")

            #Extract category (if available):
            web_category = vres.get("category", None)

        else:
            adfobj.debug_log(f"global_latlon_vect_map: Skipping '{var}' as no variable defaults were found")
            continue
        #End if

        #Make sure that variable is part of a vector pair:
        if "vector_pair" in vres:
            var_pair = vres["vector_pair"]
            var_name = vres["vector_name"]
        else:
            adfobj.debug_log(f"variable '{var}' not a vector pair")
            continue
        #End if

        #Add variables to "skipped vars" set:
        skip_vars.add(var)
        skip_vars.add(var_pair)

        # For global maps, also set the central longitude:
        # can be specified in adfobj basic info as 'central_longitude' or supplied as a number,
        # otherwise defaults to 180
        vres['central_longitude'] = pf.get_central_longitude(adfobj)

        #Determine observations to compare against:
        if adfobj.compare_obs:
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
                dmsg = f"No obs found for variable `{var}`, lat/lon vector map plotting skipped."
                adfobj.debug_log(dmsg)
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
                dmsg = f"No obs found for variable `{var}`, lat/lon vector map plotting skipped."
                adfobj.debug_log(dmsg)
                continue
            #End if

        else:
            #Set "data_var" for consistent use below:
            data_var = [var, var_pair]
        #End if

        #Notify user of variable being plotted:
        print(f"\t - lat/lon vector maps for {var},{var_pair}")

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #For now, only grab one file (but convert to list for use below)
                uoclim_fils = [udclimo_loc]
                voclim_fils = [vdclimo_loc]
            else:
                uoclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{data_var[0]}_baseline.nc"))
                voclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{data_var[1]}_baseline.nc"))
            #End if

            if len(uoclim_fils) > 1:
                uoclim_ds = xr.open_mfdataset(uoclim_fils, combine='by_coords')
            elif len(uoclim_fils) == 1:
                sfil = str(uoclim_fils[0])
                uoclim_ds = xr.open_dataset(sfil)
            else:
                print("\t ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"\t INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"\t INFO: The glob is: {data_src}_{data_var[0]}_*.nc")
                continue
            #End if

            if len(voclim_fils) > 1:
                voclim_ds = xr.open_mfdataset(voclim_fils, combine='by_coords')
            elif len(voclim_fils) == 1:
                sfil = str(voclim_fils[0])
                voclim_ds = xr.open_dataset(sfil)
            else:
                print("\t ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"\t INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"\t INFO: The glob is: {data_src}_{data_var[1]}_*.nc")
                continue
            #End if

            #Extract variables of interest:
            uodata = uoclim_ds[data_var[0]].squeeze()  # squeeze in case of degenerate dimensions
            vodata = voclim_ds[data_var[1]].squeeze()  # squeeze in case of degenerate dimensions

            #Convert units if requested (assumes units between model and data are the same):
            uodata = uodata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
            vodata = vodata * vres.get("scale_factor",1) + vres.get("add_offset", 0)

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                if (syear_cases[case_idx] and eyear_cases[case_idx]) == None:
                     #Time series files (to be used for climo years):
                     cam_ts_locs = adfobj.get_cam_info('cam_ts_loc', required=True)
                     print("No case climo years given, extracting from timeseries file...")
                     starting_location = Path(cam_ts_locs[case_idx])
                     files_list = sorted(starting_location.glob('*nc'))
                     syear_case = int(files_list[0].stem[-13:-9])
                     eyear_case = int(files_list[0].stem[-6:-2])

                else:
                    syear_case = syear_cases[case_idx]
                    eyear_case = eyear_cases[case_idx]
                    #syear_case = str(syear_case).zfill(4)

                #Set case nickname:
                case_nickname = test_nicknames[case_idx]

                #Set plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print("\t    {} not found, making new directory".format(plot_loc))
                    plot_loc.mkdir(parents=True)
                #End if

                # load re-gridded model files:
                umclim_fils = sorted(mclimo_rg_loc.glob(f"{data_src}_{case_name}_{var}_*.nc"))
                vmclim_fils = sorted(mclimo_rg_loc.glob(f"{data_src}_{case_name}_{var_pair}_*.nc"))

                if len(umclim_fils) > 1:
                    umclim_ds = xr.open_mfdataset(umclim_fils, combine='by_coords')
                else:
                    umclim_ds = xr.open_dataset(umclim_fils[0])
                #End if

                if len(vmclim_fils) > 1:
                    vmclim_ds = xr.open_mfdataset(vmclim_fils, combine='by_coords')
                elif len(vmclim_fils) == 1:
                    vmclim_ds = xr.open_dataset(vmclim_fils[0])
                else:
                    #The vector pair was never processed, so skip varaible:
                    print(f"\t Missing vector pair '{var_pair}' for variable '{var}', so skipping variable")
                    continue
                #End if

                #Extract variable of interest
                umdata = umclim_ds[var].squeeze()
                vmdata = vmclim_ds[var_pair].squeeze()

                #Convert units if requested:
                umdata = umdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                vmdata = vmdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)

                #Check dimensions:
                has_lat, has_lev = pf.zm_validate_dims(umdata)  # assumes will work for both mdata & odata

                # update units
                # NOTE: looks like our climo files don't have all their metadata
                uodata.attrs['units'] = vres.get("new_unit", uodata.attrs.get('units', 'none'))
                vodata.attrs['units'] = vres.get("new_unit", vodata.attrs.get('units', 'none'))
                umdata.attrs['units'] = vres.get("new_unit", umdata.attrs.get('units', 'none'))
                vmdata.attrs['units'] = vres.get("new_unit", vmdata.attrs.get('units', 'none'))

                #Determine if observations/baseline have the correct dimensions:
                if has_lev:
                    has_dims = pf.lat_lon_validate_dims(uodata.isel(lev=0))
                else:
                    has_dims = pf.lat_lon_validate_dims(uodata)
                #End if

                if has_dims:
                    #If observations/baseline CAM have the correct
                    #dimensions, does the input CAM run have correct
                    #dimensions as well?
                    if has_lev:
                        has_dims_cam = pf.lat_lon_validate_dims(umdata.isel(lev=0))
                    else:
                        has_dims_cam = pf.lat_lon_validate_dims(umdata)
                    #End if
                #End if

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
                    umseasons = {}
                    vmseasons = {}
                    uoseasons = {}
                    voseasons = {}
                    udseasons = {} # hold the differences
                    vdseasons = {} # hold the differences

                    if has_lev:

                        # Loop over levels
                        for lv in pres_levs:

                            #Check that the user-requested pressure level
                            #exists in the model data, which should already
                            #have been interpolated to the standard reference
                            #pressure levels:
                            if not (lv in umclim_ds['lev']):
                                #Move on to the next pressure level:
                                print(f"\t plot_press_levels value '{lv}' not a standard reference pressure, so skipping.")
                                continue
                            #End if

                            #Loop over season dictionary:
                            for s in seasons:
                                umseasons[s] = umdata.sel(time=seasons[s],lev=lv).mean(dim='time')
                                vmseasons[s] = vmdata.sel(time=seasons[s],lev=lv).mean(dim='time')
                                uoseasons[s] = uodata.sel(time=seasons[s],lev=lv).mean(dim='time')
                                voseasons[s] = vodata.sel(time=seasons[s],lev=lv).mean(dim='time')
                                # difference: each entry should be (lat, lon)
                                udseasons[s] = umseasons[s] - uoseasons[s]
                                vdseasons[s] = vmseasons[s] - voseasons[s]

                                # time to make plot; here we'd probably loop over whatever plots we want for this variable
                                # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                                plot_name = plot_loc / f"{var_name}_{lv}hpa_{s}_LatLon_Vector_Mean.{plot_type}"


                                # Check redo_plot. If set to True: remove old plot, if it already exists:
                                if (not redo_plot) and plot_name.is_file():
                                    #Add already-existing plot to website (if enabled):
                                    adfobj.add_website_data(plot_name, f"{var_name}_{lv}hpa", case_name, category=web_category,
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
                                pf.plot_map_vect_and_save(plot_name, case_nickname, base_nickname, 
                                                        [syear_case,eyear_case],[syear_baseline,eyear_baseline],lv,
                                                        umseasons[s], vmseasons[s],
                                                        uoseasons[s], voseasons[s],
                                                        udseasons[s], vdseasons[s], **vres)

                                #Add plot to website (if enabled):
                                adfobj.add_website_data(plot_name, f"{var_name}_{lv}hpa", case_name, category=web_category,
                                                        season=s, plot_type="LatLon_Vector")

                            #End for (seasons)
                        #End for (pressure levels)
                    else:

                        #Loop over season dictionary:
                        for s in seasons:
                            umseasons[s] = umdata.sel(time=seasons[s]).mean(dim='time')
                            vmseasons[s] = vmdata.sel(time=seasons[s]).mean(dim='time')
                            uoseasons[s] = uodata.sel(time=seasons[s]).mean(dim='time')
                            voseasons[s] = vodata.sel(time=seasons[s]).mean(dim='time')
                            # difference: each entry should be (lat, lon)
                            udseasons[s] = umseasons[s] - uoseasons[s]
                            vdseasons[s] = vmseasons[s] - voseasons[s]

                            # time to make plot; here we'd probably loop over whatever plots we want for this variable
                            # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                            plot_name = plot_loc / f"{var_name}_{s}_LatLon_Vector_Mean.{plot_type}"

                            # Check redo_plot. If set to True: remove old plot, if it already exists:
                            redo_plot = adfobj.get_basic_info('redo_plot')
                            if (not redo_plot) and plot_name.is_file():
                                #Add already-existing plot to website (if enabled):
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
                            pf.plot_map_vect_and_save(plot_name, case_nickname, base_nickname,
                                                      [syear_case,eyear_case],[syear_baseline,eyear_baseline], None,
                                                      umseasons[s], vmseasons[s],
                                                      uoseasons[s], voseasons[s],
                                                      udseasons[s], vdseasons[s], **vres)

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