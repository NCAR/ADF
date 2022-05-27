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
            print("No observations found to plot against, so no vector maps will be generated.")
            return

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = adfobj.get_baseline_info("cam_climo_loc", required=True)
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
                dclimo_loc = var_obs_dict[var]["obs_file"]
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
                dclimo_loc = var_obs_dict[var_pair]["obs_file"]
                #Extract target list (eventually will be a list, for now need to convert):
                data_list = [var_obs_dict[var_pair]["obs_name"]]
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
            uoclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{data_var[0]}_*.nc"))
            voclim_fils = sorted(dclimo_loc.glob(f"{data_src}_{data_var[1]}_*.nc"))

            if len(uoclim_fils) > 1:
                uoclim_ds = xr.open_mfdataset(uoclim_fils, combine='by_coords')
            elif len(uoclim_fils) == 1:
                sfil = str(uoclim_fils[0])
                uoclim_ds = xr.open_dataset(sfil)
            else:
                print("ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{data_var[0]}_*.nc")
                continue
            #End if

            if len(voclim_fils) > 1:
                voclim_ds = xr.open_mfdataset(voclim_fils, combine='by_coords')
            elif len(voclim_fils) == 1:
                sfil = str(voclim_fils[0])
                voclim_ds = xr.open_dataset(sfil)
            else:
                print("ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{data_var[1]}_*.nc")
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
                    print(f"Missing vector pair '{var_pair}' for variable '{var}', so skipping variable")
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

                if has_lev:

                    #For now, there is no easy way to use observations with specified pressure levels, so
                    #bail out here:
                    if adfobj.compare_obs:
                        print(f"\t - plot_press_levels currently doesn't work with observations, so skipping case for variable '{var}'.")
                        continue #Skip variable
                    #End if

                    print("latlon_vect: variable has lev dimension.")
                    # need hyam, hybm, P0 once, and need PS for both datasets
                    # note in future, they may have different vertical levels or one may need pressure level interp and one may not

                    if 'hyam' not in umclim_ds:
                        print(f"\t - plot_pres_levels currently only works with hybrid coordinates, so skipping '{var}'.")
                        continue #Skip variable
                    #End if
                    mhya = umclim_ds['hyam']
                    mhyb = umclim_ds['hybm']
                    if 'time' in mhya.dims:
                        mhya = mhya.isel(time=0).squeeze()
                    if 'time' in mhyb.dims:
                        mhyb = mhyb.isel(time=0).squeeze()
                    if 'P0' in umclim_ds:
                        P0 = umclim_ds['P0']
                    else:
                        P0 = 100000.0  # Pa
                    if 'PS' in umclim_ds:
                        mps = umclim_ds['PS']
                    else:
                        # look for the file (this isn't great b/c we'd have to constantly re-load)
                        mps_files = sorted(list(mclimo_rg_loc.glob("{}_{}_PS_*.nc".format(data_src, case_name))))
                        if len(mps_files) > 0:
                            mps_ds = _load_dataset(mps_files)
                            mps = mps_ds['PS']
                        else:
                            print("\t - can't find surface pressure (PS) anywhere, so skipping '{var}'.")
                            continue  # what else could we do?
                        #End if
                    #End if
                    if 'PS' in uoclim_ds:
                        ops = uoclim_ds['PS']
                    else:
                        # look for the file (this isn't great b/c we'd have to constantly re-load)
                        ops_files = sorted(list(oclimo_rg_loc.glob("{}_{}_PS_*.nc".format(data_src, case_name))))
                        if len(ops_files) > 0:
                            ops_ds = _load_dataset(ops_files)
                            ops = ops_ds['PS']
                        else:
                            print("\t - can't find surface pressure (PS) anywhere, so skipping '{var}'.")
                            continue  # what else could we do?
                        #End if
                    #End if

                    if not pres_levs:
                        print("vector plot only works with pressure levels")
                        continue
                    #End if

                    # now add in syntax to interpolate to a pressure level with geocat
                    # this needs to be improved by checking if it's on plevs already, hybrid or sigma

                    umdata = pf.lev_to_plev(umdata, mps, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                    vmdata = pf.lev_to_plev(vmdata, mps, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)

                    #Only do this for the first case:
                    if case_idx == 0:
                        uodata = pf.lev_to_plev(uodata, ops, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                        vodata = pf.lev_to_plev(vodata, ops, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                    #End if

                # end if for has_lev

                # update units
                # NOTE: looks like our climo files don't have all their metadata
                uodata.attrs['units'] = vres.get("new_unit", uodata.attrs.get('units', 'none'))
                vodata.attrs['units'] = vres.get("new_unit", vodata.attrs.get('units', 'none'))
                umdata.attrs['units'] = vres.get("new_unit", umdata.attrs.get('units', 'none'))
                vmdata.attrs['units'] = vres.get("new_unit", vmdata.attrs.get('units', 'none'))

                #Determine dimensions of variable:
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
                                    continue
                                elif (redo_plot) and plot_name.is_file():
                                    plot_name.unlink()

                                # pass in casenames
                                vres["case_name"] = case_name
                                vres["baseline"] = data_name
                                vres["var_name"] = var_name

                                #Create new plot:
                                # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                # This relies on `plot_map_and_save` knowing how to deal with the options
                                # currently knows how to handle:
                                #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                #   *Any other entries will be ignored.
                                # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                                pf.plot_map_vect_and_save(plot_name, lv, umseasons[s], vmseasons[s], uoseasons[s], voseasons[s], udseasons[s], vdseasons[s], **vres)
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
                            plot_name = plot_loc / "{}_{}_LatLon_Vector_Mean.{}".format(var_name, s, plot_type)

                            # Check redo_plot. If set to True: remove old plot, if it already exists:
                            redo_plot = adfobj.get_basic_info('redo_plot')
                            if (not redo_plot) and plot_name.is_file():
                                continue
                            elif (redo_plot) and plot_name.is_file():
                                plot_name.unlink()
                     
                            # pass in casenames
                            vres["case_name"] = case_name
                            vres["baseline"] = data_name
                            vres["var_name"] = var_name

                            #Create new plot:
                            # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                            # This relies on `plot_map_and_save` knowing how to deal with the options
                            # currently knows how to handle:
                            #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                            #   *Any other entries will be ignored.
                            # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.
                            pf.plot_map_vect_and_save(plot_name, None, umseasons[s], vmseasons[s], uoseasons[s], voseasons[s], udseasons[s], vdseasons[s], **vres)
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
