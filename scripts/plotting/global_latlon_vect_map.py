def global_latlon_vect_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model fields with continental overlays.

    This script is actually the same as 'plot_example', except
    that it uses the shared diagnostics plotting library, as
    opposed to being entirely self-contained.

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

    #Import necessary modules:
    #------------------------
    from pathlib import Path  # python standard library

    # data loading / analysis
    import xarray as xr
    import numpy as np
    import geocat.comp as gcomp

    #CAM diagnostic plotting functions:
    import plotting_functions as pf
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("  Generating lat/lon vector maps...")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output path for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location
    plot_location = plot_locations[0]

    #CAM simulation variables:
    case_names = adfobj.get_cam_info("cam_case_name", required=True)
    case_name = case_names[0]

   # CAUTION:
   # "data" here refers to either obs or a baseline simulation,
   # Until those are both treated the same (via intake-esm or similar)
   # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):
        data_name = "obs"  # does not get used, is just here as a placemarker
        data_list = adfobj.read_config_var("obs_type_list")  # Double caution!
        data_loc  = adfobj.get_basic_info("obs_climo_loc", required=True)

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

    #Set input/output data path variables:
    #------------------------------------
    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_loc      = Path(plot_location)
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

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    skip_vars = set()

    # probably want to do this one variable at a time:
    for var in var_list:
        if var in skip_vars:
            continue
        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"global_latlon_map: Found variable defaults for {var}")

        else:
            adfobj.debug_log(f"global_latlon_map: Skipping '{var}' as no variable defaults were found")
            continue
        if "vector_pair" in vres:
            var_pair = vres["vector_pair"]
            var_name = vres["vector_name"]
        else:
            adfobj.debug_log(f"variable '{var}' not a vector pair")
            continue

        skip_vars.add(var)
        skip_vars.add(var_pair)        


        #Notify user of variable being plotted:
        print(f"\t \u231B lat/lon vector maps for {var},{var_pair}")


        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            uoclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))
            voclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var_pair))))

            if len(uoclim_fils) > 1:
                uoclim_ds = xr.open_mfdataset(uoclim_fils, combine='by_coords')
            elif len(uoclim_fils) == 1:
                sfil = str(uoclim_fils[0])
                uoclim_ds = xr.open_dataset(sfil)
            else:
                print("ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{var}_*.nc")
                continue

            if len(voclim_fils) > 1:
                voclim_ds = xr.open_mfdataset(voclim_fils, combine='by_coords')
            elif len(voclim_fils) == 1:
                sfil = str(voclim_fils[0])
                voclim_ds = xr.open_dataset(sfil)
            else:
                print("ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{var_pair}_*.nc")
                continue

            # load re-gridded model files:
            umclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))
            vmclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var_pair))))


            if len(umclim_fils) > 1:
                umclim_ds = xr.open_mfdataset(umclim_fils, combine='by_coords')
            else:
                umclim_ds = xr.open_dataset(umclim_fils[0])
            if len(vmclim_fils) > 1:
                vmclim_ds = xr.open_mfdataset(vmclim_fils, combine='by_coords')
            else:
                vmclim_ds = xr.open_dataset(vmclim_fils[0])


            #Extract variable of interest
            uodata = uoclim_ds[var].squeeze()  # squeeze in case of degenerate dimensions
            vodata = voclim_ds[var_pair].squeeze()  # squeeze in case of degenerate dimensions
            umdata = umclim_ds[var].squeeze()
            vmdata = vmclim_ds[var_pair].squeeze()

            # APPLY UNITS TRANSFORMATION IF SPECIFIED:

            uodata = uodata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
            vodata = vodata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
            umdata = umdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
            vmdata = vmdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)



            has_lat, has_lev = pf.zm_validate_dims(umdata)  # assumes will work for both mdata & odata

            if has_lev:
                print("variable has lev dimension.")
                # need hyam, hybm, P0 once, and need PS for both datasets
     # note in future, they may have different vertical levels or one may need pressure level interp and one may not
                if 'hyam' not in umclim_ds:
                    print("\u2757 PROBLEM -- NO hyam")
                    print(umclim_ds)
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
                        continue  # what else could we do?
                if 'PS' in uoclim_ds:
                    ops = uoclim_ds['PS']
                else:
                    # look for the file (this isn't great b/c we'd have to constantly re-load)
                    ops_files = sorted(list(oclimo_rg_loc.glob("{}_{}_PS_*.nc".format(data_src, case_name))))
                    if len(ops_files) > 0:
                        ops_ds = _load_dataset(ops_files)
                        ops = ops_ds['PS']
                    else:
                        continue  # what else could we do?

                if not pres_levs:
                   print("vector plot only works with pressure levels")
                   continue


        # now add in syntax to interpolate to a pressure level with geocat
        # this needs to be improved by checking if it's on plevs already, hybrid or sigma

                umdata =     pf.lev_to_plev(umdata, mps, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                vmdata =     pf.lev_to_plev(vmdata, mps, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                uodata =     pf.lev_to_plev(uodata, ops, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)
                vodata =     pf.lev_to_plev(vodata, ops, mhya, mhyb, P0=100000.0, new_levels=np.array(np.array(pres_levs)*100,dtype='float32'),convert_to_mb=True)

                
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
            if has_dims:
                #If observations/baseline CAM have the correct
                #dimensions, does the input CAM run have correct
                #dimensions as well?
                if has_lev:
                    has_dims_cam = pf.lat_lon_validate_dims(umdata.isel(lev=0))
                else:
                    has_dims_cam = pf.lat_lon_validate_dims(umdata)

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
                                plot_name = plot_loc / "{}_{}_{}_LatLon_Vector_Mean.{}".format(var_name, s, lv, plot_type)

                        #Remove old plot, if it already exists:
                                if plot_name.is_file():
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

                             #Remove old plot, if it already exists:
                                    if plot_name.is_file():
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


      #Notify user that script has ended:
    print("  ...lat/lon vector maps have been generated successfully.")

##############
#END OF SCRIPT
