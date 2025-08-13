from pathlib import Path  # python standard library

# data loading / analysis
import xarray as xr
import numpy as np

# ADF library
import plotting_functions as pf

def polar_map(adfobj):
    """
    This script/function generates polar maps of model fields with continental overlays.
    Plots style follows old AMWG diagnostics:
      - plots for ANN, DJF, MAM, JJA, SON
      - separate files for each hemisphere, denoted `_nh` and `_sh` in file names.
      - mean files shown on top row, difference on bottom row (centered)
    [based on global_latlon_map.py]
    """
    #Notify user that script has started:
    msg = "\n  Generating polar maps..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #
    # Use ADF api to get all necessary information
    #
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)
    #model_rgrid_locs = adfobj.get_cam_info("cam_climo_regrid_loc", required=True)

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    plot_locations = adfobj.plot_location

    kwargs = {}

    #
    unstruct_plotting = adfobj.unstructured_plotting
    print("unstruct_plotting", unstruct_plotting)
    if unstruct_plotting:
        kwargs["unstructured_plotting"] = unstruct_plotting
        #mesh_file = '/glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc'#adfobj.mesh_file
        #kwargs["mesh_file"] = mesh_file
    else:
        unstructured=False
    print("kwargs", kwargs)

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
            print("\t No observations found to plot against, so no polar maps will be generated.")
            return
    else:
        obs = False
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc  = model_rgrid_loc #Just use the re-gridded model data path
        #data_loc = Path(adfobj.get_baseline_info("cam_climo_regrid_loc", required=True))
    #End if

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

    #Grab all case nickname(s)
    test_nicknames = adfobj.case_nicknames["test_nicknames"]
    base_nickname = adfobj.case_nicknames["base_nickname"]

    comp = adfobj.model_component
    if comp == "atm":
        hemis = ["NHPolar", "SHPolar"]
    if comp == "lnd":
        hemis = ["Arctic"]

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

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    # probably want to do this one variable at a time:
    for var in var_list:
        #Notify user of variable being plotted:
        print(f"\t - polar maps for {var}")

        if var not in adfobj.data.ref_var_nam:
            dmsg = f"\t    WARNING: No reference data found for variable `{var}`, polar lat/lon mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            continue

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
                dmsg = f"\t    WARNING: No obs found for variable `{var}`, polar map skipped."
                adfobj.debug_log(dmsg)
                continue
        else:
            #Set "data_var" for consistent use below:
            data_var = var
        #End if

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"polar_map: Found variable defaults for {var}")

            #Extract category (if available):
            web_category = vres.get("category", None)

        else:
            vres = {}
            web_category = None
        #End if

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #Set data name:
                data_name = data_src

            if unstruct_plotting:
                mesh_file = adfobj.mesh_files["baseline_mesh_file"]
                kwargs["mesh_file"] = mesh_file
                odata = adfobj.data.load_reference_climo_da(data_name, data_var, **kwargs)
                #if ('ncol' in odata.dims) or ('lndgrid' in odata.dims):
                if 1==1:
                    unstruct_base = True
                    odataset = adfobj.data.load_reference_climo_dataset(data_name, data_var, **kwargs) 
                    area = odataset.area.isel(time=0)
                    landfrac = odataset.landfrac.isel(time=0)
                    # calculate weights
                    wgt_base = area * landfrac / (area * landfrac).sum()
            else:
                odata = adfobj.data.load_reference_regrid_da(data_name, data_var, **kwargs)
            if odata is None:
                print("\t    WARNING: Did not find any regridded reference climo files. Will try to skip.")
                print(f"\t    INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"\t      The glob is: {data_src}_{data_var}_*.nc")
                continue

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):
                #mclimo_rg_loc = Path(model_rgrid_locs[case_idx])

                #Set case nickname:
                case_nickname = test_nicknames[case_idx]

                #Set output plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print(f"    {plot_loc} not found, making new directory")
                    plot_loc.mkdir(parents=True)

                if unstruct_plotting:
                    mesh_file = adfobj.mesh_files["test_mesh_file"][case_idx]
                    kwargs["mesh_file"] = mesh_file
                    mdata = adfobj.data.load_climo_da(case_name, var, **kwargs)
                    #if ('ncol' in mdata.dims) or ('lndgrid' in mdata.dims):
                    if 1==1:
                        unstruct_case = True
                        mdataset = adfobj.data.load_climo_dataset(case_name, var, **kwargs) 
                        area = mdataset.area.isel(time=0)
                        landfrac = mdataset.landfrac.isel(time=0)
                        # calculate weights
                        wgt = area * landfrac / (area * landfrac).sum()
                else:
                    mdata = adfobj.data.load_regrid_da(case_name, var, **kwargs)

                if mdata is None:
                    print("\t    WARNING: Did not find any regridded test climo files. Will try to skip.")
                    print(f"\t    INFO: Data Location, mclimo_rg_loc, is {mclimo_rg_loc}")
                    print(f"\t      The glob is: {data_src}_{case_name}_{var}_*.nc")
                    continue
                #End if

                if unstruct_plotting:
                    has_dims = {}
                    if len(wgt.n_face) == len(wgt_base.n_face):
                        vres["wgt"] = wgt
                        has_dims = {}
                        has_dims['has_lev'] = False
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

                #Determine dimensions of variable:
                has_dims = pf.lat_lon_validate_dims(odata)
                has_lat_ref, has_lev_ref = pf.zm_validate_dims(odata)
                has_lat, has_lev = pf.zm_validate_dims(mdata)
                #if has_dims:
                if (not has_lev) and (not has_lev_ref):
                    #If observations/baseline CAM have the correct
                    #dimensions, does the input CAM run have correct
                    #dimensions as well?
                    has_dims_cam = pf.lat_lon_validate_dims(mdata)

                    #If both fields have the required dimensions, then
                    #proceed with plotting:
                    #if has_dims_cam:
                    if 2==2:

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
                        pseasons = {} # hold percent change

                        #Loop over season dictionary:
                        for s in seasons:
                            mseasons[s] = pf.seasonal_mean(mdata, season=s, is_climo=True)
                            oseasons[s] = pf.seasonal_mean(odata, season=s, is_climo=True)
                            # difference: each entry should be (lat, lon)
                            dseasons[s] = mseasons[s] - oseasons[s]
                            dseasons[s].attrs['units'] = mseasons[s].attrs['units']
                            
                            # percent change 
                            pseasons[s] = (mseasons[s] - oseasons[s]) / np.abs(oseasons[s]) * 100.0 # relative change
                            pseasons[s].attrs['units'] = '%'

                            # make plots: northern and southern hemisphere separately:
                            for hemi_type in hemis:

                                #Create plot name and path:
                                plot_name = plot_loc / f"{var}_{s}_{hemi_type}_Mean.{plot_type}"

                                # If redo_plot set to True: remove old plot, if it already exists:
                                if (not redo_plot) and plot_name.is_file():
                                    #Add already-existing plot to website (if enabled):
                                    adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                                    adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                                            season=s, plot_type=hemi_type)

                                    #Continue to next iteration:
                                    continue
                                else:
                                    if plot_name.is_file():
                                        plot_name.unlink()

                                    #Create new plot:
                                    # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                    # This relies on `plot_map_and_save` knowing how to deal with the options
                                    # currently knows how to handle:
                                    #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                    #   *Any other entries will be ignored.
                                    # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.

                                    if comp == "atm":
                                        #Determine hemisphere to plot based on plot file name:
                                        if hemi_type == "NHPolar":
                                            hemi = "NH"
                                        else:
                                            hemi = "SH"
                                        #End if
                                    if comp == "lnd":
                                        hemi = hemi_type
                                    # Exclude certain plots, this may get difficult
                                    if (var != 'GRAINC_TO_FOOD') or (var != 'DSTFLXT') or (var != 'MEG_isoprene') or (var != 'FAREA_BURNED'):
                                        pf.make_polar_plot(plot_name, case_nickname, base_nickname,
                                                           [syear_cases[case_idx],eyear_cases[case_idx]],
                                                           [syear_baseline,eyear_baseline],
                                                           mseasons[s], oseasons[s], dseasons[s], pseasons[s],
                                                           hemisphere=hemi, obs=obs, unstructured=unstructured,
                                                           **vres)

                                        #Add plot to website (if enabled):
                                        adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                                                season=s, plot_type=hemi_type)

                    else: #mdata dimensions check
                        print(f"\t    WARNING: skipping polar map for {var} as it doesn't have only lat/lon dims.")
                    #End if (dimensions check)

                elif pres_levs: #Is the user wanting to interpolate to a specific pressure level?

                    #Check that case inputs have the correct dimensions (including "lev"):
                    has_lat, has_lev = pf.zm_validate_dims(mdata)  # assumes will work for both mdata & odata

                    """# check if there is a lat dimension:
                    if not has_lat:
                        print(
                            f"\t    WARNING: Variable {var} is missing a lat dimension for '{case_name}', cannot continue to plot."
                        )
                        continue
                    # End if"""

                    #Check that case inputs have the correct dimensions (including "lev"):
                    has_lat_ref, has_lev_ref = pf.zm_validate_dims(odata)

                    """# check if there is a lat dimension:
                    if not has_lat_ref:
                        print(
                            f"\t    WARNING: Variable {var} is missing a lat dimension for '{data_name}', cannot continue to plot."
                        )
                        continue"""

                    #Check if both cases have vertical levels to continue
                    if (has_lev) and (has_lev_ref):

                        #Loop over pressure levels:
                        for pres in pres_levs:

                            #Check that the user-requested pressure level
                            #exists in the model data, which should already
                            #have been interpolated to the standard reference
                            #pressure levels:
                            if not (pres in mdata['lev']):
                                #Move on to the next pressure level:
                                print(f"\t    WARNING: plot_press_levels value '{pres}' not a standard reference pressure, so skipping.")
                                continue
                            #End if

                            #Create new dictionaries:
                            mseasons = {}
                            oseasons = {}
                            dseasons = {} # hold the differences
                            pseasons = {} # hold percent change

                            #Loop over season dictionary:
                            for s in seasons:
                                mseasons[s] = (pf.seasonal_mean(mdata, season=s, is_climo=True)).sel(lev=pres)
                                oseasons[s] = (pf.seasonal_mean(odata, season=s, is_climo=True)).sel(lev=pres)
                                # difference: each entry should be (lat, lon)
                                dseasons[s] = mseasons[s] - oseasons[s]
                                dseasons[s].attrs['units'] = mseasons[s].attrs['units']
                                
                                # percent change
                                pseasons[s] = (mseasons[s] - oseasons[s]) / abs(oseasons[s]) * 100.0 # relative change
                                pseasons[s].attrs['units'] = '%'
                                #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
                                #pseasons[s] = pseasons[s].where(np.isfinite(pseasons[s]), np.nan)
                                #pseasons[s] = pseasons[s].fillna(0.0)

                                # make plots: northern and southern hemisphere separately:
                                for hemi_type in hemis:
                                    print("mseasons[s].shape",mseasons[s].shape)
                                    print("oseasons[s].shape",oseasons[s].shape)
                                    print("dseasons[s].shape",dseasons[s].shape)
                                    print("pseasons[s].shape",pseasons[s].shape)

                                    #Create plot name and path:
                                    plot_name = plot_loc / f"{var}_{pres}hpa_{s}_{hemi_type}_Mean.{plot_type}"

                                    # If redo_plot set to True: remove old plot, if it already exists:
                                    if (not redo_plot) and plot_name.is_file():
                                        #Add already-existing plot to website (if enabled):
                                        adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                                        adfobj.add_website_data(plot_name, f"{var}_{pres}hpa",
                                                                case_name, category=web_category,
                                                                season=s, plot_type=hemi_type)

                                        #Continue to next iteration:
                                        continue
                                    else:
                                        if plot_name.is_file():
                                            plot_name.unlink()

                                        #Create new plot:
                                        # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
                                        # This relies on `plot_map_and_save` knowing how to deal with the options
                                        # currently knows how to handle:
                                        #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
                                        #   *Any other entries will be ignored.
                                        # NOTE: If we were doing all the plotting here, we could use whatever we want from the provided YAML file.

                                        if comp == "atm":
                                            #Determine hemisphere to plot based on plot file name:
                                            if hemi_type == "NHPolar":
                                                hemi = "NH"
                                            else:
                                                hemi = "SH"
                                            #End if
                                        if comp == "lnd":
                                            hemi = hemi_type

                                        pf.make_polar_plot(plot_name, case_nickname, base_nickname,
                                                     [syear_cases[case_idx],eyear_cases[case_idx]],
                                                     [syear_baseline,eyear_baseline],
                                                     mseasons[s], oseasons[s], dseasons[s], pseasons[s],
                                                     hemisphere=hemi, obs=obs, unstructured=unstructured,
                                                     **vres)

                                        #Add plot to website (if enabled):
                                        adfobj.add_website_data(plot_name, f"{var}_{pres}hpa",
                                                                case_name, category=web_category,
                                                                season=s, plot_type=hemi_type)

                            #End for (seasons)
                        #End for (pressure level)
                    else:
                        print(f"\t    WARNING: variable '{var}' has no vertical dimension but is not just time/lat/lon, so skipping.")
                    #End if (has_lev)

                else: #odata dimensions check
                    print(f"\t    WARNING: skipping polar map for {var} as it has more than lat/lon dims, but no pressure levels were provided")
                #End if (dimensions check and pressure levels)
            #End for (case loop)
        #End for (obs/baseline loop)
    #End for (variable loop)

    #Notify user that script has ended:
    print("  ...polar maps have been generated successfully.")

##############
#END OF `polar_map` function

##############
# END OF FILE
