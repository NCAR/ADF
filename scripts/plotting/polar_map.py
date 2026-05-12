"""Module to make polar stereographic maps."""
from pathlib import Path
import numpy as np

# ADF library
import plotting_functions as pf
import plotting_utils as plot_utils
import adf_utils as utils

def get_hemisphere(hemi_type):
    """Helper function to convert plot type to hemisphere code.
    
    Parameters
    ----------
    hemi_type : str
        if `NHPolar` set NH, otherwise SH
        
    Returns
    -------
    str
        NH or SH
    """
    return "NH" if hemi_type == "NHPolar" else "SH"

def process_seasonal_data(mdata, odata, season):
    """Helper function to calculate seasonal means and differences.
    Parameters
    ----------
    mdata : xarray.DataArray
        test case data
    odata : xarray.DataArray
        reference case data
    season : str
        season (JJA, DJF, MAM, SON)

    Returns
    -------
    mseason : xarray.DataArray
    oseason : xarray.DataArray
    dseason : xarray.DataArray
    pseason : xarray.DataArray
        Seasonal means for test, reference, difference, and percent difference    
    """
    mseason = utils.seasonal_mean(mdata, season=season, is_climo=True)
    oseason = utils.seasonal_mean(odata, season=season, is_climo=True)
    
    # Calculate differences
    dseason = mseason - oseason
    dseason.attrs['units'] = mseason.attrs['units']
    
    # Calculate percent change
    pseason = (mseason - oseason) / np.abs(oseason) * 100.0
    pseason.attrs['units'] = '%'
    pseason = pseason.where(np.isfinite(pseason), np.nan)
    pseason = pseason.fillna(0.0)
    
    return mseason, oseason, dseason, pseason

def polar_map(adfobj):
    """Generate polar maps of model fields with continental overlays."""
    #Notify user that script has started:
    msg = "\n  Generating polar maps..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    var_list = adfobj.diag_var_list

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    plot_locations = adfobj.plot_location
    kwargs = {}

    #
    unstruct_plotting = adfobj.unstructured_plotting
    if unstruct_plotting:
        kwargs["unstructured_plotting"] = unstruct_plotting
        #mesh_file = '/glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc'#adfobj.mesh_file
        #kwargs["mesh_file"] = mesh_file
    else:
        unstructured=False
    #print("kwargs", kwargs)

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    #Grab case years
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    # if doing comparison to obs, but no observations are found, quit
    if adfobj.get_basic_info("compare_obs"):
        var_obs_dict = adfobj.var_obs_dict
        if not var_obs_dict:
            print("\t No observations found to plot against, so no polar maps will be generated.")
            return


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
        print(f"\t - polar maps for {var}")

        if var not in adfobj.data.ref_var_nam:
            dmsg = f"\t    WARNING: No reference data found for variable `{var}`, polar lat/lon mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            continue

        if not adfobj.compare_obs:
            base_name = adfobj.data.ref_labels[var]
        else:
            base_name = adfobj.data.ref_case_label


        # Get variable-specific settings
        vres = res.get(var, {})
        #vres = plot_utils.add_var_to_vres(adfobj, var, vres)
        vres["var"] = var
        web_category = vres.get("category", None)
        vres["plot_type"] = __name__

        # Get all plot info and check existence
        plot_info = []
        all_plots_exist = True
        
        for case_idx, case_name in enumerate(case_names):
            plot_loc = Path(plot_locations[case_idx])
            print("Try target thing")
            tmp_ds = adfobj.data.load_regrid_dataset(case_name, var)
            if tmp_ds is None:
                print("ok, we got prob target thing")
                continue

            has_lev = "lev" in tmp_ds.dims

            for s in seasons:
                for hemi_type in ["NHPolar", "SHPolar"]:
                    if pres_levs and has_lev: # 3-D variable & lev levels specified
                        for pres in pres_levs:
                            plot_name = plot_loc / f"{var}_{pres}hpa_{s}_{hemi_type}_Mean.{plot_type}"
                            info = {
                                'path': plot_name,
                                'var': f"{var}_{pres}hpa",
                                'case': case_name,
                                'case_idx': case_idx,
                                'season': s,
                                'hemi': hemi_type,
                                'lev': pres,
                                'exists': plot_name.is_file()
                            }
                            plot_info.append(info)
                            if (redo_plot is False) and info['exists']:
                                adfobj.add_website_data(info['path'], info['var'],
                                                    info['case'], category=web_category,
                                                    season=s, plot_type=hemi_type)
                            else:
                                all_plots_exist = False
                    elif (not has_lev): # 2-D variable
                        plot_name = plot_loc / f"{var}_{s}_{hemi_type}_Mean.{plot_type}"
                        info = {
                            'path': plot_name,
                            'var': var,
                            'case': case_name,
                            'case_idx': case_idx,
                            'season': s,
                            'hemi': hemi_type,
                            'exists': plot_name.is_file()
                        }
                        plot_info.append(info)
                        if (redo_plot is False) and info['exists']:
                            adfobj.add_website_data(info['path'], info['var'],
                                                  info['case'], category=web_category,
                                                  season=s, plot_type=hemi_type)
                        else:
                            all_plots_exist = False
                    if unstruct_plotting:
                        if case_name == base_name:
                            info["mesh_file"] = adfobj.mesh_files["baseline_mesh_file"]
                        else:
                            info["mesh_file"] = adfobj.mesh_files["test_mesh_file"][case_idx]

        if all_plots_exist:
            print(f"\t    Skipping {var} - all plots already exist")
            continue

        #odata = adfobj.data.load_reference_regrid_da(base_name, var, **kwargs)
        if unstruct_plotting:
            kwargs["mesh_file"] = adfobj.mesh_files["baseline_mesh_file"]
            unstruct_base = True
        odataset = adfobj.data.load_reference_regrid_dataset(base_name, var, **kwargs)
        odata = adfobj.data.load_reference_regrid_da(base_name, var, **kwargs)
        if odataset is None:
            print(f"\t    WARNING: No reference data found for {var}")
            continue
        if comp == "lnd":
            if adfobj.native_grid[case_name] and not adfobj.unstructured_plotting:
                area = odataset.area.isel(time=0)
                landfrac = odataset.landfrac.isel(time=0)
                # calculate weights
                wgt_base = area * landfrac / (area * landfrac).sum()
        if comp == "atm":
            if adfobj.native_grid[case_name] and not adfobj.unstructured_plotting:
                wgt_base = odataset.isel(time=0)[var]

        # Process each case
        for plot in plot_info:
            if plot['exists'] and not redo_plot:
                print("ok, we got prob plot exist and redo thing??")
                continue
                
            case_name = plot['case']
            case_idx = plot['case_idx']
            vres["season"] = plot['season']
            vres["hemi"] = plot['hemi']
            plot_loc = Path(plot_locations[case_idx])
            if unstruct_plotting:
                vres["mesh_file"] = info["mesh_file"]

            if comp == "atm":
                #Determine hemisphere to plot based on plot file name:
                if hemi_type == "NHPolar":
                    hemi = "NH"
                if hemi_type == "SHPolar":
                    hemi = "SH"
                #End if
            if comp == "lnd":
                hemi = hemi_type

            # Ensure plot directory exists
            plot_loc.mkdir(parents=True, exist_ok=True)
            
            if unstruct_plotting:
                kwargs["mesh_file"] = adfobj.mesh_files["test_mesh_file"][case_idx]
                #mdata = adfobj.data.load_climo_da(case_name, var, **kwargs)
                mdata = adfobj.data.load_regrid_da(case_name, var, **kwargs)

                unstruct_case = True
                #mdataset = adfobj.data.load_climo_dataset(case_name, var, **kwargs)
                mdataset = adfobj.data.load_regrid_dataset(case_name, var, **kwargs)
                if comp == "lnd": 
                    if adfobj.native_grid[case_name] and not adfobj.unstructured_plotting:
                        area = mdataset.area.isel(time=0)
                        landfrac = mdataset.landfrac.isel(time=0)
                        # calculate weights
                        wgt = area * landfrac / (area * landfrac).sum()
                if comp == "atm":
                    if adfobj.native_grid[case_name] and not adfobj.unstructured_plotting:
                        wgt = mdataset.isel(time=0)[var]
            else:
                mdata = adfobj.data.load_regrid_da(case_name, var, **kwargs)
                """#Skip this variable/case if the regridded climo file doesn't exist:
                if mdata is None:
                    dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{var}`, polar lat/lon mean plotting skipped."
                    adfobj.debug_log(dmsg)
                    continue
                #Determine dimensions of variable:
                has_dims = utils.validate_dims(mdata, ["lat", "lon", "lev"])
                if (not has_dims['has_lat']) or (not has_dims['has_lon']):
                    print(f"\t    WARNING: skipping polar map for {var} for case {case_name} as it does not have both lat and lon")
                    continue
                else: # i.e., has lat&lon
                    if (has_dims['has_lev']) and (not pres_levs):
                        print(f"\t    WARNING: skipping polar map for {var} as it has more than lev dimension, but no pressure levels were provided")
                        continue"""

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

            #Skip this variable/case if the regridded climo file doesn't exist:
            if mdata is None:
                dmsg = f"\t    WARNING: No test file for {case_name} for variable `{var}`, polar lat/lon mean plotting skipped."
                print(dmsg)
                adfobj.debug_log(dmsg)
                continue
            

            has_dims = utils.lat_lon_validate_dims(odata)
            has_lat_ref, has_lev_ref = utils.zm_validate_dims(odata)
            has_lat, has_lev = utils.zm_validate_dims(mdata)


            # Process data based on dimensionality
            if "lev" in mdata.dims:
                has_lev = True
                #Check that case inputs have the correct dimensions (including "lev"):
                has_lat, has_lev = utils.zm_validate_dims(mdata)
            else:
                has_lev = False

            if not unstructured:
                if has_lev and pres_levs and plot.get('lev'):
                    if not all(dim in mdata.dims for dim in ['lat', 'lev']):
                        continue
                    """mdata = mdata.sel(lev=plot['lev'])
                    odata_level = odata.sel(lev=plot['lev'])
                    vres["lev"] = plot['lev']"""
                else:
                    if not utils.lat_lon_validate_dims(mdata):
                        continue
            else:
                print("OOOOOKKKKKKAAAAyYYY")

            if has_lev and pres_levs and plot.get('lev'):
                mdata = mdata.sel(lev=plot['lev'])
                odata_level = odata.sel(lev=plot['lev'])
                vres["lev"] = plot['lev']

            # Calculate seasonal means and differences
            use_odata = odata_level if has_lev else odata
            mseason, oseason, dseason, pseason = process_seasonal_data(
                mdata, 
                use_odata,
                plot['season']
            )

            # Create plot
            if plot['path'].exists():
                plot['path'].unlink()

            pf.make_polar_plot(plot['path'], test_nicknames[case_idx], base_nickname,
                [syear_cases[case_idx], eyear_cases[case_idx]],
                [syear_baseline, eyear_baseline],
                mseason, oseason, dseason, pseason,
                hemisphere=get_hemisphere(plot['hemi']),
                obs=adfobj.compare_obs, unstructured=unstructured,
                **vres
            )

            # Add to website
            adfobj.add_website_data(
                plot['path'], plot['var'], case_name,
                category=web_category, season=plot['season'],
                plot_type=plot['hemi']
            )

    print("  ...polar maps have been generated successfully.")

##############
#END OF `polar_map` function

##############
# END OF FILE