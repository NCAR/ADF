from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import plotting_utils as plot_utils

import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

def zonal_mean(adfobj):

    """
    Plots zonal average from climatological files (annual and seasonal).
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
    plot_var_list, climo_yrs, variable_defaults, read_config_var, 
    get_basic_info, add_website_data, debug_log

    Determines whether `lev` dimension is present. If not, makes
    a line plot, but if so it makes a contour plot.
    TODO: There's a flag to plot linear vs log pressure, but no
          method to infer what the user wants. 
    """

    #Notify user that script has started:
    msg = "\n  Generating zonal mean plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    var_list = adfobj.plot_var_list

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
    #print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    #print(f"\t NOTE: redo_plot is set to {redo_plot}")
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
    zonal_skip = []
    logp_zonal_skip = []

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
            #Website category of the log-p plots (vres, where it is set
            #later, does not exist yet in this pre-flight pass):
            web_category = res.get(var, {}).get("category", None)
            for s in seasons:
                #Check zonal log-p:
                plot_name_log = plot_loc / f"{var}_logp_{s}_Zonal_Mean.{plot_type}"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name_log.is_file():
                    logp_zonal_skip.append(plot_name_log)
                    #Continue to next iteration:
                    adfobj.add_website_data(plot_name_log, f"{var}_logp", case_name, season=s,
                                            plot_type="Zonal", category=web_category)
                    pass

                elif (redo_plot) and plot_name_log.is_file():
                    plot_name_log.unlink()
                #End if
                
                #Check regular zonal
                plot_name = plot_loc / f"{var}_{s}_Zonal_Mean.{plot_type}"
                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    zonal_skip.append(plot_name)
                    #Add already-existing plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s,
                                                        plot_type="Zonal")

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
        print(f"\t - zonal mean plots for {var}")

        # Whatever data there is gets plotted: without reference data the
        # test cases are plotted on their own (and the other way round).
        has_ref_var = var in adfobj.data.ref_var_nam
        if not has_ref_var:
            dmsg = f"\t    WARNING: No reference data found for variable `{var}`, plotting the test case(s) only."
            adfobj.debug_log(dmsg)
            print(dmsg)

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"\t    INFO: zonal_mean: Found variable defaults for {var}")

        else:
            vres = {}
        #End if

        unstruct_plotting = adfobj.unstructured_plotting
        if unstruct_plotting:
            #config["unstructured_plotting"] = unstruct_plotting
            unstructured = unstruct_plotting
            #mesh_file = '/glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc'#adfobj.mesh_file
            #kwargs["mesh_file"] = mesh_file
        else:
            unstructured = False
        #config["unstructured_plotting"] = unstructured
        vres["unstructured_plotting"] = unstructured

        vres = plot_utils.add_var_name_key_to_vres(adfobj, var, vres)
        vres["plot_type"] = __name__
        #Extract category (if available):
        web_category = vres.get("category", None)

        # load reference data (observational or baseline)
        if not adfobj.compare_obs:
            base_name = adfobj.data.ref_case_label
        else:
            base_name = adfobj.data.ref_labels.get(var, var)

        # Gather reference variable data
        odata = None
        vres.pop("wgt_base", None)
        if not has_ref_var:
            pass
        elif unstructured:
            # On the mesh, each case is compared with the reference on its own
            # mesh (a lat/lon reference is put there by the regridding step;
            # see AdfData._reference_files), so it is loaded with each case.
            pass
        else:
            odata = adfobj.data.load_reference_regrid_da(base_name, var)
            #Check if regridded file exists, if not plot the test case(s) on their own
            if odata is None:
                dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{var}`, plotting the test case(s) only."
                adfobj.debug_log(dmsg)
                print(dmsg)
            # End if
        # End if

        #Check zonal mean dimensions
        has_lat_ref = has_lev_ref = False
        if odata is not None:
            has_lat_ref, has_lev_ref = utils.zm_validate_dims(odata)

            if (not unstructured) and (not has_lat_ref):
                # check if there is a lat dimension:
                print(
                    f"\t    WARNING: Variable {var} is missing a lat dimension for '{base_name}', plotting the test case(s) only."
                )
                odata = None
            # End if
        # End if

        #Loop over model cases:
        for case_idx, case_name in enumerate(adfobj.data.case_names):

            #Set case nickname:
            case_nickname = adfobj.data.test_nicknames[case_idx]

            #Set output plot location:
            plot_loc = Path(plot_locations[case_idx])

            # load re-gridded model files:
            if unstructured:
                if has_ref_var:
                    odata, has_lat_ref, has_lev_ref = _mesh_reference(
                        adfobj, base_name, var, vres, case_name
                    )
                mesh_file = adfobj.mesh_files["test_mesh_file"][case_idx]
                vres["mesh_file"] = mesh_file
                # The data with its unit conversion, as on a lat/lon grid; the
                # dataset for the cell count:
                mdataset = adfobj.data.load_regrid_dataset(case_name, var, **vres)
                mdata = None if mdataset is None else adfobj.data.load_regrid_da(
                    case_name, var, **vres
                )
            else:
                mdata = adfobj.data.load_regrid_da(case_name, var)
            # End if

            if mdata is None:
                dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{var}`, plotting the baseline only."
                adfobj.debug_log(dmsg)
                print(dmsg)
            elif unstructured and ("wgt_base" in vres):
                unstruct_case = True
                wgt = mdataset.isel(time=0)[var]

                #Determine dimensions of variable:
                if len(wgt.n_face) == len(vres["wgt_base"].n_face):
                    vres["wgt"] = wgt
                    vres["indataset"] = mdataset
                else:
                    print("The weights are different between test and baseline. Won't continue, eh.")
                    return

                unstruct_base = vres["unstruct_base"]
                if (not unstruct_case) and (unstruct_base):
                    print("Base is unstructured but Test is lat/lon. Can't continue?")
                    return
                if (unstruct_case) and (not unstruct_base):
                    print("Base is lat/lon but Test is unstructured. Can't continue?")
                    return
                if (unstruct_case) and (unstruct_base):
                    unstructured = True
                if (not unstruct_case) and (not unstruct_base):
                    unstructured = False
                vres["unstructured_plotting"] = unstructured
            # End if

            # determine whether it's 2D or 3D
            # 3D triggers search for surface pressure
            # check data dimensions:
            case_odata = odata
            has_lev = has_lev_ref
            if mdata is not None:
                has_lat, has_lev = utils.zm_validate_dims(mdata)

                if (not unstructured) and (not has_lat):
                    # check if there is a lat dimension:
                    print(
                        f"\t    WARNING: Variable {var} is missing a lat dimension for '{case_name}', plotting the baseline only."
                    )
                    mdata = None
                    has_lev = has_lev_ref
                # End if
            # End if
            if (mdata is not None) and (case_odata is not None) and (has_lev != has_lev_ref):
                #A case and a reference that don't both have vertical levels
                #can't be differenced, so the test case is plotted on its own:
                print(f"\t    WARNING: expecting lev boolean for both case: {has_lev} and ref: {has_lev_ref}; plotting the test case only.")
                case_odata = None
            # End if
            if (mdata is None) and (case_odata is None):
                continue
            # End if

            #Notify user of level dimension:
            if has_lev:
                print(f"\t    INFO: {var} has lev dimension.")

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

                #Set the file name
                plot_name = plot_loc / f"{var}_{s}_Zonal_Mean.{plot_type}"
                plot_name_log = None

                if has_lev:
                    #Set the file name for log-pressure plots
                    plot_name_log = plot_loc / f"{var}_logp_{s}_Zonal_Mean.{plot_type}"
                #End if

                # Nothing to draw for this season: both plots were found above
                # and redo_plot is false.  The seasonal averages below are the
                # expensive part of a re-run, so work that out before doing them.
                if (plot_name in zonal_skip) and (
                    (plot_name_log is None) or (plot_name_log in logp_zonal_skip)
                ):
                    continue
                # End if

                # Seasonal Averages
                if mdata is not None:
                    mseasons[s] = utils.seasonal_mean(mdata, season=s, is_climo=True)
                if case_odata is not None:
                    oseasons[s] = utils.seasonal_mean(case_odata, season=s, is_climo=True)

                def make_plot(wks, log_p):
                    """Test vs baseline, or whichever of the two has data."""
                    if (mdata is not None) and (case_odata is not None):
                        pf.plot_zonal_mean_and_save(adfobj, wks, case_nickname, adfobj.data.ref_nickname,
                                                    [syear_cases[case_idx],eyear_cases[case_idx]],
                                                    [syear_baseline,eyear_baseline],
                                                    mseasons[s], oseasons[s], has_lev, log_p=log_p, obs=adfobj.compare_obs, **vres)
                    elif mdata is not None:
                        pf.plot_single_case_and_save(adfobj, wks, "zonal", "Test", case_nickname,
                                                     [syear_cases[case_idx],eyear_cases[case_idx]],
                                                     mseasons[s], has_lev, log_p=log_p, **vres)
                    else:
                        pf.plot_single_case_and_save(adfobj, wks, "zonal", "Baseline",
                                                     adfobj.data.ref_nickname or base_name,
                                                     [syear_baseline,eyear_baseline],
                                                     oseasons[s], has_lev, log_p=log_p, **vres)
                    #End if
                #End def

                #Create plots
                if plot_name not in zonal_skip:

                    #Create new plot:
                    make_plot(plot_name, log_p=False)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name, var, case_name, season=s, plot_type="Zonal")
                #End if

                #Create log-pressure plots as well (if applicable)
                if (plot_name_log) and (plot_name_log not in logp_zonal_skip):

                    make_plot(plot_name_log, log_p=True)

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_name_log, f"{var}_logp", case_name, season=s, plot_type="Zonal", category=web_category)
                #End if

            #End for (seasons loop)
        #End for (case names loop)
    #End for (variables loop)

    #Notify user that script has ended:
    print("  ...Zonal mean plots have been generated successfully.")


##############
#END OF SCRIPT


def _mesh_reference(adfobj, base_name, var, vres, test_case):
    """The reference for `var` on a test case's mesh, for 'unstructured_plotting'.

    Returns (odata, has_lat_ref, has_lev_ref); odata is None, with a warning,
    when there is no reference on that mesh.  Sets "wgt_base" and
    "unstruct_base" in `vres` to match (and removes "wgt_base" when there is
    none, so the case is plotted on its own).
    """
    vres.pop("wgt_base", None)
    kwargs = dict(vres, mesh_file=adfobj.mesh_files["baseline_mesh_file"])
    odataset = adfobj.data.load_reference_regrid_dataset(
        base_name, var, test_case=test_case, **kwargs
    )
    odata = None if odataset is None else adfobj.data.load_reference_regrid_da(
        base_name, var, test_case=test_case, **kwargs
    )
    if odata is None:
        # e.g. a reference on another mesh, which the regridding step skips:
        dmsg = f"\t    WARNING: No native-grid reference data for {base_name} for variable `{var}`, plotting the test case only."
        adfobj.debug_log(dmsg)
        print(dmsg)
        return None, False, False
    vres["wgt_base"] = odata
    vres["unstruct_base"] = True
    has_lat_ref, has_lev_ref = utils.zm_validate_dims(odata)
    return odata, has_lat_ref, has_lev_ref
