from pathlib import Path
import numpy as np
import plotting_functions as pf
import plotting_utils as plot_utils

import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning


def meridional_mean(adfobj):

    """
    Plots meridional average from climatological files (annual and seasonal).

    Follows the old AMWG convention of averaging 5S to 5N.  **Note:** that
    constraint is easily changed; the function that calculates the average
    takes any range of latitudes.  Compares CAM climatologies against other
    climatological data (observations or baseline runs).

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
    plot_var_list, plot_location, climo_yrs, variable_defaults,
    read_config_var, get_basic_info, add_website_data, debug_log,
    compare_obs

    Determines whether `lev` dimension is present. If not, makes a line plot;
    if so, makes a contour plot, plus a second one on a log-pressure axis.

    The regular plots are named the same way whether or not the variable has
    a `lev` dimension, but the log-pressure plots exist only for variables
    that do.  So a variable is settled from the file names alone (see the
    pre-flight pass below) only when its log-pressure plots are present too;
    otherwise the data is opened to find out, and each season that is
    already drawn is still skipped.
    """

    #Notify user that script has started:
    msg = "\n  Generating meridional mean plots..."
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

    # Check if plots already exist and redo_plot boolean
    # If redo_plot is false and file exists, keep track and skip the calculation
    # entirely, which is what makes re-running the ADF fast.  A set, because
    # every season of every variable is looked up in it once per case.
    merid_skip = set()

    # Loop over model cases:
    for case_idx, case_name in enumerate(adfobj.data.case_names):
        # Set output plot location:
        plot_loc = Path(plot_locations[case_idx])

        # Check if plot output directory exists, and if not, then create it:
        if not plot_loc.is_dir():
            print(f"    {plot_loc} not found, making new directory")
            plot_loc.mkdir(parents=True)
        # End if

        # Loop over the variables for each season
        for var in var_list:
            web_category = res.get(var, {}).get("category", None)
            for s in seasons:
                # Check meridional log-p (only made for variables with 'lev'):
                plot_name_log = plot_loc / f"{var}_logp_{s}_Meridional_Mean.{plot_type}"
                if (not redo_plot) and plot_name_log.is_file():
                    merid_skip.add(plot_name_log)
                    adfobj.add_website_data(
                        plot_name_log, f"{var}_logp", case_name, season=s,
                        plot_type="Meridional", category=web_category,
                    )
                elif (redo_plot) and plot_name_log.is_file():
                    plot_name_log.unlink()
                # End if

                plot_name = plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    merid_skip.add(plot_name)
                    # Add already-existing plot to website (if enabled):
                    adfobj.add_website_data(
                        plot_name, var, case_name, season=s, plot_type="Meridional"
                    )
                    continue
                elif (redo_plot) and plot_name.is_file():
                    plot_name.unlink()
                # End if
            # End for (seasons)
        # End for (variables)
    # End for (cases)
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

        # Nothing to draw for this variable at all: settle it from the file
        # names, before opening any data
        if _all_plots_present(
            plot_locations, adfobj.data.case_names, var, seasons, plot_type, merid_skip
        ):
            continue
        # End if

        if var not in adfobj.data.ref_var_nam:
            dmsg = f"\t    WARNING: No reference data found for variable `{var}`, meridional mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            continue
        #End if

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            # Copy, as season/plot_type/var_name get added below, and the
            # variable defaults are shared with every other script:
            vres = dict(res[var])
            # If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(
                f"\t    INFO: meridional_mean: Found variable defaults for {var}"
            )
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
        # End if

        # Gather reference variable data
        odata = adfobj.data.load_reference_regrid_da(base_name, var)

        # Check if regridded file exists, if not skip meridional plot for this var
        if odata is None:
            dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{var}`, meridional mean plotting skipped."
            adfobj.debug_log(dmsg)
            continue
        # End if

        # Check meridional mean dimensions
        has_dims_ref = utils.validate_dims(odata, ["lat", "lev"])

        # check if there is a lat dimension:
        # if not, skip test cases and move to next variable
        if not has_dims_ref["has_lat"]:
            print(
                f"\t    WARNING: Variable {var} is missing a lat dimension for '{base_name}', cannot continue to plot."
            )
            continue
        # End if

        # Loop over model cases:
        for case_idx, case_name in enumerate(adfobj.data.case_names):

            # Set case nickname:
            case_nickname = adfobj.data.test_nicknames[case_idx]

            # Set output plot location:
            plot_loc = Path(plot_locations[case_idx])

            # Nothing to draw for this case: settle it before opening the file
            # (only possible when the log-p plots are there too, see above)
            if all(
                (plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}") in merid_skip
                and (plot_loc / f"{var}_logp_{s}_Meridional_Mean.{plot_type}")
                in merid_skip
                for s in seasons
            ):
                continue
            #End if

            # load re-gridded model files:
            mdata = adfobj.data.load_regrid_da(case_name, var)

            if mdata is None:
                dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{var}`, meridional mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue
            # End if

            # determine whether it's 2D or 3D
            # 3D triggers search for surface pressure
            has_dims = utils.validate_dims(mdata, ["lat", "lev"])

            # check if there is a lat dimension:
            if not has_dims["has_lat"]:
                print(
                    f"\t    WARNING: Variable {var} is missing a lat dimension for '{case_name}', cannot continue to plot."
                )
                continue
            # End if

            has_lev = has_dims["has_lev"]

            # Notify user of level dimension:
            if has_lev:
                print(f"\t    INFO: {var} has lev dimension.")
            # End if

            # Check to make sure each case has vertical levels if one of the cases does
            if has_lev != has_dims_ref["has_lev"]:
                print(
                    f"\t    WARNING: expecting lev boolean for both case: {has_lev} and ref: {has_dims_ref['has_lev']}"
                )
                continue
            # End if

            #
            # Seasonal Averages
            # Note: xarray can do seasonal averaging, but depends on having time accessor,
            # which these prototype climo files don't.
            #

            # Create new dictionaries:
            mseasons = {}
            oseasons = {}

            # Loop over season dictionary:
            for s in seasons:
                vres["season"] = s

                # Set the file names
                plot_name = plot_loc / f"{var}_{s}_Meridional_Mean.{plot_type}"
                plot_name_log = None
                if has_lev:
                    plot_name_log = (
                        plot_loc / f"{var}_logp_{s}_Meridional_Mean.{plot_type}"
                    )
                # End if

                # Found above and redo_plot is false, so there is nothing to
                # draw.  The seasonal averages below are the expensive part of
                # a re-run, so work that out before doing them.
                if (plot_name in merid_skip) and (
                    (plot_name_log is None) or (plot_name_log in merid_skip)
                ):
                    continue
                # End if

                # Seasonal Averages
                mseasons[s] = utils.seasonal_mean(mdata, season=s, is_climo=True)
                oseasons[s] = utils.seasonal_mean(odata, season=s, is_climo=True)

                # Create new plot:
                if plot_name not in merid_skip:
                    pf.plot_meridional_mean_and_save(
                        adfobj,
                        plot_name,
                        case_nickname,
                        adfobj.data.ref_nickname,
                        [syear_cases[case_idx], eyear_cases[case_idx]],
                        [syear_baseline, eyear_baseline],
                        mseasons[s],
                        oseasons[s],
                        has_lev,
                        log_p=False,
                        latbounds=slice(-5, 5),
                        obs=adfobj.compare_obs,
                        **vres,
                    )

                    # Add plot to website (if enabled):
                    adfobj.add_website_data(
                        plot_name, var, case_name, season=s, plot_type="Meridional"
                    )
                # End if

                # Create log-pressure plot as well (if applicable):
                if (plot_name_log) and (plot_name_log not in merid_skip):
                    pf.plot_meridional_mean_and_save(
                        adfobj,
                        plot_name_log,
                        case_nickname,
                        adfobj.data.ref_nickname,
                        [syear_cases[case_idx], eyear_cases[case_idx]],
                        [syear_baseline, eyear_baseline],
                        mseasons[s],
                        oseasons[s],
                        has_lev,
                        log_p=True,
                        latbounds=slice(-5, 5),
                        obs=adfobj.compare_obs,
                        **vres,
                    )

                    # Add plot to website (if enabled):
                    adfobj.add_website_data(
                        plot_name_log, f"{var}_logp", case_name, season=s,
                        plot_type="Meridional", category=web_category,
                    )
                # End if

            # End for (seasons loop)
        # End for (case names loop)
    # End for (variables loop)

    #Notify user that script has ended:
    print("  ...Meridional mean plots have been generated successfully.")


#########
# Helpers
#########


def _all_plots_present(plot_locations, case_names, var, seasons, plot_type, merid_skip):
    """
    Report whether every plot this variable would produce is already there.

    Parameters
    ----------
    plot_locations : list
        output plot directory for each case
    case_names : list
        names of the test cases
    var : str
        ADF name of the variable being plotted
    seasons : dict
        the seasons being plotted, keyed by name
    plot_type : str
        file extension the plots are written with, e.g. "png"
    merid_skip : set
        plots found by the pre-flight pass above

    Returns
    -------
    bool
        ``True`` when nothing is left to draw for this variable, so the
        reference and test data never have to be opened.  The log-pressure
        plots count too: a variable without them may simply have no `lev`
        dimension, which only opening the data can tell.
    """
    for case_idx in range(len(case_names)):
        plot_loc = Path(plot_locations[case_idx])
        for s in seasons:
            for name in (f"{var}_{s}_Meridional_Mean", f"{var}_logp_{s}_Meridional_Mean"):
                if (plot_loc / f"{name}.{plot_type}") not in merid_skip:
                    return False
                # End if
            # End for
            # End if
        # End for
    # End for
    return True


##############
# END OF FILE
