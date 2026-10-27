"""
Generate global maps of vector fields (e.g. U/V wind, TAUX/TAUY stress)

Functions
---------
global_latlon_vect_map(adfobj)
    use ADF object to make vector maps
get_plot_config(adfobj)
    plotting settings shared by every vector pair
process_vector_pair(adfobj, var, ...)
    load and plot one vector pair for every test case
"""
# Import standard modules:
from pathlib import Path
import numpy as np

# Import local modules:
import adf_utils as utils
import plotting_utils as plot_utils
import plotting_functions as pf

# Warnings
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

#########

def global_latlon_vect_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model vector fields with continental
    overlays.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    Does not return a value; produces plots and saves files.

    Notes
    -----
    A variable is plotted as a vector when its variable-defaults entry has a
    ``vector_pair`` (the other component) and a ``vector_name`` (the name in
    the plot file names).  Each pair is plotted once.

    It uses the AdfDiag object's methods to get necessary information.
    Makes use of AdfDiag's data sub-class.
    Explicitly accesses:
    adfobj.plot_var_list
        List of variables
    adfobj.plot_location
        output plot path
    adfobj.climo_yrs
        start and end climo years of the case(s), `syears` & `eyears`
        start and end climo years of the reference, `syear_baseline` & `eyear_baseline`
    adfobj.variable_defaults
        dict of variable-specific plot preferences
    adfobj.var_obs_dict
        observation file, name and variable of each variable
    adfobj.read_config_var
        dict of basic info, `diag_basic_info`
        Then use to check `plot_type`
    adfobj.unstructured_plotting, adfobj.mesh_files
        plotting on the native mesh, and its mesh files
    adfobj.debug_log
        Issues debug message
    adfobj.add_website_data
        Communicates information to the website generator
    adfobj.compare_obs
        Logical to determine if comparing to observations

    The helper modules are needed for:
    plot_utils.get_central_longitude()
        determine central longitude for global plots
    utils.validate_dims()
        checks the lat, lon and lev dimensions
    utils.cell_weights()
        native-grid cell weights
    utils.seasonal_mean()
        calculate seasonal mean
    pf.plot_map_and_save()
        send information to make the plot and save the file
    """

    msg = "\n  Generating lat/lon vector maps..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    if adfobj.compare_obs and not adfobj.var_obs_dict:
        # No observations to compare against:
        print("\t No observations found to plot against, so no vector maps will be generated.")
        return

    # Get configuration
    config = get_plot_config(adfobj)
    adfobj.debug_log(f"Vector plot type is set to {config['plot_type']}")

    # Each pair is plotted once, under its first component:
    skip_vars = set()
    for var in adfobj.plot_var_list:
        if var in skip_vars:
            continue
        pair = process_vector_pair(adfobj, var, **config)
        if pair:
            skip_vars.update(pair)

    print("  ...lat/lon vector maps have been generated successfully.")


def get_plot_config(adfobj):
    """Get plotting configuration from ADF object."""
    return {
        'seasons': {
            "ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]
        },
        'plot_type': adfobj.read_config_var("diag_basic_info").get('plot_type', 'png'),
        'redo_plot': adfobj.get_basic_info('redo_plot'),
        # Empty when the config sets none (land configs, for one):
        'pres_levs': adfobj.get_basic_info("plot_press_levels") or [],
        'unstruct_plotting': bool(adfobj.unstructured_plotting),
    }


def process_vector_pair(adfobj, var, seasons, pres_levs, plot_type, redo_plot,
                        unstruct_plotting):
    """Plot the vector pair `var` belongs to, for every test case.

    Returns
    -------
    tuple or None
        ``(var, var_pair)`` once `var` is known to be the first component of
        a vector pair (whether or not it could be plotted), else None.
    """
    vres = adfobj.variable_defaults.get(var)
    if vres is None or "vector_pair" not in vres:
        adfobj.debug_log(f"variable '{var}' not a vector pair")
        return None
    vres["unstructured_plotting"] = unstruct_plotting
    vres = plot_utils.add_var_name_key_to_vres(adfobj, var, vres, vector=True)
    adfobj.debug_log(f"global_latlon_vect_map: Found variable defaults for {var}")
    vres["plot_type"] = __name__
    web_category = vres.get("category", None)
    var_pair = vres["vector_pair"]
    var_name = vres["vector_name"]

    #Notify user of variable being plotted:
    print(f"\t - lat/lon vector maps for {var},{var_pair}")

    # For global maps, also set the central longitude:
    # can be specified in adfobj basic info as 'central_longitude' or supplied as a number,
    # otherwise defaults to 180
    vres['central_longitude'] = plot_utils.get_central_longitude(adfobj)

    case_names = adfobj.data.case_names
    plot_locations = adfobj.plot_location

    # A complete set of plots can be recognised from the file names alone,
    # because a 2-D vector and a 3-D one are named differently.  Doing that
    # first means a re-run never opens the four regridded files a vector
    # pair needs just to find out which of the two it is looking at.
    settled = _existing_plot_set(
        plot_locations, case_names, var_name, seasons, pres_levs, plot_type
    )
    if settled and not redo_plot:
        for path, web_name, case_name, season in settled:
            adfobj.debug_log(f"'{path}' exists and clobber is false.")
            adfobj.add_website_data(
                path,
                web_name,
                case_name,
                category=web_category,
                season=season,
                plot_type="LatLon_Vector",
            )
        print(
            f"\t    INFO: All plots exist for {var_name}. "
            f"Redo is {redo_plot}. Existing plots added to website data."
        )
        return var, var_pair

    names = _reference_names(adfobj, var, var_pair)
    if names is None:
        return var, var_pair
    base_name, pair_base_name, data_var, data_src = names

    kwargs = {"unstructured_plotting": True} if unstruct_plotting else {}

    if not unstruct_plotting:
        ref = _load_reference(adfobj, base_name, pair_base_name, var,
                              var_pair, data_var, kwargs, False)
        if ref is None:
            return var, var_pair

    #Loop over model cases:
    for case_idx, case_name in enumerate(case_names):
        if unstruct_plotting:
            # On the mesh, each case is compared with the reference on its
            # own mesh (a lat/lon reference is put there by the regridding
            # step; see AdfData._reference_files):
            ref = _load_reference(adfobj, base_name, pair_base_name, var,
                                  var_pair, data_var, kwargs, True,
                                  test_case=case_name)
            if ref is None:
                continue
        uodata, vodata, wgt_base, _ = ref

        case = _load_case(adfobj, case_idx, case_name, var, var_pair, data_var,
                          kwargs, unstruct_plotting, pres_levs, vres, wgt_base)
        if case is None:
            continue
        umdata, vmdata, m_has_dims = case

        plot_loc = Path(plot_locations[case_idx])
        plot_loc.mkdir(parents=True, exist_ok=True)

        process_plots(adfobj, umdata, vmdata, uodata, vodata, case_name, case_idx,
                      var_name, seasons, pres_levs if m_has_dims["has_lev"] else None,
                      plot_loc, plot_type, redo_plot, vres, web_category, data_src)

    return var, var_pair


def _reference_names(adfobj, var, var_pair):
    """The reference labels and variable names for a vector pair, or None.

    Returns ``(base_name, pair_base_name, data_var, data_src)``: the reference
    label each component is found under, the reference's names of the two
    components, and the name of the reference data set (the plots' title).
    """
    if not adfobj.compare_obs:
        # reference (baseline) name
        base_name = adfobj.data.ref_case_label
        data_src = adfobj.get_baseline_info("cam_case_name", required=True)
        return base_name, base_name, [var, var_pair], data_src

    var_obs_dict = adfobj.var_obs_dict
    if (
        var not in adfobj.data.ref_var_nam
        or var not in var_obs_dict
        or var_pair not in var_obs_dict
    ):
        dmsg = f"\t    WARNING: No reference data found for variable `{var}`, lat/lon vector map skipped."
        adfobj.debug_log(dmsg)
        print(dmsg)
        return None
    #The loaders find each component's obs file by its ADF
    #name and its own reference label:
    return (
        adfobj.data.ref_labels[var],
        adfobj.data.ref_labels[var_pair],
        [var_obs_dict[var]["obs_var"], var_obs_dict[var_pair]["obs_var"]],
        var_obs_dict[var]["obs_name"],
    )


def _load_reference(adfobj, base_name, pair_base_name, var, var_pair, data_var,
                    kwargs, unstruct_plotting, test_case=None):
    """Load the reference U and V components, or return None (with a warning).

    Returns (uodata, vodata, wgt_base, o_has_dims); wgt_base is the cell
    weights on the mesh, None on a lat/lon grid.  With `unstruct_plotting`,
    `test_case` asks for the reference on that case's mesh.
    """
    kwargs = dict(kwargs)
    wgt_base = None
    if unstruct_plotting:
        kwargs["mesh_file"] = adfobj.mesh_files["baseline_mesh_file"]
        kwargs["test_case"] = test_case
        uodata = adfobj.data.load_reference_regrid_da(base_name, var, **kwargs)
        vodata = adfobj.data.load_reference_regrid_da(pair_base_name, var_pair, **kwargs)
        odataset = adfobj.data.load_reference_regrid_dataset(base_name, var, **kwargs)
        if (odataset is None) or (uodata is None) or (vodata is None):
            # e.g. a reference on another mesh, which the regridding step skips:
            dmsg = f"\t    WARNING: No native-grid reference data for {base_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            return None
        o_has_dims = utils.validate_dims(uodata, ["lev"])
        # Cell weights (area; area * landfrac for land), as global_latlon_map:
        wgt_base = utils.cell_weights(odataset, adfobj.model_component)
    else:
        uodata = adfobj.data.load_reference_regrid_da(base_name, var, **kwargs)
        vodata = adfobj.data.load_reference_regrid_da(pair_base_name, var_pair, **kwargs)
        if (uodata is None) or (vodata is None):
            dmsg = f"\t    WARNING: No regridded baseline file for {base_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            return None
        o_has_dims = utils.validate_dims(uodata, ["lat", "lon", "lev"])
        if (not o_has_dims['has_lat']) or (not o_has_dims['has_lon']):
            print(f"\t    WARNING: skipping global map for {var} as REFERENCE does not have both lat and lon")
            return None
    return uodata, vodata, wgt_base, o_has_dims


def _load_case(adfobj, case_idx, case_name, var, var_pair, data_var, kwargs,
               unstruct_plotting, pres_levs, vres, wgt_base):
    """Load a test case's U and V components, or return None (with a warning).

    Returns (umdata, vmdata, m_has_dims).  On the mesh, the case's cell
    weights go into `vres` ("wgt"), after checking they are on the same cells
    as the reference's.
    """
    if unstruct_plotting:
        mesh_file = adfobj.mesh_files["test_mesh_file"][case_idx]
        kwargs = dict(kwargs, mesh_file=mesh_file)
        vres["mesh_file"] = mesh_file
        umdata = adfobj.data.load_regrid_da(case_name, var, **kwargs)
        vmdata = adfobj.data.load_regrid_da(case_name, var_pair, **kwargs)
        mdataset = adfobj.data.load_regrid_dataset(case_name, var, **kwargs)
        if (mdataset is None) or (umdata is None) or (vmdata is None):
            dmsg = f"\t    WARNING: No native-grid regridded test file for {case_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            return None
        m_has_dims = utils.validate_dims(umdata, ["lev"])
        # Cell weights (area; area * landfrac for land), as global_latlon_map:
        wgt = utils.cell_weights(mdataset, adfobj.model_component)
        if len(wgt.n_face) != len(wgt_base.n_face):
            print(
                f"\t    WARNING: '{case_name}' and the reference are not on the same "
                f"cells, so the vector map for {var} is skipped."
            )
            return None
        vres["wgt"] = wgt
        vres["indataset"] = mdataset
    else:
        umdata = adfobj.data.load_regrid_da(case_name, var)
        vmdata = adfobj.data.load_regrid_da(case_name, var_pair)
        #Skip this variable/case if the regridded climo file doesn't exist:
        if (umdata is None) or (vmdata is None):
            dmsg = f"\t    WARNING: No regridded test file for {case_name} for variable `{data_var[0]}`/`{data_var[1]}`, global lat/lon vect plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)
            return None
        m_has_dims = utils.validate_dims(umdata, ["lat", "lon", "lev"])
        if (not m_has_dims['has_lat']) or (not m_has_dims['has_lon']):
            print(f"\t    WARNING: skipping global map for {var} for case {case_name} as it does not have both lat and lon")
            return None

    # Check pressure levels if 3D data
    if m_has_dims['has_lev'] and not pres_levs:
        print(f"\t    WARNING: skipping global map for {var} as it has more than lev dimension, but no pressure levels were provided")
        return None
    return umdata, vmdata, m_has_dims


def process_plots(adfobj, umdata, vmdata, uodata, vodata, case_name, case_idx,
                  var_name, seasons, pres_levs, plot_loc, plot_type, redo_plot,
                  vres, web_category, data_src):
    """Make the vector maps of one case: every season, and every level of a 3-D pair.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object containing configuration
    umdata, vmdata : xarray.DataArray
        Test case U and V components
    uodata, vodata : xarray.DataArray
        Reference U and V components
    case_name : str
        Name of current case
    case_idx : int
        Index of current case
    var_name : str
        The vector's name, used in plot file names
    seasons : dict
        Dictionary of season definitions
    pres_levs : list or None
        Pressure levels to plot a 3-D pair at; None for a 2-D pair
    plot_loc : Path
        Output plot directory
    plot_type : str
        Plot file type (e.g. 'png')
    redo_plot : bool
        Whether to regenerate existing plots
    vres : dict
        Variable-specific plot settings
    web_category : str
        Category for website organization
    data_src : str
        Name of the reference data set
    """
    case_nickname = adfobj.case_nicknames["test_nicknames"][case_idx]
    base_nickname = adfobj.case_nicknames["base_nickname"]
    case_yrs = [adfobj.climo_yrs["syears"][case_idx], adfobj.climo_yrs["eyears"][case_idx]]
    base_yrs = [adfobj.climo_yrs["syear_baseline"], adfobj.climo_yrs["eyear_baseline"]]

    for lev in (pres_levs if pres_levs is not None else [None]):
        if lev is None:
            web_name = var_name
        else:
            #Check that the user-requested pressure level
            #exists in the model data, which should already
            #have been interpolated to the standard reference
            #pressure levels:
            if (lev not in umdata['lev']) or (lev not in uodata['lev']):
                print(f"\t    WARNING: plot_press_levels value '{lev}' not present in {var_name} [test: {(lev in umdata['lev'])}, ref: {lev in uodata['lev']}], so skipping.")
                continue
            vres['lev'] = int(lev)
            web_name = f"{var_name}_{lev}hpa"

        for s in seasons:
            plot_name = plot_loc / f"{web_name}_{s}_LatLon_Vector_Mean.{plot_type}"

            # Check redo_plot. If set to True: remove old plot, if it already exists:
            if (not redo_plot) and plot_name.is_file():
                #Add already-existing plot to website (if enabled):
                adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
                adfobj.add_website_data(plot_name, web_name, case_name, category=web_category,
                                        season=s, plot_type="LatLon_Vector")
                continue
            if redo_plot and plot_name.is_file():
                plot_name.unlink()

            # Seasonal averages, on the level for a 3-D pair:
            umseason, vmseason, uoseason, voseason = (
                utils.seasonal_mean(fld, season=s, is_climo=True)
                for fld in (umdata, vmdata, uodata, vodata)
            )
            if lev is not None:
                umseason, vmseason, uoseason, voseason = (
                    fld.sel(lev=lev) for fld in (umseason, vmseason, uoseason, voseason)
                )
            # differences: each entry should be (lat, lon)
            udseason = umseason - uoseason
            vdseason = vmseason - voseason
            # percent difference of the zonal component:
            upseason = (umseason - uoseason) / np.abs(uoseason) * 100.0
            upseason = upseason.where(np.isfinite(upseason), np.nan)

            vres["season"] = s
            # pass in casenames
            vres["case_name"] = case_name
            vres["baseline"] = data_src

            #Create new plot:
            # NOTE: send vres as kwarg dictionary.  --> ONLY vres, not the full res
            # This relies on `plot_map_and_save` knowing how to deal with the options
            # currently knows how to handle:
            #   colormap, contour_levels, diff_colormap, diff_contour_levels, tiString, tiFontSize, mpl
            #   *Any other entries will be ignored.
            pf.plot_map_and_save(adfobj, plot_name, case_nickname, base_nickname,
                                 case_yrs, base_yrs,
                                 umseason, uoseason, udseason, upseason,
                                 obs=adfobj.compare_obs,
                                 vmdlfld=vmseason, vobsfld=voseason,
                                 vdiffld=vdseason, **vres)

            #Add plot to website (if enabled):
            adfobj.add_website_data(plot_name, web_name, case_name, category=web_category,
                                    season=s, plot_type="LatLon_Vector")


def _existing_plot_set(
    plot_locations, case_names, var_name, seasons, pres_levs, plot_type
):
    """
    Return the complete set of vector plots for `var_name`, if one is on disk.

    A 2-D vector is plotted as ``{name}_{season}_LatLon_Vector_Mean`` and a
    3-D one as ``{name}_{pressure}hpa_{season}_LatLon_Vector_Mean``, so which
    of the two a vector pair is can be read off the file names.

    Parameters
    ----------
    plot_locations : list
        output plot directory for each case
    case_names : list
        names of the test cases
    var_name : str
        the vector's name from the variable defaults, e.g. "Wind"
    seasons : list or dict
        the seasons being plotted
    pres_levs : list
        the configured pressure levels, empty when none are set
    plot_type : str
        file extension the plots are written with, e.g. "png"

    Returns
    -------
    list
        ``(path, website name, case, season)`` for every plot, when a
        complete set exists; an empty list when it does not.
    """
    flat = []
    levelled = []
    for case_idx, case_name in enumerate(case_names):
        plot_loc = Path(plot_locations[case_idx])
        for season in seasons:
            flat.append(
                (
                    plot_loc / f"{var_name}_{season}_LatLon_Vector_Mean.{plot_type}",
                    var_name,
                    case_name,
                    season,
                )
            )
            for lev in pres_levs:
                levelled.append(
                    (
                        plot_loc
                        / (
                            f"{var_name}_{lev}hpa_{season}"
                            f"_LatLon_Vector_Mean.{plot_type}"
                        ),
                        f"{var_name}_{lev}hpa",
                        case_name,
                        season,
                    )
                )
            # End for
        # End for
    # End for
    return plot_utils.first_complete_plot_set([levelled, flat])


##############
#END OF SCRIPT
