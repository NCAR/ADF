"""
Regional (land) climatology plots for the LDF web site.

Also holds the helpers shared with regional_timeseries: reading the regions
file, loading a variable with its land weights, and subsetting to a region.
"""
from pathlib import Path
import warnings  # use to warn user about missing files.

import numpy as np
import yaml
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import adf_utils as utils

warnings.formatwarning = utils.my_formatwarning

# Variables plotted when 'regional_var_list' is not in the config file:
DEFAULT_REGIONAL_VARS = [
    "TBOT", "PREC", "ELAI",
    "FSDS", "FLDS", "WIND", "ASA",
    "QRUNOFF_TO_COUPLER", "FSH", "ET", "FCEV",
    "SNOWDP", "GPP", "FSNO", "QBOT",
]

# The 4x4 figure uses panel 0 for the region map, leaving 15 panels:
MAX_REGIONAL_VARS = 15

# Map extents for the named regions; others are chosen from their bounds:
REGION_MAP_EXTENTS = {
    "N Hemisphere Land": [-180, 179, -3, 90],
    "Global": [-180, 179, -89, 90],
    "S Hemisphere Land": [-180, 179, -89, 3],
    "Polar": [-180, 179, 45, 90],
    "CONUS": [-140, -55, 10, 70],
    "S America": [-100, -20, -45, 20],
    "East Asia": [70, 145, 5, 55],
    "Eurasia": [20, 130, 10, 80],
}


def regional_climatology(adfobj):
    """
    Load climo files, subset for each region and each variable, and make a
    combined plot of the land-weighted monthly climatologies (baseline, test
    case and, where available, observations) for each region.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information.

    Notes
    -----
    Config entries: 'region_list' (regions to plot), 'regional_var_list'
    (variables, at most 15), and 'regions_file' in 'diag_basic_info' (the
    region definitions, by default lib/regions_lnd.yaml).

    Uses ``adfobj.data`` (``load_climo_dataset``, ``load_reference_climo_dataset``,
    ``apply_conversion``, ``ref_case_label``, ``ref_nickname``,
    ``test_nicknames``), ``adfobj.region_list``, ``adfobj.variable_defaults``,
    ``adfobj.unstructured_plotting``, ``adfobj.mesh_files``,
    ``adfobj.plot_location``, ``adfobj.compare_obs``, and
    ``utils.normalize_weights``.

    TODO (from Meg)
    - increase number of variables that have a climo plotted
    - make sure that climo's are being plotted with the preferred units
    - only the first test case is plotted
    """

    #Notify user that script has started:
    print("\n  --- Generating regional climatology plots... ---")

    setup = regional_setup(adfobj)
    if setup is None:
        return
    case_name = setup["case_name"]
    kwargs = setup["kwargs"]

    obs_info = {}
    base_data = {}
    case_data = {}
    for field in setup["var_list"]:
        # Load the global climatology (and land weights) for this variable
        base_data[field] = load_with_weights(
            adfobj, adfobj.data.load_reference_climo_dataset(
                adfobj.data.ref_case_label, field, **kwargs),
            field, adfobj.data.ref_case_label)
        case_data[field] = load_with_weights(
            adfobj, adfobj.data.load_climo_dataset(case_name, field, **kwargs),
            field, case_name)
        if base_data[field] is None or case_data[field] is None:
            print(f"\t    WARNING: Missing climo file for {field}, skipping it.")
            continue
        # End if

        # Now, check if observations are to be plotted for this variable
        obs = load_regional_obs(adfobj, field)
        if obs is not None:
            obs_info[field] = obs
        # End if
    # End for

    def plot_series(ax, field, region):
        """Plot the regional climatologies of `field` on `ax`."""
        base_var, base_wgt = subset_region(base_data[field], region, kwargs)
        case_var, case_wgt = subset_region(case_data[field], region, kwargs)
        base_var_wgtd = region_sum(base_var, base_wgt)
        case_var_wgtd = region_sum(case_var, case_wgt)
        months = np.arange(12) + 1
        ax.plot(months, base_var_wgtd, label=setup["base_nickname"], linewidth=2)
        ax.plot(months, case_var_wgtd, label=setup["case_nickname"], linewidth=2)
        if field in obs_info:
            obs_ds, obs_name = obs_info[field]
            obs_var, obs_wgt = subset_region(obs_ds, region, {})
            ax.plot(months, region_sum(obs_var, obs_wgt),
                    label=obs_name, color="black", linewidth=2)
            ax.legend()
        # End if
        ax.set_xticks(np.arange(1, 13, 2))
        return base_var
    # End def

    make_regional_plots(adfobj, setup, base_data, plot_series,
                        plot_kind="RegionalClimo", title="climatology")

    print("\n  --- Regional climatology plots generated successfully! ---")


#########
# Helpers shared with regional_timeseries
#########

def regional_setup(adfobj):
    """Settings shared by the regional plots.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object.

    Returns
    -------
    dict or None
        The plot location and type, redo_plot, the regions, the variables, the
        (first) test case, nicknames and loader keyword arguments; None when
        the plots cannot be made (no baseline case or no regions).
    """
    if adfobj.compare_obs:
        print("\t    NOTE: Regional plots need a baseline case, so they are skipped "
              "when comparing to observations.")
        return None
    # End if

    region_list = adfobj.region_list
    if not region_list:
        print("\t    NOTE: No 'region_list' in the config file, so no regional plots.")
        return None
    # End if

    # Read regions from yml file (by default, the land regions in lib):
    regions_file = adfobj.get_basic_info("regions_file")
    if not regions_file:
        regions_file = Path(__file__).resolve().parents[2] / "lib" / "regions_lnd.yaml"
    # End if
    with open(regions_file, encoding="UTF-8") as rfil:
        regions = yaml.safe_load(rfil)
    # End with

    # Variables to plot are set by "regional_var_list" in the config file;
    # use these defaults if it is not specified:
    var_list = adfobj.read_config_var("regional_var_list")
    if not var_list:
        print("\t NOTE: 'regional_var_list' not found in config file, using default variables")
        var_list = DEFAULT_REGIONAL_VARS
    # End if
    if len(var_list) > MAX_REGIONAL_VARS:
        warnings.warn(f"\t WARNING: regional_var_list has {len(var_list)} "
                      f"variables; only the first {MAX_REGIONAL_VARS} will be plotted.")
        var_list = var_list[:MAX_REGIONAL_VARS]
    # End if

    # Native-grid data plotted natively is read with uxarray:
    kwargs = {}
    if adfobj.unstructured_plotting:
        kwargs = {"unstructured_plotting": True,
                  "mesh_file": adfobj.mesh_files["baseline_mesh_file"]}
    # End if

    # TODO hard wired for single case name:
    case_name = adfobj.data.case_names[0]
    case_nickname = adfobj.data.test_nicknames[0] or case_name
    base_nickname = adfobj.data.ref_nickname or adfobj.data.ref_case_label

    redo_plot = adfobj.get_basic_info("redo_plot")
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    return {
        "plot_loc": Path(adfobj.plot_location[0]),
        "plot_type": adfobj.get_basic_info("plot_type") or "png",
        "redo_plot": redo_plot,
        "region_list": region_list,
        "regions": regions,
        "var_list": var_list,
        "case_name": case_name,
        "case_nickname": case_nickname,
        "base_nickname": base_nickname,
        "kwargs": kwargs,
    }


def load_with_weights(adfobj, dset, field, case):
    """The variable, converted to its plotting units, and its land weights.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object.
    dset : xarray.Dataset, uxarray.UxDataset or None
        A climo or time series dataset holding `field`, 'area' and 'landfrac'.
    field : str
        The variable.
    case : str
        The case the data belongs to (for the unit conversion).

    Returns
    -------
    xarray.Dataset or None
        `field`, plus the unnormalized weights 'area * landfrac' as "weight";
        None if `dset` is missing any of them.
    """
    if dset is None or field not in dset:
        return None
    if ("area" not in dset) or ("landfrac" not in dset):
        print(f"\t    WARNING: No 'area' or 'landfrac' with {field} for '{case}', "
              "so it can't be averaged over land.")
        return None
    # End if
    weight = dset["area"] * dset["landfrac"].fillna(0)
    if "time" in weight.dims:
        # drop time dimension to avoid confusion
        weight = weight.isel(time=0)
    # End if
    out = dset[[field]]
    out[field] = adfobj.data.apply_conversion(dset[field], field, case)
    out["weight"] = weight
    return out


def load_regional_obs(adfobj, field):
    """Observations for `field` from the variable defaults' 'obs_file', if any.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object.
    field : str
        The variable.

    Returns
    -------
    tuple or None
        The observations dataset and its name; None when there are none.
    """
    default_var_dict = adfobj.variable_defaults.get(field, {})
    if "obs_file" not in default_var_dict:
        #No observation file was specified, so print to log and skip variable:
        adfobj.debug_log(f"No observations file was listed for variable '{field}'.")
        return None
    # End if

    #Check if file exists, if not, then check if it is in "obs_data_loc":
    obs_file_path = Path(default_var_dict["obs_file"])
    obs_data_loc = adfobj.get_basic_info("obs_data_loc")
    if not obs_file_path.is_file() and obs_data_loc:
        obs_file_path = Path(obs_data_loc) / obs_file_path
    # End if
    if not obs_file_path.is_file():
        msg = f"Unable to find obs file '{default_var_dict['obs_file']}' "
        msg += f"for variable '{field}'."
        adfobj.debug_log(msg)
        return None
    # End if

    #Observations dataset and variable names (default to file and model names):
    obs_name = default_var_dict.get("obs_name", obs_file_path.name)
    obs_var_name = default_var_dict.get("obs_var_name", field)

    obs_ds = xr.open_mfdataset([str(obs_file_path)], combine="by_coords")
    # Special handling for some variables:, NOT A GOOD HACK!
    # TODO: improve this!
    if (field == "ASA") and ("BRDALB" in obs_ds.variables):
        obs_ds["BRDALB"] = obs_ds["BRDALB"].swap_dims({"lsmlat": "lat", "lsmlon": "lon"})
    # End if
    if obs_var_name not in obs_ds:
        adfobj.debug_log(f"No '{obs_var_name}' in obs file for variable '{field}'.")
        return None
    # End if

    obs_weight = _obs_weight(obs_ds)
    if obs_weight is None:
        adfobj.debug_log(f"No valid weight, area, or landmask in obs file for '{field}'.")
        return None
    # End if
    obs = obs_ds[[obs_var_name]].rename({obs_var_name: field})
    obs["weight"] = obs_weight
    return obs, obs_name


def _obs_weight(obs_ds):
    """Land weights for an observations dataset, or None if it has none."""
    # TODO is there a less brittle way to do this?
    if ("weight" in obs_ds) and ("datamask" in obs_ds):
        return obs_ds["weight"] * obs_ds["datamask"]
    if ("weight" in obs_ds) and ("LANDFRAC" in obs_ds):
        #used for MODIS albedo product
        return obs_ds["weight"] * obs_ds["LANDFRAC"]
    if ("area" in obs_ds) and ("landfrac" in obs_ds):
        return obs_ds["area"] * obs_ds["landfrac"]
    if ("area" in obs_ds) and ("landmask" in obs_ds):
        weight = obs_ds["area"] * obs_ds["landmask"]
        # Fluxnet data also has a datamask
        if "datamask" in obs_ds:
            weight = weight * obs_ds["datamask"]
        return weight
    return None


def get_region_boundaries(regions, region_name):
    """Get the boundaries of a specific region.

    Parameters
    ----------
    regions : dict
        The region definitions from the regions file.
    region_name : str
        The region.

    Returns
    -------
    tuple
        west, east, south, north (longitudes in -180 to 180) and the
        region's website category (None if it has none).
    """
    if region_name not in regions:
        raise ValueError(f"Region '{region_name}' not found in regions dictionary")

    region = regions[region_name]
    south, north = region["lat_bounds"]
    west, east = region["lon_bounds"]
    region_category = region.get("region_category")
    # YAML reads a bare "None" as the string "None":
    if region_category in ("None", ""):
        region_category = None

    return west, east, south, north, region_category


def subset_region(dset, region, kwargs):
    """The variable and its normalized weights in a region.

    Parameters
    ----------
    dset : xarray.Dataset or uxarray.UxDataset
        One variable plus its weights as "weight", see `load_with_weights`.
    region : tuple
        west, east, south, north, as from `get_region_boundaries`.
    kwargs : dict
        The loader keyword arguments; unstructured when they ask for it.

    Returns
    -------
    tuple
        The variable and its weights (normalized to sum to 1) in the region.

    Raises
    ------
    ValueError
        if there is no land in the region (the weights sum to zero).
    """
    field = [name for name in dset.data_vars if name != "weight"][0]
    if kwargs.get("unstructured_plotting"):
        return _region_uxarray(dset, field, *region)
    return _region_xarray(dset, field, *region)


def region_sum(var, wgt):
    """Weighted sum over the horizontal dimension(s), i.e. the regional mean."""
    horiz = [dim for dim in var.dims if dim in ("n_face", "lndgrid", "lat", "lon")]
    return (var * wgt).sum(dim=horiz)


def _region_uxarray(dset, field, box_w, box_e, box_s, box_n):
    """Subset unstructured data to the faces with centres inside the box."""
    grid = dset.uxgrid
    face_lons = grid.face_lon.values
    face_lats = grid.face_lat.values

    # Create a boolean mask for faces within your domain
    in_domain = ((face_lons >= box_w) & (face_lons <= box_e) &
                 (face_lats >= box_s) & (face_lats <= box_n))
    face_indices = np.where(in_domain)[0]

    domain_subset = dset[field].isel(n_face=face_indices)
    wgt_subset = utils.normalize_weights(dset["weight"].isel(n_face=face_indices))
    return domain_subset, wgt_subset


def _region_xarray(dset, field, box_w, box_e, box_s, box_n):
    """Subset lat/lon data to the box, with longitudes taken as 0 to 360."""
    # Convert the data's longitudes to [0,360] if necessary:
    if dset.lon.values.min() < 0:
        dset = dset.assign_coords(lon=dset["lon"] % 360).sortby("lon")
    # End if

    # Convert box_w and box_e to [0,360] if necessary
    if (box_w == -180) and (box_e == 180):
        box_w, box_e = 0, 360  # Special case for global domain
    if box_w < 0:
        box_w = box_w + 360
    if box_e < 0:
        box_e = box_e + 360

    # Subset the dataset using the specified box
    if box_w < box_e:
        subset = dset.sel(lat=slice(box_s, box_n), lon=slice(box_w, box_e))
    else:
        # The box crosses the prime meridian:
        lon_sel = (dset.lon >= box_w) | (dset.lon <= box_e)
        subset = dset.sel(lat=slice(box_s, box_n), lon=lon_sel)
    # End if

    wgt_subset = utils.normalize_weights(subset["weight"].fillna(0))
    return subset[field], wgt_subset


def _set_map_extent(map_ax, region_name, box_e, box_s):
    """Zoom the region map to the region."""
    if region_name in REGION_MAP_EXTENTS:
        extent = REGION_MAP_EXTENTS[region_name]
    elif (box_s >= 30) and (box_e <= -5):
        extent = [-180, 0, 30, 90]
    elif box_s >= 30:
        extent = [-10, 179, 30, 90]
    elif (box_s >= -30) and (box_e <= -5):
        extent = [-180, 0, -30, 30]
    elif box_s >= -30:
        extent = [-10, 179, -30, 30]
    elif (box_s >= -60) and (box_e >= -5):
        extent = [-10, 179, -89, -30]
    elif box_s >= -60:
        extent = [-180, 0, -89, -30]
    else:
        extent = [-180, 179, -89, -60]
    # End if
    map_ax.set_extent(extent, crs=ccrs.PlateCarree())


def _plot_region_map(fig, region_var, unstructured):
    """Draw where the region's data is, in the first panel."""
    transform = ccrs.PlateCarree()
    map_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())
    map_ax.coastlines()
    if "time" in region_var.dims:
        region_var = region_var.isel(time=0)
    # End if
    mask = region_var.copy()
    mask.values = np.where(np.isfinite(mask.values), 1.0, np.nan)
    if unstructured:
        collection = mask.to_polycollection()
        collection.set_transform(transform)
        collection.set_cmap("rainbow_r")
        collection.set_antialiased(False)
        map_ax.add_collection(collection)
    else:
        # Plot using pcolormesh for structured grids
        map_ax.pcolormesh(mask.lon, mask.lat, mask.values,
                          transform=transform, cmap="rainbow_r", shading="auto")
    # End if
    return map_ax


def make_regional_plots(adfobj, setup, base_data, plot_series, plot_kind, title):
    """Make one figure per region: a map panel and a panel per variable.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object.
    setup : dict
        From `regional_setup`.
    base_data : dict
        The baseline data for each variable (None where it is missing).
    plot_series : callable
        ``plot_series(ax, field, region)`` plots one variable's regional
        series on `ax` and returns the baseline data in the region (used to
        draw the region map); raises ValueError if the region has no land.
    plot_kind : str
        The website plot type, and part of the file name.
    title : str
        Word used in messages ("climatology", "time series").
    """
    unstructured = bool(setup["kwargs"].get("unstructured_plotting"))
    for region_name in setup["region_list"]:
        print(f"\n\t - Plotting regional {title} for: {region_name}")
        try:
            *region, region_category = get_region_boundaries(setup["regions"], region_name)
        except ValueError as err:
            warnings.warn(f"\t WARNING: {err}, skipping it.")
            continue
        # End try

        plot_loc = setup["plot_loc"] / f"{region_name}_plot_{plot_kind}_Mean.{setup['plot_type']}"
        web_kwargs = {"season": None, "multi_case": True, "non_season": True,
                      "category": region_category, "plot_type": plot_kind}

        # Check redo_plot. If set to True: remove old plots, if they already exist:
        if (not setup["redo_plot"]) and plot_loc.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_loc}' exists and clobber is false.")
            adfobj.add_website_data(plot_loc, region_name, None, **web_kwargs)
            continue
        # End if

        ## Set up figure
        ## TODO: Make the plot size/number of subplots responsive to number of fields specified
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        axs = axs.ravel()
        fig.delaxes(axs[0])

        plt_counter = 1
        for field in setup["var_list"]:
            if base_data.get(field) is None:
                continue
            # End if
            try:
                base_var = plot_series(axs[plt_counter], field, region)
            except ValueError as err:
                # e.g. no land in region, so weights sum to zero
                warnings.warn(f"\t WARNING: skipping {field} for {region_name}: {err}")
                continue
            # End try
            axs[plt_counter].set_title(field)
            axs[plt_counter].set_ylabel(base_data[field][field].attrs.get("units", ""))

            ## Define region in first subplot
            if plt_counter == 1:
                map_ax = _plot_region_map(fig, base_var, unstructured)
                _set_map_extent(map_ax, region_name, region[1], region[2])
            # End if
            plt_counter += 1
        # End for

        # Nothing was plotted for this region (e.g. no land in region), so skip it:
        if plt_counter == 1:
            warnings.warn(f"\t WARNING: no variables plotted for {region_name}, skipping figure.")
            plt.close(fig)
            continue
        # End if

        # Remove the panels that were not used:
        for ax in axs[plt_counter:]:
            fig.delaxes(ax)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Save out figure
        if plot_loc.is_file():
            plot_loc.unlink()
        fig.savefig(plot_loc, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc, region_name, None, **web_kwargs)
    # End for


##############
#END OF SCRIPT
