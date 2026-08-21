'''
Module: cam_taylor_diagram

Provides a Taylor diagram following the AMWG package. Uses spatial information only.

This module, for better or worse, provides both the computation and plotting functionality.
It depends on an ADF instance to obtain the regridded `climo` files.
It is designed to have one "reference" case (could be observations) and arbitrary test cases.
When multiple test cases are provided, they are plotted with different colors.

'''
#
# --- imports and configuration ---
#
import os
import sys
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    print("WARNING: xesmf not available, regridding in derived variables may fail")

import adf_utils as utils

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


from contextlib import redirect_stdout, contextmanager

@contextmanager
def silence_output():
    sys.stdout.flush() # Flush Python's buffer
    sys.stderr.flush()
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            # Also catch the C-level stderr/stdout if possible
            # This is the most robust way to silence C extensions
            old_stdout_fd = os.dup(sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            try:
                yield
            finally:
                os.dup2(old_stdout_fd, sys.stdout.fileno())
                os.close(old_stdout_fd)

def get_level_dim(dset):
    """Get the name of the level dimension in the dataset."""
    level_dims = ['lev', 'level', 'ilev']
    for dim in level_dims:
        if dim in dset.dims:
            return dim
    return None


#
# --- Main Function Shares Name with Module: cam_taylor_diagram ---
#
def cam_taylor_diagram(adfobj):
    """Create Taylor diagrams for specified configuration."""
    msg = "\n  Generating Taylor Diagrams..."
    logger.info(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Extract needed quantities from ADF object:
    # -----------------------------------------
    # NOTE: "baseline" == "reference" == "observations" will be called `base`
    #       test case(s) == case(s) to be diagnosed  will be called `case` (assumes a list)
    case_names: list = adfobj.get_cam_info('cam_case_name', required=True)
    test_nicknames: list = adfobj.case_nicknames["test_nicknames"]
    syear_cases: list = adfobj.climo_yrs["syears"]
    eyear_cases: list = adfobj.climo_yrs["eyears"]

    # ADF variable which contains the output path for plots and tables:
    plot_location = adfobj.plot_location
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            #Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            logger.warning(f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}")
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)

    # reference data set(s) -- if comparing with obs, these are dicts.
    data_name = adfobj.data.ref_case_label
    base_nickname = adfobj.data.ref_nickname

    #Extract baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')

    #Check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    logger.info(f"\t redo_plot is set to {redo_plot}")

    # Check for required variables
    taylor_var_set = {'U', 'PSL', 'SWCF', 'LWCF', 'LANDFRAC', 'TREFHT', 'TAUX', 'RELHUM', 'T'}
    available_vars = set(adfobj.diag_var_list)
    missing_vars = taylor_var_set - available_vars
    # Check for precipitation (Needs PRECT OR both PRECL and PRECC)
    has_prect = 'PRECT' in available_vars
    has_precl_precc = {'PRECL', 'PRECC'}.issubset(available_vars)
    if adfobj.compare_obs and has_precl_precc and not has_prect:
        # PRECC and PRECL have no observational counterpart, so deriving PRECT
        # from them only works against a baseline simulation. Comparing to obs
        # needs PRECT itself in diag_var_list -- ADF derives it from PRECC +
        # PRECL when making the time series. Without it the two tropical precip
        # entries drop out of the diagram silently.
        logger.warning("\t WARNING: comparing against observations, so the tropical "
                       "precipitation entries need 'PRECT' in diag_var_list; PRECC and "
                       "PRECL have no observations to compare against. ADF will derive "
                       "PRECT for you if you add it.")
    if missing_vars or not (has_prect or has_precl_precc):
        logger.warning("\tTaylor Diagrams skipped due to missing variables:")
        if missing_vars:
            logger.warning(f"\t Missing: {', '.join(sorted(missing_vars))}")
        if not (has_prect or has_precl_precc):
            if not has_prect:
                logger.warning("\t Missing: PRECT (Alternative PRECL + PRECC also incomplete)")
        logger.info("\n\tFull requirement: U, PSL, SWCF, LWCF, LANDFRAC, TREFHT, TAUX, RELHUM, T,")
        logger.info("\tAND (PRECT OR both PRECL & PRECC)")
        return


    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    # TAYLOR PLOT VARIABLES:
    var_list = ['PSL', 'SWCF', 'LWCF',
                'TropicalLandPrecip', 'TropicalOceanPrecip',
                'Land2mTemperature', 'EquatorialPacificStress',
                'U300', 'ColumnRelativeHumidity', 'ColumnTemperature']

    case_colors = [mpl.cm.tab20(i) for i, case in enumerate(case_names)] # change color for each case

    #
    # LOOP OVER SEASON
    #
    for season, months in seasons.items():
        logger.debug(f"TAYLOR DIAGRAM SEASON: {season}")
        plot_name = plot_loc / f"TaylorDiag_{season}_Special_Mean.{plot_type}"

        # Check redo_plot. If set to True: remove old plot, if it already exists:
        if (not redo_plot) and plot_name.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(plot_name, "TaylorDiag", None, season=season, multi_case=True)
            continue
        elif (redo_plot) and plot_name.is_file():
            plot_name.unlink()

        # hold the data in a DataFrame for each case
        # variable | correlation | stddev ratio | bias
        df_template = pd.DataFrame(index=var_list, columns=['corr', 'ratio', 'bias'])
        result_by_case = {cname: df_template.copy() for cname in case_names}

        for v in var_list:
            logger.debug(f"TAYLOR DIAGRAM VARIABLE: {v}")
            # Load reference data (already regridded to target grid)
            ref_data = _retrieve(adfobj, v, data_name)
            if ref_data is None:
                logger.warning(f"\t WARNING: No regridded reference data for {v} in {data_name}, skipping.")
                continue
            with silence_output():
                ref_data = ref_data.sel(time=months).mean(dim='time').compute()

            for casenumber, case in enumerate(case_names):
                # Load test case data regridded to match reference grid
                case_data = _retrieve(adfobj, v, case)
                if case_data is None:
                    logger.warning(f"\t WARNING: No regridded data for {v} in {case}, skipping.")
                    continue
                case_data = case_data.sel(time=months).mean(dim='time').compute()
                result_by_case[case].loc[v] = taylor_stats_single(case_data, ref_data)

        # -- PLOTTING (one per season) --
        logger.debug(f"TAYLOR DIAGRAM PLOTTING: {season}")
        fig, ax = taylor_plot_setup(title=f"Taylor Diagram - {season}",
                                    baseline=f"Baseline: {base_nickname}  yrs: {syear_baseline}-{eyear_baseline}")

        for i, case in enumerate(case_names):
            logger.debug(f"\t TAYLOR DIAGRAM CASE: {case}")
            ax = plot_taylor_data(ax, result_by_case[case], case_color=case_colors[i], use_bias=True)

        ax = taylor_plot_finalize(ax, test_nicknames, case_colors, syear_cases, eyear_cases, needs_bias_labels=True)
        logger.debug(f"TAYLOR DIAGRAM SAVING: {plot_name}")
        # add text with variable names:
        txtstrs = [f"{i+1} - {v}" for i, v in enumerate(var_list)]
        fig.text(0.9, 0.9, "\n".join(txtstrs), va='top')
        fig.savefig(plot_name, bbox_inches='tight')
        adfobj.debug_log(f"\t Taylor Diagram: completed {season}. \n\t File: {plot_name}")

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_name, "TaylorDiag", None, season=season, multi_case=True)
        plt.close(fig)
        logger.debug(f"TAYLOR DIAGRAM FINISHED WITH {season}")

    #Notify user that script has ended:
    logger.info("Taylor Diagrams have been generated successfully.")

#
# --- Local Functions ---
#

# --- DERIVED VARIABLES ---


def find_landmask(adf, casename):
    return _retrieve(adf, 'LANDFRAC', casename)


def regrid_to_target(adf, casename, source_da, target_da, method='conservative'):
    """
    Regrid source_da to match the grid of target_da using xesmf.
    
    Parameters:
    - adf: ADF object for getting output locations
    - casename: the casename of the source data - used to determine paths.
    - source_da: xarray.DataArray to regrid
    - target_da: xarray.DataArray with target grid
    - method: regridding method ('conservative', 'bilinear', etc.)
    
    Returns:
    - Regridded DataArray
    """
    if not XESMF_AVAILABLE:
        logger.error("xesmf not available, cannot regrid")
        return source_da
    
    if source_da.lat.shape == target_da.lat.shape and source_da.lon.shape == target_da.lon.shape:
        logger.debug("Grids already match, no regridding needed")
        return source_da
    
    logger.debug(f"Regridding from {source_da.lat.shape} x {source_da.lon.shape} to {target_da.lat.shape} x {target_da.lon.shape}")

    # Create clean grids for xesmf
    # source_grid = utils.create_clean_grid(source_da)
    # target_grid = utils.create_clean_grid(target_da)
    source_grid = utils.create_clean_grid(source_da.reset_coords(drop=True))
    target_grid = utils.create_clean_grid(target_da.reset_coords(drop=True))
    
    # Manage weights files -- MULTI-CASE NEEDS TO KNOW CASENAME
    regrid_loc = Path(adf.get_basic_info("cam_regrid_loc", required=True))
    case_list = adf.get_cam_info("cam_case_name", required=True)
    subdir = "regrid_weights" if casename in case_list else "obs_regrid_weights"
    regrid_weights_dir = regrid_loc / subdir
    regrid_weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate grid descriptions
    source_grid_type = "unstructured" if "ncol" in source_da.dims else "structured"
    target_grid_type = "unstructured" if "ncol" in target_da.dims else "structured"
    
    source_grid_desc = f"{source_grid_type}_{len(source_da.lat)}_{len(source_da.lon)}" if source_grid_type == "structured" else f"{source_grid_type}_{len(source_da.ncol)}"
    target_grid_desc = f"{target_grid_type}_{len(target_da.lat)}_{len(target_da.lon)}" if target_grid_type == "structured" else f"{target_grid_type}_{len(target_da.ncol)}"
    
    weights_file = regrid_weights_dir / f"weights_{source_grid_desc}_to_{target_grid_desc}_{method}.nc"
    logger.debug(f">> Weights file: {weights_file}")
    if weights_file.exists():
        logger.debug(f"Using existing regridding weights file: {weights_file}")
        with silence_output():
            logger.debug(">>Set up regridder from existing regridding weights file.")
            regridder = xe.Regridder(source_grid, target_grid, method, weights=str(weights_file))     
            logger.debug("<<Finished setting up regridder from existing regridding weights file.")
    else:
        logger.debug(f"Creating new regridding weights file: {weights_file}")
        with silence_output():
            regridder = xe.Regridder(source_grid, target_grid, method)
            regridder.to_netcdf(weights_file)
    logger.debug("Exectute regridding on silent mode.") 
    with silence_output():
        # Also catch numpy floating point errors just in case
        with np.errstate(invalid='ignore'):
            regridded = regridder(source_da)

    logger.debug("Returning regridded DataArray")
    return regridded



def _is_ref(adf, casename):
    """True if casename refers to the reference/baseline data rather than a test case."""
    return casename == adf.data.ref_case_label


def _load_ref_da(adf, variable):
    """Reference field on the target grid.

    Observations (and baseline fields that needed vertical interpolation) have a
    regridded file; a baseline simulation defines the target grid, so its plain
    climo file is already on that grid.
    """
    da = adf.data.load_reference_regrid_da(adf.data.ref_labels[variable], variable)
    if da is None:
        da = adf.data.load_reference_climo_da(adf.data.ref_case_label, variable)
    return da


def _load_ref_ds(adf, variable):
    """Reference dataset on the target grid; see _load_ref_da."""
    ds = adf.data.load_reference_regrid_dataset(adf.data.ref_labels[variable], variable)
    if ds is None:
        ds = adf.data.load_reference_climo_ds(adf.data.ref_case_label, variable)
    return ds


def _load_field(adf, casename, variable):
    """Load `variable` on the target grid, for a test case or the reference.

    Returns None when the variable is not part of this run, rather than raising,
    so callers can fall back to deriving it (e.g. PRECT from PRECC + PRECL).
    """
    if variable not in adf.data.ref_var_nam:
        return None
    if _is_ref(adf, casename):
        return _load_ref_da(adf, variable)
    return adf.data.load_regrid_da(casename, variable)


def _load_field_ds(adf, casename, variable):
    """Dataset form of _load_field; needed where hyam/hybm are also required."""
    if variable not in adf.data.ref_var_nam:
        return None
    if _is_ref(adf, casename):
        return _load_ref_ds(adf, variable)
    return adf.data.load_regrid_dataset(casename, variable)


def get_prect(adf, casename, **kwargs):
    prect = _load_field(adf, casename, 'PRECT')
    if prect is not None:
        return prect
    logger.info("\t Need to derive PRECT = PRECC + PRECL")
    precc = _load_field(adf, casename, 'PRECC')
    precl = _load_field(adf, casename, 'PRECL')
    if precc is None or precl is None:
        logger.warning(f"\t WARNING: Could not derive PRECT for {casename} "
                       "(missing PRECC or PRECL)")
        return None
    return precc + precl

def get_tropical_land_precip(adf, casename, **kwargs):
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    prect = get_prect(adf, casename)
    if prect is None:
        return None
    
    # Regrid prect to match landfrac grid if necessary
    prect = regrid_to_target(adf, casename, prect, landfrac)
    # mask to only keep land locations
    prect = xr.DataArray(np.where(landfrac >= .95, prect, np.nan),
                         dims=prect.dims,
                         coords=prect.coords,
                         attrs=prect.attrs)  # threshold could be 1
    return prect.sel(lat=slice(-30,30))


def get_tropical_ocean_precip(adf, casename, **kwargs):
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    prect = get_prect(adf, casename)
    if prect is None:
        return None
    
    # Regrid prect to match landfrac grid if necessary
    prect = regrid_to_target(adf, casename, prect, landfrac)
    
    # mask to only keep ocean locations
    prect = xr.DataArray(np.where(landfrac <= 0.05, prect, np.nan),
                         dims=prect.dims,
                         coords=prect.coords,
                         attrs=prect.attrs)
    return prect.sel(lat=slice(-30,30))


def get_surface_pressure(adf, dset, casename):
    if isinstance(dset, xr.Dataset) and 'PS' in dset.data_vars:
        ps = dset['PS']
    else:
        ps = _load_field(adf, casename, 'PS')
    if ps is None:
        logger.warning(f"\t WARNING: Could not load PS for {casename}.")
        return None    
    return ps


def get_var_at_plev(adf, casename, variable, plev):
    """Select `variable` at pressure level `plev` (hPa).

    Regridded files are already on pressure levels (regrid_and_vert_interp does
    the vertical interpolation), so this is a nearest-level selection; only the
    units of the level coordinate have to be sorted out.
    """
    da = _load_field(adf, casename, variable)
    if da is None:
        logger.warning(f"\t WARNING: {variable} unavailable for {casename}.")
        return None
    level_dim = get_level_dim(da)
    if level_dim is None:
        logger.warning(f"\t WARNING: {variable} for {casename} lacks a level dimension.")
        return None
    # Detect pressure units: if max(lev) > 2000, assume Pa, else hPa
    adjusted_plev = plev * 100 if da[level_dim].max().item() > 2000 else plev
    return da.sel(**{level_dim: adjusted_plev}, method='nearest')


def get_u_at_plev(adf, casename):
    return get_var_at_plev(adf, casename, "U", 300)


def get_vertical_average(adf, casename, varname):
    '''Collect data and apply mass-weighted vertical averaging with interface support.
    
    NOTE: the height coordinate is not weighted by density, so is biased.
    '''
    ds = _load_field_ds(adf, casename, varname)

    if ds is None: return None
    
    level_dim = get_level_dim(ds)
    if level_dim is None: return None

    if 'hyai' in ds and 'hybi' in ds:
        ps = get_surface_pressure(adf, ds, casename)
        p_int = utils.pres_from_hybrid(ps, ds['hyai'], ds['hybi'])
        # diff returns one fewer element than p_int
        # This represents the thickness of the layers BETWEEN interfaces
        dp = abs(p_int.diff(dim=get_level_dim(p_int)))
        
        if level_dim == get_level_dim(p_int):
            # Data is on interfaces (ilev). 
            # interpolate the layer thicknesses (dp) back to the interface points.
            weights = dp.rolling({level_dim: 2}, center=True).mean().fillna(0)
        else:
            # Data is on midpoints (lev). 
            # Assign the dp calculated from interfaces to the lev dimension.
            weights = dp.rename({get_level_dim(p_int): level_dim})
            # Ensure coordinates match exactly for alignment
            weights.coords[level_dim] = ds[level_dim]

    elif 'hyam' in ds and 'hybm' in ds:
        # Fallback for hybrid midpoints only
        ps = get_surface_pressure(adf, ds, casename)
        pres = utils.pres_from_hybrid(ps, ds['hyam'], ds['hybm'])
        weights = calculate_thickness_approx(pres, dim=level_dim)

    else:
        # Pure Pressure, Height, or Obs
        logger.debug("get_vertical_average: Pure Pressure, Height, or Obs")
        weights = calculate_thickness_approx(ds[level_dim], dim=level_dim)

    # Apply Weighted Average
    return weighted_vertical_average(ds[varname], weights, dim=level_dim)


def calculate_thickness_approx(coord, dim='lev'):
    """Approximates thickness using centered differences between midpoints."""
    if not isinstance(coord, (xr.DataArray, xr.Dataset)):
        # If it's just a coordinate object, convert to DataArray
        coord = xr.DataArray(coord, coords={dim: coord}, dims=[dim])
    logger.debug(f"\t calculate_thickness_approx: {dim = }, max: {coord.max().item()}, min: {coord.min().item()}")

    # (P_{k+1} - P_{k-1}) / 2
    upper = coord.shift({dim: -1})
    lower = coord.shift({dim: 1})
    diff = abs(upper - lower) / 2.0
    
    # For the edges (top/bottom), we can't do centered differences.
    # Take the distance to the only available neighbor.
    edge_diff = abs(coord.diff(dim=dim))
    
    # Fill the NaNs at the start and end of the array
    # bfill handles the first element, ffill handles the last
    return diff.fillna(edge_diff.bfill(dim).ffill(dim))


def weighted_vertical_average(da, weights, dim='lev'):
    """Computes mass-weighted average: Σ(φ * Δp) / Σ(Δp)"""
    # Ensure weights align with data (broadcasts PS if necessary)
    weighted_field = da * weights
    mask = da.notnull()
    total_weight = weights.where(mask).sum(dim=dim)
    return weighted_field.sum(dim=dim) / total_weight


def get_virh(adf, casename, **kwargs):
    '''Calculate vertically averaged relative humidity.'''
    return get_vertical_average(adf, casename, "RELHUM")


def get_vit(adf, casename, **kwargs):
    '''Calculate vertically averaged temperature.'''
    return get_vertical_average(adf, casename, "T")


def get_landt2m(adf, casename):
    '''Get 2-meter temperature over land'''
    t = _load_field(adf, casename, 'TREFHT')
    if t is None:
        return None
    
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    
    # Regrid t to match landfrac grid if necessary
    t = regrid_to_target(adf, casename, t, landfrac)
    
    t = xr.DataArray(np.where(landfrac >= .95, t, np.nan),
                     dims=t.dims, coords=t.coords, attrs=t.attrs)
    return t



def get_eqpactaux(adf, casename):
    """Get zonal surface wind stress 5°S to 5°N."""
    taux = _load_field(adf, casename, 'TAUX')
    if taux is None:
        logger.warning(f"\t WARNING: Could not load TAUX for {casename}")
        return None
    return taux.sel(lat=slice(-5, 5))


def get_derive_func(fld: str):
    '''Provide the function name for derived variables.'''
    funcs = {'TropicalLandPrecip': get_tropical_land_precip,
    'TropicalOceanPrecip': get_tropical_ocean_precip,
    'U300': get_u_at_plev,
    'ColumnRelativeHumidity': get_virh,
    'ColumnTemperature': get_vit,
    'Land2mTemperature': get_landt2m,
    'EquatorialPacificStress': get_eqpactaux
    }
    if fld not in funcs:
        logger.warning(f"We do not have a method for variable: {fld}.")
        return None
    return funcs[fld]


def _retrieve(adfobj, variable, casename, return_dataset=False):
    """Custom function that retrieves a variable using ADF loaders.
    Returns the variable as a DataArray (or Dataset if return_dataset=True).
    """
    v_to_derive = ['TropicalLandPrecip', 'TropicalOceanPrecip', 'EquatorialPacificStress',
                   'U300', 'ColumnRelativeHumidity', 'ColumnTemperature', 'Land2mTemperature']
    if variable in v_to_derive:
        func = get_derive_func(variable)
        if func is None:
            logger.error(f"No derivation function available for {variable}")
            return None
        da = func(adfobj, casename)
        if da is None:
            logger.warning(f"Derivation function for {variable} returned None for {casename}")
            return None
    else:
        da = _load_field(adfobj, casename, variable)
        if da is None:
            logger.warning(f"Failed to load {variable} for {casename}")
            return None

    if return_dataset and not isinstance(da, xr.Dataset):
        da = da.to_dataset(name=variable)
    return da


def weighted_correlation(x: xr.DataArray, y: xr.DataArray, weights: npt.ArrayLike):
    '''Calculate weighted correlation coefficient.'''
    mean_x = x.weighted(weights).mean()
    mean_y = y.weighted(weights).mean()
    dev_x = x - mean_x
    dev_y = y - mean_y
    cov_xy = (dev_x * dev_y).weighted(weights).mean()
    cov_xx = (dev_x * dev_x).weighted(weights).mean()
    cov_yy = (dev_y * dev_y).weighted(weights).mean()
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def weighted_std(x: xr.DataArray, weights: npt.ArrayLike):
    """Calculate weighted standard deviation."""
    xshape = x.shape
    wshape = weights.shape
    if xshape != wshape:
        wa = weights.broadcast_like(x)
    else:
        wa = weights
    mean_x = x.weighted(weights).mean()
    dev_x = x - mean_x
    swdev = (weights * dev_x**2).sum()
    total_weights = wa.where(x.notnull()).sum()
    return np.sqrt(swdev / total_weights)



def taylor_stats_single(casedata, refdata, w=True):
    """This replicates the basic functionality of 'taylor_stats' from NCL.
    PARAMTERS
    ---------
        casedata : input data, DataArray
        refdata  : reference case data, DataArray
        w        : if true use cos(latitude) as spatial weight, if false assume uniform weight
    RETURNS
    -------
        tuple: 
            pattern correlation, ratio of standard deviation (case/ref), bias
    """
    # Restrict both fields to the points where BOTH are valid. Masked variables
    # (land/ocean/tropics subsets) do not carry identical masks -- the model and
    # the observations disagree about coastlines -- and without this every
    # statistic below is taken over a different population: the case mean over
    # the case's points, the reference mean over the reference's, the covariance
    # over the intersection. That can put the correlation outside [-1, 1], and it
    # did: Land2mTemperature came out at 1.04, hidden on the plot because the
    # marker placement clips to 1. This is also what NCL's taylor_stats does.
    both_valid = casedata.notnull() & refdata.notnull()
    casedata = casedata.where(both_valid)
    refdata = refdata.where(both_valid)

    lat = casedata.lat
    if w:
        wgt = np.cos(np.radians(lat))
    else:
        wgt = np.ones(len(lat))
    correlation = weighted_correlation(casedata, refdata, wgt).item()
    a_sigma = weighted_std(casedata, wgt)
    b_sigma = weighted_std(refdata, wgt)
    mean_case = casedata.weighted(wgt).mean()
    mean_ref = refdata.weighted(wgt).mean()
    bias = (100*((mean_case - mean_ref)/mean_ref)).item()
    return correlation, a_sigma/b_sigma, bias


def taylor_plot_setup(title, baseline):
    """Constructs Figure and Axes objects for basic Taylor Diagram."""
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
    corr_labels = np.array([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.])
    corr_locations = np.pi/2 - np.arccos((corr_labels))  # azim. ticks in radians.
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim([0, 1.6])  # Works better than set_rmin / set_rmax
    ax.set_theta_zero_location("N") # zero at top,
    ax.set_theta_direction(-1)  # angle increases clockwise
    thetalines, thetalabels = ax.set_thetagrids(np.degrees(corr_locations), corr_labels)
    ax.grid(axis='x', linewidth=0)  # turn off radial grid
    ax.set_rgrids(np.arange(0, 1.75, .25))
    # ax.set_yticks([0.0, 1.0, 1.25]) # same effect as set_rgrids()
    ax.set_ylabel("Standardized Deviations")
    # Add tick marks along azimuth
    tick = [ax.get_rmax(),ax.get_rmax()*0.97]
    for t in corr_locations:
        ax.plot([t,t], tick, lw=0.72, color="k")
    ax.text(np.radians(50), ax.get_rmax()*1.1, "Correlation", ha='center', rotation=-50, fontsize=15)
    ax.text(np.radians(95), 1.0, "REF", ha='center')
    st = fig.suptitle(title, fontsize=18)
    st.set_y(1.)
    ax.set_title(baseline, fontsize=10,pad=15)
    return fig, ax


def plot_taylor_data(wks, df, **kwargs):
    """Apply data on top of the Taylor Diagram Axes.
        wks -> Axes object, probably from taylor_plot_setup
        df  -> DataFrame holding the Taylor stats.
        kwargs -> optional arguments
          look for 'use_bias'
          look for 'case_color'
    """
    # option is whether to stylize the markers by the bias:
    marker_list = ["v", "v", "v", "v", "o", "^", "^", "^", "^"]
    marker_size = [24, 16, 8, 4, 4, 4, 8, 16, 24]
    use_bias = bool(kwargs.get('use_bias', False))
    if use_bias:
        df['bias_digi'] = np.digitize(df['bias'].values, [-20, -10, -5, -1, 1, 5, 10, 20])
    # option: has color been specified as case_color?
    # --> expect the case labeling to be done external to this function
    if 'case_color' in kwargs:
        color = kwargs['case_color']
        if isinstance(color, int):
            # assume we should use this as an index
            color = mpl.cm.tab20(color) # convert to RGBA
            # TODO: allow colormap to be specified.
    # NOTE: number by position in the DataFrame, NOT by a running count of the
    # points actually drawn. The figure legend is built from the full variable
    # list, so a counter that skips missing variables shifts every later label
    # onto the wrong name -- which is the normal case when comparing to obs,
    # where several variables have no observational counterpart.
    for k, (ndx, row) in enumerate(df.iterrows(), start=1):
        # NOTE: ndx will be the DataFrame index, and we expect that to be the variable name
        if np.isnan(row['corr']) or np.isnan(row['ratio']):
            continue  # Skip plotting if data is missing
        theta = np.pi/2 - np.arccos(np.clip(row['corr'], -1.0, 1.0))  # Transform DATA
        if use_bias:
            mk = marker_list[row['bias_digi']]
            mksz = marker_size[row['bias_digi']]
            wks.plot(theta, row['ratio'], marker=mk, markersize=mksz, color=color)
        else:
            wks.plot(theta, row['ratio'], marker='o', markersize=16, color=color)
        wks.annotate(str(k), (theta, row['ratio']), ha='center', va='bottom',
                            xytext=(0,5), textcoords='offset points', fontsize='x-large',
                            color=color)
    return wks


def taylor_plot_finalize(wks, test_nicknames, casecolors, syear_cases, eyear_cases, needs_bias_labels=True):
    """Apply final formatting to a Taylor diagram.
        wks -> Axes object that has passed through taylor_plot_setup and plot_taylor_data
        casenames -> list of case names for the legend
        casecolors -> list of colors for the cases
        needs_bias_labels -> Bool, if T make the legend for the bias-sized markers.
    """
    # CASE LEGEND -- Color-coded
    bottom_of_text = 0.05

    height_of_lines = 0.03
    n = 0
    for case_idx, (s, c) in enumerate(zip(test_nicknames, casecolors)):

            wks.text(0.052, bottom_of_text + n*height_of_lines, f"{s}  yrs: {syear_cases[case_idx]}-{eyear_cases[case_idx]}",
            color=c, ha='left', va='bottom', transform=wks.transAxes, fontsize=10)
            n += 1
    wks.text(0.052, bottom_of_text + n*height_of_lines, "Cases:",
            color='k', ha='left', va='bottom', transform=wks.transAxes, fontsize=11)

    # BIAS LEGEND
    if needs_bias_labels:
        # produce an info-box showing the markers/sizes based on bias
        bias_legend_elements = [(Line2D([0], [0], marker="v", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0), Line2D([0], [0], marker="^", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="10-20%", markersize=16, linewidth=0), Line2D([0], [0], marker="^", color='k', label="10-20%", markersize=16, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="5-10%", markersize=8, linewidth=0), Line2D([0], [0], marker="^", color='k', label="5-10%", markersize=8, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label=">1-5%", markersize=4, linewidth=0), Line2D([0], [0], marker="^", color='k', label=">1-5%", markersize=4, linewidth=0)),
                                Line2D([0], [0], marker="o", color='k', label="< 1%", markersize=4, linewidth=0),
                                ]
        bias_legend_labels = ["> 20%", "10-20%", "5-10%", "1-5%", "< 1%"]
        wks.legend(handles=bias_legend_elements, labels=bias_legend_labels, loc='upper left', handler_map={tuple: HandlerTuple(ndivide=None, pad=2.)}, labelspacing=2, handletextpad=2, frameon=False, title=" - / + Bias",
                    title_fontsize=18)
    return wks