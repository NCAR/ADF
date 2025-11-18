import os
import joblib
from math import ceil
import warnings
from pathlib import Path

import numpy as np
import xesmf

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

import dask
import xarray as xr
import matplotlib as mpl
from shapely.geometry import Point
from shapely.prepared import prep

try:
    from numba import njit
except ImportError:

    def njit(func=None):
        if func is None:
            return njit
        # Issue a warning that numba is not available
        warnings.warn(
            "NumPy performance optimization using Numba is not available. "
            "Fallback to standard Python execution.",
            UserWarning,
        )
        return func


from dataclasses import dataclass

@dataclass(frozen=True)
class VariableNames:
    product_name: str
    data_var: str
    ht_var: str
    tau_var: str
    obs_data_var: str
    obs_ht_var: str
    obs_tau_var: str


# Consolidate data into a single dictionary of dataclass objects
ALL_VARS = {
    "FISCCP1_COSP": VariableNames(
        product_name="ISCCP",
        data_var="FISCCP1_COSP",
        ht_var="cosp_prs",
        tau_var="cosp_tau",
        obs_data_var="n_pctaudist",
        obs_ht_var="levtau",
        obs_tau_var="levpc",
    ),
    "CLD_MISR": VariableNames(
        product_name="MISR",
        data_var="CLD_MISR",
        ht_var="cosp_htmisr",
        tau_var="cosp_tau",
        obs_data_var="clMISR",
        obs_ht_var="tau",
        obs_tau_var="cth",
    ),
    "CLMODIS": VariableNames(
        product_name="MODIS",
        data_var="CLMODIS",
        ht_var="cosp_prs",
        tau_var="cosp_tau_modis",
        obs_data_var="MODIS_CLD_HISTO",
        obs_ht_var="COT",
        obs_tau_var="PRES",
    ),
}

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def cloud_regime_analysis(
    adf,
    wasserstein_or_euclidean="euclidean",
    ot_library="pot",
    emd_method=None,
    premade_cloud_regimes=None,
    lat_range=None,
    lon_range=None,
    only_ocean_or_land=None
):
    """
    Generates 2D maps and plots of Cloud Regime (CR) centers by comparing a
    test case against observations or a baseline simulation.

    This function orchestrates the ADF workflow to to generate 2-D lat/lon maps of Cloud Regimes (CRs) and plots of the CR
    centers themselves (CTP-tau histograms). It can fit data into CRs using either Wasserstein (AKA Earth Movers Distance) or
    Euclidean distance. 
    Checks for COSP variables in diag_var_list: FISCCP1_COSP, CLD_MISR, and CLMODIS. 
    Whichever are found will be processed.  
    Optionally do analysis on subsets by masking ocean or land and specifying latitude and/or longitude bounds.
    User-specified CRs can be provided, but default uses premade CRs from observational products from Davis & Medeiros (2024).

    There are 6 sets of premade CRs, two for each data product. One made with euclidean distance and one
    with Wasserstein distance for ISCCP, MODIS, and MISR. 
    Therefore when the wasserstein_or_euclidean variables is changed it is
    important to undertand that not only the distance metric used to fit data into CRs is changing, but also the CRs themselves
    unless the user is passing in a set of premade CRs with the premade_cloud_regimes variable.

    PARAMETERS
    ----------
    adf
        The ADF object
    wasserstein_or_euclidean : str ("wasserstein" | "euclidean")
        Whether to use wasserstein or euclidean distance to fit CRs. 
        This also selects the default CRs based on creation with kmeans the selected distance.
        Default is euclidean distance *because it is much faster than wasserstein*.
    ot_library : str ("pot" | "wasserstein")
        When wasserstein distance is used, this chooses tha backend for calculation:
        - "pot": python optimal transport is default, see: https://pythonot.github.io/index.html
        - "wasserstein" : wasserstein package, see: https://github.com/thaler-lab/Wasserstein
           NOTE: wasserstein was used originally (See Davis & Medeiros 2024), but as of ADF implementation, requires Numpy < 2.
    emd_method : str ("exact" | "sinkhorn")
        When wasserstein distances is used AND POT library is backend, specify the algorithm
        - "exact" is uses the exact algorithm, is default, and is recommended.
        - "sinkhorn" uses the Sinkhorn algorithm, which is faster, but is **highly experimental** and not recommeded.
    premade_cloud_regimes : Path-like to numpy array file
        Specify custom CRs to use rather than the those in ADF_variable_defaults
        - enter as a path to a numpy array of shape (k, n_tau_bins * n_pressure_bins)
        NOTE: specifying custom CRs has not been tested in ADF (caution!)
    lat_range : like of floats
        Range of latitudes to use, Example: [-30,30] 
        Default is use all available latitudes
    lon_range : list of floats
        Range of longitudes to use, Example [-90,90] 
        Default is use all available longitudes
    only_ocean_or_land : str
        Set to 
        - "O" to perform analysis with only points over water, 
        - "L" for only points over land, 
        - None or False to use data over land and water. 
        Default is None (land & water).
    """
    dask.config.set({"array.slicing.split_large_chunks": False})

    # 1. Validate user inputs & set configuration
    opts = _validate_user_inputs(
        wasserstein_or_euclidean,
        premade_cloud_regimes,
        lat_range,
        lon_range,
        only_ocean_or_land,
    )
    
    time_range = [str(adf.get_cam_info("start_year")[0]), str(adf.get_cam_info("end_year")[0])]
    opts['time_range'] = time_range
    landfrac_present = "LANDFRAC" in adf.diag_var_list
    opts['landfrac_present'] = landfrac_present
    opts['emd_method'] = emd_method
    opts['n_cpus'] = adf.get_basic_info('num_procs')

    # 2. Process each COSP cloud variable
    cr_vars = [field for field in adf.diag_var_list if field in ALL_VARS]
    for field in cr_vars:
        print(f"INFO: Processing variable: {field}")
        var_info = ALL_VARS[field]

        # 3. Load cluster centers
        cluster_spec = premade_cloud_regimes if premade_cloud_regimes is not None else opts['distance']
        cl = load_cluster_centers(adf, cluster_spec, field)
        if cl is None:
            warnings.warn(f"WARNING: Skipping {field} due to failed cluster center loading.")
            continue
        opts['cl_shape'] = cl.shape

        # 4. Load and process reference data to get reference labels
        ref_data = load_reference_data(adf, field)
        if ref_data is None: continue

        ref_labels = _get_ref_cluster_labels(adf, ref_data, field, var_info, cl, opts)
        if ref_labels is None:
            warnings.warn(f"WARNING: Could not generate reference labels for {field}. Skipping.")
            continue

        # 5. Process each test case against the reference
        for case_name in adf.data.case_names:
            print(f"\nINFO: Analyzing case: {case_name}")
            
            c_ts_da = adf.data.load_timeseries_da(case_name, field)
            if c_ts_da is None:
                warnings.warn(f"WARNING: Variable {field} for case '{case_name}' is None. Skipping.")
                continue

            # Regrid if on unstructured grid (e.g., 'ncol' dimension)
            if "ncol" in c_ts_da.dims:
                print("INFO: Regridding data from unstructured grid.")
                regrid_weights_file = Path("/glade/work/brianpm/mapping_ne30pg3_to_fv09_esmfbilin.nc")
                rg = make_se_regridder(regrid_weights_file, Method="bilinear")
                ds = regrid_se_data_bilinear(rg, c_ts_da, column_dim_name="ncol")
            else:
                ds = c_ts_da

            # Preprocess test case data
            processed_ds = _preprocess_data(ds, field, var_info, opts)
            if processed_ds is None: continue

            # Compute cluster labels for the test case
            test_labels = compute_cluster_labels(processed_ds, var_info.tau_var, var_info.ht_var, cl, opts['distance'], ot_library, method=opts['emd_method'], num_cpus=opts['n_cpus'])
            test_labels.attrs['k'] = cl.shape[0]

            # 6. Generate all plots
            print("INFO: Generating plots...")
            tau_coord = processed_ds[var_info.tau_var]
            ht_coord = processed_ds[var_info.ht_var]

            if adf.compare_obs:
                plot_hists_obs(field, cl, test_labels, ref_labels, processed_ds, ref_data, var_info.ht_var, var_info.tau_var, ht_coord, tau_coord, adf)
            else:
                plot_hists_baseline(field, cl, test_labels, ref_labels, processed_ds, ref_data, var_info.ht_var, var_info.tau_var, ht_coord, tau_coord, adf)

            plot_rfo_maps(test_labels, ref_labels, adf, field)


#
# --- local functions ---
#
def _validate_user_inputs(distance, regimes, lat_r, lon_r, land_ocean):
    """Validates inputs, returning an options dictionary."""
    opts = {}
    if distance not in ["euclidean", "wasserstein"]:
        warnings.warn('WARNING: Invalid distance metric. Defaulting to "euclidean".')
        opts['distance'] = "euclidean"
    else:
        opts['distance'] = distance
    opts['premade_cloud_regimes'] = regimes
    opts['lat_range'] = lat_r if isinstance(lat_r, list) and len(lat_r) == 2 else None
    opts['lon_range'] = lon_r if isinstance(lon_r, list) and len(lon_r) == 2 else None
    if not land_ocean:
        print("INFO: Default to using both LAND and OCEAN points.")
        opts['only_ocean_or_land'] = False
    elif land_ocean not in ["L", "O"]:
        warnings.warn('WARNING: Invalid land/ocean flag. Defaulting to False (land and ocean).')
        opts['only_ocean_or_land'] = False
    else:
        opts['only_ocean_or_land'] = land_ocean
    return opts

def _get_ref_cluster_labels(adf, ref_data, field, var_info, cl, opts):
    """Computes and returns cluster labels for the reference (obs or baseline)."""
    if adf.compare_obs:
        # If pre-computed labels are in the file, use them
        if opts['premade_cloud_regimes'] is None:
            label_var = "emd_cluster_labels" if opts['distance'] == "wasserstein" else "euclidean_cluster_labels"
            if label_var in ref_data:
                print(f"INFO: Using pre-computed reference labels: {label_var}")
                # Get the labels and ensure their longitude is standardized to -180 to 180
                labels = ref_data[label_var]
                if 'lon' in labels.coords and labels.lon.max() > 180:
                    print("INFO: Standardizing longitude for pre-computed reference labels.")
                    labels = labels.assign_coords(lon=(((labels.lon + 180) % 360) - 180)).sortby("lon")
                return labels
        # Otherwise, compute labels from histograms
        ref_ht_var = var_info.obs_ht_var
        ref_tau_var = var_info.obs_tau_var
        data_var = var_info.obs_data_var
        processed_ref = _preprocess_data(ref_data[data_var], data_var, VariableNames("", "", ref_ht_var, ref_tau_var, "", "", ""), opts)
    else: # Comparing to baseline simulation
        baseline_info = adf.get_baseline_info
        time_range_b = [str(baseline_info("start_year")), str(baseline_info("end_year"))]
        baseline_opts = {**opts, 'time_range': time_range_b}
        processed_ref = _preprocess_data(ref_data, field, var_info, baseline_opts)
        ref_ht_var, ref_tau_var = var_info.ht_var, var_info.tau_var

    if processed_ref is None:
        return None
    
    return compute_cluster_labels(processed_ref, ref_tau_var, ref_ht_var, cl, opts['distance'], ot_library, method=opts['emd_method'], num_cpus=opts['n_cpus'])

def _preprocess_data(ds, field_name, var_info, opts):
    """Performs all preprocessing steps on a data array before clustering."""
    if isinstance(ds, xr.Dataset):
        ds = ds[field_name]

    if 'lon' in ds.coords and ds.lon.max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")

    ds = apply_land_ocean_mask(ds, opts['only_ocean_or_land'], opts.get('landfrac_present'))
    if ds is None: return None
    
    ds = spatial_subset(ds, opts['lat_range'], opts['lon_range'])
    ds = temporal_subset(ds, opts.get('time_range'))
    if ds is None or 'time' not in ds.dims or ds.time.size == 0:
        warnings.warn(f"WARNING: No data remains for {field_name} after subsetting.")
        return None

    ds = select_valid_tau_height(ds, var_info.tau_var, var_info.ht_var)

    # Special handling when comparing against observational regimes
    if ALL_VARS[field_name].product_name == "ISCCP" and opts['cl_shape'][1] == 42:
        ds = ds.sel({var_info.tau_var: slice(ds[var_info.tau_var].min().item() + 1e-11, None)})
        print(f"\t Dropping smallest tau bin ({var_info.tau_var}) for obs comparison.")
    if ALL_VARS[field_name].product_name == "MISR" and opts['cl_shape'][1] == 105:
        ds = ds.sel({var_info.ht_var: slice(ds[var_info.ht_var].min().item() + 1e-11, None)})
        print(f"\t Dropping lowest height bin ({var_info.ht_var}) for obs comparison.")

    return ds


def compute_cluster_labels(ds, tau_var_name, ht_var_name, cl, wasserstein_or_euclidean, ot_library='pot', method=None, num_cpus=None):
    """
    Computes cluster labels for a given data array of histograms.
    
    PARAMETERS
    ----------
    ds : xr.DataArray
        input data with histograms
    tau_var_name : str
        tau dimension name
    ht_var_name : str
        CTH/CTP dimension name
    cl
        cluster centers
    wassterstein_or_euclidean : str
        distrance metric choice
    ot_library : str
        backend library for Wasserstein
    methos : str
        algorithm for wasserstein distance (default is exact)
    num_cpus : int
        number of CPU cores to assume for wasserstein calculation

    RETURNS
    -------
    cluster_labels : xr.DataArray
    """
    dims = [dim for dim in ds.dims if dim not in [tau_var_name, ht_var_name]]
    histograms = ds.stack(spacetime=dims, tau_ht=(tau_var_name, ht_var_name))
    mat = histograms.values
    is_valid = ~np.isnan(mat).any(axis=1)
    if not np.any(is_valid):
        warnings.warn("ERROR: No valid histograms found after removing NaNs.")
        return None
    mat_valid = mat[is_valid]
    print(f"INFO: Fitting {len(mat_valid)} valid histograms to cluster centers.")
    labels_valid = precomputed_clusters(mat_valid, cl, wasserstein_or_euclidean, ds, tau_var_name, ht_var_name, ot_library, method, num_cpus=num_cpus)
    
    cluster_labels_flat = np.full(len(mat), np.nan, dtype=np.float32)
    cluster_labels_flat[is_valid] = labels_valid
    
    cluster_labels = xr.DataArray(
        data=cluster_labels_flat,
        coords={"spacetime": histograms.spacetime},
        dims=("spacetime"),
    )
    return cluster_labels.unstack()


def precomputed_clusters(mat, cl, wasserstein_or_euclidean, ds, tau_var_name, ht_var_name, ot_library='pot', emd_method=None, num_cpus=None):
    """
    Compute cluster labels from precomputed cluster centers.
    
    PARAMETERS
    ----------
        mat
            array of histograms, reshaped to be (time-lat-lon)x(2 histogram dimensions)
        cl
            cluster centers
        wasserstein_or_euclidean : str
            choice of distance metric
        ds : xr.DataArray
            the histogram DataArray, used for the dimension/coordinate information
        tau_var_name, ht_var_name : str
            names of the tau and vertical dimensions (in ds)
        ot_library : str 
            The library to use for Optimal Transport. 
            Either 'pot' or 'wasserstein'. Defaults to 'pot'.
        emd_method : str
            When ot_library is 'pot', can use 'exact' or 'sinkhorn' for calculation
        num_cpus : int
            The number of CPU cores to specify (specified as number of threads for wasserstein)
    RETURNS
    -------
    array of cluster labels (integers)
    """
    if wasserstein_or_euclidean == "euclidean":
        distances = np.sum((mat[:, :, None] - cl.T[None, :, :]) ** 2, axis=1)        
    elif wasserstein_or_euclidean == "wasserstein":
        distances = None
        # Try preferred library first
        if ot_library == 'pot':
            distances = _compute_distances_pot(mat, cl, ds, tau_var_name, ht_var_name, method=emd_method)
        elif (ot_library is None) or (ot_library == 'wasserstein'):
            distances = _compute_distances_wasserstein(mat, cl, ds, tau_var_name, ht_var_name, num_cpus=num_cpus)
        else:
            warnings.warn(f"precomputed_clusters needs calculation backend in (`pot`,`wasserstein`), got {ot_library}")
            return None
        
    if distances is None:
        warnings.warn(f"ERROR: [precomputed_clusters] Calculation failed. Neither POT nor wasserstein library could be used successfully.")
        return None

    # find the smallest distance for each - that is the cluster classification
    return np.argmin(distances, axis=1)


def _compute_distances_pot(mat, cl, ds, tau_var_name, ht_var_name, method=None):
    """
    Computes pairwise Wasserstein distances using POT and parallelizes the
    calculation with joblib for performance.
    """
    try:
        import ot
        from ot.lp import emd2
        from ot import sinkhorn2
    except ImportError:
        warnings.warn("Python Optimal Transport (POT) package not found or corrupt. Cannot use 'pot' library.")
        return None

    print("\t INFO: Using Python Optimal Transport (POT) library with joblib for parallel execution.")
    
    # 1. Define the ground metric (cost matrix) ONCE.
    n_tau = len(ds[tau_var_name])
    n_ht = len(ds[ht_var_name])
    x_coords, y_coords = np.meshgrid(np.arange(n_ht), np.arange(n_tau))
    coords = np.vstack([y_coords.ravel(), x_coords.ravel()]).T
    
    M = ot.dist(coords, coords, metric='euclidean')
    M /= M.max()

    reg_val = 0.1 * M.mean() # used for regularization w/ Sinkhorn -- tunable parameter (bigger should be faster, but can be unstable).

    # 2. Normalize the cluster centers ONCE.
    cl_sum = cl.sum(axis=1, keepdims=True)
    cl_normalized = cl / (cl_sum + 1e-9)

    # 3. Define a helper function that INCLUDES normalization for the model data.
    def compute_single_histogram_distances(histogram, centers_normalized, cost_matrix, method=None):
        if method is None:
            method = 'sinkhorn'
        hist_sum = histogram.sum()
        if hist_sum < 1e-9:
            return np.full(centers_normalized.shape[0], np.inf)
        hist_normalized = histogram / hist_sum
        if method == 'exact':
            return [emd2(hist_normalized, center, cost_matrix) for center in centers_normalized]
        elif method == 'sinkhorn':
            # Use stabilized version. Somewhat faster than exact calculation.
            return [sinkhorn2(hist_normalized, center, cost_matrix, reg=reg_val, method='sinkhorn_stabilized', log=False) for center in centers_normalized]
        else:
            warnings.warn(f"ERROR: compute_single_histogram_distances method must be (None, sinkhorn, exact), got {method}")
            return None

    # --- FIX: Use the helper function to get the *allocated* core count ---
    n_jobs = 36 # _get_hpc_job_cores()
    print(f"\t Distributing EMD calculation across {n_jobs} allocated cores...")
    
    # 4. Use joblib to run the calculations in parallel
    distances_list = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(compute_single_histogram_distances)(mat[i, :], cl_normalized, M, method) for i in range(mat.shape[0])
    )
    
    distances = np.array(distances_list)
                
    return distances


def _compute_distances_wasserstein(mat, cl, ds, tau_var_name, ht_var_name, num_cpus=None):
    """
    Computes pairwise Wasserstein distances using the 'wasserstein' library.
    """
    try:
        import wasserstein
    except ImportError:
        warnings.warn("'wasserstein' package not found. Cannot use 'wasserstein' library.")
        return None

    print("\t INFO: Using 'wasserstein' library. Will try to JIT compile `stacking` function.")
    
    # This function is defined locally as it's highly specific to this library's API
    @njit()
    def stacking(position_matrix, centroids):
        centroid_list = []
        for i in range(len(centroids)):
            x = np.empty((3, len(mat[0]))).T
            x[:, 0] = centroids[i]
            x[:, 1] = position_matrix[0]
            x[:, 2] = position_matrix[1]
            centroid_list.append(x)
        return centroid_list

    n1 = len(ds[tau_var_name])
    n2 = len(ds[ht_var_name])
    R = (n1**2 + n2**2)**0.5

    position_matrix = np.zeros((2, n1, n2))
    position_matrix[0] = np.tile(np.arange(n2), (n1, 1))
    position_matrix[1] = np.tile(np.arange(n1), (n2, 1)).T
    position_matrix = position_matrix.reshape(2, -1)

    num_threads = num_cpus if num_cpus is not None else 1
    print(f"\t Using {num_threads} threads for calculation.")
    emds = wasserstein.PairwiseEMD(R=R, norm=True, dtype=np.float32, verbose=0, num_threads=num_threads)
    events = stacking(position_matrix, mat)
    centroid_list = stacking(position_matrix, cl)   
    emds(events, centroid_list)
    return emds.emds()

def _calculate_rfo(labels, cluster_index):
    """Calculates the spatial and total RFO for a given cluster using xarray.
    
    PARAMETERS
    ----------
    labels : xr.DataArray
        label data
    cluster_index : int
        cluster value to count

    RETURNS
    -------
    tuple of (RFO (array), total RFO (float))
    
    """
    if not isinstance(labels, xr.DataArray):
        warnings.warn(f"ERROR: Input 'labels' must be an xarray.DataArray, got {type(labels)}")
        return None

    # Spatial RFO map (% of time steps in the cluster)
    rfo_map = (labels == cluster_index).mean(dim="time", skipna=True) * 100

    # Total area-weighted RFO (scalar %)
    weights = np.cos(np.deg2rad(labels.lat))
    total_rfo_num = (labels == cluster_index).weighted(weights).sum()
    total_rfo_denom = (labels >= 0).weighted(weights).sum()
    
    total_rfo = (total_rfo_num / total_rfo_denom * 100).item() if total_rfo_denom > 0 else 0
    return rfo_map, total_rfo
def load_reference_data(adfobj, varname):
    """Load reference data, which could be an observation or a baseline simulation."""
    # ... (function content is identical to your original)
    base_name = adfobj.data.ref_case_label
    ref_var_nam = adfobj.data.ref_var_nam[varname] # shuld work for obs/sim
    print(f"[CRA: load_reference_data] {base_name = }, {ref_var_nam = }")

    if adfobj.compare_obs:
        ocase = adfobj.data.ref_case_label
        fils = adfobj.data.ref_var_loc.get(varname, None)
        if not isinstance(fils, list):
            fils = [fils]
        ds = adfobj.data.load_dataset(fils)
        if ds is None:
            warnings.warn(f"\t    WARNING: Load failed reference data for {varname}")
            return None
        print(f"[CRA: load_reference_data] return observation dataset")
        return ds
    else:
        print(f"[CRA: load_reference_data] returning simulation dataarray")
        return adfobj.data.load_reference_timeseries_da(varname) 

def load_cluster_centers(adf, cluster_spec, variablename):
    """Loads cluster center data from a specified source."""
    if isinstance(cluster_spec, str):
        if cluster_spec in ('wasserstein', 'euclidean'):
            if cluster_spec == 'wasserstein':
                algo = 'emd'
            else:
                algo = 'euclidean'
            try:
                # Use variablename to find the data product name
                data = ALL_VARS[variablename].product_name
                obs_data_loc = Path(adf.get_basic_info("obs_data_loc"))
                data_key = f"{data}_{algo}_centers"
                cluster_centers_path = adf.variable_defaults[data_key]["obs_file"]
                file_path = obs_data_loc / cluster_centers_path
            except KeyError as e:
                warnings.warn(
                    f"[ERROR] Could not find '{variablename}' in ALL_VARS or default file path for '{cluster_spec} with {algo = }'. "
                    f"Original error: {e}"
                )
                return None
        else:
            file_path = Path(cluster_spec) # Assume it's a direct file path
    elif isinstance(cluster_spec, Path):
        file_path = cluster_spec
    else:
        warnings.warn(f"ERROR: cluster_spec must be a string or Path, not {type(cluster_spec)}")
        return None

    if not file_path.exists():
        warnings.warn(f"[ERROR] Cluster center file not found at: {file_path}")
        return None

    try:
        if file_path.suffix == ".nc":
            cl = xr.open_dataset(file_path)['centers'].values
        elif file_path.suffix == ".npy":
            cl = np.load(file_path)
        else:
            warnings.warn(f"[ERROR] Unsupported file type: {file_path.suffix}")
            return None
    except Exception as e:
        warnings.warn(f"[ERROR] Could not load {file_path.name}: {e}")
        return None

    return cl

# --------
# PLOTTING
# --------
def _plot_map(ax, lon, lat, data, title, cmap, vmin, vmax):
    """Plots a single lat/lon map on a given axis."""
    ax.set_global()
    ax.coastlines()
    mesh = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        rasterized=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=4)
    return mesh

def _configure_map_axes(ax, is_left, is_bottom):
    """Configures ticks and labels for a map subplot."""
    if is_left:
        ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    if is_bottom:
        ax.set_xticks([-120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))


def _add_colorbar(fig, ax, cmap, norm):
    """Add colorbar to the figure."""
    p = [0, 0.2, 1, 2, 3, 4, 6, 8, 10, 15, 99]
    # cbar_ax = fig.add_axes([1.01, 0.25, 0.04, 0.5])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])  # Required for colorbar
    cb = fig.colorbar(sm, 
                      ax=ax, 
                      orientation='vertical', 
                      fraction=0.025, 
                      pad=0.02,
                      aspect=40,
                      ticks=p)
    cb.set_label(label="Cloud Cover (%)", size=16)
    cb.ax.tick_params(labelsize=14)


def _add_colorbar2(fig, ax, mappable, label):
    """Adds a colorbar next to a given axis."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label=label)

def plot_rfo_maps(test_labels, ref_labels, adf, field):
    """
    Plots Relative Frequency of Occurrence (RFO) maps for test, reference,
    and their difference for each cloud regime.
    """
    k = test_labels.attrs.get("k", int(np.nanmax(test_labels.values)) + 1)
    obs_or_base = "Observation" if adf.compare_obs else "Baseline"
    plt.rcParams.update({"font.size": 13, "figure.dpi": 200})
    for cluster in range(k):
        fig, axes = plt.subplots(
            nrows=2, ncols=2,
            subplot_kw={"projection": ccrs.PlateCarree()},
            figsize=(12, 7)
        )
        fig.subplots_adjust(wspace=0.15, hspace=0.15)
        ax = axes.ravel()

        # 1. Reference RFO
        rfo_ref, total_rfo_ref = _calculate_rfo(ref_labels, cluster)
        mesh1 = _plot_map(ax[0], rfo_ref.lon, rfo_ref.lat, rfo_ref,
                          f"{obs_or_base}, RFO = {total_rfo_ref:.1f}%",
                          "GnBu", 0, 100)
        
        # 2. Test Case RFO
        rfo_test, total_rfo_test = _calculate_rfo(test_labels, cluster)
        mesh2 = _plot_map(ax[1], rfo_test.lon, rfo_test.lat, rfo_test,
                          f"Test Case, RFO = {total_rfo_test:.1f}%",
                          "GnBu", 0, 100)
        _add_colorbar2(fig, ax[1], mesh2, "RFO (%)")

        # 3. Difference RFO (regrid if necessary)
        if rfo_ref.shape != rfo_test.shape:
            rfo_ref = rfo_ref.interp_like(rfo_test, method="nearest")
        
        rfo_diff = rfo_test - rfo_ref
        total_rfo_diff = total_rfo_test - total_rfo_ref
        mesh3 = _plot_map(ax[2], rfo_diff.lon, rfo_diff.lat, rfo_diff,
                          f"Test - {obs_or_base}, ΔRFO = {total_rfo_diff:.1f}%",
                          "coolwarm", -100, 100)
        _add_colorbar2(fig, ax[2], mesh3, "ΔRFO (%)")

        # Configure all axes
        _configure_map_axes(ax[0], is_left=True, is_bottom=False)
        _configure_map_axes(ax[1], is_left=False, is_bottom=False)
        _configure_map_axes(ax[2], is_left=True, is_bottom=True)
        # Manually set ticks for bottom right axis
        ax[3].remove()

        fig.suptitle(f"CR{cluster+1} Relative Frequency of Occurrence", fontsize=16, y=0.95)
        
        # Save figure
        save_path = Path(adf.plot_location[0]) / f"{field}_CR{cluster+1}_LatLon_mean.png"
        plt.savefig(save_path, bbox_inches='tight')

        if adf.create_html:
            adf.add_website_data(str(save_path), field, case_name=None, multi_case=True)
        
        plt.close(fig)


def plot_hists_baseline(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord, adf):
    """Plot cloud regime centers for observations, baseline, and test case."""
    plot_data = _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord)
    plot_data['columns'] = ['observation', 'baseline', 'test_case']
    plot_data['figsize'] = (17, plot_data['fig_height'])
    plot_data['save_suffix'] = '_CR_centers'
    
    _plot_cloud_regimes(plot_data, adf, baseline_mode=True)


def plot_hists_obs(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord, adf):
    """Plot cloud regime centers for observations and test case."""
    plot_data = _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord)
    plot_data['columns'] = ['observation', 'test_case']
    plot_data['figsize'] = (12, plot_data['fig_height'])
    plot_data['save_suffix'] = '_CR_centers'
    
    _plot_cloud_regimes(plot_data, adf, baseline_mode=False)


def _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord):
    """Prepare all data needed for plotting.
    
    fld: str
        name of variable

    cl
        cluster centers
    
    cluster_labels: xr.DataArray
        labels for case ([time], lat, lon)
    cluster_labels_o: array-like
        labels for reference data
    histograms: xr.DataAray (?)
    
    """
    data = ALL_VARS[fld].product_name
    k = len(cl)
    ylabels = htcoord.values
    xlabels = taucoord.values
    
    # Create meshgrid
    X2, Y2 = np.meshgrid(np.arange(len(xlabels) + 1), np.arange(len(ylabels) + 1))
    
    # Calculate figure height
    fig_height = (1 + 10 / 3 * ceil(k / 3)) * 3
    
    # Create weights for RFO calculations
    weights = np.cos(np.deg2rad(cluster_labels.stack(z=("time", "lat", "lon")).lat.values))
    valid_inds = ~np.isnan(cluster_labels.stack(z=("time", "lat", "lon")))
    weights = weights[valid_inds]
    
    return {
        'field': fld,
        'data_product': data,
        'k': k,
        'cl': cl,
        'cluster_labels': cluster_labels,
        'cluster_labels_o': cluster_labels_o,
        'xlabels': xlabels,
        'ylabels': ylabels,
        'X2': X2,
        'Y2': Y2,
        'fig_height': fig_height,
        'weights': weights,
        'ht_var_name': ht_var_name,
        'tau_var_name': tau_var_name,
        'fld': fld,
        'histograms': histograms,
        'histograms_ref': histograms_ref
    }


def _plot_cloud_regimes(plot_data, adf, baseline_mode=True):
    """Main plotting function that handles both baseline and obs-only modes."""
    # Setup
    plt.rcParams.update({"font.size": 14})
    cmap, norm = _create_colormap()
    
    # Create figure
    ncols = 3 if baseline_mode else 2
    fig, ax = plt.subplots(
        figsize=plot_data['figsize'],
        ncols=ncols,
        nrows=plot_data['k'],
        sharex=True, # was "all"
        sharey=True
    )
    fig.subplots_adjust(right=0.88)  # Leave space for colorbar

    # Handle y-axis inversion
    if plot_data['data_product'] != "MISR":
        ax.ravel()[1].invert_yaxis()
    
    # Plot columns
    if baseline_mode:
        _plot_observation_column(ax[:, 0], plot_data, cmap, norm, include_rfo=True)
        _plot_baseline_column(ax[:, 1], plot_data, cmap, norm)
        _plot_test_case_column(ax[:, 2], plot_data, cmap, norm)
    else:
        _plot_observation_column(ax[:, 0], plot_data, cmap, norm, include_rfo=True)
        _plot_test_case_column(ax[:, 1], plot_data, cmap, norm)
    
    # Configure axes and labels
    _configure_axes(ax, plot_data['data_product'])
    _add_figure_labels(fig, ax, plot_data, baseline_mode)
    _add_colorbar(fig, ax.ravel(), cmap, norm)
    
    # Save
    _save_figure(fig, plot_data, adf)
    plt.close()


def _plot_observation_column(ax_col, plot_data, cmap, norm, include_rfo=False):
    """Plot the observation cluster centers."""
    for i in range(plot_data['k']):
        # Plot cluster center
        im = ax_col[i].pcolormesh(
            plot_data['X2'], plot_data['Y2'],
            plot_data['cl'][i].reshape(len(plot_data['xlabels']), len(plot_data['ylabels'])).T,
            norm=norm, cmap=cmap
        )
        
        # Set title with optional RFO
        if include_rfo:
            rfo_map, rfo = _calculate_rfo(plot_data['cluster_labels_o'], i)
            ax_col[i].set_title(f"Observation CR {i+1}, RFO = {np.round(float(rfo), 1)}%")
        else:
            ax_col[i].set_title(f"Observation CR {i+1}")


def _plot_baseline_column(ax_col, plot_data, cmap, norm):
    """Plot the baseline cluster centers with weighted means and RFO."""
    for i in range(plot_data['k']):
        # Calculate RFO
        rfo_map, rfo = _calculate_rfo(cluster_labels_b, i)  # Global variable
        
        # Calculate weighted mean
        wmean = _calculate_weighted_mean_xr(i, plot_data['cluster_labels_o'], plot_data['histograms_ref'])
        # Plot
        im = ax_col[i].pcolormesh(plot_data['X2'], plot_data['Y2'], wmean, norm=norm, cmap=cmap)
        ax_col[i].set_title(f"Baseline Case CR {i+1}, RFO = {np.round(rfo, 1)}%")


def _plot_test_case_column(ax_col, plot_data, cmap, norm):
    """Plot the test case cluster centers with weighted means and RFO."""
    for i in range(plot_data['k']):
        # Calculate RFO
        rfo_map, rfo = _calculate_rfo(plot_data['cluster_labels'], i)
        
        # Calculate weighted mean
        wmean = _calculate_weighted_mean_xr(i, plot_data['cluster_labels'], plot_data['histograms'])
        
        # Plot
        im = ax_col[i].pcolormesh(plot_data['X2'], plot_data['Y2'], wmean, norm=norm, cmap=cmap)
        ax_col[i].set_title(f"Test Case CR {i+1}, RFO = {np.round(rfo, 1)}%")

def _create_colormap():
    """Create standardized colormap and normalization."""
    p = [0, 0.2, 1, 2, 3, 4, 6, 8, 10, 15, 99]
    colors = [
        "white",
        (0.19215686274509805, 0.25098039215686274, 0.5607843137254902),
        (0.23529411764705882, 0.3333333333333333, 0.6313725490196078),
        (0.32941176470588235, 0.5098039215686274, 0.6980392156862745),
        (0.39215686274509803, 0.6, 0.43137254901960786),
        (0.44313725490196076, 0.6588235294117647, 0.21568627450980393),
        (0.4980392156862745, 0.6784313725490196, 0.1843137254901961),
        (0.5725490196078431, 0.7137254901960784, 0.16862745098039217),
        (0.7529411764705882, 0.8117647058823529, 0.2),
        (0.9568627450980393, 0.8980392156862745, 0.1607843137254902),
    ]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(p, cmap.N, clip=True)
    return cmap, norm

def _configure_axes(ax, data_product):
    """Configure axis ticks and labels based on data product."""
    config_functions = {
        "MODIS": _configure_modis_axes,
        "MISR": _configure_misr_axes,
        "ISCCP": _configure_isccp_axes
    }
    
    if data_product in config_functions:
        config_functions[data_product](ax[0, 0])


def _configure_modis_axes(ax):
    """Configure axes for MODIS data."""
    ylabels = [0, 180, 310, 440, 560, 680, 800, 1000]
    xlabels = [0, 0.3, 1.3, 3.6, 9.4, 23, 60, 150]
    
    ax.set_yticks(np.arange(8))
    ax.set_xticks(np.arange(8))
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)


def _configure_misr_axes(ax):
    """Configure axes for MISR data."""
    xlabels = [0.2, 0.8, 2.4, 6.5, 16.2, 41.5, 100]
    ylabels = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 6, 8, 10, 12, 14, 16, 20]
    
    ax.set_yticks(np.arange(0, 16, 2) + 0.5)
    ax.set_yticklabels(ylabels[0::2])
    ax.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7]) - 0.5)
    ax.set_xticklabels(xlabels, fontsize=16)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)


def _configure_isccp_axes(ax):
    """Configure axes for ISCCP data."""
    xlabels = [0, 1.3, 3.6, 9.4, 22.6, 60.4, 450]
    ylabels = [10, 180, 310, 440, 560, 680, 800, 1025]
    
    # Get current ticks
    yticks = ax.get_yticks().tolist()
    xticks = ax.get_xticks().tolist()
    
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)


def _add_figure_labels(fig, ax, plot_data, baseline_mode):
    """Add figure title and axis labels."""
    data = plot_data['data_product']
    ht_var_name = plot_data['ht_var_name']
    fig_height = plot_data['fig_height']
    # bpm: hacky attempt to get coordinate units:
    if ("prs" in ht_var_name):
        ht_label = "Pressure"
        ht_unit = "hPa"
    else:
        ht_label = "Height"
        ht_unit = "m"

    # Determine height or pressure
    # height_or_pressure = "h" if data == "MISR" else "p"

    # Y-label
    x_pos = 0.07 if baseline_mode else 0.05
    fig.supylabel(f"Cloud-top {ht_label} ({ht_unit})", x=x_pos)

    # Title positioning
    fig.suptitle(
        f"{data} Cloud Regimes",
        y=0.92,  # Adjust this value slightly (e.g., 0.93) for perfect placement
        fontsize=18,
        fontweight='bold'
    )
    
    # X-label positioning
    bbox = ax[-1, -1].get_position()  # Use last subplot for positioning
    fig.supxlabel("Optical Depth", y=bbox.p0[1] - (1 / fig_height * 0.5) - 0.007)


def _save_figure(fig, plot_data, adf):
    """Save figure and add to website if requested."""
    data = plot_data['data_product']
    save_path = adf.plot_location[0] + f"/{data}{plot_data['save_suffix']}"
    plt.savefig(save_path)
    
    if adf.create_html:
        if hasattr(adf, 'compare_obs') and adf.compare_obs:
            # For obs comparison mode
            adf.add_website_data(save_path + ".png", plot_data['field'], case_name=None, multi_case=True)
        else:
            # For baseline comparison mode  
            adf.add_website_data(save_path + ".png", plot_data['field'], adf.get_baseline_info("cam_case_name"))

# ---------------------
# Data handling helpers
# ---------------------
def spatial_subset(ds, lat_range, lon_range):
    """Subsets a DataArray or Dataset by latitude and longitude ranges."""
    if lat_range:
        if ds.lat[0] > ds.lat[-1]:
            ds = ds.sel(lat=slice(lat_range[1], lat_range[0]))
        else:
            ds = ds.sel(lat=slice(lat_range[0], lat_range[1]))
    if lon_range:
        if ds.lon[0] > ds.lon[-1]:
            ds = ds.sel(lon=slice(lon_range[1], lon_range[0]))
        else:
            ds = ds.sel(lon=slice(lon_range[0], lon_range[1]))
    return ds

def temporal_subset(ds, time_range):
    """Subsets a DataArray or Dataset by a time range."""
    def is_valid_time(value):
        return value is not None and value != "None" and value != ""
    
    if not time_range or len(time_range) < 2: return ds
    
    start, end = time_range[0], time_range[1]
    start_valid, end_valid = is_valid_time(start), is_valid_time(end)
    
    if not start_valid and not end_valid: return ds
    
    start = start if start_valid else ds.time.min().item()
    end = end if end_valid else ds.time.max().item()
    
    return ds.sel(time=slice(start, end))

def select_valid_tau_height(ds, tau_var_name, ht_var_name):
    """Selects valid (non-negative) tau and height/pressure ranges."""
    ds = ds.sel({tau_var_name: slice(0, None)})
    if ds[ht_var_name][0] > ds[ht_var_name][-1]: # Pressure (decreasing)
        ds = ds.sel({ht_var_name: slice(None, 0)})
    else: # Altitude (increasing)
        ds = ds.sel({ht_var_name: slice(0, None)})
    return ds

def _calculate_weighted_mean_xr(cluster_i, cluster_labels, hists):
    weights = np.cos(np.radians(hists['lat']))
    cluster_data = xr.where(cluster_labels==cluster_i, hists, np.nan)
    dims = [dim for dim in hists.dims if dim in ["ncol","lat","lon","time"]]
    return cluster_data.weighted(weights).mean(dim=dims)


# --------------
# LAND MASK CODE (probably need to simplify and move out of here)
# --------------
def apply_land_ocean_mask(ds, only_ocean_or_land, landfrac_present=None):
    """
    Apply land or ocean mask to dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset with lat/lon coordinates
    only_ocean_or_land : str or False
        "L" for land only, "O" for ocean only, False for no masking
    landfrac_present : bool, optional
        Whether LANDFRAC variable is available. Auto-detected if None.
        
    Returns:
    --------
    ds : xarray.Dataset
        Masked dataset, or None if invalid option
    """
    # No masking requested
    if only_ocean_or_land is False:
        return ds
    
    # Validate input
    if only_ocean_or_land not in ["L", "O"]:
        warnings.warn(f'[ERROR] Invalid option for only_ocean_or_land: {only_ocean_or_land}'
              'Please enter "O" for ocean only, "L" for land only, or set to False for both')
        return None
    
    # Auto-detect LANDFRAC if not specified
    if landfrac_present is None:
        landfrac_present = "LANDFRAC" in ds.data_vars or "LANDFRAC" in ds.coords
    
    # Use LANDFRAC if available
    if landfrac_present:
        land_mask_value = 1 if only_ocean_or_land == "L" else 0
        return ds.where(ds.LANDFRAC == land_mask_value)
    
    # Otherwise use cartopy-based land mask
    land_mask = create_land_mask(ds)
    
    # Make land mask broadcastable with dataset
    land_mask = _make_mask_broadcastable(land_mask, ds)
    
    # Apply mask
    mask_value = 1 if only_ocean_or_land == "L" else 0
    return ds.where(land_mask == mask_value)


def _make_mask_broadcastable(mask, ds):
    """
    Make 2D land mask broadcastable with dataset by adding dimensions.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        2D mask array (lat, lon)
    ds : xarray.Dataset
        Target dataset
        
    Returns:
    --------
    mask : numpy.ndarray
        Broadcastable mask array
    """
    # Add dimensions for any dims that aren't lat/lon
    for i, dim in enumerate(ds.dims):
        if dim not in ("lat", "lon"):
            mask = np.expand_dims(mask, axis=i)
    return mask


def create_land_mask(ds):
    """
    Create land mask using cartopy Natural Earth data.
    Improved version with better performance and cleaner code.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with lat/lon coordinates
        
    Returns:
    --------
    land_mask : numpy.ndarray
        2D array (lat, lon) with 1 for land, 0 for ocean
    """
    from cartopy import feature as cfeature
    from shapely.geometry import Point
    from shapely.prepared import prep
    import numpy as np
    
    #TODO: Replace this with with a regionmask approach (no numba needed)
    # Get land polygons
    land_110m = cfeature.NaturalEarthFeature("physical", "land", "110m")
    land_polygons = [prep(geom) for geom in land_110m.geometries()]
    # Create coordinate arrays
    lats, lons = ds.lat.values, ds.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Flatten coordinates for easier processing
    lon_flat, lat_flat = lon_grid.flatten(), lat_grid.flatten()
    points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
    # Find land points
    land_coords = []
    for polygon in land_polygons:
        land_coords.extend([
            (point.x, point.y) for point in points if polygon.covers(point)
        ])
    # Convert to numpy array for numba processing
    land_array = np.array(land_coords)
    coord_array = np.column_stack([lon_flat, lat_flat])
    # Use numba for fast coordinate matching
    land_mask_flat = _find_land_points(coord_array, land_array)
    # Reshape to original grid
    return land_mask_flat.reshape(len(lats), len(lons))


@njit()
def _find_land_points(coord_array, land_coords):
    """
    Numba-compiled function to quickly identify land points.
    
    Parameters:
    -----------
    coord_array : numpy.ndarray
        Array of (lon, lat) coordinates
    land_coords : numpy.ndarray
        Array of known land coordinates
        
    Returns:
    --------
    mask : numpy.ndarray
        1D mask array with 1 for land, 0 for ocean
    """
    mask = np.zeros(len(coord_array), dtype=np.int32)

    for i in range(len(coord_array)):
        coord = coord_array[i]
        for j in range(len(land_coords)):
            if np.allclose(coord, land_coords[j], atol=1e-10):
                mask[i] = 1
                break
    return mask

#---------------------
# Regridding functions
#---------------------
def make_se_regridder(weight_file, Method="conservative"):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )
    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        # results seem insensitive to this method choice
        # choices are coservative_normed, coservative, and bilinear
        method=Method,
        reuse_weights=True,
        periodic=True,
    )
    return regridder


def regrid_se_data_bilinear(regridder, data_to_regrid, column_dim_name="ncol"):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [
            name
            for name in data_to_regrid.variables
            if column_dim_name in data_to_regrid[name].dims
        ]
        updated = data_to_regrid.copy().update(
            data_to_regrid[vars_with_ncol]
            .transpose(..., "ncol")
            .expand_dims("dummy", axis=-2)
        )
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(..., column_dim_name).expand_dims(
            "dummy", axis=-2
        )
    else:
        warnings.warn(
            f"[ERROR] Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}"
        )
        return None
    regridded = regridder(updated)
    return regridded


