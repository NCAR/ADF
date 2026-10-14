import os
import re
import json
import hashlib
import pickle
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute

# Add parent directory to sys.path
#parent_dir = Path('/glade/u/home/patnaude/inform/').resolve().parent.parent
#sys.path.append(str(parent_dir))
import dask.array as da
from scipy import stats

import adf_utils as utils
import time

def disk_size(obj, name="data"):
    """
    Report an object's size in MB for debug logging, computed in-memory
    (no disk I/O). For a dask-backed xr.DataArray/Dataset, ".nbytes" is
    shape/dtype metadata only, so this does NOT force a compute.
    """
    if isinstance(obj, np.ndarray):
        size = obj.nbytes

    elif isinstance(obj, pd.DataFrame):
        size = obj.memory_usage(deep=True).sum()

    elif isinstance(obj, (xr.DataArray, xr.Dataset)):
        size = obj.nbytes

    elif isinstance(obj, pd.core.series.Series):
        size = obj.memory_usage(deep=True)

    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

    print(f"{name}: {size / 1e6:.2f} MB")
    return size


# =============================================================================
# 0b. SAFE, INCREMENTAL PICKLE CACHING (dropsonde/Nd-LWC/sensitivity scripts)
# =============================================================================
# These per-case compute steps are expensive (dask reductions over full 4D
# CAM fields, or per-dropsonde netCDF collocation) and can run for hours
# across many cases, so a long-running job that gets interrupted (e.g. a
# dropped connection) needs to actually resume from whatever cases already
# finished, rather than silently redoing everything on the next attempt.
# See cached_source() below, which every INFORM diagnostic uses for its
# per-source result cache and which calls these two helpers.

def safe_pickle_load(path):
    """
    Load a pickle cache, tolerating a file left truncated/corrupted by an
    interrupted write. Returns an empty dict instead of raising, so a
    caller just treats a broken cache as "nothing cached yet" and
    recomputes, rather than crashing outright on the next run.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"WARNING: could not load cache '{path}' ({e}); starting fresh.")
        return {}


def atomic_pickle_dump(obj, path):
    """
    Write a pickle cache atomically: to a temp file in the same directory,
    then rename it into place. os.replace() is atomic on POSIX filesystems,
    so an interruption mid-write can only ever leave the temp file behind -
    the real cache at `path` stays either the old, still-valid version or
    the new, complete one, never a half-written file.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)


# =============================================================================
# 0c. PER-SOURCE RESULT CACHE (shared across configs)
# =============================================================================
# Each diagnostic caches one file per source - "Obs" or one CAM case - at
#   {cache root}/{diagnostic}/{campaign}__{source key}.pkl
# where a CAM case's key is its full case name (cam_case_name), not its
# nickname. Each file stores a fingerprint of everything its result depends
# on (case name, history stream and location, time/space window, the
# diagnostic's settings and a version number); a fingerprint that no longer
# matches the current run is recomputed. So configs that share a cache root
# reuse each other's results for the same case, while a renamed nickname,
# a different window or new settings can never pick up a stale result.

def inform_cache_root(adf, campaign):
    """
    Root directory for the INFORM diagnostics' result caches.

    Uses the config's campaigns-section ``inform_cache_dir`` (one path for
    all campaigns, or a list with one per campaign) when set; otherwise
    ``{composited_data}/inform_cache`` for the campaign; otherwise
    ``{plot_location}/inform_cache``.
    """
    try:
        campaigns = adf.campaigns_dict or {}
    except AttributeError:
        campaigns = {}
    names = campaigns.get("name") or []
    idx = names.index(campaign) if campaign in names else None

    def _for_campaign(value):
        if isinstance(value, (list, tuple)):
            return value[idx] if idx is not None and idx < len(value) else None
        return value

    cache_dir = _for_campaign(campaigns.get("inform_cache_dir"))
    if cache_dir:
        return Path(cache_dir)
    composited = _for_campaign(campaigns.get("composited_data"))
    if composited:
        return Path(composited) / "inform_cache"
    return Path(adf.plot_location[0]) / "inform_cache"


def _canonical(obj):
    """JSON-normalize a fingerprint so equal settings always compare equal."""
    def _default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        if isinstance(o, slice):
            return [o.start, o.stop, o.step]
        return str(o)
    return json.loads(json.dumps(obj, sort_keys=True, default=_default))


def case_cache_source(case_info, nickname):
    """
    Return (cache key, source fingerprint) for the CAM case shown as `nickname`.

    `case_info` maps nickname -> {"case_name", "hist_str", "cam_hist_loc",
    "tmin", "tmax", "lat_slice", "lon_slice"}, as built by
    inform_model_analysis. Without an entry for `nickname` the nickname
    itself is the key, so callers that pass no case info still work.
    """
    info = (case_info or {}).get(nickname)
    if not info:
        return nickname, {"case_name": nickname}
    return info["case_name"], dict(info)


def sonde_cache_source(df_sonde):
    """
    Source fingerprint for the dropsonde composite, from its contents (not
    its path), so configs with identical composites share cached results.
    """
    digest = hashlib.sha256(pd.util.hash_pandas_object(df_sonde, index=True).values.tobytes())
    return {"sonde_composite_sha256": digest.hexdigest(), "rows": len(df_sonde)}


def obs_cache_source(All_rf_df):
    """
    Source fingerprint for "Obs" computed from the combined flight table,
    from its contents (like sonde_cache_source), so regenerated flight files
    with new values (e.g. cloud_regime, EIS) recompute the Obs results.

    The hash is kept in the table's attrs, so the diagnostics that each call
    this hash the table once per run. It is tagged with the table's id():
    pandas copies attrs onto filtered/derived frames, which must not reuse it.
    """
    memo = All_rf_df.attrs.get("inform_sha256")
    if memo is not None and memo[0] == id(All_rf_df):
        digest = memo[1]
    else:
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(All_rf_df, index=False).values.tobytes()).hexdigest()
        All_rf_df.attrs["inform_sha256"] = (id(All_rf_df), digest)
    return {"obs": "combined flight table", "flight_table_sha256": digest,
            "rows": len(All_rf_df)}


def cached_source(cache_root, diagnostic, campaign, key, source, settings, version,
                  compute, label=None):
    """
    Return one source's result for one diagnostic, from its cache file when
    the stored fingerprint matches, otherwise from compute() (then cached).

    Parameters
    ----------
    cache_root : str or Path
        From inform_cache_root().
    diagnostic : str
        Subdirectory name for the diagnostic, e.g. "nd_lwc_pdf".
    campaign : str
        Campaign name, part of the file name.
    key : str
        "Obs" or the CAM case's full case name.
    source : dict
        What the source data depends on (from case_cache_source or
        obs_cache_source).
    settings : dict
        The diagnostic's settings that affect the result.
    version : int
        Bump in the diagnostic when its computation changes, so results
        cached by older code are recomputed.
    compute : callable
        Called with no arguments to produce the result on a cache miss.
    label : str, optional
        Name to print instead of `key` (e.g. the case nickname).
    """
    safe_key = re.sub(r"[^\w.+-]", "_", str(key))
    path = Path(cache_root) / diagnostic / f"{campaign}__{safe_key}.pkl"
    fingerprint = _canonical({"version": version, "source": source, "settings": settings})

    entry = safe_pickle_load(path) if path.is_file() else {}
    if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint:
        return entry["stats"]

    reason = "settings or source changed" if entry else "not cached yet"
    print(f"  {diagnostic}: computing {label or key} ({reason}) -> {path}")
    stats = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump({"key": str(key), "label": label, "fingerprint": fingerprint,
                        "stats": stats}, path)
    return stats


# -----------------------------------------------------------------------------
# Per-case CAM data cache
# -----------------------------------------------------------------------------
# Loading a CAM case (open the history files, derive the microphysics and
# state fields, the cloud-controlling factors and the regime masks) is the
# slowest step, and any diagnostic missing that case repeats it. So the case's
# derived fields that the diagnostics read, plus its two regime masks, are
# saved once per case at
#   {cache root}/cam/{campaign}__{case name}.nc
# with the same kind of fingerprint as cached_source (in the file's
# attributes). The regime composites are rebuilt from the masks on load.

# Bump when compute_cam_aerosol_micphys_metrics, load_cam_ccfs or
# subset_cam_by_campaign change what they compute, so saved CAM data from
# older code is rebuilt.
CAM_CACHE_VERSION = 1

# The ds_out fields the INFORM diagnostics read: RH (dropsonde) and the
# Nd/LWC PDF and sensitivity inputs.
CAM_CACHE_VARS = ("RH", "T_K", "cam_lwc", "cam_Nc", "cam_N_UHSAS", "sigma_w", "cam_Nr")


def cam_case_cache_path(cache_root, campaign, key):
    """Where one case's saved CAM data lives (see read/write_cam_case_cache)."""
    safe_key = re.sub(r"[^\w.+-]", "_", str(key))
    return Path(cache_root) / "cam" / f"{campaign}__{safe_key}.nc"


def cam_case_fingerprint(source, variables=CAM_CACHE_VARS):
    """Fingerprint for a case's saved CAM data, from case_cache_source's source."""
    return _canonical({"version": CAM_CACHE_VERSION, "source": source,
                       "settings": {"variables": list(variables)}})


def read_cam_case_cache(path, fingerprint):
    """
    Return (ds_out, masks) from a case's saved CAM data, or None when the file
    is missing, unreadable or was saved with a different fingerprint.

    ds_out holds only the saved fields; masks is {"strat": ..., "open": ...}.
    Both are opened lazily (dask-backed), as freshly derived data would be.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        ds_out = xr.open_dataset(path, chunks={})
        stored = ds_out.attrs.get("inform_fingerprint")
        if stored is None or json.loads(stored) != _canonical(fingerprint):
            ds_out.close()
            return None
        masks = xr.open_dataset(path, group="masks", chunks={})
    except (OSError, ValueError, KeyError) as e:
        print(f"WARNING: could not read saved CAM data '{path}' ({e}); rebuilding it.")
        return None
    return ds_out, {"strat": masks["mask_strat"], "open": masks["mask_open"]}


def write_cam_case_cache(path, ds_out, masks, fingerprint, variables=CAM_CACHE_VARS):
    """
    Save a case's derived fields and regime masks (from subset_cam_by_campaign)
    to `path`, computing them from the history files as it writes.

    Written to a temporary file first and then moved into place, so an
    interrupted run never leaves a partial file for the next run to read.
    Fields keep their precision; compression is lossless.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    out = ds_out[list(variables)]
    out.attrs = {**ds_out.attrs,
                 "inform_fingerprint": json.dumps(_canonical(fingerprint), sort_keys=True)}
    encoding = {v: {"zlib": True, "complevel": 1} for v in variables}
    try:
        out.to_netcdf(tmp_path, mode="w", encoding=encoding)
        xr.Dataset({"mask_strat": masks["strat"], "mask_open": masks["open"]}).to_netcdf(
            tmp_path, mode="a", group="masks")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def comp_from_masks(ds_out, masks):
    """
    Rebuild subset_cam_by_campaign's regime composites from its saved masks:
    the same inner alignment and masking, without recomputing the CCFs.
    """
    mask_strat = masks["strat"].compute()
    mask_open = masks["open"].compute()
    ds_aln, mask_strat, mask_open = xr.align(ds_out, mask_strat, mask_open, join="inner")
    return {
        "strat": ds_aln.where(mask_strat, drop=True),
        "open": ds_aln.where(mask_open, drop=True),
        "masks": {"strat": mask_strat, "open": mask_open},
    }


# =============================================================================
# 1. SONDE HELPERS (BINNED MEAN/STD BY PRESSURE)
# =============================================================================
def _autodetect_pressure_col(df, candidates=("pres","pressure","P","p")):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No pressure column found; tried {candidates}")

def _as_percent(arr, assume_percent_if_max_gt_1=True):
    amax = np.nanmax(arr)
    if assume_percent_if_max_gt_1:
        return arr if (amax > 1.5) else (arr * 100.0)
    else:
        return arr

def sonde_profile(df, sonde_var, pcol=None, bin_hPa=10, to_percent=False):
    """
    Compute binned (by pressure) mean/std sonde profile for a given variable.
    Returns a DataFrame with columns: p_hPa, mean, std (sorted high->low p).
    """
    if pcol is None:
        pcol = _autodetect_pressure_col(df)

    sub = df[[pcol, sonde_var]].dropna()
    if sub.empty:
        raise ValueError("No data for sonde_profile after dropna")

    # round to nearest bin
    pbin = (np.round(sub[pcol].values / bin_hPa) * bin_hPa).astype(int)
    vals = sub[sonde_var].values
    if to_percent:
        vals = _as_percent(vals)

    g = pd.DataFrame({"p_hPa": pbin, "val": vals}).groupby("p_hPa")
    prof = g["val"].agg(["mean","std"]).reset_index()
    prof = prof.sort_values("p_hPa", ascending=False)
    return prof


# =============================================================================
# 2. MODEL TIME / COLUMN COLLLOCATION HELPERS (CAM & ERA5)
# =============================================================================
def _get_nearest_time_index(ds, time_name, target_time):
    """
    Return integer index of ds[time_name] closest to target_time.

    - Works for DatetimeIndex and CFTimeIndex (e.g., 'noleap').
    - Avoids converting CFTimeIndex to pandas.DatetimeIndex, so no warning.
    """
    time_index = ds[time_name].to_index()  # may be DatetimeIndex or CFTimeIndex

    # Case 1: CFTimeIndex (non-standard calendar, e.g. 'noleap')
    if hasattr(time_index, "calendar"):
        # target_time is np.datetime64 or Timestamp → convert to matching cftime object
        ts = pd.Timestamp(target_time)
        cf_cls = type(time_index[0])  # e.g. cftime.DatetimeNoLeap

        target_cf = cf_cls(
            ts.year, ts.month, ts.day,
            ts.hour, ts.minute, ts.second
        )

        # compute absolute time deltas in the cftime space
        diffs = np.array([abs(t - target_cf) for t in time_index])
        idx = int(np.argmin(diffs))
        return idx

    # Case 2: normal pandas DatetimeIndex
    else:
        dt_index = time_index
        ts = pd.Timestamp(target_time)
        diffs = np.abs(dt_index - ts)
        idx = int(np.argmin(diffs))
        return idx

def get_model_column_raw(
    ds,
    var="RH",
    time=None,
    lat=None,
    lon=None,
    time_name="time",
    lat_name="latitude",
    lon_name="longitude",
    lev_name="level",
    to_percent=False,
):
    """
    Extract a single vertical column from a 4D model dataset at (nearest time, nearest lat/lon).

    Returns
    -------
    p_hPa : 1D numpy array of pressure (hPa)
    da    : 1D xarray.DataArray of the variable on the vertical level dimension

    NOTE: da is NOT converted to .values here; we keep it lazy and compute later.
    """
    if time is None or lat is None or lon is None:
        raise ValueError("Must provide time, lat, and lon for model column selection")

    # 1) Nearest time index (CFTime-safe)
    itime = _get_nearest_time_index(ds, time_name, time)

    # 2) Select that time, then nearest lat/lon (no .load here!)
    da = (
        ds[var]
        .isel({time_name: itime})
        .sel({lat_name: lat, lon_name: lon}, method="nearest")
    )  # dims: (lev_name,)

    # 3) Pressure / level coordinate
    if lev_name in ds.coords:
        p = ds[lev_name]
    else:
        for cand in ("plev", "lev", "level", "pressure"):
            if cand in ds.coords:
                lev_name = cand
                p = ds[cand]
                break
        else:
            raise ValueError("Could not find level/pressure coordinate in ds")

    p_vals = p.values
    if float(p_vals.max()) > 2000:
        p_hPa = p_vals / 100.0
    else:
        p_hPa = p_vals

    # percent conversion (lazily)
    # only convert % for RH-like fields
    if to_percent and var in ["RH", "rh", "RHUM", "relative_humidity"]:
        try:
            da_max = da.max()
            if float(da_max) <= 1.5:  # means stored as 0–1
                da = da * 100.0
        except Exception:
            pass

    return p_hPa, da  # DataArray (lazy)


# =============================================================================
# 3. COLLOCATE MODEL PROFILES FOR A GIVEN CLOUD REGIME
# =============================================================================

def collocate_model_profiles_for_regime(
    df_sonde,
    ds_cam,
    ds_era,
    regime_name,
    sonde_var="rh",
    cam_var="RH",
    era_var="RH",
    time_col="Time",
    lat_col="GGLAT",
    lon_col="GGLON",
    drop_col="drop_num",
    flight_col="RF",
    cam_lat_name="lat",
    cam_lon_name="lon",
    cam_lev_name="lev",
    era_lat_name="latitude",
    era_lon_name="longitude",
    era_lev_name="level",
    to_percent_models=True,
):
    df_reg = df_sonde[df_sonde.cloud_regime == regime_name].copy()

    groups = df_reg.groupby([flight_col, drop_col])

    cam_cols = []
    era_cols = []
    p_cam_ref = None
    p_era_ref = None

    # ds_cam = ds_out[[cam_var]].chunk({"time": -1, "lev": -1})
    # ds_era = era5_pl[[era_var]].chunk({"time": -1, "level": -1})
    # ds_cam = ds_cam.persist()
    # ds_era = ds_era.persist()
    t0 = time.perf_counter()
    for (rf, drop), sub in groups:
        if sub.empty:
            continue
    
        # Drop NaNs in GGLAT / GGLON (top of sonde always has NaNs)
        sub_valid = sub.dropna(subset=[lat_col, lon_col])
        if sub_valid.empty:
            print(f"WARNING No valid lat/lon for RF={rf}, drop={drop}, skipping")
            continue
    
        # Set representative level for location/time
        # TODO: have an option to use midpoint of profile instead of surface? - JR
        # Option A: surface (recommended)
        row = sub_valid.iloc[-1]
    
        # Option B: midpoint of profile (less typical for dropsondes)
        # mid_idx = len(sub_valid) // 2
        # row = sub_valid.iloc[mid_idx]
    
        # Extract clean location/time
        t   = np.datetime64(row[time_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
    
        # Debug print
        # print(f"RF={rf}, drop={drop}, using t={t}, lat={lat:.3f}, lon={lon:.3f}")

        # CAM column
        p_cam, da_cam = get_model_column_raw(
            ds_cam,
            var=cam_var,
            time=t,
            lat=lat,
            lon=lon,
            time_name="time",
            lat_name=cam_lat_name,
            lon_name=cam_lon_name,
            lev_name=cam_lev_name,
            to_percent=to_percent_models,
        )

        # ERA5 column
        p_era, da_era = get_model_column_raw(
            ds_era,
            var=era_var,
            time=t,
            lat=lat,
            lon=lon,
            time_name="time",
            lat_name=era_lat_name,
            lon_name=era_lon_name,
            lev_name=era_lev_name,
            to_percent=to_percent_models,
        )

        if p_cam_ref is None:
            p_cam_ref = p_cam
        else:
            if not np.allclose(p_cam_ref, p_cam):
                raise ValueError("CAM pressure levels not consistent across columns")

        if p_era_ref is None:
            p_era_ref = p_era
        else:
            if not np.allclose(p_era_ref, p_era):
                raise ValueError("ERA5 pressure levels not consistent across columns")

        # 👉 compute each column immediately (small graph), instead of concatenating dask objects
        cam_cols.append(da_cam.load().values)
        era_cols.append(da_era.load().values)

    cam_arr = np.vstack(cam_cols)  # (n_drops, n_lev_cam)
    era_arr = np.vstack(era_cols)  # (n_drops, n_lev_era)

    t0 = utils.timer("loaded:  collocate_model_profiles_for_regime", t0)

    return df_reg, p_cam_ref, cam_arr, p_era_ref, era_arr


def collocate_profiles_for_regime(
    df_sonde,
    ds,
    regime_name,
    var="RH",
    lat_name="lat",
    lon_name="lon",
    lev_name="lev",
    time_col="Time",
    lat_col="GGLAT",
    lon_col="GGLON",
    drop_col="drop_num",
    flight_col="RF",
    to_percent=True,
    label="model",
):
    """
    Collocate ONE model dataset with every dropsonde in `regime_name`.

    Same selection and column extraction as collocate_model_profiles_for_regime,
    which does CAM and ERA5 together; this lets the dropsonde diagnostic
    collocate ERA5 once per campaign and each CAM case on its own.

    Returns
    -------
    df_reg : pandas.DataFrame
        The sonde rows in this regime.
    p_ref : numpy.ndarray
        The dataset's pressure levels (hPa).
    arr : numpy.ndarray
        Columns at each sonde, shape (n_drops, n_lev).
    """
    df_reg = df_sonde[df_sonde.cloud_regime == regime_name].copy()

    cols = []
    p_ref = None
    t0 = time.perf_counter()
    for (rf, drop), sub in df_reg.groupby([flight_col, drop_col]):
        if sub.empty:
            continue

        # Drop NaNs in GGLAT / GGLON (top of sonde always has NaNs)
        sub_valid = sub.dropna(subset=[lat_col, lon_col])
        if sub_valid.empty:
            print(f"WARNING No valid lat/lon for RF={rf}, drop={drop}, skipping")
            continue

        # Surface point of the sonde sets the column's location/time
        row = sub_valid.iloc[-1]
        p, da = get_model_column_raw(
            ds,
            var=var,
            time=np.datetime64(row[time_col]),
            lat=float(row[lat_col]),
            lon=float(row[lon_col]),
            time_name="time",
            lat_name=lat_name,
            lon_name=lon_name,
            lev_name=lev_name,
            to_percent=to_percent,
        )

        if p_ref is None:
            p_ref = p
        elif not np.allclose(p_ref, p):
            raise ValueError(f"{label} pressure levels not consistent across columns")

        # compute each column immediately (small graph)
        cols.append(da.load().values)

    utils.timer(f"collocated {label} for {regime_name}", t0)
    return df_reg, p_ref, np.vstack(cols)

# =============================================================================
# 4. MEAN / STD FROM MODEL PROFILE MATRIX
# =============================================================================

def mean_std_from_profiles(p, arr_2d):
    """
    arr_2d: shape (n_profiles, n_lev)
    Returns p_sorted, mean_sorted, std_sorted with p high->low.
    """
    mean = np.nanmean(arr_2d, axis=0)
    std  = np.nanstd(arr_2d,  axis=0)

    sorter = np.argsort(p)[::-1]  # high -> low pressure
    return p[sorter], mean[sorter], std[sorter]

'''def plot_three_panels_cam_obs_delta_cam_minus_obs(
    prof_open_sonde, prof_strat_sonde,
    p_cam_open,   cam_open_mean,   cam_open_std,
    p_cam_strat,  cam_strat_mean,  cam_strat_std,
    N_open, N_strat,
    var_label="Relative Humidity (%)",
    ylim_hPa=(1000, 600),
    xlim_hPa=(-10, 100),
    xlim_delta=(-40, 40),
    cam_label="CAM6",obs="Obs"
):
    """
    3-panel figure:
      (1) CAM only (Open & Strat)
      (2) Obs only (Open & Strat)
      (3) CAM − Obs for each regime separately
    """

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })

    # ============================================================
    # 1. CAM PANEL (Open vs Strat)
    # ============================================================
    ax = axes[0]

    # Open CAM
    ax.plot(cam_open_mean, p_cam_open, lw=2, ls="-", label="Open")
    ax.fill_betweenx(
        p_cam_open,
        cam_open_mean - cam_open_std,
        cam_open_mean + cam_open_std,
        alpha=0.2,
    )

    # Strat CAM
    ax.plot(cam_strat_mean, p_cam_strat, lw=2, ls="--", label="StratoCu")
    ax.fill_betweenx(
        p_cam_strat,
        cam_strat_mean - cam_strat_std,
        cam_strat_mean + cam_strat_std,
        alpha=0.2,
    )

    ax.set_title(f"{cam_label}")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 2. OBS PANEL (Open vs Strat)
    # ============================================================
    ax = axes[1]

    # Open Obs
    ax.plot(prof_open_sonde["mean"], prof_open_sonde["p_hPa"],
            lw=2, ls="-", label="Open")
    ax.fill_betweenx(
        prof_open_sonde["p_hPa"],
        prof_open_sonde["mean"] - prof_open_sonde["std"],
        prof_open_sonde["mean"] + prof_open_sonde["std"],
        alpha=0.2,
    )

    # Strat Obs
    ax.plot(prof_strat_sonde["mean"], prof_strat_sonde["p_hPa"],
            lw=2, ls="--", label="StratoCu")
    ax.fill_betweenx(
        prof_strat_sonde["p_hPa"],
        prof_strat_sonde["mean"] - prof_strat_sonde["std"],
        prof_strat_sonde["mean"] + prof_strat_sonde["std"],
        alpha=0.2,
    )

    #ax.set_title("Observed")
    ax.set_title(obs)
    ax.set_xlabel(var_label)
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 3. Δ(CAM − Obs) FOR OPEN & STRAT
    # ============================================================
    ax = axes[2]

    # --- OPEN Δ(CAM − Obs) ---
    # interp obs onto CAM pressure grid
    obs_open_interp = np.interp(
        p_cam_open[::-1],
        prof_open_sonde["p_hPa"].values[::-1],
        prof_open_sonde["mean"].values[::-1],
    )[::-1]

    delta_open = cam_open_mean - obs_open_interp

    # --- STRAT Δ(CAM − Obs) ---
    obs_strat_interp = np.interp(
        p_cam_strat[::-1],
        prof_strat_sonde["p_hPa"].values[::-1],
        prof_strat_sonde["mean"].values[::-1],
    )[::-1]

    delta_strat = cam_strat_mean - obs_strat_interp

    # Plot deltas
    ax.plot(delta_open,  p_cam_open,  lw=2, ls="-",  label="Open")
    ax.plot(delta_strat, p_cam_strat, lw=2, ls="--", label="StratoCu")
    #ax.plot(delta_open,  p_cam_open,  lw=2, ls="-",  label="Δ(CAM − Obs) Open")
    #ax.plot(delta_strat, p_cam_strat, lw=2, ls="--", label="Δ(CAM − Obs) StratoCu")

    #ax.set_title("Bias: CAM − Obs")
    ax.set_title(f"Bias:\n{cam_label} − {obs}")
    ax.set_xlabel(var_label)
    ax.set_xlim(*xlim_delta)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.legend()

    fig.tight_layout()
    return fig

def plot_three_panels_profiles_with_era(
    prof_open_sonde, prof_strat_sonde,
    p_cam_open,   cam_open_mean,   cam_open_std,
    p_cam_strat,  cam_strat_mean,  cam_strat_std,
    p_era_open,   era_open_mean,   era_open_std,
    p_era_strat,  era_strat_mean,  era_strat_std,
    N_open, N_strat,
    var_label="U-wind (m/s)",
    ylim_hPa=(1000, 600), xlim_hPa=(-20, 20),xlim_delta=(-30,30)
):
    # --- 3 panel setup ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })

    # ============================================================
    # 1. OPEN-CELL PANEL
    # ============================================================
    ax = axes[0]

    # Sonde
    ax.plot(prof_open_sonde["mean"], prof_open_sonde["p_hPa"],
            lw=2, label="Sonde")
    ax.fill_betweenx(
        prof_open_sonde["p_hPa"],
        prof_open_sonde["mean"] - prof_open_sonde["std"],
        prof_open_sonde["mean"] + prof_open_sonde["std"],
        alpha=0.2,
    )

    # CAM
    ax.plot(cam_open_mean, p_cam_open, lw=2, ls="--", label="CAM6")
    ax.fill_betweenx(
        p_cam_open,
        cam_open_mean - cam_open_std,
        cam_open_mean + cam_open_std,
        alpha=0.2,
    )

    # ERA5
    ax.plot(era_open_mean, p_era_open, lw=2, ls=":", label="ERA5")
    ax.fill_betweenx(
        p_era_open,
        era_open_mean - era_open_std,
        era_open_mean + era_open_std,
        alpha=0.2,
    )

    ax.set_title(f"Open-Cell (N = {N_open})")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 2. STRAT PANEL
    # ============================================================
    ax = axes[1]

    # Sonde
    ax.plot(prof_strat_sonde["mean"], prof_strat_sonde["p_hPa"],
            lw=2, label="Sonde")
    ax.fill_betweenx(
        prof_strat_sonde["p_hPa"],
        prof_strat_sonde["mean"] - prof_strat_sonde["std"],
        prof_strat_sonde["mean"] + prof_strat_sonde["std"],
        alpha=0.2,
    )

    # CAM
    ax.plot(cam_strat_mean, p_cam_strat, lw=2, ls="--", label="CAM6")
    ax.fill_betweenx(
        p_cam_strat,
        cam_strat_mean - cam_strat_std,
        cam_strat_mean + cam_strat_std,
        alpha=0.2,
    )

    # ERA5
    ax.plot(era_strat_mean, p_era_strat, lw=2, ls=":", label="ERA5")
    ax.fill_betweenx(
        p_era_strat,
        era_strat_mean - era_strat_std,
        era_strat_mean + era_strat_std,
        alpha=0.2,
    )

    ax.set_title(f"Stratocumulus (N = {N_strat})")
    ax.set_xlabel(var_label)
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 3. Δ STRAT - OPEN DIFFERENCE PANEL
    # ============================================================
    ax = axes[2]

    # Extract as arrays
    p_open  = prof_open_sonde["p_hPa"].values        # sorted high -> low
    m_open  = prof_open_sonde["mean"].values
    p_strat = prof_strat_sonde["p_hPa"].values       # sorted high -> low
    m_strat = prof_strat_sonde["mean"].values
    
    # np.interp requires xp ascending, so work in ascending p,
    # then flip back to descending for plotting
    strat_interp_asc = np.interp(
        p_open[::-1],      # x: Open pressures in ascending order
        p_strat[::-1],     # xp: Strat pressures in ascending order
        m_strat[::-1],     # fp: Strat means on ascending p
    )
    
    # Flip back to descending to match p_open ordering
    strat_interp = strat_interp_asc[::-1]
    
    # Δ(Strat - Open) on the Open pressure grid
    delta_sonde = strat_interp - m_open
    
    # Model deltas (already on matching p grids)
    delta_cam = cam_strat_mean - cam_open_mean
    delta_era = era_strat_mean - era_open_mean
    
    # --- Plot ---
    ax.plot(delta_sonde, p_open, lw=2, label="Sonde Δ")
    ax.plot(delta_cam,   p_cam_open, lw=2, ls="--", label="CAM6 Δ")
    ax.plot(delta_era,   p_era_open, lw=2, ls=":",  label="ERA5 Δ")
    
    ax.set_title("Regime Difference (Strat − Open)")
    ax.set_xlabel(var_label)
    ax.set_xlim(*xlim_delta)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.legend()

    fig.tight_layout()
    return fig'''







def dropsonde_cam_obs_2x2(
    enough,
    var_label="Relative Humidity (%)",
    ylim_hPa=(1000, 600),
    xlim_hPa=(-10, 100),
    xlim_delta=(-40, 40),
    obs="Obs"
):
    """
    3-panel figure:
      (1) CAM only (Open & Strat)
      (2) Obs only (Open & Strat)
      (3) CAM − Obs for each regime separately
    """


    """enough[case] = {"prof_sonde_open":prof_sonde_open, "prof_sonde_strat":prof_sonde_strat,
                  "p_cam_open":p_cam_open, "cam_open_mean":cam_open_mean, "cam_open_std":cam_open_std,
                  "p_cam_strat":p_cam_strat, "cam_strat_mean":cam_strat_mean, "cam_strat_std":cam_strat_std,
                  "N_open":N_open, "N_strat":N_strat}"""
    #if len(enough.keys()) > 1:
    # Legend column width grows with the longest case label
    lens = [len(name) for name in enough]
    label_width = max(0.5, max(lens) / 20)

    
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[label_width, 1, 1, 0.3],
        wspace=0.3,
        hspace=0.4,
    )

    axes = np.empty((2, 4), dtype=object)

    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[0, 2] = fig.add_subplot(gs[0, 2])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[1, 2] = fig.add_subplot(gs[1, 2])

    axes_on = [
        axes[0, 1],
        axes[0, 2],
        axes[1, 1],
        axes[1, 2],
    ]

    for ax in axes_on:
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_hPa)
        ax.grid(True, alpha=0.3)

    open_legend_ax = fig.add_subplot(gs[0,0])
    pos = open_legend_ax.get_position()
    open_legend_ax.set_position([
        pos.x0 - 0.06,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    open_legend_ax.axis("off")

    diff_open_legend_ax = fig.add_subplot(gs[0,3])
    pos = diff_open_legend_ax.get_position()
    diff_open_legend_ax.set_position([
        pos.x0 - 0.03,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    diff_open_legend_ax.axis("off")

    strat_legend_ax = fig.add_subplot(gs[1,0])
    pos = strat_legend_ax.get_position()
    strat_legend_ax.set_position([
        pos.x0 - 0.06,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    strat_legend_ax.axis("off")

    diff_strat_legend_ax = fig.add_subplot(gs[1,3])
    pos = diff_strat_legend_ax.get_position()
    diff_strat_legend_ax.set_position([
        pos.x0 - 0.03,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    diff_strat_legend_ax.axis("off")


    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })

    # Grab the Obs data before looping over the CAM cases
    cam_ds_0 = enough[list(enough.keys())[0]]

    # Open Obs
    axes[0,1].plot(cam_ds_0["sonde_open_mean"], cam_ds_0["p_sonde_open"],
            lw=2, ls="-", label="Obs", color="k")
    axes[0,1].fill_betweenx(
        cam_ds_0["p_sonde_open"],
        cam_ds_0["sonde_open_mean"] - cam_ds_0["sonde_open_std"],
        cam_ds_0["sonde_open_mean"] + cam_ds_0["sonde_open_std"],
        alpha=0.2, color="k"
    )

    # Strat Obs
    axes[1,1].plot(cam_ds_0["sonde_strat_mean"], cam_ds_0["p_sonde_strat"],
            lw=2, ls="-", label="Obs", color="k")
    axes[1,1].fill_betweenx(
        cam_ds_0["p_sonde_strat"],
        cam_ds_0["sonde_strat_mean"] - cam_ds_0["sonde_strat_std"],
        cam_ds_0["sonde_strat_mean"] + cam_ds_0["sonde_strat_std"],
        alpha=0.2, color="k"
    )
    
    for idx, (case, cam_ds) in enumerate(enough.items()):

        # ============================================================
        # Row 0, Column 1 Panel: Open - Obs and CAM case(s)
        # ============================================================
        # Open CAM
        axes[0,1].plot(cam_ds["cam_open_mean"], cam_ds["p_cam_open"], lw=2, ls="--", label=f"{case}")
        if len(enough) == 1:
            axes[0,1].fill_betweenx(
                cam_ds["p_cam_open"],
                cam_ds["cam_open_mean"] - cam_ds["cam_open_std"],
                cam_ds["cam_open_mean"] + cam_ds["cam_open_std"],
                alpha=0.2,
            )

        axes[0,1].set_title("Open-Cell")
        #ax.legend()

        # ============================================================
        # Row 1, Column 1 Panel: Stratocumulus - Obs and CAM case(s)
        # ============================================================
        # Strat CAM
        axes[1,1].plot(cam_ds["cam_strat_mean"], cam_ds["p_cam_strat"], lw=2, ls="--", label=f"{case}")
        if len(enough) == 1:
            axes[1,1].fill_betweenx(
                cam_ds["p_cam_strat"],
                cam_ds["cam_strat_mean"] - cam_ds["cam_strat_std"],
                cam_ds["cam_strat_mean"] + cam_ds["cam_strat_std"],
                alpha=0.2,
            )

        axes[1,1].set_title("Stratocumulus")

        # ============================================================
        # Row 0, Column 2: Δ(CAM − Obs) (Open)
        # ============================================================

        # interp obs onto CAM pressure grid
        obs_open_interp = np.interp(
            cam_ds["p_cam_open"][::-1],
            cam_ds["p_sonde_open"][::-1],
            cam_ds["sonde_open_mean"][::-1],
        )[::-1]
        delta_open = cam_ds["cam_open_mean"] - obs_open_interp
        axes[0,2].plot(delta_open,  cam_ds["p_cam_open"],  lw=2, ls="--",  label=f"Δ(CAM {idx+1} − Obs)")

        axes[0,2].set_title(f"Open-Cell Bias: CAM − {obs}")
        axes[0,2].invert_yaxis()
        axes[0,2].set_ylim(*ylim_hPa)
        axes[0,2].set_xlim(*xlim_delta)
        axes[0,2].axvline(0, color="k", lw=1)

        # ============================================================
        # Row 1, Column 2: Δ(CAM − Obs) (Stratocumulus)
        # ============================================================

        obs_strat_interp = np.interp(
            cam_ds["p_cam_strat"][::-1],
            cam_ds["p_sonde_strat"][::-1],
            cam_ds["sonde_strat_mean"][::-1],
        )[::-1]
        delta_strat = cam_ds["cam_strat_mean"] - obs_strat_interp
        axes[1,2].plot(delta_strat, cam_ds["p_cam_strat"], lw=2, ls="--", label=f"Δ(CAM {idx+1} − Obs)")

        axes[1,2].set_title(f"StratoCu Bias: CAM − {obs}")
        axes[1,2].invert_yaxis()
        axes[1,2].set_ylim(*ylim_hPa)
        axes[1,2].set_xlim(*xlim_delta)
        axes[1,2].axvline(0, color="k", lw=1)

    # Legends
    #--------
    # Open-Cell Legend
    open_handles, open_labels = axes[0,1].get_legend_handles_labels()
    open_legend_ax.legend(
        open_handles,
        open_labels,
        loc="upper left",
        frameon=False,
    )
    diff_open_handles, diff_open_labels = axes[0,2].get_legend_handles_labels()
    diff_open_legend_ax.legend(
        diff_open_handles,
        diff_open_labels,
        loc="upper left",
        frameon=False,
    )

    # Stratocumulus Legend
    strat_handles, strat_labels = axes[1,1].get_legend_handles_labels()
    strat_legend_ax.legend(
        strat_handles,
        strat_labels,
        loc="upper left",
        frameon=False,
    )
    diff_strat_handles, diff_strat_labels = axes[1,2].get_legend_handles_labels()
    diff_strat_legend_ax.legend(
        diff_strat_handles,
        diff_strat_labels,
        loc="upper left",
        frameon=False,
    )

    fig.canvas.draw()
    left = axes[0,1].get_position().x0
    right = axes[0,2].get_position().x1

    xcenter = (left + right) / 2
    fig.suptitle("Dropsonde Vertical Profiles", x=xcenter, y=0.95)
    fig.tight_layout()
    return fig









def plot_three_panels_cam_obs_delta_cam_minus_obs(
    enough,
    var_label="Relative Humidity (%)",
    ylim_hPa=(1000, 600),
    xlim_hPa=(-10, 100),
    xlim_delta=(-40, 40),
    obs="Obs",
    save_indiv_panels=False,
    indiv_plot_loc=None,
    indiv_img_base=None,
    indiv_dpi=300,
):
    """
    2x2 grid figure (each row/col pair sharing an off-axis legend):
      (0,1) Open-Cell:      CAM (per case) vs Obs
      (1,1) Stratocumulus:  CAM (per case) vs Obs
      (0,2) Open-Cell Bias: CAM - Obs
      (1,2) StratoCu Bias:  CAM - Obs

    If `save_indiv_panels` is True, each of the four panels above is also
    redrawn on its own standalone figure (same title/axis labels/limits,
    with an on-panel legend instead of the shared off-axis one) and saved
    as its own PNG under `indiv_plot_loc`, named:
        {indiv_img_base}_OpenCell.png
        {indiv_img_base}_Stratocumulus.png
        {indiv_img_base}_OpenCellBias.png
        {indiv_img_base}_StratoCuBias.png
    """

    """enough[case] = {"prof_sonde_open":prof_sonde_open, "prof_sonde_strat":prof_sonde_strat,
                  "p_cam_open":p_cam_open, "cam_open_mean":cam_open_mean, "cam_open_std":cam_open_std,
                  "p_cam_strat":p_cam_strat, "cam_strat_mean":cam_strat_mean, "cam_strat_std":cam_strat_std,
                  "N_open":N_open, "N_strat":N_strat}"""
    #if len(enough.keys()) > 1:
    # Legend column width grows with the longest case label
    lens = [len(name) for name in enough]
    label_width = max(0.5, max(lens) / 20)

    
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[label_width, 1, 1, 0.3],
        wspace=0.3,
        hspace=0.4,
    )

    axes = np.empty((2, 4), dtype=object)

    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[0, 2] = fig.add_subplot(gs[0, 2])
    axes[1, 1] = fig.add_subplot(gs[1, 1])
    axes[1, 2] = fig.add_subplot(gs[1, 2])

    axes_on = [
        axes[0, 1],
        axes[0, 2],
        axes[1, 1],
        axes[1, 2],
    ]

    for ax in axes_on:
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_hPa)
        ax.grid(True, alpha=0.3)

    open_legend_ax = fig.add_subplot(gs[0,0])
    pos = open_legend_ax.get_position()
    open_legend_ax.set_position([
        pos.x0 - 0.06,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    open_legend_ax.axis("off")

    diff_open_legend_ax = fig.add_subplot(gs[0,3])
    pos = diff_open_legend_ax.get_position()
    diff_open_legend_ax.set_position([
        pos.x0 - 0.03,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    diff_open_legend_ax.axis("off")

    strat_legend_ax = fig.add_subplot(gs[1,0])
    pos = strat_legend_ax.get_position()
    strat_legend_ax.set_position([
        pos.x0 - 0.06,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    strat_legend_ax.axis("off")

    diff_strat_legend_ax = fig.add_subplot(gs[1,3])
    pos = diff_strat_legend_ax.get_position()
    diff_strat_legend_ax.set_position([
        pos.x0 - 0.03,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])
    diff_strat_legend_ax.axis("off")


    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })

    # Grab the Obs data before looping over the CAM cases
    cam_ds_0 = enough[list(enough.keys())[0]]

    # ------------------------------------------------------------------
    # Panel-drawing helpers. Each one fully draws a single panel (data,
    # title, axis labels/limits) onto whatever `ax` it is given, so the
    # exact same code can populate the combined 2x2 figure below and, if
    # requested, a standalone single-axis figure for individual export.
    # ------------------------------------------------------------------
    def draw_open_cell(ax, with_legend=False):
        ax.plot(cam_ds_0["sonde_open_mean"], cam_ds_0["p_sonde_open"],
                lw=2, ls="-", label="Obs", color="k")
        ax.fill_betweenx(
            cam_ds_0["p_sonde_open"],
            cam_ds_0["sonde_open_mean"] - cam_ds_0["sonde_open_std"],
            cam_ds_0["sonde_open_mean"] + cam_ds_0["sonde_open_std"],
            alpha=0.2, color="k"
        )
        for case, cam_ds in enough.items():
            ax.plot(cam_ds["cam_open_mean"], cam_ds["p_cam_open"], lw=2, ls="--", label=f"{case}")
            if len(enough) == 1:
                ax.fill_betweenx(
                    cam_ds["p_cam_open"],
                    cam_ds["cam_open_mean"] - cam_ds["cam_open_std"],
                    cam_ds["cam_open_mean"] + cam_ds["cam_open_std"],
                    alpha=0.2,
                )
        ax.set_title("Open-Cell")
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_hPa)
        ax.grid(True, alpha=0.3)
        if with_legend:
            ax.legend(loc="best", frameon=False)

    def draw_stratocumulus(ax, with_legend=False):
        ax.plot(cam_ds_0["sonde_strat_mean"], cam_ds_0["p_sonde_strat"],
                lw=2, ls="-", label="Obs", color="k")
        ax.fill_betweenx(
            cam_ds_0["p_sonde_strat"],
            cam_ds_0["sonde_strat_mean"] - cam_ds_0["sonde_strat_std"],
            cam_ds_0["sonde_strat_mean"] + cam_ds_0["sonde_strat_std"],
            alpha=0.2, color="k"
        )
        for case, cam_ds in enough.items():
            ax.plot(cam_ds["cam_strat_mean"], cam_ds["p_cam_strat"], lw=2, ls="--", label=f"{case}")
            if len(enough) == 1:
                ax.fill_betweenx(
                    cam_ds["p_cam_strat"],
                    cam_ds["cam_strat_mean"] - cam_ds["cam_strat_std"],
                    cam_ds["cam_strat_mean"] + cam_ds["cam_strat_std"],
                    alpha=0.2,
                )
        ax.set_title("Stratocumulus")
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_hPa)
        ax.grid(True, alpha=0.3)
        if with_legend:
            ax.legend(loc="best", frameon=False)

    def draw_open_bias(ax, with_legend=False):
        for idx, (case, cam_ds) in enumerate(enough.items()):
            # interp obs onto CAM pressure grid
            obs_open_interp = np.interp(
                cam_ds["p_cam_open"][::-1],
                cam_ds["p_sonde_open"][::-1],
                cam_ds["sonde_open_mean"][::-1],
            )[::-1]
            delta_open = cam_ds["cam_open_mean"] - obs_open_interp
            ax.plot(delta_open, cam_ds["p_cam_open"], lw=2, ls="--", label=f"Δ(CAM {idx+1} − Obs)")
        ax.set_title(f"Open-Cell Bias: CAM − {obs}")
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_delta)
        #ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="k", lw=1)
        if with_legend:
            ax.legend(loc="best", frameon=False)

    def draw_strat_bias(ax, with_legend=False):
        for idx, (case, cam_ds) in enumerate(enough.items()):
            obs_strat_interp = np.interp(
                cam_ds["p_cam_strat"][::-1],
                cam_ds["p_sonde_strat"][::-1],
                cam_ds["sonde_strat_mean"][::-1],
            )[::-1]
            delta_strat = cam_ds["cam_strat_mean"] - obs_strat_interp
            ax.plot(delta_strat, cam_ds["p_cam_strat"], lw=2, ls="--", label=f"Δ(CAM {idx+1} − Obs)")
        ax.set_title(f"StratoCu Bias: CAM − {obs}")
        ax.set_xlabel(var_label)
        ax.set_ylabel("Pressure (hPa)")
        ax.set_ylim(*ylim_hPa)
        ax.set_xlim(*xlim_delta)
        #ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="k", lw=1)
        if with_legend:
            ax.legend(loc="best", frameon=False)

    # ------------------------------------------------------------------
    # Draw the four panels onto the combined figure (legends stay in the
    # separate off-axis legend columns, as before).
    # ------------------------------------------------------------------
    draw_open_cell(axes[0, 1])
    draw_stratocumulus(axes[1, 1])
    draw_open_bias(axes[0, 2])
    draw_strat_bias(axes[1, 2])

    # Legends
    #--------
    # Open-Cell Legend
    open_handles, open_labels = axes[0,1].get_legend_handles_labels()
    open_legend_ax.legend(
        open_handles,
        open_labels,
        loc="upper left",
        frameon=False,
    )
    diff_open_handles, diff_open_labels = axes[0,2].get_legend_handles_labels()
    diff_open_legend_ax.legend(
        diff_open_handles,
        diff_open_labels,
        loc="upper left",
        frameon=False,
    )

    # Stratocumulus Legend
    strat_handles, strat_labels = axes[1,1].get_legend_handles_labels()
    strat_legend_ax.legend(
        strat_handles,
        strat_labels,
        loc="upper left",
        frameon=False,
    )
    diff_strat_handles, diff_strat_labels = axes[1,2].get_legend_handles_labels()
    diff_strat_legend_ax.legend(
        diff_strat_handles,
        diff_strat_labels,
        loc="upper left",
        frameon=False,
    )

    fig.canvas.draw()
    left = axes[0,1].get_position().x0
    right = axes[0,2].get_position().x1

    xcenter = (left + right) / 2
    fig.suptitle("Dropsonde Vertical Profiles", x=xcenter, y=0.95)
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Optionally re-draw and save each panel as its own standalone image,
    # using the same helpers so titles/labels/limits stay identical. Each
    # standalone panel gets its own on-plot legend since there's no
    # shared legend column to borrow from outside the combined figure.
    # ------------------------------------------------------------------
    if save_indiv_panels:
        if indiv_plot_loc is None or indiv_img_base is None:
            raise ValueError(
                "save_indiv_panels=True requires both indiv_plot_loc and indiv_img_base."
            )
        plot_loc = Path(indiv_plot_loc)
        panel_drawers = {
            "OpenCell": draw_open_cell,
            "Stratocumulus": draw_stratocumulus,
            "OpenCellBias": draw_open_bias,
            "StratoCuBias": draw_strat_bias,
        }
        for suffix, drawer in panel_drawers.items():
            panel_fig, panel_ax = plt.subplots(figsize=(8, 7))
            drawer(panel_ax, with_legend=True)
            panel_fig.tight_layout()
            panel_fig.savefig(plot_loc / f"{indiv_img_base}_{suffix}.png",
                               dpi=indiv_dpi, bbox_inches="tight")
            plt.close(panel_fig)

    return fig










def plot_three_panels_profiles_with_era(
    p_sonde_open,  sonde_open_mean,  sonde_open_std,
    p_sonde_strat, sonde_strat_mean, sonde_strat_std,
    p_cam_open,   cam_open_mean,   cam_open_std,
    p_cam_strat,  cam_strat_mean,  cam_strat_std,
    p_era_open,   era_open_mean,   era_open_std,
    p_era_strat,  era_strat_mean,  era_strat_std,
    N_open, N_strat,
    var_label="U-wind (m/s)",
    ylim_hPa=(1000, 600), xlim_hPa=(-20, 20),xlim_delta=(-30,30)
):
    # --- 3 panel setup ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
    })

    # ============================================================
    # 1. OPEN-CELL PANEL
    # ============================================================
    ax = axes[0]

    # Sonde
    ax.plot(sonde_open_mean, p_sonde_open,
            lw=2, label="Sonde")
    ax.fill_betweenx(
        p_sonde_open,
        sonde_open_mean - sonde_open_std,
        sonde_open_mean + sonde_open_std,
        alpha=0.2,
    )

    # CAM
    ax.plot(cam_open_mean, p_cam_open, lw=2, ls="--", label="CAM6")
    ax.fill_betweenx(
        p_cam_open,
        cam_open_mean - cam_open_std,
        cam_open_mean + cam_open_std,
        alpha=0.2,
    )

    # ERA5
    ax.plot(era_open_mean, p_era_open, lw=2, ls=":", label="ERA5")
    ax.fill_betweenx(
        p_era_open,
        era_open_mean - era_open_std,
        era_open_mean + era_open_std,
        alpha=0.2,
    )

    ax.set_title(f"Open-Cell (N = {N_open})")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Pressure (hPa)")
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 2. STRAT PANEL
    # ============================================================
    ax = axes[1]

    # Sonde
    ax.plot(sonde_strat_mean, p_sonde_strat,
            lw=2, label="Sonde")
    ax.fill_betweenx(
        p_sonde_strat,
        sonde_strat_mean - sonde_strat_std,
        sonde_strat_mean + sonde_strat_std,
        alpha=0.2,
    )

    # CAM
    ax.plot(cam_strat_mean, p_cam_strat, lw=2, ls="--", label="CAM6")
    ax.fill_betweenx(
        p_cam_strat,
        cam_strat_mean - cam_strat_std,
        cam_strat_mean + cam_strat_std,
        alpha=0.2,
    )

    # ERA5
    ax.plot(era_strat_mean, p_era_strat, lw=2, ls=":", label="ERA5")
    ax.fill_betweenx(
        p_era_strat,
        era_strat_mean - era_strat_std,
        era_strat_mean + era_strat_std,
        alpha=0.2,
    )

    ax.set_title(f"Stratocumulus (N = {N_strat})")
    ax.set_xlabel(var_label)
    ax.set_ylim(*ylim_hPa)
    ax.set_xlim(*xlim_hPa)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ============================================================
    # 3. Δ STRAT - OPEN DIFFERENCE PANEL
    # ============================================================
    ax = axes[2]

    # Extract as arrays
    p_open  = p_sonde_open        # sorted high -> low
    m_open  = sonde_open_mean
    p_strat = p_sonde_strat       # sorted high -> low
    m_strat = sonde_strat_mean
    
    # np.interp requires xp ascending, so work in ascending p,
    # then flip back to descending for plotting
    strat_interp_asc = np.interp(
        p_open[::-1],      # x: Open pressures in ascending order
        p_strat[::-1],     # xp: Strat pressures in ascending order
        m_strat[::-1],     # fp: Strat means on ascending p
    )
    
    # Flip back to descending to match p_open ordering
    strat_interp = strat_interp_asc[::-1]
    
    # Δ(Strat - Open) on the Open pressure grid
    delta_sonde = strat_interp - m_open
    
    # Model deltas (already on matching p grids)
    delta_cam = cam_strat_mean - cam_open_mean
    delta_era = era_strat_mean - era_open_mean
    
    # --- Plot ---
    ax.plot(delta_sonde, p_open, lw=2, label="Sonde Δ")
    ax.plot(delta_cam,   p_cam_open, lw=2, ls="--", label="CAM6 Δ")
    ax.plot(delta_era,   p_era_open, lw=2, ls=":",  label="ERA5 Δ")
    
    ax.set_title("Regime Difference (Strat − Open)")
    ax.set_xlabel(var_label)
    ax.set_xlim(*xlim_delta)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="k", lw=1)
    ax.legend()

    fig.tight_layout()
    return fig


















'''def plot_obs_cam_nd_lwc_pdfs(
    All_rf_df,
    ds_cam_comp,
    labels_of_interest=("In-Cloud Level FT", "In-Cloud Profiles"),
    regimes_of_interest=("Stratocumulus", "Open-Cell"),
    cam_regime_keys=None,
    Nd_bins=None,
    LWC_bins=None,
    obs_Nd_min=10.0,
    obs_LWC_min=0.001,
    cam_Nd_min=10.0,
    cam_LWC_min=0.001,
    T_min_C=0.0,
    figsize=(12, 5),
    savepath=None,
):
    """
    Plot normalized 1D PDFs of Nd and LWC for observations and CAM6 by cloud regime.

    Assumes:
      Obs dataframe has:
        - block_label
        - cloud_regime
        - ATX in deg C
        - CONCD* column for Nd
        - PLWCD* or PLWC column for LWC

      CAM datasets have:
        - T_K in K
        - cam_Nc in cm^-3
        - cam_lwc in g m^-3
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import dask.array as da

    if Nd_bins is None:
        Nd_bins = np.logspace(-1, 3, 15)

    if LWC_bins is None:
        LWC_bins = np.logspace(-3, 1, 15)

    Nd_centers = np.sqrt(Nd_bins[:-1] * Nd_bins[1:])
    LWC_centers = np.sqrt(LWC_bins[:-1] * LWC_bins[1:])

    if cam_regime_keys is None:
        cam_regime_keys = {
            "Stratocumulus": "strat",
            "Open-Cell": "open",
        }

    cam_ds = {
        regime: ds_cam_comp[cam_regime_keys[regime]]
        for regime in regimes_of_interest
    }

    def _find_obs_columns(df):
        concd_col = next((c for c in df.columns if "CONCD" in c), None)
        if concd_col is None:
            raise ValueError("Could not find a 'CONCD*' column in obs dataframe.")

        plwc_col = next((c for c in df.columns if "PLWCD" in c), None)
        if plwc_col is None:
            plwc_col = next((c for c in df.columns if c == "PLWC" or "PLWC" in c), None)
        if plwc_col is None:
            raise ValueError("Could not find a PLWC/PLWCD column in obs dataframe.")

        return concd_col, plwc_col

    def compute_pdf_1d(vals, bins):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            return np.zeros(len(bins) - 1, dtype=float)

        counts, _ = np.histogram(vals, bins=bins)

        if counts.sum() == 0:
            return np.zeros_like(counts, dtype=float)

        return counts.astype(float) / counts.sum()

    def compute_pdf_dask(da_vals, bins):
        arr = da_vals.data
        arr = arr[da.isfinite(arr)]

        hist, _ = da.histogram(arr, bins=bins)
        hist = hist.compute().astype(float)

        if hist.sum() == 0:
            return np.zeros_like(hist, dtype=float)

        return hist / hist.sum()

    concd_col_obs, plwc_col_obs = _find_obs_columns(All_rf_df)

    pdf_Nd_obs = {}
    pdf_LWC_obs = {}
    pdf_Nd_cam = {}
    pdf_LWC_cam = {}

    for regime in regimes_of_interest:

        # --------------------------
        # Observations
        # --------------------------
        df_sub = All_rf_df[
            (All_rf_df["block_label"].isin(labels_of_interest)) &
            (All_rf_df["cloud_regime"] == regime) &
            (All_rf_df[concd_col_obs] > obs_Nd_min) &
            (All_rf_df[plwc_col_obs] > obs_LWC_min) &
            (All_rf_df["ATX"] > T_min_C)
        ]

        pdf_Nd_obs[regime] = compute_pdf_1d(df_sub[concd_col_obs].values, Nd_bins)
        pdf_LWC_obs[regime] = compute_pdf_1d(df_sub[plwc_col_obs].values, LWC_bins)

        # --------------------------
        # CAM6
        # --------------------------
        ds_reg = cam_ds[regime]

        T_C = ds_reg["T_K"] - 273.15

        mask = (
            (T_C > T_min_C) &
            (ds_reg["cam_lwc"] > cam_LWC_min) &
            (ds_reg["cam_Nc"] > cam_Nd_min)
        )

        Nd_cam_da = ds_reg["cam_Nc"].where(mask)
        LWC_cam_da = ds_reg["cam_lwc"].where(mask)

        pdf_Nd_cam[regime] = compute_pdf_dask(Nd_cam_da, Nd_bins)
        pdf_LWC_cam[regime] = compute_pdf_dask(LWC_cam_da, LWC_bins)

    # --------------------------
    # Plotting
    # --------------------------
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    color_map_obs = {
        "Stratocumulus": "#1f77b4",
        "Open-Cell": "#b22222",
    }

    color_map_cam = {
        "Stratocumulus": "#6fa8dc",
        "Open-Cell": "#ff7f50",
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Nd panel
    ax = axes[0]

    for regime in regimes_of_interest:
        ax.plot(
            Nd_centers,
            pdf_Nd_obs[regime],
            label=f"{regime} Obs",
            color=color_map_obs[regime],
            lw=3,
            ls="-",
        )

        ax.plot(
            Nd_centers,
            pdf_Nd_cam[regime],
            label=f"{regime} CAM6",
            color=color_map_cam[regime],
            lw=2,
            ls="--",
        )

    ax.set_xscale("log")
    ax.set_xlim(1, 1e3)
    ax.set_xlabel(r"$N_d$ (cm$^{-3}$)")
    ax.set_ylabel("Probability")
    ax.set_title(rf"$N_d$ (T > {T_min_C:g}°C)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # LWC panel
    ax = axes[1]

    for regime in regimes_of_interest:
        ax.plot(
            LWC_centers,
            pdf_LWC_obs[regime],
            label=f"{regime} Obs",
            color=color_map_obs[regime],
            lw=3,
            ls="-",
        )

        ax.plot(
            LWC_centers,
            pdf_LWC_cam[regime],
            label=f"{regime} CAM6",
            color=color_map_cam[regime],
            lw=2,
            ls="--",
        )

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 10)
    ax.set_xlabel(r"LWC (g m$^{-3}$)")
    ax.set_ylabel("Probability")
    ax.set_title(rf"LWC (T > {T_min_C:g}°C)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    #fig.suptitle('',)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, axes, {
        "Nd_obs": pdf_Nd_obs,
        "LWC_obs": pdf_LWC_obs,
        "Nd_cam": pdf_Nd_cam,
        "LWC_cam": pdf_LWC_cam,
        "Nd_bins": Nd_bins,
        "LWC_bins": LWC_bins,
        "Nd_centers": Nd_centers,
        "LWC_centers": LWC_centers,
    }'''


def find_obs_nd_lwc_columns(df):
    """
    Auto-detect the droplet-number (CONCD*) and liquid-water-content
    (PLWCD*/PLWC) column names in an observation dataframe.
    """
    concd_col = next((c for c in df.columns if "CONCD" in c), None)
    if concd_col is None:
        raise ValueError("Could not find a 'CONCD*' column in obs dataframe.")

    plwc_col = next((c for c in df.columns if "PLWCD" in c), None)
    if plwc_col is None:
        plwc_col = next((c for c in df.columns if c == "PLWC" or "PLWC" in c), None)
    if plwc_col is None:
        raise ValueError("Could not find a PLWC/PLWCD column in obs dataframe.")

    return concd_col, plwc_col


def compute_pdf_1d(vals, bins):
    """Normalized 1D PDF (sums to 1) of a plain numpy/pandas array of values."""
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)

    counts, _ = np.histogram(vals, bins=bins)

    if counts.sum() == 0:
        return np.zeros_like(counts, dtype=float)

    return counts.astype(float) / counts.sum()


def compute_pdf_dask(da_vals, bins):
    """Normalized 1D PDF (sums to 1) of a dask-backed xarray DataArray."""
    import dask.array as da

    arr = da_vals.data
    arr = arr[da.isfinite(arr)]

    hist, _ = da.histogram(arr, bins=bins)
    hist = hist.compute().astype(float)

    if hist.sum() == 0:
        return np.zeros_like(hist, dtype=float)

    return hist / hist.sum()


def plot_obs_cam_nd_lwc_pdfs(
    All_rf_df,
    cam_cases,
    labels_of_interest=("In-Cloud Level FT", "In-Cloud Profiles"),
    regimes_of_interest=("Stratocumulus", "Open-Cell"),
    cam_regime_keys=None,
    Nd_bins=None,
    LWC_bins=None,
    obs_Nd_min=10.0,
    obs_LWC_min=0.001,
    cam_Nd_min=10.0,
    cam_LWC_min=0.001,
    T_min_C=0.0,
    figsize=(15, 10),
    savepath=None,
    save_indiv_panels=False,
    indiv_plot_loc=None,
    indiv_img_base=None,
    indiv_dpi=300,
    pdf_stats=None,
):
    """
    Plot normalized 1D PDFs of Nd and LWC for observations and CAM6 by cloud regime.

    `cam_cases` maps each case name to its composited CAM dataset.  When
    `pdf_stats` is given it holds the PDFs already computed (e.g. cached) by
    compute_obs_pdf_stats/compute_case_pdf_stats in cam_obs_nd_lwc_pdf.py,
    {"Obs": {...}, <case>: {...}}, with the same bins and thresholds as here;
    they are drawn as they are, `cam_cases` only needs the case names and
    `All_rf_df` is not read.

    Assumes:
      Obs dataframe has:
        - block_label
        - cloud_regime
        - ATX in deg C
        - CONCD* column for Nd
        - PLWCD* or PLWC column for LWC

      CAM datasets have:
        - T_K in K
        - cam_Nc in cm^-3
        - cam_lwc in g m^-3
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if Nd_bins is None:
        Nd_bins = np.logspace(-1, 3, 15)

    if LWC_bins is None:
        LWC_bins = np.logspace(-3, 1, 15)

    Nd_centers = np.sqrt(Nd_bins[:-1] * Nd_bins[1:])
    LWC_centers = np.sqrt(LWC_bins[:-1] * LWC_bins[1:])

    if cam_regime_keys is None:
        cam_regime_keys = {
            "Stratocumulus": "strat",
            "Open-Cell": "open",
        }

    """cam_ds = {
        regime: ds_cam_comp[cam_regime_keys[regime]]
        for regime in regimes_of_interest
    }"""

    pdf_Nd_obs = {}
    pdf_LWC_obs = {}
    pdf_Nd_cam = {}   # {cam_desc: {regime: array}}
    pdf_LWC_cam = {}  # {cam_desc: {regime: array}}

    # --------------------------
    # Plotting
    # --------------------------
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })

    color_map_obs = {
        "Stratocumulus": "#1f77b4",
        "Open-Cell": "#b22222",
    }

    color_map_cam = {
        "Stratocumulus": "#6fa8dc",
        "Open-Cell": "#ff7f50",
    }

    """#fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    legend_ax = fig.add_subplot(axes[0,2])
    legend_ax.axis("off")
    print("regimes_of_interest",regimes_of_interest,"\n")
    """

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 1],
        wspace=0.3,
        hspace=0.3,
    )

    axes = np.empty((2, 2), dtype=object)

    """axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(
        gs[0, 1],
        sharey=axes[0,0]
    )

    axes[1, 0] = fig.add_subplot(
        gs[1,0],
        sharex=axes[0,0]
    )

    axes[1,1] = fig.add_subplot(
        gs[1,1],
        sharex=axes[0,1],
        sharey=axes[1,0]
    )"""

    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1]) #, sharey=axes[0, 0]


    legend_ax = fig.add_subplot(gs[0,2])
    pos = legend_ax.get_position()

    legend_ax.set_position([
        pos.x0 - 0.06,   # move left
        pos.y0,
        pos.width,
        pos.height,
    ])

    legend_ax.axis("off")





    """outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[2.0, 0.5],
        wspace=0.05,      # small gap to legend column
    )

    left = outer[0].subgridspec(
        2,
        2,
        wspace=0.4,       # gap between science panels
        hspace=0.25,
    )

    axes[0,0] = fig.add_subplot(left[0,0])
    axes[0,1] = fig.add_subplot(left[0,1])

    axes[1,0] = fig.add_subplot(left[1,0])
    axes[1,1] = fig.add_subplot(left[1,1])

    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")"""


    # --------------------------------------------------------------------
    # Compute every PDF (Obs + each CAM case, both regimes) up front, so
    # the combined-grid panels and any standalone panel exports below both
    # draw from these same precomputed values instead of recomputing the
    # (non-cheap, dask-histogram-backed) CAM PDFs a second time.
    # --------------------------------------------------------------------
    if pdf_stats is not None:
        for regime in regimes_of_interest:
            key = cam_regime_keys[regime]
            pdf_Nd_obs[regime] = pdf_stats["Obs"][f"Nd_{key}"]
            pdf_LWC_obs[regime] = pdf_stats["Obs"][f"LWC_{key}"]
            for cam_desc in cam_cases:
                pdf_Nd_cam.setdefault(cam_desc, {})[regime] = pdf_stats[cam_desc][f"Nd_{key}"]
                pdf_LWC_cam.setdefault(cam_desc, {})[regime] = pdf_stats[cam_desc][f"LWC_{key}"]
    else:
        concd_col_obs, plwc_col_obs = find_obs_nd_lwc_columns(All_rf_df)

        # "Stratocumulus", "Open-Cell"
        for regime in regimes_of_interest:
            # --------------------------
            # Observations
            # --------------------------
            df_sub = All_rf_df[
                (All_rf_df["block_label"].isin(labels_of_interest)) &
                (All_rf_df["cloud_regime"] == regime) &
                (All_rf_df[concd_col_obs] > obs_Nd_min) &
                (All_rf_df[plwc_col_obs] > obs_LWC_min) &
                (All_rf_df["ATX"] > T_min_C)
            ]

            pdf_Nd_obs[regime] = compute_pdf_1d(df_sub[concd_col_obs].values, Nd_bins)
            pdf_LWC_obs[regime] = compute_pdf_1d(df_sub[plwc_col_obs].values, LWC_bins)

            # --------------------------
            # CAM
            # --------------------------
            for cam_desc, ds_cam_comp in cam_cases.items():

                cam_ds = {regime: ds_cam_comp[cam_regime_keys[regime]]}
                ds_reg = cam_ds[regime]

                T_C = ds_reg["T_K"] - 273.15

                mask = (
                    (T_C > T_min_C) &
                    (ds_reg["cam_lwc"] > cam_LWC_min) &
                    (ds_reg["cam_Nc"] > cam_Nd_min)
                )

                Nd_cam_da = ds_reg["cam_Nc"].where(mask)
                LWC_cam_da = ds_reg["cam_lwc"].where(mask)

                pdf_Nd_cam.setdefault(cam_desc, {})[regime] = compute_pdf_dask(Nd_cam_da, Nd_bins)
                pdf_LWC_cam.setdefault(cam_desc, {})[regime] = compute_pdf_dask(LWC_cam_da, LWC_bins)

    # ------------------------------------------------------------------
    # Panel-drawing helper. Fully draws a single panel (data, title, axis
    # labels/limits) onto whatever `ax` it is given, so the exact same
    # code can populate the combined 2x2 figure below and, if requested,
    # a standalone single-axis figure for individual export.
    # ------------------------------------------------------------------
    def draw_panel(ax, regime, var, with_legend=False):
        """var: "Nd" or "LWC"."""
        if var == "Nd":
            centers = Nd_centers
            pdf_obs = pdf_Nd_obs[regime]
            pdf_cam = pdf_Nd_cam
            xlim = (1, 1e3)
            xlabel = r"$N_d$ (cm$^{-3}$)"
            var_title = r"$N_d$"
        else:
            centers = LWC_centers
            pdf_obs = pdf_LWC_obs[regime]
            pdf_cam = pdf_LWC_cam
            xlim = (1e-3, 10)
            xlabel = r"LWC (g m$^{-3}$)"
            var_title = "LWC"

        ax.plot(
            centers,
            pdf_obs,
            label="Obs",
            color="k",
            lw=3,
            ls="-",
        )

        for cam_desc in cam_cases:
            ax.plot(
                centers,
                pdf_cam[cam_desc][regime],
                label=cam_desc,
                lw=2,
                ls="--",
            )

        ax.set_ylabel("Probability")
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.01, 0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(rf"{regime} {var_title} (T > {T_min_C:g}°C)")
        if with_legend:
            ax.legend(loc="best", frameon=False)

    # ------------------------------------------------------------------
    # Draw the four panels onto the combined figure (same axes layout,
    # same draw order per axes, as before: Open-Cell -> row 0,
    # Stratocumulus -> row 1, Nd -> col 0, LWC -> col 1).
    # ------------------------------------------------------------------
    regime_row = {"Open-Cell": 0, "Stratocumulus": 1}
    for regime, row in regime_row.items():
        draw_panel(axes[row, 0], regime, "Nd")
        draw_panel(axes[row, 1], regime, "LWC")

    # legend panel
    handles, labels = axes[0,0].get_legend_handles_labels()

    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
    )

    fig.suptitle('Nd LWC PDF', y=0.95,)
    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Optionally re-draw and save each panel as its own standalone image,
    # using the same helper so titles/labels/limits stay identical. Each
    # standalone panel gets its own on-panel legend since there's no
    # shared legend column to borrow from outside the combined figure.
    # ------------------------------------------------------------------
    if save_indiv_panels:
        if indiv_plot_loc is None or indiv_img_base is None:
            raise ValueError(
                "save_indiv_panels=True requires both indiv_plot_loc and indiv_img_base."
            )
        indiv_plot_loc = Path(indiv_plot_loc)
        panel_specs = {
            "OpenCellNd":       ("Open-Cell", "Nd"),
            "OpenCellLWC":      ("Open-Cell", "LWC"),
            "StratocumulusNd":  ("Stratocumulus", "Nd"),
            "StratocumulusLWC": ("Stratocumulus", "LWC"),
        }
        for suffix, (regime, var) in panel_specs.items():
            panel_fig, panel_ax = plt.subplots(figsize=(7, 6))
            draw_panel(panel_ax, regime, var, with_legend=True)
            panel_fig.tight_layout()
            panel_fig.savefig(indiv_plot_loc / f"{indiv_img_base}_{suffix}.png",
                               dpi=indiv_dpi, bbox_inches="tight")
            plt.close(panel_fig)

    return fig, axes, {
        "Nd_obs": pdf_Nd_obs,
        "LWC_obs": pdf_LWC_obs,
        "Nd_cam": pdf_Nd_cam,    # {cam_desc: {regime: array}}
        "LWC_cam": pdf_LWC_cam,  # {cam_desc: {regime: array}}
        "Nd_bins": Nd_bins,
        "LWC_bins": LWC_bins,
        "Nd_centers": Nd_centers,
        "LWC_centers": LWC_centers,
    }













def _linregress_from_moments(n, mean_x, mean_y, var_x, var_y, cov_xy):
    """
    Return slope, intercept, r, stderr, p for y = a + b x
    using summary moments.
    """
    if n is None or n < 3 or var_x <= 0 or var_y < 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    b = cov_xy / var_x
    a = mean_y - b * mean_x

    # correlation
    denom = np.sqrt(var_x * var_y) if var_y > 0 else np.nan
    r = cov_xy / denom if denom and np.isfinite(denom) and denom > 0 else np.nan

    # stderr of slope (classic OLS)
    if np.isfinite(r):
        stderr = np.sqrt((1.0 - r**2) * var_y / (var_x * (n - 2)))
        tstat = b / stderr if stderr > 0 else np.nan
        p = 2 * stats.t.sf(np.abs(tstat), df=n - 2) if np.isfinite(tstat) else np.nan
    else:
        stderr = np.nan
        p = np.nan

    return b, a, r, stderr, p

def weighted_linear_fit(x, y, w=None):
    """
    Fit y = a + b*x.
    If w is provided, do weighted least squares with weights w.
    Returns slope b and intercept a.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if w is None:
        A = np.vstack([x, np.ones_like(x)]).T
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]
        return b, a

    w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if m.sum() < 2:
        return np.nan, np.nan

    x = x[m]; y = y[m]; w = w[m]

    W = np.diag(w)
    A = np.vstack([x, np.ones_like(x)]).T
    beta = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ y)
    b, a = beta
    return b, a

def r2_of_line(x, y, a, b, w=None):
    """
    R^2 for yhat = a + b*x.
    If w is provided, uses weighted R^2.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 2:
        return np.nan

    x = x[m]; y = y[m]
    yhat = a + b * x

    if w is None:
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
    else:
        w = np.asarray(w, float)[m]
        w = np.clip(w, 0, np.inf)
        ybar = np.sum(w * y) / np.sum(w)
        ss_res = np.sum(w * (y - yhat) ** 2)
        ss_tot = np.sum(w * (y - ybar) ** 2)

    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def cam_binned_alpha_vs_sigmaw(
    ds_cam_comp,
    *,
    cam_Nd_var="cam_Nc",
    cam_CCN_var="cam_N_UHSAS",
    cam_sigmaw_var="sigma_w",
    cam_rain_var="cam_Nr",
    cam_rain_threshold=0.0,
    cam_lwc_var="cam_lwc",        # <-- set to whatever your CAM LWC is called
    lwc_threshold=1e-3,           # kg/kg or g/m3? use appropriate threshold
    Nc_threshold=0.0,             # optional extra in-cloud gate
    cam_temp_var="T_K",
    temp_threshold_K=268.15,
    regimes=("Stratocumulus", "Open-Cell"),
    cam_regime_map=None,
    n_bins=8,
    binning="quantile",
    min_n=50,
    w_mid_method="center",        # "center" fastest; "mean" ok too
):
    """
    Fast CAM alpha(σw) in-cloud only.
    Drizzle split uses cam_rain_var > cam_rain_threshold (after broadcast if needed).
    """

    if cam_regime_map is None:
        cam_regime_map = {"Stratocumulus": "strat", "Open-Cell": "open"}

    rows = []

    for reg in regimes:
        key = cam_regime_map.get(reg, reg)
        ds = ds_cam_comp[key]

        Nd  = ds[cam_Nd_var]
        CCN = ds[cam_CCN_var]
        W   = ds[cam_sigmaw_var]
        T = ds[cam_temp_var] if cam_temp_var in ds else None

        R = ds[cam_rain_var] if (cam_rain_var is not None and cam_rain_var in ds) else None
        LWC = ds[cam_lwc_var] if (cam_lwc_var is not None and cam_lwc_var in ds) else None

        # ---- Broadcast 3D -> 4D to match Nd (robust; no name assumptions) ----
        if "lev" in Nd.dims and "lev" not in W.dims:
            W = W.broadcast_like(Nd)
        if R is not None and "lev" in Nd.dims and "lev" not in R.dims:
            R = R.broadcast_like(Nd)
        if LWC is not None and "lev" in Nd.dims and "lev" not in LWC.dims:
            LWC = LWC.broadcast_like(Nd)
        if T is not None and "lev" in Nd.dims and "lev" not in T.dims:
            T = T.broadcast_like(Nd)
            
        # ---- Dask arrays ----
        Nd_d  = Nd.data
        CCN_d = CCN.data
        W_d   = W.data
        R_d   = R.data if R is not None else None
        LWC_d = LWC.data if LWC is not None else None
        T_d = T.data if T is not None else None
        # ---- In-cloud mask ----
        # Strongly recommended: use LWC (or ql) threshold; else fallback to Nd>0 & CCN>0
        m = da.isfinite(Nd_d) & da.isfinite(CCN_d) & da.isfinite(W_d)

        if LWC_d is not None:
            m = m & da.isfinite(LWC_d) & (LWC_d > lwc_threshold)
        if Nc_threshold is not None and Nc_threshold > 0:
            m = m & (Nd_d > Nc_threshold)
        if T_d is not None:
            m = m & da.isfinite(T_d) & (T_d > temp_threshold_K)
            
        # also require positive for logs
        m = m & (Nd_d > 0) & (CCN_d > 0) & (W_d > 0) 

        # ---- log10 (dask-safe) ----
        x = da.log(CCN_d)
        y = da.log(Nd_d)

        # ---- Flatten ----
        x = x.ravel()
        y = y.ravel()
        w = W_d.ravel()
        m = m.ravel()

        # drizzle flag (0/1)
        if R_d is None:
            dr = da.zeros_like(w, dtype=np.int8)
        else:
            dr = (R_d.ravel() > cam_rain_threshold).astype(np.int8)

        # ---- Filter to valid values first ----
        valid = da.isfinite(x) & da.isfinite(y) & da.isfinite(w) & (dr >= 0)
        x = x[valid]
        y = y[valid]
        w = w[valid]
        dr = dr[valid]
        
        if x.size == 0:
            continue  # skip if nothing valid
        
        # ---- Bin edges on σw ----
        if binning == "quantile":
            qs = np.linspace(0, 100, n_bins + 1)
            # Dask percentile
            w_edges = da.percentile(w, qs).compute()
            w_edges = np.unique(w_edges)
            if len(w_edges) < 3:
                continue  # not enough variation to bin
        
        elif binning == "uniform":
            # Dask-safe min/max, only finite values remain
            wmin = float(w.min().compute())
            wmax = float(w.max().compute())
            if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
                continue  # skip if invalid
            w_edges = np.linspace(wmin, wmax, n_bins + 1)
        
        else:
            raise ValueError("binning must be 'quantile' or 'uniform'")
    
        nb = len(w_edges) - 1
        if nb < 1:
            continue

        b = da.digitize(w, w_edges, right=False) - 1
        b = da.clip(b, 0, nb - 1)

        idx = dr * nb + b  # 0..2*nb-1

        minlength = 2 * nb
        cnt  = da.bincount(idx, minlength=minlength)
        sx   = da.bincount(idx, weights=x, minlength=minlength)
        sy   = da.bincount(idx, weights=y, minlength=minlength)
        sxx  = da.bincount(idx, weights=x*x, minlength=minlength)
        syy  = da.bincount(idx, weights=y*y, minlength=minlength)
        sxy  = da.bincount(idx, weights=x*y, minlength=minlength)
        sw   = da.bincount(idx, weights=w, minlength=minlength) if w_mid_method == "mean" else None

        if sw is None:
            cnt_, sx_, sy_, sxx_, syy_, sxy_ = da.compute(cnt, sx, sy, sxx, syy, sxy)
        else:
            cnt_, sx_, sy_, sxx_, syy_, sxy_, sw_ = da.compute(cnt, sx, sy, sxx, syy, sxy, sw)

        cnt_ = cnt_.reshape(2, nb)
        sx_  = sx_.reshape(2, nb)
        sy_  = sy_.reshape(2, nb)
        sxx_ = sxx_.reshape(2, nb)
        syy_ = syy_.reshape(2, nb)
        sxy_ = sxy_.reshape(2, nb)
        if sw is not None:
            sw_ = sw_.reshape(2, nb)

        if w_mid_method == "center":
            w_mid = 0.5 * (w_edges[:-1] + w_edges[1:])
        else:
            w_mid = np.where(cnt_ > 0, sw_ / cnt_, np.nan)

        # Build rows
        for driz in [0, 1]:
            for kbin in range(nb):
                n = int(cnt_[driz, kbin])
                if n < min_n:
                    continue

                mean_x = sx_[driz, kbin] / n
                mean_y = sy_[driz, kbin] / n
                var_x  = sxx_[driz, kbin] / n - mean_x**2
                var_y  = syy_[driz, kbin] / n - mean_y**2
                cov_xy = sxy_[driz, kbin] / n - mean_x * mean_y

                slope, intercept, r, stderr, p = _linregress_from_moments(
                    n, mean_x, mean_y, var_x, var_y, cov_xy
                )

                rows.append({
                    "source": "CAM",
                    "regime": reg,
                    "drizzle": bool(driz),
                    "w_mid": float(w_mid[kbin]),
                    "n": n,
                    "slope": slope,
                    "stderr": stderr,
                    "r": r,
                    "p": p,
                })

    out = pd.DataFrame(rows)
    return out.sort_values(["regime", "drizzle", "w_mid"]) if len(out) else out

def slope_logNd_logCCN_vs_sigmaw(
    df: pd.DataFrame,
    *,
    Nd_col: str,
    CCN_col: str,
    lwc_col: str,
    sigmaw_col: str,
    regime_col: str,
    Ndriz_col: str,
    regimes=("Stratocumulus", "Open-Cell"),
    n_bins: int = 8,
    binning: str = "quantile",
    min_n: int = 50,
    drizzle_threshold: float = 0.0,
):
    """
    Returns dataframe with:
      regime, drizzle, w_mid, n, slope, stderr, r, p, nuhsas100_med
    slope is from log10(Nd) ~ log10(CCN) within sigma_w bins.
    """
    d = df.copy()

    # numeric + clean
    for c in [Nd_col, CCN_col, sigmaw_col, Ndriz_col]:  # <-- NEW
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=[Nd_col, CCN_col, sigmaw_col, regime_col, Ndriz_col])  # <-- NEW

    # log requires strictly positive Nd/CCN; sigmaw positive
    d = d[(d[lwc_col] > 0.001) & (d[CCN_col] > 0) & (d[sigmaw_col] > 0) & (d['ATX'] > -5)]
    # d = d[(d['PLWCD_LWOI'] > 0.001) & (d[CCN_col] > 0) & (d[sigmaw_col] > 0) & (d['ATX'] > -5)]

    # drizzle flag from numeric Ndriz
    d["_drizzle_flag_"] = d[Ndriz_col] > drizzle_threshold

    rows = []

    for reg in regimes:
        for driz in [False, True, None]:   # None => all data
            if driz is None:
                panel = d[d[regime_col] == reg].copy()
            else:
                panel = d[(d[regime_col] == reg) & (d["_drizzle_flag_"] == driz)].copy()
    
            if len(panel) == 0:
                continue

            w = panel[sigmaw_col].to_numpy()

            # define bins in sigma_w
            if binning == "quantile":
                try:
                    panel["wbin"] = pd.qcut(panel[sigmaw_col], q=n_bins, duplicates="drop")
                except ValueError:
                    continue
            elif binning == "uniform":
                edges = np.linspace(np.nanmin(w), np.nanmax(w), n_bins + 1)
                panel["wbin"] = pd.cut(panel[sigmaw_col], bins=edges, include_lowest=True)
            else:
                raise ValueError("binning must be 'quantile' or 'uniform'")

            for _, g in panel.groupby("wbin", observed=True):
                if len(g) < min_n:
                    continue

                x = np.log(g[CCN_col].to_numpy())
                y = np.log(g[Nd_col].to_numpy())
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < min_n:
                    continue

                res = stats.linregress(x[m], y[m])
                w_mid = float(np.nanmedian(g[sigmaw_col].to_numpy()))

                rows.append({
                    "regime": reg,
                    "drizzle": driz,   # keep None for "all"
                    "w_mid": w_mid,
                    "n": int(m.sum()),
                    "slope": res.slope,
                    "stderr": res.stderr,
                    "r": res.rvalue,
                    "p": res.pvalue,
                })


    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values(["regime", "drizzle", "w_mid"])





'''def plot_sensitivity_vs_sigmaw_2x2_obs_cam(
    df_obs: pd.DataFrame,
    ds_cam_comp,
    *,
    # OBS columns
    Nd_col: str,
    CCN_col: str,
    lwc_col: str,
    sigmaw_col: str,
    regime_col: str,
    Ndriz_col: str,
    drizzle_threshold: float = 0.0,
    # CAM vars
    cam_Nd_var="cam_Nc",
    cam_CCN_var="cam_N_UHSAS",
    cam_sigmaw_var="sigma_w",
    cam_temp_var="T_K",
    cam_temp_threshold_K=273.15,
    cam_regime_map=None,
    cam_bl_masks=None,
    cam_rain_var="cam_Nr",
    cam_rain_threshold=0.0,
    # shared
    regimes=("Stratocumulus", "Open-Cell"),
    drizzle_labels=("Non-drizzling", "Drizzling"),
    n_bins=8,
    binning="quantile",
    min_n=50,
    figsize=(8, 6),
):
    # OBS binned α
    obs_out = slope_logNd_logCCN_vs_sigmaw(
        df_obs,
        Nd_col=Nd_col,
        CCN_col=CCN_col,
        lwc_col=lwc_col,
        sigmaw_col=sigmaw_col,
        regime_col=regime_col,
        Ndriz_col=Ndriz_col,
        regimes=regimes,
        n_bins=n_bins,
        binning=binning,
        min_n=min_n,
        drizzle_threshold=drizzle_threshold,
    ).copy()
    if len(obs_out):
        obs_out["source"] = "OBS"

    # CAM binned α
    cam_out = cam_binned_alpha_vs_sigmaw(
        ds_cam_comp,
        cam_Nd_var=cam_Nd_var,
        cam_CCN_var=cam_CCN_var,
        cam_sigmaw_var=cam_sigmaw_var,
        cam_rain_var=cam_rain_var,
        cam_rain_threshold=cam_rain_threshold,
        temp_threshold_K=cam_temp_threshold_K,
        cam_lwc_var="cam_lwc",      # <-- set correctly
        lwc_threshold=1e-3,         # <-- set correctly for your units
        regimes=regimes,
        cam_regime_map=cam_regime_map,
        n_bins=n_bins,
        binning=binning,
        min_n=min_n,
    )

    out = pd.concat([obs_out, cam_out], ignore_index=True) if len(cam_out) else obs_out

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    # styling
    style = {
        "OBS": dict(marker="o", linestyle="none", alpha=0.9),
        "CAM": dict(marker="s", linestyle="none", alpha=0.9),
    }

    for i, reg in enumerate(regimes):
        for j, driz in enumerate([False, True]):
            ax = axes[i, j]
            ax.axhline(0, linewidth=1)
            ax.set_title(f"{reg} | {drizzle_labels[j]}")
            # --- ALWAYS define these first (prevents UnboundLocalError) ---
            sub_obs = out[(out["source"] == "OBS") & (out["regime"] == reg) & (out["drizzle"] == driz)]
            sub_cam = out[(out["source"] == "CAM") & (out["regime"] == reg) & (out["drizzle"] == driz)]

            # --- add best-fit lines for OBS and CAM (weighted by n) ---
            # OBS fit
            if len(sub_obs) >= 3:
                xfit = sub_obs["w_mid"].to_numpy()
                yfit = sub_obs["slope"].to_numpy()
                wfit = sub_obs["n"].to_numpy()
            
                b_obs, a_obs = weighted_linear_fit(xfit, yfit, w=wfit)  # y = a + b x
                r2_obs = r2_of_line(xfit, yfit, a_obs, b_obs, w=wfit)
            
                if np.isfinite(b_obs):
                    xx = np.linspace(np.nanmin(xfit), np.nanmax(xfit), 100)
                    ax.plot(xx, a_obs + b_obs * xx, linewidth=2, label=None,linestyle='-')
            
                if np.isfinite(r2_obs):
                    # OBS text (lower)
                    ax.text(
                        0.02, 0.06, rf"OBS: m={b_obs:.2g}, $R^2$={r2_obs:.2f}",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=9
                    )
                    
            # CAM fit
            if len(sub_cam) >= 3:
                xfit = sub_cam["w_mid"].to_numpy()
                yfit = sub_cam["slope"].to_numpy()
                wfit = sub_cam["n"].to_numpy()
            
                b_cam, a_cam = weighted_linear_fit(xfit, yfit, w=wfit)
                r2_cam = r2_of_line(xfit, yfit, a_cam, b_cam, w=wfit)
            
                if np.isfinite(b_cam):
                    xx = np.linspace(np.nanmin(xfit), np.nanmax(xfit), 100)
                    ax.plot(xx, a_cam + b_cam * xx, linewidth=2, label=None,linestyle='-')
            
                if np.isfinite(r2_cam):
                    # CAM text (just above OBS)
                    ax.text(
                        0.02, 0.14,
                        rf"CAM: m={b_cam:.2g}, $R^2$={r2_cam:.2f}",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=9
                    )
            # OBS (drizzle split)
            sub_obs = out[(out["source"] == "OBS") & (out["regime"] == reg) & (out["drizzle"] == driz)]
            if len(sub_obs):
                # ax.errorbar(
                #     sub_obs["w_mid"], sub_obs["slope"], yerr=sub_obs["stderr"],
                #     fmt="none", ecolor="k", alpha=0.6
                # )
                obs_style = dict(style["OBS"])
                obs_style.pop("linestyle", None)  # scatter doesn't support linestyle
                ax.scatter(
                    sub_obs["w_mid"], sub_obs["slope"],
                    s=45,
                    label="OBS" if (i == 0 and j == 0) else None,
                    **obs_style
                )
            
            # CAM (drizzle split)
            sub_cam = out[(out["source"] == "CAM") & (out["regime"] == reg) & (out["drizzle"] == driz)]
            if len(sub_cam):
                # ax.errorbar(
                #     sub_cam["w_mid"], sub_cam["slope"], yerr=sub_cam["stderr"],
                #     fmt="none", ecolor="k", alpha=0.6
                # )
                cam_style = dict(style["CAM"])
                cam_style.pop("linestyle", None)  # safe even if not present
                ax.scatter(
                    sub_cam["w_mid"], sub_cam["slope"],
                    s=45,
                    label="CAM" if (i == 0 and j == 0) else None,
                    **cam_style
                )


    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\sigma(w)$ (m/s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\frac{d\log(N_d)}{d\log(\mathrm{CCN})}$")

    axes[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig, axes, out'''





def plot_sensitivity_vs_sigmaw_2x2_obs_cam(
    sources,
    *,
    regimes=("Stratocumulus", "Open-Cell"),
    drizzle_labels=("Non-drizzling", "Drizzling"),
    figsize=(18, 8),
    save_indiv_panels=False,
    indiv_plot_loc=None,
    indiv_img_base=None,
    indiv_dpi=300,
):
    """
    `sources` is a dict of precomputed per-source stats tables - one
    entry for "Obs" (from slope_logNd_logCCN_vs_sigmaw) and one per CAM
    case (from cam_binned_alpha_vs_sigmaw), each already carrying
    "source"/"case"/"regime"/"drizzle"/"w_mid"/"n"/"slope" columns. See
    compute_obs_sensitivity_stats/compute_case_sensitivity_stats in
    sensitivity_vs_sigmaw_2x2_obs_cam.py, which compute and pickle-cache
    these once per case rather than recomputing them on every plot call.
    """

    # =========================
    # Split the precomputed sources back into Obs vs CAM cases
    # =========================

    REF_NAME = "Obs"
    obs_out = sources.get(REF_NAME, pd.DataFrame())
    cam_outs = {name: df for name, df in sources.items() if name != REF_NAME}

    non_empty_cam = [df for df in cam_outs.values() if len(df)]
    cam_all = pd.concat(non_empty_cam, ignore_index=True) if non_empty_cam else pd.DataFrame()

    non_empty_all = [df for df in (obs_out, cam_all) if len(df)]
    out = pd.concat(non_empty_all, ignore_index=True) if non_empty_all else pd.DataFrame()


    # =========================
    # FIGURE LAYOUT
    # =========================

    from matplotlib.gridspec import GridSpec

    # Legend column width grows with the longest case label
    lens = [len(name) for name in cam_outs]
    label_width = max(0.5, max(lens) / 20) if lens else 0.5

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[1, 0.35, 1, 0.35, label_width],
        wspace=0.15,
    )

    plot_axes = np.empty((2,2), dtype=object)
    stat_axes = np.empty((2,2), dtype=object)

    plot_axes[0,0] = fig.add_subplot(gs[0,0])
    stat_axes[0,0] = fig.add_subplot(gs[0,1])

    plot_axes[0,1] = fig.add_subplot(gs[0,2])
    stat_axes[0,1] = fig.add_subplot(gs[0,3])

    plot_axes[1,0] = fig.add_subplot(gs[1,0])
    stat_axes[1,0] = fig.add_subplot(gs[1,1])

    plot_axes[1,1] = fig.add_subplot(gs[1,2])
    stat_axes[1,1] = fig.add_subplot(gs[1,3])

    legend_ax = fig.add_subplot(gs[:,4])
    pos = legend_ax.get_position()

    legend_ax.set_position([
        pos.x0 - 0.02,   # move left
        pos.y0 + 0.01,
        pos.width,
        pos.height,
    ])
    

    for ax in plot_axes.flatten():
        ax.set_xlim(-0.05, 1.75)
        ax.set_ylim(-1, 1)
    for ax in stat_axes.flatten():
        ax.axis("off")

    # ------------------------------------------------------------------
    # Panel-drawing helper. Fully draws one (plot_ax, stat_ax) pair - the
    # fitted line/scatter for Obs + every CAM case, plus the adjacent
    # slope/R^2 text column - so the exact same code can populate the
    # combined 2x2 grid below and, if requested, a standalone panel
    # export. `label_lines` controls whether this call's plotted lines
    # carry a legend label at all (the combined grid only labels once,
    # from the (0,0) panel, to avoid duplicate legend entries; a
    # standalone panel needs its own labels to build its own legend).
    # ------------------------------------------------------------------
    def draw_panel(plot_ax, stat_ax, reg, driz, label_lines=False, with_legend=False):

        drizzle_label = drizzle_labels[int(driz)]

        plot_ax.axhline(
            0,
            linewidth=1
        )

        plot_ax.set_title(
            f"{reg} | {drizzle_label}"
        )

        stats = []

        # ---------------------
        # OBS
        # ---------------------

        sub_obs = out[
            (out.source=="OBS")
            &
            (out.regime==reg)
            &
            (out.drizzle==driz)
        ]

        if len(sub_obs):

            xfit = sub_obs.w_mid.values
            yfit = sub_obs.slope.values
            wfit = sub_obs.n.values

            b,a = weighted_linear_fit(
                xfit,
                yfit,
                w=wfit
            )

            r2 = r2_of_line(
                xfit,
                yfit,
                a,
                b,
                w=wfit
            )


            xx = np.linspace(
                np.nanmin(xfit),
                np.nanmax(xfit),
                100
            )

            plot_ax.plot(
                xx,
                a+b*xx,
                linewidth=2,
                color="k"
            )

            plot_ax.scatter(
                sub_obs.w_mid,
                sub_obs.slope,
                s=45,
                color="k",
                label="Obs" if label_lines else None
            )

            stats.append(
                dict(
                    marker="o",
                    color="k",
                    label="Obs",
                    slope=b,
                    r2=r2,
                )
            )

        # ---------------------
        # CAM CASES
        # ---------------------
        for case_name, cam_df in cam_outs.items():

            sub_cam = cam_df[
                (cam_df.regime==reg)
                &
                (cam_df.drizzle==driz)
            ]

            if len(sub_cam)==0:
                continue

            xfit = sub_cam.w_mid.values
            yfit = sub_cam.slope.values
            wfit = sub_cam.n.values

            b,a = weighted_linear_fit(
                xfit,
                yfit,
                w=wfit
            )

            r2 = r2_of_line(
                xfit,
                yfit,
                a,
                b,
                w=wfit
            )

            xx = np.linspace(
                np.nanmin(xfit),
                np.nanmax(xfit),
                100
            )

            line, = plot_ax.plot(
                xx,
                a+b*xx,
                linewidth=1.5,
                alpha=0.7
            )

            color = line.get_color()

            plot_ax.scatter(
                sub_cam.w_mid,
                sub_cam.slope,
                s=40,
                marker="s",
                alpha=0.7,
                color=color,
                label=case_name if label_lines else None
            )

            stats.append(
                dict(
                    marker="s",
                    color=color,
                    label=case_name,
                    slope=b,
                    r2=r2,
                )
            )

        # ---------------------
        # Slope / R^2 text column
        # ---------------------
        stat_ax.axis("off")

        y = 0.95
        for item in stats:
            stat_ax.scatter(
                -0.1,
                y,
                marker=item["marker"],
                color=item["color"],
                s=40,
                transform=stat_ax.transAxes,
                clip_on=False,
            )

            stat_ax.text(
                -0.05,
                y,
                f"m={item['slope']:.2f}  $R^2$={item['r2']:.2f}",
                va="center",
                fontsize=8,
                transform=stat_ax.transAxes,
                clip_on=False,
            )

            y -= 0.08

        if with_legend:
            plot_ax.legend(loc="best", frameon=False)

    # =========================
    # Draw the four panels onto the combined figure (same layout, same
    # per-axes draw order as before: only the (0,0) panel's lines carry
    # legend labels, matching the original single shared legend column).
    # =========================
    legend_ax.axis("off")

    for i, reg in enumerate(regimes):
        for j, driz in enumerate([False, True]):
            draw_panel(
                plot_axes[i,j], stat_axes[i,j], reg, driz,
                label_lines=(i == 0 and j == 0),
            )

    # labels
    for ax in plot_axes[1, :]:
        ax.set_xlabel(
            r"$\sigma(w)$ (m/s)"
        )

    for ax in plot_axes[:, 0]:
        ax.set_ylabel(
            r"$\frac{d\log(N_d)}{d\log(\mathrm{CCN})}$"
        )


    # legend panel

    handles, labels = plot_axes[0,0].get_legend_handles_labels()

    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        #frameon=False,
    )

    fig.suptitle("Sensitivity vs Sigma w")
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Optionally re-draw and save each of the four (plot + stat) panel
    # pairs as its own standalone image. Each standalone panel gets its
    # own on-panel legend (label_lines=True) since there's no shared
    # legend column to borrow from outside the combined figure, and both
    # axis labels (the combined grid only labels its bottom row/left
    # column, sharing axes across the grid).
    # ------------------------------------------------------------------
    if save_indiv_panels:
        if indiv_plot_loc is None or indiv_img_base is None:
            raise ValueError(
                "save_indiv_panels=True requires both indiv_plot_loc and indiv_img_base."
            )
        indiv_plot_loc = Path(indiv_plot_loc)
        panel_specs = {
            "StratocumulusNonDrizzling": ("Stratocumulus", False),
            "StratocumulusDrizzling":    ("Stratocumulus", True),
            "OpenCellNonDrizzling":      ("Open-Cell", False),
            "OpenCellDrizzling":         ("Open-Cell", True),
        }
        for suffix, (reg, driz) in panel_specs.items():
            panel_fig = plt.figure(figsize=(9, 6))
            panel_gs = GridSpec(
                1, 2,
                figure=panel_fig,
                width_ratios=[1, 0.35],
                wspace=0.15,
            )
            panel_ax = panel_fig.add_subplot(panel_gs[0,0])
            panel_stat_ax = panel_fig.add_subplot(panel_gs[0,1])
            panel_ax.set_xlim(-0.05, 1.75)
            panel_ax.set_ylim(-1, 1)

            draw_panel(
                panel_ax, panel_stat_ax, reg, driz,
                label_lines=True, with_legend=True,
            )

            panel_ax.set_xlabel(r"$\sigma(w)$ (m/s)")
            panel_ax.set_ylabel(r"$\frac{d\log(N_d)}{d\log(\mathrm{CCN})}$")

            panel_fig.tight_layout()
            panel_fig.savefig(indiv_plot_loc / f"{indiv_img_base}_{suffix}.png",
                               dpi=indiv_dpi, bbox_inches="tight")
            plt.close(panel_fig)

    # Also return the underlying per-(source, regime, drizzle, w_mid-bin)
    # stats table ("source"/"case"/"regime"/"drizzle"/"w_mid"/"n"/"slope"),
    # so callers can export the exact same numbers plotted here to JSON
    # without recomputing them (see sensitivity_vs_sigmaw_2x2_obs_cam.py).
    return fig, out



























def plot_alpha_sigmaw_regime_comparison_2x2(
        df_obs: pd.DataFrame,
        #ds_cam_comp,
        cam_cases,
        *,
        # OBS columns
        Nd_col: str,
        CCN_col: str,
        lwc_col: str,
        sigmaw_col: str,
        regime_col: str,
        Ndriz_col: str,
        drizzle_threshold: float = 0.0,
        # CAM vars
        cam_Nd_var="cam_Nc",
        cam_CCN_var="cam_N_UHSAS",
        cam_sigmaw_var="sigma_w",
        cam_temp_var="T_K",
        cam_temp_threshold_K=273.15,
        cam_regime_map=None,
        cam_bl_masks=None,
        cam_rain_var="cam_Nr",
        cam_rain_threshold=0.0,
        # shared
        regimes=("Stratocumulus", "Open-Cell"),
        drizzle_labels=("Non-drizzling", "Drizzling"),
        n_bins=8,
        binning="quantile",
        min_n=50,
        figsize=(10, 8),
        sources=None,
    ):
    """
    Compare observed and modeled

        alpha = dlog(Nd)/dlog(CCN)

    as a function of sigma_w in four cloud regimes.

    Parameters
    ----------
    obs_out : DataFrame
        Output from slope_logNd_logCCN_vs_sigmaw()

    cam_outs : dict
        {
            "Control": cam_df,
            "NudgeUVT 6h": cam_df,
            "NudgeUVT 24h": cam_df,
            ...
        }
    sources : dict, optional
        Precomputed tables, {"Obs": obs_df, <case>: cam_df, ...}, as cached by
        sensitivity_vs_sigmaw_2x2_obs_cam.load_sensitivity_sources; when given,
        `df_obs` is not read and `cam_cases` only needs the case names.
    """

    if sources is not None:
        # Tables already computed (and cached) by the sensitivity-vs-sigma_w
        # diagnostic, with the same settings: nothing is recomputed and
        # cam_cases only needs the case names.
        obs_out = sources["Obs"]
        cam_outs = {case_name: sources[case_name] for case_name in cam_cases}
    else:
        # OBS binned α
        obs_out = slope_logNd_logCCN_vs_sigmaw(
            df_obs,
            Nd_col=Nd_col,
            CCN_col=CCN_col,
            lwc_col=lwc_col,
            sigmaw_col=sigmaw_col,
            regime_col=regime_col,
            Ndriz_col=Ndriz_col,
            regimes=regimes,
            n_bins=n_bins,
            binning=binning,
            min_n=min_n,
            drizzle_threshold=drizzle_threshold,
        ).copy()
        if len(obs_out):
            obs_out["source"] = "OBS"

        # CAM binned α
        """cam_out = cam_binned_alpha_vs_sigmaw(
            ds_cam_comp,
            cam_Nd_var=cam_Nd_var,
            cam_CCN_var=cam_CCN_var,
            cam_sigmaw_var=cam_sigmaw_var,
            cam_rain_var=cam_rain_var,
            cam_rain_threshold=cam_rain_threshold,
            temp_threshold_K=cam_temp_threshold_K,
            cam_lwc_var="cam_lwc",      # <-- set correctly
            lwc_threshold=1e-3,         # <-- set correctly for your units
            regimes=regimes,
            cam_regime_map=cam_regime_map,
            n_bins=n_bins,
            binning=binning,
            min_n=min_n,
        )"""

        """cam_outs = {}

        for case_name, ds_case in cam_cases.items():

            cam_outs[case_name] = cam_binned_alpha_vs_sigmaw(
                ds_case,
                cam_Nd_var=cam_Nd_var,
                cam_CCN_var=cam_CCN_var,
                cam_sigmaw_var=cam_sigmaw_var,
                cam_rain_var=cam_rain_var,
                cam_rain_threshold=cam_rain_threshold,
                temp_threshold_K=cam_temp_threshold_K,
                cam_lwc_var="cam_lwc",
                lwc_threshold=1e-3,
                regimes=regimes,
                cam_regime_map=cam_regime_map,
                n_bins=n_bins,
                binning=binning,
                min_n=min_n,
            )"""
    
        cam_outs = {}

        for case_name, ds_cam_comp in cam_cases.items():

            print(f"Processing {case_name}")

            cam_out = cam_binned_alpha_vs_sigmaw(
                ds_cam_comp,
                cam_Nd_var=cam_Nd_var,
                cam_CCN_var=cam_CCN_var,
                cam_sigmaw_var=cam_sigmaw_var,
                cam_rain_var=cam_rain_var,
                cam_rain_threshold=cam_rain_threshold,
                temp_threshold_K=cam_temp_threshold_K,
                cam_lwc_var="cam_lwc",
                lwc_threshold=1e-3,
                regimes=regimes,
                cam_regime_map=cam_regime_map,
                n_bins=n_bins,
                binning=binning,
                min_n=min_n,
            )

            cam_out["case"] = case_name

            cam_outs[case_name] = cam_out

    #out = pd.concat([obs_out, cam_out], ignore_index=True) if len(cam_out) else obs_out

    cam_all = pd.concat(
        [df.assign(case=name)
        for name, df in cam_outs.items()
        if len(df)],
        ignore_index=True,
    )
    print("cam_outs.keys()",cam_outs.keys(),"\n")
    obs_all = obs_out.copy()
    obs_all["case"] = "OBS"

    out = pd.concat(
        [obs_all, cam_all],
        ignore_index=True,
    )


    """fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )"""

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 8))

    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.6],
        wspace=0.25,
    )

    axes = np.empty((2, 2), dtype=object)

    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1], sharey=axes[0, 0])
    axes[1, 0] = fig.add_subplot(gs[1, 0], sharex=axes[0, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1], sharex=axes[0, 1], sharey=axes[1, 0])

    # empty legend axis
    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis("off")

    for i, reg in enumerate(regimes):
        for j, driz in enumerate([False, True]):

            ax = axes[i, j]

            ax.axhline(0, color="0.7", lw=1)

            ax.set_title(
                f"{reg} | {drizzle_labels[j]}"
            )

            #
            # OBS
            #
            sub_obs = obs_out[
                (obs_out["regime"] == reg)
                & (obs_out["drizzle"] == driz)
            ]

            if len(sub_obs):

                ax.errorbar(
                    sub_obs["w_mid"],
                    sub_obs["slope"],
                    yerr=sub_obs["stderr"],
                    fmt="o-",
                    lw=3,
                    ms=6,
                    capsize=3,
                    label="OBS" if (i == 0 and j == 0) else None,
                )

            #
            # CAM CASES
            #
            for case_name, cam_df in cam_outs.items():

                sub_cam = cam_df[
                    (cam_df["regime"] == reg)
                    & (cam_df["drizzle"] == driz)
                ]

                if not len(sub_cam):
                    continue

                ax.errorbar(
                    sub_cam["w_mid"],
                    sub_cam["slope"],
                    yerr=sub_cam["stderr"],
                    fmt="-o",
                    lw=1.5,
                    ms=4,
                    alpha=0.7,
                    label=case_name if (i == 0 and j == 0) else None,
                )
            """sub_cam = cam_out[
                (cam_out["regime"] == reg)
                & (cam_out["drizzle"] == driz)
            ]
            if len(sub_cam):

                ax.errorbar(
                    sub_cam["w_mid"],
                    sub_cam["slope"],
                    yerr=sub_cam["stderr"],
                    fmt="-o",
                    lw=2,
                    ms=5,
                    alpha=0.8,
                    label="CAM" if (i == 0 and j == 0) else None,
                )"""
    
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\sigma_w$ (m s$^{-1}$)")

    for ax in axes[:, 0]:
        ax.set_ylabel(
            r"$d\log(N_d)/d\log(CCN)$"
        )

    """axes[0, 0].legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False,
    )"""

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()
    return fig, axes, out