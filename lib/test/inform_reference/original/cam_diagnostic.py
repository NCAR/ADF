from scipy.special import erf
import os, xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
import bisect
import glob
import sys
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import dask
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
import datetime
import inform_utils as inform
import cam_era5_functions as cam
# Add parent directory to sys.path
parent_dir = Path('/glade/u/home/patnaude/inform/').resolve().parent.parent
sys.path.append(str(parent_dir))
import dask.array as da
import matplotlib.ticker as mticker
from scipy import stats

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
    
    for (rf, drop), sub in groups:
        if sub.empty:
            continue
    
        # 🔥 drop NaNs in GGLAT / GGLON (top of sonde always has NaNs)
        sub_valid = sub.dropna(subset=[lat_col, lon_col])
        if sub_valid.empty:
            print(f"⚠️ No valid lat/lon for RF={rf}, drop={drop}, skipping")
            continue
    
        # 🔑 pick representative level for location/time
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

    return df_reg, p_cam_ref, cam_arr, p_era_ref, era_arr

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

def plot_three_panels_cam_obs_delta_cam_minus_obs(
    prof_open_sonde, prof_strat_sonde,
    p_cam_open,   cam_open_mean,   cam_open_std,
    p_cam_strat,  cam_strat_mean,  cam_strat_std,
    N_open, N_strat,
    var_label="Relative Humidity (%)",
    ylim_hPa=(1000, 600),
    xlim_hPa=(-10, 100),
    xlim_delta=(-40, 40),
    cam_label="CAM6",
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
    ax.plot(cam_open_mean, p_cam_open, lw=2, ls="-", label=f"{cam_label} Open")
    ax.fill_betweenx(
        p_cam_open,
        cam_open_mean - cam_open_std,
        cam_open_mean + cam_open_std,
        alpha=0.2,
    )

    # Strat CAM
    ax.plot(cam_strat_mean, p_cam_strat, lw=2, ls="--", label=f"{cam_label} StratoCu")
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
            lw=2, ls="-", label="Obs Open")
    ax.fill_betweenx(
        prof_open_sonde["p_hPa"],
        prof_open_sonde["mean"] - prof_open_sonde["std"],
        prof_open_sonde["mean"] + prof_open_sonde["std"],
        alpha=0.2,
    )

    # Strat Obs
    ax.plot(prof_strat_sonde["mean"], prof_strat_sonde["p_hPa"],
            lw=2, ls="--", label="Obs StratoCu")
    ax.fill_betweenx(
        prof_strat_sonde["p_hPa"],
        prof_strat_sonde["mean"] - prof_strat_sonde["std"],
        prof_strat_sonde["mean"] + prof_strat_sonde["std"],
        alpha=0.2,
    )

    ax.set_title("Observed")
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
    ax.plot(delta_open,  p_cam_open,  lw=2, ls="-",  label="Δ(CAM − Obs) Open")
    ax.plot(delta_strat, p_cam_strat, lw=2, ls="--", label="Δ(CAM − Obs) StratoCu")

    ax.set_title("Bias: CAM − Obs")
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
    return fig

def plot_obs_cam_nd_lwc_pdfs(
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
    show=True,
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

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes, {
        "Nd_obs": pdf_Nd_obs,
        "LWC_obs": pdf_LWC_obs,
        "Nd_cam": pdf_Nd_cam,
        "LWC_cam": pdf_LWC_cam,
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
    
def plot_sensitivity_vs_sigmaw_2x2_obs_cam(
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
    return fig, axes, out
