"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

from pathlib import Path
import time
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import cam_diagnostic as cdog
import adf_utils as utils

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

    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

    print(f"{name}: {size / 1e6:.2f} MB")
    return size


def _json_float(v):
    """
    float(v), or None for NaN/inf - json.dump would otherwise write a bare
    NaN token, which is invalid JSON and makes JSON.parse in the D3 page
    fail. (e.g. pandas' std is NaN for a pressure bin holding one sample.)
    The dashboard already skips non-finite levels, so null is safe there.
    """
    v = float(v)
    return v if np.isfinite(v) else None


def _profile_records(p, mean, std):
    """A regime's profile as tidy per-level records: [{p_hPa, mean, std}, ...]."""
    return [
        {"p_hPa": float(p_), "mean": _json_float(m_), "std": _json_float(s_)}
        for p_, m_, s_ in zip(p, mean, std)
    ]


def _bias_records(p, delta):
    """A regime's bias-vs-Obs as tidy per-level records: [{p_hPa, delta}, ...]."""
    return [
        {"p_hPa": float(p_), "delta": _json_float(d_)}
        for p_, d_ in zip(p, delta)
    ]


def _named_profile_entry(p_open, mean_open, std_open, p_strat, mean_strat, std_strat,
                          N_open=None, N_strat=None,
                          delta_open=None, delta_strat=None):
    """
    One named source's open/strat profiles - a CAM case, or a shared
    reference like "Obs"/"ERA5". N_open/N_strat (drop counts) are an
    observation property, so they're only passed for the "Obs" entry.
    delta_open/delta_strat (bias vs Obs, already computed in Python - see
    compute_case_stats) are only passed for actual CAM cases, since Obs/
    ERA5 don't have a bias against themselves.
    """
    entry = {
        "open":  _profile_records(p_open, mean_open, std_open),
        "strat": _profile_records(p_strat, mean_strat, std_strat),
    }
    if N_open is not None:
        entry["N_open"] = N_open
        entry["N_strat"] = N_strat
    if delta_open is not None:
        entry["open_bias"] = _bias_records(p_open, delta_open)
        entry["strat_bias"] = _bias_records(p_strat, delta_strat)
    return entry


def save_named_profile_json(name, campaign, out_dir, p_open, mean_open, std_open,
                             p_strat, mean_strat, std_strat,
                             N_open=None, N_strat=None,
                             delta_open=None, delta_strat=None):
    """
    Write one named profile (a CAM case, or a shared reference like "Obs"/
    "ERA5") to a browser-friendly JSON file, for use outside of Python (e.g. D3).
    Filenames are qualified by campaign since "Obs"/"ERA5" (and possibly
    case names) are reused across campaigns - without this, a second
    campaign's run would silently overwrite the first's per-source files.
    """
    record = {"case": name, **_named_profile_entry(
        p_open, mean_open, std_open, p_strat, mean_strat, std_strat,
        N_open, N_strat, delta_open, delta_strat,
    )}
    out_path = Path(out_dir) / f"sonde_profile_{campaign}_{name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote profile JSON: {out_path}")


def save_dropsonde_dashboard_json(campaign, sonde_var, enough, out_dir, config_name):
    """
    Combine every source into the single-file shape the D3 dashboard expects
    (see lib/website_templates/examples/sample_dropsonde_profile.json):
    "Obs" and "ERA5" are peers of the CAM cases under "cases" - shared
    reference data, not duplicated inside every CAM case.
    """
    # Obs/ERA5 are shared across cases (same observations/reanalysis
    # regardless of which CAM run they're compared against), so grab them
    # from whichever case happens to be first - same convention the Python
    # plotting already uses (cam_ds_0 = enough[list(enough.keys())[0]]).
    first = next(iter(enough.values()))

    cases_json = {
        "Obs": _named_profile_entry(
            first["p_sonde_open"], first["sonde_open_mean"], first["sonde_open_std"],
            first["p_sonde_strat"], first["sonde_strat_mean"], first["sonde_strat_std"],
            first["N_open"], first["N_strat"],
        ),
        "ERA5": _named_profile_entry(
            first["p_era_open"], first["era_open_mean"], first["era_open_std"],
            first["p_era_strat"], first["era_strat_mean"], first["era_strat_std"],
        ),
    }
    for case, d in enough.items():
        cases_json[case] = _named_profile_entry(
            d["p_cam_open"], d["cam_open_mean"], d["cam_open_std"],
            d["p_cam_strat"], d["cam_strat_mean"], d["cam_strat_std"],
            delta_open=d["delta_open"], delta_strat=d["delta_strat"],
        )

    dashboard_json = {"campaign": campaign, "sonde_var": sonde_var, "cases": cases_json}

    out_path = Path(out_dir) / f"dropsonde_profiles_{campaign}_{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"Wrote dropsonde dashboard JSON: {out_path}")
    return out_path


REGIMES = {"open": "Open-Cell", "strat": "Stratocumulus"}


def compute_reference_stats(ds_era5, df_sonde, sonde_var, era_var):
    """
    Obs (sonde) and ERA5 profiles for both regimes. These don't depend on the
    CAM case, so the diagnostic computes and caches them once per campaign.
    """
    ds_era = ds_era5[[era_var]].chunk({"time": -1, "level": -1}).persist()
    ref = {}
    for key, regime in REGIMES.items():
        df_reg, p_era_raw, era_arr = cdog.collocate_profiles_for_regime(
            df_sonde, ds_era, regime, var=era_var,
            lat_name="latitude", lon_name="longitude", lev_name="level",
            to_percent=True, label="ERA5",
        )
        prof_sonde = cdog.sonde_profile(df_reg, sonde_var, bin_hPa=10, to_percent=True)
        ref[f"p_sonde_{key}"] = prof_sonde["p_hPa"].values
        ref[f"sonde_{key}_mean"] = prof_sonde["mean"].values
        ref[f"sonde_{key}_std"] = prof_sonde["std"].values
        (ref[f"p_era_{key}"], ref[f"era_{key}_mean"],
         ref[f"era_{key}_std"]) = cdog.mean_std_from_profiles(p_era_raw, era_arr)
        # Counts of physical drops (RF, drop_num) in this regime
        ref[f"N_{key}"] = df_reg.groupby(["RF", "drop_num"]).ngroups
    return ref


def compute_cam_stats(ds_out, df_sonde, cam_var):
    """One CAM case's profiles at the sondes, for both regimes."""
    ds_cam_sub = ds_out[[cam_var]].chunk({"time": -1, "lev": -1}).persist()
    cam = {}
    for key, regime in REGIMES.items():
        _, p_cam_raw, cam_arr = cdog.collocate_profiles_for_regime(
            df_sonde, ds_cam_sub, regime, var=cam_var,
            lat_name="lat", lon_name="lon", lev_name="lev",
            to_percent=True, label="CAM",
        )
        (cam[f"p_cam_{key}"], cam[f"cam_{key}_mean"],
         cam[f"cam_{key}_std"]) = cdog.mean_std_from_profiles(p_cam_raw, cam_arr)
    return cam


def combine_case_stats(ref, cam):
    """
    One case's full "enough" entry: the shared Obs/ERA5 profiles, the case's
    CAM profiles, and its bias vs Obs.

    The bias (CAM - Obs) interpolates the Obs mean onto the case's own CAM
    pressure grid, the same calculation the Python plotting side does
    (draw_open_bias/draw_strat_bias in lib/cam_diagnostic.py), so the D3
    dashboard reads it instead of recomputing it client-side.
    """
    delta = {}
    for key in REGIMES:
        obs_interp = np.interp(
            cam[f"p_cam_{key}"][::-1], ref[f"p_sonde_{key}"][::-1], ref[f"sonde_{key}_mean"][::-1],
        )[::-1]
        delta[key] = cam[f"cam_{key}_mean"] - obs_interp

    return {
        "p_sonde_open": ref["p_sonde_open"], "sonde_open_mean": ref["sonde_open_mean"],
        "sonde_open_std": ref["sonde_open_std"],
        "p_sonde_strat": ref["p_sonde_strat"], "sonde_strat_mean": ref["sonde_strat_mean"],
        "sonde_strat_std": ref["sonde_strat_std"],
        "p_cam_open": cam["p_cam_open"], "cam_open_mean": cam["cam_open_mean"],
        "cam_open_std": cam["cam_open_std"],
        "p_cam_strat": cam["p_cam_strat"], "cam_strat_mean": cam["cam_strat_mean"],
        "cam_strat_std": cam["cam_strat_std"],
        "p_era_open": ref["p_era_open"], "era_open_mean": ref["era_open_mean"],
        "era_open_std": ref["era_open_std"],
        "p_era_strat": ref["p_era_strat"], "era_strat_mean": ref["era_strat_mean"],
        "era_strat_std": ref["era_strat_std"],
        "delta_open": delta["open"], "delta_strat": delta["strat"],
        "N_open": ref["N_open"], "N_strat": ref["N_strat"],
    }


def compute_case_stats(case, ds_out, ds_era5, df_sonde, sonde_var, cam_var, era_var):
    """
    Collocate model/obs profiles for a single case and return its "enough"
    entry. The diagnostic itself caches the reference and CAM parts
    separately; this computes both, for callers that want one case at once.
    """
    ref = compute_reference_stats(ds_era5, df_sonde, sonde_var, era_var)
    return combine_case_stats(ref, compute_cam_stats(ds_out, df_sonde, cam_var))


# Bump when compute_reference_stats/compute_cam_stats change what they
# compute, so results cached by older code are recomputed (see
# cdog.cached_source). 2: Obs/ERA5 cached once, separately from each case.
# 3: sonde longitudes put in the model grid's convention before collocation.
CACHE_VERSION = 3


def cam_obs_dropsonde_comp(adf, campaign, df_sonde, ds_outs, ds_era5, case_info=None):
    """
    Dropsonde-vs-CAM (and ERA5) RH profiles, by cloud regime.

    Parameters
    ----------
    adf : AdfDiag
        The ADF object.
    campaign : str
        Campaign name.
    df_sonde : pandas.DataFrame
        The dropsonde composite. A hash of its contents is part of every
        cache fingerprint, so rebuilding it with different data recomputes.
    ds_outs : Mapping
        {case nickname: CAM dataset}, only indexed for cases not cached yet.
    ds_era5 : xarray.Dataset
        ERA5 pressure levels, only read when a case is not cached yet.
    case_info : dict, optional
        {case nickname: case details} from inform_model_analysis, used to
        key each case's cache file by its full case name.
    """
    '''img_base = f'{campaign}__RHvprof_diff_coll__{cam_desc}'
    img_name = f'{img_base}.png'
    plot_loc = Path(adf.plot_location[0])
    img_path =  plot_loc / img_name
    if img_path.is_file():
        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                            category="Model Cloud State", plot_loc=str(plot_loc),cam_desc=cam_desc)
    '''
    '''
    img_name = f'{img_base}.png'
    plot_loc = Path(adf.plot_location[0])
    img_path = f"{plot_loc}/{img_name}"

    if len(ds_outs.keys()) > 1:
        img_base = f'{campaign}__RHvprof_diff_coll__multi_CAM'
    else:
        case = list(ds_outs.keys())[0]
        img_base = f'{campaign}__RHvprof_diff_coll__{case}'
    if img_path.is_file():
        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                            category="Model Cloud State", plot_loc=str(plot_loc),cam_desc="")
        return
    '''

    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Dropsonde plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")
    yup = Path(adf.config_file_dict["name"])
    plot_loc = Path(adf.plot_location[0])
    # JSON exports live alongside the dashboard/html files
    # (created already by AdfWeb.__init__, but mkdir here too to be safe):
    website_dir = plot_loc / "website"
    website_dir.mkdir(parents=True, exist_ok=True)
    sonde_var = "rh" # tdry, u_wind, v_wind, rh
    cam_var = "RH" # T_K, RH, U, V
    era_var = "RH" # T_K, RH, U, V

    # Cached per source (cdog.cached_source): the Obs/ERA5 reference once per
    # campaign, and each case's CAM profiles keyed by full case name. Each is
    # checked against a fingerprint of everything its result depends on.
    cache_root = cdog.inform_cache_root(adf, campaign)
    sonde_source = cdog.sonde_cache_source(df_sonde)
    # Kelvin copy, made only if something actually has to be computed:
    sonde_K = []

    def _sonde_K():
        if not sonde_K:
            sonde_K.append(df_sonde.assign(tdry=df_sonde["tdry"] + 273.15,
                                           dp=df_sonde["dp"] + 273.15))
        return sonde_K[0]

    # ERA5 is loaded for the campaign's flight window and box, which every
    # case shares, so any case's entry describes it:
    window = next(iter((case_info or {}).values()), {})
    era5_source = {"era5": "d633000 pressure levels",
                   **{k: window.get(k) for k in ("tmin", "tmax", "lat_slice", "lon_slice")}}
    ref = cdog.cached_source(
        cache_root, "dropsonde", campaign, "Obs_ERA5", {"sonde": sonde_source, **era5_source},
        {"sonde_var": sonde_var, "era_var": era_var, "bin_hPa": 10}, CACHE_VERSION,
        compute=lambda: compute_reference_stats(ds_era5, _sonde_K(), sonde_var, era_var),
        label="Obs and ERA5")

    enough = {}
    for case in ds_outs:
        key, source = cdog.case_cache_source(case_info, case)
        cam = cdog.cached_source(
            cache_root, "dropsonde", campaign, key, source,
            {"cam_var": cam_var, "sonde": sonde_source}, CACHE_VERSION,
            compute=lambda case=case: compute_cam_stats(ds_outs[case], _sonde_K(), cam_var),
            label=case)
        enough[case] = combine_case_stats(ref, cam)

        # This case's own CAM profile (and its bias vs Obs), for the D3
        # dashboard; Obs/ERA5 are shared across cases and written once below.
        stats = enough[case]
        save_named_profile_json(
            case, campaign, website_dir,
            stats["p_cam_open"], stats["cam_open_mean"], stats["cam_open_std"],
            stats["p_cam_strat"], stats["cam_strat_mean"], stats["cam_strat_std"],
            delta_open=stats["delta_open"], delta_strat=stats["delta_strat"],
        )

    # Obs/ERA5 are shared across cases, so write them once here rather than
    # once per case (mirrors the per-case CAM files written above).
    first = next(iter(enough.values()))
    save_named_profile_json(
        "Obs", campaign, website_dir,
        first["p_sonde_open"], first["sonde_open_mean"], first["sonde_open_std"],
        first["p_sonde_strat"], first["sonde_strat_mean"], first["sonde_strat_std"],
        first["N_open"], first["N_strat"],
    )
    save_named_profile_json(
        "ERA5", campaign, website_dir,
        first["p_era_open"], first["era_open_mean"], first["era_open_std"],
        first["p_era_strat"], first["era_strat_mean"], first["era_strat_std"],
    )

    # Combined per-campaign JSON for the D3 dashboard (all cases in `enough`,
    # whichever branch above produced them):
    dashboard_json = save_dropsonde_dashboard_json(campaign, sonde_var, enough, website_dir, yup)
    # So the website builds this campaign's interactive dropsonde page:
    adf.add_inform_page(campaign, "dropsonde", dashboard_json)

    #=========================================
    # Figure 1. 3 panels with CAM, ERA5, Obs
    #=========================================
    #
    # fig = cdog.plot_three_panels_profiles_with_era(
    #     prof_sonde_open, prof_sonde_strat,
    #     p_cam_open,  cam_open_mean,  cam_open_std,
    #     p_cam_strat, cam_strat_mean, cam_strat_std,
    #     p_era_open,  era_open_mean,  era_open_std,
    #     p_era_strat, era_strat_mean, era_strat_std,
    #     N_open, N_strat,
    #     var_label="Relative Humidity (%)",
    #     ylim_hPa=(1000, 500), xlim_hPa=(-10,110),xlim_delta=(-35,20)
    # )
    # plt.savefig('SOCRATES_RHvprof_diff_coll_cam_era5_CAM6_6hrnudgUTV.png',dpi=300,bbox_inches='tight')

    #========================================================================
    # Figure 2. 3 panels with CAM and Obs, difference between cloud regimes
    #========================================================================
    # Skip building/saving plots entirely when the config has create_html
    # turned off - the pkl cache and dashboard JSON above are already
    # written regardless, so this just avoids unused PNGs and figure work.
    if not adf.create_html:
        print("  create_html is False: skipping dropsonde plot generation.")
        return

    plot_loc = Path(adf.plot_location[0])

    # Machine key -> (filename suffix used by plot_three_panels_cam_obs_delta_cam_minus_obs,
    # human-readable label for the dashboard's panel selector).
    panel_suffixes = {
        "open_cell":      ("OpenCell",      "Open-Cell"),
        "stratocumulus":  ("Stratocumulus", "Stratocumulus"),
        "open_cell_bias": ("OpenCellBias",  "Open-Cell Bias"),
        "stratocu_bias":  ("StratoCuBias",  "StratoCu Bias"),
    }

    def _make_and_register_suite(enough_subset, cam_desc, img_base):
        """
        Build one summary (4-panel) figure + its 4 individually-saved
        component panels for `enough_subset`, and register all 5 with the
        website generator under `cam_desc`.
        """
        fig = cdog.plot_three_panels_cam_obs_delta_cam_minus_obs(
                enough_subset,
                var_label="Relative Humidity (%)",
                ylim_hPa=(1000, 500), xlim_hPa=(-10,110), xlim_delta=(-20,35),
                save_indiv_panels=True,
                indiv_plot_loc=plot_loc,
                indiv_img_base=img_base,
        )

        img_name = f'{img_base}.png'
        img_path = plot_loc / img_name

        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                                category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                panel="summary", panel_label="Summary")

        for panel_key, (suffix, panel_label) in panel_suffixes.items():
            panel_img_base = f"{img_base}_{suffix}"
            panel_img_path = plot_loc / f"{panel_img_base}.png"
            adf.add_website_data(str(panel_img_path), panel_img_base, campaign, plot_type="Dropsonde Comparison",
                                    category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                    panel=panel_key, panel_label=panel_label)

    # With more than one case, build the "multi_CAM" ensemble suite first
    # (every case overlaid on the same axes vs Obs/ERA5), so it's the first
    # entry registered for this campaign and appears first in the dashboard's
    # case dropdown, ahead of the individual cases:
    if len(enough) > 1:
        img_base = f'{campaign}__RHvprof_diff_coll__multi_CAM'
        _make_and_register_suite(enough, "multi_CAM", img_base)

    # Every case also gets its own summary + component-panel suite (that
    # case alone vs Obs/ERA5), regardless of how many cases are in this run:
    for case in enough:
        img_base = f'{campaign}__RHvprof_diff_coll__{case}'
        _make_and_register_suite({case: enough[case]}, case, img_base)