"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

import os, xarray as xr
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import cam_diagnostic as cdog


# Regime/drizzle -> the JSON key convention used by both the sample data in
# lib/website_templates/examples/sample_sensitivity_vs_sigmaw.json and
# template_sensitivity_vs_sigmaw.html, matching the "open"/"strat" style
# already used by the dropsonde/Nd-LWC dashboard JSON.
_REGIME_KEYS = {"Open-Cell": "open", "Stratocumulus": "strat"}
_DRIZZLE_KEYS = {False: "nondrizzle", True: "drizzle"}


def _json_float(v):
    """
    float(v), or None for NaN/inf - json.dump would otherwise write a bare
    NaN token, which is invalid JSON and makes JSON.parse in the D3 page
    fail. The dashboard drops non-finite records, so null is safe there.
    """
    v = float(v)
    return v if np.isfinite(v) else None


def _sensitivity_records(panel_df):
    """A (regime, drizzle) panel's binned points: [{w_mid, slope, n}, ...], sorted by w_mid."""
    return [
        {"w_mid": _json_float(row.w_mid), "slope": _json_float(row.slope), "n": int(row.n)}
        for row in panel_df.sort_values("w_mid").itertuples()
    ]


def _sensitivity_panel_entry(panel_df):
    """
    One (regime, drizzle) panel's binned points plus the same weighted
    linear fit (slope/intercept/r2) that draw_panel() computes for the
    matplotlib figure (see plot_sensitivity_vs_sigmaw_2x2_obs_cam in
    lib/cam_diagnostic.py) - computed once there, reused here rather than
    redone, so the JSON always matches what's plotted.
    """
    if not len(panel_df):
        return {"points": [], "fit": {"slope": None, "intercept": None, "r2": None}}

    xfit = panel_df["w_mid"].to_numpy()
    yfit = panel_df["slope"].to_numpy()
    wfit = panel_df["n"].to_numpy()
    b, a = cdog.weighted_linear_fit(xfit, yfit, w=wfit)
    r2 = cdog.r2_of_line(xfit, yfit, a, b, w=wfit)

    return {
        "points": _sensitivity_records(panel_df),
        "fit": {
            "slope": None if b is None else _json_float(b),
            "intercept": None if a is None else _json_float(a),
            "r2": None if r2 is None else _json_float(r2),
        },
    }


def _named_sensitivity_entry(out_df):
    """
    One named source's (Obs, or one CAM case) 4-panel sensitivity data,
    keyed by regime then drizzle state.
    """
    entry = {}
    for regime, regime_key in _REGIME_KEYS.items():
        entry[regime_key] = {}
        for driz, driz_key in _DRIZZLE_KEYS.items():
            panel_df = out_df[(out_df["regime"] == regime) & (out_df["drizzle"] == driz)]
            entry[regime_key][driz_key] = _sensitivity_panel_entry(panel_df)
    return entry


def save_named_sensitivity_json(name, campaign, out_dir, out_df):
    """
    Write one named source's (Obs, or one CAM case) sensitivity-vs-sigmaw
    data to a browser-friendly JSON file, for use outside of Python (e.g. D3).
    """
    record = {"case": name, **_named_sensitivity_entry(out_df)}
    out_path = Path(out_dir) / f"sensitivity_vs_sigmaw_{campaign}_{name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote sensitivity JSON: {out_path}")


def save_sensitivity_dashboard_json(campaign, sources, out_dir, config_name):
    """
    Combine every source into the single-file shape the D3 dashboard
    expects (see lib/website_templates/examples/sample_sensitivity_vs_sigmaw.json):
    "Obs" is a peer of the CAM cases under "cases", not duplicated inside
    every CAM case.

    The "dashboard" in the name keeps it distinct from the per-source files
    (sensitivity_vs_sigmaw_{campaign}_{name}.json): a case nickname equal to
    the config name would otherwise overwrite this file.
    """
    dashboard_json = {
        "campaign": campaign,
        "cases": {
            name: _named_sensitivity_entry(out_df)
            for name, out_df in sources.items()
        },
    }

    out_path = Path(out_dir) / f"sensitivity_vs_sigmaw_dashboard_{campaign}_{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"Wrote sensitivity dashboard JSON: {out_path}")
    return out_path


def compute_obs_sensitivity_stats(All_rf_df, Nd_col, CCN_col, lwc_col, sigmaw_col,
                                   regime_col, Ndriz_col, drizzle_threshold,
                                   n_bins, binning, min_n):
    """
    Compute the observed sensitivity-vs-sigmaw stats - case-independent,
    since these come only from All_rf_df (the aircraft observations), not
    from any particular CAM case.
    """
    obs_out = cdog.slope_logNd_logCCN_vs_sigmaw(
        All_rf_df,
        Nd_col=Nd_col,
        CCN_col=CCN_col,
        lwc_col=lwc_col,
        sigmaw_col=sigmaw_col,
        regime_col=regime_col,
        Ndriz_col=Ndriz_col,
        n_bins=n_bins,
        binning=binning,
        min_n=min_n,
        drizzle_threshold=drizzle_threshold,
    ).copy()

    if len(obs_out):
        obs_out["source"] = "OBS"
        obs_out["case"] = "OBS"

    return obs_out


def compute_case_sensitivity_stats(case, ds_cam_comp, cam_Nd_var, cam_CCN_var, cam_sigmaw_var,
                                    cam_rain_var, cam_rain_threshold, cam_temp_threshold_K,
                                    cam_regime_map, n_bins, binning, min_n):
    """
    Compute one CAM case's sensitivity-vs-sigmaw stats.
    """
    cam_out = cdog.cam_binned_alpha_vs_sigmaw(
        ds_cam_comp,
        cam_Nd_var=cam_Nd_var,
        cam_CCN_var=cam_CCN_var,
        cam_sigmaw_var=cam_sigmaw_var,
        cam_rain_var=cam_rain_var,
        cam_rain_threshold=cam_rain_threshold,
        temp_threshold_K=cam_temp_threshold_K,
        cam_lwc_var="cam_lwc",
        lwc_threshold=1e-3,
        cam_regime_map=cam_regime_map,
        n_bins=n_bins,
        binning=binning,
        min_n=min_n,
    )

    cam_out["source"] = "CAM"
    cam_out["case"] = case

    return cam_out


# Settings shared by the sensitivity and the alpha-vs-sigma_w regime
# comparison diagnostics, which both draw from the one cache below.
COMPUTE_KWARGS = dict(
    Nd_col="CONCD_RWIO",
    lwc_col="PLWCD_RWIO",
    CCN_col="Nccn_UH_CVIU",
    sigmaw_col="Sigma_w",
    regime_col="cloud_regime",
    Ndriz_col="Ndriz_2DC",
    drizzle_threshold=0.0,
    cam_temp_threshold_K=268.15,
    cam_Nd_var="cam_Nc",
    cam_CCN_var="cam_N_UHSAS",
    cam_sigmaw_var="sigma_w",
    cam_rain_var="cam_Nr",
    cam_rain_threshold=0.0,
    cam_regime_map={"Stratocumulus":"strat", "Open-Cell":"open"},
    n_bins=10,
    min_n=50,
    binning="quantile",
)


# Bump when compute_obs_sensitivity_stats/compute_case_sensitivity_stats
# change what they compute, so results cached by older code are recomputed
# (see cdog.cached_source).
CACHE_VERSION = 1


def load_sensitivity_sources(adf, All_rf_df, ds_cams, campaign, case_info=None):
    """
    Return the per-source sensitivity tables, {"Obs": df, <case>: df, ...},
    from the cache, computing (and caching) only what is missing.

    `ds_cams` is only indexed for cases not already cached, so with a warm
    cache no CAM data is read; likewise `All_rf_df` is only used when "Obs"
    is missing. `case_info` ({case nickname: case details} from
    inform_model_analysis) keys each case's cache file by its full case name.
    """
    obs_kwargs = {k: COMPUTE_KWARGS[k] for k in (
        "Nd_col", "CCN_col", "lwc_col", "sigmaw_col", "regime_col", "Ndriz_col",
        "drizzle_threshold", "n_bins", "binning", "min_n",
    )}
    cam_kwargs = {k: COMPUTE_KWARGS[k] for k in (
        "cam_Nd_var", "cam_CCN_var", "cam_sigmaw_var", "cam_rain_var", "cam_rain_threshold",
        "cam_temp_threshold_K", "cam_regime_map", "n_bins", "binning", "min_n",
    )}

    # One cache file per source ("Obs" and each case, keyed by full case
    # name), checked against a fingerprint of everything the result depends
    # on (cdog.cached_source):
    cache_root = cdog.inform_cache_root(adf, campaign)

    enough_sens = {"Obs": cdog.cached_source(
        cache_root, "sensitivity_vs_sigmaw", campaign, "Obs", cdog.obs_cache_source(All_rf_df),
        obs_kwargs, CACHE_VERSION,
        compute=lambda: compute_obs_sensitivity_stats(All_rf_df, **obs_kwargs))}
    for case in ds_cams:
        key, source = cdog.case_cache_source(case_info, case)
        enough_sens[case] = cdog.cached_source(
            cache_root, "sensitivity_vs_sigmaw", campaign, key, source, cam_kwargs, CACHE_VERSION,
            compute=lambda case=case: compute_case_sensitivity_stats(case, ds_cams[case], **cam_kwargs),
            label=case)
        # The table's "case" column holds the nickname it was computed under,
        # which another config sharing this cache may have named differently:
        if len(enough_sens[case]) and "case" in enough_sens[case]:
            enough_sens[case] = enough_sens[case].assign(case=case)

    return enough_sens


#cloud regimes
def sensitivity_vs_sigmaw_2x2_obs_cam(adf, All_rf_df, ds_cams, campaign, case_info=None):
    # Example usage of the microphysics processes plotting function
    # Adjust the variable names and datasets as needed

    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Sigma w sensitivity plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    plot_loc = Path(adf.plot_location[0])
    yup = Path(adf.config_file_dict["name"])
    # JSON exports live alongside the dashboard/html files
    # (created already by AdfWeb.__init__, but mkdir here too to be safe):
    website_dir = plot_loc / "website"
    website_dir.mkdir(parents=True, exist_ok=True)

    enough_sens = load_sensitivity_sources(adf, All_rf_df, ds_cams, campaign, case_info)

    # JSON exports: one file per named source (Obs + each CAM case), plus a
    # single combined file for the D3 dashboard.
    for name, out_df in enough_sens.items():
        if len(out_df):
            save_named_sensitivity_json(name, campaign, website_dir, out_df)
    dashboard_json = save_sensitivity_dashboard_json(campaign, enough_sens, website_dir, yup)
    # So the website builds this campaign's interactive sensitivity page:
    adf.add_inform_page(campaign, "sensitivity_vs_sigmaw", dashboard_json)

    # Skip plot generation entirely when the config has create_html turned
    # off - the pkl cache and JSON exports above are already written
    # regardless, so this just avoids unused PNGs and figure work.
    if not adf.create_html:
        print("  create_html is False: skipping sensitivity plot generation.")
        return

    # Machine key -> (filename suffix used by plot_sensitivity_vs_sigmaw_2x2_obs_cam,
    # human-readable label for the dashboard's panel selector).
    panel_suffixes = {
        "stratocumulus_nondrizzling": ("StratocumulusNonDrizzling", "Stratocumulus | Non-drizzling"),
        "stratocumulus_drizzling":    ("StratocumulusDrizzling",    "Stratocumulus | Drizzling"),
        "open_cell_nondrizzling":     ("OpenCellNonDrizzling",      "Open-Cell | Non-drizzling"),
        "open_cell_drizzling":        ("OpenCellDrizzling",         "Open-Cell | Drizzling"),
    }

    def _make_and_register_suite(sources_subset, cam_desc, img_base):
        """
        Build one summary (2x2) figure + its 4 individually-saved
        component panels from the cached/precomputed `sources_subset`
        stats tables, and register all 5 with the website generator
        under `cam_desc`.
        """
        fig, out = cdog.plot_sensitivity_vs_sigmaw_2x2_obs_cam(
                sources=sources_subset,
                save_indiv_panels=True,
                indiv_plot_loc=plot_loc,
                indiv_img_base=img_base,
        )

        img_name = f'{img_base}.png'
        img_path = plot_loc / img_name

        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Sensitivity vs Sigma W",
                                category="Model Micro State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                panel="summary", panel_label="Summary")

        for panel_key, (suffix, panel_label) in panel_suffixes.items():
            panel_img_base = f"{img_base}_{suffix}"
            panel_img_path = plot_loc / f"{panel_img_base}.png"
            adf.add_website_data(str(panel_img_path), panel_img_base, campaign, plot_type="Sensitivity vs Sigma W",
                                    category="Model Micro State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                    panel=panel_key, panel_label=panel_label)

    # With more than one case, build the "multi_CAM" ensemble suite first
    # (every case overlaid on the same axes vs Obs), so it's the first
    # entry registered for this campaign and appears first in the
    # dashboard's case dropdown, ahead of the individual cases:
    if len(ds_cams) > 1:
        img_base = f'{campaign}__sensitivity_vs_sigmaw__multi_CAM'
        _make_and_register_suite(enough_sens, "multi_CAM", img_base)

    # Every case also gets its own summary + component-panel suite (that
    # case alone vs Obs), regardless of how many cases are in this run:
    for case in ds_cams:
        img_base = f'{campaign}__sensitivity_vs_sigmaw__{case}'
        _make_and_register_suite({"Obs": enough_sens["Obs"], case: enough_sens[case]}, case, img_base)