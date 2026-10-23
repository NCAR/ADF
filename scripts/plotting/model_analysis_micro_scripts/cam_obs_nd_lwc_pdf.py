"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

#from scipy.special import erf
import os, xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
import glob
import sys
import json
import matplotlib.pyplot as plt

from metpy.units import units

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import inform_utils as inform
import cam_era5_functions as cfun
import cam_diagnostic as cdog

"""# Add parent directory to sys.path
parent_dir = Path('/glade/u/home/patnaude/inform/').resolve().parent.parent
sys.path.append(str(parent_dir))"""



import plotting_utils as plot_utils


"""def cam_obs_dropsonde_comp(icf, All_rf_df, campaign, 
                         marker_map=None, color_map=None, size_map=None):
    
    # Located folder with .nc CAM output
    cam_dir = Path("/glade/derecho/scratch/islas/archive/f.e30_cam6_4_120.FHIST_BGC.f09_f09_mg17.SOCRATES_nudgeUVTfull_withCOSP_tau6h.002/atm/hist")
    cam_desc = "nudgeUVTfull_withCOSP_tau6h.002"
    # Select campaign
    campaign = 'SOCRATES'

    if campaign == 'SOCRATES':
        lat_slice = slice(-62, -42)
        lon_slice = slice(135, 165)   # 0..360 convention
    elif campaign == 'CSET':
        lat_slice = slice(15, 45)
        lon_slice = slice(200, 240)

    file_sel = "cam.h0"
"""



def _json_float(v):
    """
    float(v), or None for NaN/inf - json.dump would otherwise write a bare
    NaN token, which is invalid JSON and makes JSON.parse in the D3 page
    fail. The dashboard drops non-finite records, so null is safe there.
    """
    v = float(v)
    return v if np.isfinite(v) else None


def _pdf_records(centers, probs):
    """A PDF as tidy per-bin records: [{"bin_center": c, "probability": p}, ...]."""
    return [
        {"bin_center": float(c), "probability": _json_float(p)}
        for c, p in zip(centers, probs)
    ]


def _named_pdf_entry(Nd_centers, LWC_centers, pdf_stats):
    """
    One named source's (a CAM case, or the shared "Obs" reference) Nd/LWC
    PDFs by regime.
    """
    return {
        "open": {
            "Nd":  _pdf_records(Nd_centers, pdf_stats["Nd_open"]),
            "LWC": _pdf_records(LWC_centers, pdf_stats["LWC_open"]),
        },
        "strat": {
            "Nd":  _pdf_records(Nd_centers, pdf_stats["Nd_strat"]),
            "LWC": _pdf_records(LWC_centers, pdf_stats["LWC_strat"]),
        },
    }


def save_named_pdf_json(name, campaign, out_dir, Nd_centers, LWC_centers, pdf_stats):
    """
    Write one named Nd/LWC PDF (a CAM case, or the shared "Obs" reference)
    to a browser-friendly JSON file, for use outside of Python (e.g. D3).
    Filenames are qualified by campaign since "Obs" (and possibly case
    names) are reused across campaigns - without this, a second campaign's
    run would silently overwrite the first's per-source files.
    """
    record = {"case": name, **_named_pdf_entry(Nd_centers, LWC_centers, pdf_stats)}
    out_path = Path(out_dir) / f"nd_lwc_pdf_{campaign}_{name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote Nd/LWC PDF JSON: {out_path}")


def save_nd_lwc_dashboard_json(campaign, Nd_bins, Nd_centers, LWC_bins, LWC_centers,
                                enough_pdf, out_dir, config_name):
    """
    Combine every source into the single-file shape the D3 dashboard expects:
    "Obs" is a peer of the CAM cases under "cases" - shared reference data,
    not duplicated inside every CAM case. Bins/centers are shared constants,
    so they live at the top level, same role campaign/sonde_var play for
    the dropsonde dashboard JSON.
    """
    dashboard_json = {
        "campaign": campaign,
        "Nd_bins": [float(v) for v in Nd_bins],
        "Nd_centers": [float(v) for v in Nd_centers],
        "LWC_bins": [float(v) for v in LWC_bins],
        "LWC_centers": [float(v) for v in LWC_centers],
        "cases": {
            name: _named_pdf_entry(Nd_centers, LWC_centers, pdf_stats)
            for name, pdf_stats in enough_pdf.items()
        },
    }

    out_path = Path(out_dir) / f"nd_lwc_pdfs_{campaign}_{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"Wrote Nd/LWC dashboard JSON: {out_path}")
    return out_path


def compute_obs_pdf_stats(All_rf_df,
                           regimes_of_interest=("Stratocumulus", "Open-Cell"),
                           labels_of_interest=("In-Cloud Level FT", "In-Cloud Profiles"),
                           Nd_bins=None, LWC_bins=None,
                           obs_Nd_min=10.0, obs_LWC_min=0.001, T_min_C=0.0):
    """
    Compute the observed Nd/LWC PDFs by regime - case-independent, since
    these come only from All_rf_df (the dropsonde/aircraft observations),
    not from any particular CAM case.
    """
    if Nd_bins is None:
        Nd_bins = np.logspace(-1, 3, 15)
    if LWC_bins is None:
        LWC_bins = np.logspace(-3, 1, 15)

    concd_col_obs, plwc_col_obs = cdog.find_obs_nd_lwc_columns(All_rf_df)

    pdf_Nd = {}
    pdf_LWC = {}
    for regime in regimes_of_interest:
        df_sub = All_rf_df[
            (All_rf_df["block_label"].isin(labels_of_interest)) &
            (All_rf_df["cloud_regime"] == regime) &
            (All_rf_df[concd_col_obs] > obs_Nd_min) &
            (All_rf_df[plwc_col_obs] > obs_LWC_min) &
            (All_rf_df["ATX"] > T_min_C)
        ]
        pdf_Nd[regime] = cdog.compute_pdf_1d(df_sub[concd_col_obs].values, Nd_bins)
        pdf_LWC[regime] = cdog.compute_pdf_1d(df_sub[plwc_col_obs].values, LWC_bins)

    return {
        "Nd_open": pdf_Nd["Open-Cell"], "LWC_open": pdf_LWC["Open-Cell"],
        "Nd_strat": pdf_Nd["Stratocumulus"], "LWC_strat": pdf_LWC["Stratocumulus"],
    }


def compute_case_pdf_stats(ds_cam_comp, cam_regime_keys=None,
                            Nd_bins=None, LWC_bins=None,
                            cam_Nd_min=10.0, cam_LWC_min=0.001, T_min_C=0.0):
    """
    Compute one CAM case's Nd/LWC PDFs for both regimes.
    """
    if Nd_bins is None:
        Nd_bins = np.logspace(-1, 3, 15)
    if LWC_bins is None:
        LWC_bins = np.logspace(-3, 1, 15)
    if cam_regime_keys is None:
        cam_regime_keys = {"Stratocumulus": "strat", "Open-Cell": "open"}

    pdf_Nd = {}
    pdf_LWC = {}
    for regime, key in cam_regime_keys.items():
        ds_reg = ds_cam_comp[key]

        T_C = ds_reg["T_K"] - 273.15
        mask = (
            (T_C > T_min_C) &
            (ds_reg["cam_lwc"] > cam_LWC_min) &
            (ds_reg["cam_Nc"] > cam_Nd_min)
        )

        Nd_cam_da = ds_reg["cam_Nc"].where(mask)
        LWC_cam_da = ds_reg["cam_lwc"].where(mask)

        pdf_Nd[regime] = cdog.compute_pdf_dask(Nd_cam_da, Nd_bins)
        pdf_LWC[regime] = cdog.compute_pdf_dask(LWC_cam_da, LWC_bins)

    return {
        "Nd_open": pdf_Nd["Open-Cell"], "LWC_open": pdf_LWC["Open-Cell"],
        "Nd_strat": pdf_Nd["Stratocumulus"], "LWC_strat": pdf_LWC["Stratocumulus"],
    }


# Bump when compute_obs_pdf_stats/compute_case_pdf_stats change what they
# compute, so results cached by older code are recomputed (see
# cdog.cached_source).
CACHE_VERSION = 1


def cam_obs_nd_lwc_pdf(adf, All_rf_df, ds_cams, campaign, case_info=None):
    """
    Obs-vs-CAM Nd and LWC probability distributions, by cloud regime.

    `ds_cams` ({case nickname: regime composites}) is only indexed for cases
    not cached yet. `case_info` ({case nickname: case details} from
    inform_model_analysis) keys each case's cache file by its full case name.
    """
    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Nd LWC PDF plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    yup = Path(adf.config_file_dict["name"])
    plot_loc = Path(adf.plot_location[0])
    # JSON exports live alongside the dashboard/html files
    # (created already by AdfWeb.__init__, but mkdir here too to be safe):
    website_dir = plot_loc / "website"
    website_dir.mkdir(parents=True, exist_ok=True)

    Nd_bins = np.logspace(-1, 3, 15)
    LWC_bins = np.logspace(-3, 1, 15)
    Nd_centers = np.sqrt(Nd_bins[:-1] * Nd_bins[1:])
    LWC_centers = np.sqrt(LWC_bins[:-1] * LWC_bins[1:])

    # One cache file per source ("Obs" and each case, keyed by full case
    # name), checked against a fingerprint of everything the result depends
    # on (cdog.cached_source):
    cache_root = cdog.inform_cache_root(adf, campaign)
    settings = {"Nd_bins": Nd_bins, "LWC_bins": LWC_bins}

    enough_pdf = {"Obs": cdog.cached_source(
        cache_root, "nd_lwc_pdf", campaign, "Obs", cdog.obs_cache_source(All_rf_df),
        settings, CACHE_VERSION,
        compute=lambda: compute_obs_pdf_stats(All_rf_df, Nd_bins=Nd_bins, LWC_bins=LWC_bins))}
    for case in ds_cams:
        key, source = cdog.case_cache_source(case_info, case)
        enough_pdf[case] = cdog.cached_source(
            cache_root, "nd_lwc_pdf", campaign, key, source, settings, CACHE_VERSION,
            compute=lambda case=case: compute_case_pdf_stats(
                ds_cams[case], Nd_bins=Nd_bins, LWC_bins=LWC_bins),
            label=case)

    # JSON exports: one file per named source (Obs + each CAM case), plus a
    # single combined file for the D3 dashboard.
    for name, pdf_stats in enough_pdf.items():
        save_named_pdf_json(name, campaign, website_dir, Nd_centers, LWC_centers, pdf_stats)
    dashboard_json = save_nd_lwc_dashboard_json(campaign, Nd_bins, Nd_centers, LWC_bins,
                                                LWC_centers, enough_pdf, website_dir, yup)
    # So the website builds this campaign's interactive Nd/LWC page:
    adf.add_inform_page(campaign, "nd_lwc_pdf", dashboard_json)

    # Skip plot generation entirely when the config has create_html turned
    # off - the cached stats and dashboard JSON above are already written
    # regardless, so this just avoids an unused PNG and figure work.
    if not adf.create_html:
        print("  create_html is False: skipping Nd/LWC PDF plot generation.")
        return

    # Machine key -> (filename suffix used by plot_obs_cam_nd_lwc_pdfs,
    # human-readable label for the dashboard's panel selector).
    panel_suffixes = {
        "open_cell_nd":      ("OpenCellNd",       "Open-Cell Nd"),
        "open_cell_lwc":     ("OpenCellLWC",      "Open-Cell LWC"),
        "stratocumulus_nd":  ("StratocumulusNd",  "Stratocumulus Nd"),
        "stratocumulus_lwc": ("StratocumulusLWC", "Stratocumulus LWC"),
    }

    def _make_and_register_suite(cases, cam_desc, img_base):
        """
        Build one summary (2x2) figure + its 4 individually-saved
        component panels for the case names in `cases`, drawn from the
        cached `enough_pdf` (so no CAM data is read), and register all 5
        with the website generator under `cam_desc`.
        """
        fig, axes, pdfs = cdog.plot_obs_cam_nd_lwc_pdfs(
                All_rf_df=All_rf_df,
                cam_cases=list(cases),
                savepath=None,
                save_indiv_panels=True,
                indiv_plot_loc=plot_loc,
                indiv_img_base=img_base,
                pdf_stats=enough_pdf,
        )

        img_name = f'{img_base}.png'
        img_path = plot_loc / img_name

        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Nd LWC PDF",
                                category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                panel="summary", panel_label="Summary")

        for panel_key, (suffix, panel_label) in panel_suffixes.items():
            panel_img_base = f"{img_base}_{suffix}"
            panel_img_path = plot_loc / f"{panel_img_base}.png"
            adf.add_website_data(str(panel_img_path), panel_img_base, campaign, plot_type="Nd LWC PDF",
                                    category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                    panel=panel_key, panel_label=panel_label)

    # With more than one case, build the "multi_CAM" ensemble suite first
    # (every case overlaid on the same axes vs Obs), so it's the first
    # entry registered for this campaign and appears first in the
    # dashboard's case dropdown, ahead of the individual cases:
    if len(ds_cams) > 1:
        img_base = f'{campaign}__nd_lwc__multi_CAM'
        _make_and_register_suite(ds_cams, "multi_CAM", img_base)

    # Every case also gets its own summary + component-panel suite (that
    # case alone vs Obs), regardless of how many cases are in this run:
    for case in ds_cams:
        img_base = f'{campaign}__nd_lwc__{case}'
        _make_and_register_suite([case], case, img_base)





