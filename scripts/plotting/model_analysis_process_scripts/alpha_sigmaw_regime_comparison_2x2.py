"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

import os, xarray as xr
from pathlib import Path

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import cam_diagnostic as cdog
from .sensitivity_vs_sigmaw_2x2_obs_cam import load_sensitivity_sources

import matplotlib.pyplot as plt


#cloud regimes
def alpha_sigmaw_regime_comparison_2x2(adf, All_rf_df, ds_cams, campaign, case_info=None):
    """
    Obs-vs-CAM alpha = dlog(Nd)/dlog(CCN) against sigma_w, by cloud regime and
    drizzle state: one figure with every case ("multi_CAM", when there is more
    than one) and one per case, matching the other INFORM diagnostics.

    Uses the sensitivity diagnostic's cached tables, so CAM data is only read
    for cases not cached yet. `case_info` ({case nickname: case details} from
    inform_model_analysis) keys each case's cache file by its full case name.
    """
    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Sigma W Regime Comparison plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    sources = load_sensitivity_sources(adf, All_rf_df, ds_cams, campaign, case_info)

    # Skip plot generation entirely when the config has create_html turned
    # off - the cached tables above are already written regardless.
    if not adf.create_html:
        print("  create_html is False: skipping sigma W regime comparison plots.")
        return

    plot_loc = Path(adf.plot_location[0])

    def _make_and_register(cases, cam_desc, img_base):
        """Build one 2x2 figure for `cases` vs Obs from the cached tables and
        register it with the website generator under `cam_desc`."""
        fig, axes, out = cdog.plot_alpha_sigmaw_regime_comparison_2x2(
                df_obs=All_rf_df,
                cam_cases=list(cases),
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
                figsize=(10, 8),
                sources=sources,
            )
        img_path = plot_loc / f"{img_base}.png"
        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Sigma W Regime Comparison",
                                category="Model Micro State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                panel="summary", panel_label="Summary")

    # With more than one case, the "multi_CAM" figure (every case overlaid)
    # comes first, so it appears first in the dashboard's case dropdown:
    if len(ds_cams) > 1:
        _make_and_register(ds_cams, "multi_CAM", f'{campaign}__sigmaw_regime_comparison__multi_CAM')

    # Every case also gets its own figure (that case alone vs Obs):
    for case in ds_cams:
        _make_and_register([case], case, f'{campaign}__sigmaw_regime_comparison__{case}')
