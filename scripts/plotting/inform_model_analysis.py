"""
adf_histogram

Create histogram(s) of all specified (2D) variables.

- Constructs global histograms: total, ocean, land.
  + for consistency with other ADF, will default to making annual and seasonal histograms.
- Saves the resulting histogram data as netCDF (into the plots directory by default)
- Plots the histograms.

Options:
  - specify whether to do spatial distribution from climo files
    or combine temporal and spatial using time series files

Possible future enhancements:
  - specify a landmask file
  - specific histogram options in the variable defaults file (looks for hist_bins)
  - ability to specify additional regions (?)
  - specify the output location for netCDF files
  - improved detection of dimensions that (like cosp height)
  - allow for rgridding; right now if LANDFRAC and data are not the same, just skips.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import os, re, sys, time, json, shutil, tempfile
import functools
from collections.abc import Mapping
from glob import glob

import cam_era5_functions as cfun
import cam_diagnostic as cdog
import inform_native_grid as ingrid
import inform_utils as inform

import warnings  # use to warn user about missing files.
import adf_utils as utils
warnings.formatwarning = utils.my_formatwarning

#
# USER-ADJUSTABLE PARAMETERS:
#

class LazyDataset:
    """
    Stand in for a dataset that is only loaded when first used.

    Indexing or attribute access loads it (once) and forwards to it, so the
    diagnostics use it exactly as they would the dataset itself.
    """

    def __init__(self, loader, what):
        self._loader = loader
        self._what = what
        self._ds = None

    def _get(self):
        if self._ds is None:
            print(f"\n  Loading {self._what} (first use)")
            self._ds = self._loader()
        return self._ds

    def __getitem__(self, key):
        return self._get()[key]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get(), name)


class LazyCases(Mapping):
    """
    {case nickname: one part of that case's CAM data}, loading each case only
    when a diagnostic first indexes it.

    Listing the cases (iterating, len, `in`) never loads anything.  Views that
    share `cache` (ds_cams and ds_outs) load each case at most once between them.
    """

    def __init__(self, loaders, part, cache):
        self._loaders = loaders
        self._part = part
        self._cache = cache

    def __getitem__(self, case):
        if case not in self._cache:
            self._cache[case] = self._loaders[case]()
        return self._cache[case][self._part]

    def __iter__(self):
        return iter(self._loaders)

    def __len__(self):
        return len(self._loaders)

    def __contains__(self, case):
        return case in self._loaders


# CAM history variables the INFORM loaders can't do without
# (cam_era5_functions.compute_cam_aerosol_micphys_metrics and load_cam_ccfs);
# the aerosol/CCN fields they use are optional and read as zero when missing.
REQUIRED_CAM_VARS = (
    "PS", "hyam", "hybm", "T", "Q", "U", "V", "OMEGA", "TS",
    "WP2_ZT_CLUBB", "CLDLIQ", "FREQL", "AWNC", "AQRAIN", "FREQR", "ANRAIN",
)

# Regionally subset CAM output appends e.g. "_130e_to_170e_35s_to_65s" to names
_REGION_SUFFIX = re.compile(r"_\d+[ew]_to_\d+[ew]_\d+[ns]_to_\d+[ns]$")


def missing_cam_vars(hist_file):
    """
    The REQUIRED_CAM_VARS that `hist_file` doesn't have (regional suffixes ignored).
    """
    with xr.open_dataset(hist_file, decode_times=False) as ds:
        have = {_REGION_SUFFIX.sub("", str(v)) for v in ds.variables}
    return [v for v in REQUIRED_CAM_VARS if v not in have]


def load_cam_case(cam_dir, file_sel, campaign, tmin, tmax, lat_slice, lon_slice,
                  cache_path=None, cache_fingerprint=None, grid=None):
    """
    Load one CAM case and build what the diagnostics use from it.

    With `cache_path`, the case's derived fields and regime masks are read
    from that file when its fingerprint matches `cache_fingerprint`;
    otherwise they are derived from the history files, saved there, and read
    back, so a fresh and a cached load give the diagnostics the same data
    (see cam_diagnostic.read_cam_case_cache/write_cam_case_cache).

    With `grid` ({"weights_file", "regrid_method", "out_dir"}), the case's
    history files are on a native grid: the ones the window needs are
    gridded to lat/lon first (only when the saved CAM data can't be used).

    Returns
    -------
    dict
        "out": microphysics/state/aerosol dataset (ds_out),
        "comp": the regime composites (ds_cam_comp).
    """
    t0 = time.perf_counter()
    if cache_path is not None:
        cached = cdog.read_cam_case_cache(cache_path, cache_fingerprint)
        if cached is not None:
            ds_out, masks = cached
            ds_cam_comp = cdog.comp_from_masks(ds_out, masks)
            utils.timer(f"loaded saved CAM data {Path(cache_path).name}", t0)
            return {"out": ds_out, "comp": ds_cam_comp}

    if grid is not None:
        cam_dir = ingrid.grid_case_for_window(
            cam_dir, file_sel, tmin, tmax, grid["weights_file"], grid["out_dir"],
            method=grid["regrid_method"], region=(lat_slice, lon_slice))
        t0 = utils.timer("gridded native-grid history files", t0)

    ds_cam = cfun.load_cam_files(cam_dir, file_sel, tmin, tmax, campaign=campaign)
    t0 = utils.timer("loaded ds_cam", t0)
    # Dataset with all the microphysics, state, and aerosol parameters
    ds_out = cfun.compute_cam_aerosol_micphys_metrics(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice)
    t0 = utils.timer("loaded ds_out", t0)
    # CAM cloud controlling factors, for compositing CAM data like the sonde/aircraft observations
    ds_cam_ccfs = cfun.load_cam_ccfs(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice)
    t0 = utils.timer("loaded ds_cam_ccfs", t0)
    ds_cam_comp = cfun.subset_cam_by_campaign(ds_out,ds_cam_ccfs,campaign=campaign)
    t0 = utils.timer("loaded ds_cam_comp", t0)

    if cache_path is None:
        return {"out": ds_out, "comp": ds_cam_comp}

    print(f"  Saving CAM data for the diagnostics to {cache_path}")
    cdog.write_cam_case_cache(cache_path, ds_out, ds_cam_comp["masks"], cache_fingerprint)
    t0 = utils.timer("saved CAM data", t0)
    ds_out, masks = cdog.read_cam_case_cache(cache_path, cache_fingerprint)
    return {"out": ds_out, "comp": cdog.comp_from_masks(ds_out, masks)}

def load_nc_cldrgme(file_paths):
    """
    Load and combine NetCDF cloud regime files into a single DataFrame.
    
    Parameters
    ----------
    file_paths : list
        List of file paths to NetCDF files
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all blocks
    """
    combined_blocks = []   
    for path in file_paths:
        ds = xr.open_dataset(path)
        
        # Convert to DataFrame for convenient filtering
        df = ds.to_dataframe().reset_index().drop(columns="index")
        
        # Get all unique (label, index) pairs
        unique_blocks = df[["block_label", "block_index"]].drop_duplicates()
        
        # Loop through each unique block
        for _, row in unique_blocks.iterrows():
            label = row["block_label"]
            idx = row["block_index"]
        
            # Filter DataFrame
            df_block = df[(df["block_label"] == label) & (df["block_index"] == idx)].copy()
        
            combined_blocks.append(df_block)
    
    all_blocks = pd.concat(combined_blocks, ignore_index=True)
    return all_blocks

def build_flight_files(adfobj, campaign, flight_dir, idx, reason):
    """
    Build a campaign's per-flight cloud regime files from its raw aircraft data.

    The files are built into a new directory beside ``flight_dir``, which
    replaces ``flight_dir`` once every flight is written.  Their names carry
    the build date, so building over old files would leave two of each flight
    for the ``*RF*.nc`` glob, and an interrupted build leaves ``flight_dir``
    as it was.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object; its ``campaigns_dict`` gives ``air_1hz_dir``
        and, optionally, ``air_25hz_dir``.
    campaign : str
        "SOCRATES" or "CSET".
    flight_dir : pathlib.Path
        Directory that holds the campaign's flight files.
    idx : int
        The campaign's position in the config's campaigns section.
    reason : str
        Why the files are being built, for the log line.

    Returns
    -------
    list of str
        The flight files, sorted.
    """
    air_dirs = adfobj.campaigns_dict.get("air_1hz_dir") or []
    if idx >= len(air_dirs) or not air_dirs[idx]:
        adfobj.end_diag_fail(f"build_flight_files is set for {campaign}, but the config's "
                             "campaigns section has no 'air_1hz_dir' to build them from.")
    flight_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{flight_dir.name}.building-", dir=flight_dir.parent))
    print(f"\nBuilding {campaign} per-flight cloud regime files in {flight_dir} ({reason})")
    t0 = time.perf_counter()
    try:
        inform.build_flight_cloud_regime_files(campaign, adfobj, tmp_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    file_names = sorted(p.name for p in tmp_dir.glob(f"*{campaign}*RF*.nc"))
    if not file_names:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        adfobj.end_diag_fail(f"No {campaign} flight files were built from {air_dirs[idx]}")
    old_dir = flight_dir.with_name(f".{flight_dir.name}.old-{os.getpid()}")
    if flight_dir.exists():
        flight_dir.rename(old_dir)
    tmp_dir.rename(flight_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
    t0 = utils.timer(f"built {len(file_names)} flight files", t0)
    return [str(flight_dir / name) for name in file_names]

def inform_model_analysis(adfobj):
    """
    Run the INFORM campaign cloud micro analysis.
    
    Parameters
    ----------
    adfobj : ADF object
        The ADF diagnostics object containing campaign information.
    """
    print('\nStarting INFORM campaign cloud micro analysis.')
    
    # Load cloud regime data for the specified campaign
    campaign_name = "SOCRATES"
    print(adfobj.plotting_scripts)
    #print(sorted(glob(f"{__file__}/*.py")))
    
    micro_diag_dir = Path("scripts/plotting/model_analysis_micro_scripts/")
    micro_all_files = sorted(micro_diag_dir.glob(f"*.py"))
    print(micro_all_files)
    micro_filtered_files = [f for f in micro_all_files if f.name != '__init__.py']
    print("\nMICRO WOWSA:  -> ",micro_filtered_files,"\n-------------------------------------\n")

    macro_diag_dir = Path("scripts/plotting/model_analysis_macro_scripts/")
    macro_all_files = sorted(macro_diag_dir.glob(f"*.py"))
    print(macro_all_files)
    macro_filtered_files = [f for f in macro_all_files
                            if f.name not in ('__init__.py', 'cam_obs_dropsonde_comp_ryan.py')]
    print("\nMACRO WOWSA:  -> ",macro_filtered_files,"\n-------------------------------------\n")

    process_diag_dir = Path("scripts/plotting/model_analysis_process_scripts/")
    process_all_files = sorted(process_diag_dir.glob(f"*.py"))
    print(process_all_files)
    process_filtered_files = [f for f in process_all_files if f.name != '__init__.py']
    print("\PROCESS WOWSA:  -> ",process_filtered_files,"\n-------------------------------------\n")
    

    plot_dict = {"micro":"",
                 "macro":"",
                 "process":""}

    # Tiers requested in the dict form of the plotting_scripts entry, e.g.
    #   - inform_model_analysis: [micro, macro, process]
    req_diags = []
    for i,item in enumerate(adfobj.plotting_scripts):
        if type(item)==dict:
            #print(item,item.keys())
            if list(item.keys())[0] == "inform_model_analysis":
                req_diags = []
                print('item["inform_model_analysis"] WHAT?',item["inform_model_analysis"])
                for diag_type in item["inform_model_analysis"]:
                    if diag_type in ["micro", "macro", "process"]:
                        #lst.index("dict")
                        req_diags.append(diag_type)

    print(req_diags)
    if not req_diags:
        print("\tNo INFORM diagnostics requested: list micro, macro and/or process under "
              "'inform_model_analysis' in plotting_scripts. Skipping INFORM model analysis.")
        return
    #sys.exit()
    # Additional analyses can be added here as needed
    print('\nStarting COMPASS model analysis plots.')
    #print("adfobj.campaigns_dict",adfobj.campaigns_dict)
    campaigns = adfobj.campaigns_dict["name"]
    #cam_paths = models_dict.keys()
    campaigns_defaults = adfobj.campaigns_dict
    #print("campaigns_defaults",campaigns_defaults)
    #models_dict = adfobj.model_dict
    cam_hist_locs = adfobj.get_cam_info("cam_hist_loc")
    cam_case_names = adfobj.get_cam_info("cam_case_name")
    #for cam_path in ["/glade/derecho/scratch/islas/archive/f.e30_cam6_4_120.FHIST_BGC.f09_f09_mg17.SOCRATES_nudgeUVTfull_withCOSP_tau6h.002/atm/hist"]:
    for idx,campaign in enumerate(campaigns):
        #composited_data_path = Path(campaigns_defaults[campaign]["composited_data"])
        composited_data_path = Path(campaigns_defaults["composited_data"][idx])
        # Find all flight data files for this campaign
        composited_data_path.mkdir(parents=True, exist_ok=True)

        # redo_composited rebuilds every composite below (the ADF-built flight
        # files, the flight table and the sondes)
        redo_list = campaigns_defaults.get("redo_composited") or []
        redo = bool(redo_list[idx]) if idx < len(redo_list) else False

        # Individual flight data files: with build_flight_files, the ADF builds
        # them from the raw aircraft data into the INFORM cache, shared by the
        # configs that share it; otherwise they are read from campaign_data.
        build_list = campaigns_defaults.get("build_flight_files") or []
        if idx < len(build_list) and build_list[idx]:
            flight_dir = cdog.inform_cache_root(adfobj, campaign) / "flights" / campaign
            file_paths = sorted(glob(f"{flight_dir}/*{campaign}*RF*.nc"))
            if redo or not file_paths:
                file_paths = build_flight_files(adfobj, campaign, flight_dir, idx,
                                                "redo_composited" if redo else "not built yet")
        else:
            campaign_data_path = Path(campaigns_defaults["campaign_data"][idx])
            file_paths = sorted(glob(f"{campaign_data_path}/*{campaign}*RF*.nc"))

            #TODO: add check for all flight files being present, and if not, print a warning and exit.
            if not file_paths:
                adfobj.end_diag_fail(f"No flight files (*{campaign}*RF*.nc) found for campaign {campaign} in {campaign_data_path}")

        # Campaign region (lon in the 0..360 convention); shared with the
        # native-grid gridding, which crops to it:
        lat_slice, lon_slice = ingrid.CAMPAIGN_REGIONS[campaign]

        # Combined flight data. A sidecar lists the flight files (name, size,
        # mtime) the CSV was built from; it is rebuilt when that list changes,
        # when it has no sidecar, or with redo_composited. Like the sonde
        # composite, it is always read back from the file, so a run that builds
        # it and a run that reuses it see exactly the same data.
        rf_csv = composited_data_path / f"{campaign}_combined_RF_data.csv"
        rf_sources = rf_csv.with_name(rf_csv.name + ".sources.json")
        built_from = [{"file": Path(p).name, "size": Path(p).stat().st_size,
                       "mtime_ns": Path(p).stat().st_mtime_ns} for p in file_paths]
        try:
            stored = json.loads(rf_sources.read_text())
        except (OSError, ValueError):
            stored = None
        if redo or not rf_csv.is_file() or stored != built_from:
            reason = ("redo_composited" if redo else "not built yet" if not rf_csv.is_file()
                      else "no record of its flight files" if stored is None
                      else "flight files changed")
            print(f"\nBuilding {rf_csv.name} ({reason})")
            tmp_csv = rf_csv.with_name(rf_csv.name + ".tmp")
            load_nc_cldrgme(file_paths).to_csv(tmp_csv, index=False)
            os.replace(tmp_csv, rf_csv)
            rf_sources.write_text(json.dumps(built_from, indent=1))
        All_rf_df = pd.read_csv(rf_csv)

        df_sonde = None
        if "macro" in req_diags:
            # The dropsonde composite lives in composited_data; build it from the
            # raw sondes the first time (or whenever redo_composited is true).
            # It is always read back from the file, so a run that builds it and
            # a run that reuses it see exactly the same data.
            sonde_file = composited_data_path / f"{campaign}_sonde_data_composite.nc"
            if redo or not sonde_file.is_file():
                sonde_dirs = campaigns_defaults.get("campn_sonde_dir") or []
                if idx >= len(sonde_dirs) or not sonde_dirs[idx]:
                    raise ValueError(f"No {campaign} dropsonde composite at {sonde_file}, and no "
                                     "'campn_sonde_dir' in the config's campaigns section to build it from.")
                print(f"\nBuilding {campaign} dropsonde composite from {sonde_dirs[idx]}")
                t0 = time.perf_counter()
                inform.build_sonde_composite(campaign, inform.find_sondes(sonde_dirs[idx]),
                                             out_file=sonde_file)
                t0 = utils.timer(f"built {sonde_file.name}", t0)
            ds_sonde = xr.open_dataset(sonde_file)
            df_sonde = ds_sonde.to_dataframe()

        air_time = All_rf_df.Time
        time_ser = air_time.astype('datetime64[ns]')
        #print("time_ser",time_ser,"\n")
        
        tmin = pd.to_datetime(time_ser.min()).to_pydatetime()
        #print("time_ser.index(time_ser.min())",list(time_ser).index(time_ser.min()))
        
        ###tmax = time_ser[list(time_ser).index(time_ser.min())+1200]
        #tmax = pd.to_datetime(time_ser.mean()).to_pydatetime()
        tmax = pd.to_datetime(time_ser.max()).to_pydatetime()

        #tmin = time_ser[list(time_ser).index(time_ser.min())+12000]
        #tmax = time_ser[list(time_ser).index(time_ser.min())+12000+24000]

        print("tmin",tmin)
        print("tmax",tmax,"\n")
        # Nothing below reads ERA5 or CAM data yet: the diagnostics only load
        # what their caches are missing, on first use (see LazyDataset/LazyCases).
        ds_era5 = LazyDataset(
            functools.partial(cfun.load_era5_files, tmin, tmax, lat_slice, lon_slice),
            f"ERA5 for {campaign}")

        cache_root = cdog.inform_cache_root(adfobj, campaign)
        case_loaders = {}
        # {nickname: details}: the diagnostics key each case's cache file by
        # its full case name and fingerprint it with these (see
        # cam_diagnostic.cached_source).
        case_info = {}
        skipped_cases = {}  # {nickname: missing variables}
        for case_idx,cam_path in enumerate(cam_hist_locs):
            cam_dir = Path(cam_path)
            print("\ncam_dir",cam_dir)
            cam_desc = adfobj.get_cam_info("case_nickname")[case_idx]
            print("cam_desc",cam_desc)
            print('adfobj.hist_string["test_hist_str"]',adfobj.hist_string["test_hist_str"])
            print('adfobj.hist_string["test_hist_str"][case_idx]',adfobj.hist_string["test_hist_str"][case_idx])
            file_sel = adfobj.hist_string["test_hist_str"][case_idx][0]

            print("CAM case description:",cam_desc)
            print("CAM case history num:",file_sel,"\n")

            if cam_desc in case_info:
                raise ValueError(f"Case nickname '{cam_desc}' is used by more than one case; "
                                 "INFORM diagnostics need unique case_nickname values.")
            # A case whose history files lack variables the loaders need is
            # left out of the analysis and plots (the other cases still run):
            window = ingrid.window_files(cam_dir, file_sel, tmin, tmax)
            missing = missing_cam_vars(window[0]) if window else []
            if missing:
                msg = (f"Skipping INFORM case '{cam_desc}' ({cam_case_names[case_idx]}): "
                       f"its {file_sel} files lack {', '.join(missing)}")
                print(f"\n\tWARNING: {msg}")
                warnings.warn(msg)
                skipped_cases[cam_desc] = missing
                continue
            case_info[cam_desc] = {
                "case_name": cam_case_names[case_idx],
                "hist_str": file_sel,
                "cam_hist_loc": str(cam_dir),
                "tmin": tmin, "tmax": tmax,
                "lat_slice": lat_slice, "lon_slice": lon_slice,
            }
            # A case on a native grid is gridded to lat/lon (into the INFORM
            # cache) before it is loaded.  Only then does its fingerprint
            # include the gridding, so lat/lon cases keep their saved data:
            grid = None
            if window and ingrid.is_native(window[0]):
                grid = {
                    "weights_file": adfobj.latlon_wgt_files["test_wgts_file"][case_idx],
                    "regrid_method": adfobj.latlon_regrid_method["test_regrid_method"][case_idx],
                }
                case_info[cam_desc]["grid"] = {k: str(v) for k, v in grid.items()}
                grid["out_dir"] = ingrid.gridded_dir(
                    cache_root, campaign, cam_case_names[case_idx], file_sel,
                    (lat_slice, lon_slice), grid["weights_file"], grid["regrid_method"])
            # Each case's derived CAM data is saved in the INFORM cache too, so
            # a later run that needs this case doesn't re-read its history files:
            cache_key, cache_source = cdog.case_cache_source(case_info, cam_desc)
            case_loaders[cam_desc] = functools.partial(
                load_cam_case, cam_dir, file_sel, campaign, tmin, tmax, lat_slice, lon_slice,
                cache_path=cdog.cam_case_cache_path(cache_root, campaign, cache_key),
                cache_fingerprint=cdog.cam_case_fingerprint(cache_source),
                grid=grid)

        if skipped_cases:
            print(f"\n{campaign}: skipped {len(skipped_cases)} of {len(cam_hist_locs)} cases "
                  f"for missing variables: {', '.join(skipped_cases)}")
        if not case_loaders:
            adfobj.end_diag_fail(f"No {campaign} case has all the CAM variables INFORM needs "
                                 f"({', '.join(REQUIRED_CAM_VARS)}).")

        loaded_cases = {}
        ds_cams = LazyCases(case_loaders, "comp", loaded_cases)
        ds_outs = LazyCases(case_loaders, "out", loaded_cases)
        print("\n\n",list(ds_outs),"\n\n")
        #if campaign == "SOCRATES":
            #sensitivity_vs_sigmaw_2x2_obs_cam(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
            #alpha_sigmaw_regime_comparison_2x2(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
        ###cam_obs_dropsonde_comp(adfobj, campaign, df_sonde, ds_outs, ds_era5)
        #cam_obs_nd_lwc_pdf(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
        
        for diag in req_diags:
            if diag == "macro":
                from model_analysis_macro_scripts import cam_obs_dropsonde_comp
                cam_obs_dropsonde_comp(adfobj, campaign, df_sonde, ds_outs, ds_era5, case_info)
            if diag == "micro":
                from model_analysis_micro_scripts import cam_obs_nd_lwc_pdf
                cam_obs_nd_lwc_pdf(adfobj, All_rf_df, ds_cams, campaign, case_info)
            if campaign == "SOCRATES":
                if diag == "process":
                    from model_analysis_process_scripts import alpha_sigmaw_regime_comparison_2x2,sensitivity_vs_sigmaw_2x2_obs_cam
                    sensitivity_vs_sigmaw_2x2_obs_cam(adfobj, All_rf_df, ds_cams, campaign, case_info)
                    alpha_sigmaw_regime_comparison_2x2(adfobj, All_rf_df, ds_cams, campaign, case_info)

        #for a in req_diags:
        #    for b in plot_dict[a]:
    print("ok yeah")