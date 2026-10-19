"""
Grid INFORM CAM history files from a native (unstructured, "ncol") grid to
lat/lon, so the INFORM diagnostics can run on native-grid history output.

The INFORM diagnostics read CAM history files directly (not ADF time series),
pick the ones covering a campaign's flight window by the date in their names,
then cut the campaign region out by lat/lon slices.  That needs lat/lon
dimensions, so a native-grid case is first gridded with the same ESMF weights
file and gridding code the ADF uses for its own native-grid time series
(adf_utils.grid_to_latlon; see AdfDiag.grid_native_time_series).

Only the history files the campaign window needs are gridded, and only the
campaign region (plus a margin) is kept, since that is all the diagnostics
use.  Each gridded file keeps its original name, so
cam_era5_functions.load_cam_files finds and date-sorts them exactly as it does
the originals, and is written to the INFORM cache, so later runs (and other
configs sharing the cache) reuse it.

Gridding a full campaign can take hours per case, so the cases can be gridded
in parallel ahead of the ADF run, one batch job per case:

    python lib/inform_native_grid.py <config.yaml> --case <index>

The ADF run then finds the gridded files in place and skips straight to the
diagnostics.
"""

import argparse
import bisect
import re
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

import adf_utils as utils

# History file names end in YYYY-MM-DD-SSSSS.nc (the same pattern
# cam_era5_functions.load_cam_files uses to date them):
_DATE_PAT = re.compile(r".*(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")

_NATIVE_DIMS = {"ncol", "ncols"}

# The region each campaign's diagnostics cut out of the CAM data
# (latitude, longitude in the 0..360 convention):
CAMPAIGN_REGIONS = {
    "SOCRATES": (slice(-62, -42), slice(135, 165)),
    "CSET": (slice(15, 45), slice(200, 240)),
}

# Degrees kept around the region when cropping gridded files, so the region's
# edge points are interior points of the gridded data:
REGION_MARGIN = 2.0


def _file_time(path):
    """The time in a history file's name, or None when it has none."""
    m = _DATE_PAT.match(path.name)
    if not m:
        return None
    y, mo, d, sod = map(int, m.groups())
    return pd.Timestamp(
        year=y, month=mo, day=d, hour=sod // 3600, minute=(sod % 3600) // 60,
        second=sod % 60,
    )


def window_files(cam_dir, file_sel, tmin, tmax):
    """
    The history files load_cam_files reads for [tmin, tmax].

    Mirrors its selection: files whose name time falls within the window
    padded by an hour, plus one more file on each side.  Kept in step with
    cam_era5_functions.load_cam_files by
    test_window_files_match_load_cam_files.
    """
    files, times = [], []
    for p in sorted(p for p in Path(cam_dir).glob("*.nc") if file_sel in p.name):
        t = _file_time(p)
        if t is not None:
            files.append(p)
            times.append(t)
    if not files:
        return []
    pad = pd.Timedelta(minutes=60)
    i0 = bisect.bisect_left(times, tmin - pad)
    i1 = bisect.bisect_right(times, tmax + pad) - 1
    if i0 > i1:
        return []
    return files[max(i0 - 1, 0) : min(i1 + 1, len(files) - 1) + 1]


def is_native(path):
    """Whether the history file at `path` is on a native (ncol) grid."""
    with xr.open_dataset(path, decode_times=False) as ds:
        return bool(_NATIVE_DIMS & set(ds.dims))


def gridded_dir(cache_root, campaign, case_name, file_sel, region, wgts_file, method):
    """
    Where one case's gridded history files go in the INFORM cache.

    The region, weights file and method are part of the path, so changing any
    of them grids afresh instead of reusing files made another way.
    """
    lat_slice, lon_slice = region
    region_tag = f"lat{lat_slice.start}_{lat_slice.stop}_lon{lon_slice.start}_{lon_slice.stop}"
    # (without a weights file, grid_case_for_window stops with a clear message)
    wgts_tag = Path(wgts_file).stem if wgts_file else "no_weights_file"
    method_tag = f"{wgts_tag}.{method or 'conservative'}"
    return (Path(cache_root) / "gridded" / campaign / f"{case_name}.{file_sel}"
            / region_tag / method_tag)


def crop_to_region(ds, region, margin=REGION_MARGIN):
    """`ds` cut to `region` (lat, lon slices, increasing) plus `margin` degrees."""
    lat_slice, lon_slice = region
    return ds.sel(
        lat=slice(lat_slice.start - margin, lat_slice.stop + margin),
        lon=slice(lon_slice.start - margin, lon_slice.stop + margin),
    )


def grid_history_file(src, dst, wgts_file, method=None, region=None):
    """
    Grid one native-grid history file to lat/lon and save it as `dst`.

    With `region` (lat, lon slices), only that region plus REGION_MARGIN is
    kept.  Times are left undecoded, so time and time_bnds are written back
    exactly as CAM wrote them.  The file is written under a temporary name and
    then renamed, so an interrupted run never leaves a partial file that a
    later run would take for a finished one.
    """
    with xr.open_dataset(src, decode_times=False) as ds:
        native = [v for v in ds.data_vars if _NATIVE_DIMS & set(ds[v].dims)]
        if not native:
            raise ValueError(f"'{src}' has no fields on a native grid to grid.")
        # grid_to_latlon grids every ncol field at once; the variable it is
        # given only sets which one has missing values filled and its
        # attributes restored.
        gridded = utils.grid_to_latlon(ds, native[0], wgt_file=wgts_file, method=method)
        if region is not None:
            gridded = crop_to_region(gridded, region)
        # Gridding returns float64 fields and drops some attributes (time's
        # units and calendar among them, which load_cam_files needs to decode
        # the times), so give every variable back its own type and attributes:
        for name, var in ds.variables.items():
            if name not in gridded.variables:
                continue
            if name in native and gridded[name].dtype != var.dtype:
                gridded[name] = gridded[name].astype(var.dtype)
            gridded[name].attrs = var.attrs
        tmp = dst.with_name(dst.name + ".tmp")
        utils.save_to_nc(gridded, tmp)
    tmp.replace(dst)


def grid_case_for_window(cam_dir, file_sel, tmin, tmax, wgts_file, out_dir,
                         method=None, region=None, overwrite=False):
    """
    Grid the native-grid history files a campaign window needs.

    Parameters
    ----------
    cam_dir : str or Path
        The case's history file directory ('cam_hist_loc').
    file_sel : str
        The history stream, e.g. "cam.h1i" ('hist_str').
    tmin, tmax : datetime
        The campaign's flight window.
    wgts_file : str or Path
        ESMF weights file from the native grid to lat/lon ('weights_file').
    out_dir : Path
        Where the gridded files go (in the INFORM cache).
    method : str, optional
        'regrid_method' ("conservative", the default, or "bilinear").
    region : (slice, slice), optional
        Keep only this (lat, lon) region plus REGION_MARGIN.
    overwrite : bool
        Re-grid files that were already gridded.

    Returns
    -------
    Path
        The directory to read the case from: `out_dir` when the case is on a
        native grid, otherwise `cam_dir` unchanged.
    """
    files = window_files(cam_dir, file_sel, tmin, tmax)
    if not files or not is_native(files[0]):
        return Path(cam_dir)
    if not wgts_file:
        raise ValueError(
            f"CAM history files in '{cam_dir}' are on a native grid, but the case "
            "has no 'weights_file' to grid them to lat/lon. Add one to "
            "'diag_cam_climo' in the config file."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Gridding {len(files)} native-grid history files to {out_dir}")
    for src in files:
        dst = out_dir / src.name
        if dst.is_file() and not overwrite:
            continue
        print(f"\t  gridding {src.name}", flush=True)
        grid_history_file(src, dst, wgts_file, method=method, region=region)
    return out_dir


def _flight_window(adf, campaign):
    """
    The campaign's (tmin, tmax), from the combined flight table the INFORM
    run builds in 'composited_data' (the same window the run uses).
    """
    campaigns = adf.campaigns_dict
    idx = campaigns["name"].index(campaign)
    rf_csv = Path(campaigns["composited_data"][idx]) / f"{campaign}_combined_RF_data.csv"
    if not rf_csv.is_file():
        sys.exit(f"No {rf_csv}: run the INFORM diagnostics once to build the flight table "
                 "(it stops before loading CAM data if interrupted), then grid.")
    times = pd.read_csv(rf_csv, usecols=["Time"])["Time"].astype("datetime64[ns]")
    return (pd.to_datetime(times.min()).to_pydatetime(),
            pd.to_datetime(times.max()).to_pydatetime())


def main(argv=None):
    """Grid one case of an INFORM config ahead of the ADF run."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("config", help="INFORM ADF config file")
    parser.add_argument("--case", type=int, required=True,
                        help="index of the case in diag_cam_climo (0 = first)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-grid files that were already gridded")
    args = parser.parse_args(argv)

    from adf_info import AdfInfo
    import cam_diagnostic as cdog

    adf = AdfInfo(args.config)
    case_name = adf.get_cam_info("cam_case_name")[args.case]
    cam_dir = adf.get_cam_info("cam_hist_loc")[args.case]
    file_sel = adf.hist_string["test_hist_str"][args.case][0]
    wgts_file = adf.latlon_wgt_files["test_wgts_file"][args.case]
    method = adf.latlon_regrid_method["test_regrid_method"][args.case]

    for campaign in adf.campaigns_dict["name"]:
        tmin, tmax = _flight_window(adf, campaign)
        region = CAMPAIGN_REGIONS[campaign]
        out_dir = gridded_dir(cdog.inform_cache_root(adf, campaign), campaign, case_name,
                              file_sel, region, wgts_file, method)
        print(f"{campaign} {case_name}: {tmin} to {tmax}")
        grid_case_for_window(cam_dir, file_sel, tmin, tmax, wgts_file, out_dir,
                             method=method, region=region, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
