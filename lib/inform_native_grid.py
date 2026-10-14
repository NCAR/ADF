"""
Grid INFORM CAM history files from a native (unstructured, "ncol") grid to
lat/lon, so the INFORM diagnostics can run on native-grid history output.

The INFORM diagnostics read CAM history files directly (not ADF time series),
pick the ones covering a campaign's flight window by the date in their names,
then cut the campaign region out by lat/lon slices.  That needs lat/lon
dimensions, so a native-grid case is first gridded with the same ESMF weights
file and gridding code the ADF uses for its own native-grid time series
(adf_utils.grid_to_latlon; see AdfDiag.grid_native_time_series).

Only the history files the campaign window needs are gridded.  Each gridded
file keeps its original name, so cam_era5_functions.load_cam_files finds and
date-sorts them exactly as it does the originals, and is written to the
INFORM cache, so later runs (and other configs sharing the cache) reuse it.
"""

import bisect
import re
from pathlib import Path

import pandas as pd
import xarray as xr

import adf_utils as utils

# History file names end in YYYY-MM-DD-SSSSS.nc (the same pattern
# cam_era5_functions.load_cam_files uses to date them):
_DATE_PAT = re.compile(r".*(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")

_NATIVE_DIMS = {"ncol", "ncols"}


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


def grid_history_file(src, dst, wgts_file, method=None):
    """
    Grid one native-grid history file to lat/lon and save it as `dst`.

    Times are left undecoded, so time and time_bnds are written back exactly
    as CAM wrote them.  The file is written under a temporary name and then
    renamed, so an interrupted run never leaves a partial file that a later
    run would take for a finished one.
    """
    with xr.open_dataset(src, decode_times=False) as ds:
        native = [v for v in ds.data_vars if _NATIVE_DIMS & set(ds[v].dims)]
        if not native:
            raise ValueError(f"'{src}' has no fields on a native grid to grid.")
        # grid_to_latlon grids every ncol field at once; the variable it is
        # given only sets which one has missing values filled and its
        # attributes restored.
        gridded = utils.grid_to_latlon(ds, native[0], wgt_file=wgts_file, method=method)
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
                         method=None, overwrite=False):
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
        print(f"\t  gridding {src.name}")
        grid_history_file(src, dst, wgts_file, method=method)
    return out_dir
