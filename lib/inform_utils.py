import netCDF4
import pathlib as path
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Iterable
import xarray as xr
import re
from typing import Callable, Optional, Union, Tuple, Iterable, List, Dict
from pathlib import Path
import glob

from scipy.spatial import cKDTree

import adf_info as ADFInfo


def _campaign_cfg_for(campaigns_dict: dict, campaign: str) -> dict:
    """
    Look up one campaign's config values out of an AdfInfo-style
    campaigns_dict.

    campaigns_dict is stored column-major - one list per config key,
    each entry at the same index describing one campaign (see
    AdfInfo.campaigns_dict / the "campaigns:" section of the config
    YAML) - e.g. {"name": ["SOCRATES", "CSET"], "air_1hz_dir": [...],
    ...}, not {"SOCRATES": {...}, "CSET": {...}}. This reconstructs the
    single-campaign "row" (e.g. {"air_1hz_dir": ..., "ccn_dir": ...})
    that the rest of this module's functions expect to do cfg.get(...)
    lookups against.
    """
    names = campaigns_dict.get("name", [])
    if campaign not in names:
        raise ValueError(f"Unknown campaign '{campaign}'")
    idx = names.index(campaign)
    return {
        key: (values[idx] if idx < len(values) else None)
        for key, values in campaigns_dict.items()
        if key != "name"
    }


def find_flight_fnames(campaign: str, freq: str = "air_1hz_dir", icf_obj=None) -> list[str]:
    """
    Return list of aircraft flight files for a campaign.

    Parameters
    ----------
    campaign : str
        Campaign name ("SOCRATES", "CSET", etc.)

    freq : str
        Data frequency key in campaigns_dict
        ("air_1hz_dir", "air_25hz_dir", "ccn_dir")

    icf_obj : an AdfInfo-like object exposing `.campaigns_dict`
        (see _campaign_cfg_for). Required.

    Returns
    -------
    list[str]
        List of flight NetCDF file paths
    """

    if icf_obj is None:
        raise ValueError("find_flight_fnames requires icf_obj (an AdfInfo-like object with .campaigns_dict)")

    campaign = campaign.upper()
    campaign_cfg = _campaign_cfg_for(icf_obj.campaigns_dict, campaign)

    print(campaign, freq)
    dir_path = campaign_cfg.get(freq)

    if dir_path is None:
        raise ValueError(f"{freq} not defined for campaign {campaign}")

    flight_fnames = sorted(
        fname for fname in os.listdir(dir_path)
        if fname.endswith(".nc")
    )

    return [os.path.join(dir_path, f) for f in flight_fnames]

def find_nc_fnames(dir_path: str) -> list[str]:
    """
    find_flight_fnames just searches a directory for all *.nc files and returns a list of them.
    
    :param dir_path: a path to the directory containing flight netcdf files
    
    :return: Returns a list of flight netcdf files.
    """
    nc_paths=[]
    nc_fnames = sorted([fname for fname in os.listdir(dir_path) if fnmatch(fname, "*.nc")])
    for i in range(len(nc_fnames)):
        nc_paths.append(dir_path + '/' + nc_fnames[i])
        
        nudg_path = [file for file in nc_paths if ".hs." in file]
        free_path = [file for file in nc_paths if ".h0." in file]
        # save dictionary with the paths for 
        paths = {'Free': free_path,'Nudg': nudg_path}
        
    return paths

def open_nc(flight_paths: str) -> netCDF4._netCDF4.Dataset:
    """
    open_flight_nc simply checks to see if the file at the provided path string exists and opens it.

    :param file_path: A path string to a flight data file, e.g. "./test/test_flight.nc"

    :return: Returns xr.open_dataset object.
    """
    fp_path = path.Path(flight_paths)
    if not fp_path.is_file():
        raise FileNotFoundError('testing excptions')

    return xr.open_dataset(flight_paths)

def read_flight_nc_1hz(nc: xr.open_dataset, read_vars) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.

    NOTE: a low-rate, 1 Hz, flight data file is assumed

    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An list of strings of variable names to be read into memory.

    :return: Returns a pandas data frame.
    """
    long_names = [nc[var].long_name if 'long_name' in nc[var].attrs else None for var in read_vars]
    data = [] # an empty list to accumulate Dataframes of each variable to be read in
    for var in read_vars:
        try:
            if var == "Time":
                # df = xr.open_dataset(nc)
                time = np.array(nc.Time)
                data.append(pd.DataFrame({var: time}))
                # dt_list = sfm_to_datetime(time, tunits)
                # data.append(pd.DataFrame({'datetime': time}))
            else:
                output = nc[var][:]
                data.append(pd.DataFrame({var: output}))
        except Exception as e:
            print(f"Issue reading {var}: {e}")
            pass
    
    dataframe = pd.concat(data, axis=1, ignore_index=False)
    dataframe.attrs['long_names'] = long_names
    # concatenate the list of dataframes into a single dataframe and return it
    return dataframe

def read_flight_nc_25hz(nc: xr.open_dataset, read_vars) -> pd.DataFrame:
    """
    read_flight_nc reads a set of variables into memory.
    
    NOTE: a high-rate, usually 25 Hz, flight data file is assumed.
    
    :param nc: netCDF4._netCDF4.Dataset object opened by open_flight_nc.
    :param read_vars: An optional list of strings of variable names to be read into memory. A default
                      list, vars_to_read, is specified above. Passing in a similar list will read in those variables
                      instead.
    
    :return: Returns a pandas data frame.
    """
    data = []
    sub_seconds = np.arange(0, 25, 1)/25.
    hz = 25
    for var in read_vars:
        try:
            if var == "Time":
                time = nc[var].values  # Get NumPy array from Xarray
                # Convert sub_seconds into timedelta in nanoseconds
                sub_seconds_ns = (sub_seconds * 1e9).astype('timedelta64[ns]')
                # Expand time into 2D, add sub-second offsets
                time_25hz = time[:, None] + sub_seconds_ns
                output = time_25hz.ravel()  # Flatten to 1D
                data.append(pd.DataFrame({var: output}))
            else:
                ndims = len(np.shape(nc[var][:]))
                if ndims == 2:
                    # 2-D, 25 Hz variables can just be raveled into 1-D time series
                    output = np.ravel(nc[var].values)
                    data.append(pd.DataFrame({var: output}))
                elif ndims == 1:
                    values = nc[var].values  # Extract as NumPy array
                    if values.shape[0] != len(time):  # Interpolation case (e.g., GGALT-style)
                        print(f"Skipping {var} due to shape mismatch: {values.shape[0]} != {len(time)}")
                        continue
                    # Interpolate to 25 Hz (fudged interpolation)
                    output_2d = np.full((len(values), hz), np.nan)
                    for i in range(len(values) - 1):
                        output_2d[i, :] = values[i] + sub_seconds * (values[i+1] - values[i])
                    output = output_2d[:-1].ravel()  # remove the last NaN row
                    data.append(pd.DataFrame({var: output}))
        except Exception as e:
            print(f"Issue reading {var}: {e}")
            pass
    # concatenate the list of dataframes into a single dataframe and return it
    dataframe = pd.concat(data, axis=1, ignore_index=False)
    return dataframe

def read_flight_nc(nc: xr.open_dataset, vars2read: list[str]) -> pd.DataFrame:
    """
    read_flight_nc simply figures out if the flight netcdf object is 1 hz or 25 hz and calls the appropriate reader.

    :param nc: A netcdf object for a flight netcdf file.
    :param read_vars: A list of variable names to be read in the netcdf object. Optional. Default is "vars_to_read" specified
                      above.

    :return: Returns Pandas DataFrame
    """
    dim_names = list(nc.dims)
    if 'sps25' in dim_names:
        df = read_flight_nc_25hz(nc, vars2read)
    else:
        df = read_flight_nc_1hz(nc, vars2read)
    return df

# Function to read in all the relevant variables from the NSF aircraft datasets
def read_vars(nc):

    var_list = nc.data_vars
    time = 'Time'
    # Spatial variables
    lat, lon, alt = 'GGLAT', 'GGLON', 'GGALT'
    
    # state variables
    temp = 'ATX'
    dwpt = 'DPXC'
    u = 'UIC' if 'UIC' in var_list else 'UIX'
    v = 'VIC' if 'VIC' in var_list else 'VIX'
    w = 'WIC' if 'WIC' in var_list else 'WIX'
    p = 'PSXC'
    ew = 'EWX'
    rh = 'RHUM'
    vars_to_read = [time, lat, lon, alt, temp, dwpt, u,  w, p, ew, rh]
    # Thermodynamic data
    if any('THETA' in var for var in var_list): 
        theta_vars = [var for var in var_list if 'THETA' in var and ('_GP' not in var)]
        vars_to_read.extend(theta_vars)
    # Cloud microphysical
    if any('CONC' in var for var in var_list): # cloud concentrations
        conc_vars = [var for var in var_list if 'CONC' in var and 'D' in var and ('R_' not in var and 'CN' not in var and \
                    'CV' not in var and '0_' not in var and 'UD' not in var)]
        # print(conc_vars)
        vars_to_read.extend(conc_vars)
    if any('PLW' in var for var in var_list): # Liquid/Ice water contents
        # v = [var for var in var_list if '2' not in var]
        wc_vars = [var for var in var_list if 'PLW' in var and ('2V' not in var)]
        vars_to_read.extend(wc_vars)
    # Aerosol data
    if any('UHSAS' in var for var in var_list) or any('CONCN' in var for var in var_list):
        aer_var = [var for var in var_list if ('UHSAS' in var or 'CONCU' in var or 'CONCN' in var) and ('AU' not in var and 'UD'  not in var and 'CUH' not in var and 'CFDC' not in var)]
        # uhsas_cells = var_list['CUHSAS_LWII'].CellSizes
        vars_to_read.extend(aer_var)
    # print("Loaded variables:")
    # print(vars_to_read)
    return vars_to_read

def read_sizedist_vars(nc):
    # Ensure we’re iterating over plain strings (variable names)
    names = list(nc.data_vars.keys())

    out = []
        # Include Time if present (coord or data var)
    if 'Time' in nc:
        out.append('Time')
        
    def add_prefix(prefix, exclude_substr=None):
        for n in names:
            if n.startswith(prefix) and (exclude_substr is None or exclude_substr not in n):
                out.append(n)

    # Cloud probe size distributions
    add_prefix('CCDP')
    add_prefix('C2DCA')
    add_prefix('C2DSA')          # <-- startswith enforces "at the start"

    # Aerosol size distributions
    add_prefix('CUHSAS') #, exclude_substr='CVI')   # exclude any CUHSAS* containing CVI
    add_prefix('CS200') # PCASP

    # Deduplicate while preserving original order
    out = list(dict.fromkeys(out))

    return out

def _prep_probe(nc, varname):
    da = nc[varname]
    # collapse any sps* to 1 Hz
    sps_dims = [d for d in da.dims if d.lower().startswith('sps')]
    if sps_dims:
        da = da.mean(dim=sps_dims, keep_attrs=True)
    # find bin dim and order (Time, Bin)
    bin_dim = next(d for d in da.dims if d.lower().startswith(('vector','bin','cell')))
    time_name = 'Time' if 'Time' in da.dims else 'time'
    da = da.transpose(time_name, bin_dim)

    # restrict to used bins
    first_bin = int(da.attrs.get('FirstBin', 0))
    last_bin  = int(da.attrs.get('LastBin', da.sizes[bin_dim]-1))
    da = da.isel({bin_dim: slice(first_bin, last_bin+1)})
    nbins = da.sizes[bin_dim]

    # upper edges for used bins
    cells_all = np.asarray(da.attrs.get('CellSizes', []), dtype=float)
    if cells_all.size == 0:
        raise ValueError(f"{varname} missing CellSizes attr")
    cells_used = cells_all[first_bin:last_bin+1]  # length == nbins

    return da, bin_dim, time_name, cells_used, nbins

def _sum_range_by_upper_edge(da, bin_dim, cells_used, lower_um=None, upper_um=None):
    """Sum across bins chosen by upper-edge thresholds."""
    nbins = da.sizes[bin_dim]

    if lower_um is None and upper_um is None:
        raise ValueError("Provide at least lower_um or upper_um")

    # choose start
    if lower_um is None:
        i0 = 0
    else:
        i0 = int(np.searchsorted(cells_used, lower_um, side='left'))

    # choose end (inclusive)
    if upper_um is None:
        i1 = nbins - 1
    else:
        i1 = int(np.searchsorted(cells_used, upper_um, side='right')) - 1

    i0 = np.clip(i0, 0, nbins - 1)
    i1 = np.clip(i1, 0, nbins - 1)

    if i1 < i0:
        # empty selection → NaNs (shape preserves time axis)
        return da.isel({bin_dim: slice(0, 0)}).sum(dim=bin_dim) * np.nan

    return da.isel({bin_dim: slice(i0, i1 + 1)}).sum(dim=bin_dim, skipna=True)

# def calc_concs_from_sd(sizedist_vars, nc):
#     cols = []

#     # --- C2DC branch (always compute Ndriz + Nprecip if present) ---
#     var_2dc = next((v for v in sizedist_vars if v.startswith('C2DC')), None)
#     if var_2dc is not None:
#         da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_2dc)
#         # drizzle: 100–500 µm
#         ndriz_2dc = _sum_range_by_upper_edge(da, bin_dim, cells_used, 100.0, 500.0)
#         # precip: ≥1000 µm
#         nprecip_2dc = _sum_range_by_upper_edge(da, bin_dim, cells_used, 1000.0, None)
#         t = pd.to_datetime(da[time_name].values)
#         df_2dc = pd.DataFrame({"Ndriz_2DC": ndriz_2dc.values, "Nprecip_2DC": nprecip_2dc.values},index=t)
#         df_2dc.index.name = "time"
#         cols.append(df_2dc)

#     # --- C2DS branch (optional; only Nprecip requested) ---
#     var_2ds = next((v for v in sizedist_vars if v.startswith('C2DS') and v.endswith('2H')), None)
#     if var_2ds is not None:
#         da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_2ds)
#         nprecip_2ds = _sum_range_by_upper_edge(da, bin_dim, cells_used, 1000.0, None)
#         ndriz_2ds = _sum_range_by_upper_edge(da, bin_dim, cells_used, 100.0, 500.0)
#         t = pd.to_datetime(da[time_name].values)
#         df_2ds = pd.DataFrame({"Ndriz_2DS": ndriz_2ds, "Nprecip_2DS": nprecip_2ds.values}, index=t)
#         df_2ds.index.name = "time"
#         cols.append(df_2ds)

#     # # # --- C2DS branch (optional; only Nprecip requested) ---
#     # var_uhsas = next((v for v in sizedist_vars if v.startswith('CUH')), None)
#     # if var_uhsas is not None:
#     #     da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_uhsas)
#     #     n_accum = _sum_range_by_upper_edge(da, bin_dim, cells_used, 100.0, None)
#     #     n_ait = _sum_range_by_upper_edge(da, bin_dim, cells_used, 70.0, 100.0)
#     #     t = pd.to_datetime(da[time_name].values)
#     #     df_uhs = pd.DataFrame({"Naitk_UH": n_ait, "Naccum_UH": n_accum.values}, index=t)
#     #     df_uhs.index.name = "time"
#     #     cols.append(df_uhs)
#     # if not cols:
#     #     return pd.DataFrame()

#     # # time-align and return
#     # return pd.concat(cols, axis=1).sort_index()

#     # --- C2DS / UHSAS branch (optional; only Nprecip requested) ---
#     uhsas_vars = [v for v in sizedist_vars if v.startswith("CUHSAS_")]  # or "CUH" if needed
#     if uhsas_vars is not None:
#         for var_uhsas in uhsas_vars:
#             da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_uhsas)

#             # thresholds are in µm (70 nm = 0.07 µm; 100 nm = 0.10 µm)
#             n_accum    = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.10, None)
#             n_ait      = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.07, 0.10)
#             n_ccn_prox = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.07, None)

        
#             suffix = var_uhsas.split("_")[-1]  # e.g. CVIU, LWII
#             t = pd.to_datetime(da[time_name].values)
        
#             df_uhs = pd.DataFrame(
#                 {
#                     f"Naitk_UH_{suffix}": np.asarray(n_ait),
#                     f"Naccum_UH_{suffix}": np.asarray(n_accum),
#                     f"Nccn_UH_{suffix}": np.asarray(n_ccn_prox),
#                 },
#                 index=t,
#             )
#             df_uhs.index.name = "time"
#             cols.append(df_uhs)
        
#         if not cols:
#             return pd.DataFrame()
    
#     # time-align and return
#     return pd.concat(cols, axis=1).sort_index()

def calc_concs_from_sd(sizedist_vars, nc, *, d_split_um=25.0):
    """
    Adds stitched Nliq using:
      - CDP: D < d_split_um
      - 2DS: D >= d_split_um
    Also keeps your existing drizzle/precip and UHSAS calculations.

    Requires helper functions you already have:
      _prep_probe, _sum_range_by_upper_edge
    """

    cols = []

    # -----------------------------
    # 2DC branch (always compute drizzle/precip if present)
    # -----------------------------
    var_2dc = next((v for v in sizedist_vars if v.startswith("C2DC")), None)
    if var_2dc is not None:
        da2, bin_dim2, time_name2, cells_used2, _ = _prep_probe(nc, var_2dc)

        # drizzle: 100–500 µm
        ndriz_2dc = _sum_range_by_upper_edge(da2, bin_dim2, cells_used2, 100.0, 500.0)
        # precip: ≥1000 µm
        nprecip_2dc = _sum_range_by_upper_edge(da2, bin_dim2, cells_used2, 1000.0, None)

        t2 = pd.to_datetime(da2[time_name2].values)
        df_2dc = pd.DataFrame(
            {
                "Ndriz_2DC": np.asarray(ndriz_2dc),
                "Nprecip_2DC": np.asarray(nprecip_2dc),
            },
            index=t2,
        )
        df_2dc.index.name = "time"
        cols.append(df_2dc)

    # -----------------------------
    # CDP branch (NEW): CDP contribution to stitched Nliq (< d_split_um)
    # -----------------------------
    # Try common CDP prefixes; adjust if your files use a different naming convention.
    var_cdp = next(
        (v for v in sizedist_vars
         if v.startswith("CCDP") or v.startswith("CDP") or v.startswith("CCDP_") or v.startswith("CCDP2")),
        None
    )

    if var_cdp is not None:
        dac, bin_dimc, time_namec, cells_usedc, _ = _prep_probe(nc, var_cdp)

        # CDP contribution below split diameter
        # Use lower bound 0.0 to capture all CDP bins below d_split_um.
        nliq_cdp_lt = _sum_range_by_upper_edge(dac, bin_dimc, cells_usedc, 0.0, d_split_um)

        tc = pd.to_datetime(dac[time_namec].values)
        df_cdp_nliq = pd.DataFrame(
            {f"Nliq_CDP_lt{int(d_split_um)}": np.asarray(nliq_cdp_lt)},
            index=tc,
        )
        df_cdp_nliq.index.name = "time"
        cols.append(df_cdp_nliq)

    # -----------------------------
    # 2DS branch (optional)
    # -----------------------------
    var_2ds = next((v for v in sizedist_vars if v.startswith("C2DS") and v.endswith("2H")), None)
    if var_2ds is not None:
        da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_2ds)
        nprecip_2ds = _sum_range_by_upper_edge(da, bin_dim, cells_used, 1000.0, None)
        ndriz_2ds = _sum_range_by_upper_edge(da, bin_dim, cells_used, 100.0, 500.0)
        t = pd.to_datetime(da[time_name].values)
        df_2ds = pd.DataFrame(
            {"Ndriz_2DS": np.asarray(ndriz_2ds), "Nprecip_2DS": np.asarray(nprecip_2ds)},
            index=t
        )
        df_2ds.index.name = "time"
        cols.append(df_2ds)

        # NEW: 2DC contribution to stitched Nliq (>= d_split_um)
        nliq_2ds_ge = _sum_range_by_upper_edge(da2, bin_dim2, cells_used2, d_split_um, None)
        df_2ds_nliq = pd.DataFrame(
            {f"Nliq_2DS_ge{int(d_split_um)}": np.asarray(nliq_2ds_ge)},
            index=t2,
        )
        df_2ds_nliq.index.name = "time"
        cols.append(df_2ds_nliq)

    # -----------------------------
    # UHSAS branch (optional)
    # -----------------------------
    uhsas_vars = [v for v in sizedist_vars if v.startswith("CUHSAS_")]  # adjust if needed
    if uhsas_vars:  # <-- fix: only loop if non-empty
        for var_uhsas in uhsas_vars:
            da, bin_dim, time_name, cells_used, _ = _prep_probe(nc, var_uhsas)

            n_accum    = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.10, None)
            n_ait      = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.07, 0.10)
            n_ccn_prox = _sum_range_by_upper_edge(da, bin_dim, cells_used, 0.07, None)

            suffix = var_uhsas.split("_")[-1]  # e.g. CVIU, LWII
            t = pd.to_datetime(da[time_name].values)

            df_uhs = pd.DataFrame(
                {
                    f"Naitk_UH_{suffix}": np.asarray(n_ait),
                    f"Naccum_UH_{suffix}": np.asarray(n_accum),
                    f"Nccn_UH_{suffix}": np.asarray(n_ccn_prox),
                },
                index=t,
            )
            df_uhs.index.name = "time"
            cols.append(df_uhs)

    if not cols:
        return pd.DataFrame()

    # -----------------------------
    # time-align and stitch Nliq if both parts exist
    # -----------------------------
    out = pd.concat(cols, axis=1).sort_index()

    cdp_name = f"Nliq_CDP_lt{int(d_split_um)}"
    twods_name = f"Nliq_2DS_ge{int(d_split_um)}"
    stitched_name = f"Nliq_stitched_{int(d_split_um)}um"

    if (cdp_name in out.columns) and (twods_name in out.columns):
        out[stitched_name] = out[cdp_name] + out[twods_name]

    return out

def load_flight_data(
    campaign: str,
    idx: int = 0,
    add_sizedist: bool = True,
    ccn_df: pd.DataFrame | None = None,
    add_sigma_w: bool = True,
    sigma_min_samples: int = 20,
    asof: bool = False,
    tol: str = "1s",
    icf_obj=None,
) -> pd.DataFrame:
    if icf_obj is None:
        raise ValueError("load_flight_data requires icf_obj (an AdfInfo-like object with .campaigns_dict)")
    cfg = _campaign_cfg_for(icf_obj.campaigns_dict, campaign)

    # --- 1 Hz aircraft ---
    flight_1hz_paths = find_flight_fnames(campaign, icf_obj=icf_obj)
    nc = open_nc(flight_1hz_paths[idx])

    vars2read = read_vars(nc)
    df = read_flight_nc(nc, vars2read)

    # --- CCN (SOCRATES only) ---
    if campaign == "SOCRATES" and ccn_df is not None:
        df = merge_two_ccn_streams_into_aircraft(df, ccn_df, tolerance="5s")

    # --- sigma(w) from 25 Hz (idx-aligned) ---
    if add_sigma_w:
        air25_dir = cfg.get("air_25hz_dir", None)
        if air25_dir is None:
            df["sigma_w"] = pd.NA
        else:
            flight_25hz_paths = find_flight_fnames(campaign, freq="air_25hz_dir", icf_obj=icf_obj)
            nc25 = open_nc(flight_25hz_paths[idx])

            w = nc25["WIC"]
            n = w.count(dim="sps25")
            sigma_w = w.std(dim="sps25", skipna=True).where(n >= sigma_min_samples)

            sigma_df = sigma_w.to_dataframe(name="Sigma_w").reset_index()

            # Merge into 1 Hz df on Time
            df2 = df.copy()
            df2["Time"] = pd.to_datetime(df2["Time"]).dt.tz_localize(None).round("S")

            sigma_df["Time"] = pd.to_datetime(sigma_df["Time"]).dt.tz_localize(None).round("S")

            df = df2.merge(sigma_df[["Time", "Sigma_w"]], on="Time", how="left")
            df = df.reindex(columns=df.columns.tolist()[:df.columns.get_loc("WIC")+1] + ["Sigma_w"] +  df.columns.tolist()[df.columns.get_loc("WIC")+1:-1])
    # --- size dist derived vars ---
    if add_sizedist:
        sd_vars = read_sizedist_vars(nc)
        conc_df = calc_concs_from_sd(sd_vars, nc)

        if conc_df is not None and not conc_df.empty:
            df2 = df.copy()
            df2["Time"] = pd.to_datetime(df2["Time"]).dt.tz_localize(None).round("S")

            conc = conc_df.copy()
            conc.index = pd.to_datetime(conc.index).tz_localize(None).round("S")

            if asof:
                df = pd.merge_asof(
                    df2.sort_values("Time"),
                    conc.reset_index().rename(columns={"index": "Time"}).sort_values("Time"),
                    on="Time",
                    direction="nearest",
                    tolerance=pd.Timedelta(tol),
                )
            else:
                df = df2.set_index("Time").join(conc, how="left").reset_index()

    return df


def find_sondes(dir_path: str) -> list[str]:
    """
    Search a directory for dropsonde files (.nc and .cls).
    Parameters
    ----------
    dir_path : str
        Directory containing dropsonde files.
    Returns
    -------
    list[str]
        Sorted list of full file paths.
    """
    valid_ext = (".nc", ".cls",".eol")
    fnames = sorted(
        fname for fname in os.listdir(dir_path)
        if fname.lower().endswith(valid_ext)
    )
    return [os.path.join(dir_path, fname) for fname in fnames]

PathLike = Union[str, Path]

def _assign_rf_cset(launch_time):
    """
    Assign CSET research flight number from launch datetime.
    Uses hard-coded RF day mapping.
    """
    if launch_time is None:
        return None
    # Use UTC date
    d = launch_time.date()
    rf_map = {
        (2015, 7, 1): 1,
        (2015, 7, 7): 2,
        (2015, 7, 9): 3,
        (2015, 7, 12): 4,
        (2015, 7, 14): 5,
        (2015, 7, 17): 6,
        (2015, 7, 19): 7,
        (2015, 7, 22): 8,
        (2015, 7, 24): 9,
        (2015, 7, 27): 10,
        (2015, 7, 29): 11,
        (2015, 8, 1): 12,
        (2015, 8, 3): 13,
        (2015, 8, 7): 14,
        (2015, 8, 9): 15,
        (2015, 8, 12): 16,
    }
    return rf_map.get((d.year, d.month, d.day), None)

def _infer_rf_from_name(name: str) -> Optional[int]:
    m = re.search(r"RF(\d{1,2})", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _sort_key_default(p: Path) -> str:
    # Keep it simple/stable: filename sort
    return p.name

def _read_cls_multi(file_path: Path) -> Tuple[List[pd.DataFrame], List[pd.Timestamp]]:
    """
    Reads a `.cls` radiosonde file and extracts multiple datasets with nominal release times.
    Returns (datasets, nominal_times).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start_indices = [i for i, line in enumerate(lines) if "Nominal Release Time" in line]
    if not start_indices:
        raise ValueError(f"No 'Nominal Release Time' entries found in file: {file_path}")

    datasets: List[pd.DataFrame] = []
    nominal_times: List[pd.Timestamp] = []

    for idx, start in enumerate(start_indices):
        # Find header row ("Time ... Press ...")
        data_start = None
        header_i = None
        for i in range(start, len(lines) - 2):
            if lines[i].strip().startswith("Time") and "Press" in lines[i]:
                header_i = i
                data_start = i + 3  # matches your original logic
                break
        if data_start is None or header_i is None:
            # Skip gracefully
            continue

        # Parse nominal release time
        try:
            date_time_str = lines[start].split("):")[1].strip()
            drop_time = pd.to_datetime(date_time_str, format="%Y, %m, %d, %H:%M:%S")
        except Exception:
            continue

        # Columns come from the header line itself
        columns = lines[header_i].strip().split()

        end = start_indices[idx + 1] if idx + 1 < len(start_indices) else len(lines)

        data_lines = [ln.strip().split() for ln in lines[data_start:end]]
        data = [row for row in data_lines if len(row) == len(columns)]
        if not data:
            continue

        df = pd.DataFrame(data, columns=columns)

        # Remove 9999.0 sentinels (string or numeric after coercion)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.replace([9999, 9999.0], np.nan).dropna(how="any")

        if df.empty:
            continue

        df.attrs["nominal_time"] = drop_time
        datasets.append(df)
        nominal_times.append(drop_time)

    return datasets, nominal_times

def _read_eol_one(file_path, campaign=None):
    """
    Read one EOL Sounding Format/1.1 dropsonde file (single drop per file).
    Returns (df, launch_time_utc).

    Output df includes a real datetime column 'Time' built from launch_time + seconds.
    Also renames Lon/Lat -> GGLON/GGLAT for consistency with your other pipeline.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with p.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # ---- 1) parse launch time ----
    launch_time = None
    for line in lines[:200]:
        if "UTC Launch Time" in line:
            # e.g. "UTC Launch Time (y,m,d,h,m,s):             2015, 07, 01, 18:08:29"
            rhs = line.split(":", 1)[1].strip()
            try:
                launch_time = pd.to_datetime(rhs, format="%Y, %m, %d, %H:%M:%S", utc=True)
            except Exception:
                # fallback: let pandas try
                launch_time = pd.to_datetime(rhs, utc=True, errors="coerce")
            break
    if launch_time is None or pd.isna(launch_time):
        raise ValueError(f"Could not parse 'UTC Launch Time' from header in {p.name}")

    # ---- 2) find table header row (the one with column names) ----
    header_i = None
    for i, line in enumerate(lines):
        s = line.strip()
        # Your file shows: "Time   -- UTC  --   Press    Temp ..."
        if s.startswith("Time") and "Press" in s and "Lon" in s and "Lat" in s:
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"Could not find data header row in {p.name}")

    columns = [
    "Time", "hh", "mm", "ss",
    "Press", "Temp", "Dewpt", "RH",
    "Uwind", "Vwind", "Wspd", "Dir",
    "dZ", "GeoPoAlt", "Lon", "Lat", "GPSAlt"
    ]

    # ---- 3) data starts after the dashed separator line ----
    dash_i = None
    for j in range(header_i, min(header_i + 10, len(lines))):
        if re.match(r"^-{5,}", lines[j].strip()):
            dash_i = j
            break
    if dash_i is None:
        raise ValueError(f"Could not find dashed separator below header in {p.name}")

    data_start = dash_i + 1

    # ---- 4) parse rows ----
    raw_rows = [ln.strip().split() for ln in lines[data_start:] if ln.strip()]
    rows = [r for r in raw_rows if len(r) == len(columns)]
    if not rows:
        raise ValueError(f"No data rows parsed in {p.name} (expected {len(columns)} columns)")

    df = pd.DataFrame(rows, columns=columns).apply(pd.to_numeric, errors="coerce")

    # ---- 5) handle missing sentinels (EOL commonly uses -999) ----
    df = df.replace([-999, -999.0, 9999, 9999.0, -9999, -9999.0], np.nan)

    # ---- 6) build actual datetime 'Time' from seconds since launch ----
    # In your example, first column is "Time" in seconds (can be negative at first record)
    if "Time" not in df.columns:
        raise ValueError(f"'Time' (seconds) column not found in {p.name}")

    # launch_time is UTC; keep timezone-aware unless you want naive
    df["Time"] = launch_time + pd.to_timedelta(df["Time"].astype(float), unit="s")

    # ---- 7) rename Lon/Lat to match your convention ----
    rename_map = {}
    if "Lon" in df.columns:
        rename_map["Lon"] = "GGLON"
    if "Lat" in df.columns:
        rename_map["Lat"] = "GGLAT"
    df = df.rename(columns=rename_map)

    # store attrs
    df.attrs["launch_time"] = launch_time
    df.attrs["nominal_time"] = launch_time  # if you want to treat nominal=launch for .eol
    # --- assign RF for CSET ---
    if campaign is not None and campaign.upper() == "CSET":
        rf = _assign_rf_cset(launch_time)
        df["RF"] = rf
    # put Time first
    lead = ["Time"] + [c for c in ("GGLAT", "GGLON") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    return df, launch_time

def _read_nc_one(
    file_path: Path,
    *,
    include_reference: bool,
    reference_prefix: str,
    broadcast_scalars: bool,
    keep_time_as_column: bool,
) -> Tuple[pd.DataFrame, Optional[pd.Timestamp]]:
    ds = xr.open_dataset(file_path)
    print(__file__,"file_path",file_path)
    n = ds.sizes.get("time", None)
    if n is None:
        ds.close()
        raise ValueError(f"{file_path} has no 'time' dimension.")

    cols: Dict[str, np.ndarray] = {}

    # coords varying with time
    for cname, coord in ds.coords.items():
        if "time" in coord.dims:
            cols[cname] = coord.values

    # data_vars varying with time
    for vname, da in ds.data_vars.items():
        if "time" in da.dims and da.ndim == 1:
            cols[vname] = da.values

    if broadcast_scalars:
        for vname, da in ds.data_vars.items():
            if da.ndim == 0:
                cols[vname] = np.repeat(da.item(), n)
        for cname, coord in ds.coords.items():
            if coord.ndim == 0 and cname not in cols:
                cols[cname] = np.repeat(coord.item(), n)

    if include_reference:
        if ds.sizes.get("obs", None) == 1:
            for vname, da in ds.data_vars.items():
                if "obs" in da.dims and da.ndim == 1:
                    val = da.isel(obs=0).values
                    out_name = vname if vname.startswith("reference_") else f"{reference_prefix}{vname}"
                    cols[out_name] = np.repeat(val, n)

    df = pd.DataFrame(cols)

    if keep_time_as_column:
        if "time" not in df.columns and "time" in ds:
            df["time"] = ds["time"].values
    else:
        if "time" in df.columns:
            df = df.set_index(pd.to_datetime(df["time"]))

    # Standardize names (your current renames)
    df = df.rename(columns={"time": "Time", "lat": "GGLAT", "lon": "GGLON"})

    # Launch/nominal time from dataset if present
    lt: Optional[pd.Timestamp] = None
    if "launch_time" in ds:
        try:
            lt = pd.to_datetime(ds["launch_time"].item())
        except Exception:
            lt = None

    df.attrs["nominal_time"] = lt
    ds.close()
    return df, lt

def read_sonde(
    paths: Union[PathLike, xr.Dataset, Iterable[Union[PathLike, xr.Dataset]]],
    *,
    # ---- nc behavior (kept from your function) ----
    campaign: Optional[str] = None,  # <-- ADD THIS
    include_reference: bool = True,
    reference_prefix: str = "ref_",
    broadcast_scalars: bool = True,
    keep_time_as_column: bool = True,
    # ---- cls behavior ----
    add_rf_and_dropnum: bool = True,
    # ---- optional extension hooks for .ict/.eol if you have a reader elsewhere ----
    ict_eol_reader: Optional[
        Callable[[Path], Tuple[Union[pd.DataFrame, List[pd.DataFrame]], Union[pd.Timestamp, List[pd.Timestamp], None]]]
    ] = None,
    # ---- sorting within RF ----
    sort_key: Callable[[Path], object] = _sort_key_default,
) -> Tuple[List[pd.DataFrame], List[Optional[pd.Timestamp]]]:
    """
    Unified reader for dropsonde/radiosonde inputs.

    Accepts:
      - .nc : reads one file -> one DataFrame
      - .cls: reads one file -> many DataFrames (multiple sondes inside)
      - .ict/.eol: optional via `ict_eol_reader(path)` hook

    Returns:
      dfs, nominal_times  (same length, aligned)
    """

    # Normalize input
    if isinstance(paths, (str, Path, xr.Dataset)):
        paths = [paths]

    # Split xarray datasets vs filesystem paths
    path_list: List[Path] = []
    ds_list: List[xr.Dataset] = []

    for p in paths:
        if isinstance(p, xr.Dataset):
            ds_list.append(p)
        else:
            path_list.append(Path(p))

    dfs: List[pd.DataFrame] = []
    nominal_times: List[Optional[pd.Timestamp]] = []

    # ---- Handle provided xr.Datasets (treated like .nc content) ----
    for ds in ds_list:
        n = ds.sizes.get("time", None)
        if n is None:
            continue
        cols: Dict[str, np.ndarray] = {}
        for cname, coord in ds.coords.items():
            if "time" in coord.dims:
                cols[cname] = coord.values
        for vname, da in ds.data_vars.items():
            if "time" in da.dims and da.ndim == 1:
                cols[vname] = da.values
        if broadcast_scalars:
            for vname, da in ds.data_vars.items():
                if da.ndim == 0:
                    cols[vname] = np.repeat(da.item(), n)
        df = pd.DataFrame(cols).rename(columns={"time": "Time", "lat": "GGLAT", "lon": "GGLON"})
        lt = None
        if "launch_time" in ds:
            try:
                lt = pd.to_datetime(ds["launch_time"].item())
            except Exception:
                lt = None
        df.attrs["nominal_time"] = lt
        dfs.append(df)
        nominal_times.append(lt)

    # ---- Read everything first, then assign RF + drop_num per RF ----
    records: List[Tuple[int, pd.Timestamp, pd.DataFrame, Optional[pd.Timestamp]]] = []
    # tuple = (RF, nominal_time_for_sort, df, nominal_time_return)
    
    for fp in sorted(path_list, key=sort_key):
        print(__file__,"fp - File?",fp)
        ext = fp.suffix.lower()
    
        if ext == ".cls":
            cls_dfs, cls_times = _read_cls_multi(fp)
            for df, t in zip(cls_dfs, cls_times):
                # nominal time exists in .cls
                nominal = pd.to_datetime(t, utc=True, errors="coerce")
    
                # RF: prefer filename RF, else CSET mapping from nominal time if campaign is CSET
                rf = _infer_rf_from_name(fp.name)
                if rf is None and campaign is not None and campaign.upper() == "CSET" and pd.notna(nominal):
                    rf = _assign_rf_cset(nominal.to_pydatetime())
    
                # Skip if RF still unknown (or set to -1 if you prefer keeping them)
                if rf is None:
                    rf = -1
    
                # Store
                records.append((int(rf), nominal, df, t))
    
        elif ext == ".eol":
            df, t = _read_eol_one(fp, campaign)   # t is launch_time (UTC)
            nominal = pd.to_datetime(t, utc=True, errors="coerce")
    
            # RF: _read_eol_one already sets df["RF"] for CSET, but we standardize here
            rf = None
            if "RF" in df.columns and pd.notna(df["RF"].iloc[0]):
                rf = int(df["RF"].iloc[0])
            else:
                rf = _infer_rf_from_name(fp.name)
                if rf is None and campaign is not None and campaign.upper() == "CSET" and pd.notna(nominal):
                    rf = _assign_rf_cset(nominal.to_pydatetime())
    
            if rf is None:
                rf = -1
    
            records.append((int(rf), nominal, df, t))
    
        elif ext == ".nc":
            df, t = _read_nc_one(
                fp,
                include_reference=include_reference,
                reference_prefix=reference_prefix,
                broadcast_scalars=broadcast_scalars,
                keep_time_as_column=keep_time_as_column,
            )
            nominal = pd.to_datetime(t, utc=True, errors="coerce") if t is not None else pd.NaT
    
            rf = _infer_rf_from_name(fp.name)
            if rf is None:
                rf = -1
    
            records.append((int(rf), nominal, df, t))
    
        else:
            print(f"⚠️ Skipping {fp.name}: unsupported extension '{ext}'")
    
    # ---- Now assign drop_num per RF (resets within each flight) ----
    # Sort by (RF, nominal_time). Put unknown RF (-1) last.
    def _rec_key(rec):
        rf, nominal, _, _ = rec
        rf_sort = 9999 if rf == -1 else rf
        t_sort = pd.Timestamp.max if pd.isna(nominal) else nominal
        return (rf_sort, t_sort)
    
    records = sorted(records, key=_rec_key)
    
    dfs = []
    nominal_times = []
    
    drop_counters: Dict[int, int] = {}
    
    for rf, nominal, df, t_return in records:
        if add_rf_and_dropnum:
            df = df.copy()
    
            # set RF column consistently
            df["RF"] = rf
    
            # increment drop counter per RF (skip -1 if you want)
            if rf == -1:
                drop_num = -1
            else:
                drop_counters[rf] = drop_counters.get(rf, 0) + 1
                drop_num = drop_counters[rf]
    
            df["drop_num"] = drop_num
    
            # standard leading columns
            lead = [c for c in ["Time", "RF", "drop_num"] if c in df.columns]
            df = df[lead + [c for c in df.columns if c not in lead]]
    
        dfs.append(df)
        nominal_times.append(t_return)
    
    return dfs, nominal_times

#======================================================================================
# ERA5 cloud controlling factors and cloud regimes for dropsondes, and the dropsonde
# composite (<campaign>_sonde_data_composite.nc) built from them.
#
# select_ERA5_4flight, wrap180, nearest_time_indices, collocate_ERA5_sonde and
# cloud_regime_sonde are ported from the cookbook's process_data_products_utils.py
# (see lib/test/inform_reference/).  select_ERA5_4flight differs only in using this
# module's `datetime`/`timedelta` names (imported from the datetime module) instead
# of `datetime.datetime`/`datetime.timedelta`.  build_sonde_composite and
# sonde_composite_dataset follow the sonde cells of the cookbook notebook
# INFORM_process_system_database.ipynb.
#======================================================================================

def select_ERA5_4flight(df, campaign, dat_type="aircraft"):
    # Define function to filter ERA5 files based on time
    def get_matching_files(pattern, start_dt, end_dt):
        file_list = glob.glob(pattern)
        selected = []
        for file in file_list:
            time_strs = file.split('.')[-2].split('_')
            file_start = datetime.strptime(time_strs[0], "%Y%m%d%H")
            file_end = datetime.strptime(time_strs[1], "%Y%m%d%H")
            if file_start <= end_dt and file_end >= start_dt:
                selected.append(file)
        return selected
    
    filepath_sfc = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"
    filepath_pl = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    # Extract the times of the research flight
    month, year = df.Time[0].month, df.Time[0].year
    day_start,day_end = df.Time[0].day, df.Time.iloc[-1].day
    start_hour, end_hour = df.Time[0].hour, df.Time.iloc[-1].hour
    
    # Select the latitude/longitude box to reduce size of era5 data
    if campaign == 'SOCRATES':
        lat_max, lat_min = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())
    elif campaign == 'CSET':
        lat_min, lat_max = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())
        
    # ---- build lat/lon selection from the flight track ----
    # Aircraft → 0..360 to match ERA5
    lon0 = ((df.GGLON.to_numpy(dtype=float) % 360.0) + 360.0) % 360.0
    lat0 = df.GGLAT.to_numpy(dtype=float)
    
    # Robust bounds (with a small pad for ERA5 0.25° grid)
    pad = 0.5
    lon_min = float(np.floor(np.nanmin(lon0) - pad))
    lon_max = float(np.ceil (np.nanmax(lon0) + pad))
    # keep inside ERA5 domain
    lon_min = max(0.0, lon_min)
    lon_max = min(359.999, lon_max)
    
    lat_min = float(np.floor(np.nanmin(lat0) - pad))
    lat_max = float(np.ceil (np.nanmax(lat0) + pad))
    
    # ERA5 latitude is usually descending (90 → -90): use slice(max, min)
    lat_slice = slice(lat_max, lat_min)
    
    # ERA5 longitudes are ascending (0 → 360): use slice(min, max)
    lon_slice = slice(lon_min, lon_max)
    
    print("Selecting ERA5 box:",
      f"lon {lon_min}→{lon_max} (0–360), lat {lat_max}→{lat_min} (descending)")

    # Make the yearmonth string for file selection
    dir_date = f"{year}{month:02d}"
    
    # Flight start and end times
    start_dt = datetime(year, month, day_start, start_hour) 
    end_dt = datetime(year, month, day_end, end_hour)+timedelta(hours=1)

    # ---- apply the SAME slices to every dataset you open ----
    ds_sp  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_sp.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_sst  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_sstk.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t2m  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_2t.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_u10  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_10u.*.nc", start_dt, end_dt),
                                combine='by_coords')[['VAR_10U']].sel(latitude=lat_slice, 
                                                                      longitude=lon_slice,time=slice(start_dt,end_dt))
    ds_v10  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_10v.*.nc", start_dt, end_dt),
                                combine='by_coords')[['VAR_10V']].sel(latitude=lat_slice, longitude=lon_slice, 
                                                                      time=slice(start_dt,end_dt))
    
    ds_w    = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_w.*.nc",  start_dt, end_dt),
                                combine='nested', concat_dim='time')
    w_700   = ds_w['W'].sel(level=700).sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    
    # ds_rh700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_r.*.nc", start_dt, end_dt),
    #                              combine='nested', concat_dim='time')[['R']].sel(level=700).drop_vars('level', errors='ignore')
    # rh      = ds_rh700['R'].rename('RH').sortby('time').sel(latitude=lat_slice, longitude=lon_slice, 
    # time=slice(start_dt,end_dt))
    ds_q700  = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_q.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['Q']].sel(level=700).drop_vars('level', errors='ignore')
    q       = ds_q700['Q'].rename('Q').sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_u700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_u.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['U']].sel(level=700).drop_vars('level', errors='ignore')
    ds_u700 = ds_u700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_v700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_v.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['V']].sel(level=700).drop_vars('level', errors='ignore')
    ds_v700 = ds_v700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    
    ds_t    = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=800).drop_vars('level', errors='ignore')
    ds_t    = ds_t.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=700).drop_vars('level', errors='ignore')
    ds_t700 = ds_t700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t850 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=850).drop_vars('level', errors='ignore')
    ds_t850 = ds_t700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))

    ws = np.sqrt(ds_u10.VAR_10U**2 + ds_v10.VAR_10V**2)
    wind_dir = (270 - np.degrees(np.arctan2(ds_v10.VAR_10V, ds_u10.VAR_10U))) % 360
    
    # Calculate RH from specfic humidity
    # Murphy & Koop (2005) saturation vapor pressure (Pa)
    def es_MK_water(T):
        # ln(esw [Pa]) valid ~123–332 K
        return xr.ufuncs.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
                             + xr.ufuncs.tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
    
    def es_MK_ice(T):
        # ln(esi [Pa]) valid ~110–273 K
        return xr.ufuncs.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
    
    esw = es_MK_water(ds_t700.T)
    esi = es_MK_ice(ds_t700.T)
    # Choose water above freezing, ice at/below (adjust threshold if you prefer 273.16)
    es = xr.where(ds_t700.T > 273.15, esw, esi)
    
    # Saturation specific humidity and RH
    eps = 0.622
    qsat = (eps * es) / (70000 - (1.0 - eps) * es)
    
    # Avoid division issues extremely near saturation/low p
    qsat = qsat.clip(min=1e-12)
    
    RH = (q / qsat) * 100.0

    # Calcualte wind shear (SFC - 700mb)
    ws700 = np.sqrt(ds_u700.U**2 + ds_v700.V**2)
    wind_shear = ws700-ws
    
    # Calculate M-value
    Rd = 287
    Cp = 1005   
    theta_sfc = ds_t2m.VAR_2T*(101325/ds_sp.SP)**(Rd/Cp)
    theta_800 = ds_t*(1013.25/800)**(Rd/Cp)
    
    M = theta_sfc.T - theta_800.T
    M = M.transpose("time", "latitude", "longitude")
    
    dt = ds_t2m.VAR_2T - ds_sst.SSTK
    
    # Constants
    Re = 6.371e6  # Earth radius in meters
    deg2rad = np.pi / 180
    phi = np.deg2rad(ds_sst.SSTK['latitude'])
    # meters per 1° at this latitude
    m_per_deg_lon = Re * np.cos(phi) * deg2rad
    m_per_deg_lat = Re * deg2rad

    if dat_type == "dropsonde": # Necessary to calculate delta x/y for dropsonde which only returns one column
        # --- SST gradients & Tadv (K/day) ---
        ds_sst2 = ds_sst.sortby(["latitude", "longitude"]).unify_chunks().chunk({"latitude": -1, "longitude": -1, "time": -1})

        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst2.SSTK.differentiate("longitude") / m_per_deg_lon   # K/m
        dT_dy = ds_sst2.SSTK.differentiate('latitude') / m_per_deg_lat   # K/m

    elif dat_type == "aircraft":
        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst.SSTK.differentiate("longitude") / m_per_deg_lon   # K/m
        dT_dy = ds_sst.SSTK.differentiate('latitude') / m_per_deg_lat   # K/m
 
    # advection: K/s -> K/day
    Tadv = -(ds_u10['VAR_10U'] * dT_dx + ds_v10['VAR_10V'] * dT_dy) * 86400.0
    # Convert to K/day
    Tadv = Tadv.rename("Tadv")

    # Calculate EIS following Wood and Bretherton (2006, J. Climate)
    cp = 1004.     # specific heat at constant pressure for dry air (J / kg / K)
    Rd = 287.         # gas constant for dry air (J / kg / K)
    kappa = Rd / cp
    Lhvap = 2.5e6    # Latent heat of vaporization (J / kg)
    g = 9.81 # m/s^2
    cp = 1004 # J/K/kg
    Lv = 2.5e6 # J/kg
    
    Rv = 461 # J/K/kg;
    Ra = 287 # J/K/kg
    
    def get_qsat(T,p):
        Tcel = T-273.15
        es=6.11*10**(7.5*Tcel/(Tcel+273.15))
        return 0.622*es/p
    
    # Calculate lower tropospheric stability (LTS)
    theta_700 = ds_t700.T*(1013.25/700)**kappa
    LTS = theta_700 - theta_sfc
    
    # T850 = (ds_t2m.VAR_2T+ds_t700.T)/2
    T850 = ds_t850.T
    
    Gammam = (g/cp*(1.0 - (1.0 + Lhvap*get_qsat(T850,850) / Rd / T850) /
                 (1.0 + Lhvap**2 * get_qsat(T850,850)/ cp/Rv/T850**2)))
    
    # Assume exponential decrease of pressure with scale height given by surface temperature
    z700 = (Rd * ds_t2m.VAR_2T / g) * np.log(1000 / 700)
    # Assume 80% relative humidity to compute LCL, appropriate for marine boundary layer
    Tadj = Tadj = ds_t2m.VAR_2T-55.  # in Kelvin
    LCL = cp/g*(Tadj - (1/Tadj - np.log(0.8)/2840.)**(-1))    
    EIS = LTS - Gammam*(z700 - LCL)

    # Convert w700 (m/s) to pa/s
    omega700 = -(70000 / (Rd * ds_t700.T)) * g * w_700
    
    # Merge the dataset variables used later
    ds = {
    'deltaT': dt,
    'Tadv': Tadv,
    'M': M,
    'omega700': omega700,
    'SST': ds_sst.SSTK,
    'WS': ws,
    'Wind_shear': wind_shear,
    'RH700': RH,
    'EIS': EIS
     }

    return ds

def wrap180(lon):
    # Map any longitude to [-180, 180)
    return (lon + 180.0) % 360.0 - 180.0

def nearest_time_indices(era5_times_ns, flight_times_ns):
    # era5_times_ns: 1D int64 nanoseconds, sorted
    # flight_times_ns: 1D int64 nanoseconds
    idx_right = np.searchsorted(era5_times_ns, flight_times_ns, side="left")
    idx_left  = np.clip(idx_right - 1, 0, len(era5_times_ns) - 1)
    idx_right = np.clip(idx_right,       0, len(era5_times_ns) - 1)
    choose_right = np.abs(era5_times_ns[idx_right] - flight_times_ns) < np.abs(era5_times_ns[idx_left] - flight_times_ns)
    return np.where(choose_right, idx_right, idx_left)

def collocate_ERA5_sonde(ds, df):
    """
    Collocate ERA5 gridded fields with dropsonde observations using
    nearest-neighbor matching in space and time.

    For each dropsonde observation, the function:
        1. Finds the nearest ERA5 grid point using a KDTree built from
           the ERA5 latitude/longitude grid.
        2. Finds the nearest ERA5 time step to the sonde observation time.
        3. Extracts ERA5 variables at that grid point and time.
        4. Appends those values as new columns to the dropsonde dataframe.

    Longitude matching is performed in a [-180°, 180°) coordinate system
    to avoid issues near the dateline.

    Parameters
    ----------
    ds : xarray.Dataset or dict-like
        ERA5 dataset containing the variables to collocate. Expected
        dimensions are ('time', 'latitude', 'longitude').

        Required variables:
            SST
            M
            omega700
            deltaT
            WS
            Wind_shear
            Tadv
            RH700
            EIS

    df : pandas.DataFrame
        Dropsonde dataframe containing observation coordinates and time.

        Required columns:
            GGLAT   : latitude (degrees)
            GGLON   : longitude (degrees)
            Time    : observation time (datetime-like)

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with additional columns containing collocated ERA5
        values at the nearest grid point and time:

            ERA5_SST
            M
            Omega700
            deltaT
            Wind_sp
            Wind_shear
            Tadv
            RH700
            EIS
    Notes
    -----
    - Spatial matching uses Euclidean distance in lat/lon space via
      scipy.spatial.cKDTree.
    - Temporal matching uses nearest neighbor in ERA5 time.
    - Rows with missing latitude, longitude, or time receive NaN values
      for all collocated variables.
    - ERA5 variables are loaded into memory as NumPy arrays for fast
      indexing.
    """
    
    # ---------------- core collocation block ----------------
    ds = xr.Dataset(ds)
    
    # ERA5 coords
    lat_vals = ds['latitude'].values.astype(float)
    lon_vals_ds = ds['longitude'].values.astype(float)
    
    # Build KDTree in [-180,180) to avoid wrap issues
    lon_vals_wrapped = wrap180(lon_vals_ds)
    lon_grid, lat_grid = np.meshgrid(lon_vals_wrapped, lat_vals, indexing="xy")  # (nx,ny) if xy; use ij below
    # use ij orientation for unravel consistency:
    YY, XX = np.meshgrid(lat_vals, lon_vals_wrapped, indexing="ij")  # (ny,nx)
    tree = cKDTree(np.c_[YY.ravel(), XX.ravel()])
    ny, nx = YY.shape  # (lat, lon)
    
    # ERA5 time (sorted)
    t_era = ds['time'].values.astype('datetime64[ns]')
    t_era_ns = t_era.view('int64')
    
    # Preload arrays into NumPy (T,Y,X) for fast indexing
    def arr3(name):
        return ds[name].transpose('time', 'latitude', 'longitude').compute().values
    
    arr = {
        'ERA5_SST':     arr3('SST'),
        'M':            arr3('M'),
        'Omega700':     arr3('omega700'),
        'deltaT':       arr3('deltaT'),
        'Wind_sp':      arr3('WS'),
        'Wind_shear':   arr3('Wind_shear'),
        'Tadv':         arr3('Tadv'),
        'RH700':        arr3('RH700'),
        'EIS':          arr3('EIS'),
    }
    
    # Flight coords/time
    flt_lat = df['GGLAT'].to_numpy(float)
    flt_lon = wrap180(df['GGLON'].to_numpy(float))  # match KDTree frame
    flt_t   = pd.to_datetime(df['Time'].values).to_numpy('datetime64[ns]')
    flt_t_ns = flt_t.view('int64')
    
    N = len(df)
    
    # Mask rows we can sample (ignore NaNs)
    valid = np.isfinite(flt_lat) & np.isfinite(flt_lon) & np.isfinite(flt_t_ns)
    
    # If nothing valid, just make the output columns full NaN and return df as-is
    if not np.any(valid):
        for out_name in arr.keys():
            df[out_name] = np.nan
    else:
        # Nearest time indices for valid rows only
        ti_valid = nearest_time_indices(t_era_ns, flt_t_ns[valid])  # (M,)
    
        # Nearest gridpoint for valid rows
        _, flat_idx = tree.query(np.c_[flt_lat[valid], flt_lon[valid]])  # (M,)
        yi, xi = np.unravel_index(flat_idx, (ny, nx))                    # (M,), (M,)
    
        # Gather each variable and scatter back into full-length columns (NaN elsewhere)
        for out_name, A in arr.items():   # A: (T,ny,nx)
            vals_valid = A[ti_valid, yi, xi]              # (M,)
            out_full = np.full(N, np.nan, dtype=float)    # default NaN
            out_full[valid] = vals_valid
            df[out_name] = out_full

    return df

def cloud_regime_sonde(df, campaign, min_valid=5):
    """
    Assign a single cloud_regime label to an entire dropsonde dataframe
    based on block-mean (NaN-excluded) cloud controlling factors.

    Parameters
    ----------
    df : pandas.DataFrame
        Dropsonde dataframe containing ERA5_SST, M, RH700, Tadv, Wind_sp,
        Wind_shear, EIS, etc.
    campaign : str
        'SOCRATES' or 'CSET'
    min_valid : int
        Minimum required valid samples when computing mean()

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with new column 'cloud_regime' filled with a single label
    """

    def mean_if_enough(x, nmin=min_valid):
        """Return mean(x) if enough valid samples exist, else NaN."""
        vals = x.to_numpy(dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).sum() >= nmin else np.nan

    # --- compute block means ---
    M_mean          = mean_if_enough(df.get("M"))
    RH700_mean      = mean_if_enough(df.get("RH700"))
    SST_mean        = mean_if_enough(df.get("ERA5_SST"))
    Tadv_mean       = mean_if_enough(df.get("Tadv"))
    Wind_sp_mean    = mean_if_enough(df.get("Wind_sp"))
    Wind_shear_mean = mean_if_enough(df.get("Wind_shear"))
    EIS_mean        = mean_if_enough(df.get("EIS"))

    # default label
    label = "Unknown"

    # ===========================
    #     SOCRATES RULE SET
    # ===========================
    if campaign.upper() == "SOCRATES":

        # --- Open-cell cumulus ---
        cond_open = (
            # 1) M >= -7 & WS >= 9
            (np.isfinite(M_mean) and np.isfinite(Wind_sp_mean) and
             M_mean >= -7 and Wind_sp_mean >= 9)
            or
            # 2) M >= -8 & EIS < 9
            (np.isfinite(M_mean) and np.isfinite(EIS_mean) and
             M_mean >= -8 and EIS_mean < 9)
            or
            # 3) M >= -8 & wshear > 6
            (np.isfinite(M_mean) and np.isfinite(Wind_shear_mean) and
             M_mean >= -8 and Wind_shear_mean > 6)
        )

        # --- Stratocumulus ---
        cond_strat = (
            # 1) M < -9 & WS < 9
            (np.isfinite(M_mean) and np.isfinite(Wind_sp_mean) and
             M_mean < -9 and Wind_sp_mean < 9)
            or
            # 2) M < -10 & EIS > 7
            (np.isfinite(M_mean) and np.isfinite(EIS_mean) and
             M_mean < -10 and EIS_mean > 7)
            or
            # 3) M < -10 & wshear < 9
            (np.isfinite(M_mean) and np.isfinite(Wind_shear_mean) and
             M_mean < -10 and Wind_shear_mean < 9)
        )

        if cond_open:
            label = "Open-Cell"
        if cond_strat:
            # Stratocumulus overrides if both are true
            label = "Stratocumulus"

    # ===========================
    #        CSET RULE SET
    # ===========================
    elif campaign.upper() == "CSET":

        cond_strat = (
            (np.isfinite(M_mean) and np.isfinite(SST_mean)  and M_mean < -10 and SST_mean < 295)
            or
            (np.isfinite(M_mean) and np.isfinite(Tadv_mean) and M_mean < -10 and Tadv_mean < 0)
        )

        cond_opencu = (
            (np.isfinite(M_mean) and np.isfinite(SST_mean)  and M_mean >= -10 and SST_mean >= 296)
            or
            (np.isfinite(M_mean) and np.isfinite(Tadv_mean) and M_mean >= -4)
        )

        if cond_strat:
            label = "Stratocumulus"
        if cond_opencu:
            label = "Open-Cell"

    # add a single label to the entire df
    df = df.copy()
    df["cloud_regime"] = label
    return df


def sonde_composite_dataset(all_sondes, campaign):
    """
    Turn the per-sonde table into the composite dataset, as the cookbook notebook
    writes it: CSET gets its dtypes fixed, columns renamed to the SOCRATES names
    and zlib compression; SOCRATES is written as it is.

    Parameters
    ----------
    all_sondes : pandas.DataFrame
        Every sonde, collocated with ERA5 and labelled with its cloud regime.
    campaign : str
        "SOCRATES" or "CSET".

    Returns
    -------
    tuple of (xarray.Dataset, dict or None)
        The dataset and the netCDF encoding to write it with.
    """
    if campaign == 'CSET':
        # --- copy to avoid side effects ---
        d = all_sondes.copy()

        # --- fix dtypes so netCDF can serialize ---
        d["Time"] = pd.to_datetime(d["Time"], utc=True, errors="coerce").dt.tz_convert(None)

        # RF / drop_num: real numpy ints (netcdf-safe)
        for c in ["RF", "drop_num"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype(np.int16)

        # strings: MUST be plain python str/object, NOT pandas "string"
        for c in ["cloud_regime", "drop_id", "trajectory"]:
            if c in d.columns:
                d[c] = d[c].fillna("").astype(str)

        # OPTIONAL: drop header junk if it exists
        for c in ["--", "UTC"]:
            if c in d.columns:
                d = d.drop(columns=[c])

        # OPTIONAL: rename to match SOCRATES naming
        rename_map = {
            "Press": "pres",
            "Temp": "tdry",
            "Dewpt": "dp",
            "RH": "rh",
            "Uwind": "u_wind",
            "Vwind": "v_wind",
            "Wspd": "wspd",
            "Dir": "wdir",
            "GPSAlt": "gpsalt",
            "GeoPoAlt": "alt",
        }
        d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})

        # --- make a simple RangeIndex called 'index' like SOCRATES ---
        d = d.reset_index(drop=True)
        d.index.name = "index"

        # --- convert to xarray in the SAME "flat table" layout ---
        ds = xr.Dataset.from_dataframe(d)
        # --- compression (skip strings + datetimes) ---
        encoding = {}
        for v in ds.data_vars:
            if ds[v].dtype.kind in ("U", "S", "O", "M"):  # M = datetime64
                continue
            encoding[v] = {"zlib": True, "complevel": 4}
        return ds, encoding

    if campaign == 'SOCRATES':
        return xr.Dataset.from_dataframe(all_sondes), None

    raise ValueError(f"No sonde composite layout for campaign '{campaign}'")


def build_sonde_composite(campaign, sonde_paths, out_file=None):
    """
    Build the dropsonde composite from raw sonde files: read every sonde,
    collocate it with ERA5 cloud controlling factors, label its cloud regime,
    and combine them into one dataset.

    Parameters
    ----------
    campaign : str
        "SOCRATES" or "CSET".
    sonde_paths : list
        Raw sonde files (.nc, .cls or .eol), e.g. from find_sondes().
    out_file : str or pathlib.Path, optional
        Where to write the composite.  It is written to a temporary file first
        and then moved into place, so an interrupted run never leaves a
        truncated composite behind for the next run to reuse.

    Returns
    -------
    xarray.Dataset
        The composite, as written.
    """
    dfs, _ = read_sonde(sonde_paths, campaign=campaign)
    sonde_dat = []
    for i in range(0, len(dfs)):
        print(f'  sonde {i + 1} of {len(dfs)}')
        ds = select_ERA5_4flight(dfs[i], campaign=campaign, dat_type='dropsonde')
        # Collocate ERA5 data and calculate environmental controlling factors
        sonde_coll = collocate_ERA5_sonde(ds, dfs[i])
        # Regime each sonde based on average column value
        regime_sonde = cloud_regime_sonde(sonde_coll, campaign=campaign)
        sonde_dat.append(regime_sonde)
    all_sondes = pd.concat(sonde_dat)

    ds, encoding = sonde_composite_dataset(all_sondes, campaign)
    if out_file is not None:
        out_file = Path(out_file)
        tmp_file = out_file.with_name(out_file.name + ".tmp")
        ds.to_netcdf(tmp_file, encoding=encoding)
        os.replace(tmp_file, out_file)
    return ds


#======================================================================================
# Per-flight cloud regime files (<CAMPAIGN>_RF<nn>_<yyyymmdd>.nc): the aircraft data
# blocked into flight maneuvers, collocated with ERA5 cloud controlling factors and
# labelled with cloud regimes.
#
# VAP_process_flight_data, assign_flight_type, block_flight, collocate_ERA5_dat,
# cloud_regime and write_RF_nc are ported from the cookbook's
# process_data_products_utils.py.  write_RF_nc differs only in using this module's
# `datetime` name (the class) and in taking an optional output directory; the
# original writes to the working directory, which is still the default.
# build_flight_cloud_regime_files follows the per-flight loop of the cookbook
# notebook INFORM_process_system_database.ipynb.
#======================================================================================

def VAP_process_flight_data(df,i):
    """
    High-Level Function for Processing Flight Data in Value Added Products.

    This function serves as the main entry point for processing flight data. It first calls 
    `assign_flight_type` to assign flight types (e.g., 'level' or 'profile') to different segments of the flight 
    based on altitude stability and time gaps. After flight types are assigned, it proceeds to categorize the data 
    into different flight blocks (e.g., level flight in boundary layer, in-cloud profile flight, etc.) by calling 
    the `block_flight` function.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing flight data with at least the following columns:
        - 'Time' (datetime): Time of each flight record.
        - 'GGALT' (float): Altitude of the aircraft.
        - 'PLWCD_' (float): Cloud Droplet Probe LWC.
        - 'CONCD_' (float): Cloud Droplet Probe Number Concentration.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'DataFrame': A modified DataFrame with assigned flight types, cloud status, and location.
        - 'flight_blocks': A dictionary of flight blocks categorized by flight type and cloud status.
        - 'cloud_blocks': A dataframe of flight blocks including blocks of aircraft data inside cloud layers.

    Notes:
    ------
    - The `assign_flight_type` function is responsible for determining whether the flight segments are 'level' or 'profile'.
    - The `block_flight` function segments the flight data based on the assigned flight types and cloud status into specific blocks (e.g., 'Level BL', 'In-Cloud Profiles', etc.).
    - The function ensures proper labeling of different flight segments for further analysis, including cloud status and location (e.g., boundary layer or free airspace).
    """
    # Function to assign flight type "Level" and "Profile" when in/out of cloud
    dict_flight_type = assign_flight_type(df)

    # Extract dataframe that has been modified from the assign_flight_type function
    df_mod = dict_flight_type['DataFrame']
    # Plot time series of aircraft defined flight blocks
    # plot_block_ts(dict_flight_type,i)

    # Run block flight function to return list of Dataframes of "blocked" flight data
    flight_blocks = block_flight(df_mod)
    # Function to assign cloud type from the HCR data
    # flight_block_comp = assign_cloud_type_HCR(flight_blocks,dir,i)
    
    # Plot time series of HCR defined cloud types  
    # plot_hcr_cloud_type(df_mod,flight_block_comp,i)
    return flight_blocks

def assign_flight_type(df):
    """
    Assigns flight type ('level' or 'profile') to each row of the input DataFrame based on stable altitude blocks 
    and gaps between these blocks. The function uses rolling standard deviation of altitude to identify level legs
    and combines consecutive blocks of stable altitude with a specified time gap threshold. Additionally, it labels 
    flight segments as "level" for level legs and "profile" for the aircraft vertical profile.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with at least the following columns:
        - 'Time' (timestamp)
        - 'GGALT' (altitude in meters)

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with a new column 'flight_type', where each row is assigned a flight type:
        - 'level' for stable altitude periods
        - 'profile' for gaps between stable altitude blocks

    Example:
    --------
    df = pd.read_csv('flight_data.csv')  # Assuming the CSV contains relevant columns
    df_with_flight_types = assign_flight_type(df)
    """

    #-----------------------------------------
    #----- Find profiles and level legs ------
    #-----------------------------------------
    
    # Define a time gap threshold to combine blocks (e.g., 120 seconds)
    time_gap_threshold = pd.Timedelta(seconds=120)
    
    # Compute rolling standard deviation of altitude to smooth noise
    df['rolling_std'] = df['GGALT'].rolling(window=10, center=True).std()
    
    # Identify where altitude remains stable within the threshold
    df['stable'] = df['rolling_std'] < 3  # You can adjust the threshold (meters)
    
    # Assign unique block IDs when stability changes
    df['block_id'] = (df['stable'] != df['stable'].shift()).cumsum()
    
    # Group by block_id and filter for long-duration stable blocks
    block_info = df[df['stable']].groupby('block_id').agg(
        start_time=('Time', 'first'),
        end_time=('Time', 'last'),
        lower_bound=('GGALT', 'min'),  # Minimum altitude (lower bound)
        upper_bound=('GGALT', 'max'),  # Maximum altitude (upper bound)
        duration=('Time', lambda x: x.max() - x.min())
    )
    
    # Filter out short-duration blocks
    valid_blocks = block_info[block_info['duration'] > pd.Timedelta(seconds=150)] ## EDIT?
    
    # Sort the blocks by start time
    valid_blocks = valid_blocks.sort_values(by='start_time')
    
    # Define a time gap threshold to combine blocks (e.g., 120 seconds)
    time_gap_threshold = pd.Timedelta(seconds=120)
    
    # Combine consecutive blocks that are less than the threshold apart
    combined_blocks = []
    previous_block = valid_blocks.iloc[0]
    
    for idx, current_block in valid_blocks.iloc[1:].iterrows():
        # Check if the gap between the end time of the previous block and start time of the current block is below the threshold
        if current_block['start_time'] - previous_block['end_time'] <= time_gap_threshold:
            # Extend the previous block's end time to the current block's end time
            previous_block['end_time'] = current_block['end_time']
        else:
            # If the gap is too large, append the previous block and update to the current block
            combined_blocks.append(previous_block)
            previous_block = current_block
    
    # Add the last block after the loop
    combined_blocks.append(previous_block)
    
    # Convert combined blocks back to DataFrame
    combined_blocks_df = pd.DataFrame(combined_blocks)
    
    # --- Identify and Label "Profiles" between "Level" (Stable) sections ---
    
    # Create a new column 'flight_type' to categorize the blocks as "level" or "profile"
    combined_blocks_df['flight_type'] = 'level'  # By default, label as 'level'
    
    # Now identify the gaps between "level" blocks and label as "profile"
    profile_blocks = []
    for i in range(len(combined_blocks_df) - 1):
        end_time_current = combined_blocks_df.iloc[i]['end_time']
        start_time_next = combined_blocks_df.iloc[i + 1]['start_time']
        
        # If there's a gap between two 'level' blocks, label the gap as 'profile'
        if start_time_next - end_time_current > time_gap_threshold:
            # Assign 'profile' to the gap between two level blocks and calculate duration
            profile_duration = start_time_next - end_time_current  # Duration of the profile block
            
            profile_blocks.append({
                'start_time': end_time_current,
                'end_time': start_time_next,
                'flight_type': 'profile',
                'duration': profile_duration
            })
    
    # Convert 'profile_blocks' to DataFrame
    profile_blocks_df = pd.DataFrame(profile_blocks)
    
    # Append profile blocks to the original combined blocks DataFrame
    combined_blocks_with_profiles = pd.concat([combined_blocks_df, profile_blocks_df], ignore_index=True)
    
    # Sort again by time
    combined_blocks_with_profiles = combined_blocks_with_profiles.sort_values(by='start_time')
    
    # Check for "Profile" after the Last Level Block
    last_end_time = combined_blocks_with_profiles.iloc[-1]['end_time']
    last_time_in_data = df['Time'].max()
    
    if last_time_in_data - last_end_time > time_gap_threshold:
        # If the gap is greater than the threshold, consider it a "profile" block
        profile_block = pd.DataFrame([{
            'start_time': last_end_time,
            'end_time': last_time_in_data,
            'flight_type': 'profile',
        }])
    
        # Concatenate the new profile block to the existing DataFrame
        combined_blocks_with_profiles = pd.concat([combined_blocks_with_profiles, profile_block], ignore_index=True)
    
    # Check for "Profile" before the First Level Block, used for takeoff/landing
    first_start_time = combined_blocks_with_profiles.iloc[0]['start_time']
    first_time_in_data = df['Time'].min()
    
    if first_start_time - first_time_in_data > time_gap_threshold:
        profile_block_before_first = pd.DataFrame([{
            'start_time': first_time_in_data,
            'end_time': first_start_time,
            'flight_type': 'profile',
        }])
    
        combined_blocks_with_profiles = pd.concat([profile_block_before_first, combined_blocks_with_profiles], ignore_index=True)
    
    # List of columns to remove
    columns_to_remove = ['rolling_std','stable','block_id']
    # Drop the specified columns from df2
    df = df.drop(columns=columns_to_remove)
    
    # Add new column "flight_type" as either "level" or "profile"
    for _, row in combined_blocks_with_profiles.iterrows():
        flight_type = row['flight_type']
        # Find rows in df2 where the time is between start_time and end_time
        mask = (df['Time'] >= row['start_time']) & (df['Time'] <= row['end_time'])
        df.loc[mask, 'flight_type'] = flight_type
    # Assign the flight_type to the first few rows that fall before the first start_time in df1
    df.loc[df['Time'] < first_start_time, 'flight_type'] = df.iloc[0]['flight_type']

    #------------------------------
    #----- Find cloud layers ------
    #------------------------------
    # Ensure 'Time' is in datetime format
    df = df.copy()  # Avoid modifying original DataFrame
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Find best match for column names dynamically
    plwc_col = next((col for col in df.columns if 'PLWCD' in col), None) or \
           next((col for col in df.columns if 'PLWC' in col), None)
    concd_col = next((col for col in df.columns if 'CONCD' in col), None)
    # Add check if there are any cloudy periods
    if not plwc_col or not concd_col:
        print("Required columns not found. Skipping cloud detection.")
        final_cloud_blocks = pd.DataFrame(columns=['start_time', 'end_time', 'lower_bound', 'upper_bound', 'duration', 'Location'])
        df['cloud_status'] = 'Out-of-cloud'
        df['Location'] = 'Free'
    else:
        df['blocked'] = (df[plwc_col] > 0.001) & (df[concd_col] > 10)
        if not df['blocked'].any():
            print("No valid cloud blocks found. Skipping cloud layer logic.")
            final_cloud_blocks = pd.DataFrame(columns=['start_time', 'end_time', 'lower_bound', 'upper_bound', 'duration', 'Location'])
            df['cloud_status'] = 'Out-of-cloud'
            df['Location'] = 'Free'
        else:
            df['block_id'] = (df['blocked'] != df['blocked'].shift()).cumsum()
            block_info = df[df['blocked']].groupby('block_id').agg(
                start_time=('Time', 'first'),
                end_time=('Time', 'last'),
                lower_bound=('GGALT', 'min'),
                upper_bound=('GGALT', 'max'),
            )

            # Calculate duration directly by subtracting start_time from end_time
            block_info['duration'] = block_info['end_time'] - block_info['start_time']
            
            # Filter out short-duration blocks
            min_vertical = 30  # Adjust as needed (100 meters in your case)
            valid_blocks = block_info[(block_info['upper_bound'] - block_info['lower_bound']) > min_vertical].reset_index(drop=True)
            
            # Define the altitude difference and time gap thresholds
            altitude_gap_threshold = 200  # Increased altitude gap threshold
            # time_gap_threshold = pd.Timedelta(minutes=20)  # Time gap threshold for merging
            
            # Sort the valid blocks by their start time to process them in sequence
            valid_blocks = valid_blocks.sort_values(by='lower_bound')
            
            # Initialize a list to store combined blocks
            combined_blocks = []
            previous_block = valid_blocks.iloc[0].to_dict()
            
            # Iterate through the blocks and merge those that are within the thresholds
            for idx, current_block in valid_blocks.iloc[1:].iterrows():
                # Calculate the altitude gap between the current block's lower bound and the previous block's upper bound
                altitude_gap = abs(current_block['lower_bound'] - previous_block['upper_bound'])
                
                # Calculate the time gap between the current block's start time and the previous block's end time
                time_gap = current_block['start_time'] - previous_block['end_time']
                # Check if the altitude gap is within the threshold or if the time gap is within the allowed range for smaller altitudes
                if altitude_gap <= altitude_gap_threshold :
                    # If both criteria are met, merge the blocks
                    previous_block['end_time'] = max(previous_block['end_time'], current_block['end_time'])  # Get the latest end time
                    previous_block['start_time'] = min(previous_block['start_time'], current_block['start_time'])  # Get the earliest start time
                    previous_block['upper_bound'] = max(previous_block['upper_bound'], current_block['upper_bound'])  # Update upper bound
                    previous_block['lower_bound'] = min(previous_block['lower_bound'], current_block['lower_bound'])  # Update lower bound
                    
                    # Recalculate the duration for the merged block
                    previous_block['duration'] = previous_block['end_time'] - previous_block['start_time']
                else:
                    # If the blocks are far apart, save the previous block and move to the next one
                    combined_blocks.append(previous_block)
                    previous_block = current_block.to_dict()
            
            # Add the last block after the loop
            combined_blocks.append(previous_block)
            
            # Convert the merged blocks back into a DataFrame
            combined_blocks_df = pd.DataFrame(combined_blocks)
            
            # Second check for merging adjacent blocks in combined_blocks_df
            final_combined_blocks = []
            previous_block = combined_blocks_df.iloc[0].to_dict()
            
            # Apply additional check for merging based on both time and altitude gap
            for idx, current_block in combined_blocks_df.iloc[1:].iterrows():
                # Calculate the altitude gap and time gap
                altitude_gap = abs(current_block['lower_bound'] - previous_block['upper_bound'])
                time_gap = current_block['start_time'] - previous_block['end_time']
                
                # Check for overlap in the altitude ranges
                overlap_check = (current_block['lower_bound'] >= previous_block['lower_bound']) and (current_block['lower_bound'] <= previous_block['upper_bound'])
            
                # Check if both the altitude gap, time gap, or overlap condition is met
                if altitude_gap <= altitude_gap_threshold or overlap_check:
                    # Merge the blocks
                    previous_block['end_time'] = max(previous_block['end_time'], current_block['end_time'])
                    previous_block['start_time'] = min(previous_block['start_time'], current_block['start_time'])
                    previous_block['upper_bound'] = max(previous_block['upper_bound'], current_block['upper_bound'])
                    previous_block['lower_bound'] = min(previous_block['lower_bound'], current_block['lower_bound'])
                    
                    # Recalculate the duration for the merged block
                    previous_block['duration'] = previous_block['end_time'] - previous_block['start_time']
                else:
                    # Save the previous block and move to the next one
                    final_combined_blocks.append(previous_block)
                    previous_block = current_block.to_dict()
            
            # Add the last block after the loop
            final_combined_blocks.append(previous_block)
            
            # Convert the final combined blocks back into a DataFrame
            final_cloud_blocks = pd.DataFrame(final_combined_blocks)

            # Add 'cloud_status' based on whether altitude and time fall within any blocked region (in the cloud or out of cloud)
            df['cloud_status'] = 'Out-of-cloud'  # Default label
            # Loop through each block and label altitudes as "In-cloud" if they fall within the block's range
            for _, block in final_cloud_blocks.iterrows():
                # Create a mask that checks both altitude and time conditions
                mask = (
                    (df['GGALT'] >= block['lower_bound']) & (df['GGALT'] <= block['upper_bound']) &
                    (df['Time'] >= block['start_time']) & (df['Time'] <= block['end_time'])
                )
                
                # Apply the 'In-cloud' label where the mask is True
                df.loc[mask, 'cloud_status'] = 'In-cloud'
            
            # List of columns to remove
            columns_to_remove = ['blocked','block_id']
            # Drop the specified columns from df2
            df = df.drop(columns=columns_to_remove)
        
            df['Location'] = 'Free'
            
            # Find the minimum in-cloud altitude
            min_ic_alt = np.min(final_cloud_blocks['lower_bound'])-5
            mask = df.GGALT < min_ic_alt
            # Define 
            df.loc[mask, 'Location'] = 'BL'
        
            # Update the Location column based on the GGALT and cloud status
            df.loc[df['GGALT'] < min_ic_alt, 'Location'] = 'BL'
        
            # --- Add Location to final_cloud_blocks DataFrame ---
            # Add 'Location' based on the minimum in-cloud altitude
            final_cloud_blocks['Location'] = final_cloud_blocks['lower_bound'].apply(
                lambda x: 'BL' if x < min_ic_alt else 'Free'
            )
                # Add 'Location' based on the minimum in-cloud altitude
            combined_blocks_with_profiles['Location'] = combined_blocks_with_profiles['lower_bound'].apply(
                lambda x: 'BL' if x < min_ic_alt else 'Free'
            )

    # Sort the dataframe by Time for continuous time grouping
    df = df.sort_values(by='Time')
    # Remove rows where 'flight_type' is NaN
    df = df.dropna(subset=['flight_type'])
    # Create a new column 'block_id' to group continuous time periods based on flight_type, cloud_status, and Location
    df['block_id'] = (df['flight_type'] != df['flight_type'].shift()) | \
                      (df['cloud_status'] != df['cloud_status'].shift()) | \
                      (df['Location'] != df['Location'].shift())
    df['block_id'] = df['block_id'].cumsum()

    Final_ds = {'DataFrame': df,
                'flight_blocks': combined_blocks_with_profiles,
                'Cloud_blocks': final_cloud_blocks
               }
    
    return Final_ds

def block_flight(df):
    """
    Segments a flight dataset into different flight block categories based on cloud status, location, and flight type.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing flight data with at least the following columns:
        - 'block_id' (int): Identifies different flight segments.
        - 'Location' (str): Can be 'BL' (Boundary Layer) or 'Free' airspace.
        - 'flight_type' (str): Can be 'level' or 'profile'.
        - 'cloud_status' (str): Either 'In-cloud' or 'Out-of-cloud'.
        - 'GGALT' (float): Altitude dbata, used for filtering profile segments.
        - 'Time' (datetime): Used to filter level flight segments.

    Returns:
    --------
    Flight_blocks : dict
        A dictionary containing categorized flight data:
        - 'Level BL': List of DataFrames for level flight in the boundary layer.
        - 'In-Cloud Profiles': List of DataFrames for in-cloud profile flights with altitude variation > 30m.
        - 'In-Cloud Level FT': List of DataFrames for level flights in free airspace within clouds.
        - 'Out-of-cloud Level FT': List of DataFrames for level flights in free airspace, lasting at least 3 minutes.

    Notes:
    ------
    - The function removes the first and last 'Level BL' periods to exclude takeoff/landing effects.
    - Only level flights lasting more than 180 seconds are included in 'Out-of-cloud Level FT'.
    - Profile flights are only included if their altitude change is greater than 30 meters.
    """
    # ---------- Level BL periods (KEEP ALL) ----------
    out_of_cloud_bl = df[(df['Location'] == 'BL') & (df['flight_type'] == 'level')]
    bl_ids = sorted(out_of_cloud_bl['block_id'].unique())
    bl_blocks_ds = [df[df['block_id'] == i] for i in bl_ids]
    
    # Find In-Cloud profile periods
    in_cloud_prof = df[(df['cloud_status'] == 'In-cloud') & (df['flight_type'] == 'profile')]
    ic_prof_ids = sorted(in_cloud_prof['block_id'].unique())
    ic_pro_blocks_ds = [df[df['block_id'] == i] for i in ic_prof_ids if df[df['block_id'] == i]['GGALT'].max() - df[df['block_id'] == i]['GGALT'].min() > 30]
    
    # Find level FT periods out-of-cloud
    level_ft = df[(df['cloud_status'] == 'Out-of-cloud') & (df['flight_type'] == 'level') & (df['Location'] == 'Free')]
    out_of_cloud_ft_ids = sorted(level_ft['block_id'].unique())
    level_ft_out_blocks_ds = [df[df['block_id'] == i] for i in out_of_cloud_ft_ids if df[df['block_id'] == i]['Time'].iloc[-1] - df[df['block_id'] == i]['Time'].iloc[0] > pd.Timedelta(seconds=180)]

    # Find level FT periods in-cloud
    level_ft_ic = df[(df['cloud_status'] == 'In-cloud') & (df['flight_type'] == 'level') & (df['Location'] == 'Free')]
    in_cloud_ft_ids = sorted(level_ft_ic['block_id'].unique())
    level_ft_ic_blocks_ds = [df[df['block_id'] == i] for i in in_cloud_ft_ids]
    
    # Save blocks of flight as dictionary for output
    Flight_blocks = {
        'Level BL': bl_blocks_ds,
        'In-Cloud Profiles': ic_pro_blocks_ds,
        'In-Cloud Level FT': level_ft_ic_blocks_ds,
        'Out-of-cloud Level FT': level_ft_out_blocks_ds
    }

    return Flight_blocks

def collocate_ERA5_dat(ds, blocks):
    """
    Collocate ERA5 fields onto flight blocks.
    Assumes ds variables have dims ('time','latitude','longitude') and longitude is 0..360 ascending.
    """

    # Ensure Dataset and time sorted
    ds = xr.Dataset(ds).sortby('time')

    # Coords (ERA5 already 0..360)
    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values
    ny, nx   = lat_vals.size, lon_vals.size

    # KDTree over regular lat-lon grid (0..360 frame)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    tree = cKDTree(np.column_stack((lat_grid.ravel(), lon_grid.ravel())))

    # ERA5 times (ns, sorted)
    t_era_ns = ds['time'].values.astype('datetime64[ns]').view('int64')

    # Pull arrays once (T,Y,X) as NumPy
    def arr3(name):
        return ds[name].transpose('time','latitude','longitude').compute().values

    arr = {
        'ERA5_SST':   arr3('SST'),
        'M':          arr3('M'),
        'omega700':       arr3('omega700'),
        'deltaT':     arr3('deltaT'),
        'Wind_sp':    arr3('WS'),
        'Wind_shear': arr3('Wind_shear'),
        'Tadv':       arr3('Tadv'),
        'RH700':      arr3('RH700'),
        'EIS':        arr3('EIS'),
    }

    # Loop blocks
    for key, blist in blocks.items():
        for i, block in enumerate(blist):
            b = block.copy().dropna(subset=['GGLAT','GGLON'])
            if len(b) == 0:
                blist[i] = b
                continue

            # Flight coords/time (convert lon to 0..360)
            flt_lat  = b['GGLAT'].to_numpy(dtype=float)
            flt_lon  = ((b['GGLON'].to_numpy(dtype=float) % 360.0) + 360.0) % 360.0
            flt_t_ns = b['Time'].to_numpy('datetime64[ns]').view('int64')

            # Nearest time index per sample
            ti = nearest_time_indices(t_era_ns, flt_t_ns)  # (N,)

            # Nearest gridpoint (lat, lon) in 0..360 frame
            _, flat_idx = tree.query(np.column_stack((flt_lat, flt_lon)))  # (N,)
            yi, xi = np.unravel_index(flat_idx, (ny, nx))                  # (N,), (N,)

            # Gather all variables
            for out_name, A in arr.items():
                b[out_name] = A[ti, yi, xi]

            blist[i] = b
        blocks[key] = blist

    return blocks

def cloud_regime(fblks, campaign):
    """
    Assign cloud_regime per block.

    - SOCRATES:
        Stratocumulus (cond_strat) if ANY of:
            1) M < -9  and Wind_sp < 9
            2) M < -10 and EIS > 7
            3) M < -10 and Wind_shear < 9

        Open-Cell (cond_open) if ANY of:
            1) M >= -7 and Wind_sp >= 9
            2) M >= -8 and EIS < 9
            3) M >= -8 and Wind_shear > 6

    - CSET:
        Stratocumulus (cond_strat) if ANY of:
            1) M < -10 and SST < 295
            2) M < -11 and Tadv < 0

        Open-Cell (cond_open) if ANY of:
            1) M >= -10 and SST >= 296
            2) M >= -10 and Tadv >= -4

    Tie policy:
        - Start everything as 'Undetermined'
        - Assign 'Stratocumulus' only where cond_strat is True
          AND cond_open is False
        - Assign 'Open-Cell' only where cond_open is True
          AND cond_strat is False
        → Points that satisfy both or neither stay 'Undetermined'.
    """

    # Normalize campaign string a bit for safety
    campaign = str(campaign).upper()

    for val in fblks:
        block_list = fblks[val]

        for i in range(len(block_list)):
            block = block_list[i].copy()

            # Default: unknown / in-between regime
            block['cloud_regime'] = pd.Series('Undetermined',
                                              index=block.index,
                                              dtype='object')

            # =====================================================
            # SOCRATES rules
            # =====================================================
            if campaign == 'SOCRATES':
                # Required / optional fields
                M   = block['M']
                WS  = block.get('Wind_sp',
                                pd.Series(np.nan, index=block.index))
                EIS = block.get('EIS',
                                pd.Series(np.nan, index=block.index))
                WSH = block.get('Wind_shear',
                                pd.Series(np.nan, index=block.index))

                # Stratocumulus-favoring conditions
                cond_strat = (
                    ((M < -9)  & (WS  < 9)) |
                    ((M < -10) & (EIS > 7)) |
                    ((M < -10) & (WSH < 9))
                )

                # Open-cell-favoring conditions
                cond_open = (
                    ((M >= -7) & (WS  >= 9)) |
                    ((M >= -8) & (EIS < 9))  |
                    ((M >= -8) & (WSH > 6))
                )

                # Apply labels, with "Undetermined" winning ties
                block.loc[cond_strat & ~cond_open, 'cloud_regime'] = 'Stratocumulus'
                block.loc[cond_open  & ~cond_strat, 'cloud_regime'] = 'Open-Cell'

            # =====================================================
            # CSET rules
            # =====================================================
            elif campaign == 'CSET':
                M    = block['M']
                SST  = block.get('ERA5_SST',
                                 block.get('sst',
                                           pd.Series(np.nan, index=block.index)))
                Tadv = block.get('Tadv',
                                 pd.Series(np.nan, index=block.index))

                # Stratocumulus-favoring conditions
                cond_strat = (
                    ((M < -10) & (SST < 296)) |
                    ((M < -11) & (Tadv < 0))
                )

                # Open-cell-favoring conditions
                cond_open = (
                    ((M >= -10) & (SST >= 296)) |
                    ((M >= -10) & (Tadv >= -3))
                )

                # Apply labels, with "Undetermined" winning ties
                block.loc[cond_strat & ~cond_open, 'cloud_regime'] = 'Stratocumulus'
                block.loc[cond_open  & ~cond_strat, 'cloud_regime'] = 'Open-Cell'

            # Write back modified block
            block_list[i] = block

        # Update this entry in fblks
        fblks[val] = block_list

    return fblks

def write_RF_nc(fblks_cr, rf, campaign, out_dir=None):
    combined = []
    if isinstance(fblks_cr, dict):
        for label, df_list in fblks_cr.items():
            for i, df in enumerate(df_list):
                df = df.copy()
                df["flight"] = rf
                df["block_label"] = label
                df["block_index"] = i
                combined.append(df)

        df_all = pd.concat(combined, ignore_index=True)
        df_all = df_all.set_index(["block_label", "block_index", "Time"])
        ds = df_all.reset_index().to_xarray()
        today = datetime.today().strftime("%Y%m%d")        
        # campaign prefix, no spaces, underscore separator
        campaign_str = str(campaign).upper().replace(" ", "")
        rf_str = str(rf).replace(" ", "_")
        name = f"{campaign_str}_{rf_str}_{today}.nc"

        if out_dir is not None:
            name = str(Path(out_dir) / name)
        ds.to_netcdf(name)
        print(f"Wrote {name}")
        return name


def build_flight_cloud_regime_files(campaign, icf_obj, out_dir, flights=None):
    """
    Build the per-flight cloud regime files from the raw aircraft data: load
    each research flight (with 25 Hz sigma_w where configured), block it
    into flight maneuvers, collocate the blocks with ERA5 cloud controlling
    factors, label their cloud regimes and write one netCDF file per flight.

    Parameters
    ----------
    campaign : str
        "SOCRATES" or "CSET".
    icf_obj : AdfInfo-like
        Object whose `campaigns_dict` gives "air_1hz_dir" and, optionally,
        "air_25hz_dir" for the campaign (see _campaign_cfg_for).  "ccn_dir" is
        not used: like the cookbook, the files carry no CCN.
    out_dir : str or pathlib.Path
        Directory the files are written to.  Each name carries today's date,
        so do not write into a directory that already holds files for the same
        flights: anything globbing *RF*.nc there would read both.
    flights : list of int, optional
        Research flight numbers to build (1 = RF01).  Default: every flight.

    Returns
    -------
    list of str
        The files written.
    """
    flight_paths = find_flight_fnames(campaign, icf_obj=icf_obj)
    if flights is None:
        flights = range(1, len(flight_paths) + 1)
    written = []
    for rf_num in flights:
        i = rf_num - 1
        # The cookbook loop passes ccn_df=None, so CCN is not merged
        df = load_flight_data(campaign, i, ccn_df=None, icf_obj=icf_obj)

        # RF12 is the only flight with this variable and it messes up the final product
        if ('PLWC' in df.columns) and (campaign == 'SOCRATES'):
            df = df.drop(columns=['PLWC'])
        blocks = VAP_process_flight_data(df, i)

        # Select ERA5 data
        ds = select_ERA5_4flight(df, campaign)
        rf_id = f"RF{i+1:02d}"
        print(rf_id)

        # Collocate ERA5 data and calculate environmental controlling factors
        fblks_coll = collocate_ERA5_dat(ds, blocks)
        # Select cloud regime type based on cloud controlling factors
        fblks_cr = cloud_regime(fblks_coll, campaign=campaign)
        # Write to NetCDF for this flight
        written.append(write_RF_nc(fblks_cr, rf_id, campaign, out_dir=out_dir))
    return written


def load_nc_cldrgme(file_paths):

    combined_blocks = []   
    for path in file_paths:
        rf = path.split("/")[-1].split(".")[0]  # e.g., "RF01"
        ds = xr.open_dataset(path)
        
        # Get unique combinations
        labels = ds["block_label"].values
        indices = ds["block_index"].values
        
        # Convert to DataFrame for convenient filtering
        df = ds.to_dataframe().reset_index().drop(columns="index")  # remove redundant index
        
        # Get all unique (label, index) pairs
        unique_blocks = df[["block_label", "block_index"]].drop_duplicates()
        
        # Loop through each unique block
        for _, row in unique_blocks.iterrows():
            label = row["block_label"]
            idx = row["block_index"]
        
            # Filter DataFrame
            df_block = df[(df["block_label"] == label) & (df["block_index"] == idx)].copy()
        
            # (Optional) add flight ID if you have it
            df_block["block_label"] = label
            df_block["block_index"] = idx
        
            combined_blocks.append(df_block)
        all_blocks = pd.concat(combined_blocks, ignore_index=True)
    return all_blocks

#======================================================================================
# Functions related to reading in CCN data and merging with 1Hz aircraft dataframe
#======================================================================================

def _parse_first_float(s: str):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

def _read_header(path):
    """
    Reads ICARTT-style header.
    Returns:
      n_header (int)
      header_lines (list[str])
      varnames (list[str])  # from last header line
    """
    with open(path, "r") as f:
        first = f.readline().strip()
        # e.g. "35, 1001" -> n_header is first integer
        n_header = int(re.split(r"[,\s]+", first)[0])

        header_lines = [first]
        for _ in range(n_header - 1):
            header_lines.append(f.readline().rstrip("\n"))

    var_line = header_lines[-1]
    varnames = [v.strip() for v in var_line.split(",") if v.strip()]
    return n_header, header_lines, varnames

def _extract_date_from_header(header_lines):
    """
    From your example there is a line like:
      2018, 01, 28, 2018, 04, 06
    We'll take the first 3 as the data date (YYYY, MM, DD).
    """
    for line in header_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            # try parse first three as ints: year, month, day
            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                # sanity check year range
                if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
                    return pd.Timestamp(year=y, month=m, day=d)
            except Exception:
                pass
    raise ValueError("Could not find a YYYY,MM,DD line in header.")

def _extract_const_ss_percent(header_lines):
    """
    Parses DATA_INFO: ... 0.43 percent supersaturation.
    Returns float (percent), e.g. 0.43
    """
    for line in header_lines:
        if line.startswith("DATA_INFO"):
            # look for "<number> percent supersaturation"
            m = re.search(r"([\d.]+)\s*percent\s+supersaturation", line, flags=re.IGNORECASE)
            if m:
                return float(m.group(1))
            # fallback: any float on DATA_INFO line
            x = _parse_first_float(line)
            if x is not None:
                return float(x)
            raise ValueError(f"DATA_INFO present but can't parse SS: {line}")
    raise ValueError("No DATA_INFO line found for constant-SS file.")

def _extract_rf_from_path(path: str) -> str:
    m = re.search(r"(RF\d{2})", path)
    if not m:
        raise ValueError(f"Could not parse RF from filename: {path}")
    return m.group(1)

def _find_col(varnames, patterns):
    """
    Finds first column whose name matches any regex in patterns.
    """
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for i, v in enumerate(varnames):
            if rx.search(v):
                return i
    return None

def load_ccn_file(path):
    """
    Returns a DataFrame with columns:
      dt_utc (datetime64[ns])
      Start_UTC (float)
      CCN (float)
      CCN_error (float)
      SS_percent (float)
      source_file (str)
    """
    n_header, header_lines, varnames = _read_header(path)
    date0 = _extract_date_from_header(header_lines)
    rf = _extract_rf_from_path(path)
    
    # Load numeric data block
    data = np.genfromtxt(path, delimiter=",", skip_header=n_header, invalid_raise=False)
    if data.ndim == 1:
        data = data[None, :]

    # Column indices
    i_t   = _find_col(varnames, [r"^Start_UTC$"])
    i_ccn = _find_col(varnames, [r"^CCN$"])
    i_err = _find_col(varnames, [r"^CCN_error$", r"^CCN.*error$"])

    if i_t is None or i_ccn is None or i_err is None:
        raise ValueError(
            f"Missing required columns in {path}\n"
            f"Found columns: {varnames}\n"
            f"Need Start_UTC, CCN, CCN_error."
        )

    start_utc = data[:, i_t].astype(float)
    ccn = data[:, i_ccn].astype(float)
    ccn_err = data[:, i_err].astype(float)

    # Supersaturation: constant or scanning column
    if "CCNconstSS_" in path:
        ss_percent = _extract_const_ss_percent(header_lines)
        ss = np.full_like(start_utc, ss_percent, dtype=float)
    else:
        # scanning file: find SS column
        i_ss = _find_col(varnames, [r"supersat", r"supersaturation", r"(^|[^a-z])ss([^a-z]|$)"])
        if i_ss is None:
            raise ValueError(
                f"Scanning file but no SS column found in {path}\n"
                f"Columns: {varnames}"
            )
        ss = data[:, i_ss].astype(float)

    # Build real datetime = date + seconds
    dt = date0 + pd.to_timedelta(start_utc, unit="s")

    df = pd.DataFrame({
        "dt_utc": dt,
        "RF_num": rf,
        "CCN": ccn,
        "CCN_error": ccn_err,
        "SS_percent": ss,
        "source_file": path,
    })

    # Optional: mask ICARTT missing flags
    # Your header shows -9999 as missing. Also ULOD/LLOD flags exist.
    # We'll at least drop -9999 for these columns.
    for col in ["CCN", "CCN_error", "SS_percent"]:
        df.loc[df[col] <= -9000, col] = np.nan

    return df

def load_all_ccn(files):
    dfs = [load_ccn_file(p) for p in files]
    out = pd.concat(dfs, ignore_index=True)

    # Sort by real datetime (NOT Start_UTC alone)
    out = out.sort_values("dt_utc").reset_index(drop=True)
    return out

def split_ccn_streams(ccn_df):
    ccn_df = ccn_df.copy()

    const = (
        ccn_df[ccn_df["source_file"].str.contains("CCNconstSS_", na=False)]
        [["dt_utc", "CCN", "CCN_error", "SS_percent"]]
        .rename(columns={
            "CCN": "CCN_const",
            "CCN_error": "CCN_error_const",
            "SS_percent": "SS_percent_const",
        })
        .sort_values("dt_utc")
    )

    scan = (
        ccn_df[ccn_df["source_file"].str.contains("CCNscanning_", na=False)]
        [["dt_utc", "CCN", "CCN_error", "SS_percent"]]
        .rename(columns={
            "CCN": "CCN_scan",
            "CCN_error": "CCN_error_scan",
            "SS_percent": "SS_percent_scan",
        })
        .sort_values("dt_utc")
    )

    return const, scan

def merge_two_ccn_streams_into_aircraft(df_aircraft, ccn_df, tolerance="5s"):
    df = df_aircraft.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    const, scan = split_ccn_streams(ccn_df)
    const["dt_utc"] = pd.to_datetime(const["dt_utc"])
    scan["dt_utc"]  = pd.to_datetime(scan["dt_utc"])

    # Merge constSS stream
    df = pd.merge_asof(
        df,
        const,
        left_on="Time",
        right_on="dt_utc",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    ).drop(columns=["dt_utc"])

    # Merge scanning stream
    df = pd.merge_asof(
        df,
        scan,
        left_on="Time",
        right_on="dt_utc",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    ).drop(columns=["dt_utc"])

    return df


def load_ccn_for_campaign(campaign: str, icf_obj=None) -> pd.DataFrame | None:
    if icf_obj is None:
        raise ValueError("load_ccn_for_campaign requires icf_obj (an AdfInfo-like object with .campaigns_dict)")
    cfg = _campaign_cfg_for(icf_obj.campaigns_dict, campaign)
    ccn_dir = cfg.get("ccn_dir")
    if not ccn_dir:
        return None

    exclude_substring = "spectra"
    fnames = sorted(
        f for f in os.listdir(ccn_dir)
        if f.endswith(".ict") and exclude_substring not in f.lower()
    )
    paths = [os.path.join(ccn_dir, f) for f in fnames]
    return load_all_ccn(paths)
