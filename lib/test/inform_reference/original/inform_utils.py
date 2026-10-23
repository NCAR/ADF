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

CAMPAIGN_CFG = {
    "SOCRATES": {
        "air_1hz_dir": "/glade/work/patnaude/SOCRATES/aircraft_1hz",
        "air_25hz_dir": "/glade/work/patnaude/SOCRATES/aircraft_25hz",
        "ccn_dir":      "/glade/work/patnaude/SOCRATES/CCN",
    },
    "CSET": {
        "air_1hz_dir": "/glade/work/patnaude/CSET/aircraft_1hz",
        "air_25hz_dir": "/glade/work/patnaude/CSET/aircraft_25hz",
        "ccn_dir": None,
    },
}

def find_flight_fnames(campaign: str, freq: str = "air_1hz_dir") -> list[str]:
    """
    Return list of aircraft flight files for a campaign.

    Parameters
    ----------
    campaign : str
        Campaign name ("SOCRATES", "CSET", etc.)

    freq : str
        Data frequency key in CAMPAIGN_CFG
        ("air_1hz", "air_25hz", "ccn")

    Returns
    -------
    list[str]
        List of flight NetCDF file paths
    """

    campaign = campaign.upper()
    print("campaign:", campaign)
    if campaign not in CAMPAIGN_CFG:
        raise ValueError(f"Unknown campaign '{campaign}'")

    dir_path = CAMPAIGN_CFG[campaign][freq]

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
) -> pd.DataFrame:
    cfg = CAMPAIGN_CFG[campaign]

    # --- 1 Hz aircraft ---
    flight_1hz_paths = find_flight_fnames(campaign)
    print("flight_1hz_paths",flight_1hz_paths)
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
            flight_25hz_paths = find_flight_fnames(campaign, freq="air_25hz_dir")
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

def load_ccn_for_campaign(campaign: str) -> pd.DataFrame | None:
    cfg = CAMPAIGN_CFG[campaign]
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

# def load_flight_data(dir_path: str, ccn_dir: str, idx: int = 0, add_sizedist: bool = True,
#                      asof: bool = False, tol: str = "1s") -> pd.DataFrame:
#     """
#     High-level loader: base 1 Hz vars + (optional) drizzle/precip from sizedists.
#     """    """
#     High-Level Function for finding and reading in flight data.

#     This function searches a directory for NetCDF (*.nc) flight data files, selects one based on the provided 
#     index, opens it, identifies relevant variables based on the dataset contents, and reads those variables 
#     into a Pandas DataFrame.

#     :param dir_path: Path to the directory containing NetCDF flight data files.
#     :param idx: Index of the file to load from the sorted list of *.nc files in the directory.

#     :return: A Pandas DataFrame containing the extracted flight data variables.
#     """
#     flight_dat_paths = find_flight_fnames(dir_path)
#     nc = open_nc(flight_dat_paths[idx])
#     vars2read = read_vars(nc)
#     df = read_flight_nc(nc,vars2read)
#     ## Function to load 

#     # 2) Load CCN data
#     exclude_substring="spectra"
#     flight_fnames = sorted(
#     fname for fname in os.listdir(ccn_dir)
#     if fnmatch.fnmatch(fname, "*.ict")
#     and (exclude_substring.lower() not in fname.lower())
# )
#     paths = [os.path.join(ccn_dir, fname) for fname in flight_fnames]
#     ccn_df = load_all_ccn(paths)
    
#     # merge CCN data with 1Hz aircraft data
#     df = merge_two_ccn_streams_into_aircraft(df, ccn_df, tolerance="5s")
    
#     # 3) derived from size distributions (returns time-indexed DF)
#     sd_vars = read_sizedist_vars(nc)          # your prefix-based picker
#     conc_df = calc_concs_from_sd(sd_vars, nc) # columns like Ndriz_2DC, Nprecip_2DC, Nprecip_2DS, ...

#     if conc_df is None or conc_df.empty:
#         return df

#     # 4) join on time
#     df2 = df.copy()
#     df2["Time"] = pd.to_datetime(df2["Time"]).dt.tz_localize(None).round("S")

#     conc = conc_df.copy()
#     conc.index = pd.to_datetime(conc.index).tz_localize(None).round("S")

#     if asof:
#         # nearest match within tolerance (useful if clocks are off by <1s)
#         out = pd.merge_asof(
#             df2.sort_values("Time"),
#             conc.reset_index().rename(columns={"index": "Time"}).sort_values("Time"),
#             on="Time",
#             direction="nearest",
#             tolerance=pd.Timedelta(tol),
#         )
#     else:
#         # exact join on second-resolution timestamps
#         out = df2.set_index("Time").join(conc, how="left").reset_index()

#     return out

# def read_sonde2df(file_path):
#     """
#     Reads a `.cls` radiosonde file and extracts multiple datasets with their nominal release times.

#     :param file_path: Path to the `.cls` file containing radiosonde data.

#     :return: A tuple containing:
#         - A list of Pandas DataFrames, each representing an individual radiosonde dataset.
#         - A list of corresponding nominal release times as Pandas Timestamps.
    
#     :raises FileNotFoundError: If the file does not exist.
#     :raises ValueError: If the file is incorrectly formatted or missing essential data.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File '{file_path}' not found.")

#     # Read the entire file into memory
#     with open(file_path, "r", encoding="utf-8") as file:
#         lines = file.readlines()

#     # Identify all "Nominal Release Time" occurrences
#     start_indices = [i for i, line in enumerate(lines) if "Nominal Release Time" in line]
    
#     if not start_indices:
#         raise ValueError(f"No 'Nominal Release Time' entries found in file: {file_path}")

#     datasets = []
#     drop_times = []
#     # Process each radiosonde dataset
#     for idx, start in enumerate(start_indices):
#         # Find the start of the tabular data
#         data_start = None
#         for i in range(start, len(lines) - 2):
#             if lines[i].strip().startswith("Time") and "Press" in lines[i]:  # Detect header row
#                 data_start = i + 3  # Data starts 2 lines after the column names
#                 break

#         if data_start is None:
#             print(f"Warning: No data start found for entry at line {start}")
#             continue  # Skip this dataset

#         # Extract and convert nominal release time
#         try:
#             date_time_str = lines[start].split("):")[1].strip()
#             drop_time = pd.to_datetime(date_time_str, format='%Y, %m, %d, %H:%M:%S')
#         except (IndexError, ValueError) as e:
#             print(f"Warning: Failed to parse drop time at line {start}: {e}")
#             continue  # Skip this dataset

#         drop_times.append(drop_time)

#         # Extract column names
#         columns = lines[data_start - 3].strip().split()

#         # Determine dataset end (next "Nominal Release Time" or end of file)
#         end = start_indices[idx + 1] if idx + 1 < len(start_indices) else len(lines)

#         # Extract and clean data
#         data_lines = [line.strip().split() for line in lines[data_start:end]]
#         data = [row for row in data_lines if len(row) == len(columns)]

#         # Convert to DataFrame
#         df = pd.DataFrame(data, columns=columns)

#         if df.empty:
#             print(f"Warning: Empty dataset at line {start}")
#             continue  # Skip empty datasets

#         # Remove rows containing 9999.0 in any column
#         df = df[(df != "9999.0").all(axis=1)]
#         # Convert numeric columns where possible
#         df = df.apply(pd.to_numeric, errors='coerce')
#         # Store drop time in DataFrame metadata
#         df.attrs["drop_time"] = drop_time
#         # Append dataset
#         datasets.append(df)

#     return datasets

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

def load_nc_cldrgme(file_paths):

    combined_blocks = []   
    for path in file_paths:
        rf = path.split("/")[-1].split(".")[0]  # e.g., "RF01"
        ds = xr.open_dataset(path)
        print(ds,"\n\n")
        
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
    
def load_ccn_for_campaign(campaign: str) -> pd.DataFrame | None:
    cfg = CAMPAIGN_CFG[campaign]
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

