from scipy.special import erf
import os, xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
import bisect
import glob
import sys
import datetime

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# Add parent directory to sys.path
parent_dir = Path('/glade/u/home/patnaude/inform/').resolve().parent.parent
sys.path.append(str(parent_dir))

def load_cam_files(cam_dir, file_sel, tmin, tmax, campaign):
    # =====================================================================
    # Your existing CAM load (unchanged)
    # =====================================================================
    pat = re.compile(r".*(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")
    
    def fname_to_dt(p: Path):
        m = pat.match(p.name)
        if not m:
            return None
        y, mo, d, sod = map(int, m.groups())
        hh = sod // 3600
        mm = (sod % 3600) // 60
        ss = sod % 60
        return pd.Timestamp(year=y, month=mo, day=d, hour=hh, minute=mm, second=ss)
    
    all_files = sorted([p for p in cam_dir.glob("*.nc") if f"{file_sel}" in p.name])
    
    times = []
    files = []
    for p in all_files:
        dt = fname_to_dt(p)
        if dt is not None:
            times.append(dt)
            files.append(p)
    
    pad = pd.Timedelta(minutes=60)
    lo = tmin - pad
    hi = tmax + pad
    
    i0 = bisect.bisect_left(times, lo)
    i1 = bisect.bisect_right(times, hi) - 1
    
    if i0 > i1 or not files:
        raise RuntimeError("No files found overlapping the requested time window.")
    
    i0_ext = max(i0 - 1, 0)
    i1_ext = min(i1 + 1, len(files) - 1)
    
    sel_files = files[i0_ext : i1_ext + 1]
    
    print(f"Selected {len(sel_files)} CAM files from {files[i0_ext].name} to {files[i1_ext].name}")

    ds_cam = xr.open_mfdataset(sel_files, combine="by_coords", parallel=True)
    print("loaded CAM ds")

    if "time" in ds_cam:
        if np.issubdtype(ds_cam.time.dtype, np.datetime64):
            ds_cam = ds_cam.sel(time=slice(tmin, tmax))
        else:
            cftype = type(ds_cam.time.values[0])
            tmin_cf = cftype(tmin.year, tmin.month, tmin.day, tmin.hour, tmin.minute, tmin.second)
            tmax_cf = cftype(tmax.year, tmax.month, tmax.day, tmax.hour, tmax.minute, tmax.second)
            ds_cam = ds_cam.sel(time=slice(tmin_cf, tmax_cf))
    else:
        print("Warning: CAM dataset has no 'time' coordinate; skipping time slice.")
    
    
    if campaign == 'SOCRATES':
        # rename variables to remove weird end
        suffix = "_130e_to_170e_35s_to_65s"
        ds_cam = ds_cam.rename({var: var.replace(suffix, "") for var in ds_cam.data_vars})
        ds_cam = ds_cam.rename({
            "lat_35s_to_65s": "lat",
            "lon_130e_to_170e": "lon"
        })
    elif campaign == 'CSET':
        suffix = "_200e_to_245e_15n_to_45n"
        ds_cam = ds_cam.rename({var: var.replace(suffix, "") for var in ds_cam.data_vars})
        ds_cam = ds_cam.rename({
            "lat_15n_to_45n": "lat",
            "lon_200e_to_245e": "lon"
        })
    
    return ds_cam

# =====================================================================
# ERA5: load all pressure levels for T, U, V and derive P, Td, RH
# =====================================================================
def load_era5_files(tmin, tmax, lat_slice, lon_slice):
    """
    Load ERA5 pressure-level data (T, U, V, Q) for a given time and spatial window,
    compute relative humidity and dew point, and return a merged xarray Dataset.

    Parameters
    ----------
    tmin : datetime.datetime
        Start time of the requested window
    tmax : datetime.datetime
        End time of the requested window
    lat_slice : slice
        Latitude slice (e.g., slice(-60, -30))
    lon_slice : slice
        Longitude slice (e.g., slice(0, 60))
    
    Returns
    -------
    xr.Dataset
        Dataset containing:
            - T  : Temperature (K)
            - U  : Zonal wind (m/s)
            - V  : Meridional wind (m/s)
            - RH : Relative humidity (%)
            - P  : Pressure (Pa)
            - Td : Dew point (K)
    """
    # Helper to find ERA5 files that overlap our time window
    def get_matching_files(pattern, start_dt, end_dt):
        file_list = glob.glob(pattern)
        selected = []
        for file in file_list:
            time_strs = file.split('.')[-2].split('_')
            file_start = datetime.datetime.strptime(time_strs[0], "%Y%m%d%H")
            file_end   = datetime.datetime.strptime(time_strs[1], "%Y%m%d%H")
            if file_start <= end_dt and file_end >= start_dt:
                selected.append(file)
        return selected
    
    # ERA5 pressure-level directory
    filepath_pl = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    
    # Assume flight is contained within a single year/month (true for individual RFs)
    dir_date = f"{tmin.year}{tmin.month:02d}"
    
    start_dt = tmin
    end_dt   = tmax
    
    def month_iter(start_dt, end_dt):
        """Yield (year, month) pairs from start_dt.month to end_dt.month inclusive."""
        y, m = start_dt.year, start_dt.month
        while (y < end_dt.year) or (y == end_dt.year and m <= end_dt.month):
            yield y, m
            if m == 12:
                y += 1
                m = 1
            else:
                m += 1
    
    def open_era5_pl_var(short_name, var_name, start_dt, end_dt):
        """
        short_name: one of 't','u','v','r','q' in filenames
        var_name:   'T','U','V','R','Q' inside the file
    
        Searches all monthly subdirs between start_dt and end_dt, then
        selects the exact time window and lat/lon slice.
        """
        all_files = []
        for y, m in month_iter(start_dt, end_dt):
            dir_date = f"{y}{m:02d}"
            pattern = f"{filepath_pl}{dir_date}/*_{short_name}.*.nc"
            monthly_files = get_matching_files(pattern, start_dt, end_dt)
            all_files.extend(monthly_files)
    
        all_files = sorted(set(all_files))
        if not all_files:
            raise RuntimeError(
                f"No ERA5 pl files found for {short_name} between {start_dt} and {end_dt}"
            )
    
        # =============== PRINT SUMMARY (like CAM printout) ==================
        print(
            f"Selected {len(all_files)} ERA5-PL files ({short_name}) from\n"
            f"  {Path(all_files[0]).name}\n"
            f"to\n"
            f"  {Path(all_files[-1]).name}"
        )
    
        # =============== load across time ==================
        ds_var = xr.open_mfdataset(all_files, combine="nested", concat_dim="time")[[var_name]]
        ds_var = ds_var.sortby("time").sel(time=slice(start_dt, end_dt))
        ds_var = ds_var.sortby("latitude")
        ds_var = ds_var.sel(latitude=lat_slice, longitude=lon_slice)
    
        return ds_var
    
    # Load T, U, V, RH on all levels
    era5_T = open_era5_pl_var("t", "T", start_dt, end_dt)
    era5_U = open_era5_pl_var("u", "U", start_dt, end_dt)
    era5_V = open_era5_pl_var("v", "V", start_dt, end_dt)
    era5_Q = open_era5_pl_var("q", "Q", start_dt, end_dt)
    
    # ---------------------------------------------------------------------
    # Build pressure field (3D) and compute dew point from T and RH
    # ---------------------------------------------------------------------
    
    # Broadcast pressure (Pa) from 'level' (hPa) to full field
    P = (era5_T["level"] * 100.0)  # Pa
    P = xr.DataArray(
        P.values,
        coords={"level": era5_T["level"]},
        dims=("level",),
        name="P"
    ).broadcast_like(era5_T["T"])
    
    def RH_from_spec_hum(T,q,P):    
        # Murphy & Koop (2005) saturation vapor pressure (Pa)
        def es_MK_water(T):
            # ln(esw [Pa]) valid ~123–332 K
            return xr.ufuncs.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
                                 + xr.ufuncs.tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
        
        def es_MK_ice(T):
            # ln(esi [Pa]) valid ~110–273 K
            return xr.ufuncs.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
        
        esw = es_MK_water(T)
        esi = es_MK_ice(T)
        # Choose water above freezing, ice at/below (adjust threshold if you prefer 273.16)
        es = xr.where(T > 273.15, esw, esi)
        
        # Saturation specific humidity and RH
        eps = 0.622
        qsat = (eps * es) / ((P) - (1.0 - eps) * es)
        
        # Avoid division issues extremely near saturation/low p
        qsat = qsat.clip(min=1e-12)
        
        RH = (q / qsat) * 100.0
        return RH
    
    RH = RH_from_spec_hum(era5_T["T"],era5_Q["Q"],P)
    
    # Dew point from T (K) and RH (%), using a Magnus-type formula
    def dewpoint_from_T_RH(T_K, RH_pct):
        T_C = T_K - 273.15
        a = 17.625
        b = 243.04  # deg C
        gamma = np.log(RH_pct / 100.0) + (a * T_C) / (b + T_C)
        Td_C = (b * gamma) / (a - gamma)
        return Td_C + 273.15  # back to Kelvin
    
    Td = dewpoint_from_T_RH(era5_T["T"], RH)
    Td = Td.rename("Td")
    
    # ---------------------------------------------------------------------
    # Final ERA5 pressure-level dataset aligned with CAM time window
    # ---------------------------------------------------------------------
    return xr.Dataset(data_vars=dict(
                T  = era5_T["T"],   # Temperature (K)
                U  = era5_U["U"],   # Zonal wind (m/s)
                V  = era5_V["V"],   # Meridional wind (m/s)
                RH = RH,  # Relative humidity (%)
                P  = P,             # Pressure (Pa)
                Td = Td,            # Dew point (K)
                )
            )

def load_cam_ccfs(ds, lat_slice, lon_slice):
    # Calculate pressure levels
    # Compute pressure altitude (palt) from CESM hybrid coordinates
    p0 = 100000 # (ds.P0 old CAM6, it's all the same)  # Reference pressure
    ps = ds.PS  # Surface pressure [=] Pa
    hyai = ds.hyam  # Hybrid A coefficient at layer interface
    hybi = ds.hybm  # Hybrid B coefficient at layer interface
    
    P_dummy = p0 * hyai
    midP = (P_dummy + (hybi * ps))  # [=] Pa

    # a(k)*p0 + b(k)*ps(n,j,i)
    p = midP * 0.01  # Convert to hPa
    p = p.sel(lat = lat_slice, lon =lon_slice)
    
    # p has dims: (lev, time, lat, lon) in hPa
    p800_idx = np.abs(p - 800.0).argmin(dim='lev').compute()  # (time, lat, lon) int64
    p700_idx = np.abs(p - 700.0).argmin(dim='lev').compute()
    
       
    # (optional) speed difference version
    # wind_shear = np.hypot(U700, V700) - np.hypot(usfc, vsfc)
    
    # # Horizontal sfc windspeed
    u = ds.U.isel(lev = 31).sel(lat = lat_slice, lon = lon_slice)
    v = ds.V.isel(lev = 31).sel(lat = lat_slice, lon = lon_slice)
    ws = np.sqrt(u**2 + v**2)
    
    # # Calcualte wind shear (SFC - 700mb)
    Usub = ds.U.sel(lat=lat_slice, lon=lon_slice)    # (lev, time, lat, lon)
    Vsub = ds.V.sel(lat=lat_slice, lon=lon_slice)
    # 3) vectorized isel along lev using the per-column indices
    U700 = Usub.isel(lev=p700_idx)                   # (time, lat, lon)
    V700 = Vsub.isel(lev=p700_idx)

    def interp_to_p(var, p_hPa, target_hPa=700.0):
        tgt = xr.DataArray(target_hPa)
        def _interp_1d(pcol, vcol, t):
            m = np.isfinite(pcol) & np.isfinite(vcol)
            if m.sum() < 2:
                return np.nan
            p1, v1 = pcol[m], vcol[m]
            o = np.argsort(p1)
            p1, v1 = p1[o], v1[o]
            return np.interp(t, p1, v1, left=np.nan, right=np.nan)
    
        return xr.apply_ufunc(
            _interp_1d, p_hPa, var, tgt,
            input_core_dims=[["lev"], ["lev"], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[var.dtype],
        )

    # --- after you compute p_hPa (lev,time,lat,lon) and Usub/Vsub ---
    U700 = interp_to_p(Usub, p, 700.0)
    V700 = interp_to_p(Vsub, p, 700.0)
    
    # near-sfc
    if "U10" in ds and "V10" in ds:
        usfc = ds.U10.sel(lat=lat_slice, lon=lon_slice)
        vsfc = ds.V10.sel(lat=lat_slice, lon=lon_slice)
    else:
        usfc = Usub.isel(lev=-1)
        vsfc = Vsub.isel(lev=-1)
    
    # shear (recommended)
    wind_shear = np.hypot(U700 - usfc, V700 - vsfc)
    # 4) wind speed at ~700 hPa and shear vs near-sfc
    # ws700 = np.hypot(U700, V700)                     # (time, lat, lon)
    # wind_shear = ws700 - ws
        
    # Constants
    Rd, cp = 287.0, 1005.0
    kappa = Rd / cp
    g = 9.81
    Re = 6.371e6
    deg2rad = np.pi / 180.0
    
    # --- subset fields to same lat/lon BEFORE vectorized isel ---
    sst  = ds.TS.sel(lat=lat_slice, lon=lon_slice)             # (time, lat, lon) [K]
    Tsub = ds.T.sel(lat=lat_slice, lon=lon_slice)               # (lev, time, lat, lon)
    Usub = ds.U.sel(lat=lat_slice, lon=lon_slice)
    Vsub = ds.V.sel(lat=lat_slice, lon=lon_slice)
    omega = ds.OMEGA.sel(lat=lat_slice, lon=lon_slice)
    # Wsub = ds.WSUB.sel(lat=lat_slice, lon=lon_slice)
    q = ds.Q.sel(lat=lat_slice, lon=lon_slice)
    # RHsub = ds.RELHUM.sel(lat=lat_slice, lon=lon_slice)
    
    # Horizontal wind at "surface/lowest" level (adjust if you have a dedicated 10m field)
    # If you already have u,v on surface levels (not lev-ed), just use those here.
    u = ds.U10.sel(lat=lat_slice, lon=lon_slice) if 'U10' in ds else Usub.isel(lev=-1)
    v = ds.V10.sel(lat=lat_slice, lon=lon_slice) if 'V10' in ds else Vsub.isel(lev=-1)
    ws = np.hypot(u, v)
    
    # --- temperatures at nearest 800/700 hPa (vectorized along lev) ---
    T800 = Tsub.isel(lev=p800_idx)                               # (time, lat, lon)
    T700 = Tsub.isel(lev=p700_idx)
    # RH700 = RHsub.isel(lev=p700_idx)
    
    # Potential temperatures
    p0_hpa = 1013.25
    theta_sfc = sst                                              # θ at ~1000 hPa; refine with ps if available
    # If you have surface pressure ps (Pa), use:
    # theta_sfc = sst * (p0_hpa / (ds.ps.sel(lat=lat_slice, lon=lon_slice) * 0.01))**kappa
    theta_800 = T800 * (p0_hpa / 800.0)**kappa
    theta_700 = T700 * (p0_hpa / 700.0)**kappa
    
    # M-value & ΔT
    M  = theta_sfc - theta_800
    dt = Tsub.isel(lev=-1) - sst                                 # lowest model level minus SST
    
    # --- SST advection in K/day (lat/lon in DEGREES) ---
    phi   = np.deg2rad(sst['lat'])
    cosphi = np.clip(np.cos(phi), 1e-6, None)                    # guard near poles
    
    # meters per degree
    m_per_deg_lon = Re * cosphi * deg2rad
    m_per_deg_lat = Re * deg2rad
    
    # spatial gradients (K/m)
    dT_dx = sst.differentiate('lon') / m_per_deg_lon
    dT_dy = sst.differentiate('lat') / m_per_deg_lat
    
    # advection (K/s) -> (K/day)
    Tadv = -(u * dT_dx + v * dT_dy) * 86400.0
    Tadv = Tadv.rename('Tadv').assign_attrs(units='K day-1')
    
    # --- EIS (Wood & Bretherton 2006) ---
    Lhvap = 2.5e6   # J/kg
    Rv    = 461.0   # J/(kg K)
    
    def get_qsat(T, p_hpa):
        # Bolton-type saturation (simple): T in K, p in hPa -> q in kg/kg
        Tcel = T - 273.15
        es = 6.11 * 10**(7.5 * Tcel / (Tcel + 273.15))          # hPa
        return 0.622 * es / p_hpa
    
    # Lower-tropospheric stability
    LTS = theta_700 - theta_sfc
    
    # Mid-layer temperature for Γ_m
    t2m  = Tsub.isel(lev=-1)                                     # near-sfc model level
    T850 = 0.5 * (t2m + T700)
    
    Gammam = (g/cp) * (1.0 - (1.0 + Lhvap * get_qsat(T850, 850.0) / (Rd * T850)) /
                       (1.0 + (Lhvap**2) * get_qsat(T850, 850.0) / (cp * Rv * T850**2)))
    
    # z700: hypsometric with surface T
    z700 = (Rd * t2m / g) * np.log(1000.0 / 700.0)
    
    # LCL (assume 80% RH over MBL); simple approximation
    Tadj = t2m - 55.0
    LCL  = (cp/g) * (Tadj - (1.0/Tadj - np.log(0.8)/2840.0)**(-1))
    
    EIS = LTS - Gammam * (z700 - LCL)
    
    # --- 700-hPa vertical motion in Pa/s ---
    omega700 = omega.isel(lev=p700_idx)
    # omega700 = omega700.assign_attrs(units='Pa s-1')

    # Murphy & Koop (2005) saturation vapor pressure (Pa)
    def es_MK_water(T):
        # ln(esw [Pa]) valid ~123–332 K
        return xr.ufuncs.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
                             + xr.ufuncs.tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
    
    def es_MK_ice(T):
        # ln(esi [Pa]) valid ~110–273 K
        return xr.ufuncs.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
    
    esw = es_MK_water(Tsub)
    esi = es_MK_ice(Tsub)
    # Choose water above freezing, ice at/below (adjust threshold if you prefer 273.16)
    es = xr.where(Tsub > 273.15, esw, esi)
    
    # Saturation specific humidity and RH
    eps = 0.622
    qsat = (eps * es) / (midP - (1.0 - eps) * es)
    
    # Avoid division issues extremely near saturation/low p
    qsat = qsat.clip(min=1e-12)
    
    RH = (q / qsat) * 100.0
    RH = RH.clip(min=0.0, max=100.0)

    RH700 = RH.isel(lev=p700_idx)

    dt = dt.drop_vars('lev')
    ws = ws.drop_vars('lev')
    M = M.drop_vars('lev')
    omega700 = omega700.drop_vars('lev')
    RH700 = RH700.drop_vars('lev')
    
    # Merge and compute daily average
    vars_dict = {
        "SST":       sst, # Surface temperature (not sure if it's the same as SST) 11/10/25
        "deltaT":    dt,
        "WS":        ws,
        "Wind_shear": wind_shear,
        "M":         M,
        "omega700":  omega700,
        "EIS":       EIS,
        "RH700":     RH700,
        "Tadv":      Tadv
    }
    # Ensure names match the keys
    vars_named = {k: (v if getattr(v, "name", None) == k else v.rename(k))
                  for k, v in vars_dict.items()}

    return xr.Dataset(vars_named).drop_vars('lev') # optional: .transpose("time","latitude","longitude")

def compute_cam_aerosol_micphys_metrics(
    ds: xr.Dataset,
    *,
    # geometric std devs by mode (dimensionless)
    sigma=(1.6, 1.6, 1.8, 1.6),
    # latitude and longitude bounds from campaign
    lat_slice,
    lon_slice,
    # diameter bounds (meters)
    bounds=dict(
        n500=(0.5e-6, 10e-6),       # dust N>0.5 µm up to 10 µm
        UHSAS=(0.07e-6, 1.0e-6),    # wing UHSAS
        GNI=(1.4e-6, 16e-6),        # GNI
    ),
    # optionally pre-select a slice, e.g. {'instr_num': 32}
    select: dict | None = None,
    ) -> xr.Dataset:
    """
    Compute CAM aerosol diagnostics from an xarray Dataset `ds`.

    Expects the following variables when available (missing ones are treated as 0):
      Mass mixing ratios (kg/kg): bc_a1, dst_a1, ncl_a1, pom_a1, so4_a1, soa_a1,
                                   so4_a2, soa_a2, ncl_a2, dst_a3, ncl_a3, so4_a3, pom_a4
      Number (1/kg):              num_a1, num_a2, num_a3, num_a4
      Dry geometric mean D (m):   dgnd_a01, dgnd_a02, dgnd_a03, dgnd_a04

    Returns an xr.Dataset with:
      - Total mass by mode & all modes (kg/kg)
      - Species mass fractions per mode (dimensionless)
      - Surface area per mode (m^2/kg) and converted to m^2/m^3 where relevant
      - Dust SA, sea-salt SA, total SA (m^2/m^3)
      - Bounded number (per m^3) for n500, UHSAS, and GNI windows
      - cam_N_UHSAS, cam_N_gni (per m^3)
      - n500_dst (per m^3) using dust fractions

    All operations are vectorized and Dask-lazy.
    """

    if select:
        ds = ds.sel(**select)

    # ---------- helpers ----------
    def _get_or_zero(name: str, like: xr.DataArray | None = None) -> xr.DataArray:
        if name in ds:
            return ds[name]
        if like is None:
            # find a template once
            for k in ("num_a1", "bc_a1", "so4_a1", "dst_a1", "ncl_a1", "pom_a1","CCN1"):
                if k in ds:
                    like = ds[k]
                    break
        if like is None:
            raise KeyError(f"Template not found to create zeros for missing var '{name}'.")
        return xr.zeros_like(like)

    def _safe_frac(numer: xr.DataArray, denom: xr.DataArray) -> xr.DataArray:
        return xr.where(denom > 0, numer / denom, 0.0)

    def _surface_area_m2_per_kg(N: xr.DataArray, dg: xr.DataArray, sig: float) -> xr.DataArray:
        """
        Lognormal surface area moment for spheres: S = N * π * E[D^2],
        with E[D^2] = dg^2 * exp(2 * ln(σ)^2)
        """
        s2 = np.log(sig) ** 2
        return N * np.pi * (dg ** 2) * np.exp(2.0 * s2)

    def _N_between(N, dg, sig, dmin, dmax):
        # ln(sigma)
        s = np.log(sig)
    
        # Avoid σ=1 ⇒ ln(σ)=0 without using xr.full_like on a scalar
        if np.isscalar(s):
            s = 1e-12 if s == 0 else s
        else:  # if you ever pass a DataArray for sig
            s = xr.where(s == 0, 1e-12, s)
    
        # Use np.log with DataArrays (works fine via __array_ufunc__)
        zmax = (np.log(dmax) - np.log(dg)) / (np.sqrt(2.0) * s)
        zmin = (np.log(dmin) - np.log(dg)) / (np.sqrt(2.0) * s)
        return N * 0.5 * (erf(zmax) - erf(zmin))

    def ilev_to_lev_midpoint_da(da_ilev: xr.DataArray, ilev_dim="ilev", lev_dim="lev") -> xr.DataArray:
        """
        Convert an interface (ilev) variable to midlevel (lev) by averaging adjacent interfaces.
        Assumes ilev length = lev length + 1 (typical CAM vertical grid).
        Result is named with lev_dim.
        """
        if ilev_dim not in da_ilev.dims:
            return da_ilev
    
        n = da_ilev.sizes[ilev_dim]
        if n < 2:
            return da_ilev.isel({ilev_dim: slice(0, 0)}).rename({ilev_dim: lev_dim})
    
        mid = 0.5 * (
            da_ilev.isel({ilev_dim: slice(0, n - 1)}) +
            da_ilev.isel({ilev_dim: slice(1, n)})
        )
        return mid.rename({ilev_dim: lev_dim})

    # sigma_w: WP2_CLUBB might be on ilev -> convert to lev
    wp2 = ds.WP2_ZT_CLUBB
    # if "ilev" in wp2.dims and "lev" not in wp2.dims:
    #     wp2 = ilev_to_lev_midpoint_da(wp2, ilev_dim="ilev", lev_dim="lev")
    
    sigma_w = np.sqrt(wp2)  # dask-safe
    
    # Compute pressure altitude (palt) from CESM hybrid coordinates
    T = ds.T
    p0 = 100000 # (ds.P0 old CAM6, it's all the same)  # Reference pressure
    ps = ds.PS  # Surface pressure [=] Pa
    hyai = ds.hyam  # Hybrid A coefficient at layer interface
    hybi = ds.hybm  # Hybrid B coefficient at layer interface
    
    P_dummy = p0 * hyai
    P_pa = (P_dummy + (hybi * ps))  # [=] Pa
    P_hpa = P_pa/100
    Rdry = 287.
    # calc mass density 
    massdens = P_pa / T / Rdry # [=] kg/m3
    
    # calcualted stp mass density
    stp_P = 101325.
    stp_T = 273.15
    massdens_stp = massdens*(stp_P/P_pa)*(T/stp_T) # [=] kg/m3

    # ---------- pull variables (or zeros) ----------
    bc_a1  = _get_or_zero("bc_a1")
    dst_a1 = _get_or_zero("dst_a1")
    ncl_a1 = _get_or_zero("ncl_a1")
    pom_a1 = _get_or_zero("pom_a1")
    so4_a1 = _get_or_zero("so4_a1")
    soa_a1 = _get_or_zero("soa_a1")

    so4_a2 = _get_or_zero("so4_a2")
    soa_a2 = _get_or_zero("soa_a2")
    ncl_a2 = _get_or_zero("ncl_a2")

    dst_a3 = _get_or_zero("dst_a3")
    ncl_a3 = _get_or_zero("ncl_a3")
    so4_a3 = _get_or_zero("so4_a3")

    pom_a4 = _get_or_zero("pom_a4")

    num_a1 = _get_or_zero("num_a1")
    num_a2 = _get_or_zero("num_a2")
    num_a3 = _get_or_zero("num_a3")
    num_a4 = _get_or_zero("num_a4")

    dg1 = _get_or_zero("dgnd_a01", like=num_a1)
    dg2 = _get_or_zero("dgnd_a02", like=num_a2)
    dg3 = _get_or_zero("dgnd_a03", like=num_a3)
    dg4 = _get_or_zero("dgnd_a04", like=num_a4)

    ccn1 = _get_or_zero("CCN1")
    ccn2 = _get_or_zero("CCN2")
    ccn3 = _get_or_zero("CCN3")
    ccn4 = _get_or_zero("CCN4")
    ccn5 = _get_or_zero("CCN5")
    ccn6 = _get_or_zero("CCN6")

    # ---------- total mass by mode ----------
    m_ac1_tot = bc_a1 + dst_a1 + ncl_a1 + pom_a1 + so4_a1 + soa_a1
    m_ac2_tot = so4_a2 + soa_a2 + ncl_a2
    m_ac3_tot = dst_a3 + ncl_a3 + so4_a3
    m_ac4_tot = pom_a4
    m_allmodes_tot = m_ac1_tot + m_ac2_tot + m_ac3_tot + m_ac4_tot

    # ---------- species mass fractions (safe) ----------
    Fbc_ac1  = _safe_frac(bc_a1,  m_ac1_tot)
    Fdst_ac1 = _safe_frac(dst_a1, m_ac1_tot)
    Fdst_ac3 = _safe_frac(dst_a3, m_ac3_tot)
    Fncl_ac1 = _safe_frac(ncl_a1, m_ac1_tot)
    Fncl_ac2 = _safe_frac(ncl_a2, m_ac2_tot)
    Fncl_ac3 = _safe_frac(ncl_a3, m_ac3_tot)
    Fpom_ac1 = _safe_frac(pom_a1, m_ac1_tot)
    Fpom_ac4 = _safe_frac(pom_a4, m_ac4_tot)
    Fso4_ac1 = _safe_frac(so4_a1, m_ac1_tot)
    Fso4_ac2 = _safe_frac(so4_a2, m_ac2_tot)
    Fso4_ac3 = _safe_frac(so4_a3, m_ac3_tot)
    Fsoa_ac1 = _safe_frac(soa_a1, m_ac1_tot)
    Fsoa_ac2 = _safe_frac(soa_a2, m_ac2_tot)

    # ---------- surface area (per kg) ----------
    s1 = _surface_area_m2_per_kg(num_a1, dg1, sigma[0])
    s2 = _surface_area_m2_per_kg(num_a2, dg2, sigma[1])
    s3 = _surface_area_m2_per_kg(num_a3, dg3, sigma[2])
    s4 = _surface_area_m2_per_kg(num_a4, dg4, sigma[3])

    # Convert selected SA to m^2/m^3 (multiply by ρ)
    ss_SA_tot = (Fncl_ac1 * s1 + Fncl_ac2 * s2 + Fncl_ac3 * s3) * massdens_stp
    ss_SA_sub = (Fncl_ac1 * s1 + Fncl_ac2 * s2) * massdens_stp
    Stot_dst  = (Fdst_ac1 * s1 + Fdst_ac3 * s3) * massdens_stp
    Stot_tot  = (s1 + s2 + s3 + s4) * massdens_stp

    # ---------- bounded number integrals (per m^3) ----------
    Dmin, Dmax = bounds["n500"]
    n500_ac1 = _N_between(num_a1, dg1, sigma[0], Dmin, Dmax) * massdens_stp
    n500_ac3 = _N_between(num_a3, dg3, sigma[2], Dmin, Dmax) * massdens_stp
    n500_dst = Fdst_ac1 * n500_ac1 + Fdst_ac3 * n500_ac3

    Dmin, Dmax = bounds["UHSAS"]
    nUHSAS_ac1 = _N_between(num_a1, dg1, sigma[0], Dmin, Dmax) * massdens_stp
    nUHSAS_ac2 = _N_between(num_a2, dg2, sigma[1], Dmin, Dmax) * massdens_stp
    nUHSAS_ac3 = _N_between(num_a3, dg3, sigma[2], Dmin, Dmax) * massdens_stp
    nUHSAS_ac4 = _N_between(num_a4, dg4, sigma[3], Dmin, Dmax) * massdens_stp
    cam_N_UHSAS = (nUHSAS_ac1 + nUHSAS_ac2 + nUHSAS_ac3 + nUHSAS_ac4) / 1e6 # /cm3

    Dmin, Dmax = bounds["GNI"]
    # sea-salt fraction * number per kg, then integrate between bounds:
    num_ncl_a1 = Fncl_ac1 * num_a1
    num_ncl_a2 = Fncl_ac2 * num_a2
    num_ncl_a3 = Fncl_ac3 * num_a3
    nGNI_ac1 = _N_between(num_ncl_a1, dg1, sigma[0], Dmin, Dmax) * massdens_stp
    nGNI_ac2 = _N_between(num_ncl_a2, dg2, sigma[1], Dmin, Dmax) * massdens_stp
    nGNI_ac3 = _N_between(num_ncl_a3, dg3, sigma[2], Dmin, Dmax) * massdens_stp
    cam_N_gni = nGNI_ac1 + nGNI_ac2 + nGNI_ac3

    T = ds.T

    #################################
    # Load microphysics parameters
    #################################
    
    # Cloud water 
    qc = (ds.CLDLIQ/ds.FREQL)*massdens*1000 # kg/kg to g/m3
    nc = (ds.AWNC/ds.FREQL) / 1e6 # /cm3
    
    qr = (ds.AQRAIN/ds.FREQR)*massdens*1000 # kg/kg to g/m3
    nr = (ds.ANRAIN/ds.FREQR) / 1e6 # /cm3
    
    qliq = qc+qr

    # Calcualate dew point from specific humidity
    # 1. vapor pressure (Pa)
    eps = 0.622
    e = (ds.Q * P_pa) / (eps + (1 - eps) * ds.Q)
    
    # prevent numerical issues
    e = xr.where(e <= 0, np.nan, e)
    
    # 2. dew point (°C) from Magnus formula
    ln_ratio = xr.ufuncs.log(e / 611.2)
    Td_C = (243.5 * ln_ratio) / (17.67 - ln_ratio)
    
    # Optionally convert to Kelvin
    Td = Td_C + 273.15
    
    # Murphy & Koop (2005) saturation vapor pressure (Pa)
    def es_MK_water(T):
        # ln(esw [Pa]) valid ~123–332 K
        return xr.ufuncs.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
                             + xr.ufuncs.tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
    
    def es_MK_ice(T):
        # ln(esi [Pa]) valid ~110–273 K
        return xr.ufuncs.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
    
    esw = es_MK_water(T)
    esi = es_MK_ice(T)
    # Choose water above freezing, ice at/below (adjust threshold if you prefer 273.16)
    es = xr.where(T > 273.15, esw, esi)
    
    # Saturation specific humidity and RH
    eps = 0.622
    qsat = (eps * es) / (P_pa - (1.0 - eps) * es)
    
    # Avoid division issues extremely near saturation/low p
    qsat = qsat.clip(min=1e-12)
    
    RH = (ds.Q / qsat) * 100.0

    Rd = 287.0      # J kg-1 K-1
    g  = 9.81       # m s-2

    # if 'WSUB' in ds:
    #     w = ds.WSUB
    # else:
    w = -ds.OMEGA * Rd * ds.T / (g * P_pa)
    
    # # Ensure names match the keys
    # vars_named = {k: (v if getattr(v, "name", None) == k else v.rename(k))
    #               for k, v in vars_dict.items()}
    
    # ds_cam_mic = xr.Dataset(vars_named).sel(lat=lat_slice, lon=lon_slice)

    # ---------- assemble output ----------
    out = xr.Dataset(
        {
            # State
            "P_hPa": P_hpa,
            "T_K": T,
            "Td_K": Td,
            "RH": RH,
            "U": ds.U,
            "V": ds.V,
            "W": w,
            "sigma_w": sigma_w,
            # masses
            "m_ac1_tot": m_ac1_tot,
            "m_ac2_tot": m_ac2_tot,
            "m_ac3_tot": m_ac3_tot,
            "m_ac4_tot": m_ac4_tot,
            "m_allmodes_tot": m_allmodes_tot,
            # mass fractions
            "Fbc_ac1": Fbc_ac1,
            "Fdst_ac1": Fdst_ac1, "Fdst_ac3": Fdst_ac3,
            "Fncl_ac1": Fncl_ac1, "Fncl_ac2": Fncl_ac2, "Fncl_ac3": Fncl_ac3,
            "Fpom_ac1": Fpom_ac1, "Fpom_ac4": Fpom_ac4,
            "Fso4_ac1": Fso4_ac1, "Fso4_ac2": Fso4_ac2, "Fso4_ac3": Fso4_ac3,
            "Fsoa_ac1": Fsoa_ac1, "Fsoa_ac2": Fsoa_ac2,
            # surface areas
            "Stot_ac1_m2kg": s1, "Stot_ac2_m2kg": s2, "Stot_ac3_m2kg": s3, "Stot_ac4_m2kg": s4,
            "ss_SA_tot": ss_SA_tot, "ss_SA_sub": ss_SA_sub,
            "Stot_dst": Stot_dst, "Stot_tot": Stot_tot,
            # bounded numbers
            "n500_ac1": n500_ac1, "n500_ac3": n500_ac3, "n500_dst": n500_dst,
            "nUHSAS_ac1": nUHSAS_ac1, "nUHSAS_ac2": nUHSAS_ac2, "nUHSAS_ac3": nUHSAS_ac3, "nUHSAS_ac4": nUHSAS_ac4,
            "cam_N_UHSAS": cam_N_UHSAS,
            "nGNI_ac1": nGNI_ac1, "nGNI_ac2": nGNI_ac2, "nGNI_ac3": nGNI_ac3,
            "cam_N_gni": cam_N_gni,
            "cam_lwc": qliq,
            "cam_Nc": nc,
            "cam_Nr": nr
        }
    )

    # annotate a few units
    out["Stot_ac1_m2kg"].attrs["units"] = "m^2 kg^-1"
    out["Stot_tot"].attrs["units"] = "m^2 m^-3"
    out["cam_N_UHSAS"].attrs["units"] = "cm^-3"
    out["cam_N_gni"].attrs["units"] = "m^-3"
    out.attrs.update(
        sigma_modes=sigma,
        massdens_units="kg m^-3",
        note="Derived using lognormal moments; safe division used for mass fractions.",
    )
    out = out.sel(lat =lat_slice, lon =lon_slice)

    return out

def subset_cam_by_campaign(
    ds_cam: xr.Dataset,
    ds_cam_ccfs: xr.Dataset,
    campaign: str,
    vars_to_keep: list[str] | None = None,
    drop: bool = True,       # drop points outside mask (shrinks time/lat/lon)
    add_qlwc: bool = True,   # add qlwc = Qc+Qr if present (g m^-3)
):
    """
    Return ds_cam masked by campaign-specific conditions built from ds_cam_ccfs.
    No averaging/compositing — you get the full variables (Qc, Qr, etc.) where the
    mask is True. Works with 4D vars (time, lev, lat, lon).

    Campaign rules:

    CSET
      STRAT: ((M < -10) & (SST < 295)) | ((M < -11) & (Tadv < 0))
      OPEN : ((M >= -10) & (SST >= 296)) | ((M >= -10) & (Tadv >= 5e-4))

    SOCRATES
      STRAT: ((M < -9)  & (WS < 9)) | ((M < -10) & (EIS > 7)) | ((M < -10) & (wshear < 9))
      OPEN : ((M >= -7) & (WS >= 9)) | ((M >= -8) & (EIS < 9)) | ((M >= -8) & (wshear > 6))
    """

    # --- alias helper so we’re resilient to naming (adjust as needed)
    _aliases = {
        "M":      ("M",),
        "WS":     ("WS", "Wind_sp", "Wind_speed"),
        "EIS":    ("EIS",),
        "wshear": ("wshear", "Wind_shear"),
        "SST":    ("SST","ERA5_SST","sst"),
        "Tadv":   ("Tadv","T_adv","Temp_adv"),
    }
    def _getv(ds, key):
        for k in _aliases.get(key, (key,)):
            if k in ds:
                return ds[k]
        raise KeyError(f"Missing '{key}' (aliases { _aliases.get(key) }) in ds_cam_ccfs.")

    # Align so masks and fields line up on (time, lat, lon)
    ds_cam_aln, ds_ccfs_aln = xr.align(ds_cam, ds_cam_ccfs, join="inner")
    
    # Add qlwc if requested and available (already g m^-3 in your case)
    if add_qlwc and ("Qc" in ds_cam_aln) and ("Qr" in ds_cam_aln):
        ds_cam_aln = ds_cam_aln.assign(qlwc=ds_cam_aln["Qc"])
    
    # Keep only requested variables (default: keep everything, so Qc/Qr stay)
    ds_vars = ds_cam_aln[vars_to_keep] if vars_to_keep is not None else ds_cam_aln

    # Build masks from ccfs set
    M = _getv(ds_ccfs_aln, "M")
    camp = campaign.strip().upper()
    if camp == "CSET":
        SST  = _getv(ds_ccfs_aln, "SST")
        Tadv = _getv(ds_ccfs_aln, "Tadv")
        mask_strat = ((M < -10) & (SST < 295)) | ((M < -11) & (Tadv < 0)).astype(bool)
        mask_open  = ((M >= -10) & (SST >= 296)) | ((M >= -10) & (Tadv >= -4)).astype(bool)
    elif camp == "SOCRATES":
        WS     = _getv(ds_ccfs_aln, "WS")
        EIS    = _getv(ds_ccfs_aln, "EIS")
        WSHEAR = _getv(ds_ccfs_aln, "wshear")
        mask_strat = ((M < -9)  & (WS < 9)) | ((M < -10) & (EIS > 7)) | ((M < -10) & (WSHEAR < 9)).astype(bool)
        mask_open  = ((M >= -7) & (WS >= 9)) | ((M >= -8) & (EIS < 9)) | ((M >= -8) & (WSHEAR > 6)).astype(bool)
    else:
        raise ValueError("campaign must be 'CSET' or 'SOCRATES'.")

    # Apply masks (broadcasts over 'lev' automatically for 4D vars)
    if drop:
        # ✅ compute masks fully before dropping to avoid boolean dask indexing
        ms = mask_strat.compute()
        mo = mask_open.compute()
        ds_strat = ds_vars.where(ms, drop=True)
        ds_open  = ds_vars.where(mo, drop=True)
    else:
        # keep lazily with NaNs (no boolean indexing)
        ds_strat = ds_vars.where(mask_strat)
        ds_open  = ds_vars.where(mask_open)

    return {
        "strat": ds_strat,   # full ds with Qc, Qr, … masked to strat condition
        "open":  ds_open,    # full ds with Qc, Qr, … masked to open condition
        "masks": {"strat": mask_strat, "open": mask_open},
    }

def collocate_cam_points_only(
    df,
    ds_cam,
    time_col="Time",
    lat_col="GGLAT",
    lon_col="GGLON",
    pressure_col="PSXC",
    regime_col="cloud_regime",
    cam_time_name="time",
    cam_lat_name="lat",
    cam_lon_name="lon",
    cam_lev_name="lev",
    cam_pressure_name="P_hPa",
    lon_to_360=True,
    drop_duplicate_cam_points=False,
    duplicate_method="majority",
    drop_undetermined=False,
    undetermined_label="Undetermined",
    compute_result=True,
):
    """
    Collocate aircraft samples to CAM by nearest time/lat/lon/pressure and
    return only the collocated CAM points, with all CAM variables preserved.

    Output dataset has dimension:
        points

    Coordinates/variables included:
    - matched CAM time/lat/lon/lev
    - all CAM variables at those matched points
    - aircraft-observed cloud regime attached to each point
    - optional majority-vote collapse to unique CAM sampled points

    Parameters
    ----------
    df : pandas.DataFrame
        Aircraft dataframe.
    ds_cam : xarray.Dataset
        CAM dataset with dims (time, lev, lat, lon).
    drop_duplicate_cam_points : bool
        If False, return one collocated CAM point per aircraft row.
        If True, collapse repeated aircraft hits to unique CAM points.
    duplicate_method : str
        Currently only "majority" is supported when collapsing duplicates.
    drop_undetermined : bool
        If True, remove Undetermined rows before duplicate collapsing.
    compute_result : bool
        If True, compute the result before returning.

    Returns
    -------
    cam_point_ds : xarray.Dataset
        Collocated CAM dataset with dimension 'points'.
    regime_map : pandas.DataFrame
        Table showing the unique matched CAM points and regime assignment.
        If drop_duplicate_cam_points=False, this is still returned as metadata.
    """

    def convert_cam_time_to_datetime64(ds, time_name):
        if np.issubdtype(ds[time_name].dtype, np.datetime64):
            return ds
        try:
            new_time = pd.to_datetime(ds[time_name].values.astype("datetime64[ns]"))
        except Exception:
            new_time = pd.to_datetime([str(t) for t in ds[time_name].values])
        return ds.assign_coords({time_name: new_time})

    def regime_to_code(series):
        mapping = {
            "Undetermined": 0,
            "Stratocumulus": 1,
            "Open-Cell": 2,
        }
        return series.map(mapping).fillna(0).astype(np.int16), mapping

    # -----------------------------
    # Checks
    # -----------------------------
    required_df_cols = [time_col, lat_col, lon_col, pressure_col, regime_col]
    missing_df = [c for c in required_df_cols if c not in df.columns]
    if missing_df:
        raise ValueError(f"Missing required dataframe columns: {missing_df}")

    required_cam = [cam_time_name, cam_lat_name, cam_lon_name, cam_lev_name, cam_pressure_name]
    missing_cam = [c for c in required_cam if c not in ds_cam.variables and c not in ds_cam.coords]
    if missing_cam:
        raise ValueError(
            f"Missing required CAM variables/coords: {missing_cam}\n"
            f"Available coords: {list(ds_cam.coords)}\n"
            f"Available data_vars: {list(ds_cam.data_vars)}"
        )

    if drop_duplicate_cam_points and duplicate_method != "majority":
        raise ValueError("Only duplicate_method='majority' is currently supported.")

    # -----------------------------
    # Fix CAM time
    # -----------------------------
    ds_cam = convert_cam_time_to_datetime64(ds_cam, cam_time_name)

    # -----------------------------
    # Clean aircraft df
    # -----------------------------
    df_use = df.copy()
    df_use[time_col] = pd.to_datetime(df_use[time_col], errors="coerce")

    valid_mask = (
        df_use[time_col].notna()
        & df_use[lat_col].notna()
        & df_use[lon_col].notna()
        & df_use[pressure_col].notna()
        & df_use[regime_col].notna()
    )
    df_use = df_use.loc[valid_mask].copy()

    if len(df_use) == 0:
        raise ValueError("No valid aircraft rows remain after dropping NaNs.")

    # -----------------------------
    # Longitude handling
    # -----------------------------
    cam_lon_max = float(ds_cam[cam_lon_name].max().values)
    if lon_to_360 and cam_lon_max > 180:
        lon_vals = (df_use[lon_col].values + 360.0) % 360.0
    else:
        lon_vals = df_use[lon_col].values

    # -----------------------------
    # Nearest time/lat/lon selection
    # -----------------------------
    points = xr.DataArray(np.arange(len(df_use)), dims="points")

    cam_horiz = ds_cam.sel(
        {
            cam_time_name: xr.DataArray(df_use[time_col].values, dims="points"),
            cam_lat_name: xr.DataArray(df_use[lat_col].values, dims="points"),
            cam_lon_name: xr.DataArray(lon_vals, dims="points"),
        },
        method="nearest",
    )

    # -----------------------------
    # Vertical match by pressure
    # -----------------------------
    p_air = df_use[pressure_col].values[:, None]
    p_cam = cam_horiz[cam_pressure_name].values

    if p_cam.ndim != 2:
        raise ValueError(
            f"{cam_pressure_name} after horiz collocation should be 2D "
            f"(points, {cam_lev_name}), got shape {p_cam.shape} and dims {cam_horiz[cam_pressure_name].dims}"
        )

    lev_idx = np.abs(p_cam - p_air).argmin(axis=1)

    cam_point_ds = cam_horiz.isel(
        {
            "points": points,
            cam_lev_name: xr.DataArray(lev_idx, dims="points"),
        }
    )

    if compute_result:
        cam_point_ds = cam_point_ds.compute()

    # -----------------------------
    # Attach aircraft info to point dataset
    # -----------------------------
    regime_str = df_use[regime_col].astype(str)
    regime_code, mapping = regime_to_code(regime_str)

    cam_point_ds = cam_point_ds.assign_coords(
        aircraft_time=("points", df_use[time_col].values),
        aircraft_lat=("points", df_use[lat_col].values),
        aircraft_lon=("points", df_use[lon_col].values),
        aircraft_pressure_hPa=("points", df_use[pressure_col].values),
        obs_cloud_regime=("points", regime_str.values),
        obs_cloud_regime_code=("points", regime_code.values),
    )

    # Diagnostics
    cam_point_ds["cam_pdiff_hPa"] = (
        "points",
        np.abs(df_use[pressure_col].values - cam_point_ds[cam_pressure_name].values),
    )
    
    dt_sec = np.abs(
        (
            pd.to_datetime(df_use[time_col].values)
            - pd.to_datetime(cam_point_ds[cam_time_name].values)
        ).total_seconds()
    )
    
    cam_point_ds["cam_timedelta_s"] = ("points", dt_sec)

    # -----------------------------
    # Build regime map for unique CAM points
    # -----------------------------
    df_match = pd.DataFrame({
        "cam_time_match": pd.to_datetime(cam_point_ds[cam_time_name].values),
        "cam_lat_match": cam_point_ds[cam_lat_name].values,
        "cam_lon_match": cam_point_ds[cam_lon_name].values,
        "cam_lev_match": cam_point_ds[cam_lev_name].values,
        "aircraft_regime": regime_str.values,
    })

    if drop_undetermined:
        df_match = df_match[df_match["aircraft_regime"] != undetermined_label].copy()

    group_cols = ["cam_time_match", "cam_lat_match", "cam_lon_match", "cam_lev_match"]

    if len(df_match) > 0:
        regime_map = (
            df_match.groupby(group_cols)["aircraft_regime"]
            .agg(
                obs_cloud_regime=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
                obs_cloud_regime_n="size",
                obs_cloud_regime_agreement=lambda x: x.value_counts(normalize=True).iloc[0],
            )
            .reset_index()
        )
    else:
        regime_map = pd.DataFrame(
            columns=group_cols + ["obs_cloud_regime", "obs_cloud_regime_n", "obs_cloud_regime_agreement"]
        )

    regime_map["obs_cloud_regime_code"] = (
        regime_map["obs_cloud_regime"].map(mapping).fillna(0).astype(np.int16)
    )

    # -----------------------------
    # Optionally collapse to unique CAM points
    # -----------------------------
    if drop_duplicate_cam_points:
        # Make a DataFrame of collocated CAM points with all variables
        cam_df = cam_point_ds.to_dataframe().reset_index()

        # Drop aircraft-specific duplicate rows by unique matched CAM point
        cam_df = cam_df.merge(
            regime_map,
            left_on=[cam_time_name, cam_lat_name, cam_lon_name, cam_lev_name],
            right_on=["cam_time_match", "cam_lat_match", "cam_lon_match", "cam_lev_match"],
            how="left",
            suffixes=("", "_majority"),
        )

        cam_df = cam_df.drop_duplicates(
            subset=[cam_time_name, cam_lat_name, cam_lon_name, cam_lev_name]
        ).copy()

        # Replace pointwise regime with majority-vote regime
        cam_df["obs_cloud_regime"] = cam_df["obs_cloud_regime_majority"]
        cam_df["obs_cloud_regime_code"] = cam_df["obs_cloud_regime_code_majority"]
        cam_df["obs_cloud_regime_n"] = cam_df["obs_cloud_regime_n"]
        cam_df["obs_cloud_regime_agreement"] = cam_df["obs_cloud_regime_agreement"]

        # Remove merge clutter if present
        drop_cols = [
            "cam_time_match", "cam_lat_match", "cam_lon_match", "cam_lev_match",
            "aircraft_time", "aircraft_lat", "aircraft_lon", "aircraft_pressure_hPa"
        ]
        for c in drop_cols:
            if c in cam_df.columns:
                pass  # keep or drop depending on preference

        cam_point_ds = xr.Dataset.from_dataframe(
            cam_df.set_index(["points"]) if "points" in cam_df.columns else cam_df
        )

    return cam_point_ds, regime_map

