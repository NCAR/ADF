"""
Everything that builds <campaign>_sonde_data_composite.nc, copied verbatim.

Source: INFORM-COMPASS-cookbook on GitHub, branch data-wrapper-updates_030626,
commit 6947666a0d311ba94fd964825261b74b42b1b20c (the pin in cookbook_ref.json):
  - INFORM_process_system_database.ipynb, code cells 0, 21, 23, 25
  - inform_utils.py: find_sondes, read_sonde and the helpers it calls
  - process_data_products_utils.py: select_ERA5_4flight, collocate_ERA5_sonde,
    cloud_regime_sonde and the helpers they call
Line and cell numbers are those of that commit.

Nothing below is edited.  Section order: the notebook cells as they run,
then the functions they call, grouped by source module in source order.
Note: notebook cell 25 uses np (CSET branch) but no notebook cell imports
numpy.
"""

# ==============================================================================
# NOTEBOOK: INFORM_process_system_database.ipynb
# ==============================================================================


# --- cell 0 (imports) ---
###### INFORM .py files necessary for processing aircraft data
import inform_utils as inform  # INFORM python module inform_utils.py
import process_data_products_utils as pdp  # INFORM python module, process_data_products

#######

import xarray as xr
from datetime import time
import warnings

warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm, trange
import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import pandas as pd


# --- cell 21 ---
# Select campaign 'SOCRATES' or 'CSET
# =================================================
campaign = "CSET"
# =================================================

# Path to dropsondes – recognizes .nc, .cls, and .eol files
# ==================================================
dir = f"/glade/work/patnaude/{campaign}/dropsondes"
# =================================================


# --- cell 23 ---
paths = inform.find_sondes(dir)
dfs, times = inform.read_sonde(paths, campaign=campaign)
sonde_dat = []
for i in tqdm(range(0, len(dfs))):
    print("sonde number", i)
    ds = pdp.select_ERA5_4flight(dfs[i], campaign=campaign, dat_type="dropsonde")
    # Collocate ERA5 data and calculate environmental controlling factors
    sonde_coll = pdp.collocate_ERA5_sonde(ds, dfs[i])
    # Regime each sonde based on average column value
    regime_sonde = pdp.cloud_regime_sonde(sonde_coll, campaign=campaign)
    sonde_dat.append(regime_sonde)

all_sondes = pd.concat(sonde_dat)


# --- cell 25 ---
if campaign == "CSET":
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

    ds.to_netcdf(f"{campaign}_sonde_data_composite.nc", encoding=encoding)

elif campaign == "SOCRATES":
    # SOCRATES
    ds = xr.Dataset.from_dataframe(all_sondes)
    ds.to_netcdf(f"{campaign}_sonde_data_composite.nc")


# ==============================================================================
# inform_utils.py (module imports: netCDF4, pathlib, pandas as pd, numpy as np, os,
# datetime, fnmatch, typing, xarray as xr, re)
# ==============================================================================


# --- inform_utils.py lines 751-768 ---
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
    valid_ext = (".nc", ".cls", ".eol")
    fnames = sorted(
        fname for fname in os.listdir(dir_path) if fname.lower().endswith(valid_ext)
    )
    return [os.path.join(dir_path, fname) for fname in fnames]


# --- inform_utils.py lines 770-770 ---
PathLike = Union[str, Path]


# --- inform_utils.py lines 772-799 ---
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


# --- inform_utils.py lines 801-803 ---
def _infer_rf_from_name(name: str) -> Optional[int]:
    m = re.search(r"RF(\d{1,2})", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


# --- inform_utils.py lines 806-808 ---
def _sort_key_default(p: Path) -> str:
    # Keep it simple/stable: filename sort
    return p.name


# --- inform_utils.py lines 810-871 ---
def _read_cls_multi(file_path: Path) -> Tuple[List[pd.DataFrame], List[pd.Timestamp]]:
    """
    Reads a `.cls` radiosonde file and extracts multiple datasets with nominal release times.
    Returns (datasets, nominal_times).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start_indices = [
        i for i, line in enumerate(lines) if "Nominal Release Time" in line
    ]
    if not start_indices:
        raise ValueError(
            f"No 'Nominal Release Time' entries found in file: {file_path}"
        )

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


# --- inform_utils.py lines 873-970 ---
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
                launch_time = pd.to_datetime(
                    rhs, format="%Y, %m, %d, %H:%M:%S", utc=True
                )
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
        "Time",
        "hh",
        "mm",
        "ss",
        "Press",
        "Temp",
        "Dewpt",
        "RH",
        "Uwind",
        "Vwind",
        "Wspd",
        "Dir",
        "dZ",
        "GeoPoAlt",
        "Lon",
        "Lat",
        "GPSAlt",
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
        raise ValueError(
            f"No data rows parsed in {p.name} (expected {len(columns)} columns)"
        )

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
    df.attrs["nominal_time"] = (
        launch_time  # if you want to treat nominal=launch for .eol
    )
    # --- assign RF for CSET ---
    if campaign is not None and campaign.upper() == "CSET":
        rf = _assign_rf_cset(launch_time)
        df["RF"] = rf
    # put Time first
    lead = ["Time"] + [c for c in ("GGLAT", "GGLON") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    return df, launch_time


# --- inform_utils.py lines 972-1036 ---
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
                    out_name = (
                        vname
                        if vname.startswith("reference_")
                        else f"{reference_prefix}{vname}"
                    )
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


# --- inform_utils.py lines 1038-1212 ---
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
        Callable[
            [Path],
            Tuple[
                Union[pd.DataFrame, List[pd.DataFrame]],
                Union[pd.Timestamp, List[pd.Timestamp], None],
            ],
        ]
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
        df = pd.DataFrame(cols).rename(
            columns={"time": "Time", "lat": "GGLAT", "lon": "GGLON"}
        )
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
                if (
                    rf is None
                    and campaign is not None
                    and campaign.upper() == "CSET"
                    and pd.notna(nominal)
                ):
                    rf = _assign_rf_cset(nominal.to_pydatetime())

                # Skip if RF still unknown (or set to -1 if you prefer keeping them)
                if rf is None:
                    rf = -1

                # Store
                records.append((int(rf), nominal, df, t))

        elif ext == ".eol":
            df, t = _read_eol_one(fp, campaign)  # t is launch_time (UTC)
            nominal = pd.to_datetime(t, utc=True, errors="coerce")

            # RF: _read_eol_one already sets df["RF"] for CSET, but we standardize here
            rf = None
            if "RF" in df.columns and pd.notna(df["RF"].iloc[0]):
                rf = int(df["RF"].iloc[0])
            else:
                rf = _infer_rf_from_name(fp.name)
                if (
                    rf is None
                    and campaign is not None
                    and campaign.upper() == "CSET"
                    and pd.notna(nominal)
                ):
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
            nominal = (
                pd.to_datetime(t, utc=True, errors="coerce")
                if t is not None
                else pd.NaT
            )

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


# ==============================================================================
# process_data_products_utils.py (module imports: pandas as pd, numpy as np,
# inform_utils as inform, glob, xarray as xr, datetime, scipy.spatial.cKDTree, ...)
# ==============================================================================


# --- process_data_products_utils.py lines 522-741 ---
def select_ERA5_4flight(df, campaign, dat_type="aircraft"):
    # Define function to filter ERA5 files based on time
    def get_matching_files(pattern, start_dt, end_dt):
        file_list = glob.glob(pattern)
        selected = []
        for file in file_list:
            time_strs = file.split(".")[-2].split("_")
            file_start = datetime.datetime.strptime(time_strs[0], "%Y%m%d%H")
            file_end = datetime.datetime.strptime(time_strs[1], "%Y%m%d%H")
            if file_start <= end_dt and file_end >= start_dt:
                selected.append(file)
        return selected

    filepath_sfc = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"
    filepath_pl = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    # Extract the times of the research flight
    month, year = df.Time[0].month, df.Time[0].year
    day_start, day_end = df.Time[0].day, df.Time.iloc[-1].day
    start_hour, end_hour = df.Time[0].hour, df.Time.iloc[-1].hour

    # Select the latitude/longitude box to reduce size of era5 data
    if campaign == "SOCRATES":
        lat_max, lat_min = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())
    elif campaign == "CSET":
        lat_min, lat_max = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())

    # ---- build lat/lon selection from the flight track ----
    # Aircraft → 0..360 to match ERA5
    lon0 = ((df.GGLON.to_numpy(dtype=float) % 360.0) + 360.0) % 360.0
    lat0 = df.GGLAT.to_numpy(dtype=float)

    # Robust bounds (with a small pad for ERA5 0.25° grid)
    pad = 0.5
    lon_min = float(np.floor(np.nanmin(lon0) - pad))
    lon_max = float(np.ceil(np.nanmax(lon0) + pad))
    # keep inside ERA5 domain
    lon_min = max(0.0, lon_min)
    lon_max = min(359.999, lon_max)

    lat_min = float(np.floor(np.nanmin(lat0) - pad))
    lat_max = float(np.ceil(np.nanmax(lat0) + pad))

    # ERA5 latitude is usually descending (90 → -90): use slice(max, min)
    lat_slice = slice(lat_max, lat_min)

    # ERA5 longitudes are ascending (0 → 360): use slice(min, max)
    lon_slice = slice(lon_min, lon_max)

    print(
        "Selecting ERA5 box:",
        f"lon {lon_min}→{lon_max} (0–360), lat {lat_max}→{lat_min} (descending)",
    )

    # Make the yearmonth string for file selection
    dir_date = f"{year}{month:02d}"

    # Flight start and end times
    start_dt = datetime.datetime(year, month, day_start, start_hour)
    end_dt = datetime.datetime(year, month, day_end, end_hour) + datetime.timedelta(
        hours=1
    )

    # ---- apply the SAME slices to every dataset you open ----
    ds_sp = xr.open_mfdataset(
        get_matching_files(f"{filepath_sfc}{dir_date}/*_sp.*.nc", start_dt, end_dt),
        combine="by_coords",
    ).sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt))
    ds_sst = xr.open_mfdataset(
        get_matching_files(f"{filepath_sfc}{dir_date}/*_sstk.*.nc", start_dt, end_dt),
        combine="by_coords",
    ).sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt))
    ds_t2m = xr.open_mfdataset(
        get_matching_files(f"{filepath_sfc}{dir_date}/*_2t.*.nc", start_dt, end_dt),
        combine="by_coords",
    ).sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt))
    ds_u10 = xr.open_mfdataset(
        get_matching_files(f"{filepath_sfc}{dir_date}/*_10u.*.nc", start_dt, end_dt),
        combine="by_coords",
    )[["VAR_10U"]].sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )
    ds_v10 = xr.open_mfdataset(
        get_matching_files(f"{filepath_sfc}{dir_date}/*_10v.*.nc", start_dt, end_dt),
        combine="by_coords",
    )[["VAR_10V"]].sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )

    ds_w = xr.open_mfdataset(
        get_matching_files(f"{filepath_pl}{dir_date}/*_w.*.nc", start_dt, end_dt),
        combine="nested",
        concat_dim="time",
    )
    w_700 = (
        ds_w["W"]
        .sel(level=700)
        .sortby("time")
        .sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt))
    )

    # ds_rh700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_r.*.nc", start_dt, end_dt),
    #                              combine='nested', concat_dim='time')[['R']].sel(level=700).drop_vars('level', errors='ignore')
    # rh      = ds_rh700['R'].rename('RH').sortby('time').sel(latitude=lat_slice, longitude=lon_slice,
    # time=slice(start_dt,end_dt))
    ds_q700 = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_q.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["Q"]]
        .sel(level=700)
        .drop_vars("level", errors="ignore")
    )
    q = (
        ds_q700["Q"]
        .rename("Q")
        .sortby("time")
        .sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt))
    )
    ds_u700 = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_u.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["U"]]
        .sel(level=700)
        .drop_vars("level", errors="ignore")
    )
    ds_u700 = ds_u700.sortby("time").sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )
    ds_v700 = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_v.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["V"]]
        .sel(level=700)
        .drop_vars("level", errors="ignore")
    )
    ds_v700 = ds_v700.sortby("time").sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )

    ds_t = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["T"]]
        .sel(level=800)
        .drop_vars("level", errors="ignore")
    )
    ds_t = ds_t.sortby("time").sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )
    ds_t700 = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["T"]]
        .sel(level=700)
        .drop_vars("level", errors="ignore")
    )
    ds_t700 = ds_t700.sortby("time").sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )
    ds_t850 = (
        xr.open_mfdataset(
            get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
            combine="nested",
            concat_dim="time",
        )[["T"]]
        .sel(level=850)
        .drop_vars("level", errors="ignore")
    )
    ds_t850 = ds_t700.sortby("time").sel(
        latitude=lat_slice, longitude=lon_slice, time=slice(start_dt, end_dt)
    )

    ws = np.sqrt(ds_u10.VAR_10U**2 + ds_v10.VAR_10V**2)
    wind_dir = (270 - np.degrees(np.arctan2(ds_v10.VAR_10V, ds_u10.VAR_10U))) % 360

    # Calculate RH from specfic humidity
    # Murphy & Koop (2005) saturation vapor pressure (Pa)
    def es_MK_water(T):
        # ln(esw [Pa]) valid ~123–332 K
        return xr.ufuncs.exp(
            54.842763
            - 6763.22 / T
            - 4.210 * np.log(T)
            + 0.000367 * T
            + xr.ufuncs.tanh(0.0415 * (T - 218.8))
            * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T)
        )

    def es_MK_ice(T):
        # ln(esi [Pa]) valid ~110–273 K
        return xr.ufuncs.exp(
            9.550426 - 5723.265 / T + 3.53068 * np.log(T) - 0.00728332 * T
        )

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
    wind_shear = ws700 - ws

    # Calculate M-value
    Rd = 287
    Cp = 1005
    theta_sfc = ds_t2m.VAR_2T * (101325 / ds_sp.SP) ** (Rd / Cp)
    theta_800 = ds_t * (1013.25 / 800) ** (Rd / Cp)

    M = theta_sfc.T - theta_800.T
    M = M.transpose("time", "latitude", "longitude")

    dt = ds_t2m.VAR_2T - ds_sst.SSTK

    # Constants
    Re = 6.371e6  # Earth radius in meters
    deg2rad = np.pi / 180
    phi = np.deg2rad(ds_sst.SSTK["latitude"])
    # meters per 1° at this latitude
    m_per_deg_lon = Re * np.cos(phi) * deg2rad
    m_per_deg_lat = Re * deg2rad

    if (
        dat_type == "dropsonde"
    ):  # Necessary to calculate delta x/y for dropsonde which only returns one column
        # --- SST gradients & Tadv (K/day) ---
        ds_sst2 = (
            ds_sst.sortby(["latitude", "longitude"])
            .unify_chunks()
            .chunk({"latitude": -1, "longitude": -1, "time": -1})
        )

        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst2.SSTK.differentiate("longitude") / m_per_deg_lon  # K/m
        dT_dy = ds_sst2.SSTK.differentiate("latitude") / m_per_deg_lat  # K/m

    elif dat_type == "aircraft":
        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst.SSTK.differentiate("longitude") / m_per_deg_lon  # K/m
        dT_dy = ds_sst.SSTK.differentiate("latitude") / m_per_deg_lat  # K/m

    # advection: K/s -> K/day
    Tadv = -(ds_u10["VAR_10U"] * dT_dx + ds_v10["VAR_10V"] * dT_dy) * 86400.0
    # Convert to K/day
    Tadv = Tadv.rename("Tadv")

    # Calculate EIS following Wood and Bretherton (2006, J. Climate)
    cp = 1004.0  # specific heat at constant pressure for dry air (J / kg / K)
    Rd = 287.0  # gas constant for dry air (J / kg / K)
    kappa = Rd / cp
    Lhvap = 2.5e6  # Latent heat of vaporization (J / kg)
    g = 9.81  # m/s^2
    cp = 1004  # J/K/kg
    Lv = 2.5e6  # J/kg

    Rv = 461  # J/K/kg;
    Ra = 287  # J/K/kg

    def get_qsat(T, p):
        Tcel = T - 273.15
        es = 6.11 * 10 ** (7.5 * Tcel / (Tcel + 273.15))
        return 0.622 * es / p

    # Calculate lower tropospheric stability (LTS)
    theta_700 = ds_t700.T * (1013.25 / 700) ** kappa
    LTS = theta_700 - theta_sfc

    # T850 = (ds_t2m.VAR_2T+ds_t700.T)/2
    T850 = ds_t850.T

    Gammam = (
        g
        / cp
        * (
            1.0
            - (1.0 + Lhvap * get_qsat(T850, 850) / Rd / T850)
            / (1.0 + Lhvap**2 * get_qsat(T850, 850) / cp / Rv / T850**2)
        )
    )

    # Assume exponential decrease of pressure with scale height given by surface temperature
    z700 = (Rd * ds_t2m.VAR_2T / g) * np.log(1000 / 700)
    # Assume 80% relative humidity to compute LCL, appropriate for marine boundary layer
    Tadj = Tadj = ds_t2m.VAR_2T - 55.0  # in Kelvin
    LCL = cp / g * (Tadj - (1 / Tadj - np.log(0.8) / 2840.0) ** (-1))
    EIS = LTS - Gammam * (z700 - LCL)

    # Convert w700 (m/s) to pa/s
    omega700 = -(70000 / (Rd * ds_t700.T)) * g * w_700

    # Merge the dataset variables used later
    ds = {
        "deltaT": dt,
        "Tadv": Tadv,
        "M": M,
        "omega700": omega700,
        "SST": ds_sst.SSTK,
        "WS": ws,
        "Wind_shear": wind_shear,
        "RH700": RH,
        "EIS": EIS,
    }

    return ds


# --- process_data_products_utils.py lines 743-745 ---
def wrap180(lon):
    # Map any longitude to [-180, 180)
    return (lon + 180.0) % 360.0 - 180.0


# --- process_data_products_utils.py lines 747-754 ---
def nearest_time_indices(era5_times_ns, flight_times_ns):
    # era5_times_ns: 1D int64 nanoseconds, sorted
    # flight_times_ns: 1D int64 nanoseconds
    idx_right = np.searchsorted(era5_times_ns, flight_times_ns, side="left")
    idx_left = np.clip(idx_right - 1, 0, len(era5_times_ns) - 1)
    idx_right = np.clip(idx_right, 0, len(era5_times_ns) - 1)
    choose_right = np.abs(era5_times_ns[idx_right] - flight_times_ns) < np.abs(
        era5_times_ns[idx_left] - flight_times_ns
    )
    return np.where(choose_right, idx_right, idx_left)


# --- process_data_products_utils.py lines 1208-1339 ---
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
    lat_vals = ds["latitude"].values.astype(float)
    lon_vals_ds = ds["longitude"].values.astype(float)

    # Build KDTree in [-180,180) to avoid wrap issues
    lon_vals_wrapped = wrap180(lon_vals_ds)
    lon_grid, lat_grid = np.meshgrid(
        lon_vals_wrapped, lat_vals, indexing="xy"
    )  # (nx,ny) if xy; use ij below
    # use ij orientation for unravel consistency:
    YY, XX = np.meshgrid(lat_vals, lon_vals_wrapped, indexing="ij")  # (ny,nx)
    tree = cKDTree(np.c_[YY.ravel(), XX.ravel()])
    ny, nx = YY.shape  # (lat, lon)

    # ERA5 time (sorted)
    t_era = ds["time"].values.astype("datetime64[ns]")
    t_era_ns = t_era.view("int64")

    # Preload arrays into NumPy (T,Y,X) for fast indexing
    def arr3(name):
        return ds[name].transpose("time", "latitude", "longitude").compute().values

    arr = {
        "ERA5_SST": arr3("SST"),
        "M": arr3("M"),
        "Omega700": arr3("omega700"),
        "deltaT": arr3("deltaT"),
        "Wind_sp": arr3("WS"),
        "Wind_shear": arr3("Wind_shear"),
        "Tadv": arr3("Tadv"),
        "RH700": arr3("RH700"),
        "EIS": arr3("EIS"),
    }

    # Flight coords/time
    flt_lat = df["GGLAT"].to_numpy(float)
    flt_lon = wrap180(df["GGLON"].to_numpy(float))  # match KDTree frame
    flt_t = pd.to_datetime(df["Time"].values).to_numpy("datetime64[ns]")
    flt_t_ns = flt_t.view("int64")

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
        yi, xi = np.unravel_index(flat_idx, (ny, nx))  # (M,), (M,)

        # Gather each variable and scatter back into full-length columns (NaN elsewhere)
        for out_name, A in arr.items():  # A: (T,ny,nx)
            vals_valid = A[ti_valid, yi, xi]  # (M,)
            out_full = np.full(N, np.nan, dtype=float)  # default NaN
            out_full[valid] = vals_valid
            df[out_name] = out_full

    return df


# --- process_data_products_utils.py lines 1341-1445 ---
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
    M_mean = mean_if_enough(df.get("M"))
    RH700_mean = mean_if_enough(df.get("RH700"))
    SST_mean = mean_if_enough(df.get("ERA5_SST"))
    Tadv_mean = mean_if_enough(df.get("Tadv"))
    Wind_sp_mean = mean_if_enough(df.get("Wind_sp"))
    Wind_shear_mean = mean_if_enough(df.get("Wind_shear"))
    EIS_mean = mean_if_enough(df.get("EIS"))

    # default label
    label = "Unknown"

    # ===========================
    #     SOCRATES RULE SET
    # ===========================
    if campaign.upper() == "SOCRATES":

        # --- Open-cell cumulus ---
        cond_open = (
            # 1) M >= -7 & WS >= 9
            (
                np.isfinite(M_mean)
                and np.isfinite(Wind_sp_mean)
                and M_mean >= -7
                and Wind_sp_mean >= 9
            )
            or
            # 2) M >= -8 & EIS < 9
            (
                np.isfinite(M_mean)
                and np.isfinite(EIS_mean)
                and M_mean >= -8
                and EIS_mean < 9
            )
            or
            # 3) M >= -8 & wshear > 6
            (
                np.isfinite(M_mean)
                and np.isfinite(Wind_shear_mean)
                and M_mean >= -8
                and Wind_shear_mean > 6
            )
        )

        # --- Stratocumulus ---
        cond_strat = (
            # 1) M < -9 & WS < 9
            (
                np.isfinite(M_mean)
                and np.isfinite(Wind_sp_mean)
                and M_mean < -9
                and Wind_sp_mean < 9
            )
            or
            # 2) M < -10 & EIS > 7
            (
                np.isfinite(M_mean)
                and np.isfinite(EIS_mean)
                and M_mean < -10
                and EIS_mean > 7
            )
            or
            # 3) M < -10 & wshear < 9
            (
                np.isfinite(M_mean)
                and np.isfinite(Wind_shear_mean)
                and M_mean < -10
                and Wind_shear_mean < 9
            )
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
            np.isfinite(M_mean)
            and np.isfinite(SST_mean)
            and M_mean < -10
            and SST_mean < 295
        ) or (
            np.isfinite(M_mean)
            and np.isfinite(Tadv_mean)
            and M_mean < -10
            and Tadv_mean < 0
        )

        cond_opencu = (
            np.isfinite(M_mean)
            and np.isfinite(SST_mean)
            and M_mean >= -10
            and SST_mean >= 296
        ) or (np.isfinite(M_mean) and np.isfinite(Tadv_mean) and M_mean >= -4)

        if cond_strat:
            label = "Stratocumulus"
        if cond_opencu:
            label = "Open-Cell"

    # add a single label to the entire df
    df = df.copy()
    df["cloud_regime"] = label
    return df
