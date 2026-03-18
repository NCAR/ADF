from pathlib import Path
import os
import xarray as xr
import xesmf as xe
import numpy as np
from adf_base import AdfError


# =========================
# Helpers
# =========================

def check_unstructured(ds, case):
    """
    Check if a dataset is unstructured based on its dimensions.
    """
    if ('lat' not in ds.dims) and ('lon' not in ds.dims):
        if ('ncol' in ds.dims) or ('lndgrid' in ds.dims):
            print(f"Looks like the case '{case}' is unstructured")
            return True
    return False


def save_to_nc(ds, outname):
    enc = {v: {'_FillValue': None} for v in ds.data_vars}
    enc.update({c: {'_FillValue': None} for c in ds.coords})
    ds.to_netcdf(outname, format='NETCDF4', encoding=enc)


def ensure_latlon(ds, src_grid_file):

    if "lat" in ds and "lon" in ds:
        return ds

    print("Adding lat/lon from grid file")

    grid = xr.open_dataset(src_grid_file)
    print("grid ncol:", grid.dims.get("ncol"))
    print("data ncol:", ds.dims.get("ncol"))

    return ds.assign_coords({
        "lat": ("ncol", grid["lat"].values),
        "lon": ("ncol", grid["lon"].values),
    })


# =========================
# Regridder
# =========================

import xarray as xr
import xesmf as xe
from pathlib import Path

def build_regridder(ds, latlon_file, method, weights_file=None):

    target = xr.open_dataset(latlon_file)

    ds_out = xr.Dataset({
        "lat": (["lat"], target["lat"].values),
        "lon": (["lon"], target["lon"].values),
    })

    # If weights exist, don't rebuild grid
    #if weights_file and Path(weights_file).exists():
    if 2==1:
        print(f"Using existing weights: {weights_file}")

        regridder = xe.Regridder(
            ds,
            ds_out,
            method,
            filename=weights_file,
            reuse_weights=True
        )

        return regridder


    # -----------------------------
    # OTHERWISE: build weights fresh
    # -----------------------------
    print("Creating new weights")

    is_unstructured = "ncol" in ds.dims

    if is_unstructured:
        if "lat" not in ds or "lon" not in ds:
            raise ValueError("SE grid requires lat/lon on ncol")

        ds_in = xr.Dataset({
            "lat": ("ncol", ds["lat"].values),
            "lon": ("ncol", ds["lon"].values),
        })
    else:
        ds_in = xr.Dataset({
            "lat": (["lat"], ds["lat"].values),
            "lon": (["lon"], ds["lon"].values),
        })

    regridder = xe.Regridder(
        ds_in,
        ds_out,
        method,
        filename=weights_file,
        reuse_weights=False,
        periodic=True
    )


    return regridder


# =========================
# Core regrid function
# =========================

def regrid_variable(ds, var, regridder, comp):

    da = ds[var]

    out = regridder(da)
    out.name = var

    dims_order = [d for d in ["time", "lev", "lat", "lon"] if d in out.dims]
    out = out.transpose(*dims_order)

    return out.to_dataset()


# =========================
# Area calculation
# =========================

def add_area(ds):
    lat = ds["lat"].values
    lon = ds["lon"].values

    R = 6.37122e3  # km

    dlat = np.gradient(lat)
    dlon = np.gradient(lon)

    lat_rad = np.deg2rad(lat)

    area = np.outer(
        R * np.deg2rad(dlat),
        R * np.cos(lat_rad)[:, None] * np.deg2rad(dlon)
    )

    ds["area"] = xr.DataArray(area, coords=[lat, lon], dims=["lat", "lon"])
    ds["area"].attrs.update({
        "units": "km2",
        "long_name": "Grid cell area"
    })

    return ds


# =========================
# Main driver
# =========================

def grid_timeseries(adfobj, **kwargs):

    ts_dir = Path(kwargs["ts_dir"])
    method = kwargs["method"]
    weight_file = kwargs["wgts_file"]
    latlon_file = kwargs["latlon_file"]
    comp = kwargs["comp"]
    vars_list = kwargs["diag_var_list"]
    case_name = kwargs["case_name"]
    hist_str = kwargs["hist_str"]
    time_string = kwargs["time_string"]
    is_baseline = kwargs["is_baseline"]

    out_dir = ts_dir / "gridded"
    #out_dir.mkdir(parents=True, exist_ok=True)
    # Check that path actually exists:
    if not out_dir.is_dir():
        print(f"    {out_dir} not found, making new directory")
        out_dir.mkdir(parents=True)


    #Check if any a weights file exists if using native grid, OPTIONAL
    if not latlon_file:
        raise AdfError("Missing lat/lon target grid file")
    

    for var in vars_list:

        print(f"\n--- Regridding {var} ---")

        ts_files = (
            adfobj.data.get_ref_timeseries_file(var)
            if is_baseline
            else adfobj.data.get_timeseries_file(case_name, var)
        )

        if not ts_files:
            print(f"Skipping {var}: no files")
            continue

        out_file = out_dir / f"{case_name}.{hist_str}.{var}.{time_string}_gridded.nc"

        if out_file.exists():
            print(f"Skipping {var}: already exists")
            #if overwrite_ts[case_idx]:
            if 2==1:
                Path(out_file).unlink()
            else:
                #msg = f"[{__name__}] Warning: '{var}' file was found "
                msg = f"\t    INFO: '{var}' gridded file was found "
                msg += "and overwrite is False. Will use existing file."
                print(msg)
                continue
            continue

        ds = adfobj.data.load_timeseries_dataset(ts_files)
        if ds is None:
            print(f"    No time series data set for variable '{var}' in case '{case_name}', skipping gridding for this variable.")
            continue
        print("ds",ds.attrs,"\n\n")
        src_grid_file = ds.attrs['initial_file']
        ds = ensure_latlon(ds, src_grid_file)

        regridder = build_regridder(
            ds,
            latlon_file,
            method,
            weights_file=weight_file
        )

        original_time = ds.time.values

        # ---- REGRID ----
        rg = regrid_variable(ds, var, regridder, comp)

        # ---- POSTPROCESS ----
        rg = rg.assign_coords(time=original_time)
        rg.attrs = ds.attrs
        rg.attrs["native_grid_to_latlon"] = f"xESMF ({method})"

        rg = add_area(rg)

        # ---- SAVE ----
        save_to_nc(rg, out_file)

        print(f"Saved: {out_file}")




# Gridding Unstructured to Lat/Lon
# Regrids unstructured SE grid to regular lat-lon
# Shamelessly borrowed from @maritsandstad with NorESM who deserves credit for this work
# https://github.com/NorESMhub/xesmf_clm_fates_diagnostic/blob/main/src/xesmf_clm_fates_diagnostic/plotting_methods.py

import xarray as xr
import xesmf
import numpy as np

def make_se_regridder(weight_file, s_data, d_data,
                      var,
                      Method='coservative',
                      ):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    # Hard code masks for now, not sure this does anything?
    if isinstance(s_data, xr.DataArray):
        s_mask = xr.DataArray(s_data.data.reshape(in_shape[0],in_shape[1]), dims=("lat", "lon"))
        dummy_in['mask']= s_mask
    if isinstance(d_data, xr.DataArray):
        d_mask = xr.DataArray(d_data.values, dims=("lat", "lon"))  
        dummy_out['mask']= d_mask
    #print("VAR:",var)            
    #print("---------------\ndummy_in",dummy_in,"\n\n")
    #print("dummy_out",dummy_out,"\n\n")


    # do source and destination grids need masks here?
    # See xesmf docs https://xesmf.readthedocs.io/en/stable/notebooks/Masking.html#Regridding-with-a-mask
    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        # results seem insensitive to this method choice
        # choices are coservative_normed, coservative, and bilinear
        method=Method,
        reuse_weights=True,
        periodic=True,
    )
    return regridder

def regrid_se_data_bilinear(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}),
                         skipna=True, na_thres=1,
                         )
    return regridded

def regrid_se_data_conservative(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}) )
    return regridded

"""def regrid_se_data_conservative(regridder, data_to_regrid, comp_grid):
    dims = data_to_regrid.dims
    #if data_to_regrid.ndim == 1:
    if len(data_to_regrid.dims) == 2:
        # (ncol,) → (1, ncol)
        updated = data_to_regrid.expand_dims("lat", axis=0)
        regridded = regridder(updated.rename({comp_grid: "lon"}))
        return regridded.squeeze("lat")

    elif len(data_to_regrid.dims) == 3:
        # (other, ncol) → (other, lat, lon)
        updated = data_to_regrid.expand_dims("lat", axis=-2)
        regridded = regridder(updated.rename({"lat": "lat", comp_grid: "lon"}))
        return regridded

    elif len(data_to_regrid.dims) == 4:
        # Assume (time, lev, ncol)
        stacked = data_to_regrid.stack(stack_dim=("time", "lev", "ilev"))
        updated = stacked.expand_dims("lat", axis=-2)
        regridded = regridder(updated.rename({"lat": "lat", comp_grid: "lon"}))
        unstacked = regridded.unstack("stack_dim")
        return unstacked.transpose("time", "lev", "ilev", "lat", "lon")

    else:
        raise ValueError(f"Unhandled data shape or dimensions: {data_to_regrid.shape} {dims}")"""




def regrid_atm_se_data_bilinear(regridder, data_to_regrid, comp_grid='ncol'):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [name for name in data_to_regrid.variables if comp_grid in data_to_regrid[name].dims]
        updated = data_to_regrid.copy().update(data_to_regrid[vars_with_ncol].transpose(..., comp_grid).expand_dims("dummy", axis=-2))
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(...,comp_grid).expand_dims("dummy",axis=-2)
    else:
        raise ValueError(f"Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}")
    regridded = regridder(updated)
    return regridded


def regrid_atm_se_data_conservative(regridder, data_to_regrid, comp_grid='ncol'):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [name for name in data_to_regrid.variables if comp_grid in data_to_regrid[name].dims]
        updated = data_to_regrid.copy().update(data_to_regrid[vars_with_ncol].transpose(..., comp_grid).expand_dims("dummy", axis=-2))
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(...,comp_grid).expand_dims("dummy",axis=-2)
    else:
        raise ValueError(f"Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}")
    regridded = regridder(updated,skipna=True, na_thres=1)
    return regridded



"""
def regrid_lnd_se_data_bilinear(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}),
                         skipna=True, na_thres=1,
                         )
    return regridded


def regrid_lnd_se_data_conservative(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}) )
    return regridded"""



def regrid(model_dataset, var_name, comp, wgt_file, method, latlon_file, **kwargs):

    """
    Function that takes a variable from a model xarray
    dataset, regrids it to another dataset's lat/lon
    coordinates (if applicable)
    ----------
    model_dataset -> The xarray dataset which contains the model variable data
    var_name      -> The name of the variable to be regridded/interpolated.
    comp          ->
    wgt_file      ->
    method        ->
    latlon_file   ->
    
    Optional inputs:

    kwargs         -> Keyword arguments that contain paths to THE REST IS NOT APPLICABLE: surface pressure
                      and mid-level pressure files, which are necessary for
                      certain types of vertical interpolation.
    This function returns a new xarray dataset that contains the gridded
    model variable.
    """

    if comp == "atm":
        comp_grid = "ncol"
    if comp == "lnd":
        comp_grid = "lndgrid"
    if latlon_file:
        latlon_ds = xr.open_dataset(latlon_file)
    else:
        print("Looks like no lat lon file is supplied. God speed!")

    model_dataset[var_name] = model_dataset[var_name].fillna(0)

    if comp == "lnd":
        model_dataset['landfrac'] = model_dataset['landfrac'].fillna(0)
        #mdata = mdata * model_dataset.landfrac  # weight flux by land frac
        model_dataset[var_name] = model_dataset[var_name] * model_dataset.landfrac  # weight flux by land frac
        s_data = model_dataset.landmask.isel(time=0)
        d_data = latlon_ds.landmask
    else:
        s_data = None #model_dataset[var_name].isel(time=0)
        d_data = None #latlon_ds.PSL.isel(time=0)

    #Grid model data to match target grid lat/lon:
    regridder = make_se_regridder(weight_file=wgt_file,
                                    s_data = s_data,
                                    d_data = d_data,
                                    Method = method,
                                    var=var_name
                                    )
 

    if method == 'coservative':
        rgdata = regrid_se_data_conservative(regridder, model_dataset, comp_grid)
    if method == 'bilinear':
        rgdata = regrid_se_data_bilinear(regridder, model_dataset, comp_grid)
    if comp == "lnd":
        rgdata[var_name] = (rgdata[var_name] / rgdata.landfrac)
        rgdata['landmask'] = latlon_ds.landmask
        rgdata['landfrac'] = rgdata.landfrac.isel(time=0)

    # calculate area
    rgdata = calc_area(rgdata)

    #Return dataset:
    return rgdata


def calc_area(rgdata):
    # calculate area
    area_km2 = np.zeros(shape=(len(rgdata['lat']), len(rgdata['lon'])))
    earth_radius_km = 6.37122e3  # in meters

    yres_degN = np.abs(np.diff(rgdata['lat'].data))  # distances between gridcell centers...
    xres_degE = np.abs(np.diff(rgdata['lon']))  # ...end up with one less element, so...
    yres_degN = np.append(yres_degN, yres_degN[-1])  # shift left (edges <-- centers); assume...
    xres_degE = np.append(xres_degE, xres_degE[-1])  # ...last 2 distances bet. edges are equal

    dy_km = yres_degN * earth_radius_km * np.pi / 180  # distance in m
    phi_rad = rgdata['lat'].data * np.pi / 180  # degrees to radians

    # grid cell area
    for j in range(len(rgdata['lat'])):
        for i in range(len(rgdata['lon'])):
            dx_km = xres_degE[i] * np.cos(phi_rad[j]) * earth_radius_km * np.pi / 180  # distance in m
            area_km2[j,i] = dy_km[j] * dx_km

    rgdata['area'] = xr.DataArray(area_km2,
                                    coords={'lat': rgdata.lat, 'lon': rgdata.lon},
                                    dims=["lat", "lon"])
    rgdata['area'].attrs['units'] = 'km2'
    rgdata['area'].attrs['long_name'] = 'Grid cell area'

    return rgdata