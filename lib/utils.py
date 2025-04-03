# utils.py

def check_unstructured(ds, case):
    """
    Check if a dataset is unstructured based on its dimensions.
    """
    if ('lat' not in ds.dims) and ('lon' not in ds.dims):
        if ('ncol' in ds.dims) or ('lndgrid' in ds.dims):
            message = f"Looks like the case '{case}' is unstructured, eh!"
            print(message)
            return True
    return False


from pathlib import Path
import os
from adf_base import AdfError

def grid_timeseries(**kwargs):
    #regrd_ts_loc = Path(test_output_loc[case_idx])
    # Check if time series directory exists, and if not, then create it:
    # Use pathlib to create parent directories, if necessary.

    ts_dir = Path(kwargs["ts_dir"])
    method = kwargs["method"]
    weight_file = kwargs["wgts_file"]
    latlon_file = kwargs["latlon_file"]
    time_file = kwargs["time_file"]
    comp = kwargs["comp"]
    diag_var_list = kwargs["diag_var_list"]
    case_name = kwargs["case_name"]
    hist_str = kwargs["hist_str"]
    time_string = kwargs["time_string"]

    regrd_ts_loc = ts_dir / "regrid"
    Path(regrd_ts_loc).mkdir(parents=True, exist_ok=True)
    # Check that path actually exists:
    if not regrd_ts_loc.is_dir():
        print(f"    {regrd_ts_loc} not found, making new directory")
        regrd_ts_loc.mkdir(parents=True)

    #Check if any a weights file exists if using native grid, OPTIONAL
    if not latlon_file:
        msg = "WARNING: This looks like an unstructured case, but missing weights file, can't continue."
        raise AdfError(msg)

    for var in diag_var_list:
        print("VAR",var,"\n")
        ts_ds = xr.open_dataset(sorted(ts_dir.glob(f"*.{var}.*nc"))[0])

        # Store the original cftime time values
        #print("ts_ds['time']",ts_ds['time'],"\n\n")
        original_time = ts_ds['time'].values

        rgdata = unstructure_regrid(ts_ds, var, comp=comp,
                                    wgt_file=weight_file,
                                    latlon_file=latlon_file,
                                    time_file=time_file,
                                    method=method)
        # Copy global attributes
        rgdata.attrs = ts_ds.attrs.copy()
        attrs_dict = {
                                    #"adf_user": adf.user,
                                    #"climo_yrs": f"{case_name}: {syear}-{eyear}",
                                    #"climatology_files": climatology_files_str,
                                    "native_grid_to_latlon":f"xesmf Regridder; method: {method}"
                                }
        ts_outfil_str = (str(ts_dir)
                         + os.sep
                         + ".".join([case_name, hist_str, var, time_string, "nc"])
                         )
        regridded_file_loc = regrd_ts_loc / Path(ts_outfil_str).parts[-1].replace(".nc","_gridded.nc")
        #rgdata = rgdata.assign_attrs(attrs_dict)
        # Restore the original cftime time values
        rgdata = rgdata.assign_coords(time=('time', original_time))
        #print("regridded_file_loc",rgdata.time,"\n\n")
        save_to_nc(rgdata, regridded_file_loc)
        #self.adf.native_grid[f"{case_type_string}_native_grid"] = False

        #file_path = os.path.join(dir_path, file_name)
        #os.remove(ts_outfil_str)
        #print("ts_outfil_str before death: ",ts_outfil_str,"\n")
        #sorted(ts_dir.glob(f"*.{var}.*nc"))[0].unlink()




# Regrids unstructured SE grid to regular lat-lon
# Shamelessly borrowed from @maritsandstad with NorESM who deserves credit for this work
# https://github.com/NorESMhub/xesmf_clm_fates_diagnostic/blob/main/src/xesmf_clm_fates_diagnostic/plotting_methods.py

import xarray as xr
import xesmf
import numpy as np

def make_se_ts_regridder(weight_file,s_data,d_data,
                      Method='coservative'
                      ):
    # Intialize dict for xesmf.Regridder
    #regridder_kwargs = {}

    if weight_file:
        weights = xr.open_dataset(weight_file)
        #regridder_kwargs['weights'] = weights
    else:
        print("No weights file given!")
    #    regridder_kwargs['method'] = 'coservative'
    
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

    if isinstance(s_data, xr.DataArray):
        s_mask = xr.DataArray(s_data.data.reshape(in_shape[0],in_shape[1]), dims=("lat", "lon"))
        dummy_in['mask']= s_mask
    if isinstance(d_data, xr.DataArray):
        d_mask = xr.DataArray(d_data.values, dims=("lat", "lon"))  
        dummy_out['mask']= d_mask         

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

import xarray as xr
import xesmf
import numpy as np

#def unstructure_regrid(model_dataset, var_name, comp, weight_file, latlon_file, method):
#def unstructure_regrid(model_dataset, var_name, comp, wgt_file, method, latlon_file=None):
def  unstructure_regrid(model_dataset, var_name, comp, wgt_file, method, latlon_file, time_file, **kwargs):
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

    #Import ADF-specific functions:
    from regrid_se_to_fv import make_se_regridder, regrid_se_data_conservative, regrid_se_data_bilinear, regrid_atm_se_data_conservative, regrid_atm_se_data_bilinear

    if comp == "atm":
        comp_grid = "ncol"
    if comp == "lnd":
        comp_grid = "lndgrid"
    if latlon_file:
        latlon_ds = xr.open_dataset(latlon_file)
    else:
        print("Looks like no lat lon file is supplied. God speed!")

    model_dataset[var_name] = model_dataset[var_name].fillna(0)
    #mdata = model_dataset[var_name]

    if comp == "lnd":
        model_dataset['landfrac'] = model_dataset['landfrac'].fillna(0)
        #mdata = mdata * model_dataset.landfrac  # weight flux by land frac
        model_dataset[var_name] = model_dataset[var_name] * model_dataset.landfrac  # weight flux by land frac
        s_data = model_dataset.landmask#.isel(time=0)
        d_data = latlon_ds.landmask

        """# Combine dimensions from both datasets while keeping ds2 attributes
        d_data = xr.Dataset(
            coords={"lat": latlon_ds["lat"], "lon": latlon_ds["lon"], "time": time_file["time"]},
            attrs=latlon_ds.attrs  # Copy attributes from ds2
        )
        print("AHHHHHH",d_data,"\n\n")
        # Add the 'temperature' variable from ds2 to new_ds
        d_data["landmask"] = time_file["landmask"]
        print("AHHHHHH2",d_data,"\n\n")
        d_data = d_data.landmask"""
    else:
        s_data = None #mdata.isel(time=0)
        d_data = None #latlon_ds[var_name]
    print("AHHHHHH3",d_data,"\n\n")
    #Grid model data to match target grid lat/lon:
    regridder = make_se_ts_regridder(weight_file=wgt_file,
                                    s_data = s_data,
                                    d_data = d_data,
                                    Method = method,
                                    )

    if comp == "lnd":
        if method == 'coservative':
            rgdata = regrid_se_data_conservative(regridder, model_dataset, comp_grid)
        if method == 'bilinear':
            rgdata = regrid_se_data_bilinear(regridder, model_dataset, comp_grid)
        rgdata[var_name] = (rgdata[var_name] / rgdata.landfrac)

    if comp == "atm":
        if method == 'coservative':
            rgdata = regrid_atm_se_data_conservative(regridder, model_dataset, comp_grid)
        if method == 'bilinear':
            rgdata = regrid_atm_se_data_bilinear(regridder, model_dataset, comp_grid)


    #rgdata['lat'] = latlon_ds.lat #???
    if comp == "lnd":
        rgdata['landmask'] = latlon_ds.landmask
        rgdata['landfrac'] = rgdata.landfrac#.isel(time=0)

    """new_ds = xr.Dataset(
                        coords={"lat": ds1["lat"], "lon": ds1["lon"], "time": ds2["time"]},
                        attrs=ds2.attrs  # Copy attributes from ds2
                    )
    """
    # calculate area
    rgdata = _calc_area(rgdata)

    #Return dataset:
    return rgdata


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


def regrid_lnd_se_data_bilinear(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}),
                         skipna=True, na_thres=1,
                         )
    return regridded


def regrid_lnd_se_data_conservative(regridder, data_to_regrid, comp_grid):
    updated = data_to_regrid.copy().transpose(..., comp_grid).expand_dims("dummy", axis=-2)
    regridded = regridder(updated.rename({"dummy": "lat", comp_grid: "lon"}) )
    return regridded



def save_to_nc(tosave, outname, attrs=None, proc=None):
    """Saves xarray variable to new netCDF file"""

    xo = tosave  # used to have more stuff here.
    # deal with getting non-nan fill values.
    if isinstance(xo, xr.Dataset):
        enc_dv = {xname: {'_FillValue': None} for xname in xo.data_vars}
    else:
        enc_dv = {}
    #End if
    enc_c = {xname: {'_FillValue': None} for xname in xo.coords}
    enc = {**enc_c, **enc_dv}
    if attrs is not None:
        xo.attrs = attrs
    if proc is not None:
        xo.attrs['Processing_info'] = f"Start from file {origname}. " + proc
    xo.to_netcdf(outname, format='NETCDF4', encoding=enc)



def _calc_area(rgdata):
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
