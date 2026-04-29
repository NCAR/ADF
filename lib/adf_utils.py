"""                                                                    .
Generic computation helper functions

Functions
---------
load_dataset()
    generalized load dataset method used for plotting/analysis functions
mask_land_or_ocean(arr, msk, use_nan=False)
    Apply a land or ocean mask to provided variable.
global_average(fld, wgt, verbose=False)
    pure numpy global average.
spatial_average(indata, weights=None, spatial_dims=None)
    Compute spatial average
wgt_rmse(fld1, fld2, wgt):
    Calculate the area-weighted RMSE.
annual_mean(data, whole_years=False, time_name='time'):
    Calculate annual averages from time series data.
seasonal_mean(data, season=None, is_climo=None):
    Calculates the time-weighted seasonal average (or average over all time).
domain_stats(data, domain):
    Provides statistics in specified region.
pres_from_hybrid(psfc, hya, hyb, p0=100000.):
    Converts a hybrid level to a pressure
vert_remap(x_mdl, p_mdl, plev)
    Interpolates to specified pressure levels.
lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None, convert_to_mb=False)
    Interpolate model hybrid levels to specified pressure levels.
pmid_to_plev(data, pmid, new_levels=None, convert_to_mb=False)
    Interpolate `data` from hybrid-sigma levels to isobaric levels using provided mid-level pressures.
zonal_mean_xr(fld)
    Average over all dimensions except `lev` and `lat`.
validate_dims(fld, list_of_dims)
    Checks if specified dimensions are in a DataArray
lat_lon_validate_dims(fld)
    Check if input field has lat and lon.
zm_validate_dims(fld)
    Check for dimensions for zonal average.

Notes
-----

"""

#import statements:
import numpy as np
import xarray as xr
import pandas as pd
import geocat.comp as gcomp

from adf_base import AdfError

import warnings  # use to warn user about missing files.

#Format warning messages:
def my_formatwarning(msg, *args, **kwargs):
    """Issue `msg` as warning."""
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]
            }


#################
#HELPER FUNCTIONS
#################

def load_dataset(fils):
    """
    This method exists to get an xarray Dataset from input file information that can be passed into the plotting methods.

    Parameters
    ----------
    fils : list
        strings or paths to input file(s)

    Returns
    -------
    xr.Dataset

    Notes
    -----
    When just one entry is provided, use `open_dataset`, otherwise `open_mfdatset`
    """
    if len(fils) == 0:
        warnings.warn(f"\t    WARNING: Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        return xr.open_dataset(fils[0])
    #End if
#End def


def mask_land_or_ocean(arr, msk, use_nan=False):
    """Apply a land or ocean mask to provided variable.

    Parameters
    ----------
    arr : xarray.DataArray
        the xarray variable to apply the mask to.
    msk : xarray.DataArray
        the xarray variable that contains the land or ocean mask,
        assumed to be the same shape as "arr".
    use_nan : bool, optional
        argument for whether to set the missing values
        to np.nan values instead of the defaul "-999." values.

    Returns
    -------
    arr : xarray.DataArray
        Same as input `arr` but masked as specified.
    """

    if use_nan:
        missing_value = np.nan
    else:
        missing_value = -999.
    #End if

    arr = xr.where(msk>=0.9,arr,missing_value)
    arr.attrs["missing_value"] = missing_value
    return(arr)



#######

def global_average(fld, wgt, verbose=False):
    """A simple, pure numpy global average.

    Parameters
    ----------
    fld : np.ndarray
        an input ndarray
    wgt : np.ndarray
        a 1-dimensional array of weights, should be same size as one dimension of `fld`
    verbose : bool, optional
        prints information when `True`

    Returns
    -------
    weighted average of `fld`
    """

    s = fld.shape
    for i in range(len(s)):
        if np.size(fld, i) == len(wgt):
            a = i
            break
    fld2 = np.ma.masked_invalid(fld)
    if verbose:
        print("(global_average)-- fraction of mask that is True: {}".format(np.count_nonzero(fld2.mask) / np.size(fld2)))
        print("(global_average)-- apply ma.average along axis = {} // validate: {}".format(a, fld2.shape))
    avg1, sofw = np.ma.average(fld2, axis=a, weights=wgt, returned=True) # sofw is sum of weights

    return np.ma.average(avg1)


def spatial_average(indata, weights=None, spatial_dims=None):
    """Compute spatial average.

    Parameters
    ----------
    indata : xr.DataArray
        input data
    weights : np.ndarray or xr.DataArray, optional
        the weights to apply, see Notes for default behavior
    spatial_dims : list, optional
        list of dimensions to average, see Notes for default behavior

    Returns
    -------
    xr.DataArray
        weighted average of `indata`

    Notes
    -----
    When `weights` is not provided, tries to find sensible values.
    If there is a 'lat' dimension, use `cos(lat)`.
    If there is a 'ncol' dimension, looks for `area` in `indata`.
    Otherwise, set to equal weights.

    Makes an attempt to identify the spatial variables when `spatial_dims` is None.
    Will average over `ncol` if present, and then will check for `lat` and `lon`.
    When none of those three are found, raise an AdfError.
    """
    import warnings

    if weights is None:
        #Calculate spatial weights:
        if 'lat' in indata.coords:
            weights = np.cos(np.deg2rad(indata.lat))
            weights.name = "weights"
        elif 'ncol' in indata.dims:
            if 'area' in indata:
                warnings.warn("area variable being used to generated normalized weights.")
                weights = indata['area'] / indata['area'].sum()
            else:
                warnings.warn("\t  We need a way to get area variable. Using equal weights.")
                weights = xr.DataArray(1.)
            weights.name = "weights"
        else:
            weights = xr.DataArray(1.)
            weights.name = "weights"
            warnings.warn("Un-recognized spatial dimensions: using equal weights for all grid points.")
        #End if
    #End if

    #Apply weights to input data:
    weighted = indata.weighted(weights)

    # we want to average over all non-time dimensions
    if spatial_dims is None:
        if 'ncol' in indata.dims:
            spatial_dims = ['ncol']
        else:
            spatial_dims = [dimname for dimname in indata.dims if (('lat' in dimname.lower()) or ('lon' in dimname.lower()))]

    if not spatial_dims:
        #Scripts using this function likely expect the horizontal dimensions
        #to be removed via the application of the mean. So in order to avoid
        #possibly unexpected behavior due to arrays being incorrectly dimensioned
        #(which could be difficult to debug) the ADF should die here:
        emsg = "spatial_average: No spatial dimensions were identified,"
        emsg += " so can not perform average."
        raise AdfError(emsg)

    return weighted.mean(dim=spatial_dims, keep_attrs=True)


def wgt_rmse(fld1, fld2, wgt):
    """Calculate the area-weighted RMSE.

    Parameters
    ----------
    fld1, fld2 : array-like
        2-dimensional spatial fields with the same shape.
        They can be xarray DataArray or numpy arrays.
    wgt : array-like
        the weight vector, expected to be 1-dimensional,
        matching length of one dimension of the data.

    Returns
    -------
    float
        root mean squared error

    Notes:
    ```rmse = sqrt( mean( (fld1 - fld2)**2 ) )```
    """
    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."
    assert fld1.shape == fld2.shape, "Input fields must have the same array shape."
    # in case these fields are in dask arrays, compute them now.
    if hasattr(fld1, "compute"):
        fld1 = fld1.compute()
    if hasattr(fld2, "compute"):
        fld2 = fld2.compute()
    if isinstance(fld1, xr.DataArray) and isinstance(fld2, xr.DataArray):
        return (np.sqrt(((fld1 - fld2)**2).weighted(wgt).mean())).values.item()
    else:
        check = [len(wgt) == s for s in fld1.shape]
        if ~np.any(check):
            raise IOError(f"Sorry, weight array has shape {wgt.shape} which is not compatible with data of shape {fld1.shape}")
        check = [len(wgt) != s for s in fld1.shape]
        dimsize = fld1.shape[np.argwhere(check).item()]  # want to get the dimension length for the dim that does not match the size of wgt
        warray = np.tile(wgt, (dimsize, 1)).transpose()   # May need more logic to ensure shape is correct.
        warray = warray / np.sum(warray) # normalize
        wmse = np.sum(warray * (fld1 - fld2)**2)
        return np.sqrt( wmse ).item()


#######
# Time-weighted averaging

def annual_mean(data, whole_years=False, time_name='time'):
    """Calculate annual averages from monthly time series data.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        monthly data values with temporal dimension
    whole_years : bool, optional
        whether to restrict endpoints of the average to
        start at first January and end at last December
    time_name : str, optional
        name of the time dimension, defaults to `time`

    Returns
    -------
    result : xr.DataArray or xr.Dataset
        `data` reduced to annual averages

    Notes
    -----
    This function assumes monthly data, and weights the average by the
    number of days in each month.

    `result` includes an attribute that reports the date range used for the average.
    """
    assert time_name in data.coords, f"Did not find the expected time coordinate '{time_name}' in the data"
    if whole_years:
        first_january = np.argwhere((data.time.dt.month == 1).values)[0].item()
        last_december = np.argwhere((data.time.dt.month == 12).values)[-1].item()
        data_to_avg = data.isel(time=slice(first_january,last_december+1)) # PLUS 1 BECAUSE SLICE DOES NOT INCLUDE END POINT
    else:
        data_to_avg = data
    date_range_string = f"{data_to_avg['time'][0]} -- {data_to_avg['time'][-1]}"

    # this provides the normalized monthly weights in each year
    # -- do it for each year to allow for non-standard calendars (360-day)
    # -- and also to provision for data with leap years
    days_gb = data_to_avg.time.dt.daysinmonth.groupby('time.year').map(lambda x: x / x.sum())
    # weighted average with normalized weights: <x> = SUM x_i * w_i  (implied division by SUM w_i)
    result =  (data_to_avg * days_gb).groupby('time.year').sum(dim='time')
    result.attrs['averaging_period'] = date_range_string
    result.attrs['units'] = data.attrs.get("units",None)
    return result


def seasonal_mean(data, season=None, is_climo=None):
    """Calculates the time-weighted seasonal average (or average over all time).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        data to be averaged
    season : str, optional
        the season to extract from `data`
        If season is `ANN` or None, average all available time.
    is_climo : bool, optional
        If True, expects data to have time or month dimenion of size 12.
        If False, then 'time' must be a coordinate,
        and the `time.dt.days_in_month` attribute must be available.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        the average of `data` in season `season`

    Notes
    -----
    If the data is a climatology, the code will make an attempt to understand the time or month
    dimension, but will assume that it is ordered from January to December.
    If the data is a climatology and is just a numpy array with one dimension that is size 12,
    it will assume that dimension is time running from January to December.
    """
    if season is not None:
        assert season in ["ANN", "DJF", "JJA", "MAM", "SON"], f"Unrecognized season string provided: '{season}'"
    elif season is None:
        season = "ANN"

    unstruct = False
    if hasattr(data, "uxgrid") and data.uxgrid is not None:
        unstruct = True
        uxgrid=data.uxgrid

    try:
        month_length = data.time.dt.days_in_month
    except (AttributeError, TypeError):
        # do our best to determine the temporal dimension and assign weights
        if not is_climo:
            raise ValueError("Non-climo file provided, but without a decoded time dimension.")
        else:
            # CLIMO file: try to determine which dimension is month
            has_time = False
            if isinstance(data, xr.DataArray):
                has_time = 'time' in data.dims
                if not has_time:
                    if "month" in data.dims:
                        data = data.rename({"month":"time"})
                        has_time = True
            if not has_time:
                # this might happen if a pure numpy array gets passed in
                # --> assumes ordered January to December.
                assert ((12 in data.shape) and (data.shape.count(12) == 1)), f"Sorry, {data.shape.count(12)} dimensions have size 12, making determination of which dimension is month ambiguous. Please provide a `time` or `month` dimension."
                time_dim_num = data.shape.index(12)
                fakedims = [f"dim{n}" for n in range(len(data.shape))]
                fakedims[time_dim_num] = "time"
                data = xr.DataArray(data, dims=fakedims, attrs=data.attrs)
            timefix = pd.date_range(start='1/1/1999', end='12/1/1999', freq='MS') # generic time coordinate from a non-leap-year
            data = data.assign_coords({"time":timefix})
        month_length = data.time.dt.days_in_month
    #End try/except

    data = data.sel(time=data.time.dt.month.isin(seasons[season])) # directly take the months we want based on season kwarg
    weighted_mean = data.weighted(data.time.dt.daysinmonth).mean(dim='time', keep_attrs=True)

    # Only wrap if the input had uxgrid
    if unstruct:
        weighted_mean = ux.UxDataArray(
            weighted_mean,
            uxgrid=uxgrid,
            attrs=weighted_mean.attrs
        )
    return weighted_mean #data.weighted(data.time.dt.daysinmonth).mean(dim='time', keep_attrs=True)

import numpy as np

def array_diff(a, b, percent=False, fill_nan=None):
    """
    Compute difference or percent difference between two data arrays
    
    Both unstructured UxDataArrays while preserving uxgrid and structure and xarray DataArrays

    Parameters
    ----------
    a, b : UxDataArray/DataArray
        Input arrays (must have same shape/grid)
    percent : bool, optional
        If True, compute percent difference
    fill_nan : float or None
        If set, replace NaNs with this value

    Returns
    -------
    UxDataArray
        Result with same uxgrid as input
    or
    DataArray
    """

    # --- sanity checks ---
    if a.shape != b.shape:
        raise ValueError("Input arrays must have same shape")

    if hasattr(a, "uxgrid") and a.uxgrid is None:
        raise ValueError("Input appears to be on an unstructured grid but is missing the uxgrid coordinate")

    # --- compute values ---
    if percent:
        if hasattr(a, "uxgrid"):
            with np.errstate(divide='ignore', invalid='ignore'):
                vals = (a.values - b.values) / np.abs(b.values) * 100.0
        else:
            vals = (a.values - b.values) / np.abs(b.values) * 100.0
    else:
        vals = a.values - b.values

    # --- handle NaNs ---
    if fill_nan is not None:
        vals = np.where(np.isfinite(vals), vals, fill_nan)

    # --- copy structure ---
    out = a.copy(deep=True)
    out.values = vals

    # --- optional: update attrs ---
    if percent:
        out.attrs = dict(a.attrs)
        out.attrs["long_name"] = f"Percent difference ({a.name})"
        out.attrs["units"] = "%"

    return out


#######


def domain_stats(data, domain):
    """Provides statistics in specified region.

    Parameters
    ----------
    data : xarray.DataArray
        data values
    domain : list or tuple or numpy.ndarray
        the domain specification as:
        [west_longitude, east_longitude, south_latitude, north_latitude]

    Returns
    -------
    x_region_mean : float
        the regional area-weighted average
    x_region_max : float
        the maximum value in the region
    x_region_min : float
        the minimum value in the region

    Notes
    -----
    Currently assumes 'lat' is a dimension and uses `cos(lat)` as weight.
    Should use `spatial_average`

    See Also
    --------
    spatial_average

    """
    x_region = data.sel(lat=slice(domain[2],domain[3]), lon=slice(domain[0],domain[1]))
    x_region_mean = x_region.weighted(np.cos(np.deg2rad(x_region['lat']))).mean().item()
    x_region_min = x_region.min().item()
    x_region_max = x_region.max().item()
    return x_region_mean, x_region_max, x_region_min




#
#  -- vertical interpolation code --
#

def pres_from_hybrid(psfc, hya, hyb, p0=100000.):
    """Calculates pressure field

    pressure derived with the formula:
    ```p = a(k)*p0 + b(k)*ps```

    Parameters
    ----------
    psfc
        surface pressure
    hya, hyb
        hybrid-sigma A and B coefficients
    p0 : optional
        reference pressure, defaults to 100000 Pa

    Returns
    -------
    pressure, size is same as `psfc` with `len(hya)` levels
    """
    return hya*p0 + hyb*psfc

#####

def vert_remap(x_mdl, p_mdl, plev):
    """Apply simple 1-d interpolation to a field

    Parameters
    ----------
    x_mdl : xarray.DataArray or numpy.ndarray
        input data
    p_mdl : xarray.DataArray or numpy.ndarray
        pressure field, same shape as `x_mdl`
    plev : xarray.DataArray or numpy.ndarray
        the new pressures

    Returns
    -------
    output
        `x_mdl` interpolated to `plev`

    Notes
    -----
    Interpolation done in log pressure
    """

    #Determine array shape of output array:
    out_shape = (plev.shape[0], x_mdl.shape[1])

    #Initialize interpolated output numpy array:
    output = np.full(out_shape, np.nan)

    #Perform 1-D interpolation in log-space:
    for i in range(out_shape[1]):
        output[:,i] = np.interp(np.log(plev), np.log(p_mdl[:,i]), x_mdl[:,i])
    #End for

    #Return interpolated output:
    return output

#####

def lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None,
                convert_to_mb=False):
    """Interpolate model hybrid levels to specified pressure levels.

    Parameters
    ----------
    data :
    ps :
        surface pressure
    hyam, hybm :
        hybrid-sigma A and B coefficients
    P0 : float, optional
        reference pressure, defaults to 100000 Pa
    new_levels : numpy.ndarray, optional
        1-D array containing pressure levels in Pascals (Pa).
        If not specified, then the levels will be set
        to the GeoCAT defaults, which are (in hPa):
        `1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50,
        30, 20, 10, 7, 5, 3, 2, 1`
    convert_to_mb : bool, optional
        If True, then vertical (lev) dimension will have
        values of mb/hPa, otherwise the units are Pa.

    Returns
    -------
    data_interp_rename
        data interpolated to new pressure levels

    Notes
    -----
    The function `interp_hybrid_to_pressure` used here is dask-enabled,
    and so can potentially be sped-up via the use of a DASK cluster.
    """

    #Temporary print statement to notify users to ignore warning messages.
    #This should be replaced by a debug-log stdout filter at some point:
    print("Please ignore the interpolation warnings that follow!")

    #Apply GeoCAT hybrid->pressure interpolation:
    if new_levels is not None:
        data_interp = gcomp.interpolation.interp_hybrid_to_pressure(data, ps,
                                                                    hyam,
                                                                    hybm,
                                                                    p0=P0,
                                                                    new_levels=new_levels
                                                                   )
    else:
        data_interp = gcomp.interpolation.interp_hybrid_to_pressure(data, ps,
                                                                    hyam,
                                                                    hybm,
                                                                    p0=P0
                                                                   )

    # data_interp may contain a dask array, which can cause
    # trouble downstream with numpy functions, so call compute() here.
    if hasattr(data_interp, "compute"):
        data_interp = data_interp.compute()

    #Rename vertical dimension back to "lev" in order to work with
    #the ADF plotting functions:
    data_interp_rename = data_interp.rename({"plev": "lev"})

    #Convert vertical dimension to mb/hPa, if requested:
    if convert_to_mb:
        data_interp_rename["lev"] = data_interp_rename["lev"] / 100.0

    return data_interp_rename

#####

def pmid_to_plev(data, pmid, new_levels=None, convert_to_mb=False):
    """Interpolate data from hybrid-sigma levels to isobaric levels.

    Parameters
    ----------
    data : xarray.DataArray
        field with a 'lev' coordinate
    pmid : xarray.DataArray
        the pressure field (Pa), same shape as `data`
    new_levels : optional
        the output pressure levels (Pa), defaults to standard levels
    convert_to_mb : bool, optional
        flag to convert output to mb (i.e., hPa), defaults to False

    Returns
    -------
    output : xarray.DataArray
        `data` interpolated onto `new_levels`
    """

    # determine pressure levels to interpolate to:
    if new_levels is None:
        pnew = 100.0 * np.array([1000, 925, 850, 700, 500, 400,
                                 300, 250, 200, 150, 100, 70, 50,
                                 30, 20, 10, 7, 5, 3, 2, 1])  # mandatory levels, converted to Pa
    else:
        pnew = new_levels
    #End if

    # save name of DataArray:
    data_name = data.name

    # reshape data and pressure assuming "lev" is the name of the coordinate
    zdims = [i for i in data.dims if i != 'lev']
    dstack = data.stack(z=zdims)
    pstack = pmid.stack(z=zdims)
    output = vert_remap(dstack.values, pstack.values, pnew)
    output = xr.DataArray(output, name=data_name, dims=("lev", "z"),
                          coords={"lev":pnew, "z":pstack['z']})
    output = output.unstack()

    # convert vertical dimension to mb/hPa, if requested:
    if convert_to_mb:
        output["lev"] = output["lev"] / 100.0
    #End if

    #Return interpolated output:
    return output




def validate_dims(fld, list_of_dims):
    """Check if specified dimensions are in a DataArray.

    Parameters
    ----------
    fld : xarray.DataArray
        field to check for named dimensions
    list_of_dims : list
        list of strings that specifiy the dimensions to check for

    Returns
    -------
    dict
        dict with keys that are "has_{x}" where x is the name from
        `list_of_dims` and values that are boolean

    """
    if not isinstance(list_of_dims, list):
        list_of_dims = list(list_of_dims)
    return { "_".join(["has",f"{v}"]):(v in fld.dims) for v in list_of_dims}


def lat_lon_validate_dims(fld):
    """Check if input field has lat and lon.

    Parameters
    ----------
    fld : xarray.DataArray
        data with named dimensions

    Returns
    -------
    bool
        True if lat and lon are both dimensions, False otherwise.

    See Also
    --------
    validate_dims
    """
    # note: we can only handle variables that reduce to (lat,lon)
    if len(fld.dims) > 3:
        return False
    validate = validate_dims(fld, ['lat','lon'])
    if not all(validate.values()):
        return  False
    else:
        return True


def zm_validate_dims(fld):
    """Check for dimensions for zonal average.

    Looks for dimensions called 'lev' and 'lat'.


    Parameters
    ----------
    fld : xarray.DataArray
        field to check for lat and/or lev dimensions
    Returns
    -------
    tuple
        (has_lat, has_lev) each are bool
    None
        If 'lat' is not in dimensions, returns None.
    """
    # note: we can only handle variables that reduce to (lev, lat) or (lat,)
    if len(fld.dims) > 4:
        print(f"Sorry, too many dimensions: {fld.dims}")
        return None
    validate = validate_dims(fld, ['lev','lat'])
    has_lev, has_lat = validate['has_lev'], validate['has_lat']
    return has_lat, has_lev


def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ('lev','lat')]
    else:
        if 1==1:
            print()
        else:
            print()
        #raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)

#####################
#END HELPER FUNCTIONS




from pathlib import Path
import os
import xarray as xr
import xesmf as xe
import numpy as np
from adf_base import AdfError


# =========================
# Helpers
# =========================

def check_unstructured(ds, case, ts_dir):
    """
    Check if a dataset is unstructured based on its dimensions.
    """
    if ('lat' not in ds.dims) and ('lon' not in ds.dims):
        if ('ncol' in ds.dims) or ('lndgrid' in ds.dims):
            print(f"\t    INFO: Looks like case '{case}' is unstructured, eh? -> {ts_dir}")
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



def grid_to_latlon(model_dataset, model_da, var_name, comp, wgt_file, method, latlon_file, **kwargs):

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

    #model_dataset[var_name] = model_dataset[var_name].fillna(0)
    model_da = model_da.fillna(0)

    if comp == "lnd":
        model_dataset['landfrac'] = model_dataset['landfrac'].fillna(0)
        #mdata = mdata * model_dataset.landfrac  # weight flux by land frac
        #model_dataset[var_name] = model_dataset[var_name] * model_dataset.landfrac  # weight flux by land frac
        model_da = model_da * model_dataset.landfrac
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