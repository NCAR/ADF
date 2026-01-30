"""                                                                    .
Generic computation and plotting helper functions

Functions
---------
load_dataset()
    generalized load dataset method used for plotting/analysis functions
use_this_norm()
    switches matplotlib color normalization method
get_difference_colors(values)
    Provide a color norm and colormap assuming `values` is a difference field.
mask_land_or_ocean(arr, msk, use_nan=False)
    Apply a land or ocean mask to provided variable.
get_central_longitude(*args)
    Determine central longitude for maps.
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
make_polar_plot(wks, case_nickname, base_nickname,
                    case_climo_yrs, baseline_climo_yrs,
                    d1:xr.DataArray, d2:xr.DataArray, difference:Optional[xr.DataArray]=None,
                    domain:Optional[list]=None, hemisphere:Optional[str]=None, **kwargs):
    Make a stereographic polar plot for the given data and hemisphere.
plot_map_vect_and_save(wks, case_nickname, base_nickname,
                           case_climo_yrs, baseline_climo_yrs,
                           plev, umdlfld_nowrap, vmdlfld_nowrap,
                           uobsfld_nowrap, vobsfld_nowrap,
                           udiffld_nowrap, vdiffld_nowrap, **kwargs):
    Plots a vector field on a map.
plot_map_and_save(wks, case_nickname, base_nickname,
                      case_climo_yrs, baseline_climo_yrs,
                      mdlfld, obsfld, diffld, **kwargs):
    Map plots of `mdlfld`, `obsfld`, and their difference, `diffld`.
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
zonal_plot(lat, data, ax=None, color=None, **kwargs)
    Make a line plot or pressure-latitude plot of `data`.
meridional_plot(lon, data, ax=None, color=None, **kwargs)
    Make a line plot or pressure-longitude plot of `data`.
prep_contour_plot
    Preparation for making contour plots.
plot_zonal_mean_and_save
    zonal mean plot
plot_meridional_mean_and_save
    meridioanl mean plot
square_contour_difference
    Produce filled contours of fld1, fld2, and their difference with square axes.

Notes
-----
This module includes several "private" methods intended for internal use only.

_plot_line(axobject, xdata, ydata, color, **kwargs)
    Create a generic line plot
_meridional_plot_line

_zonal_plot_line

_zonal_plot_preslat

_meridional_plot_preslon

"""

#import statements:
from typing import Optional
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib as mpl
import cartopy.crs as ccrs
#nice formatting for tick labels
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import geocat.comp as gcomp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import uxarray as ux  #need npl 2024a or later

from adf_diag import AdfDiag
from adf_base import AdfError

import warnings  # use to warn user about missing files.

#Format warning messages:
def my_formatwarning(msg, *args, **kwargs):
    """Issue `msg` as warning."""
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

#Set non-X-window backend for matplotlib:
mpl.use('Agg')

#Now import pyplot:
import matplotlib.pyplot as plt

empty_message = "No Valid\nData Points"
props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}


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

def load_ux_dataset(fils, mesh=None):
    """
    This method exists to get an uxarray Dataset from input file information that can be passed into the plotting methods.

    Parameters
    ----------
    fils : list
        strings or paths to input file(s)

    Returns
    -------
    ux.UxDataArray

    Notes
    -----
    When just one entry is provided, use `open_dataset`, otherwise `open_mfdatset`
    """
    if mesh == None:
        mesh = '/glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc'
        warnings.warn(f"No mesh file provided, using defaults ne30pg3 mesh file")
        
    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return ux.open_mfdataset(mesh, fils)
    else:
        return ux.open_dataset(mesh, fils[0])
    #End if
#End def


def use_this_norm():
    """Just use the right normalization; avoids a deprecation warning."""

    mplversion = [int(x) for x in mpl.__version__.split('.')]
    if mplversion[0] < 3:
        return mpl.colors.Normalize, mplversion[0]
    else:
        if mplversion[1] < 2:
            return mpl.colors.DivergingNorm, mplversion[0]
        else:
            return mpl.colors.TwoSlopeNorm, mplversion[0]


def get_difference_colors(values):
    """Provide a color norm and colormap assuming this is a difference field.

    Parameters
    ----------
    values : array-like
        can be either the data field or a set of specified contour levels.

    Returns
    -------
    dnorm
        Matplotlib color nomalization
    cmap
        Matplotlib colormap

    Notes
    -----
    Uses 'OrRd' colormap for positive definite, 'BuPu_r' for negative definite,
    and 'RdBu_r' centered on zero if there are values of both signs.
    """
    normfunc, mplv = use_this_norm()
    dmin = np.min(values)
    dmax = np.max(values)
    # color normalization for difference
    if ((dmin < 0) and (0 < dmax)):
        dnorm = normfunc(vmin=np.min(values), vmax=np.max(values), vcenter=0.0)
        cmap = mpl.cm.RdBu_r
    else:
        dnorm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        if dmin >= 0:
            cmap = mpl.cm.OrRd
        elif dmax <= 0:
            cmap = mpl.cm.BuPu_r
        else:
            dnorm = mpl.colors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
    return dnorm, cmap


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


def get_central_longitude(*args):
    """Determine central longitude for maps.

    Allows an arbitrary number of arguments.
    If any of the arguments is an instance of `AdfDiag`, then check
    whether it has a `central_longitude` in `diag_basic_info`.
    _This case takes precedence._
    _Else_, if any of the arguments are scalars in [-180, 360],
    assumes the FIRST ONE is the central longitude.
    There are no other possible conditions, so if none of those are met,
    returns the default value of 180.

    Parameters
    ----------
    *args : tuple
        Any number of objects to check for `central_longitude`.
        After that, looks for the first number between -180 and 360 in the args.

    Notes
    -----
    This allows a script to, for example, allow a config file to specify, but also have a preference:
    `get_central_longitude(AdfObj, 30.0)`
    """
    chk_for_adf = [isinstance(arg, AdfDiag) for arg in args]
    # preference is to get value from AdfDiag:
    if any(chk_for_adf):
        for arg in args:
            if isinstance(arg, AdfDiag):
                result = arg.get_basic_info('central_longitude', required=False)
                if (isinstance(result, int) or isinstance(result, float)) and \
                   (result >= -180) and (result <= 360):
                    return result
                else:
                    #If result exists, then write info to debug log:
                    if result:
                        msg = f"central_lngitude of type '{type(result).__name__}'"
                        msg += f" and value '{result}', which is not a valid longitude"
                        msg += " for the ADF."
                        arg.debug_log(msg)
                    #End if

                    #There is only one ADF object per ADF run, so if its
                    #not present or configured correctly then no
                    #reason to keep looking:
                    break
                #End if
            #End if
        #End for
    #End if

    # 2nd pass through arguments, look for numbers:
    for arg in args:
        if (isinstance(arg, float) or isinstance(arg, int)) and ((arg >= -180) and (arg <= 360)):
            return arg
        #End if
    else:
        # this is the `else` on the for loop --> if non of the arguments meet the criteria, do this.
        print("No valid central longitude specified. Defaults to 180.")
        return 180
    #End if

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

# TODO, maybe just adapt the spatial average above?
# TODO, should there be some unit conversions for this defined in a variable dictionary?
def spatial_average_lnd(indata, weights, spatial_dims=None):
    """Compute spatial average.

    Parameters
    ----------
    indata : xr.DataArray
        input data
    weights xr.DataArray
        weights (area * landfrac)
    spatial_dims : list, optional
        list of dimensions to average, see Notes for default behavior

    Returns
    -------
    xr.DataArray
        weighted average of `indata`

    Notes
    -----
    weights are required
    
    Makes an attempt to identify the spatial variables when `spatial_dims` is None.
    Will average over `ncol` if present, and then will check for `lat` and `lon`.
    When none of those three are found, raise an AdfError.        
    """
    import warnings

    #Apply weights to input data:
    weighted = indata*weights

    # we want to average over all non-time dimensions
    if spatial_dims is None:
        if 'lndgrid' in indata.dims:
            spatial_dims = ['lndgrid']
        else:
            spatial_dims = [dimname for dimname in indata.dims if (('lat' in dimname.lower()) or 
                                                                   ('lon' in dimname.lower()))]
    if not spatial_dims:
        #Scripts using this function likely expect the horizontal dimensions
        #to be removed via the application of the mean. So in order to avoid
        #possibly unexpected behavior due to arrays being incorrectly dimensioned
        #(which could be difficult to debug) the ADF should die here:
        emsg = "spatial_average: No spatial dimensions were identified,"
        emsg += " so can not perform average."
        raise AdfError(emsg)

    
    return weighted.sum(dim=spatial_dims, keep_attrs=True)


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
    wgt.fillna(0)
    assert len(fld1.shape) <= 2,     "Input fields must have less than two dimensions."
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

def annual_mean(data, whole_years=False, time_name='time', use_ux=False):
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
    days_in_month = data_to_avg.time.dt.daysinmonth
    #print("days_in_month",days_in_month,'\n')
    if not use_ux:
        days_gb = data_to_avg.time.dt.daysinmonth.groupby('time.year').map(lambda x: x / x.sum())
    else:
        # Group by the 'year' dimension
        grouped_by_year = days_in_month.groupby('time.year')
        
        # Initialize a list to store the normalized days for each year
        normalized_days = []
        
        # Loop over each group and normalize the values (divide by the sum of the group)
        for i, (year, group) in enumerate(grouped_by_year):
            # Compute the sum of days in the month for the current year
            print(group)
            year_sum = group[12*i:12*i+12].sum()
            
            # Normalize the group by dividing each value by the sum of the group
            normalized_group = group[12*i:12*i+12] / year_sum
            
            # Append the normalized values to the list
            normalized_days.append(normalized_group)
        
        # Concatenate the normalized days back together (align them with the original data)
        days_gb = xr.concat(normalized_days, dim='time')
        
        # Alternatively, if you want to make sure the result has the same coordinates as the original
        days_gb = days_in_month.copy(data=np.concatenate([g.values for g in normalized_days]))
        days_gb.coords['time'] = days_in_month.coords['time']  # Reassign the correct time coordinates
    
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
            if isinstance(data, ux.UxDataset):
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
    return data.weighted(data.time.dt.daysinmonth).mean(dim='time', keep_attrs=True)



#######

#Polar Plot functions

def domain_stats(data, domain, unstructured=False):
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
    if not unstructured:
        x_region = data.sel(lat=slice(domain[2],domain[3]), lon=slice(domain[0],domain[1]))
        x_region_mean = x_region.weighted(np.cos(np.deg2rad(x_region['lat']))).mean().item()
    else:
        x_region = data
        x_region_mean = data.mean().item()
    x_region_min = x_region.min().item()
    x_region_max = x_region.max().item()
    return x_region_mean, x_region_max, x_region_min



def make_polar_plot(wks, case_nickname, base_nickname,
                    case_climo_yrs, baseline_climo_yrs,
                    d1, d2, difference=None,pctchange=None,
                    domain:Optional[list]=None, hemisphere:Optional[str]=None, obs=False, unstructured=False, **kwargs):

    """Make a stereographic polar plot for the given data and hemisphere.

    Parameters
    ----------
    wks : str or Path
        output file path
    case_nickname : str
        short case name for `d1`
    base_nickname : str
        short case name for `d2`
    case_climo_yrs : list
        years for case `d1`, used for annotation
    baseline_climo_yrs : list
        years for case `d2`, used for annotation
    d1, d2 : xr.DataArray
        input data, must contain dimensions `lat` and `lon`
    difference : xr.DataArray, optional
        data to use as the difference, otherwise `d2 - d1`
    pctchange : xr.DataArray, optional data to use as the percent change
    domain : list, optional
        the domain to plot, specified as
        [west_longitude, east_longitude, south_latitude, north_latitude]
        Defaults to pole to 45-degrees, all longitudes
    hemisphere : {'NH', 'SH'}, optional
        Hemsiphere to plot
    kwargs : dict, optional
        variable-dependent options for plots, See Notes.

    Notes
    -----
    - Uses contourf. No contour lines (yet).
    - kwargs is checked for:
        + `colormap`
        + `contour_levels`
        + `contour_levels_range`
        + `diff_contour_levels`
        + `diff_contour_range`
        + `diff_colormap`
        + `units`
    """
    if difference is None:
        dif = d2 - d1
    else:
        dif = difference
        
    if  pctchange is None:
        pct = (d2 - d1) / np.abs(d1) * 100.0
    else:
        pct = pctchange
        
    #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
    pct = pct.where(np.isfinite(pct), np.nan)
    pct = pct.fillna(0.0)

    if isinstance(pct, ux.UxDataArray):
        pct_grid = pct.uxgrid
        pct = ux.UxDataArray(pct,uxgrid=pct_grid)
    else:
        pct = pct

    if (hemisphere.upper() == "NH") or (hemisphere == "Arctic"):
        proj = ccrs.NorthPolarStereo()
    elif hemisphere.upper() == "SH":
        proj = ccrs.SouthPolarStereo()
    else:
        raise AdfError(f'[make_polar_plot] hemisphere not specified, must be NH or SH; hemisphere set as {hemisphere}')

    if domain is None:
        if hemisphere.upper() == "NH":
            domain = [-180, 180, 45, 90]
        if hemisphere == "Arctic":
            domain = [-180, 180, 50, 90]
        else:
            domain = [-180, 180, -90, -45]

    """# statistics for annotation (these are scalars):
    d1_region_mean, d1_region_max, d1_region_min = domain_stats(d1, domain)
    d2_region_mean, d2_region_max, d2_region_min = domain_stats(d2, domain)
    dif_region_mean, dif_region_max, dif_region_min = domain_stats(dif, domain)
    pct_region_mean, pct_region_max, pct_region_min = domain_stats(pct, domain)"""
    means = []
    mins = []
    maxs = []
    if not unstructured:
        # statistics for annotation (these are scalars):
        d1_region_mean, d1_region_max, d1_region_min = domain_stats(d1, domain)
        d2_region_mean, d2_region_max, d2_region_min = domain_stats(d2, domain)
        dif_region_mean, dif_region_max, dif_region_min = domain_stats(dif, domain)
        pct_region_mean, pct_region_max, pct_region_min = domain_stats(pct, domain)
        #downsize to the specified region; makes plotting/rendering/saving much faster
        d1 = d1.sel(lat=slice(domain[2],domain[3]))
        d2 = d2.sel(lat=slice(domain[2],domain[3]))
        dif = dif.sel(lat=slice(domain[2],domain[3]))
        pct = pct.sel(lat=slice(domain[2],domain[3]))

        # add cyclic point to the data for better-looking plot
        d1_cyclic, lon_cyclic = add_cyclic_point(d1, coord=d1.lon)
        d2_cyclic, _ = add_cyclic_point(d2, coord=d2.lon)  # since we can take difference, assume same longitude coord.
        dif_cyclic, _ = add_cyclic_point(dif, coord=dif.lon)
        pct_cyclic, _ = add_cyclic_point(pct, coord=pct.lon)
        #wrap_fields = (d1_cyclic, d2_cyclic, dif_cyclic, pct_cyclic)
        wrap_fields = (d1_cyclic, d2_cyclic, pct_cyclic, dif_cyclic)
        lons, lats = np.meshgrid(lon_cyclic, d1.lat)
    else:
        wgt = kwargs["wgt"]
        #wrap_fields = (d1, d2, dif, pct)
        wrap_fields = (d1, d2, pct, dif)
        area_avg = [global_average(x, wgt) for x in wrap_fields]

        d1_region_mean, d1_region_max, d1_region_min = domain_stats(d1, domain, unstructured)
        d2_region_mean, d2_region_max, d2_region_min = domain_stats(d2, domain, unstructured)
        dif_region_mean, dif_region_max, dif_region_min = domain_stats(dif, domain, unstructured)
        pct_region_mean, pct_region_max, pct_region_min = domain_stats(pct, domain, unstructured)


        # TODO Check this is correct, weighted rmse uses xarray weighted function
        #d_rmse = wgt_rmse(a, b, wgt)  
        d_rmse = (np.sqrt(((dif**2)*wgt).sum())).values.item()

    # -- deal with optional plotting arguments that might provide variable-dependent choices

    # determine levels & color normalization:
    minval    = np.min([np.min(d1), np.min(d2)])
    maxval    = np.max([np.max(d1), np.max(d2)])
    absmaxdif = np.max(np.abs(dif))
    absmaxpct = np.max(np.abs(pct))

    means.extend([d1_region_mean,d2_region_mean, pct_region_mean, dif_region_mean])
    mins.extend([d1_region_min,d2_region_min, pct_region_min, dif_region_min])
    maxs.extend([d1_region_max,d2_region_max, pct_region_max, dif_region_max])

    # -- end options

    #lons, lats = np.meshgrid(lon_cyclic, d1.lat)

    # controling DPI makes uxplots look better
    fig = plt.figure(figsize=(10,10), dpi=300)
    gs = mpl.gridspec.GridSpec(2, 4, wspace=0.6)

    ax1 = plt.subplot(gs[0, :2], projection=proj)
    ax2 = plt.subplot(gs[0, 2:], projection=proj)
    ax3 = plt.subplot(gs[1, :2], projection=proj)
    ax4 = plt.subplot(gs[1, 2:], projection=proj)
    axs = [ax1,ax2,ax3,ax4]

    #generate a dictionary of contour plot settings:
    cp_info = prep_contour_plot(d1, d2, pct, dif, **kwargs)

    imgs = []

    # Loop over data arrays to make plots
    for i, a in enumerate(wrap_fields):
        if i == len(wrap_fields)-1:
            levels = cp_info['levelsdiff']
            cmap = cp_info['cmapdiff']
            norm = cp_info['normdiff']
        elif i == len(wrap_fields)-2:
            levels = cp_info['levelspctdiff']
            cmap = cp_info['cmappct']
            norm = cp_info['pctnorm']
        else:
            levels = cp_info['levels1']
            cmap = cp_info['cmap1']
            norm = cp_info['norm1']
        if unstructured:
            #configure for polycollection plotting
            # TODO, would be nice to have levels set from the info, above
            # raster approach should be faster
            axs[i].set_global()
            raster = a.to_raster(ax=axs[i])
            img = axs[i].imshow(
                raster, cmap=cmap, origin="lower", extent=axs[i].get_xlim() + axs[i].get_ylim()
            )
            img.set_clim(vmin=levels[0],vmax=levels[-1])
            imgs.append(img)

        else:
            
            levs = np.unique(np.array(levels))
            if len(levs) < 2:
                imgs.append(axs[i].contourf(lons,lats,a,colors="w",transform=ccrs.PlateCarree(),transform_first=True))
                axs[i].text(0.4, 0.4, empty_message, transform=axs[i].transAxes, bbox=props)
            else:
                imgs.append(axs[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm,
                                            transform=ccrs.PlateCarree(), #transform_first=True,
                                            **cp_info['contourf_opt']))
            #End if
        #End if unstructured

        # Set stats for title
        stat_mean = f"Mean: {means[i]:5.2f}"
        stat_max =  f"Max: {maxs[i]:5.2f}"
        stat_min = f"Min: {mins[i]:5.2f}"
        stats = f"{stat_mean}\n{stat_max}\n{stat_min}"
        axs[i].set_title(stats, loc='right', fontsize=8)
    #End for
  
    #Set Main title for subplots:
    st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=18)
    st.set_y(0.95)

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    axs[0].set_title(case_title, loc='left', fontsize=8) #fontsize=tiFontSize

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
        axs[1].set_title(base_title, loc='left', fontsize=8) #fontsize=tiFontSize
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        axs[1].set_title(base_title, loc='left', fontsize=8)

    axs[2].set_title("Test % Diff Baseline", loc='left', fontsize=8,fontweight="bold")
    axs[3].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=8)

    if "units" in kwargs:
        axs[1].set_ylabel(kwargs["units"])
        axs[3].set_ylabel(kwargs["units"])
    else:
        axs[1].set_ylabel(f"{d1.units}")
        axs[3].set_ylabel(f"{d1.units}")

    for a in axs:
        a.coastlines()
        a.set_extent(domain, ccrs.PlateCarree())
        # __Follow the cartopy gallery example to make circular__:
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpl.path.Path(verts * radius + center)
        a.set_boundary(circle, transform=a.transAxes)
        a.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                    lw=1, color="gray",y_inline=True, 
                    xlocs=range(-180,180,90), ylocs=range(0,90,10))

    # __COLORBARS__
    cb_mean_ax = inset_axes(axs[1],
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=axs[1].transAxes,
                    borderpad=0,
                    )
    fig.colorbar(imgs[0], cax=cb_mean_ax)
    
    cb_pct_ax = inset_axes(axs[3],
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=axs[3].transAxes,
                    borderpad=0,
                    )  

    cb_diff_ax = inset_axes(axs[2],
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=axs[2].transAxes,
                    borderpad=0,
                    )      
                    
    fig.colorbar(imgs[3], cax=cb_pct_ax)
    
    fig.colorbar(imgs[2], cax=cb_diff_ax)

    # Save files
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    # Close figures to avoid memory issues:
    plt.close(fig)

#######


def plot_map_vect_and_save(wks, case_nickname, base_nickname,
                           case_climo_yrs, baseline_climo_yrs,
                           plev, umdlfld_nowrap, vmdlfld_nowrap,
                           uobsfld_nowrap, vobsfld_nowrap,
                           udiffld_nowrap, vdiffld_nowrap, obs=False, **kwargs):

    """This plots a vector plot.

    Vector fields constructed from x- and y-components (u, v).

    Parameters
    ----------
    wks : str or Path
        output file path
    case_nickname : str
        short name for case
    base_nickname : str
        short name for base case
    case_climo_yrs : list
        list of years in case climatology, used for annotation
    baseline_climo_yrs : list
        list of years in base case climatology, used for annotation
    plev : str or float or None
        if not None, label denoting the pressure level
    umdlfld_nowrap, vmdlfld_nowrap : xarray.DataArray
        input data for case, the x- and y- components of the vectors
    uobsfld_nowrap, vobsfld_nowrap : xarray.DataArray
        input data for base case, the x- and y- components of the vectors
    udiffld_nowrap, vdiffld_nowrap : xarray.DataArray
        input difference data, the x- and y- components of the vectors
    kwargs : dict, optional
        variable-specific options, See Notes

    Notes
    -----
    kwargs expected to be a variable-specific section,
    possibly provided by an ADF Variable Defaults YAML file.
    Currently it is inspected for:
    - `central_longitude`
    - `var_name`
    - `case_name`
    - `baseline`
    - `tiString`
    - `tiFontSize`
    - `units`

    _Note_ The title string constructed by kwargs appears to not be used.
    """

    # specify the central longitude for the plot:
    cent_long = kwargs.get('central_longitude', 180)

    # generate projection:
    proj = ccrs.PlateCarree(central_longitude=cent_long)
    lat = umdlfld_nowrap['lat']
    wgt = np.cos(np.radians(lat))

    # add cyclic longitude:
    umdlfld, lon = add_cyclic_point(umdlfld_nowrap, coord=umdlfld_nowrap['lon'])
    vmdlfld, _   = add_cyclic_point(vmdlfld_nowrap, coord=vmdlfld_nowrap['lon'])
    uobsfld, _   = add_cyclic_point(uobsfld_nowrap, coord=uobsfld_nowrap['lon'])
    vobsfld, _   = add_cyclic_point(vobsfld_nowrap, coord=vobsfld_nowrap['lon'])
    udiffld, _   = add_cyclic_point(udiffld_nowrap, coord=udiffld_nowrap['lon'])
    vdiffld, _   = add_cyclic_point(vdiffld_nowrap, coord=vdiffld_nowrap['lon'])

    # create mesh for plots:
    lons, lats = np.meshgrid(lon, lat)

    # create figure:
    fig = plt.figure(figsize=(14,10))

    # LAYOUT WITH GRIDSPEC
    gs = mpl.gridspec.GridSpec(3, 6, wspace=0.5, hspace=0.0)
    gs.tight_layout(fig)
    ax1 = plt.subplot(gs[0:2, :3], projection=proj)
    ax2 = plt.subplot(gs[0:2, 3:], projection=proj)
    ax3 = plt.subplot(gs[2, 1:5], projection=proj)
    ax = [ax1,ax2,ax3]

    # formatting for tick labels
    lon_formatter = LongitudeFormatter(number_format='0.0f',
                                        degree_symbol='',
                                        dateline_direction_label=False)
    lat_formatter = LatitudeFormatter(number_format='0.0f',
                                        degree_symbol='')

    # too many vectors to see well, so prune by striding through data:
    skip=(slice(None,None,5),slice(None,None,8))

    title_string = "Missing title!"
    title_string_base = title_string
    if "var_name" in kwargs:
        var_name = kwargs["var_name"]
    else:
        var_name = "missing VAR name"
    #End if

    if "case_name" in kwargs:
        case_name = kwargs["case_name"]
        if plev:
            title_string = f"{case_name} {var_name} [{plev} hPa]"
        else:
            title_string = f"{case_name} {var_name}"
        #End if
    #End if
    if "baseline" in kwargs:
        data_name = kwargs["baseline"]
        if plev:
            title_string_base = f"{data_name} {var_name} [{plev} hPa]"
        else:
            title_string_base = f"{data_name} {var_name}"
        #End if
    #End if

    # Calculate vector magnitudes.
    # Please note that the difference field needs
    # to be calculated from the model and obs fields
    # in order to get the correct sign:
    mdl_mag_ma  = np.sqrt(umdlfld**2 + vmdlfld**2)
    obs_mag_ma  = np.sqrt(uobsfld**2 + vobsfld**2)

    #Convert vector magnitudes to xarray DataArrays:
    mdl_mag  = xr.DataArray(mdl_mag_ma)
    obs_mag  = xr.DataArray(obs_mag_ma)
    diff_mag = mdl_mag - obs_mag

    # Get difference limits, in order to plot the correct range:
    min_diff_val = np.min(diff_mag)
    max_diff_val = np.max(diff_mag)

    # Color normalization for difference
    if (min_diff_val < 0) and (0 < max_diff_val):
        normdiff = mpl.colors.TwoSlopeNorm(vmin=min_diff_val, vmax=max_diff_val, vcenter=0.0)
    else:
        normdiff = mpl.colors.Normalize(vmin=min_diff_val, vmax=max_diff_val)
    #End if

    # Generate vector plot:
    #  - contourf to show magnitude w/ colorbar
    #  - vectors (colored or not) to show flow --> subjective (?) choice for how to thin out vectors to be legible
    img1 = ax1.contourf(lons, lats, mdl_mag, cmap='Greys', transform=ccrs.PlateCarree(), transform_first=True,)
    ax1.quiver(lons[skip], lats[skip], umdlfld[skip], vmdlfld[skip], mdl_mag.values[skip], transform=ccrs.PlateCarree(), cmap='Reds')

    img2 = ax2.contourf(lons, lats, obs_mag, cmap='Greys', transform=ccrs.PlateCarree(), transform_first=True)
    ax2.quiver(lons[skip], lats[skip], uobsfld[skip], vobsfld[skip], obs_mag.values[skip], transform=ccrs.PlateCarree(), cmap='Reds')

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiString' in kwargs:
        tiString = kwargs.pop("tiString")
    else:
        tiString = ''
    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8
    #End if

    #Set Main title for subplots:
    st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=18)
    st.set_y(0.85)

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    ax[0].set_title(case_title, loc='left', fontsize=tiFontSize)

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)

    #Set stats: area_avg
    ax[0].set_title(f"Mean: {mdl_mag.weighted(wgt).mean().item():5.2f}\nMax: {mdl_mag.max():5.2f}\nMin: {mdl_mag.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[1].set_title(f"Mean: {obs_mag.weighted(wgt).mean().item():5.2f}\nMax: {obs_mag.max():5.2f}\nMin: {mdl_mag.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[-1].set_title(f"Mean: {diff_mag.weighted(wgt).mean().item():5.2f}\nMax: {diff_mag.max():5.2f}\nMin: {mdl_mag.min():5.2f}", loc='right',
                       fontsize=tiFontSize)

    # set rmse title:
    ax[-1].set_title(f"RMSE: ", fontsize=tiFontSize)
    ax[-1].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)

    if "units" in kwargs:
        ax[1].set_ylabel(f"[{kwargs['units']}]")
        ax[-1].set_ylabel(f"[{kwargs['units']}]")
    #End if

    # Add cosmetic plot features:
    for a in ax:
        a.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
        a.coastlines()
        a.set_xticks(np.linspace(-180, 120, 6), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=5, width=1.5, which='major')
        a.tick_params('both', length=5, width=1.5, which='minor')
        a.xaxis.set_major_formatter(lon_formatter)
        a.yaxis.set_major_formatter(lat_formatter)
    #End for

    # Add colorbar to vector plot:
    cb_c2_ax = inset_axes(ax2,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 100%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0, 1, 1),
                   bbox_transform=ax2.transAxes,
                   borderpad=0,
                   )
    fig.colorbar(img2, cax=cb_c2_ax)

    # Plot vector differences:
    img3 = ax3.contourf(lons, lats, diff_mag, transform=ccrs.PlateCarree(), transform_first=True, norm=normdiff, cmap='PuOr', alpha=0.5)
    ax3.quiver(lons[skip], lats[skip], udiffld[skip], vdiffld[skip], transform=ccrs.PlateCarree())

    # Add color bar to difference plot:
    cb_d_ax = inset_axes(ax3,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 100%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0, 1, 1),
                   bbox_transform=ax3.transAxes,
                   borderpad=0
                   )
    fig.colorbar(img3, cax=cb_d_ax)

    # Write final figure to file
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()


#######


def plot_map_and_save(wks, case_nickname, base_nickname,
                      case_climo_yrs, baseline_climo_yrs,
                      mdlfld, obsfld, diffld, pctld, unstructured=False,
                      obs=False, **kwargs):
    """This plots mdlfld, obsfld, diffld in a 3-row panel plot of maps.

    Parameters
    ----------
    wks : str or Path
        output file path
    case_nickname : str
        short name for case
    base_nickname : str
        short name for base case
    case_climo_yrs : list
        list of years in case climatology, used for annotation
    baseline_climo_yrs : list
        list of years in base case climatology, used for annotation
    mdlfld : xarray.DataArray
        input data for case
    obsfld : xarray.DataArray
        input data for base case
    diffld : xarray.DataArray
        input difference data
    pctld : xarray.DataArray
        input percent difference data
    kwargs : dict, optional
        variable-specific options, See Notes

    Notes
    -----
    kwargs expected to be a variable-specific section,
    possibly provided by an ADF Variable Defaults YAML file.
    Currently it is inspected for:
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
          we will check for an entry called `subplots`, `contourf`, and `colorbar`. So the YAML might looks something like:
          ```
           mpl:
             subplots:
               figsize: (3, 9)
             contourf:
               levels: 15
               cmap: Blues
             colorbar:
               shrink: 0.4
          ```
        + This is experimental, and if you find yourself doing much with this, you probably should write a new plotting script that does not rely on this module.
    When these are not provided, colormap is set to 'coolwarm' and limits/levels are set by data range.
    """

    #nice formatting for tick labels
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # preprocess
    if not unstructured:
        # - assume all three fields have same lat/lon
        lat = obsfld['lat']
        wgt = np.cos(np.radians(lat))
        mwrap, lon = add_cyclic_point(mdlfld, coord=mdlfld['lon'])
        owrap, _ = add_cyclic_point(obsfld, coord=obsfld['lon'])
        dwrap, _ = add_cyclic_point(diffld, coord=diffld['lon'])
        pwrap, _ = add_cyclic_point(pctld, coord=pctld['lon'])
        wrap_fields = (mwrap, owrap, pwrap, dwrap)
        # mesh for plots:
        lons, lats = np.meshgrid(lon, lat)
        # Note: using wrapped data makes spurious lines across plot (maybe coordinate dependent)
        lon2, lat2 = np.meshgrid(mdlfld['lon'], mdlfld['lat'])

        # get statistics (from non-wrapped)
        fields = (mdlfld, obsfld, pctld, diffld)
        area_avg = [spatial_average(x, weights=wgt, spatial_dims=None) for x in fields]

        d_rmse = wgt_rmse(mdlfld, obsfld, wgt)  # correct weighted RMSE for (lat,lon) fields.
        # specify the central longitude for the plot
        central_longitude = kwargs.get('central_longitude', 180)
    else:
        wgt = kwargs["wgt"]
        wrap_fields = (mdlfld, obsfld, pctld, diffld)
        area_avg = [global_average(x, wgt) for x in wrap_fields]

        # TODO Check this is correct, weighted rmse uses xarray weighted function
        #d_rmse = wgt_rmse(a, b, wgt)  
        d_rmse = (np.sqrt(((diffld**2)*wgt).sum())).values.item()

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiString' in kwargs:
        tiString = kwargs.pop("tiString")
    else:
        tiString = ''
    #End if

    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8
    #End if

    central_longitude = kwargs.get('central_longitude', 180)

    # generate dictionary of contour plot settings:
    cp_info = prep_contour_plot(mdlfld, obsfld, diffld, pctld, **kwargs)

    # create figure object,
    # controling DPI improves raster plots for unstructured data, but it does slow things down
    fig = plt.figure(figsize=(14,8), dpi=300)

    # LAYOUT WITH GRIDSPEC
    gs = mpl.gridspec.GridSpec(3, 6, wspace=2.0,hspace=0.0) # 2 rows, 4 columns, but each map will take up 2 columns
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax1 = plt.subplot(gs[0:2, :3], projection=proj, **cp_info['subplots_opt'])
    ax2 = plt.subplot(gs[0:2, 3:], projection=proj, **cp_info['subplots_opt'])
    ax3 = plt.subplot(gs[2, :3], projection=proj, **cp_info['subplots_opt'])
    ax4 = plt.subplot(gs[2, 3:], projection=proj, **cp_info['subplots_opt'])
    ax = [ax1,ax2,ax3,ax4]

    img = [] # contour plots
    cs = []  # contour lines, unused for now
    cb = []  # color bars, unused for now

    # formatting for tick labels
    lon_formatter = LongitudeFormatter(number_format='0.0f',
                                        degree_symbol='',
                                        dateline_direction_label=False)
    lat_formatter = LatitudeFormatter(number_format='0.0f',
                                        degree_symbol='')

    for i, a in enumerate(wrap_fields):

        if i == len(wrap_fields)-1:
            levels = cp_info['levelsdiff']
            cmap = cp_info['cmapdiff']
            norm = cp_info['normdiff']
        elif i == len(wrap_fields)-2:
            levels = cp_info['levelspctdiff']
            cmap = cp_info['cmappct']
            norm = cp_info['pctnorm']
        else:
            levels = cp_info['levels1']
            cmap = cp_info['cmap1']
            norm = cp_info['norm1']
        
        # Unstructured grid check
        if not unstructured:
            levs = np.unique(np.array(levels))
            if len(levs) < 2:
                img.append(ax[i].contourf(lons,lats,a,colors="w",transform=ccrs.PlateCarree(),
                                          transform_first=True))
                ax[i].text(0.4, 0.4, empty_message, transform=ax[i].transAxes, bbox=props)
            else:
                img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm,
                                          transform=ccrs.PlateCarree(), transform_first=True,
                                          **cp_info['contourf_opt']
                                          ))
            #End if
        else:
            #configure for raster plotting, polycollection was slower
            #TODO, would be nice to have levels set from the info, above
            ax[i].set_global()
            raster = a.to_raster(ax=ax[i])
            im = ax[i].imshow(
                raster, cmap=cmap, origin="lower",
                extent=ax[i].get_xlim() + ax[i].get_ylim()
            )
            im.set_clim(vmin=levels[0],vmax=levels[-1])
            img.append(im)
        # End if unstructured grid

        #ax[i].set_title("AVG: {0:.3f}".format(area_avg[i]), loc='right', fontsize=11)
        ax[i].set_title(f"Mean: {area_avg[i].item():5.2f}\nMax: {wrap_fields[i].max().item():5.2f}\nMin: {wrap_fields[i].min().item():5.2f}", 
                     loc='right', fontsize=tiFontSize)

        # add contour lines <- Unused for now -JN
        # TODO: add an option to turn this on -BM
        #cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k', linewidths=1))
        #ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        #ax[i].text( 10, -140, "CONTOUR FROM {} to {} by {}".format(min(cs[i].levels), max(cs[i].levels), cs[i].levels[1]-cs[i].levels[0]),
        #bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)

    # Custom setting for each subplot
    for a in ax:
        a.coastlines()
        a.set_global()
        a.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
        a.set_xticks(np.linspace(-180, 120, 6), crs=proj)
        a.set_yticks(np.linspace(-90, 90, 7), crs=proj)
        a.tick_params('both', length=5, width=1.5, which='both')
        a.xaxis.set_major_formatter(lon_formatter)
        a.yaxis.set_major_formatter(lat_formatter)

    st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=18)
    st.set_y(0.85)

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    ax[0].set_title(case_title, loc='left', fontsize=tiFontSize)

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)

    # set rmse title:
    ax[3].set_title(f"RMSE: {d_rmse:.3f}", fontsize=tiFontSize)
    ax[3].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)
    ax[2].set_title("Test % Diff Baseline", loc='left', fontsize=tiFontSize,fontweight="bold")

    # Cosmetic adjustments to avoid label overlap
    # also makes plots different sizes...
    #ax[0].set_xticklabels([])
    #ax[1].set_xticklabels([])
    #ax[1].set_yticklabels([])
    #ax[3].set_yticklabels([])

    # __COLORBARS__
    cb_mean_ax = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 100%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0, 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img[1], cax=cb_mean_ax, **cp_info['colorbar_opt'])
    

    cb_pct_ax = inset_axes(ax3,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 100%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0, 1, 1),
                    bbox_transform=ax3.transAxes,
                    borderpad=0,
                    )
    PCT_CB = fig.colorbar(img[2], cax=cb_pct_ax, **cp_info['colorbar_opt'])
    PCT_CB.ax.set_ylabel="%"

    cb_diff_ax = inset_axes(ax4,
                    width="5%",  # width = 5% of parent_bbox width
                    height="100%",  # height : 100%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0, 1, 1),
                    bbox_transform=ax4.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img[3], cax=cb_diff_ax, **cp_info['colorbar_opt'])

    # Write final figure to file
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()


### END plot_map_and_save

# I don't think this is used anywhere and could likely be removed -WW
def plot_unstructured_map_and_save(wks, case_nickname, base_nickname,
                                   case_climo_yrs, baseline_climo_yrs,
                                   mdlfld, obsfld, diffld, pctld, wgt,
                                   obs=False, projection='global',**kwargs):

    """This plots mdlfld, obsfld, diffld in a 3-row panel plot of maps.

    Parameters
    ----------
    wks : str or Path
        output file path
    case_nickname : str
        short name for case
    base_nickname : str
        short name for base case
    case_climo_yrs : list
        list of years in case climatology, used for annotation
    baseline_climo_yrs : list
        list of years in base case climatology, used for annotation
    mdlfld : uxarray.DataArray
        input data for case, needs units and long name attrubutes
    obsfld : uxarray.DataArray
        input data for base case, needs units and long name attrubutes 
    diffld : uxarray.DataArray
        input difference data, needs units and long name attrubutes
    pctld : uxarray.DataArray
        input percent difference data, needs units and long name attrubutes
    wgt : uxarray.DataArray
        weights assumed to be (area*landfrac)/(area*landfrac).sum()
    kwargs : dict, optional
        variable-specific options, See Notes

    Notes
    -----
    kwargs expected to be a variable-specific section,
    possibly provided by an ADF Variable Defaults YAML file.
    Currently it is inspected for:
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
          we will check for an entry called `subplots`, `contourf`, and `colorbar`. So the YAML might looks something like:
          ```
           mpl:
             subplots:
               figsize: (3, 9)
             contourf:
               levels: 15
               cmap: Blues
             colorbar:
               shrink: 0.4
          ```
        + This is experimental, and if you find yourself doing much with this, you probably should write a new plotting script that does not rely on this module.
    When these are not provided, colormap is set to 'coolwarm' and limits/levels are set by data range.
    """
    
    # prepare info for plotting
    wrap_fields = (mdlfld, obsfld, diffld, pctld)
    area_avg = [global_average(x, wgt) for x in wrap_fields]

    # TODO Check this is correct, weighted rmse uses xarray weighted function
    #d_rmse = wgt_rmse(a, b, wgt)  
    d_rmse = (np.sqrt(((diffld**2)*wgt).sum())).values.item()

    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiString' in kwargs:
        tiString = kwargs.pop("tiString")
    else:
        tiString = ''
        
    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8
        
    #generate a dictionary of contour plot settings:
    cp_info = prep_contour_plot(mdlfld, obsfld, diffld, pctld, **kwargs)
    
    if projection == 'global':
        transform = ccrs.PlateCarree()
        proj = ccrs.PlateCarree()
        figsize= (14, 7)
    elif projection == 'arctic':
        transform = ccrs.NorthPolarStereo()
        proj = ccrs.NorthPolarStereo()
        figsize = (8, 8)
        
    #nice formatting for tick labels
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    lon_formatter = LongitudeFormatter(number_format='0.0f',
                                    degree_symbol='',
                                    dateline_direction_label=False)
    lat_formatter = LatitudeFormatter(number_format='0.0f',
                                  degree_symbol='')

    # create figure object
    fig, axs = plt.subplots(2,2,
        figsize=figsize,
        facecolor="w",
        constrained_layout=True,
        subplot_kw=dict(projection=proj),
        dpi=300,
        **cp_info['subplots_opt']
    )
    axs=axs.flatten()
    
    # Loop over data arrays to make plots
    for i, a in enumerate(wrap_fields):
        if i == len(wrap_fields)-2:
            levels = cp_info['levelsdiff']
            cmap = cp_info['cmapdiff']
            norm = cp_info['normdiff']
        elif i == len(wrap_fields)-1:
            levels = cp_info['levelspctdiff']
            cmap = cp_info['cmappct']
            norm = cp_info['pctnorm']
        else:
            levels = cp_info['levels1']
            cmap = cp_info['cmap1']
            norm = cp_info['norm1']
    
        levs = np.unique(np.array(levels))
    
        #configure for polycollection plotting
        #TODO, would be nice to have levels set from the info, above
        axs[i].set_global()
        raster = a.to_raster(ax=axs[i])
        img = axs[i].imshow(
            raster, cmap=cmap, origin="lower", extent=axs[i].get_xlim() + axs[i].get_ylim()
        )
        img.set_clim(vmin=levels[0],vmax=levels[-1])

        if i > 0:
            cbar = plt.colorbar(img, ax=axs[i], orientation='vertical',
                                pad=0.05, shrink=0.8, **cp_info['colorbar_opt'])
            #TODO keep variable attributes on dataarrays
            #cbar.set_label(wrap_fields[i].attrs['units'])
        #Set stats: area_avg
        axs[i].set_title(f"Mean: {area_avg[i].item():5.2f}\nMax: {wrap_fields[i].max().item():5.2f}\nMin: {wrap_fields[i].min().item():5.2f}", 
                     loc='right', fontsize=tiFontSize)
   
    # Custom setting for each subplot
    for a in axs:
        a.coastlines()
        if projection=='global':
            a.set_global()
            a.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
            a.set_xticks(np.linspace(-180, 120, 6), crs=proj)
            a.set_yticks(np.linspace(-90, 90, 7), crs=proj)
            a.tick_params('both', length=5, width=1.5, which='major')
            a.tick_params('both', length=5, width=1.5, which='minor')
            a.xaxis.set_major_formatter(lon_formatter)
            a.yaxis.set_major_formatter(lat_formatter)
        elif projection == 'arctic':
            a.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
            # __Follow the cartopy gallery example to make circular__:
            # Compute a circle in axes coordinates, which we can use as a boundary
            # for the map. We can pan/zoom as much as we like - the boundary will be
            # permanently circular.
            theta = np.linspace(0, 2*np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpl.path.Path(verts * radius + center)
            a.set_boundary(circle, transform=a.transAxes)
            a.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                        lw=1, color="gray",y_inline=True, 
                        xlocs=range(-180,180,90), ylocs=range(0,90,10))
    
    st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=18)
    st.set_y(0.85)

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    axs[0].set_title(case_title, loc='left', fontsize=tiFontSize)
    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
        axs[1].set_title(base_title, loc='left', fontsize=tiFontSize)
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        axs[1].set_title(base_title, loc='left', fontsize=tiFontSize)
    axs[2].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)
    axs[2].set_title(f"RMSE: {d_rmse:.3f}", fontsize=tiFontSize)
    axs[3].set_title("Test % Diff Baseline", loc='left', fontsize=tiFontSize,fontweight="bold")
        
    fig.savefig(wks, bbox_inches='tight', dpi=300)
    
    #Close plots:
    plt.close()
    
## End of plot_unstructured_map_and_save

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

#
#  -- zonal & meridional mean code --
#

def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ('lev','lat')]
    else:
        raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)


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

def _plot_line(axobject, xdata, ydata, color, **kwargs):
    """Create a generic line plot and check for some ways to annotate."""

    if color != None:
        axobject.plot(xdata, ydata, c=color, **kwargs)
    else:
        axobject.plot(xdata, ydata, **kwargs)

    #Set Y-axis label:
    if hasattr(ydata, "units"):
        axobject.set_ylabel("[{units}]".format(units=getattr(ydata,"units")))
    elif "units" in kwargs:
        axobject.set_ylabel("[{units}]".format(kwargs["units"]))
    #End if

    return axobject

def _meridional_plot_line(ax, lon, data, color, **kwargs):
    """Create line plot with longitude as the X-axis."""

    ax = _plot_line(ax, lon, data, color, **kwargs)
    ax.set_xlim([lon.min(), lon.max()])
    #
    # annotate
    #
    ax.set_xlabel("LONGITUDE")
    if hasattr(data, "units"):
        ax.set_ylabel("{units}".format(units=getattr(data,"units")))
    elif "units" in kwargs:
        ax.set_ylabel("{units}".format(kwargs["units"]))
    return ax

def _zonal_plot_line(ax, lat, data, color, **kwargs):
    """Create line plot with latitude as the X-axis."""
    ax = _plot_line(ax, lat, data, color, **kwargs)
    ax.set_xlim([max([lat.min(), -90.]), min([lat.max(), 90.])])
    #
    # annotate
    #
    ax.set_xlabel("LATITUDE")
    if hasattr(data, "units"):
        ax.set_ylabel("{units}".format(units=getattr(data,"units")))
    elif "units" in kwargs:
        ax.set_ylabel("{units}".format(kwargs["units"]))
    return ax

def _zonal_plot_preslat(ax, lat, lev, data, **kwargs):
    """Create plot with latitude as the X-axis, and pressure as the Y-axis."""
    mlev, mlat = np.meshgrid(lev, lat)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'

    img = ax.contourf(mlat, mlev, data.transpose('lat', 'lev'), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which='minor', length=4, color='r')
    ax.set_ylim([np.max(lev), np.min(lev)])
    return img, ax


def _meridional_plot_preslon(ax, lon, lev, data, **kwargs):
    """Create plot with longitude as the X-axis, and pressure as the Y-axis."""

    mlev, mlon = np.meshgrid(lev, lon)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'

    img = ax.contourf(mlon, mlev, data.transpose('lon', 'lev'), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which='minor', length=4, color='r')
    ax.set_ylim([np.max(lev), np.min(lev)])
    return img, ax

def zonal_plot(lat, data, ax=None, color=None, **kwargs):
    """Make zonal plot

    Determine which kind of zonal plot is needed based
    on the input variable's dimensions.

    Parameters
    ----------
    lat
        latitude
    data
        input data
    ax : Axes, optional
        axes object to use
    color : str or mpl color specification
        color for the curve
    kwargs : dict, optional
        plotting options

    Notes
    -----
    Checks if there is a `lev` dimension to determine if
    it is a lat-pres plot or a line plot.
    """
    if ax is None:
        ax = plt.gca()
    if 'lev' in data.dims:
        img, ax = _zonal_plot_preslat(ax, lat, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = _zonal_plot_line(ax, lat, data, color, **kwargs)
        return ax

def meridional_plot(lon, data, ax=None, color=None, **kwargs):
    """Make meridional plot

    Determine which kind of meridional plot is needed based
    on the input variable's dimensions.


    Parameters
    ----------
    lon
        longitude
    data
        input data
    ax : Axes, optional
        axes object to use
    color : str or mpl color specification
        color for the curve
    kwargs : dict, optional
        plotting options

    Notes
    -----
    Checks if there is a `lev` dimension to determine if
    it is a lon-pres plot or a line plot.
    """
    if ax is None:
        ax = plt.gca()
    if 'lev' in data.dims:
        img, ax = _meridional_plot_preslon(ax, lon, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = _meridional_plot_line(ax, lon,  data, color, **kwargs)
        return ax

def prep_contour_plot(adata, bdata, diffdata, pctdata, **kwargs):
    """Preparation for making contour plots.

    Prepares for making contour plots of adata, bdata, diffdata, and pctdata, which is
    presumably the difference between adata and bdata.
    - set colormap from kwargs or defaults to coolwarm
    - set contour levels from kwargs or 12 evenly spaced levels to span the data
    - normalize colors based on specified contour levels or data range
    - set option for linear or log pressure when applicable
    - similar settings for difference, defaults to symmetric about zero
    - separates Matplotlib kwargs into their own dicts

    Parameters
    ----------
    adata, bdata, diffdata, pctdata
        the data to be plotted
    kwargs : dict, optional
        plotting options

    Returns
    -------
    dict
        a dict with the following:
        - 'subplots_opt': mpl kwargs for subplots
        - 'contourf_opt': mpl kwargs for contourf
        - 'colorbar_opt': mpl kwargs for colorbar
        - 'diff_colorbar_opt' : mpl kwargs for difference colorbar
        - 'normdiff': color normalization for difference panel
        - 'cmapdiff': colormap for difference panel
        - 'levelsdiff': contour levels for difference panel
        - 'cmap1': color map for a and b panels
        - 'norm1': color normalization for a and b panels
        - 'levels1' : contour levels for a and b panels
        - 'plot_log_p' : true/false whether to plot log(pressure) axis
    """
    # determine levels & color normalization:
    minval = np.min([np.min(adata), np.min(bdata)])
    maxval = np.max([np.max(adata), np.max(bdata)])

    # determine levels & color normalization:
    minval    = np.min([np.min(adata), np.min(bdata)])
    maxval    = np.max([np.max(adata), np.max(bdata)])
    absmaxdif = np.max(np.abs(diffdata.data))
    absmaxpct = np.max(np.abs(pctdata))

    # determine norm to use (deprecate this once minimum MPL version is high enough)
    normfunc, mplv = use_this_norm()

    if 'colormap' in kwargs:
        cmap1 = kwargs['colormap']
    else:
        cmap1 = 'coolwarm'
    #End if

    if 'contour_levels' in kwargs:
        levels1 = kwargs['contour_levels']
        if ('non_linear' in kwargs) and (kwargs['non_linear']):
            cmap_obj = cm.get_cmap(cmap1)
            norm1 = mpl.colors.BoundaryNorm(levels1, cmap_obj.N)
        else:
            norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    elif 'contour_levels_range' in kwargs:
        assert len(kwargs['contour_levels_range']) == 3, \
        "contour_levels_range must have exactly three entries: min, max, step"

        levels1 = np.arange(*kwargs['contour_levels_range'])
        if ('non_linear' in kwargs) and (kwargs['non_linear']):
            cmap_obj = cm.get_cmap(cmap1)
            norm1 = mpl.colors.BoundaryNorm(levels1, cmap_obj.N)
        else:
            norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    else:
        levels1 = np.linspace(minval, maxval, 12)
        if ('non_linear' in kwargs) and (kwargs['non_linear']):
            cmap_obj = cm.get_cmap(cmap1)
            norm1 = mpl.colors.BoundaryNorm(levels1, cmap_obj.N)
        else:
            norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    #End if

    #Check if the minval and maxval are actually different.  If not,
    #then set "levels1" to be an empty list, which will cause the
    #plotting scripts to add a label instead of trying to plot a variable
    #with no contours:
    if minval == maxval:
        levels1 = []
    #End if

    if ('colormap' not in kwargs) and ('contour_levels' not in kwargs):
        if ((minval < 0) and (0 < maxval)) and mplv > 2:
            norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        else:
            norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        #End if
    #End if

    # Difference options -- Check in kwargs for colormap and levels
    if "diff_colormap" in kwargs:
        cmapdiff = kwargs["diff_colormap"]
    else:
        cmapdiff = 'coolwarm'
    #End if

    if "diff_contour_levels" in kwargs:
        levelsdiff = kwargs["diff_contour_levels"]  # a list of explicit contour levels
    elif "diff_contour_range" in kwargs:
        assert len(kwargs['diff_contour_range']) == 3, \
        "diff_contour_range must have exactly three entries: min, max, step"

        levelsdiff = np.arange(*kwargs['diff_contour_range'])
    else:
        # set a symmetric color bar for diff:
        absmaxdif = np.max(np.abs(diffdata.data))
        # set levels for difference plot:
        levelsdiff = np.linspace(-1*absmaxdif, absmaxdif, 12)
        
    # Percent Difference options -- Check in kwargs for colormap and levels
    if "pct_diff_colormap" in kwargs:
        cmappct = kwargs["pct_diff_colormap"]
    else:
        cmappct = "PuOr_r"
    #End if

    if "pct_diff_contour_levels" in kwargs:
        levelspctdiff = kwargs["pct_diff_contour_levels"]  # a list of explicit contour levels
    elif "pct_diff_contour_range" in kwargs:
            assert len(kwargs['pct_diff_contour_range']) == 3, "pct_diff_contour_range must have exactly three entries: min, max, step"
            levelspctdiff = np.arange(*kwargs['pct_diff_contour_range'])
    else:
        levelspctdiff = [-100,-75,-50,-40,-30,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,30,40,50,75,100]
    pctnorm = mpl.colors.BoundaryNorm(levelspctdiff,256)

    if "plot_log_pressure" in kwargs:
        plot_log_p = kwargs["plot_log_pressure"]
    else:
        plot_log_p = False

    # color normalization for difference
    if ((np.min(levelsdiff) < 0) and (0 < np.max(levelsdiff))) and mplv > 2:
        normdiff = normfunc(vmin=np.min(levelsdiff), vmax=np.max(levelsdiff), vcenter=0.0)
    else:
        normdiff = mpl.colors.Normalize(vmin=np.min(levelsdiff), vmax=np.max(levelsdiff))

    #NOTE: Sometimes the contour levels chosen in the defaults file
    #can result in the "contourf" software stack generating a
    #'TypologyException', which should manifest itself as a
    #"PredicateError", but due to bugs in the stack itself
    #will also sometimes raise an AttributeError.

    #To prevent this from happening, the polar max and min values
    #are calculated, and if the default contour values are significantly
    #larger then the min-max values, then the min-max values are used instead:
    #-------------------------------
    if max(levels1) > 10*maxval:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    elif minval < 0 and min(levels1) < 10*minval:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    #End if

    if max(np.abs(levelsdiff)) > 10*absmaxdif:
        levelsdiff = np.linspace(-1*absmaxdif, absmaxdif, 12)

    #End if
    #-------------------------------

    subplots_opt = {}
    contourf_opt = {}
    colorbar_opt = {}
    diff_colorbar_opt = {}
    pct_colorbar_opt = {}

    # extract any MPL kwargs that should be passed on:
    if 'mpl' in kwargs:
        subplots_opt.update(kwargs['mpl'].get('subplots',{}))
        contourf_opt.update(kwargs['mpl'].get('contourf',{}))
        colorbar_opt.update(kwargs['mpl'].get('colorbar',{}))
        diff_colorbar_opt.update(kwargs['mpl'].get('diff_colorbar',{}))
        pct_colorbar_opt.update(kwargs['mpl'].get('pct_diff_colorbar',{}))
    #End if
    return {'subplots_opt': subplots_opt,
            'contourf_opt': contourf_opt,
            'colorbar_opt': colorbar_opt,
            'diff_colorbar_opt': diff_colorbar_opt,
            'pct_colorbar_opt': pct_colorbar_opt,
            'normdiff': normdiff,
            'cmapdiff': cmapdiff,
            'levelsdiff': levelsdiff,
            'pctnorm': pctnorm,
            'cmappct': cmappct,
            'levelspctdiff':levelspctdiff,
            'cmap1': cmap1,
            'norm1': norm1,
            'levels1': levels1,
            'plot_log_p': plot_log_p
            }


def plot_zonal_mean_and_save(wks, case_nickname, base_nickname,
                             case_climo_yrs, baseline_climo_yrs,
                             adata, bdata, has_lev, log_p=False, obs=False, **kwargs):

    """This is the default zonal mean plot

    Parameters
    ----------
    adata : data to plot ([lev], lat, [lon]).
            The vertical coordinate (lev) must be pressure levels.
    bdata : baseline or observations to plot adata against.

        - For 2-d variables (reduced to (lat,)):
          + 2 panels: (top) zonal mean, (bottom) difference
        - For 3-D variables (reduced to (lev,lat)):
          + 3 panels: (top) zonal mean adata, (middle) zonal mean bdata, (bottom) difference
          + pcolormesh/contour plot
    kwargs -> optional dictionary of plotting options
             ** Expecting this to be variable-specific section, possibly provided by ADF Variable Defaults YAML file.**
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
          we will check for an entry called `subplots`, `contourf`, and `colorbar`. So the YAML might looks something like:
          ```
           mpl:
             subplots:
               figsize: (3, 9)
             contourf:
               levels: 15
               cmap: Blues
             colorbar:
               shrink: 0.4
          ```
    """

    # style the plot:
    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8
    #End if

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
    if has_lev:

        # calculate zonal average:
        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)

        # calculate difference:
        diff = azm - bzm
        
        # calculate the percent change
        pct = (azm - bzm) / np.abs(bzm) * 100.0

        #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
        pct = pct.where(np.isfinite(pct), np.nan)
        pct = pct.fillna(0.0)

        if isinstance(pct, ux.UxDataArray):
            pct = ux.UxDataArray(pct)
        else:
            pct = pct

        # generate dictionary of contour plot settings:
        cp_info = prep_contour_plot(azm, bzm, diff, pct, **kwargs)

        # Generate zonal plot:
        fig, ax = plt.subplots(figsize=(10,10),nrows=4, constrained_layout=True, sharex=True, sharey=True,**cp_info['subplots_opt'])
        levs = np.unique(np.array(cp_info['levels1']))

        levs_diff = np.unique(np.array(cp_info['levelsdiff']))
        levs_pct_diff = np.unique(np.array(cp_info['levelspctdiff']))

        if len(levs) < 2:
            img0, ax[0] = zonal_plot(adata['lat'], azm, ax=ax[0])
            ax[0].text(0.4, 0.4, empty_message, transform=ax[0].transAxes, bbox=props)
            img1, ax[1] = zonal_plot(bdata['lat'], bzm, ax=ax[1])
            ax[1].text(0.4, 0.4, empty_message, transform=ax[1].transAxes, bbox=props)
        else:
            img0, ax[0] = zonal_plot(adata['lat'], azm, ax=ax[0], norm=cp_info['norm1'],cmap=cp_info['cmap1'],levels=cp_info['levels1'],**cp_info['contourf_opt'])
            img1, ax[1] = zonal_plot(bdata['lat'], bzm, ax=ax[1], norm=cp_info['norm1'],cmap=cp_info['cmap1'],levels=cp_info['levels1'],**cp_info['contourf_opt'])
            fig.colorbar(img0, ax=ax[0], location='right',**cp_info['colorbar_opt'])
            fig.colorbar(img1, ax=ax[1], location='right',**cp_info['colorbar_opt'])
        #End if

        if len(levs_diff) < 2:
            img2, ax[2] = zonal_plot(adata['lat'], diff, ax=ax[2])
            ax[2].text(0.4, 0.4, empty_message, transform=ax[2].transAxes, bbox=props)
        else:
            img2, ax[2] = zonal_plot(adata['lat'], diff, ax=ax[2], norm=cp_info['normdiff'],cmap=cp_info['cmapdiff'],levels=cp_info['levelsdiff'],**cp_info['contourf_opt'])
            fig.colorbar(img2, ax=ax[2], location='right',**cp_info['diff_colorbar_opt'])
            
        if len(levs_pct_diff) < 2:
            img3, ax[3] = zonal_plot(adata['lat'], pct, ax=ax[3])
            ax[3].text(0.4, 0.4, empty_message, transform=ax[3].transAxes, bbox=props)
        else:
            img3, ax[3] = zonal_plot(adata['lat'], pct, ax=ax[3], norm=cp_info['pctnorm'],cmap=cp_info['cmappct'],levels=cp_info['levelspctdiff'],**cp_info['contourf_opt'])
            fig.colorbar(img3, ax=ax[3], location='right',**cp_info['pct_colorbar_opt'])

        ax[0].set_title(case_title, loc='left', fontsize=tiFontSize)
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)
        ax[2].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)
        ax[3].set_title("Test % Diff Baseline", loc='left', fontsize=tiFontSize,fontweight="bold")


        # style the plot:
        #Set Main title for subplots:
        st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=15)
        st.set_y(0.85)
        ax[-1].set_xlabel("LATITUDE")

        if log_p:
            [a.set_yscale("log") for a in ax]

        fig.text(-0.03, 0.5, 'PRESSURE [hPa]', va='center', rotation='vertical')
    else:
        line = Line2D([0], [0], label="$\mathbf{Test}:$"+f"{case_nickname} - years: {case_climo_yrs[0]}-{case_climo_yrs[-1]}",
                        color="#1f77b4") # #1f77b4 -> matplotlib standard blue

        line2 = Line2D([0], [0], label=base_title,
                        color="#ff7f0e") # #ff7f0e -> matplotlib standard orange

        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)
        diff = azm - bzm
        
        # calculate the percent change
        pct = (azm - bzm) / np.abs(bzm) * 100.0
        #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
        pct = pct.where(np.isfinite(pct), np.nan)
        pct = pct.fillna(0.0)
        if isinstance(pct, ux.UxDataArray):
            pct = ux.UxDataArray(pct)
        else:
            pct = pct
        
        fig, ax = plt.subplots(nrows=3)
        ax = [ax[0],ax[1],ax[2]]

        #Set Main title for subplots:
        st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=15)
        st.set_y(1.02)

        zonal_plot(adata['lat'], azm, ax=ax[0],color="#1f77b4") # #1f77b4 -> matplotlib standard blue
        zonal_plot(bdata['lat'], bzm, ax=ax[0],color="#ff7f0e") # #ff7f0e -> matplotlib standard orange

        fig.legend(handles=[line,line2],bbox_to_anchor=(-0.15, 0.87, 1.05, .102),loc="right",
                   borderaxespad=0.0,fontsize=6,frameon=False)

        zonal_plot(adata['lat'], diff, ax=ax[1], color="k")
        ax[1].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=10)
        
        zonal_plot(adata['lat'], pct, ax=ax[2], color="k")
        ax[2].set_title("Test % Diff Baseline", loc='left', fontsize=10,fontweight="bold")

        for a in ax:
            try:
                a.label_outer()
            except:
                pass
            #End except
        #End for
    #End if

    #Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()



def plot_meridional_mean_and_save(wks, case_nickname, base_nickname,
                             case_climo_yrs, baseline_climo_yrs,
                             adata, bdata, has_lev, log_p=False, latbounds=None, obs=False, **kwargs):

    """Default meridional mean plot

    Parameters
    ----------
    wks :
        the figure object to plot in
    case_nickname : str
        short name of `adata` case, use for annotation
    base_nickname : str
        short name of `bdata` case, use for annotation
    case_climo_yrs : list
        years in the `adata` case, use for annotation
    baseline_climo_yrs : list:
        years in the `bdata` case, use for annotation
    adata : xarray.DataArray
        data to plot ([lev], [lat], lon).
        The vertical coordinate (lev) must be pressure levels.
    bdata : xarray.DataArray
        baseline or observations to plot adata against.
        It must have the same dimensions and vertical levels as adata.
    has_lev : bool
        whether lev dimension is present
    latbounds : numbers.Number or slice, optional
        indicates latitude bounds to average over
        if it is a number, assume symmetric about equator,
        otherwise expects `slice(south, north)`
        defaults to `slice(-5,5)`
    kwargs : dict, optional
        optional dictionary of plotting options, See Notes

    Notes
    -----

    - For 2-d variables (reduced to (lon,)):
        + 2 panels: (top) meridional mean, (bottom) difference
    - For 3-D variables (reduced to (lev,lon)):
        + 3 panels: (top) meridonal mean adata, (middle) meridional mean bdata, (bottom) difference
        + pcolormesh/contour plot

    - kwargs -> optional dictionary of plotting options
        ** Expecting this to be variable-specific section, possibly
        provided by ADF Variable Defaults YAML file.**
        - colormap             -> str, name of matplotlib colormap
        - contour_levels       -> list of explicit values or a tuple: (min, max, step)
        - diff_colormap        -> str, name of matplotlib colormap used for different plot
        - diff_contour_levels  -> list of explicit values or a tuple: (min, max, step)
        - tiString             -> str, Title String
        - tiFontSize           -> int, Title Font Size
        - mpl -> dict, This should be any matplotlib kwargs that should be passed along. Keep reading:
            + Organize these by the mpl function. In this function (`plot_meridional_mean_and_save`)
            we will check for an entry called `subplots`, `contourf`, and `colorbar`.
            So the YAML might looks something like:
            ```
            mpl:
                subplots:
                figsize: (3, 9)
                contourf:
                levels: 15
                cmap: Blues
                colorbar:
                shrink: 0.4
            ```
        """
    # apply averaging:
    import numbers  # built-in; just checking on the latbounds input
    if latbounds is None:
        latbounds = slice(-5, 5)
    elif isinstance(latbounds, numbers.Number):
        latbounds = slice(-1*np.absolute(latbounds), np.absolute(latbounds))
    elif not isinstance(latbounds, slice):  #If not a slice object, then quit this routine.
        print(f"ERROR: plot_meridonal_mean_and_save - received an invalid value for latbounds ({latbounds}). Must be a number or a slice.")
        return None
    #End if

    # style the plot:
    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    if 'tiFontSize' in kwargs:
        tiFontSize = kwargs.pop('tiFontSize')
    else:
        tiFontSize = 8
    #End if

    # possible that the data has time, but usually it won't
    if len(adata.dims) > 4:
        print(f"ERROR: plot_meridonal_mean_and_save - too many dimensions: {adata.dims}")
        return None

    if 'time' in adata.dims:
        adata = adata.mean(dim='time', keep_attrs=True)
    if 'lat' in adata.dims:
        latweight = np.cos(np.radians(adata.lat))
        adata = adata.weighted(latweight).mean(dim='lat', keep_attrs=True)
    if 'time' in bdata.dims:
        adata = bdata.mean(dim='time', keep_attrs=True)
    if 'lat' in bdata.dims:
        latweight = np.cos(np.radians(bdata.lat))
        bdata = bdata.weighted(latweight).mean(dim='lat', keep_attrs=True)
    # If there are other dimensions, they are still going to be there:
    if len(adata.dims) > 2:
        print(f"ERROR: plot_meridonal_mean_and_save - AFTER averaging, there are too many dimensions: {adata.dims}")
        return None

    diff = adata - bdata
    
    # calculate the percent change
    pct = (adata - bdata) / np.abs(bdata) * 100.0
    #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
    pct = pct.where(np.isfinite(pct), np.nan)
    pct = pct.fillna(0.0)
    if isinstance(pct, ux.UxDataArray):
        pct = ux.UxDataArray(pct)
    else:
        pct = pct

    # plot-controlling parameters:
    xdim = 'lon' # the name used for the x-axis dimension
    pltfunc = meridional_plot  # the plotting function ... maybe we can generalize to get zonal/meridional into one function (?)

    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"

    if has_lev:
        # generate dictionary of contour plot settings:
        cp_info = prep_contour_plot(adata, bdata, diff, pct, **kwargs)

        # generate plot objects:
        fig, ax = plt.subplots(figsize=(10,10),nrows=4, constrained_layout=True, sharex=True, sharey=True,**cp_info['subplots_opt'])
        levs = np.unique(np.array(cp_info['levels1']))
        levs_diff = np.unique(np.array(cp_info['levelsdiff']))
        levs_pctdiff = np.unique(np.array(cp_info['levelspctdiff']))

        if len(levs) < 2:
            img0, ax[0] = pltfunc(adata[xdim], adata, ax=ax[0])
            ax[0].text(0.4, 0.4, empty_message, transform=ax[0].transAxes, bbox=props)
            img1, ax[1] = pltfunc(bdata[xdim], bdata, ax=ax[1])
            ax[1].text(0.4, 0.4, empty_message, transform=ax[1].transAxes, bbox=props)
        else:
            img0, ax[0] = pltfunc(adata[xdim], adata, ax=ax[0], norm=cp_info['norm1'],cmap=cp_info['cmap1'],levels=cp_info['levels1'],**cp_info['contourf_opt'])
            img1, ax[1] = pltfunc(bdata[xdim], bdata, ax=ax[1], norm=cp_info['norm1'],cmap=cp_info['cmap1'],levels=cp_info['levels1'],**cp_info['contourf_opt'])
            cb0 = fig.colorbar(img0, ax=ax[0], location='right',**cp_info['colorbar_opt'])
            cb1 = fig.colorbar(img1, ax=ax[1], location='right',**cp_info['colorbar_opt'])
        #End if

        if len(levs_diff) < 2:
            img2, ax[2] = pltfunc(adata[xdim], diff, ax=ax[2])
            ax[2].text(0.4, 0.4, empty_message, transform=ax[2].transAxes, bbox=props)
        else:
            img2, ax[2] = pltfunc(adata[xdim], diff, ax=ax[2], norm=cp_info['normdiff'],cmap=cp_info['cmapdiff'],levels=cp_info['levelsdiff'],**cp_info['contourf_opt'])
            cb2 = fig.colorbar(img2, ax=ax[2], location='right',**cp_info['colorbar_opt'])
            
        if len(levs_pctdiff) < 2:
            img3, ax[3] = pltfunc(adata[xdim], pct, ax=ax[3])
            ax[3].text(0.4, 0.4, empty_message, transform=ax[3].transAxes, bbox=props)
        else:
            img3, ax[3] = pltfunc(adata[xdim], pct, ax=ax[3], norm=cp_info['pctnorm'],cmap=cp_info['cmappct'],levels=cp_info['levelspctdiff'],**cp_info['contourf_opt'])
            cb3 = fig.colorbar(img3, ax=ax[3], location='right',**cp_info['colorbar_opt'])

        #Set plot titles
        ax[0].set_title(case_title, loc='left', fontsize=tiFontSize)
        ax[1].set_title(base_title, loc='left', fontsize=tiFontSize)
        ax[2].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)
        ax[3].set_title("Test % Diff Baseline", loc='left', fontsize=tiFontSize, fontweight = "bold")

        # style the plot:
        #Set Main title for subplots:
        st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=15)
        st.set_y(0.85)
        ax[-1].set_xlabel("LONGITUDE")
        #if cp_info['plot_log_p']:
        #    [a.set_yscale("log") for a in ax]

        if log_p:
            [a.set_yscale("log") for a in ax]

        fig.text(-0.03, 0.5, 'PRESSURE [hPa]', va='center', rotation='vertical')

    else:
        line = Line2D([0], [0], label="$\mathbf{Test}:$"+f"{case_nickname} - years: {case_climo_yrs[0]}-{case_climo_yrs[-1]}",
                        color="#1f77b4") # #1f77b4 -> matplotlib standard blue

        line2 = Line2D([0], [0], label=base_title,
                        color="#ff7f0e") # #ff7f0e -> matplotlib standard orange



        fig, ax = plt.subplots(nrows=3)
        ax = [ax[0],ax[1],ax[2]]

        pltfunc(adata[xdim], adata, ax=ax[0],color="#1f77b4") # #1f77b4 -> matplotlib standard blue
        pltfunc(bdata[xdim], bdata, ax=ax[0],color="#ff7f0e") # #ff7f0e -> matplotlib standard orange
        pltfunc(adata[xdim], diff, ax=ax[1], color="k")
        pltfunc(adata[xdim], pct, ax=ax[2], color="k")

        ax[1].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=10)
        ax[2].set_title("Test % Diff Baseline", loc='left', fontsize=10, fontweight = "bold")

        #Set Main title for subplots:
        st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=15)
        st.set_y(1.02)

        fig.legend(handles=[line,line2],bbox_to_anchor=(-0.15, 0.87, 1.05, .102),loc="right",
                borderaxespad=0.0,fontsize=6,frameon=False)

        for a in ax:
            try:
                a.label_outer()
            except:
                pass
            #End except
        #End for
    #End if

    #Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()

#
#  -- zonal mean annual cycle --
#

def square_contour_difference(fld1, fld2, **kwargs):
    """Produce filled contours of fld1, fld2, and their difference with square axes.

    Intended use is latitude-by-month to show the annual cycle.
    Example use case: use climo files to get data, take zonal averages,
    rename "time" to "month" if desired,
    and then provide resulting DataArrays to this function.

    Parameters
    ----------
        fld1, fld2 : xarray.DataArray
            2-dimensional DataArrays with same shape
        **kwargs : dict, optional
            optional keyword arguments
            this function _only checks_ `kwargs` for `case1name`, `case2name`

    Returns
    -------
    fig
        figure object

    Notes
    -----
    Assumes `fld1.shape == fld2.shape` and `len(fld1.shape) == 2`

    Will try to label the cases by looking for
    `case1name` and `case2name` in `kwargs`,
    and then `fld1['case']` and `fld2['case']` (i.e., attributes)
    If no case name is found proceeds with empty strings.
    **IF THERE IS A BETTER CONVENTION WE SHOULD USE IT.**

    Each panel also puts the Min/Max values into the title string.

    Axis labels are upper-cased names of the coordinates of `fld1`.
    Ticks are automatic with the exception that if the
    first coordinate is "month" and is length 12, use `np.arange(1,13)`.
    """

    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."  # 2-Dimension => plot contourf
    assert fld1.shape == fld2.shape, "Input fields must have the same array shape."    # Same shape => allows difference


    if "case1name" in kwargs:
        case1name = kwargs.pop("case1name")
    elif hasattr(fld1, "case"):
        case1name = getattr(fld1, "case")
    else:
        case1name = ""

    if "case2name" in kwargs:
        case2name = kwargs.pop("case2name")
    elif hasattr(fld2, "case"):
        case2name = getattr(fld2, "case")
    else:
        case2name = ""

    # Geometry of the figure is hard-coded
    fig = plt.figure(figsize=(10,10))

    rows = 5
    columns = 5
    grid = mpl.gridspec.GridSpec(rows, columns, wspace=1, hspace=1,
                            width_ratios=[1,1,1,1,0.2],
                            height_ratios=[1,1,1,1,0.2])
    # plt.subplots_adjust(wspace= 0.01, hspace= 0.01)
    ax1 = plt.subplot(grid[0:2, 0:2])
    ax2 = plt.subplot(grid[0:2, 2:4])
    ax3 = plt.subplot(grid[0:2, 0:2])
    ax4 = plt.subplot(grid[0:2, 2:4])
    # color bars / means share top bar.
    cbax_top = plt.subplot(grid[0:2, -1])
    cbax_bot = plt.subplot(grid[-1, 1:3])

    # determine color normalization for means:
    mx = np.max([fld1.max(), fld2.max()])
    mn = np.min([fld1.min(), fld2.min()])
    mnorm = mpl.colors.Normalize(mn, mx)

    coord1, coord2 = fld1.coords  # ASSUMES xarray WITH coords AND 2-dimensions
    xx, yy = np.meshgrid(fld1[coord2], fld1[coord1])

    img1 = ax1.contourf(xx, yy, fld1.transpose())
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax1.set_xticks(np.arange(1,13))
    ax1.set_ylabel(coord2.upper())
    ax1.set_xlabel(coord1.upper())
    ax1.set_title(f"{case1name}\nMIN:{fld1.min().values:.2f}  MAX:{fld1.max().values:.2f}")

    ax2.contourf(xx, yy, fld2.transpose())
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax2.set_xticks(np.arange(1,13))
    ax2.set_xlabel(coord1.upper())
    ax2.set_title(f"{case2name}\nMIN:{fld2.min().values:.2f}  MAX:{fld2.max().values:.2f}")


    diff = fld1 - fld2
    
    pct = (fld1 - fld2) / np.abs(fld2) * 100.0
    #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
    pct = pct.where(np.isfinite(pct), np.nan)
    pct = pct.fillna(0.0)
    if isinstance(pct, ux.UxDataArray):
        pct = ux.UxDataArray(pct)
    else:
        pct = pct
    
    ## USE A DIVERGING COLORMAP CENTERED AT ZERO
    ## Special case is when min > 0 or max < 0
    dmin = diff.min()
    dmax = diff.max()
    if dmin > 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.cm.OrRd
    elif dmax < 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.cm.BuPu_r
    else:
        dnorm = mpl.colors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
        cmap = mpl.cm.RdBu_r
        
    img3 = ax3.contourf(xx, yy, diff.transpose(), cmap=cmap, norm=dnorm)
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax3.set_xticks(np.arange(1,13))
    ax3.set_ylabel(coord2.upper())
    ax3.set_xlabel(coord1.upper())
    ax3.set_title(f"DIFFERENCE (= a - b)\nMIN:{diff.min().values:.2f}  MAX:{diff.max().values:.2f}")
        
    ## USE A DIVERGING COLORMAP CENTERED AT ZERO
    ## Special case is when min > 0 or max < 0
    pmin = pct.min()
    pmax = pct.max()
    if pmin > 0:
        pnorm = mpl.colors.Normalize(pmin, pmax)
        cmap = mpl.cm.OrRd
    elif pmax < 0:
        pnorm = mpl.colors.Normalize(pmin, pmax)
        cmap = mpl.cm.BuPu_r
    else:
        pnorm = mpl.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
        cmap = mpl.cm.RdBu_r

    img4 = ax4.contourf(xx, yy, pct.transpose(), cmap=cmap, norm=pnorm)
    if (coord1 == 'month') and (fld1.shape[0] ==12):
        ax4.set_xticks(np.arange(1,13))
    ax4.set_ylabel(coord2.upper())
    ax4.set_xlabel(coord1.upper())
    ax4.set_title(f"PCT DIFFERENCE (= a % diff b)\nMIN:{pct.min().values:.2f}  MAX:{pct.max().values:.2f}")


    # Try to construct the title:
    if hasattr(fld1, "long_name"):
        tstr = getattr(fld1, "long_name")
    elif hasattr(fld2, "long_name"):
        tstr = getattr(fld2, "long_name")
    elif hasattr(fld1, "short_name"):
        tstr = getattr(fld1, "short_name")
    elif hasattr(fld2, "short_name"):
        tstr = getattr(fld2, "short_name")
    elif hasattr(fld1, "name"):
        tstr = getattr(fld1, "name")
    elif hasattr(fld2, "name"):
        tstr = getattr(fld2, "name")
    else:
        tstr = ""
    if hasattr(fld1, "units"):
        tstr = tstr + f" [{getattr(fld1, 'units')}]"
    elif hasattr(fld2, "units"):
        tstr = tstr + f" [{getattr(fld2, 'units')}]"
    else:
        tstr = tstr + "[-]"

    fig.suptitle(tstr, fontsize=18)

    cb1 = fig.colorbar(img1, cax=cbax_top)
    cb2 = fig.colorbar(img3, cax=cbax_bot, orientation='horizontal')
    cb3 = fig.colorbar(img4, cax=cbax_bot, orientation='horizontal')
    return fig

#####################
#END HELPER FUNCTIONS
