"""                                                                    .
Generic plotting helper functions

Functions
---------
use_this_norm()
    switches matplotlib color normalization method
get_difference_colors(values)
    Provide a color norm and colormap assuming `values` is a difference field.
get_central_longitude(*args)
    Determine central longitude for maps.
meridional_plot_line(ax, lon, data, color, **kwargs)
    Create line plot with longitude as the X-axis.
zonal_plot_line(ax, lat, data, color, **kwargs)
    Create line plot with latitude as the X-axis.
zonal_plot_preslat(ax, lat, lev, data, **kwargs)
    Create plot with latitude as the X-axis, and pressure as the Y-axis.
meridional_plot_preslon(ax, lon, lev, data, **kwargs)
    Create plot with longitude as the X-axis, and pressure as the Y-axis.

Notes
-----
This module includes "private" methods intended for internal use only.

_plot_line(axobject, xdata, ydata, color, **kwargs)
    Create a generic line plot
"""

#import statements:
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.cm as cm
import cartopy.crs as ccrs

from adf_diag import AdfDiag
import adf_utils as utils

import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

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


#######

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


#######

def transform_coordinates_for_projection(proj, lon, lat):
    """
    Explicitly project coordinates using the projection object.
    
    Parameters
    ----------
    proj : cartopy.ccrs.CRS
        projection object
    lat : xarray.DataArray or numpy.ndarray
        latitudes (in degrees)
    lon :array.DataArray or numpy.ndarray
        longitudes (in degrees)
        
    Returns
    -------
    x_proj : numpy.ndarray
        array of projected longitudes
    y_proj : numpy.ndarray
        array of projected latitudes

    Notes
    -----
    This is what cartopy's transform_first=True *should* be doing internally.
    We find that it sometimes fails for polar plots, so do it with this manually.
    This dramatically speeds up polar plots.    
    """
    lons, lats = np.meshgrid(lon, lat)
    x_proj, y_proj, _ = proj.transform_points(ccrs.PlateCarree(), lons, lats).T # .T to unpack, .T again to get x,y,z arrays
    return x_proj.T, y_proj.T


#######

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


#######

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

#
#  -- zonal & meridional mean code --
#

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

def meridional_plot_line(ax, lon, data, color, **kwargs):
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

def zonal_plot_line(ax, lat, data, color, **kwargs):
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

def zonal_plot_preslat(ax, lat, lev, data, **kwargs):
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

def meridional_plot_preslon(ax, lon, lev, data, **kwargs):
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


#####################
#END HELPER FUNCTIONS