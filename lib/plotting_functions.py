"""
This module contains generic
plotting helper functions that
can be used across multiple
different user-provided
plotting scripts.
"""

#import statements:
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

#Set non-X-window backend for matplotlib:
mpl.use('Agg')

#Now import pyplot:
import matplotlib.pyplot as plt

#################
#HELPER FUNCTIONS
#################

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

def global_average(fld, wgt, verbose=False):
    """
    A simple, pure numpy global average.
    fld: an input ndarray
    wgt: a 1-dimensional array of weights
    wgt should be same size as one dimension of fld
    """

    s = fld.shape
    for i in range(len(s)):
        if np.size(fld, i) == len(wgt):
            a = i
            break
    fld2 = np.ma.masked_invalid(fld)
    if verbose:
        print("(global_average)-- fraction input missing: {}".format(fraction_nan(fld)))
        print("(global_average)-- fraction of mask that is True: {}".format(np.count_nonzero(fld2.mask) / np.size(fld2)))
        print("(global_average)-- apply ma.average along axis = {} // validate: {}".format(a, fld2.shape))
    avg1, sofw = np.ma.average(fld2, axis=a, weights=wgt, returned=True) # sofw is sum of weights

    return np.ma.average(avg1)

#######

def plot_map_and_save(wks, mdlfld, obsfld, diffld, **kwargs):
    """This plots mdlfld, obsfld, diffld in a 3-row panel plot of maps."""
    # preprocess
    # - assume all three fields have same lat/lon
    lat = obsfld['lat']
    wgt = np.cos(np.radians(lat))
    mwrap, lon = add_cyclic_point(mdlfld, coord=mdlfld['lon'])
    owrap, _ = add_cyclic_point(obsfld, coord=obsfld['lon'])
    dwrap, _ = add_cyclic_point(diffld, coord=diffld['lon'])
    wrap_fields = (mwrap, owrap, dwrap)
    # mesh for plots:
    lons, lats = np.meshgrid(lon, lat)
    # Note: using wrapped data makes spurious lines across plot (maybe coordinate dependent)
    lon2, lat2 = np.meshgrid(mdlfld['lon'], mdlfld['lat'])

    # get statistics (from non-wrapped)
    fields = (mdlfld, obsfld, diffld)
    area_avg = [global_average(x, wgt) for x in fields]

    d_rmse = np.sqrt(area_avg[-1] ** 2)  # RMSE

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

    # We want to make sure mdlfld and obsfld are on the same contours:
    minval = np.min([np.min(mdlfld), np.min(obsfld)])
    maxval = np.max([np.max(mdlfld), np.max(obsfld)])
    normfunc, mplv = use_this_norm()
    if ((minval < 0) and (0 < maxval)) and mplv > 2:
        norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        cmap1 = 'coolwarm'
    else:
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        cmap1 = 'coolwarm'

    if 'cnLevels' in kwargs:
        levels1 = kwargs.pop(cnLevels)
    else:
        levels1 = np.linspace(minval, maxval, 12)
    # set a symmetric color bar for diff:
    absmaxdif = np.max(np.abs(diffld))
    # set levels for difference plot:
    levelsd = np.linspace(-1*absmaxdif, absmaxdif, 12)

    fig, ax = plt.subplots(figsize=(6,12), nrows=3, subplot_kw={"projection":ccrs.PlateCarree()})
    img = [] # contour plots
    cs = []  # contour lines
    cb = []  # color bars

    for i, a in enumerate(wrap_fields):
        if i == len(wrap_fields)-1:
            levels = levelsd #Using 'levels=12' casued len() error in mpl. -JN
            #Only use "vcenter" if "matplotlib" version is greater than 2:
            if(mplv > 2):
                norm = normfunc(vmin=-1*absmaxdif, vcenter=0., vmax=absmaxdif)
            else:
                norm = normfunc(vmin=-1*absmaxdif, vmax=absmaxdif)
            cmap = 'coolwarm'
        else:
            levels = levels1
            cmap = cmap1
            norm = norm1

        img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), **kwargs))
        cb.append(fig.colorbar(img[i], ax=ax[i], shrink=0.8))
        ax[i].set_title("AVG: {0:.3f}".format(area_avg[i]), loc='right', fontsize=tiFontSize)
        # add contour lines <- Unused for now -JN
        #cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k', linewidths=1))
        #ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        #ax[i].text( 10, -140, "CONTOUR FROM {} to {} by {}".format(min(cs[i].levels), max(cs[i].levels), cs[i].levels[1]-cs[i].levels[0]),
        #bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)

    # set rmse title:
    ax[-1].set_title("RMSE: {0:.3f}".format(d_rmse), fontsize=tiFontSize)

    for a in ax:
        a.outline_patch.set_linewidth(1)
        a.coastlines()
        a.set_xticks(np.linspace(-180, 180, 7), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=10, width=2, which='major')
        a.tick_params('both', length=5, width=1, which='minor')
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()

#
#  -- zonal mean code --
#

def pres_from_hybrid(psfc, hya, hyb, p0=100000.):
    """
    Converts a hybrid level to a pressure
    level using the forumla:

    p = a(k)*p0 + b(k)*ps

    """
    return hya*p0 + hyb*psfc


def lev_to_plev(data, ps, hyam, hybm, P0=100000., new_levels=None):
    """
    Interpolate model hybrid levels to specified pressure levels.
    """
    pressure = pres_from_hybrid(ps, hyam, hybm, P0)
    if new_levels is None:
        pnew = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]  # mandatory levels
    else:
        pnew = new_levels
    data_interp = data.interp(lev=pnew)
    return data_interp


def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ('lev','lat')]
    else:
        raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)

def lat_lon_validate_dims(fld):
    """
    Check if input field has the correct
    dimensions needed to plot on lat/lon map.
    """
    # note: we can only handle variables that reduce to (lat,lon)
    if len(fld.dims) > 3:
        return False
    has_lat = 'lat' in fld.dims
    has_lon = 'lon' in fld.dims
    if not has_lat or not has_lon:
        return  False
    else:
        return True

def zm_validate_dims(fld):
    """
    Check if input field has the correct
    dimensions needed to zonally average.
    """
    # note: we can only handle variables that reduce to (lev, lat) or (lat,)
    if len(fld.dims) > 4:
        print(f"Sorry, too many dimensions: {fld.dims}")
        return None
    has_lev = 'lev' in fld.dims
    has_lat = 'lat' in fld.dims
    if not has_lat:
        return None
    else:
        return has_lat, has_lev


def _zonal_plot_line(ax, lat, data, **kwargs):
    """Create line plot with latitude as the X-axis."""
    ax.plot(lat, data, **kwargs)
    ax.set_xlim([max([lat.min(), -90.]), min([lat.max(), 90.])])
    #
    # annotate
    #
    ax.set_xlabel("LATITUDE")
    if hasattr(data, "units"):
        ax.set_ylabel("[{units}]".format(units=getattr(data,"units")))
    elif "units" in kwargs:
        ax.set_ylabel("[{units}]".format(kwargs["units"]))
    if hasattr(data, "long_name"):
        ax.set_title(getattr(data,"long_name"), loc="left")
    elif hasattr(data, "name"):
        ax.set_title(getattr(data,"name"), loc="left")
    return ax

def _zonal_plot_preslat(ax, lat, lev, data, **kwargs):
    """Create plot with latitude as the X-axis, and pressure as the Y-axis."""
    mlev, mlat = np.meshgrid(lev, lat)
    if 'cmap' in kwargs:
        cmap = kwargs.pop('cmap')
    else:
        cmap = 'Spectral_r'
    img = ax.contourf(mlat, mlev, data.transpose('lat', "lev"), cmap=cmap, **kwargs)
    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which='minor', length=4, color='r')
    ax.set_ylim([1000, 1])
    return img, ax

def zonal_plot(lat, data, ax=None, **kwargs):
    """
    Determine which kind of zonal plot is needed based
    on the input variable's dimensions.
    """
    if ax is None:
        ax = plt.gca()
    if 'lev' in data.dims:
        img, ax = _zonal_plot_preslat(ax, lat, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = _zonal_plot_line(ax, lat, data, **kwargs)
        return ax


def plot_zonal_mean_and_save(wks, adata, apsurf, ahya, ahyb, bdata, bpsurf, bhya, bhyb):
    """This is the default zonal mean plot:
        adata: data to plot ([lev], lat, [lon])
        apsurf: surface pressure (Pa) for adata when lev present; otherwise None
        ahya, ahyb: a and b hybrid-sigma coefficients when lev present; otherwise None
        same for b*.
        - For 2-d variables (reduced to (lat,)):
          + 2 panels: (top) zonal mean, (bottom) difference
        - For 3-D variables (reduced to (lev,lat)):
          + 3 panels: (top) zonal mean adata, (middle) zonal mean bdata, (bottom) diffdata
          + pcolormesh/contour plot
    """
    if apsurf is not None:
        aplev = lev_to_plev(adata, apsurf, ahya, ahyb, P0=100000., new_levels=None)
        bplev = lev_to_plev(bdata, bpsurf, bhya, bhyb, P0=100000., new_levels=None)
        azm = zonal_mean_xr(aplev)
        bzm = zonal_mean_xr(bplev)
        diff = azm - bzm
        # determine levels & color normalization:
        minval = np.min([np.min(azm), np.min(bzm)])
        maxval = np.max([np.max(azm), np.max(bzm)])
        normfunc, mplv = use_this_norm()
        if ((minval < 0) and (0 < maxval)):
            norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
            cmap1 = 'coolwarm'
        else:
            norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
            cmap1 = None
        diffnorm = normfunc(vmin=np.min(diff), vcenter=0.0, vmax=np.max(diff))
        fig, ax = plt.subplots(nrows=3, constrained_layout=True, sharex=True, sharey=True)
        img0, ax[0] = zonal_plot(adata['lat'], azm, ax=ax[0], norm=norm1)
        img1, ax[1] = zonal_plot(bdata['lat'], bzm, ax=ax[1], norm=norm1)
        img2, ax[2] = zonal_plot(adata['lat'], diff, ax=ax[2], norm=diffnorm)
        # style the plot:
        cb0 = fig.colorbar(img0, ax=ax[0], location='right')
        cb1 = fig.colorbar(img1, ax=ax[1], location='right')
        cb2 = fig.colorbar(img2, ax=ax[2], location='right')
        ax[-1].set_xlabel("LATITUDE")
        fig.text(-0.03, 0.5, 'PRESSURE [hPa]', va='center', rotation='vertical')
    else:
        azm = zonal_mean_xr(adata)
        bzm = zonal_mean_xr(bdata)
        diff = azm - bzm
        fig, ax = plt.subplots(nrows=2, constrained_layout=True)
        zonal_plot(adata['lat'], azm, ax=ax[0])
        zonal_plot(bdata['lat'], bzm, ax=ax[0])
        zonal_plot(adata['lat'], diff, ax=ax[1])
        for a in ax:
            try:
                a.label_outer()
            except:
                pass

    #Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches='tight', dpi=300)

    #Close plots:
    plt.close()

#
#  -- zonal mean annual cycle --
#

def square_contour_difference(fld1, fld2, **kwargs):
    """Produce a figure with square axes that show fld1, fld2,
       and their difference as filled contours.

       Intended use is latitude-by-month to show the annual cycle.
       Example use case: use climo files to get data, take zonal averages,
       rename "time" to "month" if desired,
       and then provide resulting DataArrays to this function.

       Input:
           fld1 and fld2 are 2-dimensional DataArrays with same shape
           kwargs are optional keyword arguments
               this function checks kwargs for `case1name`, `case2name`

       Returns:
           figure object

       Assumptions:
           fld1.shape == fld2.shape
           len(fld1.shape) == 2

       Annnotation:
           Will try to label the cases by looking for
           case1name and case2name in kwargs,
           and then fld1['case'] & fld2['case'] (i.e., attributes)
           If no case name, will proceed with empty strings.
           ** IF THERE IS A BETTER CONVENTION WE SHOULD USE IT.

           Each panel also puts the Min/Max values into the title string.

           Axis labels are upper-cased names of the coordinates of fld1.
           Ticks are automatic with the exception that if the
           first coordinate is "month" and is length 12, use np.arange(1,13).

    """
    assert fld1.shape == fld2.shape  # Same shape => allows difference
    assert len(fld1.shape) == 2      # 2-Dimension => plot contourf

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
    ax3 = plt.subplot(grid[2:4, 1:3])
    # color bars / means share top bar.
    cbax_top = plt.subplot(grid[0:2, -1])
    cbax_bot = plt.subplot(grid[-1, 1:3])

    # determine color normalization for means:
    mx = np.max([fld1.max(), fld2.max()])
    mn = np.min([fld1.min(), fld2.min()])
    mnorm = mpl.colors.Normalize(mn, mx)

    coord1, coord2 = fld1.coords  # ASSUMES xarray WITH coords AND 2-dimensions
    print(f"{coord1}, {coord2}")
    xx, yy = np.meshgrid(fld1[coord2], fld1[coord1])
    print(f"shape of meshgrid: {xx.shape}")

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
    return fig

#####################
#END HELPER FUNCTIONS
