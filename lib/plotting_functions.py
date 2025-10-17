"""                                                                    .
Generic computation and plotting helper functions

Functions
---------
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

zonal_mean_xr(fld)
    Average over all dimensions except `lev` and `lat`.

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
"""

#import statements:
from typing import Optional
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
#nice formatting for tick labels
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D

from adf_base import AdfError
import plotting_utils as plot_utils
import adf_utils as utils

#Format warning messages:
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

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


def make_polar_plot(wks, case_nickname, 
                    base_nickname,
                    case_climo_yrs, 
                    baseline_climo_yrs,
                    d1:xr.DataArray, 
                    d2:xr.DataArray, 
                    difference:Optional[xr.DataArray]=None,
                    pctchange:Optional[xr.DataArray]=None,
                    domain:Optional[list]=None, 
                    hemisphere:Optional[str]=None, 
                    obs=False, 
                    **kwargs):

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

    if hemisphere.upper() == "NH":
        proj = ccrs.NorthPolarStereo()
    elif hemisphere.upper() == "SH":
        proj = ccrs.SouthPolarStereo()
    else:
        raise AdfError(f'[make_polar_plot] hemisphere not specified, must be NH or SH; hemisphere set as {hemisphere}')

    if domain is None:
        if hemisphere.upper() == "NH":
            domain = [-180, 180, 45, 90]
        else:
            domain = [-180, 180, -90, -45]

    # statistics for annotation (these are scalars):
    d1_region_mean, d1_region_max, d1_region_min = utils.domain_stats(d1, domain)
    d2_region_mean, d2_region_max, d2_region_min = utils.domain_stats(d2, domain)
    dif_region_mean, dif_region_max, dif_region_min = utils.domain_stats(dif, domain)
    pct_region_mean, pct_region_max, pct_region_min = utils.domain_stats(pct, domain)

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

    # -- deal with optional plotting arguments that might provide variable-dependent choices

    # determine levels & color normalization:
    minval    = np.min([np.min(d1), np.min(d2)])
    maxval    = np.max([np.max(d1), np.max(d2)])
    absmaxdif = np.max(np.abs(dif))
    absmaxpct = np.max(np.abs(pct))

    if 'colormap' in kwargs:
        cmap1 = kwargs['colormap']
    else:
        cmap1 = 'coolwarm'

    if 'contour_levels' in kwargs:
        levels1 = kwargs['contour_levels']
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    elif 'contour_levels_range' in kwargs:
        assert len(kwargs['contour_levels_range']) == 3, "contour_levels_range must have exactly three entries: min, max, step"
        levels1 = np.arange(*kwargs['contour_levels_range'])
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    else:
        levels1 = np.linspace(minval, maxval, 12)
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    if ('colormap' not in kwargs) and ('contour_levels' not in kwargs):
        norm1, cmap1 = plot_utils.get_difference_colors(levels1)  # maybe these are better defaults if nothing else is known.

    if "diff_contour_levels" in kwargs:
        levelsdiff = kwargs["diff_contour_levels"]  # a list of explicit contour levels
    elif "diff_contour_range" in kwargs:
            assert len(kwargs['diff_contour_range']) == 3, "diff_contour_range must have exactly three entries: min, max, step"
            levelsdiff = np.arange(*kwargs['diff_contour_range'])
    else:
        # set levels for difference plot (with a symmetric color bar):
        levelsdiff = np.linspace(-1*absmaxdif.data, absmaxdif.data, 12)
    #End if
    
    if "pct_diff_contour_levels" in kwargs:
        levelspctdiff = kwargs["pct_diff_contour_levels"]  # a list of explicit contour levels
    elif "pct_diff_contour_range" in kwargs:
            assert len(kwargs['pct_diff_contour_range']) == 3, "pct_diff_contour_range must have exactly three entries: min, max, step"
            levelspctdiff = np.arange(*kwargs['pct_diff_contour_range'])
    else:
        levelspctdiff = [-100,-75,-50,-40,-30,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,30,40,50,75,100]
    pctnorm = mpl.colors.BoundaryNorm(levelspctdiff,256)

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
        levelsdiff = np.linspace(-1*absmaxdif.data, absmaxdif.data, 12)
    
    
    #End if
    #-------------------------------

    # Difference options -- Check in kwargs for colormap and levels
    if "diff_colormap" in kwargs:
        cmapdiff = kwargs["diff_colormap"]
        dnorm, _ = plot_utils.get_difference_colors(levelsdiff)  # color map output ignored
    else:
        dnorm, cmapdiff = plot_utils.get_difference_colors(levelsdiff)  
        
    # Pct Difference options -- Check in kwargs for colormap and levels
    if "pct_diff_colormap" in kwargs:
        cmappct = kwargs["pct_diff_colormap"]        
    else:
        cmappct = "PuOr_r"
    #End if

    # -- end options
    lons, lats = plot_utils.transform_coordinates_for_projection(proj, lon_cyclic, d1.lat) # Explicit coordinate transform

    fig = plt.figure(figsize=(10,10))
    gs = mpl.gridspec.GridSpec(2, 4, wspace=0.9)

    ax1 = plt.subplot(gs[0, :2], projection=proj)
    ax2 = plt.subplot(gs[0, 2:], projection=proj)
    ax3 = plt.subplot(gs[1, :2], projection=proj)
    ax4 = plt.subplot(gs[1, 2:], projection=proj)

    levs = np.unique(np.array(levels1))
    levs_diff = np.unique(np.array(levelsdiff))
    levs_pctdiff = np.unique(np.array(levelspctdiff))

    # BPM: removing `transform=ccrs.PlateCarree()` from contourf calls & transform_first=True
    if len(levs) < 2:
        img1 = ax1.contourf(lons, lats, d1_cyclic, colors="w", norm=norm1)
        ax1.text(0.4, 0.4, empty_message, transform=ax1.transAxes, bbox=props)

        img2 = ax2.contourf(lons, lats, d2_cyclic, colors="w", norm=norm1)
        ax2.text(0.4, 0.4, empty_message, transform=ax2.transAxes, bbox=props)
    else:
        img1 = ax1.contourf(lons, lats, d1_cyclic, cmap=cmap1, norm=norm1, levels=levels1)
        img2 = ax2.contourf(lons, lats, d2_cyclic, cmap=cmap1, norm=norm1, levels=levels1)

    if len(levs_pctdiff) < 2:
        img3 = ax3.contourf(lons, lats, pct_cyclic, colors="w", norm=pctnorm)
        ax3.text(0.4, 0.4, empty_message, transform=ax3.transAxes, bbox=props)
    else:
        img3 = ax3.contourf(lons, lats, pct_cyclic, cmap=cmappct, norm=pctnorm, levels=levelspctdiff)

    if len(levs_diff) < 2:
        img4 = ax4.contourf(lons, lats, dif_cyclic, colors="w", norm=dnorm)
        ax4.text(0.4, 0.4, empty_message, transform=ax4.transAxes, bbox=props)
    else:
        img4 = ax4.contourf(lons, lats, dif_cyclic, cmap=cmapdiff, norm=dnorm, levels=levelsdiff)
        
    #Set Main title for subplots:
    st = fig.suptitle(wks.stem[:-5].replace("_"," - "), fontsize=18)
    st.set_y(0.95)

    #Set plot titles
    case_title = "$\mathbf{Test}:$"+f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    ax1.set_title(case_title, loc='left', fontsize=6) #fontsize=tiFontSize

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = "$\mathbf{Baseline}:$"+obs_title+"\n"+"$\mathbf{Variable}:$"+f"{obs_var}"
        ax2.set_title(base_title, loc='left', fontsize=6) #fontsize=tiFontSize
    else:
        base_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        ax2.set_title(base_title, loc='left', fontsize=6)

    ax1.text(-0.2, -0.10, f"Mean: {d1_region_mean:5.2f}\nMax: {d1_region_max:5.2f}\nMin: {d1_region_min:5.2f}", transform=ax1.transAxes)

    ax2.text(-0.2, -0.10, f"Mean: {d2_region_mean:5.2f}\nMax: {d2_region_max:5.2f}\nMin: {d2_region_min:5.2f}", transform=ax2.transAxes)

    ax3.text(-0.2, -0.10, f"Mean: {pct_region_mean:5.2f}\nMax: {pct_region_max:5.2f}\nMin: {pct_region_min:5.2f}", transform=ax3.transAxes)
    ax3.set_title("Test % diff Baseline", loc='left', fontsize=8)

    ax4.text(-0.2, -0.10, f"Mean: {dif_region_mean:5.2f}\nMax: {dif_region_max:5.2f}\nMin: {dif_region_min:5.2f}", transform=ax4.transAxes)
    ax4.set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=8)

    if "units" in kwargs:
        ax2.set_ylabel(kwargs["units"])
        ax4.set_ylabel(kwargs["units"])
    else:
        ax2.set_ylabel(f"{d1.units}")
        ax4.set_ylabel(f"{d1.units}")

    [a.set_extent(domain, ccrs.PlateCarree()) for a in [ax1, ax2, ax3, ax4]]
    [a.coastlines() for a in [ax1, ax2, ax3, ax4]]

    # __Follow the cartopy gallery example to make circular__:
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpl.path.Path(verts * radius + center)
    [a.set_boundary(circle, transform=a.transAxes) for a in [ax1, ax2, ax3, ax4]]

    # __COLORBARS__
    cb_mean_ax = inset_axes(ax2,
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=ax2.transAxes,
                    borderpad=0,
                    )
    fig.colorbar(img1, cax=cb_mean_ax)
    
    cb_pct_ax = inset_axes(ax3,
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=ax3.transAxes,
                    borderpad=0,
                    )  

    cb_diff_ax = inset_axes(ax4,
                    width="5%",  # width = 5% of parent_bbox width
                    height="90%",  # height : 90%
                    loc='lower left',
                    bbox_to_anchor=(1.05, 0.05, 1, 1),
                    bbox_transform=ax4.transAxes,
                    borderpad=0,
                    )      
                    
    fig.colorbar(img3, cax=cb_pct_ax)
    
    fig.colorbar(img4, cax=cb_diff_ax)

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
    ax[1].set_title(f"Mean: {obs_mag.weighted(wgt).mean().item():5.2f}\nMax: {obs_mag.max():5.2f}\nMin: {obs_mag.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[-1].set_title(f"Mean: {diff_mag.weighted(wgt).mean().item():5.2f}\nMax: {diff_mag.max():5.2f}\nMin: {diff_mag.min():5.2f}", loc='right',
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
                      mdlfld, obsfld, diffld, pctld, obs=False, **kwargs):
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
    fields = (mdlfld, obsfld, diffld, pctld)
    area_avg = [utils.spatial_average(x, weights=wgt, spatial_dims=None) for x in fields]

    d_rmse = utils.wgt_rmse(mdlfld, obsfld, wgt)  # correct weighted RMSE for (lat,lon) fields.

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

    # generate dictionary of contour plot settings:
    cp_info = plot_utils.prep_contour_plot(mdlfld, obsfld, diffld, pctld, **kwargs)

    # specify the central longitude for the plot
    central_longitude = kwargs.get('central_longitude', 180)

    # create figure object
    fig = plt.figure(figsize=(14,10))

    # LAYOUT WITH GRIDSPEC
    gs = mpl.gridspec.GridSpec(3, 6, wspace=2.0,hspace=0.0) # 2 rows, 4 columns, but each map will take up 2 columns
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax1 = plt.subplot(gs[0:2, :3], projection=proj, **cp_info['subplots_opt'])
    ax2 = plt.subplot(gs[0:2, 3:], projection=proj, **cp_info['subplots_opt'])
    ax3 = plt.subplot(gs[2, :3], projection=proj, **cp_info['subplots_opt'])
    ax4 = plt.subplot(gs[2, 3:], projection=proj, **cp_info['subplots_opt'])
    ax = [ax1,ax2,ax3,ax4]

    img = [] # contour plots
    cs = []  # contour lines
    cb = []  # color bars

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

        levs = np.unique(np.array(levels))
        if len(levs) < 2:
            img.append(ax[i].contourf(lons,lats,a,colors="w",transform=ccrs.PlateCarree(),transform_first=True))
            ax[i].text(0.4, 0.4, empty_message, transform=ax[i].transAxes, bbox=props)
        else:
            img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), transform_first=True, **cp_info['contourf_opt']))
        #End if
        ax[i].set_title("AVG: {0:.3f}".format(area_avg[i]), loc='right', fontsize=11)

        # add contour lines <- Unused for now -JN
        # TODO: add an option to turn this on -BM
        #cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k', linewidths=1))
        #ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        #ax[i].text( 10, -140, "CONTOUR FROM {} to {} by {}".format(min(cs[i].levels), max(cs[i].levels), cs[i].levels[1]-cs[i].levels[0]),
        #bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)

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
    ax[0].set_title(f"Mean: {mdlfld.weighted(wgt).mean().item():5.2f}\nMax: {mdlfld.max():5.2f}\nMin: {mdlfld.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[1].set_title(f"Mean: {obsfld.weighted(wgt).mean().item():5.2f}\nMax: {obsfld.max():5.2f}\nMin: {obsfld.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[2].set_title(f"Mean: {pctld.weighted(wgt).mean().item():5.2f}\nMax: {pctld.max():5.2f}\nMin: {pctld.min():5.2f}", loc='right',
                       fontsize=tiFontSize)
    ax[3].set_title(f"Mean: {diffld.weighted(wgt).mean().item():5.2f}\nMax: {diffld.max():5.2f}\nMin: {diffld.min():5.2f}", loc='right',
                       fontsize=tiFontSize)

    # set rmse title:
    ax[3].set_title(f"RMSE: {d_rmse:.3f}", fontsize=tiFontSize)
    ax[3].set_title("$\mathbf{Test} - \mathbf{Baseline}$", loc='left', fontsize=tiFontSize)
    ax[2].set_title("Test % Diff Baseline", loc='left', fontsize=tiFontSize,fontweight="bold")

    for a in ax:
        a.spines['geo'].set_linewidth(1.5) #cartopy's recommended method
        a.coastlines()
        a.set_xticks(np.linspace(-180, 120, 6), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=5, width=1.5, which='major')
        a.tick_params('both', length=5, width=1.5, which='minor')
        a.xaxis.set_major_formatter(lon_formatter)
        a.yaxis.set_major_formatter(lat_formatter)

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


#######

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
        img, ax = plot_utils.zonal_plot_preslat(ax, lat, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = plot_utils.zonal_plot_line(ax, lat, data, color, **kwargs)
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
        img, ax = plot_utils.meridional_plot_preslon(ax, lon, data['lev'], data, **kwargs)
        return img, ax
    else:
        ax = plot_utils.meridional_plot_line(ax, lon,  data, color, **kwargs)
        return ax


#######

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
        azm = plot_utils.zonal_mean_xr(adata)
        bzm = plot_utils.zonal_mean_xr(bdata)

        # calculate difference:
        diff = azm - bzm
        
        # calculate the percent change
        pct = (azm - bzm) / np.abs(bzm) * 100.0
        #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
        pct = pct.where(np.isfinite(pct), np.nan)
        pct = pct.fillna(0.0)

        # generate dictionary of contour plot settings:
        cp_info = plot_utils.prep_contour_plot(azm, bzm, diff, pct, **kwargs)

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

        azm = plot_utils.zonal_mean_xr(adata)
        bzm = plot_utils.zonal_mean_xr(bdata)
        diff = azm - bzm
        
        # calculate the percent change
        pct = (azm - bzm) / np.abs(bzm) * 100.0
        #check if pct has NaN's or Inf values and if so set them to 0 to prevent plotting errors
        pct = pct.where(np.isfinite(pct), np.nan)
        pct = pct.fillna(0.0)
        
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


#######

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
    log_p : bool, optional
        (Not implemented) use log(pressure) vertical axis
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
        cp_info = plot_utils.prep_contour_plot(adata, bdata, diff, pct, **kwargs)

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
        if cp_info['plot_log_p']:
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


#######

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