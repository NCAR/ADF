"""
Generic computation and plotting helper functions

Functions
---------
make_polar_plot(adfobj, wks, case_nickname, base_nickname,
                case_climo_yrs, baseline_climo_yrs, d1, d2, difference=None,
                pctchange=None, domain=None, hemisphere=None, obs=False, **kwargs)
    Make a stereographic polar plot for the given data and hemisphere.
plot_map_and_save(adfobj, wks, case_nickname, base_nickname,
                  case_climo_yrs, baseline_climo_yrs,
                  mdlfld, obsfld, diffld, pctld, obs=False, **kwargs)
    Map plots of `mdlfld`, `obsfld`, and their difference, `diffld`.
    With ``vector=True`` in kwargs it plots a vector field instead
    (this replaces the old ``plot_map_vect_and_save``).
zonal_plot(lat, data, ax=None, color=None, **kwargs)
    Make a line plot or pressure-latitude plot of `data`.
meridional_plot(lon, data, ax=None, color=None, **kwargs)
    Make a line plot or pressure-longitude plot of `data`.
prep_contour_plot
    Preparation for making contour plots.
plot_zonal_mean_and_save(adfobj, wks, case_nickname, base_nickname,
                         case_climo_yrs, baseline_climo_yrs,
                         adata, bdata, has_lev, log_p, obs=False, **kwargs)
    zonal mean plot
plot_meridional_mean_and_save(adfobj, wks, case_nickname, base_nickname,
                              case_climo_yrs, baseline_climo_yrs, adata, bdata,
                              has_lev, log_p=False, latbounds=None, obs=False, **kwargs)
    meridional mean plot
square_contour_difference
    Produce filled contours of fld1, fld2, and their difference with square axes.
"""

# This module collects many plotting functions, so allow it to be long:
# pylint: disable=too-many-lines

# ++++++++++++++++++++++++++++++
# Import standard python modules
# ++++++++++++++++++++++++++++++

from typing import Optional

import warnings  # use to warn user about missing files.

# +++++++++++++++++++++++++++++++++++++++++++++++++
# import non-standard python modules, including ADF
# +++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs

# nice formatting for tick labels
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

from adf_base import AdfError
import plotting_utils as plot_utils
import adf_utils as utils

# Format warning messages:
warnings.formatwarning = utils.my_formatwarning

# Set non-X-window backend for matplotlib:
mpl.use("Agg")

# Now import pyplot (must come after the backend is set above):
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-order,ungrouped-imports

EMPTY_MESSAGE = "No Valid\nData Points"
props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9}


# Set seasonal ranges:
seasons = {
    "ANN": np.arange(1, 13, 1),
    "DJF": [12, 1, 2],
    "JJA": [6, 7, 8],
    "MAM": [3, 4, 5],
    "SON": [9, 10, 11],
}

CBAR_SIZE = 8
CBAR_LABELPAD = 5

#################
# HELPER FUNCTIONS
#################


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def make_polar_plot(
    adfobj,
    wks,
    case_nickname,
    base_nickname,
    case_climo_yrs,
    baseline_climo_yrs,
    d1: xr.DataArray,
    d2: xr.DataArray,
    difference: Optional[xr.DataArray] = None,
    pctchange: Optional[xr.DataArray] = None,
    domain: Optional[list] = None,
    hemisphere: Optional[str] = None,
    obs=False,
    unstructured=False,
    **kwargs,
):
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

    if pctchange is None:
        pct = (d2 - d1) / np.abs(d1) * 100.0
    else:
        pct = pctchange

    # Mask NaN's and Inf's (e.g. division by zero), so they are not plotted:
    pct = pct.where(np.isfinite(pct))

    if hemisphere.upper() == "NH":
        proj = ccrs.NorthPolarStereo()
    elif hemisphere.upper() == "SH":
        proj = ccrs.SouthPolarStereo()
    else:
        raise AdfError(
            "[make_polar_plot] hemisphere not specified, must be NH or SH; "
            f"hemisphere set as {hemisphere}"
        )

    if domain is None:
        if hemisphere.upper() == "NH":
            domain = [-180, 180, 45, 90]
        else:
            domain = [-180, 180, -90, -45]

    means = []
    mins = []
    maxs = []
    if not unstructured:
        # statistics for annotation (these are scalars):
        d1_region_mean, d1_region_max, d1_region_min = utils.domain_stats(d1, domain)
        d2_region_mean, d2_region_max, d2_region_min = utils.domain_stats(d2, domain)
        dif_region_mean, dif_region_max, dif_region_min = utils.domain_stats(
            dif, domain
        )
        pct_region_mean, pct_region_max, pct_region_min = utils.domain_stats(
            pct, domain
        )
        # downsize to the specified region; makes plotting/rendering/saving much faster
        d1 = d1.sel(lat=slice(domain[2], domain[3]))
        d2 = d2.sel(lat=slice(domain[2], domain[3]))
        dif = dif.sel(lat=slice(domain[2], domain[3]))
        pct = pct.sel(lat=slice(domain[2], domain[3]))

        # add cyclic point to the data for better-looking plot
        d1_cyclic, lon_cyclic = add_cyclic_point(d1, coord=d1.lon)
        d2_cyclic, _ = add_cyclic_point(
            d2, coord=d2.lon
        )  # since we can take difference, assume same longitude coord.
        dif_cyclic, _ = add_cyclic_point(dif, coord=dif.lon)
        pct_cyclic, _ = add_cyclic_point(pct, coord=pct.lon)
        # wrap_fields = (d1_cyclic, d2_cyclic, dif_cyclic, pct_cyclic)
        wrap_fields = (d1_cyclic, d2_cyclic, pct_cyclic, dif_cyclic)
        # lons, lats = np.meshgrid(lon_cyclic, d1.lat)
        lons, lats = plot_utils.transform_coordinates_for_projection(
            proj, lon_cyclic, d1.lat
        )  # Explicit coordinate transform
    else:
        lat = d1.uxgrid.face_lat.values

    wrap_fields = (d1_cyclic, d2_cyclic, pct_cyclic, dif_cyclic)

    # -- deal with optional plotting arguments that might provide variable-dependent choices
    kwargs["adfobj"] = adfobj
    cp_info = plot_utils.prep_contour_plot(d1, d2, dif, pct, **kwargs)
    units = cp_info["units_label"]
    diff_units = cp_info["diff_units_label"]
    pct_units = cp_info["pct_units_label"]

    lons, lats = plot_utils.transform_coordinates_for_projection(
        proj, lon_cyclic, d1.lat
    )  # Explicit coordinate transform

    fig = plt.figure(figsize=(10, 10))
    gs = mpl.gridspec.GridSpec(2, 4, wspace=0.9)

    ax1 = plt.subplot(gs[0, :2], projection=proj)
    ax2 = plt.subplot(gs[0, 2:], projection=proj)
    ax3 = plt.subplot(gs[1, :2], projection=proj)
    ax4 = plt.subplot(gs[1, 2:], projection=proj)
    axs = [ax1, ax2, ax3, ax4]

    means = []
    mins = []
    maxs = []

    means.extend([d1_region_mean, d2_region_mean, pct_region_mean, dif_region_mean])
    mins.extend([d1_region_min, d2_region_min, pct_region_min, dif_region_min])
    maxs.extend([d1_region_max, d2_region_max, pct_region_max, dif_region_max])

    pctdiff_idx = 2
    diff_idx = 3
    imgs = []

    # Loop over data arrays to make plots
    for i, a in enumerate(wrap_fields):
        if i == diff_idx:
            levels = cp_info["levels_diff"]
            cmap = cp_info["cmap_diff"]
            norm = cp_info["norm_diff"]
            extend = cp_info["extend_diff"]
        elif i == pctdiff_idx:
            levels = cp_info["levels_pctdiff"]
            cmap = cp_info["cmap_pctdiff"]
            norm = cp_info["norm_pctdiff"]
            extend = cp_info["extend_pctdiff"]
        else:
            levels = cp_info["levels_sim"]
            cmap = cp_info["cmap_sim"]
            norm = cp_info["norm_sim"]
            extend = cp_info["extend_sim"]

        contour_levs = np.unique(np.array(levels))
        if len(contour_levs) < 2:
            imgs.append(axs[i].contourf(lons, lats, a, colors="w"))
            axs[i].text(0.4, 0.4, EMPTY_MESSAGE, transform=axs[i].transAxes, bbox=props)
        else:
            imgs.append(
                axs[i].contourf(
                    lons,
                    lats,
                    a,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    extend=extend,
                    **cp_info["contourf_opt"],
                )
            )

        # Set stats for title
        stat_mean = f"Mean: {means[i]:5.2f}"
        stat_max = f"Max: {maxs[i]:5.2f}"
        stat_min = f"Min: {mins[i]:5.2f}"
        stats = f"{stat_mean}\n{stat_max}\n{stat_min}"
        axs[i].text(-0.05, 0.0, stats, transform=axs[i].transAxes, fontsize=CBAR_SIZE)

    # End for

    # Set Main title for subplots:
    var_name = kwargs["var_name"]
    if "lev" in kwargs:
        vert_lev = kwargs["lev"]
        var_name = f"{var_name} at {vert_lev} hPa"
    st = fig.suptitle(
        f"{var_name}: {kwargs['season']}",
        fontsize=14,
        fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
    )
    st.set_y(0.95)

    # Set plot titles
    case_title = (
        r"$\mathbf{Test}:$"
        + f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    )
    axs[0].set_title(case_title, loc="left", fontsize=8)  # fontsize=ti_font_size

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = (
            r"$\mathbf{Baseline}:$"
            + obs_title
            + "\n"
            + r"$\mathbf{Variable}:$"
            + f"{obs_var}"
        )
        axs[1].set_title(base_title, loc="left", fontsize=8)  # fontsize=ti_font_size
    else:
        base_title = (
            r"$\mathbf{Baseline}:$"
            + f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        )
        axs[1].set_title(base_title, loc="left", fontsize=8)

    axs[2].set_title("Test % Diff Baseline", loc="left", fontsize=8, fontweight="bold")
    axs[3].set_title(r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=8)

    for a in axs:
        # __Follow the cartopy gallery example to make circular__:
        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpl.path.Path(verts * radius + center)
        a.set_boundary(circle, transform=a.transAxes)
        a.set_extent(domain, ccrs.PlateCarree())
        a.coastlines()
        a.gridlines(
            draw_labels=False,
            crs=ccrs.PlateCarree(),
            lw=1,
            color="gray",
            y_inline=True,
            xlocs=range(-180, 180, 90),
            ylocs=range(-80, 90, 10),
        )

    # __COLORBARS__
    cb_mean_ax = inset_axes(
        axs[1],
        width="5%",
        height="90%",  # height : 90%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.05, 1, 1),
        bbox_transform=axs[1].transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(imgs[0], cax=cb_mean_ax, **cp_info["colorbar_opt"])
    cbar.ax.set_title(units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left")
    cbar.ax.tick_params(labelsize=CBAR_SIZE)

    cb_pct_ax = inset_axes(
        axs[2],
        width="5%",
        height="90%",  # height : 90%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.05, 1, 1),
        bbox_transform=axs[3].transAxes,
        borderpad=0,
    )

    cb_diff_ax = inset_axes(
        axs[3],
        width="5%",
        height="90%",  # height : 90%
        loc="lower left",
        bbox_to_anchor=(1.05, 0.05, 1, 1),
        bbox_transform=axs[2].transAxes,
        borderpad=0,
    )

    pctdiff_cbar = fig.colorbar(imgs[2], cax=cb_pct_ax, **cp_info["pct_colorbar_opt"])
    pctdiff_cbar.ax.set_title(
        pct_units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left"
    )
    pctdiff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

    diff_cbar = fig.colorbar(imgs[3], cax=cb_diff_ax, **cp_info["diff_colorbar_opt"])
    diff_cbar.ax.set_title(
        diff_units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left"
    )
    diff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

    # Save files
    fig.savefig(wks, bbox_inches="tight", dpi=300)

    # Close figures to avoid memory issues:
    plt.close(fig)


#######

# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def plot_map_and_save(
    adfobj,
    wks,
    case_nickname,
    base_nickname,
    case_climo_yrs,
    baseline_climo_yrs,
    mdlfld,
    obsfld,
    diffld,
    pctld,
    obs=False,
    *,
    vmdlfld=None,
    vobsfld=None,
    vdiffld=None,
    **kwargs,
):
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
    obs : bool, optional
        True when the baseline is observations
    vmdlfld, vobsfld, vdiffld : xarray.DataArray, optional
        meridional components of a vector field, for a vector plot. The zonal
        components are then ``mdlfld``, ``obsfld`` and ``diffld``.
        The percent difference panel is not drawn for vector plots.
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
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along.
      Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
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
        + This is experimental, and if you find yourself doing much with this, you
          probably should write a new plotting script that does not rely on this module.
    When these are not provided, colormap is set to 'coolwarm' and limits/levels are
    set by data range.
    """

    kwargs["adfobj"] = adfobj
    vector = vmdlfld is not None

    # Only set for vector plots (when the V components are given):
    skip = sim_mags = diff_mag = None
    udif_wrap = vdif_wrap = umdl_wrap = vmdl_wrap = uobs_wrap = vobs_wrap = None
    mdl_mag = obs_mag = None

    if vector:
        # too many vectors to see well, so prune by striding through data:
        skip = (slice(None, None, 5), slice(None, None, 8))

        lat = mdlfld["lat"]

        # add cyclic longitude:
        umdl_wrap, lon = add_cyclic_point(mdlfld, coord=mdlfld["lon"])
        vmdl_wrap, _ = add_cyclic_point(vmdlfld, coord=vmdlfld["lon"])
        uobs_wrap, _ = add_cyclic_point(obsfld, coord=obsfld["lon"])
        vobs_wrap, _ = add_cyclic_point(vobsfld, coord=vobsfld["lon"])
        udif_wrap, _ = add_cyclic_point(diffld, coord=diffld["lon"])
        vdif_wrap, _ = add_cyclic_point(vdiffld, coord=vdiffld["lon"])
        upct_wrap, _ = add_cyclic_point(pctld, coord=pctld["lon"])

        # create mesh for plots:
        lons, lats = np.meshgrid(lon, lat)

        # Calculate vector magnitudes.
        # Please note that the difference field needs
        # to be calculated from the model and obs fields
        # in order to get the correct sign:
        mdl_mag_ma = np.sqrt(umdl_wrap**2 + vmdl_wrap**2)
        obs_mag_ma = np.sqrt(uobs_wrap**2 + vobs_wrap**2)

        # Convert vector magnitudes to xarray DataArrays:
        mdl_mag = xr.DataArray(mdl_mag_ma)
        obs_mag = xr.DataArray(obs_mag_ma)
        sim_mags = [mdl_mag, obs_mag]
        diff_mag = mdl_mag - obs_mag

        wrap_fields = (umdl_wrap, uobs_wrap, udif_wrap, upct_wrap)

        # get statistics from the non-wrapped vector magnitudes, so the
        # cyclic point is not counted twice:
        wgt = np.cos(np.radians(lat))
        mdl_mag_nowrap = np.sqrt(mdlfld**2 + vmdlfld**2)
        obs_mag_nowrap = np.sqrt(obsfld**2 + vobsfld**2)
        stat_fields = {
            0: mdl_mag_nowrap,
            1: obs_mag_nowrap,
            2: mdl_mag_nowrap - obs_mag_nowrap,
        }
        d_rmse = utils.wgt_rmse(
            mdl_mag_nowrap, obs_mag_nowrap, wgt
        )  # correct weighted RMSE for (lat,lon) fields
    else:
        # preprocess
        # - assume all three fields have same lat/lon
        lat = obsfld["lat"]
        wgt = np.cos(np.radians(lat))
        mwrap, lon = add_cyclic_point(mdlfld, coord=mdlfld["lon"])
        owrap, _ = add_cyclic_point(obsfld, coord=obsfld["lon"])
        dwrap, _ = add_cyclic_point(diffld, coord=diffld["lon"])
        pwrap, _ = add_cyclic_point(pctld, coord=pctld["lon"])
        wrap_fields = (mwrap, owrap, pwrap, dwrap)
        # mesh for plots:
        lons, lats = np.meshgrid(lon, lat)

        # get statistics (from non-wrapped)
        stat_fields = {0: mdlfld, 1: obsfld, 2: pctld, 3: diffld}
        d_rmse = utils.wgt_rmse(
            mdlfld, obsfld, wgt
        )  # correct weighted RMSE for (lat,lon) fields.

    # We should think about how to do plot customization and defaults.
    if "tiFontSize" in kwargs:
        ti_font_size = kwargs.pop("tiFontSize")
    else:
        ti_font_size = 8
    # End if

    # generate dictionary of contour plot settings:
    cp_info = plot_utils.prep_contour_plot(mdlfld, obsfld, diffld, pctld, **kwargs)
    units = cp_info["units_label"]
    diff_units = cp_info["diff_units_label"]
    pct_units = cp_info["pct_units_label"]

    # specify the central longitude for the plot
    central_longitude = kwargs.get("central_longitude", 180)

    # create figure object
    fig = plt.figure(figsize=(14, 10))

    # LAYOUT WITH GRIDSPEC
    wspace = 1.66
    if vector:
        hspace = 0
    else:
        hspace = -0.25
    gs = mpl.gridspec.GridSpec(
        3, 6, wspace=wspace, hspace=hspace
    )  # 2 rows, 4 columns, but each map will take up 2 columns
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax1 = plt.subplot(gs[0:2, :3], projection=proj, **cp_info["subplots_opt"])
    ax2 = plt.subplot(gs[0:2, 3:], projection=proj, **cp_info["subplots_opt"])

    if vector:
        ax3 = plt.subplot(gs[2, 1:5], projection=proj, **cp_info["subplots_opt"])
        ax = [ax1, ax2, ax3]
    else:
        ax3 = plt.subplot(gs[2, :3], projection=proj, **cp_info["subplots_opt"])
        ax4 = plt.subplot(gs[2, 3:], projection=proj, **cp_info["subplots_opt"])
        ax = [ax1, ax2, ax3, ax4]

    img = []  # contour plots

    # formatting for tick labels
    lon_formatter = LongitudeFormatter(
        number_format="0.0f", degree_symbol="", dateline_direction_label=False
    )
    # The latitude formatter is wrapped in a FuncFormatter so cartopy never
    # attaches it to the axis: an attached cartopy formatter reprojects every
    # tick through PROJ, and PROJ >= 9.8 returns NaN for points near the poles
    # on a PlateCarree with a non-zero central longitude.  For PlateCarree
    # the projected y-coordinate already equals latitude, so no reprojection
    # is needed.
    lat_formatter = FuncFormatter(
        lambda lat, pos, fmt=LatitudeFormatter(
            number_format="0.0f", degree_symbol=""
        ): fmt(lat, pos)
    )

    if vector:  # no percent difference panel for vectors
        pctdiff_idx = None
        diff_idx = 2
    else:
        pctdiff_idx = 2
        diff_idx = 3

    levels = cmap = norm = extend = None
    for i, a in enumerate(wrap_fields):
        if i == diff_idx:
            levels = cp_info["levels_diff"]
            cmap = cp_info["cmap_diff"]
            extend = cp_info["extend_diff"]
            if vector:
                # Get difference limits, in order to plot the correct range:
                min_diff_val = np.min(diff_mag)
                max_diff_val = np.max(diff_mag)

                # Color normalization for difference
                if min_diff_val < 0 < max_diff_val:
                    norm = mpl.colors.TwoSlopeNorm(
                        vmin=min_diff_val, vmax=max_diff_val, vcenter=0.0
                    )
                else:
                    norm = mpl.colors.Normalize(vmin=min_diff_val, vmax=max_diff_val)
                # End if
            else:
                norm = cp_info["norm_diff"]
        if (
            not vector
        ):  # non vector lat/lon gets percent diff as third (2) index For now, JR
            if i == pctdiff_idx:
                levels = cp_info["levels_pctdiff"]
                cmap = cp_info["cmap_pctdiff"]
                extend = cp_info["extend_pctdiff"]
                norm = cp_info["norm_pctdiff"]
        if i in [0, 1]:
            levels = cp_info["levels_sim"]
            cmap = cp_info["cmap_sim"]
            norm = cp_info["norm_sim"]
            extend = cp_info["extend_sim"]

        contour_levs = np.unique(np.array(levels))
        if len(contour_levs) < 2:
            img.append(
                ax[i].contourf(
                    lons,
                    lats,
                    a,
                    colors="w",
                    transform=ccrs.PlateCarree(),
                    transform_first=True,
                )
                ax[i].text(
                    0.4, 0.4, EMPTY_MESSAGE, transform=ax[i].transAxes, bbox=props
                )
            else:
                if vector:
                    if i == diff_idx:
                        ua, va = other_wrap_fields[i]
                        imgs.append(
                            ax[i].contourf(
                                lons,
                                lats,
                                diff_mag,
                                transform=ccrs.PlateCarree(),
                                transform_first=True,
                                norm=norm,
                                cmap="PuOr",
                                alpha=0.5,
                                extend=extend,
                            )
                        )
                        ax[i].quiver(
                            lons[skip],
                            lats[skip],
                            ua[skip],
                            va[skip],
                            transform=ccrs.PlateCarree(),
                        )
                    """if i == pctdiff_idx:
                        img.append(ax[i].contourf(lons, lats, pctdiff_mag, transform=ccrs.PlateCarree(),
                                                transform_first=True, norm=norm,
                                cmap='PuOr', extend=extend
                                ))
                        #ax[i].quiver(lons[skip], lats[skip], upctdiffld[skip], vpctdiffld[skip],
                        #             transform=ccrs.PlateCarree())"""
                    if i in [0, 1]:
                        ua, va = other_wrap_fields[i]
                        imgs.append(
                            ax[i].contourf(
                                lons,
                                lats,
                                sim_mags[i],
                                cmap="Greys",
                                transform=ccrs.PlateCarree(),
                                transform_first=True,
                                extend=extend,
                            )
                        )
                        ax[i].quiver(
                            lons[skip],
                            lats[skip],
                            ua[skip],
                            va[skip],
                            mdl_mag.values[skip],
                            transform=ccrs.PlateCarree(),
                            cmap="Reds",
                        )
                        ax[i].quiver(
                            lons[skip],
                            lats[skip],
                            ua[skip],
                            va[skip],
                            obs_mag.values[skip],
                            transform=ccrs.PlateCarree(),
                            cmap="Reds",
                        )
                else:
                    imgs.append(
                        ax[i].contourf(
                            lons,
                            lats,
                            fld,
                            levels=levels,
                            cmap=cmap,
                            norm=norm,
                            transform=ccrs.PlateCarree(),
                            transform_first=True,
                            extend=extend,
                            **cp_info["contourf_opt"],
                        )
                    )
                # img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), transform_first=True, **cp_info['contourf_opt']))
            # End if
        else:
            if vector:
                if i == diff_idx:
                    img.append(
                        ax[i].contourf(
                            lons,
                            lats,
                            diff_mag,
                            transform=ccrs.PlateCarree(),
                            transform_first=True,
                            norm=norm,
                            cmap="PuOr",
                            alpha=0.5,
                            extend=extend,
                        )
                    )
                    ax[i].quiver(
                        lons[skip],
                        lats[skip],
                        udif_wrap[skip],
                        vdif_wrap[skip],
                        transform=ccrs.PlateCarree(),
                    )
                if i in [0, 1]:
                    img.append(
                        ax[i].contourf(
                            lons,
                            lats,
                            sim_mags[i],
                            cmap="Greys",
                            transform=ccrs.PlateCarree(),
                            transform_first=True,
                            extend=extend,
                        )
                    )
                    # Test vectors on the test panel, baseline vectors on the baseline one:
                    if i == 0:
                        ufld, vfld = umdl_wrap, vmdl_wrap
                    else:
                        ufld, vfld = uobs_wrap, vobs_wrap
                    ax[i].quiver(
                        lons[skip],
                        lats[skip],
                        ufld[skip],
                        vfld[skip],
                        sim_mags[i].values[skip],
                        transform=ccrs.PlateCarree(),
                        cmap="Reds",
                    )
            else:
                img.append(
                    ax[i].contourf(
                        lons,
                        lats,
                        a,
                        levels=levels,
                        cmap=cmap,
                        norm=norm,
                        transform=ccrs.PlateCarree(),
                        transform_first=True,
                        extend=extend,
                        **cp_info["contourf_opt"],
                    )
                )
        # End if

    var_name = kwargs["var_name"]
    if "lev" in kwargs:
        vert_lev = kwargs["lev"]
        var_name = f"{var_name} at {vert_lev} hPa"
    st = fig.suptitle(
        f"{var_name}: {kwargs['season']}",
        fontsize=14,
        fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
    )
    st.set_y(0.825)

    # Set plot titles
    case_title = (
        r"$\mathbf{Test}:$"
        + f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    )
    ax[0].set_title(case_title, loc="left", fontsize=ti_font_size)

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = (
            r"$\mathbf{Baseline}:$"
            + obs_title
            + "\n"
            + r"$\mathbf{Variable}:$"
            + f"{obs_var}"
        )
        ax[1].set_title(base_title, loc="left", fontsize=ti_font_size)
    else:
        base_title = (
            r"$\mathbf{Baseline}:$"
            + f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        )
        ax[1].set_title(base_title, loc="left", fontsize=ti_font_size)

    # Set stats: area-weighted (cos(lat)) mean, as for the other ADF plots.
    # Vector plots give the stats of the vector magnitudes:
    for i, fld in stat_fields.items():
        ax[i].set_title(
            f"Mean: {utils.spatial_average(fld, weights=wgt).item():5.2f}\n"
            f"Max: {fld.max().item():5.2f}\nMin: {fld.min().item():5.2f}",
            loc="right",
            fontsize=ti_font_size,
        )

    if not vector:
        ax[pctdiff_idx].set_title(
            "Test % Diff Baseline", loc="left", fontsize=ti_font_size, fontweight="bold"
        )
    ax[diff_idx].set_title(f"RMSE: {d_rmse:.3f}", fontsize=ti_font_size)
    ax[diff_idx].set_title(
        r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=ti_font_size
    )

    for a in ax:
        a.tick_params(axis="both", labelsize=8)
        a.spines["geo"].set_linewidth(1.5)  # cartopy's recommended method
        a.coastlines()
        a.set_xticks(np.linspace(-180, 120, 6), crs=ccrs.PlateCarree())
        # Latitude ticks are set in projected coordinates (equal to latitude
        # for PlateCarree) to avoid PROJ returning NaN at the poles:
        a.set_yticks(np.linspace(-90, 90, 7))
        a.tick_params("both", length=5, width=1.5, which="major")
        a.tick_params("both", length=5, width=1.5, which="minor")
        a.xaxis.set_major_formatter(lon_formatter)
        a.yaxis.set_major_formatter(lat_formatter)

    # __COLORBARS__
    cb_mean_ax = inset_axes(
        ax[1],
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax2.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(img[1], cax=cb_mean_ax, **cp_info["colorbar_opt"])
    cbar.ax.set_title(units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left")
    cbar.ax.tick_params(labelsize=CBAR_SIZE)

    if not vector:
        cb_pct_ax = inset_axes(
            ax[pctdiff_idx],
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0, 1, 1),
            bbox_transform=ax3.transAxes,
            borderpad=0,
        )
        pctdiff_cbar = fig.colorbar(
            img[pctdiff_idx], cax=cb_pct_ax, **cp_info["pct_colorbar_opt"]
        )
        pctdiff_cbar.ax.set_title(
            pct_units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left"
        )
        pctdiff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

    cb_diff_ax = inset_axes(
        ax[diff_idx],
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0, 1, 1),
        bbox_transform=ax[diff_idx].transAxes,
        borderpad=0,
    )
    diff_cbar = fig.colorbar(
        img[diff_idx], cax=cb_diff_ax, **cp_info["diff_colorbar_opt"]
    )
    diff_cbar.ax.set_title(
        diff_units, fontsize=CBAR_SIZE, pad=CBAR_LABELPAD, loc="left"
    )
    diff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

    # Write final figure to file
    fig.savefig(wks, bbox_inches="tight", dpi=300)

    # Close plots:
    plt.close()


#######


"""def ux_to_lat_binned(da, bins=2):

    da_vals = np.array(da)
    lat = np.array(da.uxgrid.face_lat)

    lat_bins = np.arange(-90, 90 + bins, bins)
    lat_mid = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    #bin_index = np.digitize(lat, lat_bins) - 1
    bin_index = np.clip(np.digitize(lat, lat_bins) - 1, 0, len(lat_mid) - 1)

    # --- 1D case ---
    if da_vals.ndim == 1:
        out = np.full((len(lat_mid), len(lat)), np.nan)

        for i in range(len(lat_mid)):
            mask = bin_index == i
            out[i, mask] = da_vals[mask]

        dims = ["lat", "n_face"]
        coords = {"lat": lat_mid}

    # --- 2D case ---
    elif da_vals.ndim == 2:
        nlev = da_vals.shape[0]
        out = np.full((nlev, len(lat_mid), len(lat)), np.nan, dtype=float)

        for i in range(len(lat_mid)):
            mask = bin_index == i
            out[:, i, mask] = da_vals[:, mask]

        dims = ["lev", "lat", "n_face"]
        coords = {
            "lev": da["lev"].values,
            "lat": lat_mid
        }

    else:
        raise ValueError("Only 1D or 2D supported")
    coords["n_face"] = np.arange(len(lat))
    #return xr.DataArray(out, dims=dims, coords=coords)
    return xr.DataArray(
        out,
        dims=dims,
        coords=coords,
        name=da.name,
        attrs=da.attrs
    )"""


def ux_to_lat_binned(da, bins=2):

    lat = np.asarray(da.uxgrid.face_lat)

    lat_bins = np.arange(-90, 90 + bins, bins)
    lat_mid = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    bin_index = np.clip(np.digitize(lat, lat_bins) - 1, 0, len(lat_mid) - 1)

    face_dim = da.dims[-1]

    da_vals = da.values

    new_shape = (*da_vals.shape[:-1], len(lat_mid), da_vals.shape[-1])

    out = np.full(new_shape, np.nan, dtype=float)

    for i in range(len(lat_mid)):
        mask = bin_index == i
        out[..., i, mask] = da_vals[..., mask]

    new_dims = (*da.dims[:-1], "lat", face_dim)

    coords = {}

    for dim in da.dims[:-1]:
        coords[dim] = da.coords[dim]

    coords["lat"] = lat_mid
    coords[face_dim] = np.arange(len(lat))

    return xr.DataArray(out, dims=new_dims, coords=coords, name=da.name, attrs=da.attrs)


def ux_to_lon_binned(da, bins=2):

    da_vals = np.array(da)
    lon = np.array(da.uxgrid.face_lon)
    lon = (lon + 360) % 360  # converts -180→180 into 0→360

    lon_bins = np.arange(0, 360 + bins, bins)
    lon_mid = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    bin_index = np.clip(np.digitize(lon, lon_bins) - 1, 0, len(lon_mid) - 1)

    # --- 1D case ---
    if da_vals.ndim == 1:
        out = np.full((len(lon_mid), len(lon)), np.nan)

        for i in range(len(lon_mid)):
            mask = bin_index == i
            out[i, mask] = da_vals[mask]

        dims = ["lon", "n_face"]
        coords = {"lon": lon_mid}

    # --- 2D case ---
    elif da_vals.ndim == 2:
        nlev = da_vals.shape[0]
        out = np.full((nlev, len(lon_mid), len(lon)), np.nan, dtype=float)

        for i in range(len(lon_mid)):
            mask = bin_index == i
            out[:, i, mask] = da_vals[:, mask]

        dims = ["lev", "lon", "n_face"]
        coords = {"lev": da["lev"].values, "lon": lon_mid}

    else:
        raise ValueError("Only 1D or 2D supported")

    coords["n_face"] = np.arange(len(lon))
    # coords["lon_orig"] = np.array(da.uxgrid.face_lon)
    da_attrs = da.attrs.copy()
    da_attrs["lon_orig"] = np.array(da.uxgrid.face_lon)
    return xr.DataArray(
        out, dims=dims, coords=coords, name=da.name, attrs=da_attrs  # da.attrs
    )


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
    if "lev" in data.dims:
        img, ax = plot_utils.zonal_plot_preslat(ax, lat, data["lev"], data, **kwargs)
        return img, ax
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
    if "lev" in data.dims:
        img, ax = plot_utils.meridional_plot_preslon(
            ax, lon, data["lev"], data, **kwargs
        )
        return img, ax
    ax = plot_utils.meridional_plot_line(ax, lon, data, color, **kwargs)
    return ax


def zonal_mean_xr(fld):
    """Average over all dimensions except `lev` and `lat`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ("lev", "lat")]
    else:
        raise IOError("zonal_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)


def _lat_band(fld, latbounds):
    """The part of `fld` within the latitudes of the slice `latbounds`.

    Parameters
    ----------
    fld : xarray.DataArray or uxarray.UxDataArray
        Data with a `lat` dimension, or on an unstructured mesh.
    latbounds : slice
        ``slice(south, north)``, in degrees.

    Returns
    -------
    xarray.DataArray or uxarray.UxDataArray
        `fld` restricted to the latitude band; on a mesh, to the cells whose
        centres lie within it.
    """
    south, north = sorted((latbounds.start, latbounds.stop))
    if hasattr(fld, "uxgrid"):
        face_lat = fld.uxgrid.face_lat.values
        return fld.isel(n_face=np.where((face_lat >= south) & (face_lat <= north))[0])
    # By value rather than with .sel, so either latitude ordering works:
    return fld.isel(lat=((fld.lat >= south) & (fld.lat <= north)).values)


def meridional_mean_xr(fld):
    """Average over all dimensions except `lev` and `lon`."""
    if isinstance(fld, xr.DataArray):
        d = fld.dims
        davgovr = [dim for dim in d if dim not in ("lev", "lon")]
    else:
        raise IOError("meridional_mean_xr requires Xarray DataArray input.")
    return fld.mean(dim=davgovr)


#######


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def plot_zonal_mean_and_save(
    adfobj,
    wks,
    case_nickname,
    base_nickname,
    case_climo_yrs,
    baseline_climo_yrs,
    adata,
    bdata,
    has_lev,
    log_p,
    obs=False,
    **kwargs,
):
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
             ** Expecting this to be variable-specific section, possibly provided by
             ADF Variable Defaults YAML file.**
    - colormap -> str, name of matplotlib colormap
    - contour_levels -> list of explict values or a tuple: (min, max, step)
    - diff_colormap
    - diff_contour_levels
    - tiString -> str, Title String
    - tiFontSize -> int, Title Font Size
    - mpl -> dict, This should be any matplotlib kwargs that should be passed along.
      Keep reading:
        + Organize these by the mpl function. In this function (`plot_map_and_save`)
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
    kwargs["adfobj"] = adfobj

    # style the plot:
    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    ti_font_size = kwargs.pop("tiFontSize", 8)

    azm = utils.zonal_mean_xr(adata)
    bzm = utils.zonal_mean_xr(bdata)
    diff = azm - bzm

    # calculate the percent change
    pct = (azm - bzm) / np.abs(bzm) * 100.0
    # Mask NaN's and Inf's (e.g. division by zero), so they are not plotted:
    pct = pct.where(np.isfinite(pct))

    # generate dictionary of contour plot settings:
    cp_info = plot_utils.prep_contour_plot(azm, bzm, diff, pct, **kwargs)
    units = cp_info["units_label"]
    diff_units = cp_info["diff_units_label"]
    pct_units = cp_info["pct_units_label"]
    levels_diff = cp_info["levels_diff"]
    cmap_diff = cp_info["cmap_diff"]
    norm_diff = cp_info["norm_diff"]
    extend_diff = cp_info["extend_diff"]

    levels_pctdiff = cp_info["levels_pctdiff"]
    cmap_pctdiff = cp_info["cmap_pctdiff"]
    norm_pctdiff = cp_info["norm_pctdiff"]
    extend_pctdiff = cp_info["extend_pctdiff"]

    levels_sim = cp_info["levels_sim"]
    cmap_sim = cp_info["cmap_sim"]
    norm_sim = cp_info["norm_sim"]
    extend_sim = cp_info["extend_sim"]

    # Set plot titles
    case_title = (
        r"$\mathbf{Test}:$"
        + f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    )

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = (
            r"$\mathbf{Baseline}:$"
            + obs_title
            + "\n"
            + r"$\mathbf{Variable}:$"
            + f"{obs_var}"
        )
    else:
        base_title = (
            r"$\mathbf{Baseline}:$"
            + f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        )
    if has_lev:
        # Generate zonal plot:
        fig, ax = plt.subplots(
            figsize=(8, 10),
            nrows=4,
            constrained_layout=True,
            sharey=True,
            **cp_info["subplots_opt"],
        )

        contour_levs = np.unique(np.array(levels_sim))
        alat = adata["lat"]
        blat = bdata["lat"]

        levs_diff = np.unique(np.array(levels_diff))
        levs_pct_diff = np.unique(np.array(levels_pctdiff))
        cbar_labelpad_zonal = 5
        if len(contour_levs) < 2:
            img0, ax[0] = zonal_plot(alat, azm, ax=ax[0])
            ax[0].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[0].transAxes, bbox=props)
            img1, ax[1] = zonal_plot(blat, bzm, ax=ax[1])
            ax[1].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[1].transAxes, bbox=props)
        else:
            img0, ax[0] = zonal_plot(
                alat,
                azm,
                ax=ax[0],
                norm=norm_sim,
                cmap=cmap_sim,
                levels=levels_sim,
                extend=extend_sim,
                **cp_info["contourf_opt"],
            )
            img1, ax[1] = zonal_plot(
                blat,
                bzm,
                ax=ax[1],
                norm=norm_sim,
                cmap=cmap_sim,
                levels=levels_sim,
                extend=extend_sim,
                **cp_info["contourf_opt"],
            )

            cb_mean_ax = inset_axes(
                ax[0],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[0].transAxes,
                borderpad=0,
            )
            cbar0 = fig.colorbar(
                img0, cax=cb_mean_ax, location="right", **cp_info["colorbar_opt"]
            )
            cbar0.ax.set_title(
                units, fontsize=CBAR_SIZE, pad=cbar_labelpad_zonal, loc="left"
            )
            cbar0.ax.tick_params(labelsize=CBAR_SIZE)

            cb_mean_ax = inset_axes(
                ax[1],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[1].transAxes,
                borderpad=0,
            )
            cbar1 = fig.colorbar(
                img1, cax=cb_mean_ax, location="right", **cp_info["colorbar_opt"]
            )
            cbar1.ax.set_title(
                units, fontsize=CBAR_SIZE, pad=cbar_labelpad_zonal, loc="left"
            )
            cbar1.ax.tick_params(labelsize=CBAR_SIZE)
        # End if

        if len(levs_diff) < 2:
            img2, ax[2] = zonal_plot(alat, diff, ax=ax[2])
            ax[2].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[2].transAxes, bbox=props)
        else:
            img2, ax[2] = zonal_plot(
                alat,
                diff,
                ax=ax[2],
                norm=norm_diff,
                cmap=cmap_diff,
                levels=levels_diff,
                extend=extend_diff,
                **cp_info["contourf_opt"],
            )
            diffcb_mean_ax = inset_axes(
                ax[2],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[2].transAxes,
                borderpad=0,
            )
            diff_cbar = fig.colorbar(
                img2,
                cax=diffcb_mean_ax,
                location="right",
                **cp_info["diff_colorbar_opt"],
            )
            diff_cbar.ax.set_title(
                diff_units, fontsize=CBAR_SIZE, pad=cbar_labelpad_zonal, loc="left"
            )
            diff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

        if len(levs_pct_diff) < 2:
            img3, ax[3] = zonal_plot(alat, pct, ax=ax[3])
            ax[3].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[3].transAxes, bbox=props)
        else:
            img3, ax[3] = zonal_plot(
                alat,
                pct,
                ax=ax[3],
                norm=norm_pctdiff,
                cmap=cmap_pctdiff,
                levels=levels_pctdiff,
                extend=extend_pctdiff,
                **cp_info["contourf_opt"],
            )
            pctdiffcb_mean_ax = inset_axes(
                ax[3],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[3].transAxes,
                borderpad=0,
            )
            pctdiff_cbar = fig.colorbar(
                img3,
                cax=pctdiffcb_mean_ax,
                location="right",
                **cp_info["pct_colorbar_opt"],
            )
            pctdiff_cbar.ax.set_title(
                pct_units, fontsize=CBAR_SIZE, pad=cbar_labelpad_zonal, loc="left"
            )
            pctdiff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

        ax[0].set_title(case_title, loc="left", fontsize=ti_font_size)
        ax[1].set_title(base_title, loc="left", fontsize=ti_font_size)
        ax[2].set_title(
            r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=ti_font_size
        )
        ax[3].set_title(
            "Test % Diff Baseline", loc="left", fontsize=ti_font_size, fontweight="bold"
        )

        # style the plot:
        # Set Main title for subplots:
        var_name = kwargs["var_name"]
        st = fig.suptitle(
            f"{var_name}: {kwargs['season']}",
            fontsize=14,
            fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
        )
        st.set_y(0.85)
        ax[-1].set_xlabel("LATITUDE", fontsize=ti_font_size)

        for a in ax:
            a.tick_params("both", length=5, width=1.5, which="major")
            a.tick_params("both", length=5, width=1.5, which="minor")
            a.tick_params(axis="both", labelsize=7)
            a.xaxis.set_major_locator(MultipleLocator(20))  # every 20°
        if log_p:
            for a in ax:
                a.set_yscale("log")

        fig.text(-0.03, 0.5, "PRESSURE [hPa]", va="center", rotation="vertical")
    else:
        line = Line2D(
            [0], [0], label=r"$\mathbf{Test}:$" + case_nickname, color="#1f77b4"
        )  # #1f77b4 -> matplotlib standard blue
        line2 = Line2D(
            [0], [0], label=r"$\mathbf{Baseline}:$" + base_nickname, color="#ff7f0e"
        )  # #ff7f0e -> matplotlib standard orange
        byears = f"years: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        tyears = f"years: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"

        fig, ax = plt.subplots(nrows=3)
        ax[0].set_title(f"{tyears}\n{byears}", loc="right", fontsize=6)

        # Set Main title for subplots:
        var_name = kwargs["var_name"]
        st = fig.suptitle(
            f"{var_name}: {kwargs['season']}",
            fontsize=12,
            fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
        )
        st.set_y(1)

        zonal_plot(
            adata["lat"], azm, ax=ax[0], color="#1f77b4"
        )  # #1f77b4 -> matplotlib standard blue
        zonal_plot(
            bdata["lat"], bzm, ax=ax[0], color="#ff7f0e"
        )  # #ff7f0e -> matplotlib standard orange

        fig.legend(
            handles=[line, line2],
            bbox_to_anchor=(0.125, 0.84, 1.05, 0.102),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=6,
            frameon=False,
            labelspacing=0.3,
        )

        use_cmap = kwargs.get("colormap_2d", False)
        # One copy of kwargs, shared by the difference and percent difference plots:
        line_kwargs = kwargs.copy()
        line_kwargs["colormap_2d"] = use_cmap
        line_kwargs.pop("norm", None)
        line_kwargs["type"] = "diff"
        zonal_plot(
            adata["lat"],
            diff,
            ax=ax[1],
            norm=norm_diff,
            **line_kwargs,
        )
        ax[1].set_title(r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=6)
        ax[1].set_ylabel(units, fontsize=6)

        line_kwargs["type"] = "pctdiff"
        zonal_plot(
            adata["lat"],
            pct,
            ax=ax[2],
            norm=norm_pctdiff,
            **line_kwargs,
        )
        ax[2].set_title(
            "Test % Diff Baseline", loc="left", fontsize=6, fontweight="bold"
        )
        ax[2].set_ylabel(pct_units, fontsize=6)

        ax[-1].set_xlabel("LATITUDE", fontsize=6)

        for a in ax:
            a.tick_params("both", length=3, width=1, which="major")
            a.tick_params("both", length=3, width=1, which="minor")
            a.tick_params(axis="both", labelsize=6)
            a.xaxis.set_major_locator(MultipleLocator(20))  # every 20°
            a.yaxis.set_major_locator(MaxNLocator(nbins=6))
            a.grid(True)
        # End for

        plt.subplots_adjust(wspace=0.01, hspace=0.5, right=0.85)
        if use_cmap:
            # Create colorbar axes (same width for both)
            cax1 = fig.add_axes(
                [0.86, 0.4, 0.02, 0.1925]
            )  # [left, bottom, width, height]
            cax2 = fig.add_axes([0.86, 0.11, 0.02, 0.1925])
            ax[1].set_facecolor("0.5")  # dark gray
            ax[2].set_facecolor("0.5")

            diff_cbar = fig.colorbar(
                ax[1].adf_linecollection, cax=cax1, **cp_info["diff_colorbar_opt"]
            )
            diff_cbar.ax.tick_params(labelsize=6)
            pctdiff_cbar = fig.colorbar(
                ax[2].adf_linecollection, cax=cax2, **cp_info["pct_colorbar_opt"]
            )
            pctdiff_cbar.ax.tick_params(labelsize=6)
    # End if

    # Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches="tight", dpi=300)

    # Close plots:
    plt.close()


#######


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def plot_meridional_mean_and_save(
    adfobj,
    wks,
    case_nickname,
    base_nickname,
    case_climo_yrs,
    baseline_climo_yrs,
    adata,
    bdata,
    has_lev,
    log_p=False,
    latbounds=None,
    obs=False,
    **kwargs,
):
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
        use log(pressure) vertical axis
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
        - mpl -> dict, This should be any matplotlib kwargs that should be passed along.
          Keep reading:
            + Organize these by the mpl function. In this function
            (`plot_meridional_mean_and_save`)
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
    kwargs["adfobj"] = adfobj

    # apply averaging:
    import numbers  # built-in; just checking on the latbounds input

    if latbounds is None:
        latbounds = slice(-5, 5)
    elif isinstance(latbounds, numbers.Number):
        latbounds = slice(-1 * np.absolute(latbounds), np.absolute(latbounds))
    elif not isinstance(
        latbounds, slice
    ):  # If not a slice object, then quit this routine.
        print(
            "ERROR: plot_meridonal_mean_and_save - received an invalid value for "
            f"latbounds ({latbounds}). Must be a number or a slice."
        )
        return None
    # End if

    # style the plot:
    # We should think about how to do plot customization and defaults.
    # Here I'll just pop off a few custom ones, and then pass the rest into mpl.
    ti_font_size = kwargs.pop("tiFontSize", 8)

    # possible that the data has time, but usually it won't
    if len(adata.dims) > 4:
        print(
            f"ERROR: plot_meridonal_mean_and_save - too many dimensions: {adata.dims}"
        )
        return None

    if "time" in adata.dims:
        adata = adata.mean(dim="time", keep_attrs=True)
    if "lat" in adata.dims:
        adata = _lat_band(adata, latbounds)
        latweight = np.cos(np.radians(adata.lat))
        adata = adata.weighted(latweight).mean(dim="lat", keep_attrs=True)
    if "time" in bdata.dims:
        # bdata, not adata: this took the reference's time mean and put it in
        # the test case, throwing away the latitude weighting just applied to
        # it.  Latent, because seasonal_mean(is_climo=True) has already
        # removed 'time' by the time the ADF calls this.
        bdata = bdata.mean(dim="time", keep_attrs=True)
    if "lat" in bdata.dims:
        bdata = _lat_band(bdata, latbounds)
        latweight = np.cos(np.radians(bdata.lat))
        bdata = bdata.weighted(latweight).mean(dim="lat", keep_attrs=True)

    if "unstructured_plotting" in kwargs:
        unstructured = kwargs["unstructured_plotting"]
    else:
        unstructured = False

    if unstructured:
        # Only the cells whose centres lie within the latitude bounds; the
        # longitude binning then averages over them:
        adata = ux_to_lon_binned(_lat_band(adata, latbounds))
        bdata = ux_to_lon_binned(_lat_band(bdata, latbounds))

    adata = meridional_mean_xr(adata)
    bdata = meridional_mean_xr(bdata)

    # If there are other dimensions, they are still going to be there:
    if len(adata.dims) > 2:
        print(
            "ERROR: plot_meridonal_mean_and_save - AFTER averaging, there are too "
            f"many dimensions: {adata.dims}"
        )
        return None

    # plot-controlling parameters:
    xdim = "lon"  # the name used for the x-axis dimension
    # the plotting function ... maybe we can generalize to get zonal/meridional
    # into one function (?)
    pltfunc = meridional_plot

    diff = adata - bdata

    # calculate the percent change
    pct = (adata - bdata) / np.abs(bdata) * 100.0
    # Mask NaN's and Inf's (e.g. division by zero), so they are not plotted:
    pct = pct.where(np.isfinite(pct))

    # generate dictionary of contour plot settings:
    cp_info = plot_utils.prep_contour_plot(adata, bdata, diff, pct, **kwargs)
    units = cp_info["units_label"]
    diff_units = cp_info["diff_units_label"]
    pct_units = cp_info["pct_units_label"]

    levels_diff = cp_info["levels_diff"]
    cmap_diff = cp_info["cmap_diff"]
    norm_diff = cp_info["norm_diff"]
    extend_diff = cp_info["extend_diff"]

    levels_pctdiff = cp_info["levels_pctdiff"]
    cmap_pctdiff = cp_info["cmap_pctdiff"]
    norm_pctdiff = cp_info["norm_pctdiff"]
    extend_pctdiff = cp_info["extend_pctdiff"]

    levels_sim = cp_info["levels_sim"]
    cmap_sim = cp_info["cmap_sim"]
    norm_sim = cp_info["norm_sim"]
    extend_sim = cp_info["extend_sim"]

    # Set plot titles
    case_title = (
        r"$\mathbf{Test}:$"
        + f"{case_nickname}\nyears: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"
    )

    if obs:
        obs_var = kwargs["obs_var_name"]
        obs_title = kwargs["obs_file"][:-3]
        base_title = (
            r"$\mathbf{Baseline}:$"
            + obs_title
            + "\n"
            + r"$\mathbf{Variable}:$"
            + f"{obs_var}"
        )
    else:
        base_title = (
            r"$\mathbf{Baseline}:$"
            + f"{base_nickname}\nyears: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        )
    if has_lev:

        # Generate meridional plot:
        fig, ax = plt.subplots(
            figsize=(8, 10),
            nrows=4,
            constrained_layout=True,
            sharey=True,
            **cp_info["subplots_opt"],
        )

        contour_levs = np.unique(np.array(levels_sim))
        levs_diff = np.unique(np.array(levels_diff))
        levs_pct_diff = np.unique(np.array(levels_pctdiff))
        cbar_labelpad_merd = 5
        if len(contour_levs) < 2:
            img0, ax[0] = pltfunc(adata[xdim], adata, ax=ax[0])
            ax[0].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[0].transAxes, bbox=props)
            img1, ax[1] = pltfunc(bdata[xdim], bdata, ax=ax[1])
            ax[1].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[1].transAxes, bbox=props)
        else:
            img0, ax[0] = pltfunc(
                adata[xdim],
                adata,
                ax=ax[0],
                norm=norm_sim,
                cmap=cmap_sim,
                levels=levels_sim,
                extend=extend_sim,
                **cp_info["contourf_opt"],
            )
            img1, ax[1] = pltfunc(
                bdata[xdim],
                bdata,
                ax=ax[1],
                norm=norm_sim,
                cmap=cmap_sim,
                levels=levels_sim,
                extend=extend_sim,
                **cp_info["contourf_opt"],
            )

            cb_mean_ax = inset_axes(
                ax[0],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[0].transAxes,
                borderpad=0,
            )
            cbar0 = fig.colorbar(
                img0, cax=cb_mean_ax, location="right", **cp_info["colorbar_opt"]
            )
            cbar0.ax.set_title(
                units, fontsize=CBAR_SIZE, pad=cbar_labelpad_merd, loc="left"
            )
            cbar0.ax.tick_params(labelsize=CBAR_SIZE)

            cb_mean_ax = inset_axes(
                ax[1],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[1].transAxes,
                borderpad=0,
            )
            cbar1 = fig.colorbar(
                img1, cax=cb_mean_ax, location="right", **cp_info["colorbar_opt"]
            )
            cbar1.ax.set_title(
                units, fontsize=CBAR_SIZE, pad=cbar_labelpad_merd, loc="left"
            )
            cbar1.ax.tick_params(labelsize=CBAR_SIZE)
        # End if

        if len(levs_diff) < 2:
            img2, ax[2] = pltfunc(adata[xdim], diff, ax=ax[2])
            ax[2].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[2].transAxes, bbox=props)
        else:
            img2, ax[2] = pltfunc(
                adata[xdim],
                diff,
                ax=ax[2],
                norm=norm_diff,
                cmap=cmap_diff,
                levels=levels_diff,
                extend=extend_diff,
                **cp_info["contourf_opt"],
            )
            diffcb_mean_ax = inset_axes(
                ax[2],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[2].transAxes,
                borderpad=0,
            )
            diff_cbar = fig.colorbar(
                img2,
                cax=diffcb_mean_ax,
                location="right",
                **cp_info["diff_colorbar_opt"],
            )
            diff_cbar.ax.set_title(
                diff_units, fontsize=CBAR_SIZE, pad=cbar_labelpad_merd, loc="left"
            )
            diff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

        if len(levs_pct_diff) < 2:
            img3, ax[3] = pltfunc(adata[xdim], pct, ax=ax[3])
            ax[3].text(0.4, 0.4, EMPTY_MESSAGE, transform=ax[3].transAxes, bbox=props)
        else:
            img3, ax[3] = pltfunc(
                adata[xdim],
                pct,
                ax=ax[3],
                norm=norm_pctdiff,
                cmap=cmap_pctdiff,
                levels=levels_pctdiff,
                extend=extend_pctdiff,
                **cp_info["contourf_opt"],
            )
            pctdiffcb_mean_ax = inset_axes(
                ax[3],
                width="2%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(1.02, 0, 1, 1),
                bbox_transform=ax[3].transAxes,
                borderpad=0,
            )
            pctdiff_cbar = fig.colorbar(
                img3,
                cax=pctdiffcb_mean_ax,
                location="right",
                **cp_info["pct_colorbar_opt"],
            )
            pctdiff_cbar.ax.set_title(
                pct_units, fontsize=CBAR_SIZE, pad=cbar_labelpad_merd, loc="left"
            )
            pctdiff_cbar.ax.tick_params(labelsize=CBAR_SIZE)

        ax[0].set_title(case_title, loc="left", fontsize=ti_font_size)
        ax[1].set_title(base_title, loc="left", fontsize=ti_font_size)
        ax[2].set_title(
            r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=ti_font_size
        )
        ax[3].set_title(
            "Test % Diff Baseline", loc="left", fontsize=ti_font_size, fontweight="bold"
        )

        # style the plot:
        # Set Main title for subplots:
        var_name = kwargs["var_name"]
        st = fig.suptitle(
            f"{var_name}: {kwargs['season']}",
            fontsize=14,
            fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
        )
        st.set_y(0.85)
        ax[-1].set_xlabel("LONGITUDE", fontsize=ti_font_size)

        for a in ax:
            a.tick_params("both", length=5, width=1.5, which="major")
            a.tick_params("both", length=5, width=1.5, which="minor")
            a.tick_params(axis="both", labelsize=7)
            a.xaxis.set_major_locator(MultipleLocator(30))  # every 30°
        if log_p:
            for a in ax:
                a.set_yscale("log")

        fig.text(-0.03, 0.5, "PRESSURE [hPa]", va="center", rotation="vertical")
    else:
        line = Line2D(
            [0], [0], label=r"$\mathbf{Test}:$" + case_nickname, color="#1f77b4"
        )  # #1f77b4 -> matplotlib standard blue
        line2 = Line2D(
            [0], [0], label=r"$\mathbf{Baseline}:$" + base_nickname, color="#ff7f0e"
        )  # #ff7f0e -> matplotlib standard orange
        byears = f"years: {baseline_climo_yrs[0]}-{baseline_climo_yrs[-1]}"
        tyears = f"years: {case_climo_yrs[0]}-{case_climo_yrs[-1]}"

        fig, ax = plt.subplots(nrows=3)
        ax[0].set_title(f"{tyears}\n{byears}", loc="right", fontsize=6)

        # Set Main title for subplots:
        var_name = kwargs["var_name"]
        st = fig.suptitle(
            f"{var_name}: {kwargs['season']}",
            fontsize=12,
            fontfamily=["DejaVu Sans", "Liberation Sans", "sans-serif"],
        )
        st.set_y(1)

        pltfunc(
            adata[xdim], adata, ax=ax[0], color="#1f77b4"
        )  # #1f77b4 -> matplotlib standard blue
        pltfunc(
            bdata[xdim], bdata, ax=ax[0], color="#ff7f0e"
        )  # #ff7f0e -> matplotlib standard orange

        fig.legend(
            handles=[line, line2],
            bbox_to_anchor=(0.125, 0.84, 1.05, 0.102),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=6,
            frameon=False,
            labelspacing=0.3,
        )

        use_cmap = kwargs.get("colormap_2d", False)
        # One copy of kwargs, shared by the difference and percent difference plots:
        line_kwargs = kwargs.copy()
        line_kwargs["colormap_2d"] = use_cmap
        line_kwargs.pop("norm", None)
        line_kwargs["type"] = "diff"
        pltfunc(
            adata[xdim],
            diff,
            ax=ax[1],
            norm=norm_diff,
            **line_kwargs,
        )

        ax[1].set_title(r"$\mathbf{Test} - \mathbf{Baseline}$", loc="left", fontsize=6)
        ax[1].set_ylabel(units, fontsize=6)

        line_kwargs["type"] = "pctdiff"
        pltfunc(
            adata[xdim],
            pct,
            ax=ax[2],
            norm=norm_pctdiff,
            **line_kwargs,
        )

        ax[2].set_title(
            "Test % Diff Baseline", loc="left", fontsize=6, fontweight="bold"
        )
        ax[2].set_ylabel(pct_units, fontsize=6)

        ax[-1].set_xlabel("LONGITUDE", fontsize=6)

        for a in ax:
            a.tick_params("both", length=3, width=1, which="major")
            a.tick_params("both", length=3, width=1, which="minor")
            a.tick_params(axis="both", labelsize=6)
            a.xaxis.set_major_locator(MaxNLocator(nbins=12))
            a.yaxis.set_major_locator(MaxNLocator(nbins=6))
            a.grid(True)
        # End for

        plt.subplots_adjust(wspace=0.01, hspace=0.5, right=0.85)

        if use_cmap:
            # Create colorbar axes (same width for both)
            cax1 = fig.add_axes(
                [0.86, 0.4, 0.02, 0.1925]
            )  # [left, bottom, width, height]
            cax2 = fig.add_axes([0.86, 0.11, 0.02, 0.1925])
            ax[1].set_facecolor("0.5")  # dark gray
            ax[2].set_facecolor("0.5")

            diff_cbar = fig.colorbar(
                ax[1].adf_linecollection, cax=cax1, **cp_info["diff_colorbar_opt"]
            )
            diff_cbar.ax.tick_params(labelsize=6)
            pctdiff_cbar = fig.colorbar(
                ax[2].adf_linecollection, cax=cax2, **cp_info["pct_colorbar_opt"]
            )
            pctdiff_cbar.ax.tick_params(labelsize=6)
    # End if

    # Write the figure to provided workspace/file:
    fig.savefig(wks, bbox_inches="tight", dpi=300)

    # Close plots:
    plt.close()

    return None


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

    assert (
        len(fld1.shape) == 2
    ), "Input fields must have exactly two dimensions."  # 2-Dimension => plot contourf
    assert (
        fld1.shape == fld2.shape
    ), "Input fields must have the same array shape."  # Same shape => allows difference

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
    fig = plt.figure(figsize=(10, 10))

    rows = 5
    columns = 5
    grid = mpl.gridspec.GridSpec(
        rows,
        columns,
        wspace=1,
        hspace=1,
        width_ratios=[1, 1, 1, 1, 0.2],
        height_ratios=[1, 1, 1, 1, 0.2],
    )
    # plt.subplots_adjust(wspace= 0.01, hspace= 0.01)
    ax1 = plt.subplot(grid[0:2, 0:2])
    ax2 = plt.subplot(grid[0:2, 2:4])
    ax3 = plt.subplot(grid[0:2, 0:2])
    ax4 = plt.subplot(grid[0:2, 2:4])
    # color bars / means share top bar.
    cbax_top = plt.subplot(grid[0:2, -1])
    cbax_bot = plt.subplot(grid[-1, 1:3])

    coord1, coord2 = fld1.coords  # ASSUMES xarray WITH coords AND 2-dimensions
    xx, yy = np.meshgrid(fld1[coord2], fld1[coord1])

    img1 = ax1.contourf(xx, yy, fld1.transpose())
    if (coord1 == "month") and (fld1.shape[0] == 12):
        ax1.set_xticks(np.arange(1, 13))
    ax1.set_ylabel(coord2.upper())
    ax1.set_xlabel(coord1.upper())
    ax1.set_title(
        f"{case1name}\nMIN:{fld1.min().values:.2f}  MAX:{fld1.max().values:.2f}"
    )

    ax2.contourf(xx, yy, fld2.transpose())
    if (coord1 == "month") and (fld1.shape[0] == 12):
        ax2.set_xticks(np.arange(1, 13))
    ax2.set_xlabel(coord1.upper())
    ax2.set_title(
        f"{case2name}\nMIN:{fld2.min().values:.2f}  MAX:{fld2.max().values:.2f}"
    )

    diff = fld1 - fld2

    pct = (fld1 - fld2) / np.abs(fld2) * 100.0
    # Mask NaN's and Inf's (e.g. division by zero), so they are not plotted:
    pct = pct.where(np.isfinite(pct))

    ## USE A DIVERGING COLORMAP CENTERED AT ZERO
    ## Special case is when min > 0 or max < 0
    dmin = diff.min()
    dmax = diff.max()
    if dmin > 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.colormaps["OrRd"]
    elif dmax < 0:
        dnorm = mpl.colors.Normalize(dmin, dmax)
        cmap = mpl.colormaps["BuPu_r"]
    else:
        dnorm = mpl.colors.TwoSlopeNorm(vmin=dmin, vcenter=0, vmax=dmax)
        cmap = mpl.colormaps["RdBu_r"]

    img3 = ax3.contourf(xx, yy, diff.transpose(), cmap=cmap, norm=dnorm)
    if (coord1 == "month") and (fld1.shape[0] == 12):
        ax3.set_xticks(np.arange(1, 13))
    ax3.set_ylabel(coord2.upper())
    ax3.set_xlabel(coord1.upper())
    ax3.set_title(
        f"DIFFERENCE (= a - b)\nMIN:{diff.min().values:.2f}  MAX:{diff.max().values:.2f}"
    )

    ## USE A DIVERGING COLORMAP CENTERED AT ZERO
    ## Special case is when min > 0 or max < 0
    pmin = pct.min()
    pmax = pct.max()
    if pmin > 0:
        pnorm = mpl.colors.Normalize(pmin, pmax)
        cmap = mpl.colormaps["OrRd"]
    elif pmax < 0:
        pnorm = mpl.colors.Normalize(pmin, pmax)
        cmap = mpl.colormaps["BuPu_r"]
    else:
        pnorm = mpl.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
        cmap = mpl.colormaps["RdBu_r"]

    img4 = ax4.contourf(xx, yy, pct.transpose(), cmap=cmap, norm=pnorm)
    if (coord1 == "month") and (fld1.shape[0] == 12):
        ax4.set_xticks(np.arange(1, 13))
    ax4.set_ylabel(coord2.upper())
    ax4.set_xlabel(coord1.upper())
    ax4.set_title(
        f"PCT DIFFERENCE (= a % diff b)\nMIN:{pct.min().values:.2f}  MAX:{pct.max().values:.2f}"
    )

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

    fig.colorbar(img1, cax=cbax_top)
    fig.colorbar(img3, cax=cbax_bot, orientation="horizontal")
    fig.colorbar(img4, cax=cbax_bot, orientation="horizontal")
    return fig


#####################
# END HELPER FUNCTIONS
