"""Generic plotting helper functions.

Functions
---------
first_complete_plot_set(layouts)
    pick the set of plots that is already on disk, if there is one
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
try_load_ncl_cmap(adfobj, cmap_name, fallback="viridis")
    Load an NCL colormap, from lib/plotting/colormaps/NCL or else NCL's website.
read_ncl_colormap(adfobj, fil) / parse_ncl_colormap(lines)
    Read the RGB values of an NCL colormap file.
download_ncl_colormap(adfobj, cmap_name)
    Download an NCL colormap that is not shipped with the ADF.
ncl_to_mpl(adfobj, nclmap, name)
    Turn an NCL colormap table into a matplotlib colormap and its reverse.
get_cmap(adfobj, plotty, plot_type_dict, kwargs, polar_names)
    Pick the case or difference colormap from the variable defaults.
resolve_hemi_level(adfobj, data, kwargs, polar_names)
    Pick the hemisphere and/or pressure level specific value of a setting.
resolve_levels(adfobj, plotty, plot_type_dict, kwargs, polar_names)
    Pick the case or difference contour levels from the variable defaults.
colormap_object(cmap)
    Return a matplotlib Colormap, given either one or the name of one.
prep_contour_plot(adata, bdata, diffdata, pctdata, **kwargs)
    Colormaps, contour levels and color normalization for contour plots.
add_var_name_key_to_vres(adfobj, var, vres, vector=False)
    Copy of a variable's defaults with the name to title its plots with.

Notes
-----
This module includes "private" methods intended for internal use only.

_plot_line(axobject, xdata, ydata, color, **kwargs)
    Create a generic line plot
"""

# This module collects many plotting helpers, so allow it to be long:
# pylint: disable=too-many-lines

# ++++++++++++++++++++++++++++++
# Import standard python modules
# ++++++++++++++++++++++++++++++

import os
import re
from pathlib import Path
from urllib.request import urlopen
import warnings

# +++++++++++++++++++++++++++++++++++++++++++++++++
# import non-standard python modules, including ADF
# +++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import uxarray as ux  # need npl 2024a or later

from adf_diag import AdfDiag
import adf_utils as utils

# use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

# Set non-X-window backend for matplotlib:
mpl.use("Agg")

# Now import pyplot (must come after the backend is set above):
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-order,ungrouped-imports

script_name = os.path.splitext(os.path.basename(__file__))[0]

_LINE2D_PROPS = set(mlines.Line2D([0], [0]).properties().keys())

#################
# HELPER FUNCTIONS
#################


def load_dataset(fils):
    """
    This method exists to get an xarray Dataset from input file information that
    can be passed into the plotting methods.

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
        warnings.warn("\t    WARNING: Input file list is empty.")
        return None
    if len(fils) > 1:
        return xr.open_mfdataset(fils, combine="by_coords")
    return xr.open_dataset(fils[0])


# End def


#######


def first_complete_plot_set(layouts):
    """
    Return the first set of plots whose files all exist.

    A plotting script can usually tell what it would produce from the
    variable defaults and the configured seasons and pressure levels, without
    opening any data -- except that a 2-D variable and a 3-D one are named
    differently, and which one it has is a property of the data.  Both
    spellings can be checked instead: if either set is complete on disk, the
    script knows both that nothing needs drawing and which kind of variable it
    was looking at, and never has to open the file.

    Parameters
    ----------
    layouts : list
        candidate sets, most specific first.  Each is a list of entries whose
        first element is the plot's path; the rest is whatever the caller
        needs to register the plot on the website.

    Returns
    -------
    list
        The first complete set, or an empty list when none is complete, which
        means the data has to be opened to find out what is missing.
    """
    for entries in layouts:
        if entries and all(Path(entry[0]).is_file() for entry in entries):
            return entries
        # End if
    # End for
    return []


def use_this_norm():
    """Just use the right normalization; avoids a deprecation warning."""

    mplversion = [int(x) for x in mpl.__version__.split(".")]
    if mplversion[0] < 3:
        return mpl.colors.Normalize, mplversion[0]
    if mplversion[1] < 2:
        # Only exists in old matplotlib versions:
        return mpl.colors.DivergingNorm, mplversion[0]  # pylint: disable=no-member
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

    The source CRS is ``ccrs.Geodetic()`` (plain longitude/latitude) rather
    than ``ccrs.PlateCarree()``.  Cartopy defines PlateCarree as PROJ's
    ellipsoidal ``eqc`` projection, and from PROJ 9.8 that shifts latitudes by
    up to ~0.2 degrees and returns NaN within ~0.15 degrees of the poles,
    which breaks ``contourf`` on grids that include latitude +/-90.
    """
    lons, lats = np.meshgrid(lon, lat)
    x_proj, y_proj, _ = proj.transform_points(
        ccrs.Geodetic(), lons, lats
    ).T  # .T to unpack, .T again to get x,y,z arrays
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
    normfunc, _ = use_this_norm()
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        # Nothing to take a range from: a pressure level above the model top,
        # or a field that is missing everywhere.  Hand back a usable norm so
        # the caller draws an empty panel instead of raising -- the old code
        # left 'cmap' unset here and died with an UnboundLocalError.  Note
        # np.min/np.max were used before, so a single missing point anywhere in
        # the field was enough to reach this.
        return mpl.colors.Normalize(vmin=-1.0, vmax=1.0), mpl.colormaps["RdBu_r"]
    dmin = finite.min()
    dmax = finite.max()
    # color normalization for difference
    cmap = mpl.colormaps["RdBu_r"]
    if dmin < 0 < dmax:
        dnorm = normfunc(vmin=dmin, vmax=dmax, vcenter=0.0)
    else:
        dnorm = mpl.colors.Normalize(vmin=dmin, vmax=dmax)
        if dmin >= 0:
            cmap = mpl.colormaps["OrRd"]
        elif dmax <= 0:
            cmap = mpl.colormaps["BuPu_r"]
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
    This allows a script to, for example, allow a config file to specify, but also
    have a preference:
    `get_central_longitude(AdfObj, 30.0)`
    """
    chk_for_adf = [isinstance(arg, AdfDiag) for arg in args]
    # preference is to get value from AdfDiag:
    if any(chk_for_adf):
        for arg in args:
            if isinstance(arg, AdfDiag):
                result = arg.get_basic_info("central_longitude", required=False)
                if isinstance(result, (int, float)) and -180 <= result <= 360:
                    return result
                # End if

                # If result exists, then write info to debug log:
                if result:
                    msg = f"central_lngitude of type '{type(result).__name__}'"
                    msg += f" and value '{result}', which is not a valid longitude"
                    msg += " for the ADF."
                    arg.debug_log(msg)
                # End if

                # There is only one ADF object per ADF run, so if its
                # not present or configured correctly then no
                # reason to keep looking:
                break
            # End if
        # End for
    # End if

    # 2nd pass through arguments, look for numbers:
    for arg in args:
        if isinstance(arg, (float, int)) and -180 <= arg <= 360:
            return arg
        # End if
    # End for

    # None of the arguments meet the criteria:
    print("No valid central longitude specified. Defaults to 180.")
    return 180


#######

#
#  -- zonal & meridional mean code --
#


def _plot_line(axobject, xdata, ydata, color=None, **kwargs):
    """Create a generic line plot and check for some ways to annotate."""

    use_cmap = kwargs.get("colormap_2d", False)
    cmap = kwargs.get("cmap", "viridis")
    norm = kwargs.get("norm", None)
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    if use_cmap:
        if kwargs["type"] == "pctdiff":
            cmap = kwargs.get("pct_diff_colormap", "PuOr_r")
        if kwargs["type"] == "diff":
            cmap = kwargs.get("diff_colormap", "RdBu_r")

        from matplotlib.collections import LineCollection

        x = np.asarray(xdata)
        y = np.asarray(ydata)

        # pylint: disable-next=too-many-function-args
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=cmap)

        if norm is not None:
            lc.set_norm(norm)
        else:
            if vmin is None:
                vmin = np.nanmin(y)
            if vmax is None:
                vmax = np.nanmax(y)
            lc.set_clim(vmin, vmax)

        lc.set_array(y)

        axobject.add_collection(lc)
        axobject.autoscale()

        axobject.adf_linecollection = lc

    else:
        # Only keep kwargs valid for Line2D / ax.plot()
        plot_kwargs = {k: v for k, v in kwargs.items() if k in _LINE2D_PROPS}

        if color is not None:
            axobject.plot(xdata, ydata, c=color, **plot_kwargs)
        else:
            axobject.plot(xdata, ydata, **plot_kwargs)

    # Y label logic
    if hasattr(ydata, "units"):
        axobject.set_ylabel(f"[{getattr(ydata, 'units')}]")
    elif "units" in kwargs:
        axobject.set_ylabel(f"[{kwargs['units']}]")
    # End if

    return axobject


def meridional_plot_line(ax, lon, data, color, **kwargs):
    """Create line plot with longitude as the X-axis."""

    ax = _plot_line(ax, lon, data, color, **kwargs)
    ax.set_xlim([lon.min(), lon.max()])
    #
    # annotate
    #
    if hasattr(data, "units"):
        ax.set_ylabel(f"{getattr(data, 'units')}", fontsize=6)
    elif "units" in kwargs:
        ax.set_ylabel(f"{kwargs['units']}")
    return ax


def zonal_plot_line(ax, lat, data, color, **kwargs):
    """Create line plot with latitude as the X-axis."""
    ax = _plot_line(ax, lat, data, color, **kwargs)
    ax.set_xlim([max([lat.min(), -90.0]), min([lat.max(), 90.0])])
    #
    # annotate
    #
    if hasattr(data, "units"):
        ax.set_ylabel(f"{getattr(data, 'units')}", fontsize=6)
    elif "units" in kwargs:
        ax.set_ylabel(f"{kwargs['units']}")
    return ax


def zonal_plot_preslat(ax, lat, lev, data, **kwargs):
    """Create plot with latitude as the X-axis, and pressure as the Y-axis."""
    mlev, mlat = np.meshgrid(lev, lat)
    if "cmap" in kwargs:
        cmap = kwargs.pop("cmap")
    else:
        cmap = "Spectral_r"

    img = ax.contourf(mlat, mlev, data.transpose("lat", "lev"), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which="minor", length=4, color="r")
    ax.set_ylim([np.max(lev), np.min(lev)])
    return img, ax


def meridional_plot_preslon(ax, lon, lev, data, **kwargs):
    """Create plot with longitude as the X-axis, and pressure as the Y-axis."""

    mlev, mlon = np.meshgrid(lev, lon)
    if "cmap" in kwargs:
        cmap = kwargs.pop("cmap")
    else:
        cmap = "Spectral_r"

    img = ax.contourf(mlon, mlev, data.transpose("lon", "lev"), cmap=cmap, **kwargs)

    minor_locator = mpl.ticker.FixedLocator(lev)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.tick_params(which="minor", length=4, color="r")
    ax.set_ylim([np.max(lev), np.min(lev)])
    return img, ax


# Color Map Functions
# --------------------

# NCL's color tables, shipped with the ADF so that runs (usually on compute
# nodes without internet access) never have to download them:
# https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml
NCL_COLORMAP_DIR = Path(__file__).resolve().parent / "plotting" / "colormaps" / "NCL"
NCL_COLORMAP_URL = "https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files"

# Seconds to wait for NCL's website before giving up on a download:
_NCL_DOWNLOAD_TIMEOUT = 30

# The first three numbers on a line are its red, green and blue values;
# anything after them (a fourth column, or a "#", ";" or "/* */" comment) is not:
_NCL_RGB_REGEX = re.compile(r"^\s*([-+]?[\d.]+)\s+([-+]?[\d.]+)\s+([-+]?[\d.]+)")

# NCL colormaps already loaded, or that could not be, so that each one is
# read (or downloaded) only once per run:
_ncl_cmap_cache = {}


def parse_ncl_colormap(lines):
    """
    Read the RGB values out of the lines of an NCL colormap file.

    Parameters
    ----------
    lines : iterable of str
        lines of an NCL ".rgb" file

    Returns
    -------
    numpy.ndarray or None
        (N, 3) array, one row per color, or ``None`` if there are no colors

    Notes
    -----
    NCL's files are not regular enough for ``read_csv``: headers vary in
    length, and some lines end with comments or a fourth column. Lines that do
    not start with three numbers are skipped, as are rows with negative values
    (NCL's placeholders for the background and foreground colors).
    """
    rows = []
    for line in lines:
        match = _NCL_RGB_REGEX.match(line)
        if match is None:
            continue
        try:
            row = [float(val) for val in match.groups()]
        except ValueError:
            continue  # e.g. "1.2.3"
        if min(row) < 0:
            continue
        rows.append(row)
    if not rows:
        return None
    return np.array(rows)


def read_ncl_colormap(adfobj, fil):
    """
    Read an NCL colormap file into an (N, 3) array of RGB values.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    fil : str or pathlib.Path
        the NCL ".rgb" file

    Returns
    -------
    numpy.ndarray or None
        one row per color, or ``None`` if the file has no colors in it
    """
    with open(fil, encoding="utf-8") as rgb_file:
        table = parse_ncl_colormap(rgb_file)
    if table is None:
        adfobj.debug_log(
            f"{script_name}: read_ncl_colormap()\n\tWARNING: no RGB values in {fil}"
        )
    return table


def download_ncl_colormap(adfobj, cmap_name):
    """
    Download an NCL colormap that is not one of the ones shipped with the ADF.

    The file is saved to ``NCL_COLORMAP_DIR`` when that is writable, so later
    runs can use it without downloading it again.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    cmap_name : str
        NCL colormap name

    Returns
    -------
    numpy.ndarray or None
        (N, 3) RGB values, or ``None`` if the file has no colors in it

    Raises
    ------
    OSError
        if NCL's website cannot be reached or has no such colormap
    """
    msg = f"{script_name}: download_ncl_colormap()"
    url = f"{NCL_COLORMAP_URL}/{cmap_name}.rgb"
    msg += f"\n\t{cmap_name} is not one of the NCL colormaps in {NCL_COLORMAP_DIR},"
    msg += f" downloading it from {url}"
    with urlopen(url, timeout=_NCL_DOWNLOAD_TIMEOUT) as response:
        text = response.read().decode("utf-8", errors="replace")
    table = parse_ncl_colormap(text.splitlines())
    if table is not None:
        try:
            (NCL_COLORMAP_DIR / f"{cmap_name}.rgb").write_text(text, encoding="utf-8")
            msg += f"\n\tSaved to {NCL_COLORMAP_DIR}"
        except OSError:
            msg += f"\n\tCould not save it to {NCL_COLORMAP_DIR}, using it for this run only"
    adfobj.debug_log(msg)
    return table


def ncl_to_mpl(adfobj, nclmap, name):
    """
    Turn an NCL colormap table into a matplotlib colormap and its reverse.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    nclmap : numpy.ndarray
        (N, 3) RGB values, either 0-255 or 0-1
    name : str
        name to give the colormap

    Returns
    -------
    tuple
        the colormap and its reversed version, or ``(None, None)`` if
        ``nclmap`` is not an (N, 3) table
    """
    msg = f"{script_name}: ncl_to_mpl()"
    vals = np.atleast_2d(np.asarray(nclmap, dtype=float))
    if vals.ndim != 2 or vals.shape[1] != 3:
        msg += f"\n\tWARNING: {name} values should be (N, 3), not {vals.shape}"
        adfobj.debug_log(msg)
        return None, None
    if vals.max() > 1:
        vals = vals / 255
    else:
        msg += f"\n\t{name} seems to be 0-1"
    if vals.shape[0] > 100:
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, vals)
    else:
        my_cmap = mpl.colors.ListedColormap(vals, name)
    adfobj.debug_log(msg)
    return my_cmap, my_cmap.reversed()


def try_load_ncl_cmap(adfobj, cmap_name, fallback="viridis"):
    """
    Load an NCL colormap by name.

    Uses the copy shipped in ``NCL_COLORMAP_DIR``, and only downloads it from
    NCL's website when it is not there.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    cmap_name : str
        NCL colormap name (the file name without ".rgb"); a "_r" ending
        gives the reversed colormap
    fallback : str, optional
        matplotlib colormap to use when the NCL one cannot be loaded

    Returns
    -------
    tuple
        the colormap (or ``fallback``) and the debug message
    """
    msg = f"{script_name}: try_load_ncl_cmap()"
    msg += f"\n\tTrying {cmap_name} as an NCL color map:"

    if cmap_name not in _ncl_cmap_cache:
        base_name = cmap_name
        reverse = False
        local_file = NCL_COLORMAP_DIR / f"{cmap_name}.rgb"
        if not local_file.is_file() and cmap_name.endswith("_r"):
            base_name = cmap_name[:-2]
            reverse = True
            local_file = NCL_COLORMAP_DIR / f"{base_name}.rgb"
        cmap = None
        try:
            if local_file.is_file():
                data = read_ncl_colormap(adfobj, local_file)
            else:
                base_name = cmap_name
                reverse = False
                data = download_ncl_colormap(adfobj, cmap_name)
            if data is not None:
                cmap, cmap_r = ncl_to_mpl(adfobj, data, base_name)
                if reverse:
                    cmap = cmap_r
        # Download, read or conversion failures all mean using the fallback:
        except Exception as err:  # pylint: disable=broad-exception-caught
            msg += f"\n\tCould not load NCL colormap {cmap_name}: {err}"
        # Remember failures too, so a missing colormap is not retried
        # (or re-downloaded) for every variable and season:
        _ncl_cmap_cache[cmap_name] = cmap

    cmap = _ncl_cmap_cache[cmap_name]
    if cmap is None:
        msg += f"\n\tWARNING: {cmap_name} is not a matplotlib or NCL colormap,"
        msg += f" defaulting to '{fallback}'."
        cmap = fallback
    adfobj.debug_log(msg)
    return cmap, msg


def get_cmap(adfobj, plotty, plot_type_dict, kwargs, polar_names):
    """
    Gather colormap from variable defaults file, if applicable.

    Names that are not matplotlib colormaps are looked up as NCL colormaps.
    Falls back to 'viridis' (case) or 'BrBG' (diff) if none is found.
    """

    key_map = {
        "diff": ("diff_colormap", "BrBG"),
        "case": ("colormap", "viridis"),
    }
    colormap_key, default_cmap = key_map.get(plotty, ("colormap", "viridis"))

    cmap_case = None

    msg = f"{script_name}: get_cmap()"

    # Priority 1: YAML dict
    if colormap_key in plot_type_dict:
        cmap_entry = plot_type_dict[colormap_key]
        msg += f"\n\tUser supplied cmap for {plotty}: {cmap_entry}"

        if isinstance(cmap_entry, str):
            cmap_case = cmap_entry
        elif isinstance(cmap_entry, dict):
            resolved = resolve_hemi_level(adfobj, cmap_entry, kwargs, polar_names)
            if isinstance(resolved, str):
                cmap_case = resolved

    # Priority 2: kwargs dict
    elif colormap_key in kwargs and isinstance(kwargs[colormap_key], str):
        cmap_case = kwargs[colormap_key]
        msg += f"\n\tUser supplied cmap for {plotty}: {cmap_case}"

    # Fallback default
    if not cmap_case:
        msg += f"\n\tNo cmap for {plotty} found, defaulting to {default_cmap}"
        cmap_case = default_cmap

    # Not a matplotlib colormap: try NCL, which falls back to the default
    elif cmap_case not in plt.colormaps():
        cmap_case, ncl_msg = try_load_ncl_cmap(adfobj, cmap_case, fallback=default_cmap)
        msg += ncl_msg

    adfobj.debug_log(msg)

    return cmap_case


# Contour Plot Prep Functions
# ----------------------------
def resolve_hemi_level(adfobj, data, kwargs, polar_names):
    """Resolve hemisphere and/or vertical level specific values from a dict."""
    msg = f"{script_name}: resolve_hemi_level()"
    hemi = kwargs.get("hemi")
    lev = kwargs.get("lev")

    if hemi and polar_names.get(hemi) in data:
        hemi_data = data[polar_names[hemi]]
        if isinstance(hemi_data, dict) and lev in hemi_data:
            msg += f"\n\tPolar {hemi} with vertical level {lev}"
            adfobj.debug_log(msg)
            return hemi_data[lev]
        msg += f"\n\tPolar {hemi} without vertical levels"
        adfobj.debug_log(msg)
        return hemi_data
    if lev and lev in data:
        msg += f"\n\tVertical level {lev}"
        adfobj.debug_log(msg)
        return data[lev]

    adfobj.debug_log(msg)
    return None


def resolve_levels(adfobj, plotty, plot_type_dict, kwargs, polar_names):
    """Resolve contour levels based on user input and defaults."""
    levels_sim = None
    msg = f"{script_name}: resolve_levels()"

    # Map keys based on plot type
    key_map = {
        "diff": ("diff_contour_levels", "diff_contour_range", "diff_contour_linspace"),
        "case": ("contour_levels", "contour_levels_range", "contour_levels_linspace"),
    }
    contour_levels, contour_range, contour_linspace = key_map.get(
        plotty, (None, None, None)
    )

    # pylint: disable-next=too-many-return-statements
    def create_contour_levels(entry, kind):
        """
        Turn one levels/range/linspace setting into contour levels.

        ``entry`` is a list, or a dict of hemisphere and/or vertical level
        specific lists. Explicit levels are used as given; a range or
        linspace needs exactly three values (start, stop, step/num).
        Returns ``None`` when the setting cannot be used.
        """
        if isinstance(entry, dict):
            entry = resolve_hemi_level(adfobj, entry, kwargs, polar_names)
        if entry is None:
            return None
        if not isinstance(entry, (list, tuple, np.ndarray)):
            adfobj.debug_log(
                f"{msg}\n\tWARNING: contour {kind} for {plotty} should be a list,"
                f" not {entry!r}. Will get contours from data range."
            )
            return None
        if kind == "levels":
            if len(entry) < 2:
                adfobj.debug_log(
                    f"{msg}\n\tWARNING: contour levels for {plotty} need at least"
                    f" two values, not {entry}. Will get contours from data range."
                )
                return None
            return entry
        if len(entry) != 3:
            adfobj.debug_log(
                f"{msg}\n\tWARNING: contour {kind} for {plotty} needs exactly three"
                f" values, not {entry}. Will get contours from data range."
            )
            return None
        adfobj.debug_log(f"{msg}\n\tLevels specified for {plotty}: numpy.{kind}.")
        if kind == "range":
            return np.arange(*entry)
        return np.linspace(*entry)

    # Priority: explicit contour levels → range → linspace
    for key, kind in [
        (contour_levels, "levels"),
        (contour_range, "range"),
        (contour_linspace, "linspace"),
    ]:
        entry = None
        if key in plot_type_dict:
            entry = plot_type_dict[key]
        elif key in kwargs:
            entry = kwargs[key]

        if entry is not None:
            levels_sim = create_contour_levels(entry, kind)
            if levels_sim is not None:
                break  # stop once a valid setting is found
    adfobj.debug_log(msg)
    return levels_sim


def colormap_object(cmap):
    """
    Return a matplotlib Colormap, given either one or the name of one.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        a colormap, or the name the variable defaults give for one

    Returns
    -------
    matplotlib.colors.Colormap

    Notes
    -----
    Replaces the lookup matplotlib removed in 3.9, which the ADF still called.
    """
    if isinstance(cmap, str):
        return mpl.colormaps[cmap]
    # End if
    return cmap


class _NoDebugLog:
    """Stand-in for the ADF object when a caller does not pass one."""

    def debug_log(self, msg):
        """Discard the message."""


def prep_contour_plot(adata, bdata, diffdata, pctdata, **kwargs):
    """Preparation for making contour plots.

    Prepares for making contour plots of adata, bdata, diffdata, and pctdata, which is
    presumably the difference between adata and bdata.
    - set colormap from kwargs (matplotlib or NCL) or defaults to viridis
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
        - 'pct_colorbar_opt': mpl kwargs for percent difference colorbar
        - 'units_label', 'diff_units_label', 'pct_units_label': colorbar title
          text, from the mpl colorbar "label" settings, else the units (or "%")
        - 'norm_diff': color normalization for difference panel
        - 'cmap_diff': colormap for difference panel
        - 'levels_diff': contour levels for difference panel
        - 'norm_pctdiff': color normalization for percent difference panel
        - 'cmap_pctdiff': colormap for percent difference panel
        - 'levels_pctdiff':contour levels for percent difference panel
        - 'cmap_sim': color map for a and b panels
        - 'norm_sim': color normalization for a and b panels
        - 'levels_sim' : contour levels for a and b panels
        - 'plot_log_p' : true/false whether to plot log(pressure) axis
        - 'extend_sim': colorbar extend for a and b panels
        - 'extend_diff': colorbar extend for difference panel
        - 'units': ADF cleaned units string for labeling colorbars
    """

    # Callers written for the earlier interface (tem.py, for example) do not
    # pass the ADF object, which is only needed here for debug logging:
    adfobj = kwargs.get("adfobj") or _NoDebugLog()

    polar_names = {"NHPolar": "nh", "SHPolar": "sh"}

    # Start color map/bar configurations
    # ----------------------------------
    if "plot_type" in kwargs:
        plot_type = kwargs["plot_type"]
        plot_type_dict = kwargs.get(plot_type, kwargs)
    else:
        # No plot-type section to look in, so use the top-level settings
        # (as above when the plot type has no section of its own):
        plot_type = None
        plot_type_dict = kwargs

    msg = f"{script_name}: prep_contour_plot()"
    msg_detail = f"\n\n\tPreparing contour map for {adata.name}"
    if "lev" in kwargs:
        msg_detail += f' - {kwargs["lev"]}'
    if "hemi" in kwargs:
        msg_detail += f' : {kwargs["hemi"]}'
    if plot_type:
        msg_detail += f" for {plot_type} plot"
    start_msg = msg + f"{msg_detail}\n\t{'-' * (len(msg_detail)-2)}\n"
    adfobj.debug_log(start_msg)

    # determine levels & color normalization:
    minval = np.min([np.min(adata), np.min(bdata)])
    maxval = np.max([np.max(adata), np.max(bdata)])

    # determine norm to use (deprecate this once minimum MPL version is high enough)
    normfunc, mplv = use_this_norm()

    # Case/Baseline  options -- Check in kwargs for colormap and levels
    # COLOR MAP
    # ---------
    cmap_case = get_cmap(adfobj, "case", plot_type_dict, kwargs, polar_names)
    msg = f"\n\tFinal case colormap: {cmap_case}\n\n"

    # CONTOUR LEVELS
    # ---------------
    levels_sim = resolve_levels(adfobj, "case", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tPre check levels: {type(levels_sim)}\n\t\t{levels_sim}\n"
    if levels_sim is None:
        msg += "\n\tSetting the levels from max/min"
        levels_sim = np.linspace(minval, maxval, 12)
    msg += f"\n\tFinal levels: {type(levels_sim)}\n\t\t{levels_sim}\n\n"

    # Check whether data exceeds limits
    vmin, vmax = levels_sim[0], levels_sim[-1]

    extend = "neither"
    if minval < vmin and maxval > vmax:
        extend = "both"
    elif minval < vmin:
        extend = "min"
    elif maxval > vmax:
        extend = "max"

    if kwargs.get("non_linear", False):
        cmap_obj = colormap_object(cmap_case)
        norm_sim = mpl.colors.BoundaryNorm(levels_sim, cmap_obj.N)
    else:
        norm_sim = mpl.colors.Normalize(vmin=min(levels_sim), vmax=max(levels_sim))
    # End if

    # Check if the minval and maxval are actually different.  If not,
    # then set "levels_sim" to be an empty list, which will cause the
    # plotting scripts to add a label instead of trying to plot a variable
    # with no contours:
    if minval == maxval:
        levels_sim = []
    # End if

    if ("colormap" not in plot_type_dict) and ("contour_levels" not in plot_type_dict):
        if (minval < 0 < maxval) and mplv > 2:
            norm_sim = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        else:
            norm_sim = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        # End if
    # End if

    # Difference options -- Check in kwargs for colormap and levels
    # determine levels & color normalization:
    minval = np.nanmin(diffdata)
    maxval = np.nanmax(diffdata)

    # COLOR MAP
    # ----------
    cmap_diff = get_cmap(adfobj, "diff", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tFinal difference colormap: {cmap_diff}\n\n"

    # CONTOUR LEVELS
    # ---------------
    levels_diff = resolve_levels(adfobj, "diff", plot_type_dict, kwargs, polar_names)

    msg += f"\n\tPre check difference LEVELS: {type(levels_diff)}\n\t\t{levels_diff}\n"
    if levels_diff is None:
        msg += "\n\tSetting the difference levels from max/min"
        absmaxdif = np.nanmax(np.abs(diffdata))
        levels_diff = np.linspace(-absmaxdif, absmaxdif, 12)
    msg += f"\n\tFinal difference levels: {type(levels_diff)}\n\t\t{levels_diff}\n"

    # Check whether data exceeds limits
    vmin, vmax = levels_diff[0], levels_diff[-1]
    extend_diff = "neither"
    if minval < vmin and maxval > vmax:
        extend_diff = "both"
    elif minval < vmin:
        extend_diff = "min"
    elif maxval > vmax:
        extend_diff = "max"

    # color normalization for difference
    if (np.min(levels_diff) < 0 < np.max(levels_diff)) and mplv > 2:
        norm_diff = normfunc(
            vmin=np.min(levels_diff), vmax=np.max(levels_diff), vcenter=0.0
        )
    else:
        norm_diff = mpl.colors.Normalize(
            vmin=np.min(levels_diff), vmax=np.max(levels_diff)
        )

    # Percent Difference options -- Check in kwargs for colormap and levels.
    # As for the case and difference panels, settings in the plot type's
    # section win, then the variable's top-level settings:
    def pct_option(key):
        if key in plot_type_dict:
            return plot_type_dict[key]
        return kwargs.get(key)

    # COLOR MAP
    # ----------
    cmap_pctdiff = pct_option("pct_diff_colormap") or "PuOr_r"

    if cmap_pctdiff not in plt.colormaps():
        msg += (
            f"\n\tPercent Difference: {cmap_pctdiff} is not a matplotlib standard "
            "color map. Trying if this an NCL color map"
        )
        cmap_pctdiff, ncl_msg = try_load_ncl_cmap(
            adfobj, cmap_pctdiff, fallback="PuOr_r"
        )
        msg += ncl_msg

    # CONTOUR LEVELS
    # ---------------
    levels_pctdiff = pct_option("pct_diff_contour_levels")  # explicit levels
    pct_range = pct_option("pct_diff_contour_range")
    if levels_pctdiff is None and pct_range is not None:
        if len(pct_range) == 3:
            levels_pctdiff = np.arange(*pct_range)
        else:
            msg += (
                "\n\tWARNING: pct_diff_contour_range must have exactly three entries:"
                f" min, max, step, not {pct_range}. Using the default levels."
            )
    if levels_pctdiff is None:
        levels_pctdiff = [
            -100,
            -75,
            -50,
            -40,
            -30,
            -20,
            -10,
            -8,
            -6,
            -4,
            -2,
            0,
            2,
            4,
            6,
            8,
            10,
            20,
            30,
            40,
            50,
            75,
            100,
        ]
    norm_pctdiff = mpl.colors.BoundaryNorm(levels_pctdiff, 256)

    vmin, vmax = levels_pctdiff[0], levels_pctdiff[-1]
    extend_pctdiff = "neither"
    if pctdata is not None:
        minval = np.nanmin(pctdata)
        maxval = np.nanmax(pctdata)
        if minval < vmin and maxval > vmax:
            extend_pctdiff = "both"
        elif minval < vmin:
            extend_pctdiff = "min"
        elif maxval > vmax:
            extend_pctdiff = "max"

    plot_log_p = kwargs.get("plot_log_pressure", False)

    subplots_opt = {}
    contourf_opt = {}
    colorbar_opt = {}
    diff_colorbar_opt = {}
    pct_colorbar_opt = {}

    # extract any MPL kwargs that should be passed on:
    if "mpl" in kwargs:
        subplots_opt.update(kwargs["mpl"].get("subplots", {}))
        contourf_opt.update(kwargs["mpl"].get("contourf", {}))
        colorbar_opt.update(kwargs["mpl"].get("colorbar", {}))
        diff_colorbar_opt.update(kwargs["mpl"].get("diff_colorbar", {}))
        pct_colorbar_opt.update(kwargs["mpl"].get("pct_diff_colorbar", {}))
    # End if

    units = kwargs.get("units", adata.attrs.get("units", ""))

    # A colorbar "label" from the variable defaults is drawn as the title
    # above that colorbar (where the units go), not as matplotlib's side
    # label, which would show the units a second time:
    units_label = colorbar_opt.pop("label", units)
    diff_units_label = diff_colorbar_opt.pop("label", units_label)
    pct_units_label = pct_colorbar_opt.pop("label", "%")

    cp_info = {
        "subplots_opt": subplots_opt,
        "contourf_opt": contourf_opt,
        "colorbar_opt": colorbar_opt,
        "diff_colorbar_opt": diff_colorbar_opt,
        "pct_colorbar_opt": pct_colorbar_opt,
        "units_label": units_label,
        "diff_units_label": diff_units_label,
        "pct_units_label": pct_units_label,
        "norm_diff": norm_diff,
        "cmap_diff": cmap_diff,
        "levels_diff": levels_diff,
        "norm_pctdiff": norm_pctdiff,
        "cmap_pctdiff": cmap_pctdiff,
        "levels_pctdiff": levels_pctdiff,
        "cmap_sim": cmap_case,
        "norm_sim": norm_sim,
        "levels_sim": levels_sim,
        "plot_log_p": plot_log_p,
        "extend_sim": extend,
        "extend_diff": extend_diff,
        "extend_pctdiff": extend_pctdiff,
        "units": units,
    }

    # The names these had before, which scripts such as tem.py still read:
    cp_info.update(
        {
            "cmap1": cmap_case,
            "norm1": norm_sim,
            "levels1": levels_sim,
            "cmapdiff": cmap_diff,
            "normdiff": norm_diff,
            "levelsdiff": levels_diff,
            "cmappct": cmap_pctdiff,
            "pctnorm": norm_pctdiff,
            "levelspctdiff": levels_pctdiff,
        }
    )
    return cp_info


def add_var_name_key_to_vres(adfobj, var, vres, vector=False):
    """
    Set ``vres["var_name"]``, the name plots show for this variable.

    Uses the variable's nickname when there is one, else the variable name.
    For vector plots (``vector=True``), a vector component (U, TAUX, ...)
    is named after its vector instead: the vector's nickname, or name.

    Returns a copy: ``vres`` is usually an entry of the variable defaults,
    which every plotting script shares, so the plot settings scripts add to
    it (season, plot_type, lev, ...) must not end up there.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, for the variable defaults of the vector
    var : str
        variable name
    vres : dict
        the variable's defaults
    vector : bool, optional
        True when plotting the variable as part of its vector

    Returns
    -------
    dict
        a copy of ``vres`` with "var_name" set
    """
    vres = dict(vres)
    if vector and "vector_name" in vres:
        vect_var_name = vres["vector_name"]
        vec_vres = adfobj.variable_defaults.get(vect_var_name, {})
        vres["var_name"] = vec_vres.get("nickname", vect_var_name)
    else:
        vres["var_name"] = vres.get("nickname", var)
    return vres


def prep_ux_for_plot(fld, grid_source):
    """
    Get a native-grid seasonal field ready for plotting with uxarray.

    Takes the time mean if there is a time dimension, sets infinities to NaN,
    and makes sure the result is still a ``UxDataArray`` on the source grid --
    xarray operations on a ``UxDataArray`` can hand back a plain DataArray.

    NaNs are kept, so that cells without data (e.g. ocean cells of a land
    field) are left blank in the raster and are skipped by the statistics,
    rather than being drawn and counted as 0.

    Parameters
    ----------
    fld : xarray.DataArray or uxarray.UxDataArray
        The field to plot, on the native grid.
    grid_source : uxarray.UxDataArray
        Data on the same grid, whose ``uxgrid`` is used if `fld` lost its own.

    Returns
    -------
    uxarray.UxDataArray
        The field, ready for plotting.
    """
    if "time" in fld.dims:
        fld = fld.mean("time")
    fld = fld.where(np.isfinite(fld))
    if not isinstance(fld, ux.UxDataArray):
        fld = ux.UxDataArray(
            data=fld.data,
            dims=fld.dims,
            coords=fld.coords,
            uxgrid=grid_source.uxgrid,
        )
    return fld


#####################
# END HELPER FUNCTIONS
