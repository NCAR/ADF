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
from urllib.parse import urlparse
from urllib.request import urlretrieve
import warnings

# +++++++++++++++++++++++++++++++++++++++++++++++++
# import non-standard python modules, including ADF
# +++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.lines as mlines
import cartopy.crs as ccrs

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
            cmap = kwargs.get("pct_diff_colormap", "PuOr")
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

ncl_defaults = ["ncl_default"]


def guess_ncl_url(cmap):
    """Return the URL of the NCL colormap file named ``cmap``."""
    return f"https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/{cmap}.rgb"


def download_ncl_colormap(url, dest):
    """Download the NCL colormap file at ``url`` to ``dest``."""
    urlretrieve(url, dest)


def read_ncl_colormap(adfobj, fil):
    """
    Read an NCL colormap file into an (N, 3) array of RGB values.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    fil : str or pathlib.Path
        local file, or the URL of one to download first

    Returns
    -------
    numpy.ndarray
        one row per color
    """
    # determine if fil is a URL:
    # if so, we have to download it

    msg = f"{script_name}: read_ncl_colormap()"
    if isinstance(fil, str):
        pars = urlparse(fil)
        if pars.scheme in ["http", "https", "ftp"]:
            filename = Path.cwd() / fil.split("/")[-1]
            if filename.is_file():
                msg += f"\n\tFile already downloaded as {filename}"
            else:
                msg += f"\n\tFile will be downloaded and saved as {filename}"
                download_ncl_colormap(fil, str(filename))
        else:
            filename = Path(fil)
    elif isinstance(fil, Path):
        filename = fil
    else:
        raise ValueError(f"\tERROR: what to do with type {type(fil)}")

    # NCL's colormaps are not regularized enough to just use read_csv.
    # We have to determine how many lines to skip because it varies.
    # NCL has some files that have "comments" at the end
    # which will look like additional columnns
    # We basically are forced to just read line-by-line

    # ASSUME ALL NCL COLORMAPS ARE N rows BY 3 COLUMNS,
    # AND THE VALUES ARE INTEGERS 0-255.
    with open(filename, encoding="utf-8") as f:
        table_exists = False
        for line in f:
            line_str = line.strip()  # remove leading/trailing whitespace
            if (len(line_str) == 0) or (not line_str[0].isdigit()):
                continue  # empty line or non-data line
                # NOTE: also skips if the first value is negative (hlu_default)
            if re.search(
                r"[^\s0-9-\.]", line_str
            ):  # any non number characters ASSUMED AT END
                # take the string up to the non-matching character
                line_vals = (
                    line_str[: re.search(r"[^\s0-9-\.]", line_str).start() - 1]
                    .strip()
                    .split()
                )
            else:
                line_vals = line_str.split()
            try:
                row = [float(r) for r in line_vals]
            except ValueError:
                # Skip it, rather than adding the previous line's row again:
                msg += f"\n\tERROR reading RGB file {line_vals}"
                continue
            if table_exists:
                table = np.vstack([table, row])
            else:
                table = np.array(row)
                table_exists = True
    adfobj.debug_log(msg)
    if not table_exists:
        raise ValueError(f"no RGB values found in NCL colormap file {filename}")
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
        the colormap and its reversed version
    """
    msg = f"{script_name}: ncl_to_mpl()"
    if nclmap.max() > 1:
        try:
            vals = nclmap / 255
        except TypeError:
            msg += f"\n\tERROR: could not divide by 255. {type(nclmap) = }"
            msg += f" {nclmap}"
            adfobj.debug_log(msg)
            return None, None
    else:
        msg += f"\n\t{name} seems to be 0-1"
        vals = nclmap
    assert vals.shape[1] == 3, "vals.shape should be (N,3)"
    ncolors = vals.shape[0]
    if ncolors > 100:
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, vals)
        my_cmap_r = my_cmap.reversed()
    else:
        my_cmap = mpl.colors.ListedColormap(vals, name)
        my_cmap_r = my_cmap.reversed()
    # my_cmap, my_cmap_r from reversing a colormap
    # ALLOW MPL TO KNOW ABOUT THE COLORMAP:
    # mpl.colormaps.register(cmap=my_cmap)
    # mpl.colormaps.register(cmap=my_cmap_r)

    adfobj.debug_log(msg)
    return my_cmap, my_cmap_r


def choose_colormap_type(levels, threshold_symmetry=0.25):
    """
    Suggest "diverging" or "sequential" for a set of contour levels.

    Diverging when the levels cross zero and are roughly symmetric about it.
    """
    levels = np.array(levels)
    minval, maxval = np.min(levels), np.max(levels)
    crosses_zero = minval < 0 < maxval
    symmetry_ratio = abs(abs(maxval) - abs(minval)) / max(abs(maxval), abs(minval))
    is_symmetric = symmetry_ratio < threshold_symmetry
    return "diverging" if crosses_zero and is_symmetric else "sequential"


def load_colormap(adfobj, cmap_name):
    """
    Return a matplotlib colormap name, or a colormap loaded from NCL.

    Falls back to "viridis" when neither has it.
    """
    msg = f"{script_name}: load_colormap()"
    if cmap_name in plt.colormaps():
        adfobj.debug_log(msg)
        return cmap_name
    adfobj.debug_log(
        msg + f"\n\t{cmap_name} not a standard Matplotlib colormap. Trying NCL..."
    )
    cmap, _ = try_load_ncl_cmap(adfobj, cmap_name)
    return cmap


def try_load_ncl_cmap(adfobj, cmap_name, fallback="viridis"):
    """
    Load an NCL colormap by name, from a local "<name>.rgb" file or from NCL.

    Parameters
    ----------
    adfobj : AdfDiag
        ADF object, used for debug logging
    cmap_name : str
        NCL colormap name
    fallback : str, optional
        matplotlib colormap to use when the NCL one cannot be loaded

    Returns
    -------
    tuple
        the colormap (or ``fallback``) and the debug message
    """
    msg = f"{script_name}: try_load_ncl_cmap()"
    msg += f"\n\tTrying {cmap_name} as an NCL color map:"
    locfil = Path(".") / f"{cmap_name}.rgb"
    source = locfil if locfil.is_file() else guess_ncl_url(cmap_name)
    try:
        data = read_ncl_colormap(adfobj, source)
        cmap, _ = ncl_to_mpl(adfobj, data, cmap_name)
    # Download, read or conversion failures all mean using the fallback:
    except Exception as err:  # pylint: disable=broad-exception-caught
        cmap = None
        msg += f"\n\tCould not load NCL colormap {cmap_name}: {err}"
    if cmap is None:
        msg += f"\n\tDefaulting to '{fallback}'."
        cmap = fallback
    adfobj.debug_log(msg)
    return cmap, msg


def get_cmap(adfobj, plotty, plot_type_dict, kwargs, polar_names):
    """
    Gather colormap from variable defaults file, if applicable.
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

    # Priority 3: NCL support
    if cmap_case in ncl_defaults:
        cmap_case, msg = try_load_ncl_cmap(adfobj, cmap_case)

    # Last attempt fallback default
    if not cmap_case:
        msg += f"\n\tNo cmap for {plotty} found, defaulting to {default_cmap}"
        cmap_case = default_cmap

    # Final check: must exist in matplotlib or NCL
    if isinstance(cmap_case, str):
        if (cmap_case not in plt.colormaps()) and (cmap_case not in ncl_defaults):
            msg += f"\n\tInvalid cmap '{cmap_case}' for {plotty}, defaulting to {default_cmap}"
            cmap_case = default_cmap

    adfobj.debug_log(msg)

    return cmap_case


# Conour Plot Prep Functions
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
    def process_entry(entry, kind, msg):
        """Handle lists and dicts for levels/ranges/linspace."""
        if isinstance(entry, list):
            if len(entry) == 3:
                if kind == "range":
                    msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                    adfobj.debug_log(msg)
                    return np.arange(*entry)
                if kind == "linspace":
                    msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                    adfobj.debug_log(msg)
                    return np.linspace(*entry)
                msg += (
                    f"\n\tLevels specified for {plotty} as list of 3 values, "
                    "please add more values."
                )
                msg += " Will get contrours from data range."
                adfobj.debug_log(msg)
                return None
            if len(entry) < 3:
                msg += f"\n\tNot enough {kind} entries for {plotty} (<3) — ambiguous"
                adfobj.debug_log(msg)
                return None
            adfobj.debug_log(msg)
            return entry
        if isinstance(entry, dict):
            resolved = resolve_hemi_level(adfobj, entry, kwargs, polar_names)
            if isinstance(resolved, list) and len(resolved) == 3:
                if kind == "range":
                    msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                    adfobj.debug_log(msg)
                    return np.arange(*resolved)
                if kind == "linspace":
                    msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                    adfobj.debug_log(msg)
                    return np.linspace(*resolved)
            adfobj.debug_log(msg)
            return resolved
        adfobj.debug_log(msg)
        return entry

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
            levels_sim = process_entry(entry, kind, msg)
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


# "pctdata" is unused, but is kept so as to not break existing calls:
# pylint: disable-next=unused-argument
class _NoDebugLog:
    """Stand-in for the ADF object when a caller does not pass one."""

    def debug_log(self, msg):
        """Discard the message."""


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
        absmaxdif = np.max(np.abs(diffdata))
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

    # Percent Difference options -- Check in kwargs for colormap and levels
    # COLOR MAP
    # ----------
    # determine levels & color normalization:
    # minval = np.min(diffdata)
    # maxval = np.max(diffdata)
    cmap_pctdiff = plot_type_dict.get("pct_diff_colormap", "PuOr_r")

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
    if "pct_diff_contour_levels" in plot_type_dict:
        levels_pctdiff = plot_type_dict[
            "pct_diff_contour_levels"
        ]  # a list of explicit contour levels
    elif "pct_diff_contour_range" in plot_type_dict:
        assert (
            len(plot_type_dict["pct_diff_contour_range"]) == 3
        ), "pct_diff_contour_range must have exactly three entries: min, max, step"
        levels_pctdiff = np.arange(*plot_type_dict["pct_diff_contour_range"])
    else:
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


def add_var_to_vres(adfobj, var, vres):
    """
    Set ``vres["var_name"]``, the name plots show for this variable.

    Uses the variable's nickname (or, for a vector component, the vector's
    nickname or name) when there is one, else the variable name.

    Returns a copy: ``vres`` is usually an entry of the variable defaults,
    which every plotting script shares, so the plot settings scripts add to
    it (season, plot_type, lev, ...) must not end up there.
    """
    vres = dict(vres)
    if "vector_name" in vres:
        vect_var_name = vres["vector_name"]
        vec_vres = adfobj.variable_defaults.get(vect_var_name, {})
        if "nickname" in vec_vres:
            vres["var_name"] = vec_vres["nickname"]
        else:
            vres["var_name"] = vect_var_name
    else:

        if "nickname" in vres:
            vres["var_name"] = vres["nickname"]
        else:
            vres["var_name"] = var
    return vres


#####################
# END HELPER FUNCTIONS
