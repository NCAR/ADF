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
import os

import urllib
from urllib.parse import urlparse
from urllib.request import urlretrieve
from pathlib import Path
import re

from adf_diag import AdfDiag
import adf_utils as utils

import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

#Set non-X-window backend for matplotlib:
mpl.use('Agg')

#Now import pyplot:
import matplotlib.pyplot as plt

script_name = os.path.splitext(os.path.basename(__file__))[0]

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


# Color Map Functions
#--------------------

ncl_defaults = ["ncl_default"]

def guess_ncl_url(cmap):
    return f"https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files/{cmap}.rgb"


def download_ncl_colormap(url, dest):
    urlretrieve(url, dest)


def read_ncl_colormap(adfobj, fil):
    # determine if fil is a URL:
    # if so, we have to download it

    msg = f"{script_name}: read_ncl_colormap()"
    if isinstance(fil, str):
        pars = urlparse(fil)
        if pars.scheme in ['http', 'https', 'ftp']:
            filename = Path.cwd() / fil.split("/")[-1]
            if filename.is_file():
                msg += f"\n\tFile already downloaded as {filename}"
            else:
                msg += f"\n\tFile will be downloaded and saved as {filename}"
                download_ncl_colormap(fil, str(filename))
        else:
            is_url = False
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
    with open(filename) as f:
        table_exists = False
        for count, line in enumerate(f):
            line_str = line.strip() # remove leading/trailing whitespace
            if (len(line_str) == 0) or (not line_str[0].isdigit()):
                continue # empty line or non-data line 
                # NOTE: also skips if the first value is negative (hlu_default) 
            else:
                if re.search(r'[^\s0-9-\.]', line_str): # any non number characters ASSUMED AT END
                    # take the string up to the non-matching character
                    line_vals = line_str[:re.search(r'[^\s0-9-\.]', line_str).start()-1].strip().split()
                else:
                    line_vals = line_str.split()
                try: 
                    row = [float(r) for r in line_vals]
                except:
                    msg += f"\n\tERROR reading RGB file {line_vals}"
                if table_exists:
                    table = np.vstack([table, row])
                else:
                    table = np.array(row)
                    table_exists=True
    adfobj.debug_log(msg)
    return table


def ncl_to_mpl(adfobj, nclmap, name):
    msg = f"{script_name}: ncl_to_mpl()"
    if nclmap.max() > 1:
        try:
            vals = nclmap / 255
        except:
            msg += f"\n\tERROR: could not divide by 255. {type(nclmap) = }"
            msg += f" {nclmap}"
            adfobj.debug_log(msg)
            return None
    else:
        msg += f"\n\t{name} seems to be 0-1"
        vals = nclmap
    assert vals.shape[1] == 3, 'vals.shape should be (N,3)'
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
    levels = np.array(levels)
    minval, maxval = np.min(levels), np.max(levels)
    crosses_zero = (minval < 0) and (maxval > 0)
    symmetry_ratio = abs(abs(maxval) - abs(minval)) / max(abs(maxval), abs(minval))
    is_symmetric = symmetry_ratio < threshold_symmetry
    return 'diverging' if crosses_zero and is_symmetric else 'sequential'


def load_colormap(adfobj, cmap_name):
    msg = f"{script_name}: load_colormap()"
    if cmap_name in plt.colormaps():
        adfobj.debug_log(msg)
        return cmap_name
    else:
        msg += f"\n\t{cmap_name} not a standard Matplotlib colormap. Trying NCL..."
        url = guess_ncl_url(cmap_name)
        locfil = Path(".") / f"{cmap_name}.rgb"
        data = read_ncl_colormap(locfil,msg) if locfil.is_file() else read_ncl_colormap(url)
        cm, cmr = ncl_to_mpl(adfobj, data, cmap_name)
        if not cm:
            msg += f"\n\tFailed to load {cmap_name}. Defaulting to 'coolwarm'."
            adfobj.debug_log(msg)
            return 'coolwarm'
        adfobj.debug_log(msg)
        return cm
    

def try_load_ncl_cmap(adfobj, cmap_case):
    """Try to load an NCL colormap, fallback to PRECT special case or 'coolwarm'."""
    msg = f"{script_name}: try_load_ncl_cmap()"
    msg += f"\n\tTrying {cmap_case} as an NCL color map:"
    try:
        url = guess_ncl_url(cmap_case)
        locfil = Path(".") / f"{cmap_case}.rgb"
        if locfil.is_file():
            data = read_ncl_colormap(locfil)
        else:
            try:
                data = read_ncl_colormap(url)
            except urllib.error.HTTPError:
                msg += f"\n\tNCL colormap file not found"

        if isinstance(data, np.ndarray):
            cm, cmr = ncl_to_mpl(data, cmap_case)
            adfobj.debug_log(msg)
            return cm, msg
    except Exception:
        pass

    adfobj.debug_log(msg)
    return "coolwarm", msg


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

    # Priority 3: fallback default
    if not cmap_case:
        msg += f"\n\tNo cmap for {plotty} found, defaulting to {default_cmap}"
        cmap_case = default_cmap

    # NCL support
    if cmap_case in ncl_defaults:
        cmap_case, msg = try_load_ncl_cmap(adfobj, cmap_case)

    # Final check: must exist in matplotlib or NCL
    if isinstance(cmap_case, str):
        if (cmap_case not in plt.colormaps()) and (cmap_case not in ncl_defaults):
            msg += f"\n\tInvalid cmap '{cmap_case}' for {plotty}, defaulting to {default_cmap}"
            cmap_case = default_cmap
    
    adfobj.debug_log(msg)

    return cmap_case


# Conour Plot Prep Functions
#----------------------------
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
    elif lev and lev in data:
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
        contour_levels, contour_range, contour_linspace = key_map.get(plotty, (None, None, None))

        def process_entry(entry, kind, msg):
            """Handle lists and dicts for levels/ranges/linspace."""
            if isinstance(entry, list):
                if len(entry) == 3:
                    if kind == "range":
                        msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                        adfobj.debug_log(msg)
                        return np.arange(*entry)
                    elif kind == "linspace":
                        msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                        adfobj.debug_log(msg)
                        return np.linspace(*entry)
                    else:
                        msg += f"\n\tLevels specified for {plotty} as list of 3 values, please add more values."
                        msg += " Will get contrours from data range."
                        adfobj.debug_log(msg)
                        return None
                elif len(entry) < 3:
                    msg += f"\n\tNot enough {kind} entries for {plotty} (<3) — ambiguous"
                    adfobj.debug_log(msg)
                else:
                    adfobj.debug_log(msg)
                    return entry
            elif isinstance(entry, dict):
                resolved = resolve_hemi_level(adfobj, entry, kwargs, polar_names)
                if isinstance(resolved, list) and len(resolved) == 3:
                    if kind == "range":
                        msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                        adfobj.debug_log(msg)
                        return np.arange(*resolved)
                    elif kind == "linspace":
                        msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                        adfobj.debug_log(msg)
                        return np.linspace(*resolved)
                adfobj.debug_log(msg)
                return resolved
            else:
                adfobj.debug_log(msg)
                return entry

        # Priority: explicit contour levels → range → linspace
        for key, kind in [(contour_levels, "levels"),
                        (contour_range, "range"),
                        (contour_linspace, "linspace")]:
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

    adfobj = kwargs["adfobj"]

    polar_names = {"NHPolar":"nh",
                   "SHPolar":"sh"}
    
    # Start color map/bar configurations
    # ----------------------------------
    if "plot_type" in kwargs:
        plot_type = kwargs["plot_type"]
        if plot_type in kwargs:
            plot_type_dict = kwargs[plot_type]
        else:
            plot_type_dict = kwargs
    else:
        plot_type = None
        plot_type_dict = {}

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
    #---------
    cmap_case = get_cmap(adfobj, "case", plot_type_dict, kwargs, polar_names)
    msg = f"\n\tFinal case colormap: {cmap_case}\n\n"
    
    # CONTOUR LEVELS
    #---------------
    levels_sim = resolve_levels(adfobj, "case", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tPre check levels: {type(levels_sim)}\n\t\t{levels_sim}\n"
    if levels_sim is None:
        msg += "\n\tSetting the levels from max/min"
        levels_sim = np.linspace(minval, maxval, 12)
    msg += f"\n\tFinal levels: {type(levels_sim)}\n\t\t{levels_sim}\n\n"

    # Check whether data exceeds limits
    vmin, vmax = levels_sim[0], levels_sim[-1]

    extend = 'neither'
    if minval < vmin and maxval > vmax:
        extend = 'both'
    elif minval < vmin:
        extend = 'min'
    elif maxval > vmax:
        extend = 'max'
    
    if kwargs.get('non_linear', False):
        cmap_obj = cm.get_cmap(cmap_case)
        norm_sim = mpl.colors.BoundaryNorm(levels_sim, cmap_obj.N)
    else:
        norm_sim = mpl.colors.Normalize(vmin=min(levels_sim), vmax=max(levels_sim))
    #End if

    #Check if the minval and maxval are actually different.  If not,
    #then set "levels_sim" to be an empty list, which will cause the
    #plotting scripts to add a label instead of trying to plot a variable
    #with no contours:
    if minval == maxval:
        levels_sim = []
    #End if

    if ('colormap' not in plot_type_dict) and ('contour_levels' not in plot_type_dict):
        if ((minval < 0) and (0 < maxval)) and mplv > 2:
            norm_sim = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        else:
            norm_sim = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        #End if
    #End if
    
    # Difference options -- Check in kwargs for colormap and levels
    # determine levels & color normalization:
    minval = np.nanmin(diffdata)
    maxval = np.nanmax(diffdata)

    # COLOR MAP
    #----------
    cmap_diff = get_cmap(adfobj, "diff", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tFinal difference colormap: {cmap_diff}\n\n"

    # CONTOUR LEVELS
    #---------------
    levels_diff = resolve_levels(adfobj, "diff", plot_type_dict, kwargs, polar_names)

    msg += f"\n\tPre check difference LEVELS: {type(levels_diff)}\n\t\t{levels_diff}\n"
    if levels_diff is None:
        msg += f"\n\tSetting the difference levels from max/min"
        absmaxdif = np.max(np.abs(diffdata))
        levels_diff = np.linspace(-absmaxdif, absmaxdif, 12)
    msg += f"\n\tFinal difference levels: {type(levels_diff)}\n\t\t{levels_diff}\n"

    # Check whether data exceeds limits
    vmin, vmax = levels_diff[0], levels_diff[-1]
    extend_diff = 'neither'
    if minval < vmin and maxval > vmax:
        extend_diff = 'both'
    elif minval < vmin:
        extend_diff = 'min'
    elif maxval > vmax:
        extend_diff = 'max'

    # color normalization for difference
    if ((np.min(levels_diff) < 0) and (0 < np.max(levels_diff))) and mplv > 2:
        norm_diff = normfunc(vmin=np.min(levels_diff), vmax=np.max(levels_diff), vcenter=0.0)
    else:
        norm_diff = mpl.colors.Normalize(vmin=np.min(levels_diff), vmax=np.max(levels_diff))

    # Percent Difference options -- Check in kwargs for colormap and levels
    # COLOR MAP
    #----------
    # determine levels & color normalization:
    #minval = np.min(diffdata)
    #maxval = np.max(diffdata)
    if "pct_diff_colormap" in plot_type_dict:
        cmap_pctdiff = plot_type_dict["pct_diff_colormap"]
    else:
        cmap_pctdiff = "PuOr_r"
    #End if

    if cmap_pctdiff not in plt.colormaps():
        msg += f"\n\tPercent Difference: {cmap_pctdiff} is not a matplotlib standard color map."
        msg += f" Trying if this an NCL color map"

        url = guess_ncl_url(cmap_pctdiff)
        locfil = "." / f"{cmap_pctdiff}.rgb"
        if locfil.is_file():
            data = read_ncl_colormap(locfil)
        else:
            data = read_ncl_colormap(url)
        cm, cmr = ncl_to_mpl(data, cmap_pctdiff)
        #ncl_colors[cm.name] = cm
        #ncl_colors[cmr.name] = cmr
        
        if not cm:
            msg += f"\n\tPercent Difference: {cmap_pctdiff} is not a matplotlib or NCL color map."
            cmap_pctdiff = 'PuOr_r'
            msg += f" Defaulting to '{cmap_pctdiff}'"
            adfobj.debug_log(msg)
        else:
            cmap_pctdiff = cm

    # CONTOUR LEVELS
    #---------------
    if "pct_diff_contour_levels" in plot_type_dict:
        levels_pctdiff = plot_type_dict["pct_diff_contour_levels"]  # a list of explicit contour levels
    elif "pct_diff_contour_range" in plot_type_dict:
            assert len(plot_type_dict['pct_diff_contour_range']) == 3, "pct_diff_contour_range must have exactly three entries: min, max, step"
            levels_pctdiff = np.arange(*plot_type_dict['pct_diff_contour_range'])
    else:
        levels_pctdiff = [-100,-75,-50,-40,-30,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,30,40,50,75,100]
    norm_pctdiff = mpl.colors.BoundaryNorm(levels_pctdiff,256)

    vmin, vmax = levels_pctdiff[0], levels_pctdiff[-1]
    extend_pctdiff = 'neither'
    if pctdata is not None:
        minval = np.nanmin(pctdata)
        maxval = np.nanmax(pctdata)
        if minval < vmin and maxval > vmax:
            extend_pctdiff = 'both'
        elif minval < vmin:
            extend_pctdiff = 'min'
        elif maxval > vmax:
            extend_pctdiff = 'max'

    if "plot_log_pressure" in kwargs:
        plot_log_p = kwargs["plot_log_pressure"]
    else:
        plot_log_p = False

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

    if "units" in kwargs:
        units = kwargs["units"]
    else:
        units = adata.units

    return {'subplots_opt': subplots_opt,
            'contourf_opt': contourf_opt,
            'colorbar_opt': colorbar_opt,
            'diff_colorbar_opt': diff_colorbar_opt,
            'pct_colorbar_opt': pct_colorbar_opt,
            'norm_diff': norm_diff,
            'cmap_diff': cmap_diff,
            'levels_diff': levels_diff,
            'norm_pctdiff': norm_pctdiff,
            'cmap_pctdiff': cmap_pctdiff,
            'levels_pctdiff':levels_pctdiff,
            'cmap_sim': cmap_case,
            'norm_sim': norm_sim,
            'levels_sim': levels_sim,
            'plot_log_p': plot_log_p,
            'extend_sim': extend,
            'extend_diff': extend_diff,
            'extend_pctdiff': extend_pctdiff,
            'units': units
            }


#####################
#END HELPER FUNCTIONS