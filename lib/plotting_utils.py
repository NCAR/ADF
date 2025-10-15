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
import numpy as np
import xarray as xr
import matplotlib as mpl
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

empty_message = "No Valid\nData Points"
props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}


#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]
            }

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

    msg = f"\n\t{script_name}: read_ncl_colormap()"
    if isinstance(fil, str):
        pars = urlparse(fil)
        if pars.scheme in ['http', 'https', 'ftp']:
            filename = Path.cwd() / fil.split("/")[-1]
            if filename.is_file():
                msg += f"\n\t\tFile already downloaded as {filename}"
            else:
                msg += f"\n\t\tFile will be downloaded and saved as {filename}"
                download_ncl_colormap(fil, str(filename))
        else:
            is_url = False
            filename = Path(fil)
    elif isinstance(fil, Path):
        filename = fil
    else:
        raise ValueError(f"\t\tERROR: what to do with type {type(fil)}")
        
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
                    msg += f"\n\t\tERROR reading RGB file {line_vals}"
                    adfobj.debug_log(msg)
                    return None
                if table_exists:
                    table = np.vstack([table, row])
                else:
                    table = np.array(row)
                    table_exists=True
    adfobj.debug_log(msg)
    return table


def ncl_to_mpl(adfobj, nclmap, name):
    msg = f"\n\t{script_name}: ncl_to_mpl()"
    print(msg)
    if nclmap.max() > 1:
        try:
            vals = nclmap / 255
        except:
            msg += f"\n\t\tERROR: could not divide by 255. {type(nclmap) = }"
            msg += f" {nclmap}"
            return None, msg
    else:
        msg += f"\n\t\t{name} seems to be 0-1"
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
    

def try_load_ncl_cmap(adfobj, cmap_case):
    """Try to load an NCL colormap, fallback to PRECT special case or 'coolwarm'."""
    msg = f"\n\t{script_name}: try_load_ncl_cmap()"
    msg += f"\n\t\tTrying {cmap_case} as an NCL color map:"
    try:
        url = guess_ncl_url(cmap_case)
        locfil = Path(".") / f"{cmap_case}.rgb"
        if locfil.is_file():
            data = read_ncl_colormap(adfobj, locfil)
        else:
            try:
                data = read_ncl_colormap(adfobj, url)
            except urllib.error.HTTPError:
                msg += f"\n\t\tNCL colormap file not found"

        if isinstance(data, np.ndarray):
            try:
                cm, cmr = ncl_to_mpl(adfobj, data, cmap_case)
            except Exception as e:
                print("Exception in ncl_to_mpl:", e)
                import traceback; traceback.print_exc()
                adfobj.debug_log(f"Exception in ncl_to_mpl: {e}")
                return "coolwarm"
            adfobj.debug_log(msg)
            return cm
    except Exception:
        pass

    adfobj.debug_log(msg)
    return "coolwarm"


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
    
    msg = f"\n\t{script_name}: get_cmap()"

    # Priority 1: YAML dict
    if colormap_key in plot_type_dict:
        cmap_entry = plot_type_dict[colormap_key]
        msg += f"\n\t\tUser supplied cmap for {plotty}: {cmap_entry}"

        if isinstance(cmap_entry, str):
            cmap_case = cmap_entry
        elif isinstance(cmap_entry, dict):
            resolved = resolve_hemi_level(adfobj, cmap_entry, kwargs, polar_names)
            if isinstance(resolved, str):
                cmap_case = resolved

    # Priority 2: kwargs dict
    elif colormap_key in kwargs and isinstance(kwargs[colormap_key], str):
        cmap_case = kwargs[colormap_key]
        msg += f"\n\t\tUser supplied cmap for {plotty}: {cmap_case}"

    # Priority 3: fallback default
    if not cmap_case:
        msg += f"\n\t\tNo cmap for {plotty} found, defaulting to {default_cmap}"
        cmap_case = default_cmap

    # NCL support
    if cmap_case in ncl_defaults:
        cmap_case = try_load_ncl_cmap(adfobj, cmap_case)

    # Final check: must exist in matplotlib or NCL
    if isinstance(cmap_case, str):
        if (cmap_case not in plt.colormaps()) and (cmap_case not in ncl_defaults):
            msg += f"\n\t\tInvalid cmap '{cmap_case}' for {plotty}, defaulting to {default_cmap}"
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
        levels1 = None
        msg = f"{script_name}: resolve_levels()"

        # Map keys based on plot type
        key_map = {
            "diff": ("diff_contour_levels", "diff_contour_range", "diff_contour_linspace"),
            "case": ("contour_levels", "contour_levels_range", "contour_levels_linspace"),
        }
        contour_levels, contour_range, contour_linspace = key_map.get(plotty, (None, None, None))

        def process_entry(entry, kind, msg):
            msg += f"\n\tprocess_entry()"
            """Handle lists and dicts for levels/ranges/linspace."""
            if isinstance(entry, list):
                if len(entry) == 3:
                    if kind == "range":
                        msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                        #adfobj.debug_log(msg)
                        return np.arange(*entry), msg
                    elif kind == "linspace":
                        msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                        #adfobj.debug_log(msg)
                        return np.linspace(*entry), msg
                    else:
                        msg += f"\n\tLevels specified for {plotty} as list of 3 values, please add more values."
                        msg += " Will get contrours from data range."
                        #adfobj.debug_log(msg)
                        return None, msg
                elif len(entry) < 3:
                    msg += f"\n\tNot enough {kind} entries for {plotty} (<3) — ambiguous"
                    #adfobj.debug_log(msg)
                else:
                    #adfobj.debug_log(msg)
                    return entry, msg
            elif isinstance(entry, dict):
                resolved = resolve_hemi_level(adfobj, entry, kwargs, polar_names)
                if isinstance(resolved, list) and len(resolved) == 3:
                    if kind == "range":
                        msg += f"\n\tLevels specified for {plotty}: numpy.arange."
                        #adfobj.debug_log(msg)
                        return np.arange(*resolved), msg
                    elif kind == "linspace":
                        msg += f"\n\tLevels specified for {plotty}: numpy.linspace."
                        #adfobj.debug_log(msg)
                        return np.linspace(*resolved), msg
                #adfobj.debug_log(msg)
                return resolved, msg
            else:
                #adfobj.debug_log(msg)
                return entry, msg

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
                levels1, msg = process_entry(entry, kind, msg)
                if levels1 is not None:
                    break  # stop once a valid setting is found
        adfobj.debug_log(msg)
        return levels1


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
    msg = f"\n\tFinal case colormap: {cmap_case}"
    
    # CONTOUR LEVELS
    #---------------
    levels1 = resolve_levels(adfobj, "case", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tPre check levels: {type(levels1)}\n\t\t{levels1}\n"
    if levels1 is None:
        msg += "\n\tSetting the levels from max/min"
        levels1 = np.linspace(minval, maxval, 12)
    msg += f"\n\tFinal levels: {type(levels1)}\n\t\t{levels1}\n"

    # Check whether data exceeds limits
    vmin, vmax = levels1[0], levels1[-1]

    extend = 'neither'
    if minval < vmin and maxval > vmax:
        extend = 'both'
    elif minval < vmin:
        extend = 'min'
    elif maxval > vmax:
        extend = 'max'
    
    if kwargs.get('non_linear', False):
        cmap_obj = cm.get_cmap(cmap_case)
        norm1 = mpl.colors.BoundaryNorm(levels1, cmap_obj.N)
    else:
        norm1 = mpl.colors.Normalize(vmin=min(levels1), vmax=max(levels1))
    #End if

    #Check if the minval and maxval are actually different.  If not,
    #then set "levels1" to be an empty list, which will cause the
    #plotting scripts to add a label instead of trying to plot a variable
    #with no contours:
    if minval == maxval:
        levels1 = []
    #End if

    if ('colormap' not in plot_type_dict) and ('contour_levels' not in plot_type_dict):
        if ((minval < 0) and (0 < maxval)) and mplv > 2:
            norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        else:
            norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        #End if
    #End if
    
    # Difference options -- Check in kwargs for colormap and levels
    # determine levels & color normalization:
    minval = np.min(diffdata)
    maxval = np.max(diffdata)

    # COLOR MAP
    #----------
    cmap_diff = get_cmap(adfobj, "diff", plot_type_dict, kwargs, polar_names)
    msg += f"\n\tFinal difference colormap: {cmap_diff}\n"

    # CONTOUR LEVELS
    #---------------
    levelsdiff = resolve_levels(adfobj, "diff", plot_type_dict, kwargs, polar_names)

    msg += f"\n\tPre check difference levels: {type(levelsdiff)}\n\t\t{levelsdiff}\n"
    if levelsdiff is None:
        msg += f"\n\tSetting the difference levels from max/min"
        absmaxdif = np.max(np.abs(diffdata))
        levelsdiff = np.linspace(-absmaxdif, absmaxdif, 12)
    msg += f"\n\tFinal difference levels: {type(levelsdiff)}\n\t\t{levelsdiff}\n"

    # Check whether data exceeds limits
    vmin, vmax = levelsdiff[0], levelsdiff[-1]
    extend_diff = 'neither'
    if minval < vmin and maxval > vmax:
        extend_diff = 'both'
    elif minval < vmin:
        extend_diff = 'min'
    elif maxval > vmax:
        extend_diff = 'max'

    # color normalization for difference
    if ((np.min(levelsdiff) < 0) and (0 < np.max(levelsdiff))) and mplv > 2:
        normdiff = normfunc(vmin=np.min(levelsdiff), vmax=np.max(levelsdiff), vcenter=0.0)
    else:
        normdiff = mpl.colors.Normalize(vmin=np.min(levelsdiff), vmax=np.max(levelsdiff))

    # Percent Difference options -- Check in kwargs for colormap and levels
    # COLOR MAP
    #----------
    # determine levels & color normalization:
    #minval = np.min(diffdata)
    #maxval = np.max(diffdata)
    if "pct_diff_colormap" in plot_type_dict:
        cmappct = plot_type_dict["pct_diff_colormap"]
    else:
        cmappct = "PuOr_r"
    #End if

    if cmappct not in plt.colormaps():
        msg += f"\n\tPercent Difference: {cmappct} is not a matplotlib standard color map."
        msg += f" Trying if this an NCL color map"

        url = guess_ncl_url(cmappct)
        locfil = "." / f"{cmappct}.rgb"
        if locfil.is_file():
            data = read_ncl_colormap(adfobj, locfil)
        else:
            data = read_ncl_colormap(adfobj, url)
        cm, cmr = ncl_to_mpl(data, cmappct)
        #ncl_colors[cm.name] = cm
        #ncl_colors[cmr.name] = cmr
        
        if not cm:
            msg += f"\n\tPercent Difference: {cmappct} is not a matplotlib or NCL color map."
            cmappct = 'PuOr_r'
            msg += f" Defaulting to '{cmappct}'"
            adfobj.debug_log(msg)
        else:
            cmappct = cm

    # CONTOUR LEVELS
    #---------------
    if "pct_diff_contour_levels" in plot_type_dict:
        levelspctdiff = plot_type_dict["pct_diff_contour_levels"]  # a list of explicit contour levels
    elif "pct_diff_contour_range" in plot_type_dict:
            assert len(plot_type_dict['pct_diff_contour_range']) == 3, "pct_diff_contour_range must have exactly three entries: min, max, step"
            levelspctdiff = np.arange(*plot_type_dict['pct_diff_contour_range'])
    else:
        levelspctdiff = [-100,-75,-50,-40,-30,-20,-10,-8,-6,-4,-2,0,2,4,6,8,10,20,30,40,50,75,100]
    pctnorm = mpl.colors.BoundaryNorm(levelspctdiff,256)

    if "plot_log_pressure" in kwargs:
        plot_log_p = kwargs["plot_log_pressure"]
    else:
        plot_log_p = False

    # extract any MPL kwargs that should be passed on:
    subplots_opt = {}
    contourf_opt = {}
    colorbar_opt = {}
    diff_colorbar_opt = {}
    pct_colorbar_opt = {}

    if 'mpl' in kwargs:
        subplots_opt.update(kwargs['mpl'].get('subplots',{}))
        contourf_opt.update(kwargs['mpl'].get('contourf',{}))
        colorbar_opt.update(kwargs['mpl'].get('colorbar',{}))
        diff_colorbar_opt.update(kwargs['mpl'].get('diff_colorbar',{}))
        pct_colorbar_opt.update(kwargs['mpl'].get('pct_diff_colorbar',{}))
    #End ifs

    msg += "\n----------------------------------------------------\n"
    adfobj.debug_log(msg)

    return {'subplots_opt': subplots_opt,
            'contourf_opt': contourf_opt,
            'colorbar_opt': colorbar_opt,
            'diff_colorbar_opt': diff_colorbar_opt,
            'pct_colorbar_opt': pct_colorbar_opt,
            'normdiff': normdiff,
            'cmapdiff': cmap_diff,
            'levelsdiff': levelsdiff,
            'pctnorm': pctnorm,
            'cmappct': cmappct,
            'levelspctdiff':levelspctdiff,
            'cmap1': cmap_case,
            'norm1': norm1,
            'levels1': levels1,
            'plot_log_p': plot_log_p,
            'extend': extend,
            'extend_diff': extend_diff
            }


#####################
#END HELPER FUNCTIONS