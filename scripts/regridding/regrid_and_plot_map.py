from pathlib import Path  # python standard library
import logging

# data loading / analysis
import xarray as xr
import numpy as np

# regridding
# Try just using the xarray method
# import xesmf as xe  # This package is for regridding, and is just one potential solution.

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import cartopy.crs as ccrs

logging.basicConfig(level=logging.INFO)

logging.info(f"XARRAY VERSION: {xr.__version__}")
logging.info(f"NUMPY VERSION: {np.__version__}")
logging.info(f"MATPLOTLIB VERSION: {mpl.__version__}")

# Steps: 
# - load climo files for model and obs
# - calculate all-time and seasonal fields (from individual months)
# - regrid one to the other (probably should be a choice)
# - Take difference, calculate statistics
# - make plot


def use_this_norm():
    """Just use the right normalization; avoids a deprecation warning."""
    mplversion = [int(x) for x in mpl.__version__.split('.')]
    if mplversion[0] < 3:
        logging.warning("Your Matplotlib version is pretty old.")
        return mpl.colors.DivergingNorm
    else:
        if mplversion[1] < 2:
            return mpl.colors.DivergingNorm
        else:
            return mpl.colors.TwoSlopeNorm



def global_average(fld, wgt, verbose=False):
    """A simple, pure numpy global average.
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
        print(f"(global_average)-- fraction input missing: {fraction_nan(fld)}")
        print(f"(global_average)-- fraction of mask that is True: {np.count_nonzero(fld2.mask) / np.size(fld2)}")
        print(f"(global_average)-- apply ma.average along axis = {a} // validate: {fld2.shape}")
    avg1, sofw = np.ma.average(fld2, axis=a, weights=wgt, returned=True) # sofw is sum of weights
    return np.ma.average(avg1)


def regrid_data(fromthis, tothis):
    return fromthis.interp_like(tothis)


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
    normfunc = use_this_norm()
    if np.sign(minval) != np.sign(maxval):
        norm1 = normfunc(vmin=minval, vmax=maxval, vcenter=0.0)
        cmap1 = 'coolwarm'
    else:
        norm1 = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        cmap1 = 'cividis'
    
    if 'cnLevels' in kwargs:
        levels1 = kwargs.pop(cnLevels)
    else:
        levels1 = np.linspace(minval, maxval, 12)
    # set a symmetric color bar for diff:
    absmaxdif = np.max(np.abs(diffld))
    
    fig, ax = plt.subplots(figsize=(6,12), nrows=3, subplot_kw={"projection":ccrs.PlateCarree()})
    img = []  # contour plots
    cs = []  # contour lines
    cb = []  # color bars
    for i, a in enumerate(wrap_fields):
        if i == len(wrap_fields)-1:
            levels = 12
            norm = normfunc(vmin=-1*absmaxdif, vcenter=0., vmax=absmaxdif)
            cmap = 'coolwarm'
        else:
            levels = levels1
            cmap = cmap1
            norm = norm1
        img.append(ax[i].contourf(lons, lats, a, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), **kwargs))
        cb.append(fig.colorbar(img[i], ax=ax[i], shrink=0.8))
        ax[i].set_title(f"AVG: {area_avg[i]:6.3f}", loc='right', fontsize=tiFontSize)
        # add contour lines
        cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k'))
        ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        ax[i].text( 10, -140, f"CONTOUR FROM {min(cs[i].levels)} to {max(cs[i].levels)} by {cs[i].levels[1]-cs[i].levels[0]}", bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)
                   
    # set rmse title:
    ax[-1].set_title(f"RMSE: {d_rmse:6.3f}", fontsize=tiFontSize)

    for a in ax:
        a.outline_patch.set_linewidth(2)
        a.coastlines()
        a.set_xticks(np.linspace(-180, 180, 7), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=10, width=2, which='major')
        a.tick_params('both', length=5, width=1, which='minor')
    fig.savefig(wks, bbox_inches='tight', dpi=300)

#
# --- Main Body ---
#
logging.debug("Starting.")
oclimo_loc = Path('/glade/work/brianpm/observations/climo_files')
mclimo_loc = Path('/glade/scratch/brianpm/post_process/climo_files')
mcase_name = 'f2000.v211.fv1d.trop.L42.1'
variables = ['LWCF', 'SWCF']
seasons = {"ANN": np.arange(1,13,1), 
           "DJF": [12, 1, 2],
           "JJA": [6, 7, 8],
           "MAM": [3, 4, 5],
           "SON": [9, 10, 11]}
regrid_file = '/glade/scratch/brianpm/post_process/regrid_files/bilinear_cesmf09_to_ceres1deg.nc'
plot_root = Path('/glade/scratch/brianpm/post_process/diagnostic_plots')
plot_location = (plot_root / f'{mcase_name}_vs_obs')
if not plot_location.is_dir():
    plot_location.mkdir(parents=True, exist_ok=True)
plot_type = 'png'

logging.debug(f"Plots are going to be sent to {plot_location}")

logging.debug("Initialization done, let's loop through variables.")
# probably want to do this one variable at a time:
for v in variables:
    logging.debug(f"Working on variable {v}")
    # load files (we should explore intake as an alternative to having this kind of repeated code)
    oclim_fils = sorted(list(oclimo_loc.glob(f"*_{v}_*")))  # this will match the naming scheme used in postproc.py / post_process_climo.json
    logging.debug(f"Found {len(oclim_fils)} files: {oclim_fils}")
    if len(oclim_fils) > 1:
        oclim_ds = xr.open_mfdataset(oclim_fils, combine='by_coords')
    else:
        sfil = str(oclim_fils[0])
        oclim_ds = xr.open_dataset(sfil)
    mclim_fils = sorted(list((mclimo_loc / mcase_name).glob(f"*_{v}_*")))
    logging.debug(f"Found {len(mclim_fils)} model files.")
    if len(mclim_fils) > 1:
        mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
    else:
        mclim_ds = xr.open_dataset(mclim_fils[0])
    odata = oclim_ds[v]
    mdata = mclim_ds[v]
    #
    # Seasonal Averages
    # Note: xarray can do seasonal averaging, but depends on having time accessor, which these prototype climo files don't.
    #
    mseasons = dict()
    oseasons = dict()
    mseasons_regrid = dict()
    dseasons = dict() # hold the differences
    for s in seasons:
        logging.debug(f"Working on season: {s}")
        mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
        oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
        mseasons_regrid[s] = regrid_data(mseasons[s], oseasons[s])
        # difference: each entry should be (lat, lon)
        dseasons[s] = mseasons_regrid[s] - oseasons[s]
        # time to make plot; here we'd probably loop over whatever plots we want for this variable
        # I'll just call this one "Mean_LatLon"  ... would this work as a pattern [operation]_[AxesDescription] ?
        plot_name = plot_location / f'{v}_{s}_Mean_LatLon.{plot_type}'
        logging.info(f"Send to file {plot_name}")
        plot_map_and_save(plot_name, mseasons_regrid[s], oseasons[s], dseasons[s])
        