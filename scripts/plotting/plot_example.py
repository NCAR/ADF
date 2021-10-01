def plot_example(case_name, model_rgrid_loc, data_name, data_loc,
                 var_list, data_list, plot_location):

    """
    This is an example function showing how to set-up a
    plotting script that compares CAM climatologies against
    other climatological data (observations or baseline runs).

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
    var_list         -> List of CAM output variables provided by "diag_var_list"
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".
    """

    #Import necessary modules:
    #------------------------
    from pathlib import Path  # python standard library

    # data loading / analysis
    import xarray as xr
    import numpy as np
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("  Generating diagnostic plots...")

    #Set input/output data path variables:
    #------------------------------------
    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_loc      = Path(plot_location)
    #-----------------------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Set plot file type:
    plot_type = 'png'

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)

    # probably want to do this one variable at a time:
    for var in var_list:

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))

            if len(oclim_fils) > 1:
                oclim_ds = xr.open_mfdataset(oclim_fils, combine='by_coords')
            else:
                sfil = str(oclim_fils[0])
                oclim_ds = xr.open_dataset(sfil)

            # load re-gridded model files:
            mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))

            if len(mclim_fils) > 1:
                mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
            else:
                mclim_ds = xr.open_dataset(mclim_fils[0])

            #Extract variable of interest
            odata = oclim_ds[var].squeeze()  # squeeze in case of degenerate dimensions
            mdata = mclim_ds[var].squeeze()

            #
            # Seasonal Averages
            # Note: xarray can do seasonal averaging, but depends on having time accessor, which these prototype climo files don't.
            #

            #Create new dictionaries:
            mseasons = dict()
            oseasons = dict()
            dseasons = dict() # hold the differences

            #Loop over season dictionary:
            for s in seasons:
                mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
                # difference: each entry should be (lat, lon)
                dseasons[s] = mseasons[s] - oseasons[s]

                # time to make plot; here we'd probably loop over whatever plots we want for this variable
                # I'll just call this one "Mean_LatLon"  ... would this work as a pattern [operation]_[AxesDescription] ?
                plot_name = plot_loc / "{}_{}_Mean_LatLon.{}".format(var, s, plot_type)

                #Remove old plot, if it already exists:
                if plot_name.is_file():
                    plot_name.unlink()

                #Create new plot:
                plot_map_and_save(plot_name, mseasons[s], oseasons[s], dseasons[s])

    #Notify user that script has ended:
    print("  ...Diagnostic plots have been generated successfully.")

#################
#HELPER FUNCTIONS
#################

def use_this_norm():
    """Just use the right normalization; avoids a deprecation warning."""

    #Import statements:
    import matplotlib as mpl

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

    #import statements:
    import numpy as np

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

    #import statements:
    import numpy as np
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point

    #Set non-X-window backend for matplotlib:
    mpl.use('Agg')

    #Now import pyplot:
    import matplotlib.pyplot as plt

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
        # add contour lines
        cs.append(ax[i].contour(lon2, lat2, fields[i], transform=ccrs.PlateCarree(), colors='k'))
        ax[i].clabel(cs[i], cs[i].levels, inline=True, fontsize=tiFontSize-2, fmt='%1.1f')
        ax[i].text( 10, -140, "CONTOUR FROM {} to {} by {}".format(min(cs[i].levels), max(cs[i].levels), cs[i].levels[1]-cs[i].levels[0]),
        bbox=dict(facecolor='none', edgecolor='black'), fontsize=tiFontSize-2)

    # set rmse title:
    ax[-1].set_title("RMSE: {0:.3f}".format(d_rmse), fontsize=tiFontSize)

    for a in ax:
        a.outline_patch.set_linewidth(2)
        a.coastlines()
        a.set_xticks(np.linspace(-180, 180, 7), crs=ccrs.PlateCarree())
        a.set_yticks(np.linspace(-90, 90, 7), crs=ccrs.PlateCarree())
        a.tick_params('both', length=10, width=2, which='major')
        a.tick_params('both', length=5, width=1, which='minor')
    fig.savefig(wks, bbox_inches='tight', dpi=300)

##############
#END OF SCRIPT
