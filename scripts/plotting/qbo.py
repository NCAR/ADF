import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import adf_utils as utils
import warnings # use to warn user about missing files
warnings.formatwarning = utils.my_formatwarning

def qbo(adfobj):
    """
    This subroutine plots...

    (1)  the times series of the 5S to 5N zonal mean U (QBOts.png)
     - this uses the same record length for each dataset and compares
       with ERA5.

    (2) the Dunkerton and Delisi QBO amplitude (QBOamp.png)
     - this uses the full record length for each dataset and compares
       with ERA5.

    Isla Simpson (islas@ucar.edu) 22nd March 2022

    """
    #Notify user that script has started:
    msg = "\n  Generating qbo plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Extract relevant info from the ADF:
    case_names = adfobj.get_cam_info('cam_case_name', required=True)
    case_loc = adfobj.get_cam_info('cam_ts_loc', required=True)
    base_name = adfobj.get_baseline_info('cam_case_name')
    base_loc = adfobj.get_baseline_info('cam_ts_loc')
    obsdir = adfobj.get_basic_info('obs_data_loc', required=True)
    plot_locations = adfobj.plot_location
    plot_type = adfobj.get_basic_info('plot_type')

    #Grab all case nickname(s)
    test_nicknames = adfobj.case_nicknames["test_nicknames"]
    base_nickname = adfobj.case_nicknames["base_nickname"]
    case_nicknames = test_nicknames + [base_nickname]

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    if not plot_type:
        plot_type = 'png'
    #End if

    #Check if zonal wind ("U") variable is present.  If not then skip
    #this script:
    if not ('U' in adfobj.diag_var_list):
        msg = "No zonal wind ('U') variable present"
        msg += " in 'diag_var_list', so QBO plots will"
        msg += " be skipped."
        print(msg)
        return
    #End if

    #Set path for QBO figures:
    plot_loc_ts  = Path(plot_locations[0]) / f'QBO_TimeSeries_Special_Mean.{plot_type}'
    plot_loc_amp = Path(plot_locations[0]) / f'QBO_Amplitude_Special_Mean.{plot_type}'

    #Until a multi-case plot directory exists, let user know
    #that the QBO plot will be kept in the first case directory:
    print(f"\t QBO plots will be saved here: {plot_locations[0]}")

    # Check redo_plot. If set to True: remove old plots, if they already exist:
    if (not redo_plot) and plot_loc_ts.is_file() and plot_loc_amp.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.debug_log(f"'{plot_loc_ts}' and '{plot_loc_amp}' exist and clobber is false.")
        adfobj.add_website_data(plot_loc_ts, "QBO", None, season="TimeSeries", multi_case=True, non_season=True, plot_type="Tropics")
        adfobj.add_website_data(plot_loc_amp, "QBO", None, season="Amplitude", multi_case=True, non_season=True, plot_type="Tropics")

        #Continue to next iteration:
        return
    elif (redo_plot):
        if plot_loc_ts.is_file():
            plot_loc_ts.unlink()
        if plot_loc_amp.is_file():
            plot_loc_amp.unlink()
    #End if

    #Check if model vs model run, and if so, append baseline to case lists:
    if not adfobj.compare_obs:
        case_loc.append(base_loc)
        case_names.append(base_name)
    #End if

    #----Read in the OBS (ERA5, 5S-5N average already
    obs = xr.open_dataset(obsdir+"/U_ERA5_5S_5N_1979_2019.nc").U_5S_5N

    #----Read in the case data and baseline
    ncases = len(case_loc)
    casedat = [utils.load_dataset(sorted(Path(case_loc[i]).glob(f"{case_names[i]}.*.U.*.nc"))) for i in range(0,ncases,1)]

    #Find indices for all case datasets that don't contain a zonal wind field (U):
    bad_idxs = []
    for idx, dat in enumerate(casedat):
        if 'U' not in dat.variables:
            warnings.warn(f"\t    WARNING: Case {case_names[idx]} contains no 'U' field, skipping...")
            bad_idxs.append(idx)
        #End if
    #End for

    #Pare list down to cases that actually contain a zonal wind field (U):
    if bad_idxs:
        for bad_idx in bad_idxs:
            casedat.pop(bad_idx)
        #End for
    #End if

    #----Calculate the zonal mean
    casedatzm = []
    for i in range(0,ncases,1):
        has_dims = utils.validate_dims(casedat[i].U, ['lon'])
        if not has_dims['has_lon']:
            print(f"\t    WARNING: Variable U is missing a lat dimension for '{case_loc[i]}', cannot continue to plot.")
        else:
            casedatzm.append(casedat[i].U.mean("lon"))
    if len(casedatzm) == 0:
        print(f"\t  WARNING: No available cases found, exiting script.")
        exitmsg = "\tNo QBO plots will be made."
        print(exitmsg)
        return
    if len(casedatzm) != ncases:
        print(f"\t  WARNING: Number of available cases does not match number of cases. Will exit script for now.")
        exitmsg = "\tNo QBO plots will be made."
        print(exitmsg)
        return

    #----Calculate the 5S-5N average
    casedat_5S_5N = [ cosweightlat(casedatzm[i],-5,5) for i in range(0,ncases,1) ]

    #----Find the minimum number of years across dataset for plotting the timeseries.
    nyobs = np.floor(obs.time.size/12.)
    nycase = [ np.floor(casedat_5S_5N[i].time.size/12.) for i in range(0,ncases,1) ]
    nycase.append(nyobs)
    minny = int(np.min(nycase))

    #----QBO timeseries plots
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('QBO Time Series', fontsize=14)

    x1, x2, y1, y2 = plotpos()
    ax = plotqbotimeseries(fig, obs, minny, x1[0], x2[0], y1[0], y2[0],'ERA5')

    casecount=0
    for icase in range(0,ncases,1):
        if (icase < 11 ): # only only going to work with 12 panels currently
            ax = plotqbotimeseries(fig, casedat_5S_5N[icase],minny,
                x1[icase+1],x2[icase+1],y1[icase+1],y2[icase+1], case_names[icase])
            casecount=casecount+1
        else:
            warnings.warn("The QBO diagnostics can only manage up to twelve cases!")
            break
        #End if
    #End for

    ax = plotcolorbar(fig, x1[0]+0.2, x2[2]-0.2,y1[casecount]-0.035,y1[casecount]-0.03)

    #Save figure to file:
    fig.savefig(plot_loc_ts, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_ts, "QBO", None, season="TimeSeries", multi_case=True, non_season=True, plot_type="Tropics")

    #-----------------

    #---Dunkerton and Delisi QBO amplitude
    obsamp = calcddamp(obs)
    modamp = [ calcddamp(casedat_5S_5N[i]) for i in range(0,ncases,1) ]

    fig = plt.figure(figsize=(16,16))

    ax = fig.add_axes([0.05,0.6,0.4,0.4])
    ax.set_ylim(-np.log10(150),-np.log10(1))
    ax.set_yticks([-np.log10(100),-np.log10(30),-np.log10(10),-np.log10(3),-np.log10(1)])
    ax.set_yticklabels(['100','30','10','3','1'], fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_xlabel('Dunkerton and Delisi QBO amplitude (ms$^{-1}$)', fontsize=12)
    ax.set_title('Dunkerton and Delisi QBO amplitude', fontsize=14)

    ax.plot(obsamp, -np.log10(obsamp.pre), color='black', linewidth=2, label='ERA5')

    for icase in range(0,ncases,1):
        ax.plot(modamp[icase], -np.log10(modamp[icase].lev), linewidth=2, label=case_nicknames[icase])

    ax.legend(loc='upper left')
    fig.savefig(plot_loc_amp, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_amp, "QBO", None, season="Amplitude", multi_case=True, non_season=True, plot_type="Tropics")

    #-------------------

    #Notify user that script has ended:
    print("  ...QBO plots have been generated successfully.")

    #End QBO plotting script:
    return

#-----------------For Calculating-----------------------------

def cosweightlat(darray, lat1, lat2):
    """Calculate the weighted average for an [:,lat] array over the region
    lat1 to lat2
    """

    # flip latitudes if they are decreasing
    if (darray.lat[0] > darray.lat[darray.lat.size -1]):
        print("QBO: flipping latitudes")
        darray = darray.sortby('lat')

    region = darray.sel(lat=slice(lat1, lat2))
    weights=np.cos(np.deg2rad(region.lat))
    regionw = region.weighted(weights)
    regionm = regionw.mean("lat")

    return regionm

def calcddamp(data):
    """Calculate the Dunkerton and Delisi QBO amplitude"""
    datseas = data.groupby('time.month').mean('time')
    datdeseas = data.groupby('time.month')-datseas
    ddamp = np.sqrt(2)*datdeseas.std(dim='time')
    return ddamp


#---------------------------------For Plotting------------------------------------------
def plotpos():
    """ Positionings to position the plots nicely (3x4)"""
    x1 = [0.05,0.37,0.69,0.05,0.37,0.69,0.05,0.37,0.69,0.05,0.37,0.69]
    x2 = [0.32,0.64,0.95,0.32,0.64,0.95,0.32,0.64,0.95,0.32,0.64,0.95]
    y1 = [0.8,0.8,0.8,0.59,0.59,0.59,0.38,0.38,0.38,0.17,0.17,0.17]
    y2 = [0.95,0.95,0.95,0.74,0.74,0.74,0.53,0.53,0.53,0.32,0.32,0.32]
    return x1, x2, y1, y2

def plotqbotimeseries(fig, dat, ny, x1, x2, y1, y2, title):
    """ Function for plotting each QBO time series panel

    Input:

    fig = the figure axis
    dat = the data to plot of the form (time, lev)
    ny = the number of years to plot
    x1, x2, y1, y2 = plot positioning arguments

    """

    ax = fig.add_axes([x1, y1, (x2-x1), (y2-y1)])
    datplot = dat.isel(time=slice(0,ny*12)).transpose()
    ci = 1 ; cmax=45
    nlevs = (cmax - (-1*cmax))/ci + 1
    clevs = np.arange(-1*cmax, cmax+ci, ci)
    mymap = blue2red_cmap(nlevs)

    plt.rcParams['font.size'] = '12'

    if "lev" in datplot.dims:
        ax.contourf(datplot.time.dt.year + (datplot.time.dt.month/12.), -1.*np.log10(datplot.lev), datplot,
                levels = clevs, cmap=mymap, extent='both')
    elif "pre" in datplot.dims:
        ax.contourf(datplot.time.dt.year + (datplot.time.dt.month/12.), -1.*np.log10(datplot.pre), datplot,
                levels = clevs, cmap=mymap, extent='both')
    else:
        raise ValueError("Cannot find either 'lev' or 'pre' in datasets for QBO diagnostics")

    ax.set_ylim(-np.log10(1000.), -np.log10(1))
    ax.set_yticks([-np.log10(1000),-np.log10(300),-np.log10(100),-np.log10(30),-np.log10(10),
                   -np.log10(3),-np.log10(1)])
    ax.set_yticklabels(['1000','300','100','30','10','3','1'])
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_title(title, fontsize=14)

    return ax

def plotcolorbar(fig, x1, x2, y1, y2):
    """ Plotting the color bar at location [x1, y1, x2-x2, y2-y1 ] """
    ci = 1 ; cmax=45
    nlevs = (cmax - (-1*cmax))/ci + 1
    clevs = np.arange(-1.*cmax, cmax+ci, ci)
    mymap = blue2red_cmap(nlevs)

    ax = fig.add_axes([x1, y1, x2-x1, y2-y1])
    norm = mpl.colors.Normalize(vmin=-1.*cmax, vmax=cmax)

    clb = mpl.colorbar.ColorbarBase(ax, cmap=mymap,
           orientation='horizontal', norm=norm, values=clevs)

    clb.ax.tick_params(labelsize=12)
    clb.set_label('U (ms$^{-1}$)', fontsize=14)

    return ax

def blue2red_cmap(n, nowhite = False):
    """ combine two existing color maps to create a diverging color map with white in the middle
    n = the number of contour intervals
    """

    if (int(n/2) == n/2):
        # even number of contours
        nwhite=1
        nneg=n/2
        npos=n/2
    else:
        nwhite=2
        nneg = (n-1)/2
        npos = (n-1)/2

    if (nowhite):
        nwhite=0

    colors1 = plt.cm.Blues_r(np.linspace(0,1, int(nneg)))
    colors2 = plt.cm.YlOrRd(np.linspace(0,1, int(npos)))
    colorsw = np.ones((nwhite,4))

    colors = np.vstack((colors1, colorsw, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return mymap