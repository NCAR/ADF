import xarray as xr
import numpy as np
import warnings # use to warn user about missing files
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap ## used to create custom colormaps
import matplotlib.colors as mcolors
import matplotlib as mpl

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

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
    print("\n  Generating qbo plots...")

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

    if len(case_names) > 1:
        multi_case = True
        main_site_assets_path = adfobj.main_site_paths["main_site_assets_path"]
    else:
        multi_case = False
    #End if (check for multiple cases)

   

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
    plot_loc_ts  = Path(plot_locations[0]) / f'QBOts.{plot_type}'
    plot_loc_amp = Path(plot_locations[0]) / f'QBOamp.{plot_type}'

    #Until a multi-case plot directory exists, let user know
    #that the QBO plot will be kept in the first case directory:
    print(f"\t QBO plots will be saved here: {plot_locations[0]}")

    #Check if model vs model run, and if so, append baseline to case lists:
    if not adfobj.compare_obs:
        case_loc.append(base_loc)
        case_names.append(base_name)
    #End if

    #----Read in the OBS (ERA5, 5S-5N average already
    obs = xr.open_dataset(obsdir+"/U_ERA5_5S_5N_1979_2019.nc").U_5S_5N

    #----Read in the case data and baseline
    ncases = len(case_loc)

    casedat = [ _load_dataset(case_loc[i], case_names[i],'U') for i in range(0,ncases,1) ]

    #Find indices for all case datasets that don't contain a zonal wind field (U):
    bad_idxs = []
    for idx, dat in enumerate(casedat):
        if 'U' not in dat.variables:
            warnings.warn(f"QBO: case {case_names[idx]} contains no 'U' field, skipping...")
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
    casedatzm = [ casedat[i].U.mean("lon") for i in range(0,ncases,1) ]

    #----Calculate the 5S-5N average
    casedat_5S_5N = [ cosweightlat(casedatzm[i],-5,5) for i in range(0,ncases,1) ]

    #----Find the minimum number of years across dataset for plotting the timeseries.
    nyobs = np.floor(obs.time.size/12.)
    nycase = [ np.floor(casedat_5S_5N[i].time.size/12.) for i in range(0,ncases,1) ]
    nycase.append(nyobs)
    minny = np.int(np.min(nycase))

    #----QBO timeseries plots
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('QBO Time Series', fontsize=14)

    x1, x2, y1, y2 = plotpos()
    #ax = plotqbotimeseries(fig, obs, minny, x1[0], x2[0], y1[0], y2[0],'ERA5')

    if (adfobj.compare_obs) and (not multi_case):
        xo1 = 0.18
        xo2 = 0.45
        xm1 = 0.55
        xm2 = 0.82
        ax = plotqbotimeseries(fig, obs, minny, xo1, xo2, y1[0], y2[0],'ERA5')
    else:
        ax = plotqbotimeseries(fig, obs, minny, x1[0], x2[0], y1[0], y2[0],'ERA5')

    #casecount=0
    for icase in range(0,ncases,1):
        if (icase < 11 ): # only only going to work with 12 panels currently
            #Check if this is multi-case diagnostics
            if multi_case:
                if icase != ncases-1:
                    plot_loc_ts  = Path(plot_locations[icase]) / f'QBOts.{plot_type}'

                    #----QBO timeseries plots
                    fig_m = plt.figure(figsize=(16,16))
                    fig_m.suptitle('QBO Time Series', fontsize=14)

                    """#Plot ERA5 data
                    ax_m = plotqbotimeseries(fig_m, obs, minny, x1[0], x2[0], y1[0], y2[0],'ERA5')"""
                    
                    """#Plot individual case
                    ax_m = plotqbotimeseries(fig_m, casedat_5S_5N[icase],minny,
                                            x1[1],x2[1],y1[1],y2[1], 
                                            case_nicknames[icase])"""

                    #Check if compared vs baseline obs, alter x-positions
                    if adfobj.compare_obs:
                        """#Plot ERA5 data
                        ax_m = plotqbotimeseries(fig_m, obs, minny, 0.18, 0.45, y1[0], y2[0],'ERA5')

                        #Plot individual case
                        ax_m = plotqbotimeseries(fig_m, casedat_5S_5N[icase],minny,
                                                0.55, 0.82, y1[2], y2[2], 
                                                case_nicknames[icase])"""

                        xo1 = 0.18
                        xo2 = 0.45
                        xm1 = 0.55
                        xm2 = 0.82
                    #No observation baseline
                    else:
                        """#Plot ERA5 data
                        ax_m = plotqbotimeseries(fig_m, obs, minny, x1[0], x2[0], y1[0], y2[0],'ERA5')

                        #Plot individual case
                        ax_m = plotqbotimeseries(fig_m, casedat_5S_5N[icase],minny,
                                                x1[1],x2[1],y1[1],y2[1], 
                                                case_nicknames[icase])"""

                        xo1 = x1[0]
                        xo2 = x2[0]
                        xm1 = x1[1]
                        xm2 = x2[1]

                        ax_m = plotqbotimeseries(fig_m, casedat_5S_5N[-1],minny,
                                                    x1[2],x2[2],y1[2],y2[2], 
                                                    base_nickname)
                    #End if (compare obs)

                    #Plot ERA5 data
                    ax_m = plotqbotimeseries(fig_m, obs, minny, xo1, xo2, y1[0], y2[0],'ERA5')

                    #Plot individual case
                    ax_m = plotqbotimeseries(fig_m, casedat_5S_5N[icase],minny,
                                            xm1, xm2, y1[2], y2[2], 
                                            case_nicknames[icase])

                    #Plot colorbar
                    ax_m = plotcolorbar(fig_m, x1[0]+0.2, x2[2]-0.2,y1[2]-0.035,y1[2]-0.03)

                    #Save figure to file:
                    fig_m.savefig(plot_loc_ts, bbox_inches='tight', facecolor='white')

                    #Add plot to website (if enabled):
                    adfobj.add_website_data(plot_loc_ts, "QBO", case_names[icase], category=None, season="QBOts",
                                            multi_case=True,plot_type="Special")
            #End if (multi-case)

            #Check if compared vs baseline obs, alter x-positions
            if (adfobj.compare_obs) and (not multi_case):
                ax = plotqbotimeseries(fig, casedat_5S_5N[icase],minny,
                        xm1,xm2,y1[icase+1],y2[icase+1], case_nicknames[icase])
            else:
                ax = plotqbotimeseries(fig, casedat_5S_5N[icase],minny,
                        x1[icase+1],x2[icase+1],y1[icase+1],y2[icase+1], case_nicknames[icase])
            #casecount=casecount+1
        else:
            warnings.warn("The QBO diagnostics can only manage up to twelve cases!")
            break
        #End if
    #End for
    #ax = plotcolorbar(fig, x1[0]+0.2, x2[2]-0.2,y1[casecount]-0.035,y1[casecount]-0.03)
    ax = plotcolorbar(fig, x1[0]+0.2, x2[2]-0.2,y1[ncases]-0.035,y1[ncases]-0.03)
    
    if multi_case:#Notify user that script has started:
        print("\n  Generating qbo multi-case plots...")
        

        plot_loc_ts_multi = main_site_assets_path / f'QBO_QBOts_Special_multi_plot.{plot_type}'
        fig.savefig(plot_loc_ts_multi, bbox_inches='tight', facecolor='white')
        adfobj.add_website_data(plot_loc_ts_multi, "QBO", None, category=None, season="QBOts",
                                multi_case=True,plot_type="Special")
    
    else:
        #Save figure to file:
        fig.savefig(plot_loc_ts, bbox_inches='tight', facecolor='white')

        #Add plot to website (if enabled):
        #adfobj.add_website_data(plot_loc_ts, "QBO", None, season="QBOts", multi_case=True,plot_type = "Special") #multi_case=True
        adfobj.add_website_data(plot_loc_ts, "QBO", case_names[0], category=None, season="QBOts",
                                multi_case=True,plot_type="Special")
    #-----------------

    #---Dunkerton and Delisi QBO amplitude
    obsamp = calcddamp(obs)
    modamp = [ calcddamp(casedat_5S_5N[i]) for i in range(0,ncases,1) ]

    if multi_case:
        for icase in range(0,ncases,1):
            #Skip baseline case
            if icase != ncases-1:
                fig = plt.figure(figsize=(16,16))

                ax = fig.add_axes([0.05,0.6,0.4,0.4])
                ax.plot(modamp[icase], -np.log10(modamp[icase].lev), linewidth=2, label=case_nicknames[icase])
                
                #Check to plot baseline if not compared to obs
                if not adfobj.compare_obs:
                    ax.plot(modamp[-1], -np.log10(modamp[-1].lev), linewidth=2, label=base_nickname)

                ax.plot(obsamp, -np.log10(obsamp.pre), color='black', linewidth=2, label='ERA5')

                ax.set_ylim(-np.log10(150),-np.log10(1))
                ax.set_yticks([-np.log10(100),-np.log10(30),-np.log10(10),-np.log10(3),-np.log10(1)])
                ax.set_yticklabels(['100','30','10','3','1'], fontsize=12)
                ax.set_ylabel('Pressure (hPa)', fontsize=12)
                ax.set_xlabel('Dunkerton and Delisi QBO amplitude (ms$^{-1}$)', fontsize=12)
                ax.set_title('Dunkerton and Delisi QBO amplitude', fontsize=14)

                ax.legend(loc='upper left')

                plot_loc_amp = Path(plot_locations[icase]) / f'QBOamp.{plot_type}'

                fig.savefig(plot_loc_amp, bbox_inches='tight', facecolor='white')
                plt.close()
                #Add plot to website (if enabled):
                adfobj.add_website_data(plot_loc_amp, "QBO", case_names[icase], category = None, season="QBOamp", multi_case=True,plot_type = "Special")
            #End if (not baseline)
        #End for (cases)
    #End if (multi-case)
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

    #
    if multi_case:
        plot_loc_amp_multi = main_site_assets_path / f'QBO_QBOamp_Special_multi_plot.{plot_type}'
        fig.savefig(plot_loc_amp_multi, bbox_inches='tight', facecolor='white')

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc_amp_multi, "QBO", None, category=None, season="QBOamp",
                                multi_case=True,plot_type = "Special")
    else:
        fig.savefig(plot_loc_amp, bbox_inches='tight', facecolor='white')
        
        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc_amp, "QBO", case_names[0], category = None, season="QBOamp", multi_case=True,plot_type = "Special")
    
    #Close main fig
    plt.close()
    #-------------------

    #Notify user that script has ended:
    print("  ...QBO plots have been generated successfully.")

    #End QBO plotting script:
    return

#-------------------For Reading Data------------------------

def _load_dataset(data_loc, case_name, variable, other_name=None):
    """
    This method exists to get an xarray Dataset that can be passed into the plotting methods.

    This could (should) be changed to use an intake-esm catalog if (when) that is available.
    * At some point, we hope ADF will provide functions that can be used directly to replace this step,
      so the user will not need to know how the data gets saved.

    In this example, assume timeseries files are available via the ADF api.

    """

    dloc    = Path(data_loc)

    # a hack here: ADF uses different file names for "reference" case and regridded model data,
    # - try the longer name first (regridded), then try the shorter name

    fils = sorted(dloc.glob(f"{case_name}.*.{variable}.*.nc"))
    if (len(fils) == 0):
        warnings.warn("QBO: Input file list is empty.")
        return None
    elif (len(fils) > 1):
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)

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
    #ax.set_xlabel("Years",fontsize=12,labelpad=10)

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

def _set_ymargin(ax, top, bottom):
    """
    Allow for custom padding of plot lines and axes borders
    -----
    """
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    top = lim[1] + delta*top
    bottom = lim[0] - delta*bottom
    ax.set_ylim(bottom,top)


