import os.path
import sys
import glob
from pathlib import Path

# Import necessary packages for the new script
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

import xarray as xr

#
# load libraries: ncar_pylib
#
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from math import nan
import pandas as pd
import sys

import dask

def tape_recorder(adfobj):
    """
    Plot the values of Q against two sets of obseravations, MLS and ERA5, for the tropics
    between 10S and 10N.

    MLS h2o data is for 04/2009-11/2021
    ERA5 Q data is for 01/1980-12/2020

    This will calculate the weighted latitude average for the simulations/baseline observations in the tropics.
    NOTE: If the baseline case are observations and ERA5 is set for Q in the obs location, it will be ignored
        since ERA5 is used as a defualt obs in the tape recorder.
    
    
    
    """
    #Notify user that script has started:
    print("\n  Generating tape recorder plots...")

    #Special ADF variable which contains the output paths for
    # ADF variable which contains the output path for plots and tables:
    plot_location = adfobj.plot_location
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            #Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                print(f"\t    {pl} not found, making new directory")
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            print(f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}")
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)

    case_names = adfobj.get_cam_info('cam_case_name', required=True)
    data_name = adfobj.get_baseline_info('cam_case_name', required=False)

    case_ts_locs = adfobj.get_cam_info("cam_ts_loc", required=True)
    data_ts_loc = adfobj.get_baseline_info("cam_ts_loc", required=False)

    start_years = adfobj.climo_yrs["syears"]
    end_years = adfobj.climo_yrs["eyears"]

    data_start_year = adfobj.climo_yrs["syear_baseline"]
    data_end_year = adfobj.climo_yrs["eyear_baseline"]

    #Grab test case nickname(s)
    test_nicknames = adfobj.get_cam_info('case_nickname')
    if test_nicknames == None:
        test_nicknames = case_names

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if not adfobj.get_basic_info("compare_obs"):
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST

        #Grab baseline case nickname
        base_nickname = adfobj.get_baseline_info('case_nickname')
        if base_nickname == None:
            base_nickname = data_name
        case_ts_locs = case_ts_locs+[data_ts_loc]
        test_nicknames = test_nicknames+[base_nickname]
        start_years = start_years+[data_start_year]
        end_years = end_years+[data_end_year]
    #End if
    
    # Default colormap
    cmap='precip_nowhite'

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    #-----------------------------------------
    
    runs_LT2={}
    for i,val in enumerate(test_nicknames):
        runs_LT2[val] = case_ts_locs[i]

    # MLS data
    mls = xr.open_dataset("/glade/campaign/cgd/cas/islas/CAM7validation/MLS/mls_h2o_latNpressNtime_3d_monthly_v5.nc")
    mls = mls.rename(x='lat', y='lev', t='time')
    time = pd.date_range("2004-09","2021-11",freq='M')
    mls['time'] = time
    mls = cosweightlat(mls.H2O,-10,10)
    mls = mls.groupby('time.month').mean('time')
    mls = mls*18.015280/(1e6*28.964)

    # ERA5 data
    era5 = xr.open_dataset("/glade/campaign/cgd/cas/islas/CAM7validation/ERA5/ERA5_Q_10Sto10N_1980to2020.nc")
    era5 = era5.groupby('time.month').mean('time')

    alldat=[]
    runname_LT=[]
    for key in runs_LT2:
        dat = xr.open_dataset(glob.glob(runs_LT2[key]+'/*h0.Q.*.nc')[0])
        dat = fixcesmtime(dat)
        datzm = dat.mean('lon')
        dat_tropics = cosweightlat(datzm.Q, -10, 10)
        dat_mon = dat_tropics.groupby('time.month').mean('time').load()
        alldat.append(dat_mon)
        runname_LT.append(key)

    runname_LT=xr.DataArray(runname_LT, dims='run', coords=[np.arange(0,len(runname_LT),1)], name='run')
    alldat_concat_LT = xr.concat(alldat, dim=runname_LT)

    fig = plt.figure(figsize=(16,16))
    x1, x2, y1, y2 = get5by5coords_zmplots()

    plot_step = 0.5e-7
    plot_min = 1.5e-6
    plot_max = 3e-6

    #1e-7,1.5e-6,3e-6
    ax = plot_pre_mon(fig, mls, mls.lev, plot_step,plot_min,plot_max,'MLS',
                      x1[0],x2[0],y1[0],y2[0],cmap=cmap, paxis='lev',
                      taxis='month')

    ax = plot_pre_mon(fig, era5.Q, era5.pre, plot_step,plot_min,plot_max,
                      'ERA5',x1[1],x2[1],y1[1],y2[1], cmap=cmap, paxis='pre',
                      taxis='month')

    #Start count at 2 to account for MLS and ERA5 plots above
    count=2
    for irun in np.arange(0,alldat_concat_LT.run.size,1):
        title = f"{alldat_concat_LT.run.isel(run=irun).values}"
        ax = plot_pre_mon(fig, alldat_concat_LT.isel(run=irun), 
                          alldat_concat_LT.isel(run=irun).lev,
                          plot_step, plot_min, plot_max, title,
                          x1[count],x2[count],y1[count],y2[count],cmap=cmap, paxis='lev',
                          taxis='month',climo_yrs=f"{start_years[irun]}-{end_years[irun]}")
        count=count+1
    
    #Shift colorbar if there are less than 5 subplots
    # There will always be at least 2 (MLS and ERA5)
    if len(case_ts_locs) == 0:
        print("Seems like there are no simulations to plot, exiting script.")
        return
    if len(case_ts_locs) == 1:
        x1_loc = (x1[1]-x1[0])/2
        x2_loc = ((x2[2]-x2[1])/2)+x2[1]
    elif len(case_ts_locs) == 2:
        x1_loc = (x1[1]-x1[0])/2
        x2_loc = ((x2[3]-x2[2])/2)+x2[2]
    else:
        x1_loc = x1[0]
        x2_loc = x2[3]

    y1_loc = y1[count]-0.03
    y2_loc = y1[count]-0.02

    ax = plotcolorbar(fig, plot_step, plot_min, plot_max, 'Q (kg/kg)',
                      x1_loc, x2_loc, y1_loc, y2_loc,
                      cmap=cmap)

    plot_name = plot_loc / f"TapeRecorder_ANN_Special_Mean.{plot_type}"
    print(f"\t - Plotting annual tape recorder for Q")

    # Check redo_plot. If set to True: remove old plot, if it already exists:
    if (not redo_plot) and plot_name.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.add_website_data(plot_name, "tape_recorder", None, season="ANN", multi_case=True)

    elif (redo_plot) and plot_name.is_file():
        plot_name.unlink()

    #Save image
    fig.savefig(plot_name, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_name, "tape_recorder", None, season="ANN", multi_case=True)

    #Notify user that script has ended:
    print("  ...Tape recorder plots have been generated successfully.")

    #End QBO plotting script:
    return


# Helper Functions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap ## used to create custom colormaps
import matplotlib.colors as mcolors
import numpy as np

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

def red2blue_cmap(n, nowhite = False):
    """ combine two existing color maps to create a diverging color map with white in the middle
    n = the number of contour intervals """

    if (int(n/2) == n/2):
        #even number of contours
        nwhite=1
        nneg = n/2
        npos = n/2
    else:
        nwhite=2
        nneg = (n-1)/2
        npos = (n-1)/2

    if (nowhite):
        nwhite=0

    colors1 = plt.cm.YlOrRd_r(np.linspace(0.1,1,int(npos)))
    colors2 = plt.cm.Blues(np.linspace(0.1,1,int(nneg)))
    colorsw = np.ones((nwhite,4))

    if (nowhite):
        colors = np.vstack( (colors1, colors2))
    else:
        colors = np.vstack((colors1, colorsw, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
  
    return mymap



def precip_cmap(n, nowhite=False):
    """ combine two existing color maps to create a diverging color map with white in the middle.
    browns for negative, blues for positive
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

    colors1 = plt.cm.YlOrBr_r(np.linspace(0,1, int(nneg)))
    colors2 = plt.cm.GnBu(np.linspace(0,1, int(npos)))
    colorsw = np.ones((nwhite,4))

    colors = np.vstack((colors1, colorsw, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return mymap

def precip_cmap_nowhite(n):
    """ combine two existing color maps to create a diverging color map with white in the middle.
    browns for negative, blues for positive
    n = the number of contour intervals
    """
    if (int(n/2) == n/2):
        # even number of contours
        nneg=n/2
        npos=n/2
    else:
        nneg = (n-1)/2
        npos = (n-1)/2

    colors1 = plt.cm.YlOrBr_r(np.linspace(0,0.8, int(nneg)))
    colors2 = plt.cm.GnBu(np.linspace(0.2,1, int(npos)))

    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    return mymap



def fixcesmtime(dat,timebndsvar='time_bnds'):
    """ Fix the CESM timestamp using the average of time_bnds"""

    try:
        timebndavg = np.array(dat.isel(M=0)[timebndsvar],
                  dtype='datetime64[s]').view('i8').mean(axis=1).astype('datetime64[s]')
        dat['time'] = timebndavg
    except:
        timebndavg = np.array(dat[timebndsvar],
                  dtype='datetime64[s]').view('i8').mean(axis=1).astype('datetime64[s]')
        dat['time'] = timebndavg

    return dat


def get5by5coords_zmplots():
    """ positioning for 5x5 plots """
    x1 = [0.02,0.225,0.43,0.635,0.84,
          0.02,0.225,0.43,0.635,0.84,
          0.02,0.225,0.43,0.635,0.84,
          0.02,0.225,0.43,0.635,0.84,
          0.02,0.225,0.43,0.635,0.84] 
    x2 = [0.18,0.385,0.59,0.795,1,
          0.18,0.385,0.59,0.795,1,
          0.18,0.385,0.59,0.795,1,
          0.18,0.385,0.59,0.795,1,
          0.18,0.385,0.59,0.795,1]
    y1 = [0.8,0.8,0.8,0.8,0.8,
          0.6,0.6,0.6,0.6,0.6,
          0.4,0.4,0.4,0.4,0.4,
          0.2,0.2,0.2,0.2,0.2,
          0.,0.,0.,0.,0.]
    y2 = [0.95,0.95,0.95,0.95,0.95,
          0.75,0.75,0.75,0.75,0.75,
          0.55,0.55,0.55,0.55,0.55,
          0.35,0.35,0.35,0.35,0.35,
          0.15,0.15,0.15,0.15,0.15]
    
    return x1, x2, y1, y2



import matplotlib.pyplot as plt
import matplotlib as mpl

import importlib

import numpy as np

def plotcolorbar(fig, ci, cmin, cmax, titlestr, x1, x2, y1, y2, 
   cmap='blue2red', orient='horizontal', posneg='both', ticks=None, fsize=14, nowhite=False,
   contourlines=False, contourlinescale=1):
    """plot a color bar
       Input:
           fig = the figure identified
           ci = the contour interval for the color map
           cmin = the minimum extent of the contour range
           cmax = the maximum extent of the contour range
           titlestr = the label for the color bar
           x1 = the location of the left edge of the color bar
           x2 = the location of the right edge of the color bar
           y1 = the location of the bottom edge of the color bar
           y2 = the location of the top edge of the color bar
           cmap = the color map to be used (only set up for blue2red at the moment)
           orient = the orientation (horizontal or vertical)
           posneg = if "both", both positive and negative sides are plotted
                    if "pos", only the positive side is plotted
                    if "net", only the negative side is plotted
           ticks = user specified ticklabels
           fsize = user specified font size
           contourlines = used to overplot contour lines
           contourlinescale = scale factor for contour lines to be overplotted
    """

    # set up contour levels and color map
    nlevs = (cmax-cmin)/ci + 1
    clevs = ci * np.arange(cmin/ci, (cmax+ci)/ci, 1) 

    if (cmap == "blue2red"):
        mymap = blue2red_cmap(nlevs, nowhite)

    if (cmap == "precip"):
        mymap = precip_cmap(nlevs, nowhite)

    if (cmap == "precip_nowhite"):
        mymap = precip_cmap_nowhite(nlevs)

    if (cmap == 'red2blue'):
        mymap = red2blue_cmap(nlevs, nowhite)

    clevplot=clevs
    if (posneg == "pos"):
        clevplot = clevs[clevs >= 0]
    if (posneg == "neg"):
        clevplot = clevs[clevs <= 0]

    ax = fig.add_axes([x1, y1, x2-x1, y2-y1])
    norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
    
    if (ticks):
        clb = mpl.colorbar.ColorbarBase(ax, cmap=mymap,
           orientation=orient, norm=norm, values=clevplot, ticks=ticks)
    else:
        clb = mpl.colorbar.ColorbarBase(ax, cmap=mymap, 
           orientation=orient, norm=norm, values=clevplot)

    clb.ax.tick_params(labelsize=fsize)
    clb.set_label(titlestr, fontsize=fsize+2)

    if (contourlines):
        #clevlines = (clevs-ci/2.)*contourlinescale
        clevlines = clevs*contourlinescale
        clevlines = clevlines[np.abs(clevlines) > ci/2.]
        if (orient=='horizontal'):
            ax.vlines(clevlines[clevlines > 0],-5,5, colors='black', linestyle='solid')
            ax.vlines(clevlines[clevlines < 0],-5,5, colors='black', linestyle='dashed')
        if (orient=='vertical'):
            ax.hlines(clevlines[clevlines > 0],-10,15, colors='black', linestyle='solid')
            ax.hlines(clevlines[clevlines < 0],-10,15, colors='black', linestyle='dashed')

    return ax


def cosweightlat(darray, lat1, lat2):
    """Calculate the weighted average for an [:,lat] array over the region
    lat1 to lat2
    """

    # flip latitudes if they are decreasing
    if (darray.lat[0] > darray.lat[darray.lat.size -1]):
        print("flipping latitudes")
        darray = darray.sortby('lat')

    region = darray.sel(lat=slice(lat1, lat2))
    weights=np.cos(np.deg2rad(region.lat))
    regionw = region.weighted(weights)
    regionm = regionw.mean("lat")

    return regionm


import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_pre_mon(fig, data, pre, ci, cmin, cmax, expname, x1=None, x2=None, y1=None, y2=None, 
                 oplot=False, ax=None, cmap='precip', taxis='time', paxis='lev', climo_yrs=None):
    """
    Plot seasonal cycle, pressure versus time.
    """

    # move the time axis to the first
    if (data.dims[1] != taxis):
        data = data.transpose(..., taxis)

    nlevs = (cmax - cmin)/ci + 1
    clevs = np.arange(cmin, cmax+ci, ci)

    if (cmap == "blue2red"):
        mymap = blue2red_cmap(nlevs)

    if (cmap == "precip"):
        mymap = precip_cmap(nlevs)

    if (cmap == "precip_nowhite"):
        mymap = precip_cmap_nowhite(nlevs)

    # if overplotting, check for axis input
    if (oplot and (not ax)):
        print("This isn't going to work.  If overplotting, specify axis")
        sys.exit()

    plt.rcParams['font.size'] = '14'

    monticks_temp = np.arange(0,12,1)
    monticks2_temp = np.arange(0,12,1)+0.5

    monticks = monticks_temp
    monticks2 = np.zeros([len(monticks2_temp)+2])
    monticks2[0] = -0.5 ; monticks2[len(monticks2)-1] = 12.5
    monticks2[1:len(monticks2)-1] = monticks2_temp

    dataplot = np.zeros([data[paxis].size,len(monticks2)])
    dataplot[:,0] = data[:,11]
    dataplot[:,len(monticks2)-1] = data[:,0]
    dataplot[:,1:len(monticks2)-1] = data[:,:]

    #Check for over plotting
    if not oplot:
        if (x1):
            ax = fig.add_axes([x1, y1, x2-x1, y2-y1])
        else:
            ax = fig.add_axes()
    if climo_yrs:
        ax.set_title(f"\n{climo_yrs}", loc='right',
                       fontsize=8)

    ax.contourf(monticks2, -np.log10(data[paxis]), dataplot, levels=clevs, cmap=mymap, extend='max')
    ax.set_ylim(-np.log10(100),-np.log10(3))
    ax.set_yticks([-np.log10(100),-np.log10(30),-np.log10(10),-np.log10(3)])
    ax.set_yticklabels(['100','30','10','3'])
    ax.set_xlim([0,12])
    ax.tick_params(which='minor', length=0)
    ax.set_xticks(monticks)
    ax.set_xticklabels([])
    ax.set_xticks(monticks2[1:13], minor=True)
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], minor=True, fontsize=14)
    ax.set_title(expname, fontsize=16)

    return ax


