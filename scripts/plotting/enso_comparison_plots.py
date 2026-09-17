"""
Generate plots that compare ENSO characteristics across various versions of CESM development
"""

import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import warnings # use to warn user about missing files
from pathlib import Path

def enso_comparison_plots(adfobj):
    """
    This script/function is designed to generate ENSO-related plots across
    various CESM simulations

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    Does not return a value; produces plots and saves files.
    """
    
    # Notify user that script has started:
    msg = "\n  Generating ENSO plots to compare against all runs..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    plot_locations = adfobj.plot_location
    plot_type = adfobj.get_basic_info('plot_type')
    if not plot_type:
        plot_type = 'png'

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    #Grab saved files
    obs_ds   = xr.open_dataset('/glade/derecho/scratch/mdfowler/ENSOmetrics_Obs.nc')
    cesm1_ds = xr.open_dataset("/glade/derecho/scratch/mdfowler/ENSOmetrics_CESM1.nc")
    cesm2_ds = xr.open_dataset("/glade/derecho/scratch/mdfowler/ENSOmetrics_CESM2.nc")
    dev_ds   = xr.open_dataset("/glade/derecho/scratch/mdfowler/ENSOmetrics_CESM3dev.nc")

    # + + + + + + + + + + + + + + + + + + + +
    #  Make Nino variance comparison plots
    # + + + + + + + + + + + + + + + + + + + +
#                          J   F   M    A  M   J   J   A    S   O    N  D
    daysPerMonth = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    ninoSelString =  ['Nino34', 'Nino12', 'Nino3', 'Nino4']

    for iNino in range(len(ninoSelString)):
        #Set path for variance figures:
        plot_loc_ts  = Path(plot_locations[0]) / f'NinoVarianceComparison_{ninoSelString[iNino]}_ENSO_Mean.{plot_type}'
        print('Creating plot for ', ninoSelString[iNino])

        # Check redo_plot. If set to True: remove old plots, if they already exist:
        if (not redo_plot) and plot_loc_ts.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_loc_ts}' exists and clobber is false.")
            adfobj.add_website_data(plot_loc_ts, "NinoVarianceComparison", None, season=ninoSelString[iNino], multi_case=True, non_season=True, plot_type = "ENSO")

            #Continue to next iteration:
            return
        elif (redo_plot):
            if plot_loc_ts.is_file():
                plot_loc_ts.unlink()
        #End if

        if ( (ninoSelString[iNino]=='Nino34') ):
             pltVar = 'nino34_variance'
        elif ( (ninoSelString[iNino]=='Nino12') ):
             pltVar = 'nino12_variance'
        elif ( (ninoSelString[iNino]=='Nino3') ):
             pltVar = 'nino3_variance'
        elif ( (ninoSelString[iNino]=='Nino4') ):
             pltVar = 'nino4_variance'
        else:
            print('Invalid choice for ninoSelString supplied.\n Valid options: Nino34, Nino12, Nino3, Nino4')

        fig,axs=plt.subplots(1,1,figsize=(15,5))

        for iCase in range(len(dev_ds.case.values)):
            case_max = np.nanmax(dev_ds[pltVar].values[iCase,:])
            case_min = np.nanmin(dev_ds[pltVar].values[iCase,:])
            # Weighted mean
            weights = ( daysPerMonth / daysPerMonth.sum() )
            weighted_mean = (dev_ds[pltVar].values[iCase,:] * weights).sum() / weights.sum()
            
            axs.plot(iCase+np.ones(2), [case_min, case_max], 'k-')
            axs.plot(iCase+1, weighted_mean,'o', color='k')
            axs.plot(iCase+1, case_min,'^',color='k')
            axs.plot(iCase+1, case_max,'v',color='k')

        cesm1_xCenter = len(dev_ds.case.values)+1.5
        cesm2_xCenter = len(dev_ds.case.values)+3
        obs_xCenter   = len(dev_ds.case.values)+4

        offset = np.linspace(cesm1_xCenter-0.5, cesm1_xCenter+0.5, len(cesm1_ds.event.values))
        for iEvent in range(len(cesm1_ds.event.values)):
            case_max = np.nanmax(cesm1_ds[pltVar].values[iEvent,:])
            case_min = np.nanmin(cesm1_ds[pltVar].values[iEvent,:])
            # Weighted mean
            weights = ( daysPerMonth / daysPerMonth.sum() )
            weighted_mean = (cesm1_ds[pltVar].values[iEvent,:] * weights).sum() / weights.sum()
            
            axs.plot(offset[iEvent]*np.ones(2), [case_min, case_max],
                    '-', color='mediumpurple', alpha=0.2)
            axs.plot(offset[iEvent], weighted_mean,'o', color='mediumpurple',alpha=0.4)
            axs.plot(offset[iEvent], case_min,'^',color='mediumpurple',alpha=0.4)
            axs.plot(offset[iEvent],case_max,'v',color='mediumpurple',alpha=0.4)


        offset = np.linspace(cesm2_xCenter-0.5, cesm2_xCenter+0.5, len(cesm2_ds.event.values))
        for iEvent in range(len(cesm2_ds.event.values)):
            case_max = np.nanmax(cesm2_ds[pltVar].values[iEvent,:])
            case_min = np.nanmin(cesm2_ds[pltVar].values[iEvent,:])
            # Weighted mean
            weights = ( daysPerMonth / daysPerMonth.sum() )
            weighted_mean = (cesm2_ds[pltVar].values[iEvent,:] * weights).sum() / weights.sum()
            
            axs.plot(offset[iEvent]*np.ones(2), [case_min, case_max],
                    '-', color='orange', alpha=0.2)
            axs.plot(offset[iEvent], weighted_mean,'o', color='orange',alpha=0.4)
            axs.plot(offset[iEvent], case_min,'^',color='orange',alpha=0.4)
            axs.plot(offset[iEvent],case_max,'v',color='orange',alpha=0.4)


        # Get obs 
        obs_max = np.nanmax(obs_ds[pltVar].values)
        obs_min = np.nanmin(obs_ds[pltVar].values)
        # Weighted mean
        weights = ( daysPerMonth / daysPerMonth.sum() )
        weighted_mean = (obs_ds[pltVar].values * weights).sum() / weights.sum()

        axs.plot(obs_xCenter*np.ones(2), [obs_min, obs_max], '-', color='firebrick')
        axs.plot(obs_xCenter, weighted_mean,'o', color='firebrick')
        axs.plot(obs_xCenter, obs_min,'^',color='firebrick')
        axs.plot(obs_xCenter, obs_max,'v',color='firebrick')

        ## General plot settings 
        ticks = np.append(1+np.arange(len(dev_ds.case.values)), cesm1_xCenter)
        ticks = np.append(ticks, cesm2_xCenter)
        ticks = np.append(ticks, obs_xCenter)

        tickLabels = np.append(dev_ds.case.values, 'CESM1')
        tickLabels = np.append(tickLabels, 'CESM2')
        tickLabels = np.append(tickLabels, 'HadiSST')

        axs.set_xticks(ticks)
        axs.set_xticklabels(tickLabels)
        plt.setp( axs.xaxis.get_majorticklabels(), rotation=45 )

        axs.axhline(obs_min, color='firebrick',alpha=0.3)
        axs.axhline(obs_max, color='firebrick',alpha=0.3)

        axs.set_title('Monthly '+ninoSelString[iNino]+' variance')

        #Save figure to file:
        fig.savefig(plot_loc_ts, bbox_inches='tight', facecolor='white')

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc_ts, "NinoVarianceComparison", None, season=ninoSelString[iNino], multi_case=True, non_season=True, plot_type = "ENSO")

    # + + + + + + + + + + + + + + + + + + + +
    #  Make autocorrelation comparison plot
    # + + + + + + + + + + + + + + + + + + + +

    #Set path for variance figures:
    plot_loc_autocorr  = Path(plot_locations[0]) / f'NinoAutocorrelation_Nino34_ENSO_Mean.{plot_type}'
    print('Creating plot for Nino 3.4 autocorrelation transition')

    # Check redo_plot. If set to True: remove old plots, if they already exist:
    if (not redo_plot) and plot_loc_autocorr.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.debug_log(f"'{plot_loc_autocorr}' exists and clobber is false.")
        adfobj.add_website_data(plot_loc_autocorr, "NinoAutocorrelation", None, season='Nino34', multi_case=True, non_season=True, plot_type = "ENSO")

        #Continue to next iteration:
        return
    elif (redo_plot):
        if plot_loc_autocorr.is_file():
            plot_loc_autocorr.unlink()
    #End if

    fig,axs=plt.subplots(1,1,figsize=(15,5))

    axs.scatter(dev_ds.case.values, dev_ds.nino34_transMonth, color='k')
    axs.plot(np.full([len(cesm1_ds.event.values)], 'CESM1'), cesm1_ds.nino34_transMonth, 'o', color='mediumpurple', alpha=0.4)
    axs.plot(np.full([len(cesm2_ds.event.values)], 'CESM2'), cesm2_ds.nino34_transMonth, 'o', color='orange', alpha=0.4)
    axs.scatter('HadiSST', obs_ds.nino34_transMonth, color='k', marker='*', s=200)
    axs.axhline(obs_ds.nino34_transMonth, color='grey',alpha=0.5)
    axs.set_title('Month of transition in Autocorrelation for Nino3.4', fontsize=14)
    axs.set_ylabel('Lag of nMonths',fontsize=12)

    ticks = np.arange(len(dev_ds.case.values)+3)

    tickLabels = np.append(dev_ds.case.values, 'CESM1')
    tickLabels = np.append(tickLabels, 'CESM2')
    tickLabels = np.append(tickLabels, 'HadiSST')

    axs.set_xticks(ticks)
    axs.set_xticklabels(tickLabels, fontsize=11)
    axs.set_ylim([5,30])
    plt.setp( axs.xaxis.get_majorticklabels(), rotation=40 )
    
    #Save figure to file:
    fig.savefig(plot_loc_autocorr, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_autocorr, "NinoAutocorrelation", None, season='Nino34', multi_case=True, non_season=True, plot_type = "ENSO")


   # + + + + + + + + + + + + + + + + + + + +
    #  Make western extent comparison plot
    # + + + + + + + + + + + + + + + + + + + +

    #Set path for variance figures:
    plot_loc_westext  = Path(plot_locations[0]) / f'WesternExtent_Nino34_ENSO_Mean.{plot_type}'
    print('Creating plot for Nino3.4 SST anomaly western extent')

    # Check redo_plot. If set to True: remove old plots, if they already exist:
    if (not redo_plot) and plot_loc_westext.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.debug_log(f"'{plot_loc_westext}' exists and clobber is false.")
        adfobj.add_website_data(plot_loc_westext, "WesternExtent", None, season='Nino34', multi_case=True, non_season=True, plot_type = "ENSO")

        #Continue to next iteration:
        return
    elif (redo_plot):
        if plot_loc_westext.is_file():
            plot_loc_westext.unlink()
    #End if

    fig,axs=plt.subplots(1,2,figsize=(10,9))
    axs = axs.ravel()

    axs[0].scatter(dev_ds.nino34_zeroContour, dev_ds.case.values, color='dodgerblue', alpha=1)
    axs[0].set_title('Westernmost longitude of \n0 contour related to Nino 3.4 max')
    axs[0].invert_yaxis()
    axs[0].plot(cesm1_ds.nino34_zeroContour, np.full([len(cesm1_ds.event.values)], 'CESM1'), 'o', color='mediumpurple', alpha=0.4)
    axs[0].plot(cesm2_ds.nino34_zeroContour, np.full([len(cesm2_ds.event.values)], 'CESM2'), 'o', color='orange', alpha=0.4)
    axs[0].plot(obs_ds.nino34_zeroContour, 'HadiSST', '*', color='k',markersize=14)

    axs[1].scatter(dev_ds.nino34_0p5Contour, dev_ds.case.values, marker = 's', color='dodgerblue', alpha=1)
    axs[1].set_title('Westernmost longitude of \n0.5 contour related to Nino 3.4 max')
    axs[1].invert_yaxis()
    axs[1].plot(cesm1_ds.nino34_0p5Contour, np.full([len(cesm1_ds.event.values)], 'CESM1'), 's', color='mediumpurple', alpha=0.4)
    axs[1].plot(cesm2_ds.nino34_0p5Contour, np.full([len(cesm2_ds.event.values)], 'CESM2'), 's', color='orange', alpha=0.4)
    axs[1].plot(obs_ds.nino34_0p5Contour, 'HadiSST', '*', color='k',markersize=14)

    axs[0].set_xlim(120,170)
    axs[1].set_xlim(120,170)

    fig.subplots_adjust(wspace=0.5)

    #Save figure to file:
    fig.savefig(plot_loc_westext, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_westext, "WesternExtent", None, season='Nino34', multi_case=True, non_season=True, plot_type = "ENSO")

    # + + + + + + + + + + + + + + + + + + + +
    #  Make scatterplot comparison plots
    # + + + + + + + + + + + + + + + + + + + +
    case_shortName = adfobj.get_cam_info("case_nickname", required=True)[0]

    #Set path for variance figures:
    plot_loc_coldPoolScat  = Path(plot_locations[0]) / f'Scatter_ColdPoolExtent_ENSO_Mean.{plot_type}'
    print('Creating plot for cold pool extent vs. western extent')

    # Check redo_plot. If set to True: remove old plots, if they already exist:
    if (not redo_plot) and plot_loc_coldPoolScat.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.debug_log(f"'{plot_loc_coldPoolScat}' exists and clobber is false.")
        adfobj.add_website_data(plot_loc_coldPoolScat, "Scatter", None, season='ColdPoolExtent', multi_case=True, non_season=True, plot_type = "ENSO")

        #Continue to next iteration:
        return
    elif (redo_plot):
        if plot_loc_coldPoolScat.is_file():
            plot_loc_coldPoolScat.unlink()
    #End if

    fig,axs = plt.subplots(1,1, figsize=(10,8))

    xArr = np.append(dev_ds.sst_lon299, obs_ds.sst_lon299)
    yArr = np.append(dev_ds.nino34_0p5Contour, obs_ds.nino34_0p5Contour)
    cArr = np.append(dev_ds.sst_raw_nino34.mean(dim='season').values, (obs_ds.sst_raw_nino34.mean(dim='season').values + 273.15) )

    caseLabels = np.append(dev_ds.case.values, 'HadiSST')
    axs.tick_params(labelsize=12)  # Adjust 12 to your desired tick font size
    s = axs.scatter(xArr, yArr, c=cArr)
    cb = fig.colorbar(s,ax=axs,orientation='vertical')
    cb.set_label('Mean Nino3.4 SST', size=14)
    cb.ax.tick_params(labelsize=12)  # Adjust 12 to your desired tick font size 

    for iCase in range(len(xArr)): 
        if caseLabels[iCase]=='HadiSST':
            axs.text(xArr[iCase]*0.975, 
                    yArr[iCase], 
                    # corrs.case.values[iCase], color='k', alpha=0.5, fontsize=11)
                    caseLabels[iCase], color='g', fontsize=11)
        elif caseLabels[iCase]==case_shortName:
            axs.text(xArr[iCase], 
                     yArr[iCase], 
                     # corrs.case.values[iCase], color='k', alpha=0.5, fontsize=11)
                     caseLabels[iCase], color='r', alpha=1, fontsize=14)
        else: 
            axs.text(xArr[iCase], 
                    yArr[iCase], 
                    # corrs.case.values[iCase], color='k', alpha=0.5, fontsize=11)
                    caseLabels[iCase], color='k', alpha=0.7, fontsize=12)


    axs.set_title('Extent of cold tongue vs. Western extent of 0.5 contour for ENSO',fontsize=14)
    axs.set_xlabel('Westernmost point of 299 K contour in SSTs',fontsize=12)
    axs.set_ylabel('Westernmost point of 0.5 contour\n in SST anomaly correlations with nino3.4 index', fontsize=12)

    #Save figure to file:
    fig.savefig(plot_loc_coldPoolScat, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_coldPoolScat, "Scatter", None, season='ColdPoolExtent', multi_case=True, non_season=True, plot_type = "ENSO")


    # --- Next Scatter: Autocorrelation and mean SST ---- 
    plot_loc_durationSST  = Path(plot_locations[0]) / f'Scatter_NinoDuration&meanSST_ENSO_Mean.{plot_type}'
    print('Creating plot for Nino duration and mean SST')

    # Check redo_plot. If set to True: remove old plots, if they already exist:
    if (not redo_plot) and plot_loc_durationSST.is_file():
        #Add already-existing plot to website (if enabled):
        adfobj.debug_log(f"'{plot_loc_durationSST}' exists and clobber is false.")
        adfobj.add_website_data(plot_loc_durationSST, "Scatter", None, season='NinoDuration&meanSST', multi_case=True, non_season=True, plot_type = "ENSO")

        #Continue to next iteration:
        return
    elif (redo_plot):
        if plot_loc_durationSST.is_file():
            plot_loc_durationSST.unlink()
    #End if

    fig,axs = plt.subplots(1,1,figsize=(8,5))

    selSeason = 'DJF'

    sst_raw_season = dev_ds.sst_raw.sel(season=selSeason).copy(deep=True)
    transit_month  = dev_ds.nino34_transMonth
    # sst_cesm2_raw_season = sst_cesm2.sel(season=selSeason)

    # Define bounding box
    sst_sel      = sst_raw_season.sel(lat=slice(-5,5), lon=slice(120,280))
    sst_sel_mean = sst_sel.mean(dim='lon').mean(dim='lat')
    axs.set_title('Averages over '+selSeason+' & Full Pacific (-5S to 5N, 120 to 280)')

    axs.scatter(sst_sel_mean.values, transit_month.values)

    for iCase in range(len(sst_raw_season.case.values)): 
        if caseLabels[iCase]==case_shortName: 
            axs.plot(sst_sel_mean.values[iCase], 
                    transit_month[iCase], 'ro')
            axs.text(sst_sel_mean.values[iCase], 
                    transit_month[iCase], 
                    dev_ds.case.values[iCase], color='r', alpha=1, fontsize=14)
        else:
            if transit_month[iCase]<=25:
                axs.text(sst_sel_mean.values[iCase], 
                        transit_month[iCase], 
                        dev_ds.case.values[iCase], color='k', alpha=0.4, fontsize=12)

    axs.set_xlabel('Mean SST')
    axs.set_ylabel('Autocorrelation Transition (nMonths)')
    axs.set_ylim([7,25])

    axs.plot(273.15+obs_ds.sst_obs_raw_seasonal.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values, 
                    obs_ds.nino34_transMonth.values, 'g*', markersize=12)
    axs.text(273.15+obs_ds.sst_obs_raw_seasonal.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values, 
                obs_ds.nino34_transMonth.values*0.91, 
                'HadSST', color='g', alpha=0.8, fontsize=12)
    
    ## Add CESM ensembles
    axs.scatter(cesm2_ds.sst_raw.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values, 
                cesm2_ds.nino34_transMonth.values, marker='s', c='orange', alpha=0.8)
    axs.text(np.nanmean(cesm2_ds.sst_raw.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values)*1.001, 
            np.nanmean(cesm2_ds.nino34_transMonth.values), 'CESM2', color='orange', alpha=0.8, fontsize=12)


    axs.scatter(cesm1_ds.sst_raw.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values, 
                cesm1_ds.nino34_transMonth.values, marker='^', c='mediumorchid', alpha=0.8)
    axs.text(np.nanmean(cesm1_ds.sst_raw.sel(season=selSeason,lat=slice(-5,5), lon=slice(120,280)).mean(dim='lon').mean(dim='lat').values)*0.998, 
             np.nanmean(cesm1_ds.nino34_transMonth.values), 'CESM1', color='mediumorchid', alpha=0.8, fontsize=12)

    #Save figure to file:
    fig.savefig(plot_loc_durationSST, bbox_inches='tight', facecolor='white')

    #Add plot to website (if enabled):
    adfobj.add_website_data(plot_loc_durationSST, "Scatter", None, season='NinoDuration&meanSST', multi_case=True, non_season=True, plot_type = "ENSO")

    return