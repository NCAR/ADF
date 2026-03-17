import os
import glob
import numpy as np 
import xarray as xr
import pandas as pd
import datetime
from datetime import date, timedelta
import scipy.stats as stats
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs

# Import necessary ADF modules:
from pathlib import Path
from adf_base import AdfError

## Functions ## 


def ENSO_acrossRuns(adf, clobber=False, search=None):
    """
    This function computes ENSO statistics for new cases 

    Description of needed inputs from ADF:

    case_name    -> Name of CAM case provided by "cam_case_name"
    input_ts_loc -> Location of CAM time series files provided by "cam_ts_loc"
    output_loc   -> Location to write new dev file to, provided by "cam_climo_loc"
    var_list     -> List of CAM output variables provided by "diag_var_list"

    """
    ## Define some basics 
    lat_n = 10.0    
    lat_s = -10.0

    # Nino3.4
    lat_n34 = 5
    lat_s34 = -5
    lon_e34 = 190 
    lon_w34 = 240

    # Nino3
    lat_n3 = 5
    lat_s3 = -5
    lon_e3 = 210 
    lon_w3 = 270

    # Nino 4
    lat_n4 = 5
    lat_s4 = -5
    lon_e4 = 160 
    lon_w4 = 210

    # Nino 1+2
    lat_n12 = 0
    lat_s12 = -10
    lon_e12 = 270 
    lon_w12 = 280

    ## Read in ENSO characteristics from obs file 
    obs_ds = xr.open_dataset('/glade/derecho/scratch/mdfowler/ENSOmetrics_Obs.nc')


    ## Read in data for new case
    # - - - - - - - - - - - - - - - - - - - - - - - 
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    input_ts_locs = adf.get_cam_info("cam_ts_loc", required=True)
    #Extract simulation years:
    start_year = adf.climo_yrs["syears"]
    end_year   = adf.climo_yrs["eyears"]


    #Loop over CAM cases:
    for case_idx, case_name in enumerate(case_names):
        #Create "Path" objects:
        input_location  = Path(input_ts_locs[case_idx])

        case_shortName = adf.get_cam_info("case_nickname", required=True)[case_idx]
        
        #Check that time series input directory actually exists:
        if not input_location.is_dir():
            errmsg = f"Time series directory '{input_ts_locs}' not found.  Script is exiting."
            raise AdfError(errmsg)
        
        #Check model year bounds:
        syr, eyr = check_averaging_interval(start_year[case_idx], end_year[case_idx])

        ts       = adf.data.load_timeseries_da(case_name, 'TS')
        landfrac = adf.data.load_timeseries_da(case_name, 'LANDFRAC')

        ts = ts.assign_coords({"case":  case_shortName})

        #Extract data subset using provided year bounds:
        tslice = get_time_slice_by_year(ts.time, int(syr), int(eyr))
        ts       = ts.isel(time=tslice).sel(lat=slice(-10,10))
        landfrac = landfrac.isel(time=tslice).sel(lat=slice(-10,10))

    # - - - - - - - - - - - - - - - - - - - - - - - 
    # Start computing ENSO-relevant pieces 
    # - - - - - - - - - - - - - - - - - - - - - - - 

        # Use ocean points only
        ocnMask = landfrac.values
        ocnMask[ocnMask>0.45] = np.nan
        ocnMask[ocnMask<=0.45] = 1

        # Detrend data 
        TS = ts
        ts_detrend = signal.detrend(TS, axis=0, type='linear')
        # Get SST 
        sst = ts_detrend * ocnMask
        sst_raw_data = TS.values * ocnMask

        sst = xr.DataArray(sst, 
            coords={'time': ts.time.values,
                    'lat':  ts.lat.values, 
                    'lon':  ts.lon.values}, 
            dims=["time", "lat", "lon"])
        
        # Remove annual cycle from monthly data 
        sst_anom = rmMonAnnCyc(sst)

        ## Compute nino 3.4 index
        ## - - - - - - - - - - - - 
        ilats = np.where((sst_anom.lat.values>=lat_s34)  & (sst_anom.lat.values<=lat_n34))[0]
        ilons = np.where((sst_anom.lon.values>=lon_e34)  & (sst_anom.lon.values<=lon_w34))[0]

        regionTS = sst_anom.isel(lat=ilats, lon=ilons)
        coswgt   = np.cos(np.deg2rad(regionTS.lat))
        nino34   = regionTS.weighted(coswgt).mean(('lon','lat'))
        del ilats,ilons

        ## Compute nino 3 index
        ## - - - - - - - - - - - - 
        ilats = np.where((sst_anom.lat.values>=lat_s3)  & (sst_anom.lat.values<=lat_n3))[0]
        ilons = np.where((sst_anom.lon.values>=lon_e3)  & (sst_anom.lon.values<=lon_w3))[0]

        regionTS = sst_anom.isel(lat=ilats, lon=ilons)
        coswgt   = np.cos(np.deg2rad(regionTS.lat))
        nino3    = regionTS.weighted(coswgt).mean(('lon','lat'))
        del ilats,ilons

        ## Compute nino 4 index
        ## - - - - - - - - - - - - 
        ilats = np.where((sst_anom.lat.values>=lat_s4)  & (sst_anom.lat.values<=lat_n4))[0]
        ilons = np.where((sst_anom.lon.values>=lon_e4)  & (sst_anom.lon.values<=lon_w4))[0]

        regionTS = sst_anom.isel(lat=ilats, lon=ilons)
        coswgt   = np.cos(np.deg2rad(regionTS.lat))
        nino4    = regionTS.weighted(coswgt).mean(('lon','lat'))
        del ilats,ilons

        ## Compute nino 1.2 index
        ## - - - - - - - - - - - - 
        ilats = np.where((sst_anom.lat.values>=lat_s12)  & (sst_anom.lat.values<=lat_n12))[0]
        ilons = np.where((sst_anom.lon.values>=lon_e12)  & (sst_anom.lon.values<=lon_w12))[0]

        regionTS = sst_anom.isel(lat=ilats, lon=ilons)
        coswgt   = np.cos(np.deg2rad(regionTS.lat))
        nino12   = regionTS.weighted(coswgt).mean(('lon','lat'))
        del ilats,ilons

        ## Western extent of SST correlation 
        ## - - - - - - - - - - - - - - - - - -

        cor_case       = getLagCorr(0, nino34, sst_anom)

        ## Figure out western-most longitude of zero contour in pacific 
        #     This works by creating a plot with a contour and identifying the western point, but I'm closing that plot
        fig,axs = plt.subplots(1,1,figsize=(20,7),subplot_kw={"projection":ccrs.PlateCarree(central_longitude=210)})

        lon0, lat0     = getWesternPoint(cor_case, 0)
        lon0p5, lat0p5 = getWesternPoint(cor_case, 0.5)

        plt.close()

        ## Variance and Auto-correlation
        ## - - - - - - - - - - - - - - - - 

        nino34_var    = np.full([12], np.nan)
        nino34_ac     = np.full([48], np.nan)
        transit_month = np.nan

        nino3_var      = np.full([12], np.nan)
        nino3_ac       = np.full([48], np.nan)
        transit_month3 = np.nan

        nino4_var      = np.full([12], np.nan)
        nino4_ac       = np.full([48], np.nan)
        transit_month4 =  np.nan

        nino12_var      = np.full([12], np.nan)
        nino12_ac       = np.full([48], np.nan)
        transit_month12 = np.nan


        ## Reproduce the variance plot from Rich
        #   NOTE: doesn't match exactly, likely because I use cos(lat) for weights instead of a gaussian like Rich 
        nino34_var = nino34.groupby('time.month').var()
        nino3_var  = nino3.groupby('time.month').var()
        nino4_var  = nino4.groupby('time.month').var()
        nino12_var = nino12.groupby('time.month').var()

    
        ## Reproduce the autocorrelation plot from Rich 
        nino34_pd = pd.Series(nino34.values)
        nino3_pd  = pd.Series(nino3.values)
        nino4_pd  = pd.Series(nino4.values)
        nino12_pd = pd.Series(nino12.values)

        for iLag in range(48):
            nino34_ac[iLag] = nino34_pd.autocorr(lag=iLag)
            if ((nino34_ac[iLag-1]>0) & (nino34_ac[iLag]<0) & (np.isfinite(transit_month)==False)):
                transit_month = iLag-1

            nino3_ac[iLag] = nino3_pd.autocorr(lag=iLag)
            if ((nino3_ac[iLag-1]>0) & (nino3_ac[iLag]<0) & (np.isfinite(transit_month3)==False)):
                transit_month3 = iLag-1

            nino4_ac[iLag] = nino4_pd.autocorr(lag=iLag)
            if ((nino4_ac[iLag-1]>0) & (nino4_ac[iLag]<0) & (np.isfinite(transit_month4)==False)):
                transit_month4 = iLag-1

            nino12_ac[iLag] = nino12_pd.autocorr(lag=iLag)
            if ((nino12_ac[iLag-1]>0) & (nino12_ac[iLag]<0) & (np.isfinite(transit_month12)==False)):
                transit_month12 = iLag-1


        ## SST biases 
        ## - - - - - - - - - - - - - - - - 
        sst_bias = np.full([ 4, len(ts.lat.values), len(ts.lon.values)], np.nan)
        sst_raw  = np.full([ 4, len(ts.lat.values), len(ts.lon.values)], np.nan)
        sst_anom = np.full([ 4, len(ts.lat.values), len(ts.lon.values)], np.nan)

        sst_raw_data = xr.DataArray(sst_raw_data, 
            coords={'time': ts.time.values,
                    'lat':  ts.lat.values, 
                    'lon':  ts.lon.values}, 
            dims=["time", "lat", "lon"])

        # Get seasonal means 
        month_length = sst.time.dt.days_in_month
        weights = ( month_length.groupby("time.season") / month_length.groupby("time.season").sum() )
        # Calculate the weighted average
        sst_case_weighted = (sst * weights).groupby("time.season").sum(dim="time")
        sst_raw_case_weighted = (sst_raw_data * weights).groupby("time.season").sum(dim="time")
        del month_length,weights

        # Compute bias vs. observations (seasonally) 
        sst_bias[:,:,:] = (sst_case_weighted - obs_ds.sst_obs_weighted) * ocnMask[0,:,:]
        sst_raw[:,:,:]  = (sst_raw_case_weighted) * ocnMask[0,:,:]
        sst_anom[:,:,:] = sst_case_weighted * ocnMask[0,:,:]

        sst_bias = xr.DataArray(sst_bias, 
            coords={
                    'season': sst_case_weighted.season.values,
                    'lat': sst_case_weighted.lat.values, 
                    'lon':sst_case_weighted.lon.values}, 
            dims=["season", "lat", "lon"])

        sst_raw = xr.DataArray(sst_raw, 
            coords={
                    'season': sst_case_weighted.season.values,
                    'lat': sst_case_weighted.lat.values, 
                    'lon':sst_case_weighted.lon.values}, 
            dims=["season", "lat", "lon"])

        sst_anom = xr.DataArray(sst_anom, 
            coords={
                    'season': sst_case_weighted.season.values,
                    'lat': sst_case_weighted.lat.values, 
                    'lon':sst_case_weighted.lon.values}, 
            dims=["season", "lat", "lon"])
        
        ## Get the longitude of the "cold tongue", approximated by the contour of the 299 K SST line
        #     This works by creating a plot with a contour and identifying the western point, but I'm closing that plot
        fig,axs = plt.subplots(1,1,figsize=(20,7),subplot_kw={"projection":ccrs.PlateCarree(central_longitude=210)})

        lon299, lat299 = getWesternPoint(sst_raw.mean(dim='season'),299)

        plt.close()

        # Mean SST in nino3.4 region
        ilats = np.where((sst_raw.lat.values>=lat_s34)  & (sst_raw.lat.values<=lat_n34))[0]
        ilons = np.where((sst_raw.lon.values>=lon_e34)  & (sst_raw.lon.values<=lon_w34))[0]

        # # Compute weights and get Nino3.4 
        regionTS = sst_raw.isel(lat=ilats, lon=ilons)
        coswgt   = np.cos(np.deg2rad(regionTS.lat))
        sst_raw_nino34 = regionTS.weighted(coswgt).mean(('lon','lat'))


        # - - - - - - - - - - - - - - - - - - - - - - - 
        # Add ENSO stats for new case to existing DS
        # - - - - - - - - - - - - - - - - - - - - - - - 
        ## Combine all of above into a single DS, as in dev_ds 
    
        thisDev_DS = xr.Dataset(
            data_vars = dict( 
                startYear = (['case'], [ts['time.year'].values[0]]),
                endYear   = (['case'], [ts['time.year'].values[-1]]),
                nino34_zeroContour = (['case'], [lon0]),
                nino34_0p5Contour  = (['case'], [lon0p5]),
                nino34_variance    = (['case', 'nMonths_12'], [nino34_var]),
                nino34_autocorr    = (['case', 'nMonths_48'], [nino34_ac]),
                nino34_transMonth  = (['case'], [transit_month]),
    
                nino3_variance    = (['case', 'nMonths_12'], [nino3_var]),
                nino3_autocorr    = (['case', 'nMonths_48'], [nino3_ac]),
                nino3_transMonth  = (['case'], [transit_month3]),
    
                nino4_variance    = (['case', 'nMonths_12'], [nino4_var]),
                nino4_autocorr    = (['case', 'nMonths_48'], [nino4_ac]),
                nino4_transMonth  = (['case'], [transit_month4]),
    
                nino12_variance    = (['case', 'nMonths_12'], [nino12_var]),
                nino12_autocorr    = (['case', 'nMonths_48'], [nino12_ac]),
                nino12_transMonth  = (['case'], [transit_month12]),
    
                sst_raw         = (['case', 'season', 'lat', 'lon'], [sst_raw.values]),
                sst_raw_nino34  = (['case', 'season'], [sst_raw_nino34.values]),   
                sst_bias = (['case', 'season', 'lat', 'lon'], [sst_bias.values]),
                sst_anom = (['case', 'season', 'lat', 'lon'], [sst_anom.values]),
    
                sst_lon299 = (['case'], [lon299]),
                
            ), 
            
            coords = dict(
                case=([ts.case.values]), 
                nMonths_48=np.arange(48),
                nMonths_12=np.arange(12),
                season=sst_bias.season.values,
                lat=sst_bias.lat.values,
                lon=sst_bias.lon.values,
            )
        )
    
        dev_ds   = xr.open_dataset("/glade/derecho/scratch/mdfowler/ENSOmetrics_CESM3dev.nc")
        newDS    = xr.concat([dev_ds, thisDev_DS], dim="case")
    
        ## Save that new DS out to dev_ds and reload 
        dev_ds.close()
    
        newDS.to_netcdf("/glade/derecho/scratch/mdfowler/ENSOmetrics_CESM3dev.nc", mode='w')



def get_time_slice_by_year(time, startyear, endyear):
    if not hasattr(time, 'dt'):
        print("Warning: get_time_slice_by_year requires the `time` parameter to be an xarray time coordinate with a dt accessor. Returning generic slice (which will probably fail).")
        return slice(startyear, endyear)
    start_time_index = np.argwhere((time.dt.year >= startyear).values).flatten().min()
    end_time_index = np.argwhere((time.dt.year <= endyear).values).flatten().max()
    return slice(start_time_index, end_time_index+1)


## Calculate nino anomalies 
def rmMonAnnCyc(DS): 
    
    climatology = DS.groupby("time.month").mean("time")
    anomalies   = DS.groupby("time.month") - climatology    

    return anomalies

def getLagCorr(lag, nino34, corDS):

    if lag>0: 
        A  = nino34[:-lag]
        # B  = sst_anom.isel(case=iCase).shift(time=lag).isel(time=slice(lag,len(sst_anom.time.values)))
        # B['time'] = sst_anom.time.values[:-lag]  # To get xr.corr to work, need to have the "same" time for computing corrs
        B  = corDS.shift(time=-lag).isel(time=slice(0,len(corDS.time.values)-lag))
        B['time'] = corDS.time.values[:-lag]  # To get xr.corr to work, need to have the "same" time for computing corrs
    elif lag<0:
        A  = nino34[-lag:]
        A['time'] = nino34.time.values[:lag]
        B  = corDS.isel(time=slice(0,len(corDS.time.values)+lag))
    elif lag==0:
        A = nino34
        B = corDS
            
    cor = xr.corr(A, B, dim="time")

    return cor

def getWesternPoint(DS, contourLev):
    corrs_sel = DS.sel(lon=slice(120,240), lat=slice(-10,10))
    c2 = plt.contour(corrs_sel.lon.values,corrs_sel.lat.values ,  corrs_sel, [contourLev], transform=ccrs.PlateCarree())
    
    # Add contour marker
    maybeLon = []
    maybeLat = []
    lenSeg = 0
    for iSegs in range(len(c2.allsegs[0])): 
        dat0 = c2.allsegs[0][iSegs]
        western_most_lon = np.nanmin(dat0[:,0])
        iMatchLat = np.where(dat0[:,0]==western_most_lon)[0]
        maybeLon = np.append(maybeLon, western_most_lon)
        maybeLat = np.append(maybeLat, dat0[int(iMatchLat[0]),1])
        if len(dat0)>lenSeg:
            lenSeg= len(dat0)
            iselSeg = iSegs
    
    # axs.plot(maybeLon[iselSeg], maybeLat[iselSeg], 'o', color='limegreen', markersize=5, transform=ccrs.PlateCarree() )
    
    return maybeLon[iselSeg],maybeLat[iselSeg]

def check_averaging_interval(syear_in, eyear_in):
    #For now, make sure year inputs are integers or None,
    #in order to allow for the zero additions done below:
    if syear_in:
        check_syr = int(syear_in)
    else:
        check_syr = None
    #end if

    if eyear_in:
        check_eyr = int(eyear_in)
    else:
        check_eyr = None

    #Need to add zeros if year values aren't long enough:
    #------------------
    #start year:
    if check_syr:
        assert check_syr >= 0, 'Sorry, values must be positive whole numbers.'
        try:
            syr = f"{check_syr:04d}"
        except:
            errmsg = " 'start_year' values must be positive whole numbers"
            errmsg += f"not '{syear_in}'."
            raise AdfError(errmsg)
    else:
        syr = None
    #End if

    #end year:
    if check_eyr:
        assert check_eyr >= 0, 'Sorry, end_year values must be positive whole numbers.'
        try:
            eyr = f"{check_eyr:04d}"
        except:
            errmsg = " 'end_year' values must be positive whole numbers"
            errmsg += f"not '{eyear_in}'."
            raise AdfError(errmsg)
    else:
        eyr = None
    #End if
    return syr, eyr

