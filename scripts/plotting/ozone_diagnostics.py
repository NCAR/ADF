"""
Generates an ozone-specific diagnostics based on the previous ncl-based 
diagnostics package

Created 19 January, 2024
Shawn Honomichl
Associate Scientist III
NCAR/ACOM
shawnh@ucar.edu

Functions
---------
ozone_diagnostics(adfobj)
    use ADF object to make maps

define_regions(InputRegion)

define_stations(InputStation)

open_process_sonde_data_simone(obsdir)

Subplot_O3(ax,X0,Y0,X1,Model0,Model1,SondeMean,SondeLow,SondeHigh,PLevel,plottype,case_base,case_test,ylim,Quadrant,Compare_Obs)

Plot_Seasonal_Cycle_Profile(X0,X01,Y0,X1,X11,Y1,X2,X22,Y2,X3,X33,Y3,SondeMean0,SondeMean1,SondeMean2,SondeMean3,SondeLow0,SondeLow1,SondeLow2,SondeLow3,SondeHigh0,SondeHigh1,SondeHigh2,SondeHigh3,Model0I,Model01I,Model1I,Model11I,Model2I,Model21I,Model3I,Model31I,ylim0,ylim1,ylim2,ylim3,PLevel0,PLevel1,PLevel2,PLevel3,case_base,case_test,Station,oFile,plottype,Compare_Obs)

get_model_data(ClimoFile)

process_model_seasonal_cycle(MinLon,MaxLon,MinLat,MaxLat,Model_Dat,pnew,intyp,kxtrp,Station_Lons,Station_Lats)

process_model_profiles(Model_Dat,O3_0,PS_0,pnew,intyp,kxtrp,ILAT,ILON,lat_0,lon_0)
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    Does not return a value; produces plots and saves files.

    Notes
    -----
    This function imports `pandas` and `plotting_functions`

    It uses the AdfDiag object's methods to get necessary information.
    Specificially:
    adfobj.diag_var_list
        List of variables
    adfobj.get_basic_info
        Regrid data path, checks `compare_obs`, checks `redo_plot`, checks `plot_press_levels`
    adfobj.plot_location
        output plot path
    adfobj.get_cam_info
        Get `cam_case_name` and `case_nickname`
    adfobj.climo_yrs
        start and end climo years of the case(s), `syears` & `eyears`
        start and end climo years of the reference, `syear_baseline` & `eyear_baseline`
    adfobj.var_obs_dict
        reference data (conditional)
    adfobj.get_baseline_info
        get reference case, `cam_case_name`
    adfobj.variable_defaults 
        dict of variable-specific plot preferences
    adfobj.read_config_var
        dict of basic info, `diag_basic_info`
        Then use to check `plot_type`
    adfobj.compare_obs
        Used to set data path
    adfobj.debug_log
        Issues debug message
    adfobj.add_website_data
        Communicates information to the website generator
    
"""

#Import modules:
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import Ngl
from scipy.interpolate import RegularGridInterpolator
from scipy import stats
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os.path

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

#-----------------------------------------------------------------------------------------
#Lookup pertinent region info
#-----------------------------------------------------------------------------------------

#Region_Number : ['region short name', 'region long name',region_min_lat, region_max_lat,region_min_lon,region_max_lon]
def define_regions(InputRegion):
   
   Region_Lookup={0 : ['not_assigned','not assigned',0,0,0,0],
                  1 : ['nh_polar_west','NH Polar West',70.0,90.0,-135.0+360,-45.0+360],
                  2 : ['nh_polar_east','NH Polar East',58.0,90.0,-45.0,45.0],
                  3 : ['canada','Canada',48.0,62.0,-135.0+360,-45.0+360],
                  4 : ['west_europe','Western Europe',43.0,57.5,0.0,25.0],
                  5 : ['eastern_us','Eastern US',34.0,40.0,-95.0+360,-75.0+360],
                  6 : ['japan','Japan',30.0,40.0,120.0,150.0],
                  7 : ['nh_tropic','NH Sub Tropics',15.0,30.0,90.0,225.0],
                  8 : ['tropics1','W-Pacific/E-Indian Ocean',-20.0,0.0,90.0,225.0],
                  9 : ['tropics2','Equatorial Americas',-15.0,15.0,225.0,315.0],
                  10 : ['tropics3','Atlantic/Africa',-15.0,15.0,-45.0,45.0],
                  11 : ['sh_midlat','SH Midlatitudes',-57.5,-40.0,135.0,180.0],
                  12 : ['sh_polar','SH Polar',-90.0,-58.0,0,360],
                  13 : ['Boulder','Boulder',37.0,42.0,-110.0+360.0,-100.0+360.0],
                 }
   
   #--------------------------------------------------------------------------------------
   #try to locate the region
   #--------------------------------------------------------------------------------------
   try:
      RegionInfo=Region_Lookup[InputRegion]
   except:
      print("Input region not defined") #if region not in the lookup then auto-assign to 0
      RegionInfo=Region_Lookup[0]
      
   #--------------------------------------------------------------------------------------
   #return the region info and the total number of regions in the table.
   #--------------------------------------------------------------------------------------
   return RegionInfo,len(Region_Lookup)-1

#-----------------------------------------------------------------------------------------
#Lookup pertinent ozonesonde station info
#-----------------------------------------------------------------------------------------
def define_stations(InputStation):

   #Station : ['country',lat N,lon E]
   Station_Lookup={0 : ['Alert','Canada',82.5,-62.33],
                   1 : ['Eureka','Canada',80.05,-86.42],
                   2 : ['Ny_Alesund','Norway',78.92,11.93],
                   3 : ['Resolute','Canada',74.7,-95.0],
                   4 : ['Scoresbysund','Greenland',70.48,-21.97],
                   5 : ['Lerwick','UK',60.13,-1.1+360.0],
                   6 : ['Churchill','Canada',58.77,-94.17],
                   7 : ['Edmonton','Canada',53.53,-113.5],
                   8 : ['Goose_Bay','Canada',53.29,-60.39],
                   9 : ['Legionowo','Poland',52.4,20.97],
                   10 : ['Lindenberg','Germany',52.21,14.12],
                   11 : ['DeBilt','Netherlands',52.10,5.18],
                   12 : ['Uccle','Belgium',50.8,4.35],
                   13 : ['Praha','Czech Republic',50.01,14.45],
                   14 : ['Hohenpeissenberg','Germany',47.80,11.02],
                   15 : ['Payerne','Switzerland',46.82,6.95],
                   16 : ['Madrid','Spain',40.45,-3.72],
                   17 : ['Boulder','Colorado',39.99,-105.26],
                   18 : ['Wallops_Island','Maryland',37.94,-75.46],
                   19 : ['Trinidad Head','California',41.05,-124.15],
                   20 : ['Huntsville','Alabama',34.73,-86.64],
                   21 : ['Sapporo','Japan',43.06,141.35],
                   22 : ['Tateno','Japan',36.05,140.13],
                   23 : ['Kagoshima','Japan',31.6,130.6],
                   24 : ['Naha','Japan',26.21,127.68],
                   25 : ['Hongkong','China',22.32,114.17],
                   26 : ['Paramaribo','Suriname',5.75,55.2],
                   27 : ['Hilo','USA',19.72,155.07],
                   28 : ['San Cristobal','Galapagos',-0.87,-89.44],
                   29 : ['Nairobi','Kenya',-1.29,36.82],
                   30 : ['Natal','Brazil',-5.83,-35.2],
                   31 : ['Ascension','Atlantic',-7.95,-14.36],
                   32 : ['Watukosek','Indonesia',-7.57,112.66],
                   33 : ['Samoa','Pacific',-14.25,-170.56],
                   34 : ['Fiji','Fiji',-17.71,178.07],
                   35 : ['Reunion','Africa',-21.1,55.4],
                   36 : ['Broadmeadows','Australia',-37.68,144.92],
                   37 : ['Lauder','New Zealand',-45.04,169.68],
                   38 : ['Macquarie','Australia',-54.50,158.95],
                   39 : ['Marambio','Antarctica',-64.0,-56.0],
                   40 : ['Neumayer','Antarctica',-70.62,8.37],
                   41 : ['Syowa','Antarctica',-69.01,39.59]}

   
   #--------------------------------------------------------------------------------------
   #try to locate the station
   #--------------------------------------------------------------------------------------
   try :
      StationInfo=Station_Lookup[InputStation]
   except:
      print("Input station not defined")
      
   return StationInfo,len(Station_Lookup)-1
   
#-----------------------------------------------------------------------------------------
#Process ozone sonde climatology
#-----------------------------------------------------------------------------------------
def open_process_sonde_data_simone(obsdir):

   #Grab and package all of the data that will be used for plotting
   NRegions=define_regions(0)[1]
   
   O3_MeanC=[]
   O3_WidthC=[]
   O3_StdDevC=[]
   Region=[]
   PressureC=[]
   
   for i in range(1,NRegions+1):
      Region_Info=define_regions(i)[0]
      ifileID=Region_Info[0]
      
      Region.append(ifileID)
      
      Check_File=obsdir+'ozonesondes_'+ifileID+'1995_2011.nc'
      
      if (os.path.isfile(Check_File)):
         print("Using Ozone Climatology File: ",Check_File)
         O3_Mean = xr.open_dataset(Check_File).o3_mean
         O3_Width = xr.open_dataset(Check_File).o3_width
         O3_StdDev = xr.open_dataset(Check_File).o3_std
         if (ifileID == 'Boulder'):
             Pressure = xr.open_dataset(Check_File).press #hPa
         else:
             Pressure = xr.open_dataset(Check_File).levels #hPa
             
         if (i == 1):
            O3_MeanC=O3_Mean
            O3_WidthC=O3_Width
            O3_StdDevC=O3_StdDev
            PressureC=Pressure
         else:
            O3_MeanC = np.dstack( (O3_MeanC,O3_Mean))
            O3_WidthC = np.dstack( (O3_WidthC,O3_Width))
            O3_StdDevC = np.dstack( (O3_StdDevC,O3_Width))
            PressureC=np.dstack( (PressureC,Pressure))
      else:
         print("Error opening ozonesonde file ",Check_File)
         return -1
   
   return O3_MeanC,O3_WidthC,O3_StdDevC,np.squeeze(PressureC),Region

#-----------------------------------------------------------------------------------------
#Subroutine - plots the ozone data subplots
#-----------------------------------------------------------------------------------------
def Subplot_O3(ax,X0,Y0,X1,Model0,Model1,SondeMean,SondeLow,SondeHigh,PLevel,plottype,case_base,case_test,ylim,Quadrant,Compare_Obs):

   #plottype == 0 ; Seasonal Cycle
   #plottype == 1 ; Profiles
   
   #Blue is the base case  - if comparing obs will not be plotted
   #Red is the test case - should always be plotted

   Model_linewidth=0.75
   
   if plottype == 0:
       ax.plot(Y0,X1,'tab:red',linestyle='dashed',linewidth=Model_linewidth)
       ax.plot(Y0,Model1,'tab:red',linestyle='solid',linewidth=Model_linewidth)
       if Compare_Obs <= 0:
           ax.plot(Y0,X0,'tab:blue',linestyle='dashed',linewidth=Model_linewidth)
           ax.plot(Y0,Model0,'tab:blue',linestyle='solid',linewidth=Model_linewidth)
           
       ax.set_title(PLevel,loc='left')
       
       props = dict(boxstyle='round',facecolor='wheat',alpha=0.5)
       if Compare_Obs <= 0:
           ax.text(0.05,0.95,"     Mean:                                            \nAbs. Diff:\n            r:",transform=ax.transAxes,fontsize=5,verticalalignment='top',bbox=props,color='black')
       else:
           ax.text(0.05,0.95,"     Mean:                                \nAbs. Diff:\n            r:",transform=ax.transAxes,fontsize=5,verticalalignment='top',bbox=props,color='black')
       
       meanS0=np.mean(SondeMean)
       meanS0S='%.1f' % meanS0
       
       meanX01=np.mean( [X1,Model1])
       meanX01S='%.1f' % meanX01
       AbsDif1=np.absolute(meanX01-meanS0)
       AbsDif1S='%.1f' % AbsDif1
       PCorr01=stats.pearsonr(X1,SondeMean)
       PCorr01S='%.2f' % PCorr01[0]
       
       if Compare_Obs <= 0:
           ax.text(0.70,0.92,meanX01S,fontsize=5,color='tab:red',transform=ax.transAxes)
           ax.text(0.70,0.86,AbsDif1S,fontsize=5,color='tab:red',transform=ax.transAxes)
           ax.text(0.70,0.80,PCorr01S,fontsize=5,color='tab:red',transform=ax.transAxes)
       else:
           ax.text(0.50,0.92,meanX01S,fontsize=5,color='tab:red',transform=ax.transAxes)
           ax.text(0.50,0.86,AbsDif1S,fontsize=5,color='tab:red',transform=ax.transAxes)
           ax.text(0.50,0.80,PCorr01S,fontsize=5,color='tab:red',transform=ax.transAxes)
       
       ax.text(0.30,0.92,meanS0S,fontsize=5,color='black',transform=ax.transAxes)
       
       if Compare_Obs <= 0:
           
           meanX0=np.mean([X0,Model0])
           meanX0S='%.1f' % meanX0
           AbsDif0=np.absolute(meanX0-meanS0)
           AbsDif0S='%.1f' % AbsDif0
           PCorr0=stats.pearsonr(X0,SondeMean)
           PCorr0S='%.2f' % PCorr0[0]
       
           ax.text(0.50,0.92,meanX0S,fontsize=5,color='tab:blue',transform=ax.transAxes)
           ax.text(0.50,0.86,AbsDif0S,fontsize=5,color='tab:blue',transform=ax.transAxes)
           ax.text(0.50,0.80,PCorr0S,fontsize=5,color='tab:blue',transform=ax.transAxes)
       
       #legend (bottom left) side of bottom left plot
       if Quadrant == 2:
          props1 = dict(boxstyle='square',facecolor='white',alpha=0.5)
          #ax.text(0.03,0.03,"                                                       \n  \n  \n  \n  ",transform=ax.transAxes,fontsize=5.25,verticalalignment='bottom',bbox=props1,color='black')
          
          if Compare_Obs >0:
              ax.text(0.12,0.15,"Region Avg.",fontsize=5.0,color='tab:red',transform=ax.transAxes)
              ax.text(0.12,0.09,case_test,fontsize=5.0,color='tab:red',transform=ax.transAxes)
              ax.plot([0.03,0.11],[0.16,0.16],'tab:red',linestyle='dashed',transform=ax.transAxes,linewidth=0.75)
              ax.plot([0.03,0.11],[0.10,0.10],'tab:red',linestyle='solid',transform=ax.transAxes,linewidth=0.75)
          else:
              ax.text(0.12,0.27,"Region Avg.",fontsize=5.0,color='tab:blue',transform=ax.transAxes)
              ax.text(0.12,0.21,case_base,fontsize=5.0,color='tab:blue',transform=ax.transAxes)
              ax.plot([0.03,0.11],[0.28,0.28],'tab:blue',linestyle='dashed',transform=ax.transAxes,linewidth=0.75)
              ax.plot([0.03,0.11],[0.22,0.22],'tab:blue',linestyle='solid',transform=ax.transAxes,linewidth=0.75)
              ax.text(0.12,0.15,"Region Avg.",fontsize=5.0,color='tab:red',transform=ax.transAxes)
              ax.text(0.12,0.09,case_test,fontsize=5.0,color='tab:red',transform=ax.transAxes)
              ax.plot([0.03,0.11],[0.16,0.16],'tab:red',linestyle='dashed',transform=ax.transAxes,linewidth=0.75)
              ax.plot([0.03,0.11],[0.10,0.10],'tab:red',linestyle='solid',transform=ax.transAxes,linewidth=0.75)
              
          ax.plot([0.06],[0.04],'ko',markersize=1.25,transform=ax.transAxes)
          ax.text(0.12,0.03,"Ozonesondes",fontsize=5.0,color='black',transform=ax.transAxes)
    
       #plot the ozonesonde data for a type 0 plot
       ax.plot(Y0,SondeMean,'ko',markersize=1.25)
       for i in range(1,13):
          ax.plot([i,i],[SondeLow[i-1],SondeHigh[i-1]],color='black',linestyle='solid',linewidth=1)
         
       #set the appropriate x/y axis information
       ax.set(xlabel="Month", ylabel="Ozone (ppb)", xlim=(1,12),ylim=ylim,xscale='linear',yscale='linear')
       ax.xaxis.set_major_locator(MultipleLocator(2))
       
   else:

       ax.plot(X1,Y0,'tab:red',linestyle='dashed',linewidth=Model_linewidth)
       ax.plot(Model1,Y0,'tab:red',linestyle='solid',linewidth=Model_linewidth)
       ax.set_title(PLevel,loc='left')
       
       if Compare_Obs <= 0:
           ax.plot(X0,Y0,'tab:blue',linestyle='dashed',linewidth=Model_linewidth)
           ax.plot(Model0,Y0,'tab:blue',linestyle='solid',linewidth=Model_linewidth)
       
       if Quadrant == 2: 
          offset=0.67
          props1 = dict(boxstyle='square',facecolor='white',alpha=0.5)
          #ax.text(0.03,0.03+offset,"                                                       \n  \n  \n  \n  ",transform=ax.transAxes,fontsize=5.25,verticalalignment='bottom',bbox=props1,color='black')

          if Compare_Obs > 0:
              ax.text(0.12,0.16+offset,"Ozonesondes",fontsize=5.0,color='black',transform=ax.transAxes)
              ax.plot([0.06],[0.17+offset],'ko',markersize=1.25,transform=ax.transAxes)
          else:
              ax.text(0.12,0.03+offset,"Ozonesondes",fontsize=5.0,color='black',transform=ax.transAxes)
              ax.plot([0.06],[0.04+offset],'ko',markersize=1.25,transform=ax.transAxes)
          
          ax.text(0.12,0.27+offset,"Region Avg.",fontsize=5.0,color='tab:red',transform=ax.transAxes)
          ax.text(0.12,0.21+offset,case_test,fontsize=5.0,color='tab:red',transform=ax.transAxes)
          ax.plot([0.03,0.11],[0.28+offset,0.28+offset],'tab:red',linestyle='dashed',transform=ax.transAxes,linewidth=0.75)
          ax.plot([0.03,0.11],[0.22+offset,0.22+offset],'tab:red',linestyle='solid',transform=ax.transAxes,linewidth=0.75)
          
          
          if Compare_Obs <= 0:
              ax.text(0.12,0.15+offset,"Region Avg.",fontsize=5.0,color='tab:blue',transform=ax.transAxes)
              ax.text(0.12,0.09+offset,case_base,fontsize=5.0,color='tab:blue',transform=ax.transAxes)
              ax.plot([0.03,0.11],[0.16+offset,0.16+offset],'tab:blue',linestyle='dashed',transform=ax.transAxes,linewidth=0.75)
              ax.plot([0.03,0.11],[0.10+offset,0.10+offset],'tab:blue',linestyle='solid',transform=ax.transAxes,linewidth=0.75)
          
       #plot the ozonesonde data for a type 1 plot
       Step=1
       SondeMean=SondeMean[1::Step]
       Y00=Y0[1::Step]
       SondeLow=SondeLow[1::Step]
       SondeHigh=SondeHigh[1::Step]
       ax.plot(SondeMean,Y00,'ko',markersize=1.25)
       for i in range(0,len(Y00)):
           ax.plot([SondeLow[i],SondeHigh[i]],[Y00[i],Y00[i]],color='black',linestyle='solid',linewidth=1)
           
       #set the appropriate x/y axis information
       ax.set(xlabel="Ozone (ppb)", ylabel="Pressure (hPa)", xlim=ylim,ylim=[1000.0,0.0],xscale='linear',yscale='linear')
       
   #Set tick parameters
   ax.xaxis.set_minor_locator(AutoMinorLocator())
   ax.yaxis.set_minor_locator(AutoMinorLocator())
   ax.tick_params(top=True, bottom=True, left=True, right=True)
       
   return True

#-----------------------------------------------------------------------------------------
#Subroutine - plot the seasonal cycle
#-----------------------------------------------------------------------------------------
def Plot_Seasonal_Cycle_Profile(X0,X01,Y0,X1,X11,Y1,X2,X22,Y2,X3,X33,Y3,SondeMean0,SondeMean1,SondeMean2,SondeMean3,SondeLow0,SondeLow1,SondeLow2,SondeLow3,SondeHigh0,SondeHigh1,SondeHigh2,SondeHigh3,Model0I,Model01I,Model1I,Model11I,Model2I,Model21I,Model3I,Model31I,ylim0,ylim1,ylim2,ylim3,PLevel0,PLevel1,PLevel2,PLevel3,case_base,case_test,Station,oFile,plottype,Compare_Obs):
      
   #--------------------------------------------------------------------------------------
   #set the default font size for the plots
   #--------------------------------------------------------------------------------------
   mpl.rcParams['font.size'] = 8
   plt.rcParams['figure.figsize'] = [6.4,4.8]
   
   #--------------------------------------------------------------------------------------
   #set the subplots
   #--------------------------------------------------------------------------------------
   fig, axs = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False)
   
   #--------------------------------------------------------------------------------------
   #Primary subplots
   #--------------------------------------------------------------------------------------
   PTL = Subplot_O3(axs[0,0],X0,Y0,X01,Model0I,Model01I,SondeMean0,SondeLow0,SondeHigh0,PLevel0,plottype,case_base,case_test,ylim0,0,Compare_Obs)#top left quadrant 0
   PTR = Subplot_O3(axs[0,1],X1,Y1,X11,Model1I,Model11I,SondeMean1,SondeLow1,SondeHigh1,PLevel1,plottype,case_base,case_test,ylim1,1,Compare_Obs)#top right quadrant 1
   PBL = Subplot_O3(axs[1,0],X2,Y2,X22,Model2I,Model21I,SondeMean2,SondeLow2,SondeHigh2,PLevel2,plottype,case_base,case_test,ylim2,2,Compare_Obs)#bottom left quadrant 2
   PBR = Subplot_O3(axs[1,1],X3,Y3,X33,Model3I,Model31I,SondeMean3,SondeLow3,SondeHigh3,PLevel3,plottype,case_base,case_test,ylim3,3,Compare_Obs)#bottom right quadrant 3
   
   #--------------------------------------------------------------------------------------
   #A couple of other plotting adjustments
   #--------------------------------------------------------------------------------------
   fig.suptitle(Station)
   plt.subplots_adjust(wspace=0.4,hspace=0.4,left=0.3) #Adjusts the spacing between the plots
   plt.rcParams['font.size'] = '7'
   
   Output_IMG = oFile #set the output filename
   
   #--------------------------------------------------------------------------------------
   #Try to save the plot as an image file
   #--------------------------------------------------------------------------------------
   try:
        plt.savefig(Output_IMG, bbox_inches = 'tight', dpi = 200, format = 'png') #there will be more to add here...
        plt.clf() 
        plt.close(fig)
   except:
        print("ERROR: Could not save file ",Output_FName,flush=True)
        
   return True

#-----------------------------------------------------------------------------------------
#Subroutine - retrieve model data from a netCDF file
#-----------------------------------------------------------------------------------------
def get_model_data(ClimoFile):

    class Model_Data:
        #Get the Reference Case data
        if (os.path.isfile(ClimoFile)):
            print("ADF Climatology File Located: ",ClimoFile)
            lon = xr.open_dataset(ClimoFile).lon
            lat = xr.open_dataset(ClimoFile).lat
            #lev = xr.open_dataset(ClimoFile).lev
            hyam = xr.open_dataset(ClimoFile).hyam
            hybm = xr.open_dataset(ClimoFile).hybm
            o3=xr.open_dataset(ClimoFile).O3
            ps=xr.open_dataset(ClimoFile).PS
            #time=xr.open_dataset(ClimoFile).time
      
            #these typically won't change with time so only select the time 0
            hyam=hyam[0,:]
            hybm=hybm[0,:]
           
    MDAT=Model_Data()
    return MDAT
    
#-----------------------------------------------------------------------------------------
#Subroutine - Process Model Seasonal Cycle 
#-----------------------------------------------------------------------------------------
def process_model_seasonal_cycle(MinLon,MaxLon,MinLat,MaxLat,Model_Dat,pnew,intyp,kxtrp,Station_Lons,Station_Lats):

    class model_dat_proc:

        #get the model data from the base and test cases for the region
        if (MinLon < 0 and MaxLon > 0): #if the region crosses the date line - do different processing
            O3_00=Model_Dat.o3.sel(lon=slice(MinLon+360.0,360.0),lat=slice(MinLat,MaxLat))
            O3_01=Model_Dat.o3.sel(lon=slice(0,MaxLon),lat=slice(MinLat,MaxLat))
            O3_0 = np.concatenate( (O3_00,O3_01),axis=3)
            PS_00=Model_Dat.ps.sel(lon=slice(MinLon+360.0,360.0),lat=slice(MinLat,MaxLat))
            PS_01=Model_Dat.ps.sel(lon=slice(0,MaxLon),lat=slice(MinLat,MaxLat))
            PS_0 = np.concatenate( (PS_00,PS_01),axis=2)
            lon_00=Model_Dat.lon.sel(lon=slice(MinLon+360.0,360.0))
            lon_01=Model_Dat.lon.sel(lon=slice(0,MaxLon))
            lon_0 = np.concatenate( (lon_00,lon_01))
         
            #resort the arrays as needed
            lon_sort=lon_0.argsort()
            O3_0 = O3_0[:,:,:,lon_sort]
            PS_0 = PS_0[:,:,lon_sort]
            lon_0  = lon_0[lon_sort]
         
            O3_sfc=np.squeeze(O3_0[:,-1,:,:])*1.0e9 #get the lowest model surface level data
         
        else: #if the region does not cross the date line
         
            O3_0=Model_Dat.o3.sel(lon=slice(MinLon,MaxLon),lat=slice(MinLat,MaxLat))
            PS_0=Model_Dat.ps.sel(lon=slice(MinLon,MaxLon),lat=slice(MinLat,MaxLat))
            lon_0=Model_Dat.lon.sel(lon=slice(MinLon,MaxLon))
            O3_sfc=np.squeeze(O3_0.values[:,-1,:,:])*1.0e9
         

        lat_0=Model_Dat.lat.sel(lat=slice(MinLat,MaxLat))

        O3_sfc_04=np.mean(O3_sfc,axis=(1,2))

        O3_0I = Ngl.vinth2p(O3_0,Model_Dat.hyam,Model_Dat.hybm,pnew,PS_0,intyp,1000.0,1,kxtrp)*1.0e9
      
        #Get the seasonal cycle of the base case at each needed pressure level
        #and average over the region.
        O3_01=np.mean(O3_0I[:,0,:,:],axis=(1,2))
        O3_02=np.mean(O3_0I[:,1,:,:],axis=(1,2))
        O3_03=np.mean(O3_0I[:,2,:,:],axis=(1,2))
        O3_04=np.mean(O3_0I[:,3,:,:],axis=(1,2))
      
        #Interpolate the model at each ozonesonde location and extract the needed information for the plot
        Station_Find = np.where( (Station_Lons >= MinLon) & (Station_Lons <= MaxLon) & (Station_Lats >= MinLat) & (Station_Lats <= MaxLat))
        ILAT=Station_Lats[Station_Find]
        ILON=Station_Lons[Station_Find]
      
        #Set the points to interpolate to
        for i in range(1,13):
            for j in range(0,len(ILAT)):
                if (i == 1 and j == 0):
                    O3_Pt=[i,float(ILAT[j]),float(ILON[j])]
                else:
                    O3_Pt=np.vstack( (O3_Pt,[i,float(ILAT[j]),float(ILON[j])] ))
      
        months=[1,2,3,4,5,6,7,8,9,10,11,12]
      
        #set up the regular grid interpolator for each case and level
        interp_0 = RegularGridInterpolator((months,lat_0,lon_0), np.squeeze(O3_0I[:,0,:,:]))
        interp_1 = RegularGridInterpolator((months,lat_0,lon_0), np.squeeze(O3_0I[:,1,:,:]))
        interp_2 = RegularGridInterpolator((months,lat_0,lon_0), np.squeeze(O3_0I[:,2,:,:]))
        interp_3 = RegularGridInterpolator((months,lat_0,lon_0), O3_sfc)
      
        #interpolate the model data at each case and pressure level
        O3_station_0 = interp_0(O3_Pt)
        O3_station_1 = interp_1(O3_Pt)
        O3_station_2 = interp_2(O3_Pt)
        O3_station_3 = interp_3(O3_Pt)
      
        #loop through each station interpolation and average like months
        O3M=np.squeeze(O3_Pt[:,0])
      
        O3_station_00=[]
        O3_station_10=[]
        O3_station_20=[]
        O3_station_30=[]
      
        for i in range(1,13):
            Elms = np.where(O3M == i)
            O3_station_00.append(np.mean(O3_station_0[Elms]))
            O3_station_10.append(np.mean(O3_station_1[Elms]))
            O3_station_20.append(np.mean(O3_station_2[Elms]))
            O3_station_30.append(np.mean(O3_station_3[Elms]))
      
        #For consistency convert the O3_station variables into numpy arrays
        O3_station_00=np.array(O3_station_00)
        O3_station_10=np.array(O3_station_10)
        O3_station_20=np.array(O3_station_20)
        O3_station_30=np.array(O3_station_30)
    
    Processed_Model_Dat=model_dat_proc()
      
    return Processed_Model_Dat
    
#-----------------------------------------------------------------------------------------
#Subroutine - Process Model Profiles
#-----------------------------------------------------------------------------------------
def process_model_profiles(Model_Dat,O3_0,PS_0,pnew,intyp,kxtrp,ILAT,ILON,lat_0,lon_0):

    class model_dat_proc:

      O3_0I1 = Ngl.vinth2p(O3_0,Model_Dat.hyam,Model_Dat.hybm,pnew,PS_0,intyp,1000.0,1,kxtrp)*1.0e9
      
      Locate_Bad=np.where(O3_0I1 > 10000.0)
      if len(Locate_Bad) > 0:
          O3_0I1[Locate_Bad]=np.nan
      
      #Get the monthly of the case at each needed pressure level
      #and average over the region.
      O3_011=np.nanmean(O3_0I1[0,:,:,:],axis=(1,2))
      O3_021=np.nanmean(O3_0I1[3,:,:,:],axis=(1,2))
      O3_031=np.nanmean(O3_0I1[6,:,:,:],axis=(1,2))
      O3_041=np.nanmean(O3_0I1[9,:,:,:],axis=(1,2))
      
      #Set the points to interpolate to
      for i in range(0,len(pnew)):
         for j in range(0,len(ILAT)):
            if (i == 0 and j == 0):
                O3_Pt=[pnew[i],float(ILAT[j]),float(ILON[j])]
            else:
                O3_Pt=np.vstack( (O3_Pt,[pnew[i],float(ILAT[j]),float(ILON[j])] ))
      
      #set up the regular grid interpolator for each case and level
      interp_01 = RegularGridInterpolator((pnew,lat_0,lon_0), np.squeeze(O3_0I1[0,:,:,:]))
      interp_11 = RegularGridInterpolator((pnew,lat_0,lon_0), np.squeeze(O3_0I1[3,:,:,:]))
      interp_21 = RegularGridInterpolator((pnew,lat_0,lon_0), np.squeeze(O3_0I1[6,:,:,:]))
      interp_31 = RegularGridInterpolator((pnew,lat_0,lon_0), np.squeeze(O3_0I1[9,:,:,:]))
      
      #interpolate the model data at each case and pressure level
      O3_station_01 = interp_01(O3_Pt)
      O3_station_11 = interp_11(O3_Pt)
      O3_station_21 = interp_21(O3_Pt)
      O3_station_31 = interp_31(O3_Pt)
      
      #loop through each station interpolation and average like plevs
      O3M=np.squeeze(O3_Pt[:,0])
      
      O3_station_001=[]
      O3_station_101=[]
      O3_station_201=[]
      O3_station_301=[]
      
      for i in range(0,len(pnew)):
         Elms = np.where(O3M == pnew[i])
         O3_station_001.append(np.nanmean(O3_station_01[Elms]))
         O3_station_101.append(np.nanmean(O3_station_11[Elms]))
         O3_station_201.append(np.nanmean(O3_station_21[Elms]))
         O3_station_301.append(np.nanmean(O3_station_31[Elms]))

      #For consistency convert the O3_station variables into numpy arrays
      O3_station_001=np.array(O3_station_001)
      O3_station_101=np.array(O3_station_101)
      O3_station_201=np.array(O3_station_201)
      O3_station_301=np.array(O3_station_301)
      
    Processed_Model_Dat=model_dat_proc()
      
    return Processed_Model_Dat

#-----------------------------------------------------------------------------------------
#Primary Ozone Diagnostics Routine
#-----------------------------------------------------------------------------------------
def ozone_diagnostics (adfobj):
   
   #----------------------------------------------------------------------------------
   #Extract relevant info from the ADF
   #----------------------------------------------------------------------------------
   obsdir='/glade/campaign/acom/acom-climate/tilmes/obs_data/amwg/amwg_data/obs_data/cam-chem/' #location of the ozonesonde data files
   cam_climo_loc = adfobj.get_cam_info('cam_climo_loc',required=True)[0]
   baseline_climo_loc = adfobj.get_baseline_info('cam_climo_loc',required=True)
   ncases = len(cam_climo_loc)
   
   #check the number of cases.  For the O3 diagnostics there needs to be 2.
   #If there is more than 2, then only the first two will be used.
   case_test = adfobj.get_cam_info('cam_case_name', required=True)[0]
   case_base = adfobj.get_baseline_info('cam_case_name',required=True)
   
   #Grab all case nickname(s)
   test_nicknames = adfobj.case_nicknames["test_nicknames"]
   base_nickname = adfobj.case_nicknames["base_nickname"]
   
   plot_locations = adfobj.plot_location[0]
   
   climo_base=baseline_climo_loc+'/'+case_base+"_O3_climo.nc"
   if os.path.isfile(climo_base):
       print(climo_base," located")
   else:
       print(climo_base," could not be located.  Exiting O3 diagnostics.")
       return
   
   climo_test=cam_climo_loc+'/'+case_test+"_O3_climo.nc"
   if os.path.isfile(climo_test):
       print(climo_test," located")
   else:
       print(climo_test," could not be located. Exiting O3 diagnostics.")
       return
      
   #check if existing plots need to be redone
   redo_plot = adfobj.get_basic_info('redo_plot')
   print(f"\t NOTE: redo_plot is set to {redo_plot}")
         
   Compare_Obs=0 #initially assumes that user is comparing two models
   if adfobj.compare_obs:
       Compare_Obs=1 #if comparing a model with observations then don't need to include the base case in the plots
     
   print("-------------------------------")
   print("Processing Ozone Diagnostics...")
   print("-------------------------------")      
   
   #--------------------------------------------------------------------------------------
   #Check and make sure that if the O3 Diagnostics are selected in the .yaml file that
   #ozone (O3) is listed in the diag_var_list.
   #--------------------------------------------------------------------------------------
   if not ('O3' in adfobj.diag_var_list):
      msg = "No ozone ('O3') variable present"
      msg += " in 'diag_var_list', so O3 diagnostic plots will"
      msg += " be skipped."
      print(msg)
      return
   
   #--------------------------------------------------------------------------------------
   #Get the ozone sonde data from all of the stations
   #--------------------------------------------------------------------------------------
   Data = open_process_sonde_data_simone(obsdir)
   Obs_Mean=Data[0] #mean values
   Obs_Width=Data[1] #width of the data
   Obs_StdDev=Data[2] #Standard deviation of the data
   Obs_Pressure=Data[3] #Pressure of the obs data
   Regions=Data[4] #regions
   
   #--------------------------------------------------------------------------------------
   #Get model data from get_model_data()
   #--------------------------------------------------------------------------------------
   Test_Data=get_model_data(climo_test) #Grab test model data
   if Compare_Obs <= 0:
      Base_Data=get_model_data(climo_base) #Grab base model data
   
   #--------------------------------------------------------------------------------------   
   #Loop through each station and get the station coordinates
   #--------------------------------------------------------------------------------------
   Station_Lats=[]
   Station_Lons=[]
   Station_SNames=[]
   
   NStations=define_stations(0)[1]
   for i in range(0,NStations+1):
      Station_Info=define_stations(i)[0]
      if (Station_Info[3] < 0.0): 
          Station_Lons=np.append(Station_Lons,float(Station_Info[3])+360.0)
      else:
          Station_Lons=np.append(Station_Lons,float(Station_Info[3]))
      Station_Lats=np.append(Station_Lats,float(Station_Info[2]))
      Station_SNames=np.append(Station_SNames,Station_Info[0])

   #--------------------------------------------------------------------------------------
   #set the parameters for interpolation to pressure levels
   #--------------------------------------------------------------------------------------
   pnew = [50,250,500,900]
   intyp = 2     # 1=linear, 2=log, 3=log-log
   kxtrp = False # True=extrapolate (when the output pressure level is outside of the range of psrf)
   months=[1,2,3,4,5,6,7,8,9,10,11,12]

   #--------------------------------------------------------------------------------------
   #loop through each region and plot the Data
   #--------------------------------------------------------------------------------------
   for i in range(1,len(Regions)+1):
      
      #-----------------------------------------------------------------------------------
      #Grab the info for the current region of interest
      #-----------------------------------------------------------------------------------
      Region_Info=define_regions(i)[0]
      SName=Region_Info[0]
      LName=Region_Info[1]
      MinLat=Region_Info[2]
      MaxLat=Region_Info[3]
      MinLon=Region_Info[4]
      MaxLon=Region_Info[5]
      oFile_Seasonal = plot_locations+'/O3SeasonalCycle_'+SName+'_Special.png'
      oFile_Profile = plot_locations+'/O3Profile_'+SName+'_Special.png'
      
      #-----------------------------------------------------------------------------------
      #Check if redo_plot set and if not and plots exist already then
      #add them to the website (if enabled) and then exit the routine
      #-----------------------------------------------------------------------------------
      if (not(redo_plot)) and (os.path.isfile(oFile_Seasonal)) and (os.path.isfile(oFile_Profile)):
          print(SName,' region plots exist and redo_plot is false.  Adding to website and Skipping plot.')
          adfobj.add_website_data(oFile_Seasonal,SName.replace("_","")+"_SeasonalCycle", None, season="ANN",multi_case=True,category="O3_DIAGNOSTICS")
          adfobj.add_website_data(oFile_Profile,SName.replace("_","")+"_Profile", None, season="ANN", multi_case=True,category="O3_DIAGNOSTICS")
          continue
      else:
          print("Plotting Region ",LName)
      #-----------------------------------------------------------------------------------
      #Process the model data
      #-----------------------------------------------------------------------------------
      Processed_Seasonal_Cycle_Test_Data = process_model_seasonal_cycle(MinLon,MaxLon,MinLat,MaxLat,Test_Data,pnew,intyp,kxtrp,Station_Lons,Station_Lats)
      if Compare_Obs <= 0:
          Processed_Seasonal_Cycle_Base_Data = process_model_seasonal_cycle(MinLon,MaxLon,MinLat,MaxLat,Base_Data,pnew,intyp,kxtrp,Station_Lons,Station_Lats)
      else:
          Processed_Seasonal_Cycle_Base_Data=Processed_Seasonal_Cycle_Test_Data #allows the plotting routine to run without having to call it different ways.
      
      #-----------------------------------------------------------------------------------
      #Filter to the appropriate ozone sonde regional data.
      #-----------------------------------------------------------------------------------
      Obs_Mean_0=Obs_Mean[:,:,i-1]
      Obs_Width_0=Obs_Width[:,:,i-1]
      Obs_StdDev_0=Obs_StdDev[:,:,i-1]
      Obs_Pressure_0=Obs_Pressure[:,i-1]
      Regions_0=Regions[i-1]
      
      #-----------------------------------------------------------------------------------
      #Get the pressure level index for each of the 4 levels to 
      #filter the ozone sonde data to
      #-----------------------------------------------------------------------------------
      Locate_P0=np.where(Obs_Pressure_0 == pnew[0])[0]
      Locate_P1=np.where(Obs_Pressure_0 == pnew[1])[0]
      Locate_P2=np.where(Obs_Pressure_0 == pnew[2])[0]
      Locate_P3=np.where(Obs_Pressure_0 == np.max(Obs_Pressure_0))[0]
      
      #-----------------------------------------------------------------------------------
      #For Boulder specifically, the Index should be 2.
      #-----------------------------------------------------------------------------------
      if LName == 'Boulder':
          Locate_P3=2
      
      #-----------------------------------------------------------------------------------
      #Trim the ozonesonde data to the appropriate pressure levels
      #-----------------------------------------------------------------------------------
      Obs_Mean_00=np.squeeze(Obs_Mean_0[Locate_P0,:])
      Obs_Mean_10=np.squeeze(Obs_Mean_0[Locate_P1,:])
      Obs_Mean_20=np.squeeze(Obs_Mean_0[Locate_P2,:])
      Obs_Mean_30=np.squeeze(Obs_Mean_0[Locate_P3,:])
      
      Obs_Width_00=np.squeeze(Obs_Width_0[Locate_P0,:])
      Obs_Width_10=np.squeeze(Obs_Width_0[Locate_P1,:])
      Obs_Width_20=np.squeeze(Obs_Width_0[Locate_P2,:])
      Obs_Width_30=np.squeeze(Obs_Width_0[Locate_P3,:])
      
      Obs_StdDev_00=np.squeeze(Obs_StdDev_0[Locate_P0,:])
      Obs_StdDev_10=np.squeeze(Obs_StdDev_0[Locate_P1,:])
      Obs_StdDev_20=np.squeeze(Obs_StdDev_0[Locate_P2,:])
      Obs_StdDev_30=np.squeeze(Obs_StdDev_0[Locate_P3,:])
      
      Obs_Low_00=Obs_Mean_00-Obs_StdDev_00
      Obs_Low_10=Obs_Mean_10-Obs_StdDev_10
      Obs_Low_20=Obs_Mean_20-Obs_StdDev_20
      Obs_Low_30=Obs_Mean_30-Obs_StdDev_30
      
      Obs_High_00=Obs_Mean_00+Obs_StdDev_00
      Obs_High_10=Obs_Mean_10+Obs_StdDev_10
      Obs_High_20=Obs_Mean_20+Obs_StdDev_20
      Obs_High_30=Obs_Mean_30+Obs_StdDev_30
         
      #-----------------------------------------------------------------------------------   
      #Assign the output seasonal cycle filename and Plot the Seasonal Cycle Plot
      #-----------------------------------------------------------------------------------
      Plot_Seasonal_Cycle_Profile(Processed_Seasonal_Cycle_Base_Data.O3_01,Processed_Seasonal_Cycle_Test_Data.O3_01,months,Processed_Seasonal_Cycle_Base_Data.O3_02,Processed_Seasonal_Cycle_Test_Data.O3_02,months,Processed_Seasonal_Cycle_Base_Data.O3_03,Processed_Seasonal_Cycle_Test_Data.O3_03,months,Processed_Seasonal_Cycle_Base_Data.O3_sfc_04,Processed_Seasonal_Cycle_Test_Data.O3_sfc_04,months,Obs_Mean_00,Obs_Mean_10,Obs_Mean_20,Obs_Mean_30,Obs_Low_00,Obs_Low_10,Obs_Low_20,Obs_Low_30,Obs_High_00,Obs_High_10,Obs_High_20,Obs_High_30,Processed_Seasonal_Cycle_Base_Data.O3_station_00,Processed_Seasonal_Cycle_Test_Data.O3_station_00,Processed_Seasonal_Cycle_Base_Data.O3_station_10,Processed_Seasonal_Cycle_Test_Data.O3_station_10,Processed_Seasonal_Cycle_Base_Data.O3_station_20,Processed_Seasonal_Cycle_Test_Data.O3_station_20,Processed_Seasonal_Cycle_Base_Data.O3_station_30,Processed_Seasonal_Cycle_Test_Data.O3_station_30,[0,6000],[0,750],[0,120],[0,75],'50 hPa','250 hPa','500 hPa','Sfc',base_nickname,test_nicknames,LName,oFile_Seasonal,0,Compare_Obs)

      #-----------------------------------------------------------------------------------
      #Set up for the monthly profile plots - I am calculating my own pressure level data here to ensure a tight match with the ozone sonde climatology
      #-----------------------------------------------------------------------------------
      pnew1=[0.01,0.05,0.1,0.15,0.25,0.5,0.75,1.0,5.0,10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,125.0,150.0,175.0,200.0,225.0,250.0,275.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0,900.0,950.0,1000.0]
      pnew1=np.flip(Obs_Pressure_0)
      
      #-----------------------------------------------------------------------------------
      #Process the model data 
      #-----------------------------------------------------------------------------------
      Processed_Profiles_Test = process_model_profiles(Test_Data,Processed_Seasonal_Cycle_Test_Data.O3_0,Processed_Seasonal_Cycle_Test_Data.PS_0,pnew1,intyp,kxtrp,Processed_Seasonal_Cycle_Test_Data.ILAT,Processed_Seasonal_Cycle_Test_Data.ILON,Processed_Seasonal_Cycle_Test_Data.lat_0,Processed_Seasonal_Cycle_Test_Data.lon_0)
      if Compare_Obs<=0:
          Processed_Profiles_Base = process_model_profiles(Base_Data,Processed_Seasonal_Cycle_Base_Data.O3_0,Processed_Seasonal_Cycle_Base_Data.PS_0,pnew1,intyp,kxtrp,Processed_Seasonal_Cycle_Base_Data.ILAT,Processed_Seasonal_Cycle_Base_Data.ILON,Processed_Seasonal_Cycle_Base_Data.lat_0,Processed_Seasonal_Cycle_Base_Data.lon_0)      
      else:
          Processed_Profiles_Base=Processed_Profiles_Test
      
      #-----------------------------------------------------------------------------------
      #Trim the ozonesonde data to the appropriate pressure levels
      #-----------------------------------------------------------------------------------
      Obs_Mean_001=np.squeeze(Obs_Mean_0[:,0])
      Obs_Mean_101=np.squeeze(Obs_Mean_0[:,3])
      Obs_Mean_201=np.squeeze(Obs_Mean_0[:,6])
      Obs_Mean_301=np.squeeze(Obs_Mean_0[:,9])
      
      Obs_Width_001=np.squeeze(Obs_Width_0[:,0])
      Obs_Width_101=np.squeeze(Obs_Width_0[:,3])
      Obs_Width_201=np.squeeze(Obs_Width_0[:,6])
      Obs_Width_301=np.squeeze(Obs_Width_0[:,9])
      
      Obs_StdDev_001=np.squeeze(Obs_StdDev_0[:,0])
      Obs_StdDev_101=np.squeeze(Obs_StdDev_0[:,3])
      Obs_StdDev_201=np.squeeze(Obs_StdDev_0[:,6])
      Obs_StdDev_301=np.squeeze(Obs_StdDev_0[:,9])
      
      Obs_Low_001=Obs_Mean_001-Obs_StdDev_001
      Obs_Low_101=Obs_Mean_101-Obs_StdDev_101
      Obs_Low_201=Obs_Mean_201-Obs_StdDev_201
      Obs_Low_301=Obs_Mean_301-Obs_StdDev_301
      
      Obs_High_001=Obs_Mean_001+Obs_StdDev_001
      Obs_High_101=Obs_Mean_101+Obs_StdDev_101
      Obs_High_201=Obs_Mean_201+Obs_StdDev_201
      Obs_High_301=Obs_Mean_301+Obs_StdDev_301

      #-----------------------------------------------------------------------------------
      #Assign the output monthly profile filename and Plot the Seasonal Cycle Plot
      #-----------------------------------------------------------------------------------
      Plot_Seasonal_Cycle_Profile(Processed_Profiles_Base.O3_011,Processed_Profiles_Test.O3_011,pnew1,Processed_Profiles_Base.O3_021,Processed_Profiles_Test.O3_021,pnew1,Processed_Profiles_Base.O3_031,Processed_Profiles_Test.O3_031,pnew1,Processed_Profiles_Base.O3_041,Processed_Profiles_Test.O3_041,pnew1,np.flip(Obs_Mean_001),np.flip(Obs_Mean_101),np.flip(Obs_Mean_201),np.flip(Obs_Mean_301),np.flip(Obs_Low_001),np.flip(Obs_Low_101),np.flip(Obs_Low_201),np.flip(Obs_Low_301),np.flip(Obs_High_001),np.flip(Obs_High_101),np.flip(Obs_High_201),np.flip(Obs_High_301),Processed_Profiles_Base.O3_station_001,Processed_Profiles_Test.O3_station_001,Processed_Profiles_Base.O3_station_101,Processed_Profiles_Test.O3_station_101,Processed_Profiles_Base.O3_station_201,Processed_Profiles_Test.O3_station_201,Processed_Profiles_Base.O3_station_301,Processed_Profiles_Test.O3_station_301,[0,125],[0,125],[0,125],[0,125],'Jan', 'Apr','Jul','Oct',base_nickname,test_nicknames,LName,oFile_Profile,1,Compare_Obs)

      #-----------------------------------------------------------------------------------
      #Once the plots have successfully run, add the web page entries (if enabled).
      #-----------------------------------------------------------------------------------
      adfobj.add_website_data(oFile_Seasonal,SName.replace("_","")+"_SeasonalCycle", None, season="ANN",multi_case=True,category="O3_DIAGNOSTICS")
      adfobj.add_website_data(oFile_Profile,SName.replace("_","")+"_Profile", None, season="ANN", multi_case=True,category="O3_DIAGNOSTICS")
      
   print("Ozone Diagnostics Generated Successfully!")
