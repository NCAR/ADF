#!/usr/bin/env python
# coding: utf-8
#CESM and MOPITT monthly comparisons with maps --- 2002-2021

# loading
import h5py                                # For loading he5 files
import glob
# Processing
import xarray as xr
import numpy as np
from scipy import interpolate              # for vertical interpolation
import time                                # for timing code
#plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs                 # For plotting maps
import cartopy.feature as cfeature         # For plotting maps
from cartopy.util import add_cyclic_point  # For plotting maps
import datetime
import os


def load_and_extract_grid_hdf(filename,varname):
    he5_load = h5py.File(filename, mode='r')
    lat = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Latitude"][:]
    lon = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Longitude"][:]
    alt = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Pressure2"][:]
    alt_short = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/Pressure"][:]
    
    #LAT-LON variables
    if varname=='column':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/RetrievedCOTotalColumnDay"][:]
    elif varname=='apriori_col':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOTotalColumnDay"][:]
    elif varname=='apriori_surf':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOSurfaceMixingRatioDay"][:]
    elif varname=='pressure_surf':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/SurfacePressureDay"][:]
    #LAT-LON-ALT variables    
    elif varname=='ak_col':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/TotalColumnAveragingKernelDay"][:]
    elif varname=='apriori_prof':
        data_loaded = he5_load["/HDFEOS/GRIDS/MOP03/Data Fields/APrioriCOMixingRatioProfileDay"][:]

    # create xarray DataArray
    if (varname=='column' or varname=='apriori_col'
        or varname=='apriori_surf'or varname=='pressure_surf'):
        dataset_new = xr.DataArray(data_loaded, dims=["lon","lat"], coords=[lon,lat])
    elif (varname=='ak_col'):
        dataset_new = xr.DataArray(data_loaded, dims=["lon","lat","alt"], coords=[lon,lat,alt])
    elif (varname=='apriori_prof'):
        dataset_new = xr.DataArray(data_loaded, dims=["lon","lat","alt"], coords=[lon,lat,alt_short])
    
    # missing value -> nan
    ds_masked = dataset_new.where(dataset_new != -9999.)
    he5_load.close()
    
    return ds_masked

def collect_mopitt_data(files, varname):
    count = 0
    for filename in files:
        data = load_and_extract_grid_hdf(filename, varname)
        if count == 0:
            data_array = data
            count += 1
        else:
            data_array = xr.concat([data_array, data], 'time')
            
    return data_array

def vertical_regrid(input_press, input_values, output_press):
    ''' Dimensions must be: '''

    regrid_array = xr.full_like(output_press, np.nan)
    for t in range (input_press.shape[0]):
        print(t)
        # Latitude values
        for y in range (input_press.shape[2]):
            # Longitude values
            for x in range (input_press.shape[3]):
                xx = input_press.isel(time=t, lat=y, lon=x)
                yy = input_values.isel(time=t, lat=y, lon=x)
                f = interpolate.interp1d(xx, yy, fill_value="extrapolate")
                xnew = output_press.isel(time=t, lat=y, lon=x)
                regrid_array[t, x, y, :] = f(xnew)
    return regrid_array

def MOPITT(adfobj):

    # ### Load Measurements
    # MOPITT CO V9 daytime only joint product, L3 gridded 1x1.
    # Climatology data archived through NCAR GDEX, Data Reference: https://gdex.ucar.edu/dataset/369_buchholz.html

    files = sorted(glob.glob('/glade/campaign/acom/acom-da/buchholz/MOPITT_v9_climo/*2002_2021.he5', recursive=False))

    data_array = collect_mopitt_data(files, "column")

    sd_files = sorted(glob.glob('/glade/campaign/acom/acom-da/buchholz/MOPITT_v9_climo/*2002_2021_SD.he5', recursive=False))

    # Latitude and lanogitude not loading correctly
    sd_array = collect_mopitt_data(sd_files, "column")
    #Replace coordinate variables (because SD of coordinate variables was saved and that is zero)
    sd_array = sd_array.assign_coords(lon=data_array.lon, lat=data_array.lat)

    #surface pressure
    sat_psurf  = collect_mopitt_data(files, "pressure_surf")

    # load column and surface a priori
    prior_col_array = collect_mopitt_data(files, "apriori_col")
    prior_surf = collect_mopitt_data(files, "apriori_surf")

    # load profile a priori
    prior_prof_array = collect_mopitt_data(files, "apriori_prof")

    # averaging kernel
    ak_array = collect_mopitt_data(files, "ak_col")

    # broadcast 9 levels 900 to 100 hPa repeated everywhere
    dummy, press_dummy_arr = xr.broadcast(prior_prof_array,prior_prof_array.alt)

    # create array to hold 9 regular spaced, plus floating surface pressure
    sat_pressure_array = xr.full_like(ak_array, np.nan)
    sat_pressure_array[:,:,:,0] = sat_psurf
    sat_pressure_array[:,:,:,1:10] = press_dummy_arr
    #print(sat_pressure_array.isel(time=0, lat=-45, lon=-80))

    #Correct for where MOPITT surface pressure <900 hPa
    #calculate pressure differences
    dp = xr.full_like(ak_array, np.nan)
    dp.shape
    dp[:,:,:,9] = 1000
    for z in range (0, 9):
        dp[:,:,:,z] = sat_pressure_array[:,:,:,0] - sat_pressure_array[:,:,:,z+1]

    #Repeat surface values at all levels to replace in equivalent position in parray if needed
    psurfarray = xr.full_like(ak_array, np.nan)
    for z in range (0, 10):
        psurfarray[:,:,:,z] = sat_psurf
    
    #Add fill values below true surface
    new_pressure_array = sat_pressure_array.copy()
    new_pressure_array = new_pressure_array.where(dp>0)
    #replace lowest pressure with surface pressure
    new_pressure_array = psurfarray.where((dp>0) & (dp<100),new_pressure_array)

    # Model layer values are averages for the whole box, centred at an altitude,
    # while MOPITT values are averages described for the whole box above level.
    # Therefore need to interp MOPITT pressures to mid-box locations
    pinterp = xr.full_like(ak_array, np.nan)
    pinterp[:,:,:,9] = 87.
    for z in range (0, 9):
        pinterp[:,:,:,z] = new_pressure_array[:,:,:,z] - (new_pressure_array[:,:,:,z]-new_pressure_array[:,:,:,z+1])/2

    # MOPITT surface values are stored separately to profile values because of the floating surface pressure.
    # So, for calculations, need to combine
    # Repeat surface a priori values at all levels to replace if needed
    apsurfarray = xr.full_like(ak_array, np.nan)
    for z in range (0, 10):
        apsurfarray[:,:,:,z] = prior_surf

    aparray = xr.full_like(ak_array, np.nan)
    aparray[:,:,:,0] = prior_surf
    aparray[:,:,:,1:10] = prior_prof_array
    aparray = aparray.where(dp>0)
    aparray = apsurfarray.where((dp>0) & (dp<100),aparray)

    # ### Define the directories and file of interest for your model results.
    
    ntest = len(adfobj.get_cam_info('cam_case_name', required=True))
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    
    for i in range(0,ntest):
    
        test_case = adfobj.get_cam_info('cam_case_name', required=True)[i]
        cam_climo_loc = adfobj.get_cam_info('cam_climo_loc',required=True)
        cam_hist_loc = adfobj.get_cam_info('cam_hist_loc',required=True)[i]
        test_nicknames = adfobj.case_nicknames["test_nicknames"][i]
        plot_locations = adfobj.plot_location[i]

        # Will need to calculate and load a climatology of output for ADF processing
        result_dir = cam_hist_loc
        
        hist_str = adfobj.get_basic_info('hist_str')
        hfiles = result_dir+"/*cam.h0a.*"

        #the netcdf file is now held in an xarray dataset named 'nc_load' and can be referenced later in the notebook
        nc_load = xr.open_mfdataset(hfiles,combine='nested',concat_dim='time')
        nc_load=nc_load.groupby('time.month').mean(dim='time')
        nc_load['month']=[datetime.datetime(2003,month,1) for month in nc_load['month'].values]
        nc_load=nc_load.rename({'month': 'time'})
        #to see what the netCDF file contains, just call the variable you read it into nc_load

        # Convert longitudes to -180 to 180
        nc_load = nc_load.assign_coords(lon=(((nc_load.lon + 180) % 360) - 180)).sortby('lon')

        #extract variable
        var_sel = nc_load['CO'].load()/1e-09

        # Model pressure values
        # model pressure is in Pa
        psurf = nc_load['PS'].load()
        pdel = nc_load['PDELDRY'].load()/100

        # Load values to create pressure array id PDELDRY was not saved
        # using hybrid coordinate variable definitons
        # https://www2.cesm.ucar.edu/models/atm-cam/docs/usersguide/node25.html
        # model pressure is in Pa
        # interfaces (edges)
        hyai = nc_load['hyai'].load()
        hybi = nc_load['hybi'].load()
        p0 = 100000.0
        lev = var_sel.coords['lev']
        num_lev = lev.shape[0]

        # Initialize pressure edge arrays
        mod_press_low = xr.zeros_like(var_sel)
        mod_press_top = xr.zeros_like(var_sel)

        # Calculate pressure edge arrays
        # CAM-chem layer indices start at the top and end at the bottom
        for j in range(num_lev):
            mod_press_top[:,j,:,:] = hyai[:,j]*p0 + hybi[:,j]*psurf
            mod_press_low[:,j,:,:] = hyai[:,j+1]*p0 + hybi[:,j+1]*psurf

        # Delta P in hPa
        mod_deltap = (mod_press_low - mod_press_top)/100
        # CHECK ----->
        # Calculated in python should be within 3 to 4 decimal places of model calculated PDELDRY

        # Pressure mid-layer values
        mod_press_mid = (mod_press_top + mod_press_low)/200

        dates = var_sel.coords['time']

        # ### Regrid model to MOPITT horizontal grid
        # Horizontal regridding can have many options. Currently using xarray interp

        # linear interp
        tracer_regrid = var_sel.interp(coords=dict(lat=data_array.lat, lon=data_array.lon), method='linear')
        mod_deltap_regrid = pdel.interp(coords=dict(lat=data_array.lat, lon=data_array.lon), method='linear')
        psurf_regrid = psurf.interp(coords=dict(lat=data_array.lat, lon=data_array.lon), method='linear')
        model_pressure_regrid = mod_press_mid.interp(coords=dict(lat=data_array.lat, lon=data_array.lon), method='linear')

        # ### Vertical interpolate model grid to MOPITT vertical layers

        start = time.perf_counter()
        model_vert_regrid = vertical_regrid(model_pressure_regrid, tracer_regrid, pinterp)
        end = time.perf_counter()
        print('This took '+ str((end-start)/60) + ' minutes to run')

        # ### Consistency check: plot profile
        # Need to check the interpolation and AK application is doing what I think it is doing.

        def profile_plot(x,y,color_choice,label_string,linewidth,marker):
            plt.plot(x, y, marker, label=label_string,
               color=color_choice,
               markersize=10, linewidth=linewidth,
               markerfacecolor=color_choice,
               markeredgecolor='grey',
               markeredgewidth=1)

        plt.figure(figsize=(8,10))
        ax = plt.axes()
        ax.invert_yaxis()

        #-------------------| variable |------------------------| pressure |---------
        profile_plot(aparray.isel(time=0, lat=45, lon=105), pinterp.isel(time=0, lat=45, lon=105), 'blue','MOPITT a priori',8,'-ok')
        profile_plot(var_sel.isel(time=0, lat=45, lon=105), mod_press_mid.isel(time=0, lat=45, lon=105), 'red','CAM-chem',8,'-ok')
        profile_plot(tracer_regrid.isel(time=0, lat=45, lon=105), model_pressure_regrid.isel(time=0, lat=45, lon=105), 'green','CAM-chem horizontal regrid',8,'-ok')
        profile_plot(model_vert_regrid.isel(time=0, lat=45, lon=105), pinterp.isel(time=0, lat=45, lon=105), 'gold','CAM-chem horizontal and vertical regrid',2,'-ok')

        #titles
        plt.title('Profile example at ',fontsize=24)        
        plt.xlabel('VMR',fontsize=24)
        plt.ylabel('Pressure',fontsize=24)

        # legend
        plt.legend(bbox_to_anchor=(1.9, 0.78),loc='lower right',fontsize=18)

        plt.show() 

        # ### Convert to total column for comparison

        # Convert base model and regridded model values to total column amounts in case you need for comparison. This step is not necessary for the final plots, but a good consistency check if needed at a later stage.

        #-------------------------------
        #CONSTANTS and conversion factor
        #-------------------------------
        NAv = 6.0221415e+23                       #--- Avogadro's number
        g = 9.81                                  #--- m/s - gravity
        MWair = 28.94                             #--- g/mol
        xp_const = (NAv* 10)/(MWair*g)*1e-09      #--- scaling factor for turning vmr into pcol
                                                 #--- (note 1*e-09 because in ppb)

        var_tcol = xr.dot(pdel, xp_const*var_sel, dims=["lev"])
        var_tcol_regrid = xr.dot(mod_deltap_regrid, xp_const*tracer_regrid, dims=["lev"])

        # ### Apply MOPITT AK
        # Smooth model data to measurement space according to the User Guide:
        # https://www2.acom.ucar.edu/sites/default/files/documents/v9_users_guide_20220203.pdf
        # 
        # C<sub>smooth</sub> = C<sub>prior</sub> + AK(x<sub>model</sub>−x<sub>prior</sub>)
        # 
        # C = column
        # x = profile
        # 
        # Note MOPITT AKs are applied to log<sub>10</sub>(VMR)
        # 

        #from math import log10
        log_ap = np.log10(aparray)
        log_model=np.log10(model_vert_regrid)

        diff_array = log_model-log_ap

        ak_appl = ak_array * diff_array
        smoothed_model = prior_col_array + np.sum(ak_appl, axis=3)

        # ### Difference
        # The model - observations array

        tcol_diff = smoothed_model - data_array

        # ### Plot the map comparisons

        # Add cyclic point to avoid white line over Africa
        lon_model = var_sel.lon

        var_tcol_cyc, lon_cyc = add_cyclic_point(var_tcol, coord=lon_model)

        month_titles = np.array(['January', 'February', 'March', 'April', 'May', 'June',
                           'July','August','September','October','November','December'])

        #Sub-plots function
        def map_subplot(lon,lat,var,contours,colormap,labelbar,position1,position2):
            axs[position1,position2].contourf(lon,lat,var,contours,cmap=colormap,extend=labelbar)
            axs[position1,position2].coastlines()
            #gridlines
            gl = axs[position1,position2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='grey', 
                                                    linewidth=2, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 42, 'color': 'gray'}
            gl.ylabel_style = {'size': 42, 'color': 'gray'}
            #latitude limits in degrees
            axs[position1,position2].set_ylim(-70,70)
            # add coastlines
            axs[position1,position2].add_feature(cfeature.COASTLINE)
            axs[position1,position2].add_feature(cfeature.LAKES)
            axs[position1,position2].add_feature(cfeature.BORDERS)

        #Create monthly plot
        for monthval in range(0,12):
    
            #MONTH definiton for plotting (0 to 11):
            
            oCompare_Plot=plot_locations+'/MOPITT_'+month_titles[monthval]+'_ANN_Special_Mean.png'
            
            if (not(redo_plot)) and (os.path.isfile(oCompare_Plot)):
                print(month_titles[monthval],' monthly plot exists and redo_plot is false.  Adding to website and Skipping plot.')
                adfobj.add_website_data(oCompare_Plot,"MOPITT_"+month_titles[monthval], None,season="ANN", multi_case=True,category="MOPITT_DIAGNOSTICS")
                continue
            else:
                print("Plotting month ",month_titles[monthval])

            #-----------------columns, rows
            fig, axs = plt.subplots(2,2,figsize=(50,21),
                                    subplot_kw={'projection': ccrs.PlateCarree()},
                                    constrained_layout=True)

            fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
            fig.suptitle('Total column carbon monoxide (CO): '+month_titles[monthval], fontsize=56)

            #------------------ Create the plots
            #define contour levels
            clev = np.arange(0.4, 3.6, 0.1)

            # Plot MOPITT columns
            x = 0
            y = 0
            map_subplot(data_array.lon,data_array.lat,data_array[monthval,:,:].transpose()/1e18,clev,'Spectral_r','both',x,y)
            axs[x,y].set_title('MOPITT CO 2002-2021', fontsize=56)

            # Plot MOPITT CO standard deviation
            x = 0
            y = 1
            map_subplot(data_array.lon,data_array.lat,sd_array[monthval,:,:].transpose()/1e17,clev,'Spectral_r','both',x,y)
            axs[x,y].set_title('MOPITT CO standard deviation', fontsize=56)

            # Plot CAM-chem columns smooth
            x = 1
            y = 0
            map_subplot(smoothed_model.lon,smoothed_model.lat,smoothed_model[monthval,:,:].transpose()/1e18,clev,'Spectral_r','both',x,y)
            axs[x,y].set_title('Model CO (smoothed)', fontsize=56)
    
            # Plot CAM-chem -- MOPITT diff
            #new contour levels
            clevII = np.arange(-3.5, 3.6, 0.1)
            x = 1
            y = 1
            map_subplot(tcol_diff.lon,tcol_diff.lat,tcol_diff[monthval,:,:].transpose()/1e18,clevII,'bwr','both',x,y)
            axs[x,y].set_title('Model - Obs difference', fontsize=56)

            #------------------ Axes titles
            # x-axis
            for j in range(2):
                axs[1,j].text(0.50, -0.25, 'Longitude', va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=axs[1,j].transAxes, fontsize=56)

            # y-axis
            for j in range(2):
                axs[j,0].text(-0.15, 0.52, 'Latitude', va='bottom', ha='center',
                    rotation='vertical', rotation_mode='anchor',
                    transform=axs[j,0].transAxes, fontsize=56)
    
            #------------------ Colorbar definitions
            # defining these colorbars re-plots the plots and can take some time... It would be great to find a better way to do this
            # column
            cbar_set = axs[0,0].contourf(data_array.lon,data_array.lat,data_array[monthval,:,:].transpose()/1e18,clev,cmap='Spectral_r',extend='both')
            cb = fig.colorbar(cbar_set, ax=axs[0, 0], shrink=0.95)
            cb.set_label(label='CO (x $10^{18}$ molec./cm$^2$)', fontsize=22)
            cb.ax.tick_params(labelsize=22)
            cbar_set = axs[1,0].contourf(smoothed_model.lon,smoothed_model.lat,smoothed_model[monthval,:,:].transpose()/1e18,clev,cmap='Spectral_r',extend='both')
            cb = fig.colorbar(cbar_set, ax=axs[1, 0], shrink=0.95)
            cb.set_label(label='CO (x $10^{18}$ molec./cm$^2$)', fontsize=22)
            cb.ax.tick_params(labelsize=22)
            # column SD
            cbar_set = axs[0,1].contourf(sd_array.lon,sd_array.lat,sd_array[monthval,:,:].transpose()/1e17,clev,cmap='Spectral_r',extend='both')
            cb = fig.colorbar(cbar_set, ax=axs[0, 1], shrink=0.95)
            cb.set_label(label='CO (x $10^{17}$ molec./cm$^2$)', fontsize=22)
            cb.ax.tick_params(labelsize=22)
            # difference
            cbarII_set = axs[1,1].contourf(tcol_diff.lon,tcol_diff.lat,tcol_diff[monthval,:,:].transpose()/1e18,clevII,cmap='bwr',extend='both')
            cbII = fig.colorbar(cbarII_set, ax=axs[1, 1], shrink=0.95)
            cbII.set_label(label='CO difference (x $10^{18}$ molec./cm$^2$)', fontsize=22)
            cbII.ax.tick_params(labelsize=22)

            plt.savefig(oCompare_Plot)
            print(oCompare_Plot)
            adfobj.add_website_data(oCompare_Plot,"MOPITT_"+month_titles[monthval], None,season="ANN", multi_case=True,category="MOPITT_DIAGNOSTICS")

            # ### Seasonal plot of differences

        # Create seasonal averages
        oCompare_Season=plot_locations+'/MOPITT_SEASONAL_ANN_Special_Mean.png'
        if (not(redo_plot)) and (os.path.isfile(oCompare_Season)):
            print(month_titles[monthval],' seasonal plot exists and redo_plot is false.  Adding to website and Skipping plot.')
            adfobj.add_website_data(oCompare_Season,'MOPITT_SEASONAL', None, season="ANN", multi_case=True,category="MOPITT_DIAGNOSTICS")
            continue
        else:
            print("Plotting seasonal")
        
        seas_diff =  xr.full_like(tcol_diff.isel(time=[0,1,2,3]), np.nan)
        seas_diff[0,:,:] = tcol_diff.isel(time=[0,1,11]).mean('time')
        seas_diff[1,:,:] = tcol_diff.isel(time=[2,3,4]).mean('time')
        seas_diff[2,:,:] = tcol_diff.isel(time=[5,6,7]).mean('time')
        seas_diff[3,:,:] = tcol_diff.isel(time=[8,9,10]).mean('time')
        seas_diff.shape

        titles = np.array([['(a) DJF','(b) MAM'],['(c) JJA','(d) SON']])

        fig, axs = plt.subplots(2,2,figsize=(50,21),
                                subplot_kw={'projection': ccrs.PlateCarree()},
                                constrained_layout=True)

        fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)

        fig.suptitle('Total column carbon monoxide difference: Model - MOPITT', fontsize=56)

        #define contour levels
        clev = np.arange(-3.5, 3.6, 0.1)

        for x in range(2):
            for y in range(2):
                season_index = x*2 + y
                map_subplot(seas_diff.lon,seas_diff.lat,seas_diff[season_index,:,:].transpose()/1e18,clev,'bwr','both',x,y)
                #sub-titles
                axs[x,y].set_title(titles[x,y], fontsize=48)
        
        #------------------ Axes titles
        # x-axis
        for j in range(2):
            axs[1,j].text(0.50, -0.25, 'Longitude', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=axs[1,j].transAxes, fontsize=56)

        # y-axis
        for j in range(2):
            axs[j,0].text(-0.15, 0.52, 'Latitude', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=axs[j,0].transAxes, fontsize=56)

        #------------------ Colorbar definitions
        # note the hard-coded value here... this overplots the top left plot
        cbar_set = axs[0,0].contourf(seas_diff.lon,seas_diff.lat,seas_diff[0,:,:].transpose()/1e18,clev,cmap='bwr',extend='both')
        cb = fig.colorbar(cbar_set, ax=axs[:, 1], shrink=0.7)
        cb.set_label(label='CO (x $10^{18}$ molec./cm$^2$)', fontsize=42)
        cb.ax.tick_params(labelsize=42)

        plt.savefig(oCompare_Season)
        adfobj.add_website_data(oCompare_Season,'MOPITT_SEASONAL', None, season='ANN', multi_case=True,category="MOPITT_DIAGNOSTICS")
