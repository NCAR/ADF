#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.
import matplotlib.pyplot as plt
import pandas as pd

#Format warning messages:
def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

def tem(adf):
    """
    Plot the contents of the TEM dignostic ouput of 2-D latitude vs vertical pressure maps.
    
    Steps:
     - loop through TEM variables
     - calculate all-time fields (from individual months)
     - Take difference, calculate statistics
     - make plot

    """

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    plot_location = Path(adf.plot_location[0])

    #Check if plot output directory exists, and if not, then create it:
    if not plot_location.is_dir():
        print(f"    {plot_location} not found, making new directory")
        plot_location.mkdir(parents=True)

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adf.get_cam_info("cam_case_name", required=True)

    res = adf.variable_defaults # will be dict of variable-specific plot preferences

    #Check if comparing against observations
    if adf.compare_obs:
        obs = True
        base_name = "Obs"
    else:
        obs = False
        base_name = adf.get_baseline_info("cam_case_name", required=True)
    #End if

    #Extract test case years
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    #Extract baseline years (which may be empty strings if using Obs):
    syear_baseline = adf.climo_yrs["syear_baseline"]
    eyear_baseline = adf.climo_yrs["eyear_baseline"]

    #Grab all case nickname(s)
    test_nicknames = adf.case_nicknames["test_nicknames"]
    base_nickname = adf.case_nicknames["base_nickname"]
    case_nicknames = test_nicknames + [base_nickname]
 
    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adf.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adf.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    #-----------------------------------------

    #Grab TEM diagnostics options
    tem_opts = adf.read_config_var("tem_info")

    #Location of saved TEM netCDF files
    tem_loc = tem_opts["tem_loc"]

    #If path not specified, skip TEM calculation
    if tem_loc is None:
        print("'tem_loc' not found in config file, so TEM plots will be skipped.")
        return
    else:
        #Notify user that script has started:
        print("\n  Generating TEM plots...")
    
    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    #Suggestion from Rolando, if QBO is being produced, add utendvtem and utendwtem?
    if "qbo" in adf.plotting_scripts:
        var_list = ['uzm','epfy','epfz','vtem','wtem',
                    'psitem','utendepfd','utendvtem','utendwtem']
    #Otherwise keep it simple
    else:
        var_list = ['uzm','epfy','epfz','vtem','wtem','psitem','utendepfd']

    #Check if comparing against obs
    #If so, create the obs TEM netCDF file
    if obs:
        #Create TEM file for observations
        input_loc_idx = Path(tem_loc) / base_name
        tem_base = input_loc_idx / f'{base_name}.TEMdiag.nc'
        ds_base = xr.open_dataset(tem_base)

    else:
        #Open the baseline TEM file, if it exists
        input_loc_idx = Path(tem_loc) / base_name
        tem_base = input_loc_idx / f'{base_name}.TEMdiag_{syear_baseline}-{eyear_baseline}.nc'
        ds_base = xr.open_dataset(tem_base)

    #Setup TEM plots
    nrows = len(var_list)
    ncols = len(case_nicknames)+1
    fig_width = 20

    #try and dynamically create size of fig based off number of cases
    fig_height = 15+(ncols*nrows)

    #Loop over season dictionary:
    for s in seasons:
        #Location to save plots
        plot_name = plot_location / f"{s}_TEM_Mean.png"
        
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width,fig_height),
                                facecolor='w', edgecolor='k')

        #Loop over model cases:
        for idx,case_name in enumerate(case_names):

            # Check redo_plot. If set to True: remove old plot, if it already exists:
            if (not redo_plot) and plot_name.is_file():
                #Add already-existing plot to website (if enabled):
                adf.add_website_data(plot_name, "TEM", case_name, season=s)

                #Continue to next iteration:
                continue
            elif (redo_plot) and plot_name.is_file():
                plot_name.unlink()

            #Extract start and end year values:
            start_year = syear_cases[idx]
            end_year   = eyear_cases[idx]

            #Open the TEM file
            output_loc_idx = Path(tem_loc) / case_name
            tem = output_loc_idx / f'{case_name}.TEMdiag_{start_year}-{end_year}.nc'

            #Grab the data for the TEM netCDF files
            ds = xr.open_dataset(tem)

            climo_yrs = {"test":[syear_cases[idx], eyear_cases[idx]],
                         "base":[syear_baseline, eyear_baseline]}

            #Setup and plot the sub-plots
            tem_plot(ds, ds_base, case_nicknames, axs, s, var_list, res, obs, climo_yrs)

        #Set figure title
        plt.suptitle(f'TEM Diagnostics: {s}', fontsize=20, y=.928)

        #Write the figure to provided workspace/file:
        fig.savefig(plot_name, bbox_inches='tight', dpi=300)

        #Add plot to website (if enabled):
        adf.add_website_data(plot_name, "TEM", case_name, season=s)

# Helper functions
##################

def tem_plot(ds, ds_base, case_names, axs, s, var_list, res, obs, climo_yrs):
    """
    TEM subplots
    
    """
    #Set empty message for comparison of cases with different vertical levels
    #TODO: Work towards getting the vertical and horizontal interpolations!! - JR
    empty_message = "These have different vertical levels\nCan't compare cases currently"
    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
    prop_x = 0.18
    prop_y = 0.42

    for var in var_list:
        #Grab variable defaults for this variable
        vres = res[var]

        #Gather data for both cases
        mdata = ds[var].squeeze()
        odata = ds_base[var].squeeze()

        #Create array to avoid weighting missing values:
        md_ones = xr.where(mdata.isnull(), 0.0, 1.0)
        od_ones = xr.where(odata.isnull(), 0.0, 1.0)

        month_length = mdata.time.dt.days_in_month
        weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())

        #Calculate monthly-weighted seasonal averages:
        if s == 'ANN':

            #Calculate annual weights (i.e. don't group by season):
            weights_ann = month_length / month_length.sum()

            mseasons = (mdata * weights_ann).sum(dim='time')
            mseasons = mseasons / (md_ones*weights_ann).sum(dim='time')

            #Calculate monthly weights based on number of days:
            if obs:
                month_length_obs = odata.time.dt.days_in_month
                weights_ann_obs = month_length_obs / month_length_obs.sum()
                oseasons = (odata * weights_ann_obs).sum(dim='time')
                oseasons = oseasons / (od_ones*weights_ann_obs).sum(dim='time')
            else:
                month_length_base = odata.time.dt.days_in_month
                weights_ann_base = month_length_base / month_length_base.sum()
                oseasons = (odata * weights_ann_base).sum(dim='time')
                oseasons = oseasons / (od_ones*weights_ann_base).sum(dim='time')

        else:
            #this is inefficient because we do same calc over and over
            mseasons = (mdata * weights).groupby("time.season").sum(dim="time").sel(season=s)
            wgt_denom = (md_ones*weights).groupby("time.season").sum(dim="time").sel(season=s)
            mseasons = mseasons / wgt_denom

            if obs:
                month_length_obs = odata.time.dt.days_in_month
                weights_obs = (month_length_obs.groupby("time.season") / month_length_obs.groupby("time.season").sum())
                oseasons = (odata * weights_obs).groupby("time.season").sum(dim="time").sel(season=s)
                wgt_denom = (od_ones*weights_obs).groupby("time.season").sum(dim="time").sel(season=s)
                oseasons = oseasons / wgt_denom
            else:
                month_length_base = odata.time.dt.days_in_month
                weights_base = (month_length_base.groupby("time.season") / month_length_base.groupby("time.season").sum())
                oseasons = (odata * weights_base).groupby("time.season").sum(dim="time").sel(season=s)
                wgt_denom_base = (od_ones*weights_base).groupby("time.season").sum(dim="time").sel(season=s)
                oseasons = oseasons / wgt_denom_base

        #difference: each entry should be (lat, lon)
        dseasons = mseasons-oseasons

        #Run through variables and plot each against the baseline on each row
        #Each column will be a case, ie (test, base, difference)

        # uzm
        #------------------------------------------------------------------------------------------
        if var == "uzm":
            mseasons.plot(ax=axs[0,0], y='lev', yscale='log',ylim=[1e3,1],
                                    cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[0,1], y='lev', yscale='log',ylim=[1e3,1],
                                    cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[0,2].text(prop_x, prop_y, empty_message, transform=axs[0,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[0,2], y='lev', yscale='log', ylim=[1e3,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # epfy
        #------------------------------------------------------------------------------------------
        if var == "epfy":
            mseasons.plot(ax=axs[1,0], y='lev', yscale='log',vmax=1e6,ylim=[1e2,1],
                                    cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[1,1], y='lev', yscale='log',vmax=1e6,ylim=[1e2,1],
                                    cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[1,2].text(prop_x, prop_y, empty_message, transform=axs[1,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[1,2], y='lev', yscale='log', vmax=1e6,
                            ylim=[1e2,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})
        
        # epfz
        #------------------------------------------------------------------------------------------
        if var == "epfz":
            mseasons.plot(ax=axs[2,0], y='lev', yscale='log',vmax=1e5,ylim=[1e2,1],
                                    cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[2,1], y='lev', yscale='log',vmax=1e5,ylim=[1e2,1],
                                    cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[2,2].text(prop_x, prop_y, empty_message, transform=axs[2,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[2,2], y='lev', yscale='log', vmax=1e5,
                            ylim=[1e2,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # vtem
        #------------------------------------------------------------------------------------------
        if var == "vtem":
            mseasons.plot.contourf(ax=axs[3,0], levels = 21, y='lev', yscale='log',
                                                vmax=3,vmin=-3,ylim=[1e2,1], cmap='RdBu_r',
                                                cbar_kwargs={'label': ds[var].units})
            mseasons.plot.contour(ax=axs[3,0], levels = 11, y='lev', yscale='log',
                                                vmax=3,vmin=-3,ylim=[1e2,1],
                                                colors='black', linestyles=None)

            oseasons.plot.contourf(ax=axs[3,1], levels = 21, y='lev', yscale='log',
                                                vmax=3,vmin=-3,ylim=[1e2,1], cmap='RdBu_r',
                                                cbar_kwargs={'label': ds[var].units})
            oseasons.plot.contour(ax=axs[3,1], levels = 11, y='lev', yscale='log',
                                                vmax=3,vmin=-3,ylim=[1e2,1],
                                                colors='black', linestyles=None)

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[3,2].text(prop_x, prop_y, empty_message, transform=axs[3,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[3,2], y='lev', yscale='log', vmax=3,vmin=-3,
                            ylim=[1e2,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # wtem
        #------------------------------------------------------------------------------------------
        if var == "wtem":
            mseasons.plot.contourf(ax=axs[4,0], levels = 21, y='lev', yscale='log',
                                                vmax=0.005, vmin=-0.005, ylim=[1e2,1], cmap='RdBu_r',
                                                cbar_kwargs={'label': ds[var].units})
            mseasons.plot.contour(ax=axs[4,0], levels = 7, y='lev', yscale='log',
                                            vmax=0.03, vmin=-0.03, ylim=[1e2,1],
                                            colors='black', linestyles=None)

            oseasons.plot.contourf(ax=axs[4,1], levels = 21, y='lev', yscale='log',
                                                vmax=0.005, vmin=-0.005, ylim=[1e2,1], cmap='RdBu_r',
                                                cbar_kwargs={'label': ds[var].units})
            oseasons.plot.contour(ax=axs[4,1], levels = 7, y='lev', yscale='log',
                                            vmax=0.03, vmin=-0.03, ylim=[1e2,1],
                                            colors='black', linestyles=None)

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[4,2].text(prop_x, prop_y, empty_message, transform=axs[4,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[4,2], y='lev', yscale='log',vmax=0.005, vmin=-0.005,
                            ylim=[1e2,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # psitem
        #------------------------------------------------------------------------------------------
        if var == "psitem":
            mseasons.plot.contourf(ax=axs[5,0], levels = 21, y='lev', yscale='log',
                                                vmax=5e9, ylim=[1e2,2],
                                                cbar_kwargs={'label': ds[var].units})

            oseasons.plot.contourf(ax=axs[5,1], levels = 21, y='lev', yscale='log',
                                                vmax=5e9, ylim=[1e2,2],
                                                cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[5,2].text(prop_x, prop_y, empty_message, transform=axs[5,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[5,2], y='lev', yscale='log',vmax=5e9,
                                    ylim=[1e2,2],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # utendepfd
        #------------------------------------------------------------------------------------------
        if var == "utendepfd":
            mseasons.plot(ax=axs[6,0], y='lev', yscale='log',
                                            vmax=0.0001, vmin=-0.0001, ylim=[1e2,2],
                                            cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[6,1], y='lev', yscale='log',
                                            vmax=0.0001, vmin=-0.0001, ylim=[1e2,2],
                                            cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[6,2].text(prop_x, prop_y, empty_message, transform=axs[6,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[6,2], y='lev', yscale='log',vmax=0.0001, vmin=-0.0001,
                                    ylim=[1e2,2],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # utendvtem
        #------------------------------------------------------------------------------------------
        if var == "utendvtem":
            mseasons.plot(ax=axs[7,0], y='lev', yscale='log',vmax=0.001, ylim=[1e3,1],
                                            cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[7,1], y='lev', yscale='log',vmax=0.001, ylim=[1e3,1],
                                            cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[7,2].text(prop_x, prop_y, empty_message, transform=axs[7,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[7,2], y='lev', yscale='log', vmax=0.001, ylim=[1e3,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

        # utendwtem
        #------------------------------------------------------------------------------------------
        if var == "utendwtem":
            mseasons.plot(ax=axs[8,0], y='lev', yscale='log',vmax=0.0001, ylim=[1e3,1],
                                            cbar_kwargs={'label': ds[var].units})

            oseasons.plot(ax=axs[8,1], y='lev', yscale='log',vmax=0.0001, ylim=[1e3,1],
                                            cbar_kwargs={'label': ds[var].units})

            #Check if difference plot has contour levels, if not print notification
            if len(dseasons.lev) == 0:
                axs[8,2].text(prop_x, prop_y, empty_message, transform=axs[8,2].transAxes, bbox=props)
            else:
                dseasons.plot(ax=axs[8,2], y='lev', yscale='log', vmax=0.0001, ylim=[1e3,1],cmap="BrBG",
                                    cbar_kwargs={'label': ds[var].units})

    # Set the ticks and ticklabels for all x-axes
    #NOTE: This has to come after all subplots have been done,
    #I am assuming this is because of the way xarray plots info automatically for labels and titles
    #This is to change the default xarray labels for each instance of the xarray plot method
    plt.setp(axs, xticks=np.arange(-80,81,20), xlabel='latitude', title="")

    #Set titles of subplots
    #Set case names in first subplot only
    uzm = ds["uzm"].long_name.replace(" ", "\ ")

    test_yrs = f"{climo_yrs['test'][0]}-{climo_yrs['test'][1]}"
    axs[0,0].set_title(f"\n\n"+"$\mathbf{Test}$"+f"  yrs: {test_yrs}\n"+f"{case_names[0]}\n\n\n",fontsize=14)

    if obs:
        obs_title = Path(vres["obs_name"]).stem
        axs[0,1].set_title(f"\n\n"+"$\mathbf{Baseline}$\n"+f"{obs_title}\n\n"+"$\mathbf{"+uzm+"}$"+"\n",fontsize=14)

    else:
        base_yrs = f"{climo_yrs['base'][0]}-{climo_yrs['base'][1]}"
        axs[0,1].set_title(f"\n\n"+"$\mathbf{Baseline}$"+f"  yrs: {base_yrs}\n"+f"{case_names[1]}\n\n"+"$\mathbf{"+uzm+"}$"+"\n",fontsize=14)
    
    #Set main title for difference plots column
    axs[0,2].set_title("$\mathbf{Test} - \mathbf{Baseline}$"+"\n\n\n",fontsize=14)
    
    #Set variable name on center plot (except first plot, see above)
    for i in range(1,len(var_list)):
        var_name = ds[var_list[i]].long_name.replace(" ", "\ ")
        axs[i,1].set_title("$\mathbf{"+var_name+"}$"+"\n",fontsize=14)
    
    #Adjust subplots
    #May need to adjust hspace and wspace depending on if multi-case diagnostics ever happen for TEM diags
    hspace = 0.4
    plt.subplots_adjust(wspace=0.5, hspace=hspace)

    return axs

##############
#END OF SCRIPT