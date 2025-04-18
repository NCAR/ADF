#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.
import matplotlib.pyplot as plt
import matplotlib as mpl
import metpy.calc.thermo as thermo
from metpy.units import units

import plotting_functions as pf

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
     - take difference, calculate statistics
     - make plots

    Notes:
     - If any of the TEM cases are missing, the ADF skips this plotting script and moves on.

    """

    #Notify user that script has started:
    msg = "\n  Generating TEM plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

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
    
    #Initialize list of input TEM file locations
    tem_locs = []

    #Extract TEM file save locations
    tem_case_locs = adf.get_cam_info("cam_tem_loc",required=True)
    tem_base_loc = adf.get_baseline_info("cam_tem_loc")

    #If path not specified, skip TEM calculation?
    if tem_case_locs is None:
        print("\t 'cam_tem_loc' not found for test case(s) in config file, so no TEM plots will be generated.")
        return
    else:
        for tem_case_loc in tem_case_locs:
            tem_case_loc = Path(tem_case_loc)
            #Check if TEM directory exists, and if not, then create it:
            if not tem_case_loc.is_dir():
                print(f"    {tem_case_loc} not found, making new directory")
                tem_case_loc.mkdir(parents=True)
            #End if
            tem_locs.append(tem_case_loc)
        #End for

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]
               }

    #Suggestion from Rolando, if QBO is being produced, add utendvtem and utendwtem?
    if "qbo" in adf.plotting_scripts:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM",
                    "PSITEM","UTENDEPFD","UTENDVTEM","UTENDWTEM"]
    else:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","UTENDEPFD"]

    #Baseline TEM location
    input_loc_idx = Path(tem_base_loc)

    #Check if comparing against obs
    if adf.compare_obs:
        obs = True
        #Set TEM file for observations
        base_file_name = 'Obs.TEMdiag.nc'
        input_loc_idx = Path(tem_locs[0])
    else:
        #Set TEM file for baseline
        base_file_name = f'{base_name}.TEMdiag_{syear_baseline}-{eyear_baseline}.nc'
    
    #Set full path for baseline/obs file
    tem_base = input_loc_idx / base_file_name

    #Check to see if baseline/obs TEM file exists    
    if tem_base.is_file():
        ds_base = xr.open_dataset(tem_base)
    else:
        print(f"\t'{base_file_name}' does not exist. TEM plots will be skipped.")
        return

    input_ts_locs = adf.get_cam_info("cam_ts_loc", required=True)

    #Loop over variables:
    for var in var_list:
        #Notify user of variable being plotted:
        print(f"\t - TEM plots for {var}")

        #Loop over model cases:
        for idx,case_name in enumerate(case_names):

            tem_loc = tem_case_locs[idx]

            #Extract start and end year values:
            start_year = syear_cases[idx]
            end_year   = eyear_cases[idx]

            #Open the TEM file
            output_loc_idx = Path(tem_loc)
            case_file_name = f'{case_name}.TEMdiag_{start_year}-{end_year}.nc'
            tem_case = output_loc_idx / case_file_name

            #Grab the data for the TEM netCDF files
            if tem_case.is_file():
                ds = xr.open_dataset(tem_case)
            else:
                print(f"\t'{tem_case}' does not exist. TEM plots will be skipped.")
                return

            #Loop over season dictionary:
            for s in seasons:

                #Location to save plots
                plot_name = plot_location / f"{var}_{s}_WACCM_SeasonalCycle_Mean.png"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    #Add already-existing plot to website (if enabled):
                    adf.debug_log(f"'{plot_name}' exists and clobber is false.")
                    adf.add_website_data(plot_name, var, None, season=s, plot_type="WACCM",ext="SeasonalCycle_Mean",category="TEM",multi_case=True)

                #plot_name = plot_loc / f"CPT_ANN_WACCM_SeasonalCycle_Mean.{plot_type}"
                elif ((redo_plot) and plot_name.is_file()) or (not plot_name.is_file()):
                    if plot_name.is_file():
                        plot_name.unlink()

                #Grab variable defaults for this variable
                vres = res[var]

                #Gather data for both cases
                mdata = ds[var].squeeze()
                odata = ds_base[var].squeeze()

                # APPLY UNITS TRANSFORMATION IF SPECIFIED:
                # NOTE: looks like our climo files don't have all their metadata
                mdata = mdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                # update units
                mdata.attrs['units'] = vres.get("new_unit", mdata.attrs.get('units', 'none'))

                # Do the same for the baseline case if need be:
                if not obs:
                    odata = odata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                    # update units
                    odata.attrs['units'] = vres.get("new_unit", odata.attrs.get('units', 'none'))
                # Or for observations
                else:
                    odata = odata * vres.get("obs_scale_factor",1) + vres.get("obs_add_offset", 0)
                    # Note: we are going to assume that the specification ensures the conversion makes the units the same. Doesn't make sense to add a different unit.

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

                # Derive zonal mean temp from potential temp
                if var == "thzm":
                    path = input_ts_locs[idx]
                    ds_pmid = xr.open_dataset(f"{path}{case_name}.cam.h0.PMID.{start_year}01-{end_year}12.nc")

                    ds_pmid_interp = ds_pmid.interp(lat=mseasons.zalat,method="nearest")
                    pmid = ds_pmid_interp["PMID"]
                    pmid.attrs['units'] = 'Pa'

                    #Create array to avoid weighting missing values:
                    pmid_ones = xr.where(pmid.isnull(), 0.0, 1.0)

                    if s == 'ANN':

                        #Calculate annual weights (i.e. don't group by season):
                        weights_ann = month_length / month_length.sum()

                        pmid = (pmid * weights_ann).sum(dim='time')
                        pmid = pmid / (pmid_ones*weights_ann).sum(dim='time')
                    else:
                        #this is inefficient because we do same calc over and over
                        pmid = (pmid * weights).groupby("time.season").sum(dim="time").sel(season=s)
                        wgt_denom = (pmid_ones*weights).groupby("time.season").sum(dim="time").sel(season=s)
                        pmid = pmid / wgt_denom


                    mseasons.attrs['units'] = "K"
                    oseasons.attrs['units'] = "K"
                    pmid = pmid.mean(dim="lon")

                    mseasons = thermo.temperature_from_potential_temperature(pmid* units.Pa,mseasons* units.kelvin)
                    oseasons = thermo.temperature_from_potential_temperature(pmid* units.Pa,oseasons* units.kelvin)

                if var == "utendepfd":
                    mseasons = mseasons*1000
                    oseasons = oseasons*1000
                #difference: each entry should be (lat, lon)
                dseasons = mseasons-oseasons
                
                #Gather contour plot options
                cp_info = pf.prep_contour_plot(mseasons, oseasons, dseasons, **vres)
                clevs = np.unique(np.array(cp_info['levels1']))
                norm = cp_info['norm1']
                cmap = cp_info['cmap1']
                clevs_diff = np.unique(np.array(cp_info['levelsdiff']))

                # mesh for plots:
                lat = mseasons['zalat']
                lev = mseasons['lev']
                lats, levs = np.meshgrid(lat, lev)

                # Find the next value below highest vertical level
                prev_major_tick = 10 ** (np.floor(np.log10(np.min(levs))))
                prev_major_tick

                # Set padding for colorbar form axis
                cmap_pad = 0.005

                # create figure object
                fig = plt.figure(figsize=(14,10))
                # LAYOUT WITH GRIDSPEC
                # 4 rows, 8 columns, but each map will take up 4 columns and 2 rows
                gs = mpl.gridspec.GridSpec(4, 8, wspace=0.75,hspace=0.5)
                ax1 = plt.subplot(gs[0:2, :4], **cp_info['subplots_opt'])
                ax2 = plt.subplot(gs[0:2, 4:], **cp_info['subplots_opt'])
                ax3 = plt.subplot(gs[2:, 2:6], **cp_info['subplots_opt'])
                ax = [ax1,ax2,ax3]

                #Contour fill
                img0 = ax[0].contourf(lats, levs,mseasons, levels=clevs, norm=norm, cmap=cmap)
                img1 = ax[1].contourf(lats, levs,oseasons, levels=clevs, norm=norm, cmap=cmap)
                    
                #Add contours for highlighting
                c0 = ax[0].contour(lats,levs,mseasons,levels=clevs[::2], norm=norm,
                                    colors="k", linewidths=0.5)

                #Check if contour labels need to be adjusted
                #ie if the values are large and/or in scientific notation, just label the 
                #contours with the leading numbers.
                #EXAMPLE: plot values are 200000; plot the contours as 2.0 and let the colorbar
                #         indicate that it is e5.
                fmt = {}
                if 'contour_adjust' in vres:
                    test_strs = c0.levels/float(vres['contour_adjust'])
                    for l, str0 in zip(c0.levels, test_strs):
                        fmt[l] = str0

                    # Add contour labels
                    plt.clabel(c0, inline=True, fontsize=8, levels=c0.levels, fmt=fmt)
                else:
                    # Add contour labels
                    plt.clabel(c0, inline=True, fontsize=8, levels=c0.levels)

                #Add contours for highlighting
                c1 = ax[1].contour(lats,levs,oseasons,levels=clevs[::2], norm=norm,
                                    colors="k", linewidths=0.5)

                #Check if contour labels need to be adjusted
                #ie if the values are large and/or in scientific notation, just label the 
                #contours with the leading numbers.
                #EXAMPLE: plot values are 200000; plot the contours as 2.0 and let the colorbar
                #         indicate that it is e5.
                fmt = {}
                if 'contour_adjust' in vres:
                    base_strs = c1.levels/float(vres['contour_adjust'])
                    for l, str0 in zip(c1.levels, base_strs):
                        fmt[l] = str0

                    # Add contour labels
                    plt.clabel(c1, inline=True, fontsize=8, levels=c1.levels, fmt=fmt)
                else:
                    # Add contour labels
                    plt.clabel(c1, inline=True, fontsize=8, levels=c1.levels)


                #Check if difference plot has contour levels, if not print notification
                if len(dseasons.lev) == 0:
                    #Set empty message for comparison of cases with different vertical levels
                    #TODO: Work towards getting the vertical and horizontal interpolations!! - JR
                    empty_message = "These have different vertical levels\nCan't compare cases currently"
                    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
                    prop_x = 0.18
                    prop_y = 0.42
                    ax[2].text(prop_x, prop_y, empty_message,
                                    transform=ax[2].transAxes, bbox=props)
                else:
                    img2 = ax[2].contourf(lats, levs, dseasons,
                                            #cmap="BrBG",
                                            cmap=cp_info['cmapdiff'],
                                            levels=clevs_diff,
                                            norm=cp_info['normdiff'])
                    ax[2].contour(lats, levs, dseasons, colors="k", linewidths=0.5,
                                    levels=clevs_diff[::2], norm=cp_info['normdiff'])
                    cp_info['diff_colorbar_opt']["label"] = cp_info['colorbar_opt']["label"]
                    plt.colorbar(img2, ax=ax[2], location='right', pad=cmap_pad,**cp_info['diff_colorbar_opt'])

                #Format y-axis
                for i,a in enumerate(ax[:]):
                    a.set_yscale("log")
                    a.set_xlabel("Latitude")
                    # Only plot y-axis label for test case
                    if i == 0:
                        a.set_ylabel('Pressure [hPa]', va='center', rotation='vertical')
                    if 'ylim' in vres:
                        y_lims = [float(lim) for lim in vres['ylim']]
                        y_lims[-1]=prev_major_tick
                        a.set_ylim(y_lims)
                    else:
                        a.set_ylim(a.get_ylim()[::-1])

                # Format color bars
                plt.colorbar(img1, ax=ax[1], location='right', pad=cmap_pad,**cp_info['colorbar_opt'])
                # Remove the colorbar label for baseline
                cp_info['colorbar_opt'].pop("label", None)
                plt.colorbar(img0, ax=ax[0], location='right', pad=cmap_pad,**cp_info['colorbar_opt'])

                #Variable plot title name
                longname = vres["long_name"]
                plt.suptitle(f'{longname}: {s}', fontsize=20, y=.97)

                test_yrs = f"{start_year}-{end_year}"
                
                plot_title = "$\mathbf{Test}:$"+f"{test_nicknames[idx]}\nyears: {test_yrs}"
                ax[0].set_title(plot_title, loc='left', fontsize=10)

                if obs:
                    obs_title = Path(vres["obs_name"]).stem
                    ax[1].set_title(f"{obs_title}\n",fontsize=10)

                else:
                    base_yrs = f"{syear_baseline}-{eyear_baseline}"
                    plot_title = "$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {base_yrs}"
                    ax[1].set_title(plot_title, loc='left', fontsize=10)
                
                #Set main title for difference plots column
                ax[2].set_title("$\mathbf{Test} - \mathbf{Baseline}$",fontsize=10)

                #Write the figure to provided workspace/file:
                fig.savefig(plot_name, bbox_inches='tight', dpi=300)

                #Add plot to website (if enabled):
                adf.add_website_data(plot_name, var, case_name, season=s, plot_type="WACCM",
                                     ext="SeasonalCycle_Mean",category="TEM")

                plt.close()
    print("  ...TEM plots have been generated successfully.")

# Helper functions
##################