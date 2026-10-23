"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.ticker as mticker
from matplotlib import patheffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as patheffects
import os, glob, re
from pathlib import Path
import xarray as xr
import plotting_utils as plot_utils




import matplotlib.pyplot as plt
from matplotlib import patheffects
from collections import OrderedDict
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from datetime import datetime
from glob import glob

from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import xarray as xr
from pathlib import Path
import os
import cftime
import matplotlib.ticker as ticker
import re

def model_cloud_state_analysis(model_state_yaml):

    """
    Script to analyze COMPASS model output against reanalysis data.
    Focuses on temperature comparisons at various pressure levels.
    Generates plots comparing model output to ERA5 reanalysis.

    Requirements:
    - xarray
    - matplotlib
    - cartopy
    - numpy
    - pandas

    Adjust file paths and parameters as needed for your specific analysis.
    Good luck, and may all your plots be helpful!
    """

    # Helper Functions
    # ----------------
    #cftime.DatetimeNoLeap(2018, 2, 19, 16)
    import cftime
    from datetime import datetime


    def format_time_str(date, time_hs):
        # Your datetime string
        dt_str = f"{date} {time_hs}"
        
        # Parse to Python datetime first
        py_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        
        # Construct a cftime DatetimeNoLeap
        cf_dt = cftime.DatetimeNoLeap(
            py_dt.year, py_dt.month, py_dt.day,
            py_dt.hour, py_dt.minute, py_dt.second
        )
        
        #print(cf_dt)     # 2018-02-19 16:30:00
        #print(type(cf_dt))
        return cf_dt
    #++++++++++++++++++++++++++++++

    def clean_ds(ds, sdate, stime_hs, edate, etime_hs):
        fincl_lat_str = None
        fincl_lon_str = None
        for key in ds.dims.keys():
            if 'lat_' in key:
                #print(key,"\n",key[4:])
                fincl_lat_str = key[4:]
            if 'lon_' in key:
                #print(key,"\n",key[4:])
                fincl_lon_str = key[4:]
        if fincl_lat_str and fincl_lon_str:
            fincl_latlon_str = f"{fincl_lon_str}_{fincl_lat_str}"
        elif fincl_lat_str and not fincl_lon_str:
            fincl_latlon_str = f"{fincl_lat_str}"
        elif not fincl_lat_str and fincl_lon_str:
            fincl_latlon_str = f"{fincl_lon_str}"
        else:
            fincl_latlon_str = ""
        print(f"boff y'all: {fincl_lon_str}_{fincl_lat_str}")
        print("fincl_latlon_str",fincl_latlon_str)

        # Rename the variables in the second dataset by removing the substring
        new_var_names = {var: var.replace(f"_{fincl_latlon_str}", "") for var in ds.data_vars}

        # Apply the renaming to the dataset
        ds = ds.rename(new_var_names)

        # Rename the variables in the second dataset by removing the substring
        new_dim_names = {var: var.replace(f"_{fincl_lon_str}", "") for var in ds.dims}
        # Apply the renaming to the dataset
        ds = ds.rename(new_dim_names)

        # Rename the variables in the second dataset by removing the substring
        new_dim_names = {var: var.replace(f"_{fincl_lat_str}", "") for var in ds.dims}
        # Apply the renaming to the dataset
        ds = ds.rename(new_dim_names)

        # Define your desired approximate start and end times
        #t_start = cftime.DatetimeNoLeap(2018, 2, 19, 16)   # approximate start
        t_start = format_time_str(sdate, stime_hs)

        #t_end   = cftime.DatetimeNoLeap(2018, 2, 20, 7)  # approximate end
        t_end = format_time_str(edate, etime_hs)

        print("Attempt nearest start:", t_start)
        print("Attempt nearest end:  ", t_end)

        # Find the nearest available timestamps
        t_start_nearest = ds.sel(time=t_start, method="nearest").time.values
        t_end_nearest   = ds.sel(time=t_end, method="nearest").time.values

        print("Nearest start:", t_start_nearest)
        print("Nearest end:  ", t_end_nearest)

        # Slice between those nearest timestamps
        ds = ds.sel(
                                    lon=slice(lon_min, lon_max),
                                    lat=slice(lat_min, lat_max)
        ).sel(time=slice(t_start_nearest, t_end_nearest))

        ds = ds.sortby('time')
        #print("ACTUAL TIMES:", ds.time.values[0],ds.time.values[-1])
        return ds
    #++++++++++++++++++++++++++++++


    def plot_model_vs_reanly(case_name, cam_ds, var_name, target_time, lev, minsies, maxsies, plot_loc):
        case_cam_ds = OrderedDict()
        case_cam_diff_ds = OrderedDict()
        do_it = True

        shrink = 0.625
        pad = 0.02

        cam_lon = cam_ds['lon']
        cam_lat = cam_ds['lat']

        image_dir = Path(plot_loc)
        if not image_dir.is_dir():
            image_dir.mkdir(parents=True)

        time_part = target_time.strftime("%Y_%m_%d_%H:%M")
        #print("Full time:",time_part)
        #lev_part = f"_lev{lev_idx}" if lev_idx is not None else "_nolev"
        lev_r = int(lev)
        filename = f"{var_name}_{time_part}_{lev_r}hPa.png"
        filepath = os.path.join(image_dir, filename)

        if not Path(filepath).is_file():
            print(f"Saved plot: {filepath}")
            do_it = True
        else:
            print(f"Skipped (already exists): {filepath}")
            do_it = False

        if do_it:
            try:
                target_time_idx = list(era5_ds.TimeStamp.values).index(np.datetime64(target_time.strftime()))
            except:
                print("Skipping, must have a time coordinate mismatch. Will try next time step...\n")
                return
            #print(target_time_idx)
            era5_ds_sfc = era5_ds["T"].sel(lev=lev,method='nearest').isel(time=target_time_idx)#.where(
                                        #(era_lon >= extent[0]) & (era_lon <= extent[1]) &
                                        #(era_lat <= extent[2]) & (era_lat >= extent[3]),
                                        #drop=True
                                    #)
            cam_ds_sfc = cam_ds[var_name].sel(lev=lev,method='nearest').sel(time=target_time)#.where(
                                        # (cam_lon >= extent[0]) & (cam_lon <= extent[1]) &
                                            #(cam_lat <= extent[2]) & (cam_lat >= extent[3]),
                                        # drop=True
                                    # )
            #case_cam_ds[f"{case_name}_{var_name}_{lev_r}_{time_part}"] = cam_ds_sfc
            if case_name not in case_cam_ds:
                case_cam_ds[case_name] = {}
            if var_name not in case_cam_ds[case_name]:
                case_cam_ds[case_name][var_name] = {}
            if lev_r not in case_cam_ds[case_name][var_name]:
                case_cam_ds[case_name][var_name][lev_r] = {}
            if time_part not in case_cam_ds[case_name][var_name][lev_r]:
                case_cam_ds[case_name][var_name][lev_r][time_part] = cam_ds_sfc

            diff = era5_ds_sfc - cam_ds_sfc
            era5_ds_sfc.close()
            cam_ds_sfc.close()

            #case_cam_diff_ds[f"{case_name}_{var_name}_{lev_r}_{time_part}"] = diff
            if case_name not in case_cam_diff_ds:
                case_cam_diff_ds[case_name] = {}
            if var_name not in case_cam_diff_ds[case_name]:
                case_cam_diff_ds[case_name][var_name] = {}
            if lev_r not in case_cam_diff_ds[case_name][var_name]:
                case_cam_diff_ds[case_name][var_name][lev_r] = {}
            if time_part not in case_cam_diff_ds[case_name][var_name][lev_r]:
                case_cam_diff_ds[case_name][var_name][lev_r][time_part] = diff

        #print("do_it",do_it,"\n")
        if do_it:
            # === Plotting ===
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})

            axes[0].set_ylabel("Latitude", labelpad=40)
            axes[0].set_yticklabels([])
            axes[0].set_yticks([])

            lev_unit = cam_ds_sfc.lev.units

            levels = np.arange(minsies, maxsies + 1)  # Include the upper bound
            norm = BoundaryNorm(levels, ncolors=plt.cm.coolwarm.N, clip=True)

            c0cm = era5_ds_sfc.plot.pcolormesh(
                    ax=axes[0],
                    transform=ccrs.PlateCarree(),
                    cmap="coolwarm",
                    #vmin=270,
                    #vmax=300,
                    norm=norm,
                    #cbar_kwargs={'label': f"{var_name} ({h0a_ds.attrs.get('units', '')})"},
                    add_colorbar=False,  # We'll add manually
                    **coords
                )
            axes[0].set_title(f"ERA5: {var_name} @ {str(era5_ds_sfc['time'].values)}\nlev={lev_r} {lev_unit}")
            cbar = axes[0].figure.colorbar(c0cm, ax=axes[0], orientation='vertical', shrink=shrink, pad=pad)
            cbar.set_label(f"({era5_ds[var_name].attrs.get('units', '')})")

            mcm = cam_ds_sfc.plot.pcolormesh(
                    ax=axes[1],
                    transform=ccrs.PlateCarree(),
                    cmap="coolwarm",
                    #vmin=270,
                    #vmax=300,
                    norm=norm,
                    #cbar_kwargs={'label': f"{var_name} ({era5_ds.attrs.get('units', '')})"},
                    add_colorbar=False,  # We'll add manually
                    **coords
                )
            axes[1].set_title(f"CESM: {var_name} @ {str(cam_ds_sfc['time'].values)}\nlev={lev_r} {lev_unit}")
                # Custom colorbar matching height of axes
            cbar = axes[1].figure.colorbar(mcm, ax=axes[1], orientation='vertical', shrink=shrink, pad=pad)
            cbar.set_label(f"({cam_ds[var_name].attrs.get('units', '')})")

            c0cm_target = diff.plot.pcolormesh(
                    ax=axes[2],
                    transform=ccrs.PlateCarree(),
                    cmap="coolwarm",
                    vmin=-10,
                    vmax=10,
                    #cbar_kwargs={'label': f"{var_name} ({era5_ds.attrs.get('units', '')})"},
                    add_colorbar=False,  # We'll add manually
                    **coords
                )
            axes[2].set_title(f"ERA5 {var_name} minus CESM {var_name} @ {str(cam_ds_sfc['time'].values)}\nlev={lev_r} {lev_unit}")
                # Custom colorbar matching height of axes
            cbar = axes[2].figure.colorbar(c0cm_target, ax=axes[2], orientation='vertical', shrink=shrink, pad=pad)
            cbar.set_label(f"({cam_ds[var_name].attrs.get('units', '')})")

            # Flatten the 2D array of axes for easy iteration
            #for ax in axes.flat:
            for ax in axes:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND, zorder=0)
                ax.add_feature(cfeature.COASTLINE)
                gl = ax.gridlines(draw_labels=True)
                gl.right_labels = False
                gl.top_labels = False
                """
                nudge_lon_min = h0a_ds.lon.min().values
                nudge_lon_max = h0a_ds.lon.max().values
                nudge_lat_max = h0a_ds.lat.max().values
                nudge_lat_min = h0a_ds.lat.min().values
                box = Rectangle((nudge_lon_min, nudge_lat_min),  # lower-left corner
                                    nudge_lon_max - nudge_lon_min,   # width
                                    nudge_lat_max - nudge_lat_min,   # height
                                    linewidth=2,
                                    edgecolor='black',
                                    facecolor='none',
                                    transform=ccrs.PlateCarree())  # Important: use map projection

                # Add the box to the map
                ax.add_patch(box)
                """

            plt.tight_layout()
            #metadata={"Creator": USER,
            #          #"":,
            #          #"":,
            #          #""
            #         }
            fig.savefig(filepath, dpi=150,
                        #metadata=metadata
                    )
            plt.close(fig)

    #++++++++++++++++++++++++++++++

    import yaml
    #Open YAML file:
    with open(model_state_yaml, encoding='UTF-8') as dfil:
        model_state_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)

    merra_ds = xr.open_dataset('/glade/work/richling/cesm-diagnostics/COMPASS/merra2_12012017-02282018_fixed.nc')
    era5_ds = xr.open_dataset('/glade/work/richling/cesm-diagnostics/COMPASS/era5_12012017-02282018_fixed.nc')

    extent = model_state_defaults["extent"]
    lat_min, lat_max = extent[3], extent[2]
    lon_min, lon_max = extent[0], extent[1]

    era5_ds = era5_ds.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_min, lat_max)
    )

    era_lon = era5_ds["lon"]
    era_lat = era5_ds["lat"]

    # Lat/lon detection
    lat_name = next((dim for dim in era5_ds.dims if 'lat' in dim.lower()), None)
    lon_name = next((dim for dim in era5_ds.dims if 'lon' in dim.lower()), None)

    coords = {}
    if lat_name and lon_name and era5_ds[lat_name].ndim == 2:
        coords['x'] = era5_ds[lon_name]
        coords['y'] = era5_ds[lat_name]

    """
    sdate = "2018-02-19"
    stime_hs = "16:30"
    edate = "2018-02-20"
    etime_hs = "07:00"
    """

    sdate = model_state_defaults["sdate"]
    stime_hs = model_state_defaults["stime_hs"]
    edate = model_state_defaults["edate"]
    etime_hs = model_state_defaults["etime_hs"]

    exp_casenames = model_state_defaults["compass_cases"]
    if isinstance(exp_casenames, dict):
        cases = list(exp_casenames.keys())

    """exp_case_paths = {}
    for case in cases:
        #print(case)
        exp_case_runnames = []
        exp_case_rundirs = []
        #for dirs in matching_dirs:
        for dirs in exp_casenames[case]:
            if case in str(dirs):
                #print("\nhasd",dirs)
                exp_case_runnames.append(Path(dirs).parts[-1])
                exp_case_rundirs.append(Path(dirs) / "atm/hist")
        exp_case_paths[case] = sorted(exp_case_rundirs)"""
    exp_case_paths = {}
    case_names = {}
    case_cam_ds = {}
    #case here refers to main set of sub set of runs
    # ie f.e21.FHIST_BGC.f09_f09_mg17.SOCRATES_nudgeUVTfull is the case and 
    # the runs are the tau12h and tau24h, etc. runs within that case (case_names)
    for case in [cases[0]]:
    #for case in cases:
        print("CASE",case,"\n-----------------------\n")
        exp_case_runnames = []
        exp_case_rundirs = []
        for dirs in exp_casenames[case]:
            if case in str(dirs):
                #print("\nhasd",dirs)
                exp_case_runnames.append(Path(dirs).parts[-1])
                exp_case_rundirs.append(Path(dirs) / "atm/hist")
        exp_case_paths = sorted(exp_case_rundirs)
        case_names[case] = exp_case_runnames
        print("case_names",case_names)
        print("exp_case_paths",exp_case_paths)

        if case not in case_cam_ds:
            case_cam_ds[case] = {}
        for case_run_path in exp_case_paths:
        #for case_run_path in [exp_case_paths[0]]:
            h0i_ds = None
            h0a_ds = None
            h0_ds = None
            #print("case_run_path",case_run_path)

            h0i_files = sorted(case_run_path.glob("*h0i*00.nc"))
            if h0i_files:
                print("GOING WITH h0i THAN\n")
                h0i_ds = xr.open_mfdataset(h0i_files, combine="nested", concat_dim="time")
                cam_ds = h0i_ds
            else:
                print("no h0i files found, eh?\n")

            h0a_files = sorted(case_run_path.glob("*h0a*00.nc"))
            if h0a_files:
                print("GOING WITH h0a THAN\n")
                h0a_ds = xr.open_mfdataset(h0a_files, combine="nested", concat_dim="time")
                cam_ds = h0a_ds
            else:
                print("no h0a files found, eh?\n")

            if not h0a_files and not h0i_files:
                h0_files = sorted(case_run_path.glob("*h0.*00.nc"))
                if h0_files:
                    print("GOING WITH h0 THAN\n")
                    #print("len(h0_files)",len(h0_files))
                    h0_ds = xr.open_mfdataset(h0_files, combine="nested", concat_dim="time")
                    h0_ds = clean_ds(h0_ds, sdate, stime_hs, edate, etime_hs)
                    cam_ds = h0_ds
                else:
                    print("no files found at all, eh?\n")

            print("HERE WE GO YEAH YAY YEHAWW")
            #for case_name in list(case_names.keys()):
            for case_name in list(case_names[case]):
                print("CASE_NAME:",case_name)
                for var in ["T"]:#, "Q"
                    #plot_loc = f"plots/{case_name}/{var}/ERA5/RF13/"
                    plot_loc = f"adf_try_plots_new/{case_name}/{var}/ERA5/RF13/"
                    #for lev in cam_ds.sel(lev=slice(700,1000))['lev'].values:
                    for lev in cam_ds.sel(lev=slice(700,771))['lev'].values:
                        #print("attempted lev",lev)
                        era5_ds_sfc_all_times = era5_ds[var].sel(lev=lev,method='nearest')

                        #print("Check obs vertical lev, is it closesies:",era5_ds_sfc_all_times.lev.values)
                        #print("vs model lev:",lev)

                        cam_ds_sfc_all_times = cam_ds[var].sel(lev=lev,method='nearest')
                        cam_min = cam_ds_sfc_all_times.min()
                        cam_max = cam_ds_sfc_all_times.max()

                        merra_min = era5_ds_sfc_all_times.min()
                        merra_max = era5_ds_sfc_all_times.max()
            
                        minsies = int(np.floor(np.min([merra_min, cam_min])))
                        maxsies = int(np.ceil(np.max([merra_max, cam_max])))
            
                        for time in cam_ds['time'].values:
                            print("TIME:",time)
                            plot_model_vs_reanly(case_name, cam_ds, var, time, lev, minsies, maxsies, plot_loc)

        """#for case_name in [list(case_names.keys())[0]]:
        for case_name in list(case_names.keys()):
            haech = []
            if case_name not in case_cam_ds[case]:
                case_cam_ds[case][case_name] = {}
                if h0i_ds:
                    if "h0i" not in case_cam_ds[case][case_name]:
                        case_cam_ds[case][case_name]["h0i"] = h0i_ds
                    #haech.append(h0i_ds)
                if h0a_ds:
                    if "h0a" not in case_cam_ds[case][case_name]:
                        case_cam_ds[case][case_name]["h0a"] = h0a_ds
                    #haech.append(h0a_ds)
                if h0_ds:
                    if "h0" not in case_cam_ds[case][case_name]:
                        case_cam_ds[case][case_name]["h0"] = h0_ds
                    #haech.append(h0_ds)
                #case_cam_ds[case][case_name] = haech

            print("HERE WE GO YEAH YAY YEHAWW")

            #case_run_path = [exp_case_paths[0]]
            #case_name = case_name[case_name_dict]
            #print("case_name",case_name,"\ncase_run_path",case_run_path,"\n")
            #h0_ds = case_cam_ds[case_name_dict][case_name]["h0"]

            # Loop over time and lev indices
            time_indices = range(len(cam_ds['time'])) if 'time' in cam_ds.dims else [None]
            lev_indices = range(len(cam_ds['lev'])) if 'lev' in cam_ds.dims else [None]
            lev_unit = cam_ds.lev.units

            for var in ["T"]:#, "Q"
                #plot_loc = f"plots/{case_name}/{var}/ERA5/RF13/"
                plot_loc = f"adf_try_plots/{case_name}/{var}/ERA5/RF13/"
                #for lev in cam_ds.sel(lev=slice(700,1000))['lev'].values:
                for lev in cam_ds.sel(lev=slice(700,771))['lev'].values:
                    #print("attempted lev",lev)
                    era5_ds_sfc_all_times = era5_ds[var].sel(lev=lev,method='nearest')

                    #print("Check obs vertical lev, is it closesies:",era5_ds_sfc_all_times.lev.values)
                    #print("vs model lev:",lev)

                    cam_ds_sfc_all_times = cam_ds[var].sel(lev=lev,method='nearest')
                    cam_min = cam_ds_sfc_all_times.min()
                    cam_max = cam_ds_sfc_all_times.max()

                    merra_min = era5_ds_sfc_all_times.min()
                    merra_max = era5_ds_sfc_all_times.max()
        
                    minsies = int(np.floor(np.min([merra_min, cam_min])))
                    maxsies = int(np.ceil(np.max([merra_max, cam_max])))
        
                    for time in cam_ds['time'].values:
                        print("TIME:",time)
                        plot_model_vs_reanly(case_name, cam_ds, var, time, lev, minsies, maxsies, plot_loc)"""







































