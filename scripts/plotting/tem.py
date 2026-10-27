"""
Module to plot the Transformed Eulerian Mean (TEM) diagnostics.
"""

#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import warnings  # use to warn user about missing files.
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotting_utils as plot_utils

#Format warning messages:
import adf_utils as utils
warnings.formatwarning = utils.my_formatwarning

#Constants for the potential temperature -> temperature conversion:
P0 = 1.0e5              # reference pressure, Pa
KAPPA = 287.0 / 1004.0  # R_dry / cp_dry

#TEM variables to plot. TZM is derived here from THZM (and PMID):
TEM_PLOT_VARS = ["UZM","THZM","TZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","DELF"]

def tem(adf):
    """
    Plot the contents of the TEM diagnostic output as 2-D latitude vs pressure maps.

    Steps:
     - loop through TEM variables
     - calculate seasonal/annual fields (from individual months)
     - take difference (interpolating onto the baseline grid if need be)
     - make plots

    Notes:
     - If any of the TEM files are missing, the ADF skips this plotting script and moves on.
     - THZM is plotted as potential temperature; TZM is derived from it using
       each case's own zonal mean PMID, which create_TEM_files copies into the
       TEM file. Without PMID the hybrid 'lev' coordinate is used instead.
     - When the test case and baseline/obs are on different vertical (or
       latitude) grids, the test case is interpolated onto the baseline grid
       (linearly in log-pressure) for the difference panel, over the range
       the two have in common.
    """

    #Notify user that script has started:
    msg = "\n  Generating TEM plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    plot_locations = [Path(loc) for loc in adf.plot_location]

    #Check if plot output directories exist, and if not, then create them:
    for plot_loc in plot_locations:
        if not plot_loc.is_dir():
            print(f"    {plot_loc} not found, making new directory")
            plot_loc.mkdir(parents=True)
        #End if
    #End for

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
    #print(f"\t NOTE: Plot type is set to {plot_type}")

    # check if existing plots need to be redone
    redo_plot = adf.get_basic_info('redo_plot')
    #print(f"\t NOTE: redo_plot is set to {redo_plot}")
    #-----------------------------------------

    #Extract TEM file save locations
    tem_case_locs = adf.get_cam_info("cam_tem_loc")
    tem_base_loc = adf.get_baseline_info("cam_tem_loc")

    #If path not specified, skip TEM plots:
    if tem_case_locs is None:
        print("\t 'cam_tem_loc' not found for test case(s) in config file, so no TEM plots will be generated.")
        return
    #End if

    #Check if comparing against obs
    if obs:
        #Observation TEM file is saved to the first test case location
        base_file = Path(tem_case_locs[0]) / 'Obs.TEMdiag.nc'
    else:
        if tem_base_loc is None:
            print("\t 'cam_tem_loc' not found for baseline in config file, so no TEM plots will be generated.")
            return
        #End if
        base_file = Path(tem_base_loc) / f'{base_name}.TEMdiag_{syear_baseline}-{eyear_baseline}.nc'
    #End if

    #Check to see if baseline/obs TEM file exists
    if not base_file.is_file():
        print(f"\t'{base_file}' does not exist. TEM plots will be skipped.")
        return
    #End if
    ds_base = _open_tem_file(base_file)

    #Open each test case's TEM file once, up front, so a missing file is
    #reported before any plotting work is done:
    case_datasets = []
    for idx, case_name in enumerate(case_names):
        tem_case = Path(tem_case_locs[idx]) / \
                   f'{case_name}.TEMdiag_{syear_cases[idx]}-{eyear_cases[idx]}.nc'
        if not tem_case.is_file():
            print(f"\t'{tem_case}' does not exist. TEM plots will be skipped.")
            return
        #End if
        case_datasets.append(_open_tem_file(tem_case))
    #End for

    #Loop over variables:
    for var in TEM_PLOT_VARS:

        #TZM is derived from THZM, so that is the field that has to be in the files:
        file_var = "THZM" if var == "TZM" else var

        if file_var not in ds_base:
            if obs and file_var == "THZM":
                print(f"\t - Observations have no potential temperature, so {var} will be skipped.")
            else:
                warnings.warn(f"\t    INFO: '{file_var}' not in {base_name} TEM file, so {var} will be skipped.")
            #End if
            continue
        #End if

        if var not in res:
            warnings.warn(f"\t    INFO: '{var}' not in variable defaults, skipping.")
            continue
        #End if

        #Notify user of variable being plotted:
        print(f"\t - TEM plots for {var}")

        #Grab variable defaults for this variable
        vres = res[var]

        #Loop over model cases:
        for idx, case_name in enumerate(case_names):

            ds = case_datasets[idx]
            plot_location = plot_locations[idx]

            if file_var not in ds:
                warnings.warn(f"\t    INFO: '{file_var}' not in {case_name} TEM file, skipping.")
                continue
            #End if

            #Extract start and end year values:
            start_year = syear_cases[idx]
            end_year   = eyear_cases[idx]

            #Loop over the seasons the ADF knows about:
            for s in utils.seasons:

                #Location to save plots
                plot_name = plot_location / f"{var}_{s}_WACCM_TEM_Mean.{plot_type}"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    #Add already-existing plot to website (if enabled):
                    adf.debug_log(f"'{plot_name}' exists and clobber is false.")
                    adf.add_website_data(plot_name, var, case_name, season=s, plot_type="WACCM",
                                         ext="TEM_Mean", category="TEM")
                    continue
                elif redo_plot and plot_name.is_file():
                    plot_name.unlink()
                #End if

                #Gather data for both cases
                mdata = ds[file_var].squeeze()
                odata = ds_base[file_var].squeeze()

                # Apply the unit conversion from the variable defaults. This is
                # done directly rather than through adf.data.apply_conversion,
                # which only converts variables in 'diag_var_list' (the TEM
                # variables are not). The TEM files are always written
                # unconverted by create_TEM_files, so there is no risk of
                # scaling twice. Observations have their own scaling, which is
                # assumed to bring them to the same units.
                if var != "TZM":
                    mdata = _convert_units(mdata, vres)
                    if obs:
                        odata = _convert_units(odata, vres, prefix="obs_")
                    else:
                        odata = _convert_units(odata, vres)
                    #End if
                #End if

                #Month-length weighted seasonal (or annual) mean. The weighted
                #mean renormalizes around missing values.
                mseasons = utils.seasonal_mean(mdata, season=s, is_climo=False)
                oseasons = utils.seasonal_mean(odata, season=s, is_climo=False)

                # Derive zonal mean temperature from potential temperature,
                # each side with its own pressure:
                if var == "TZM":
                    if s == "ANN":
                        print("\t       INFO: deriving zonal mean temperature from potential temperature")
                    #End if
                    mseasons = _to_temperature(ds, mseasons, s, case_name)
                    oseasons = _to_temperature(ds_base, oseasons, s, base_name)
                #End if

                #Difference, on the baseline grid:
                dseasons = _difference(mseasons, oseasons)
                comparable = dseasons is not None
                if not comparable:
                    dseasons = xr.zeros_like(oseasons)
                #End if

                #Gather contour plot options (NaNs outside the common range of
                #the two grids are left out of the automatic levels):
                cp_info = plot_utils.prep_contour_plot(mseasons, oseasons,
                                                       dseasons.fillna(0.0), None,
                                                       adfobj=adf, **vres)
                clevs = np.unique(np.array(cp_info['levels_sim']))
                norm = cp_info['norm_sim']
                cmap = cp_info['cmap_sim']
                clevs_diff = np.unique(np.array(cp_info['levels_diff']))

                # mesh for plots -- each panel gets its own, because the baseline
                # (especially observations) can be on a different grid than the
                # test case. The difference is on the baseline grid.
                mlats, mlevs = _mesh(mseasons)
                olats, olevs = _mesh(oseasons)

                # Find the next value below highest vertical level
                prev_major_tick = min(10 ** (np.floor(np.log10(np.min(mlevs)))),
                                      10 ** (np.floor(np.log10(np.min(olevs)))))

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

                #Contour fill, then black highlight contours with labels
                img0 = ax[0].contourf(mlats, mlevs, mseasons, levels=clevs, norm=norm, cmap=cmap)
                img1 = ax[1].contourf(olats, olevs, oseasons, levels=clevs, norm=norm, cmap=cmap)
                _labelled_contours(ax[0], mlats, mlevs, mseasons, clevs[::2], norm, vres)
                _labelled_contours(ax[1], olats, olevs, oseasons, clevs[::2], norm, vres)

                #Check if difference plot can be made, if not print notification
                if not comparable:
                    #Set empty message for comparison of cases with no common vertical levels
                    empty_message = "These have no vertical levels in common\nCan't compare cases currently"
                    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
                    prop_x = 0.18
                    prop_y = 0.42
                    ax[2].text(prop_x, prop_y, empty_message,
                                    transform=ax[2].transAxes, bbox=props)
                else:
                    img2 = ax[2].contourf(olats, olevs, dseasons,
                                            cmap=cp_info['cmap_diff'],
                                            levels=clevs_diff,
                                            norm=cp_info['norm_diff'])
                    ax[2].contour(olats, olevs, dseasons, colors="k", linewidths=0.5,
                                    levels=clevs_diff[::2], norm=cp_info['norm_diff'])
                    #prep_contour_plot hands the colorbar labels back separately:
                    plt.colorbar(img2, ax=ax[2], location='right', pad=cmap_pad,
                                 label=cp_info['diff_units_label'], **cp_info['diff_colorbar_opt'])
                #End if

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
                    #End if
                #End for

                # Format color bars
                plt.colorbar(img1, ax=ax[1], location='right', pad=cmap_pad,
                             label=cp_info['units_label'], **cp_info['colorbar_opt'])
                # No colorbar label for the test case
                plt.colorbar(img0, ax=ax[0], location='right', pad=cmap_pad,**cp_info['colorbar_opt'])

                #Variable plot title name
                longname = vres["long_name"]
                plt.suptitle(f'{longname}: {s}', fontsize=20, y=.97)

                test_yrs = f"{start_year}-{end_year}"
                plot_title = r"$\mathbf{Test}:$"+f"{test_nicknames[idx]}\nyears: {test_yrs}"
                ax[0].set_title(plot_title, loc='left', fontsize=10)

                if obs:
                    obs_title = Path(vres["obs_name"]).stem
                    ax[1].set_title(f"{obs_title}\n",fontsize=10)
                else:
                    base_yrs = f"{syear_baseline}-{eyear_baseline}"
                    plot_title = r"$\mathbf{Baseline}:$"+f"{base_nickname}\nyears: {base_yrs}"
                    ax[1].set_title(plot_title, loc='left', fontsize=10)
                #End if

                #Set main title for difference plots column
                ax[2].set_title(r"$\mathbf{Test} - \mathbf{Baseline}$",fontsize=10)

                #Write the figure to provided workspace/file:
                fig.savefig(plot_name, bbox_inches='tight', dpi=300)

                #Add plot to website (if enabled):
                adf.add_website_data(plot_name, var, case_name, season=s, plot_type="WACCM",
                                     ext="TEM_Mean", category="TEM")

                plt.close()
            #End for (seasons)
        #End for (cases)
    #End for (variables)
    print("  ...TEM plots have been generated successfully.")

# Helper functions
##################

def _open_tem_file(path):
    """
    Open a TEM file, with interval-midpoint time stamps and the zonal mean
    latitude called 'zalat' (newer CESM output calls it 'zmlat').
    """
    ds = xr.open_dataset(path)
    ds = utils.use_time_bounds_midpoint(ds)
    if 'zmlat' in ds.dims or 'zmlat' in ds.variables:
        if 'zalat' in ds.dims:
            ds = ds.drop_vars('zmlat')
        else:
            ds = ds.rename({'zmlat': 'zalat'})
        #End if
    #End if
    return ds

#########

def _convert_units(data, vres, prefix=""):
    """
    Apply the variable defaults' '<prefix>scale_factor' and '<prefix>add_offset'
    to `data`, and set its units to 'new_unit' if given.
    """
    attrs = data.attrs
    data = data * float(vres.get(f"{prefix}scale_factor", 1)) \
           + float(vres.get(f"{prefix}add_offset", 0))
    data.attrs = attrs
    if "new_unit" in vres:
        data.attrs['units'] = vres["new_unit"]
    #End if
    return data

#########

def _mesh(da):
    """Return the (latitude, level) meshes for plotting `da`."""
    return np.meshgrid(da['zalat'], da[utils.vertical_dim(da)])

#########

def _to_temperature(ds, theta, season, label):
    """
    Convert the seasonal mean potential temperature `theta` to temperature,
    using the seasonal mean zonal mean PMID from the TEM dataset `ds`.

    Falls back to the hybrid vertical coordinate (with a warning) when PMID is
    not available, since that coordinate is only approximately pressure.
    """
    lev_name = utils.vertical_dim(theta)
    pres = None
    if "PMID" in ds and lev_name == "lev":
        pmid = ds["PMID"].squeeze()
        #CAM writes PMID in Pa:
        if str(pmid.attrs.get("units", "Pa")).lower() in ("hpa", "mb", "millibars"):
            pmid = pmid * 100.0
        #End if
        pres = utils.seasonal_mean(pmid, season=season, is_climo=False)
        #Only use it if it is on the same grid as theta:
        if (pres.sizes.get(lev_name) != theta.sizes[lev_name]
                or pres.sizes.get("zalat") != theta.sizes["zalat"]):
            pres = None
        #End if
    #End if
    if pres is None:
        if season == "ANN":
            warnings.warn(f"\t    WARNING: no usable PMID found for {label}, converting "
                          f"THZM with the hybrid '{lev_name}' coordinate instead.")
        #End if
        pres = theta[lev_name] * 100.0
    #End if
    temperature = theta * (pres / P0) ** KAPPA
    temperature.attrs['units'] = "K"
    return temperature

#########

def _difference(mseasons, oseasons):
    """
    Return test minus baseline on the baseline grid, or None if the two have
    fewer than two vertical levels in common.

    If the grids differ, the test case is interpolated onto the baseline
    latitudes (linearly) and levels (linearly in log-pressure). Points outside
    the range the two grids share are left as NaN rather than extrapolated.

    NOTE: the TEM fields are calculated with the model 'lev' values taken as
    pressure (see create_TEM_files.calc_tem), so the same is assumed here.
    This is accurate in the stratosphere, but only approximate near the
    surface where the hybrid levels follow the terrain.
    """
    lev_m = utils.vertical_dim(mseasons)
    lev_b = utils.vertical_dim(oseasons)
    if lev_m is None or lev_b is None:
        return None
    #End if

    same_grid = (lev_m == lev_b
                 and np.array_equal(mseasons[lev_m].values, oseasons[lev_b].values)
                 and np.array_equal(mseasons['zalat'].values, oseasons['zalat'].values))
    if same_grid:
        return mseasons - oseasons
    #End if

    #Interpolate in log-pressure, so make the vertical coordinate log(p):
    mlog = mseasons.assign_coords({lev_m: np.log(mseasons[lev_m].values)})
    if lev_m != lev_b:
        mlog = mlog.rename({lev_m: lev_b})
    #End if
    target_lev = np.log(oseasons[lev_b].values)
    minterp = mlog.interp({lev_b: target_lev, 'zalat': oseasons['zalat'].values},
                          method="linear")
    minterp = minterp.assign_coords({lev_b: oseasons[lev_b].values})

    #Need at least two levels with data to plot a difference:
    if int(minterp.notnull().any(dim='zalat').sum()) < 2:
        return None
    #End if
    return minterp - oseasons

#########

def _labelled_contours(axis, xmesh, ymesh, data, levels, norm, vres):
    """
    Draw labelled black contours over a filled contour plot.

    A 'contour_adjust' entry in the variable defaults divides the labels by a
    common factor, so a field plotted in units of 1e7 is labelled 2.0 rather
    than 20000000.0 and the colorbar carries the exponent.
    """
    contours = axis.contour(xmesh, ymesh, data, levels=levels, norm=norm,
                            colors="k", linewidths=0.5)
    fmt = None
    if 'contour_adjust' in vres:
        adjust = float(vres['contour_adjust'])
        fmt = {level: level / adjust for level in contours.levels}
    #End if
    plt.clabel(contours, inline=True, fontsize=8, levels=contours.levels, fmt=fmt)
