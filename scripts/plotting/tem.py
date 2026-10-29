"""
Module to plot the Transformed Eulerian Mean (TEM) diagnostics.
"""

#Import standard modules:
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotting_utils as plot_utils

#Format warning messages:
import adf_utils as utils
import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

#Constants for the potential temperature -> temperature conversion:
P0 = 1.0e5              # reference pressure, Pa
KAPPA = 287.0 / 1004.0  # R_dry / cp_dry


def _tem_pressure(adf, case_name, season):
    """
    Return the season-averaged zonal mean model pressure from the TEM file.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object, whose data layer reads the TEM files.
    case_name : str or None
        A test case name, or None for the baseline.
    season : str
        'ANN', 'DJF', 'JJA', 'MAM' or 'SON'.

    Returns
    -------
    xarray.DataArray or None
        Pressure in Pa on the TEM grid, or None when the TEM file does not
        carry it, which leaves the caller to fall back.

    Notes
    -----
    create_TEM_files stores the model's own pressure (PMID or PINT, to match
    the TEM grid) as 'PZM', for the same time steps as the TEM fields, so it
    is averaged here exactly as they are.
    """
    if case_name is None:
        pres = adf.data.load_reference_tem_da("PZM", apply_scaling=False)
    else:
        pres = adf.data.load_tem_da(case_name, "PZM", apply_scaling=False)
    #End if
    if pres is None:
        return None
    #End if
    return utils.seasonal_mean(pres, season=season, is_climo=False)


def _to_temperature(adf, case_name, theta, season):
    """
    Convert zonal mean potential temperature to temperature.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object, passed through to _tem_pressure.
    case_name : str or None
        A test case name, or None for the baseline.
    theta : xarray.DataArray
        Zonal mean potential temperature, in K.
    season : str
        'ANN', 'DJF', 'JJA', 'MAM' or 'SON'.

    Returns
    -------
    xarray.DataArray
        Temperature in K.

    Notes
    -----
    Uses the model's own pressure, which the TEM file carries as 'PZM'. A TEM
    file without it (one written by an older ADF, or a case with no PMID/PINT
    time series) is converted with the hybrid vertical coordinate instead and
    a warning says so, since that coordinate is only the true pressure for a
    hybrid-sigma vertical coordinate.
    """
    pres = _tem_pressure(adf, case_name, season)
    lev_name = utils.vertical_dim(theta)
    if pres is None:
        field = utils.pressure_field_name(lev_name)
        label = case_name if case_name else "the baseline"
        warnings.warn(f"\t    WARNING: no model pressure (PZM) in the TEM file "
                      f"for {label}, converting THZM with the hybrid '{lev_name}' "
                      f"coordinate instead. For the exact conversion, have "
                      f"'{field}' time series and remake the TEM files "
                      "('overwrite_tem: true').")
        pres = theta[lev_name] * 100.0
    #End if
    temperature = theta * (pres / P0) ** KAPPA
    temperature.attrs['units'] = "K"
    return temperature


def _labelled_contours(axis, xmesh, ymesh, data, levels, norm, vres):
    """
    Draw labelled black contours over a filled contour plot.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes to draw on.
    xmesh, ymesh : numpy.ndarray
        Coordinate meshes matching `data`.
    data : xarray.DataArray
        Field being contoured.
    levels : array-like
        Contour levels to draw.
    norm : matplotlib.colors.Normalize
        Color normalization, shared with the filled contours.
    vres : dict
        Variable defaults, read for the optional 'contour_adjust' entry.

    Returns
    -------
    None
        Does not return a value, draws on `axis`.

    Notes
    -----
    A 'contour_adjust' entry divides the labels by a common factor, so a field
    plotted in units of 1e7 is labelled 2.0 rather than 20000000.0 and the
    colorbar carries the exponent.
    """
    contours = axis.contour(xmesh, ymesh, data, levels=levels, norm=norm,
                            colors="k", linewidths=0.5)
    fmt = None
    if 'contour_adjust' in vres:
        adjust = float(vres['contour_adjust'])
        fmt = {level: level / adjust for level in contours.levels}
    #End if
    plt.clabel(contours, inline=True, fontsize=8, levels=contours.levels, fmt=fmt)


def tem(adf):
    """
    Plot the TEM diagnostics as latitude versus pressure panels.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    None
        Does not return a value, produces files.

    Notes
    -----
    Directly uses adf for the following:
    get_cam_info, get_baseline_info, get_basic_info, read_config_var,
    climo_yrs, case_nicknames, compare_obs, variable_defaults,
    plotting_scripts, plot_location, add_website_data, debug_log, data

    Reads the TEM files written by create_TEM_files from 'cam_tem_loc', through
    data.get_tem_file, data.load_tem_da and their reference versions. Each
    figure has three panels: the test case, the baseline or observations, and
    their difference. The difference is left as a message when the two are on
    different vertical grids, which is normal against observations, since ERA5
    has 37 pressure levels where WACCM has far more.

    THZM is stored as potential temperature and plotted as temperature. The
    conversion uses the model's own pressure (PMID or PINT, depending on the
    grid), which the TEM file carries as 'PZM', and falls back to the hybrid
    vertical coordinate with a warning when the file does not have it.

    If a TEM file is missing the whole script is skipped; a variable missing
    from a file that is present is reported and skipped on its own.
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
    
    #Without a TEM location there are no TEM files to plot:
    if adf.get_cam_info("cam_tem_loc", required=True) is None:
        print("\t 'cam_tem_loc' not found for test case(s) in config file, so no TEM plots will be generated.")
        return
    #End if

    #Suggestion from Rolando, if QBO is being produced, add utendvtem and utendwtem?
    if "qbo" in adf.plotting_scripts:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM",
                    "PSITEM","UTENDEPFD","UTENDVTEM","UTENDWTEM"]
    else:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","UTENDEPFD"]

    #Check that every TEM file is there before any plotting work is done; the
    #data layer finds them (in 'cam_tem_loc', named for the climatology years):
    if not adf.data.get_ref_tem_file():
        print(f"\t TEM file for '{base_name}' does not exist. TEM plots will be skipped.")
        return
    #End if
    for case_name in case_names:
        if not adf.data.get_tem_file(case_name):
            print(f"\t TEM file for '{case_name}' does not exist. TEM plots will be skipped.")
            return
        #End if
    #End for

    #Loop over variables:
    for var in var_list:
        #Notify user of variable being plotted:
        print(f"\t - TEM plots for {var}")

        #Read through the data layer, which applies the unit conversion from
        #the variable defaults without scaling a TEM file already holding
        #converted values a second time. Observations have their own scaling,
        #which is assumed to bring them to the same units, so they keep the
        #unit string they arrived with.
        odata = adf.data.load_reference_tem_da(var)
        if odata is None:
            continue
        #End if

        #Loop over model cases:
        for idx,case_name in enumerate(case_names):

            plot_location = plot_locations[idx]

            mdata = adf.data.load_tem_da(case_name, var)
            if mdata is None:
                continue
            #End if

            #Extract start and end year values:
            start_year = syear_cases[idx]
            end_year   = eyear_cases[idx]

            #Loop over the seasons the ADF knows about:
            for s in utils.seasons:

                #Location to save plots
                plot_name = plot_location / f"{var}_{s}_WACCM_SeasonalCycle_Mean.png"

                # Check redo_plot. If set to True: remove old plot, if it already exists:
                if (not redo_plot) and plot_name.is_file():
                    #Add already-existing plot to website (if enabled):
                    adf.debug_log(f"'{plot_name}' exists and clobber is false.")
                    adf.add_website_data(plot_name, var, case_name, season=s, plot_type="WACCM",
                                         ext="SeasonalCycle_Mean", category="TEM")
                    continue
                elif redo_plot and plot_name.is_file():
                    plot_name.unlink()
                #End if

                #Grab variable defaults for this variable
                vres = res[var]

                #Month-length weighted seasonal (or annual) mean. The weighted
                #mean renormalizes around missing values, so gaps in either
                #record are handled without extra bookkeeping.
                mseasons = utils.seasonal_mean(mdata, season=s, is_climo=False)
                oseasons = utils.seasonal_mean(odata, season=s, is_climo=False)

                # Derive zonal mean temperature from potential temperature.
                if var == "THZM":
                    #Each side is converted with its own pressure; the test case
                    #and the baseline need not share a vertical coordinate.
                    mseasons = _to_temperature(adf, case_name, mseasons, s)
                    if not obs:
                        oseasons = _to_temperature(adf, None, oseasons, s)
                    #End if
                #End if

                #The two cases can be differenced only if they are on the same
                #kind of vertical grid and share at least two levels. xarray
                #aligns them on their common levels, and equal level counts do
                #not imply equal levels: ERA5 has 37 pressure levels where
                #WACCM has 71, and CAM may write the zonal mean stream on
                #midpoints for one case and interfaces for another.
                lev_m = utils.vertical_dim(mseasons)
                lev_b = utils.vertical_dim(oseasons)
                comparable = lev_m == lev_b
                dseasons = mseasons-oseasons if comparable else None
                if dseasons is None or dseasons[lev_m].size < 2:
                    comparable = False
                    dseasons = xr.zeros_like(mseasons)
                #End if

                #percent change, following the convention in plotting_functions
                pseasons = (dseasons / np.abs(oseasons) * 100.0 if comparable
                            else dseasons)
                pseasons = pseasons.where(np.isfinite(pseasons), np.nan).fillna(0.0)

                #Gather contour plot options
                cp_info = plot_utils.prep_contour_plot(mseasons, oseasons, dseasons,
                                                       pseasons, adfobj=adf, **vres)
                clevs = np.unique(np.array(cp_info['levels_sim']))
                norm = cp_info['norm_sim']
                cmap = cp_info['cmap_sim']
                clevs_diff = np.unique(np.array(cp_info['levels_diff']))

                # mesh for plots -- each panel gets its own, because the baseline
                # (especially observations) can be on a different vertical grid
                # than the test case, and the difference is on their intersection.
                def _mesh(da):
                    return np.meshgrid(da['zalat'], da[utils.vertical_dim(da)])
                lats, levs = _mesh(mseasons)
                lats_b, levs_b = _mesh(oseasons)

                # Find the next value below highest vertical level
                prev_major_tick = 10 ** (np.floor(np.log10(np.min(levs))))

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

                #Filled contours, then black highlight contours with labels
                img0 = ax[0].contourf(lats, levs, mseasons, levels=clevs, norm=norm, cmap=cmap)
                img1 = ax[1].contourf(lats_b, levs_b, oseasons, levels=clevs, norm=norm, cmap=cmap)
                _labelled_contours(ax[0], lats, levs, mseasons, clevs[::2], norm, vres)
                _labelled_contours(ax[1], lats_b, levs_b, oseasons, clevs[::2], norm, vres)

                #Nothing to draw when the two cases could not be differenced:
                if not comparable:
                    #Set empty message for comparison of cases with different vertical levels
                    #TODO: Work towards getting the vertical and horizontal interpolations!! - JR
                    empty_message = "These have different vertical levels\nCan't compare cases currently"
                    props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.9}
                    prop_x = 0.18
                    prop_y = 0.42
                    ax[2].text(prop_x, prop_y, empty_message,
                                    transform=ax[2].transAxes, bbox=props)
                else:
                    lats_d, levs_d = _mesh(dseasons)
                    img2 = ax[2].contourf(lats_d, levs_d, dseasons,
                                            #cmap="BrBG",
                                            cmap=cp_info['cmap_diff'],
                                            levels=clevs_diff,
                                            norm=cp_info['norm_diff'])
                    ax[2].contour(lats_d, levs_d, dseasons, colors="k", linewidths=0.5,
                                    levels=clevs_diff[::2], norm=cp_info['norm_diff'])
                    #prep_contour_plot hands the colorbar labels back on their own:
                    plt.colorbar(img2, ax=ax[2], location='right', pad=cmap_pad,
                                 label=cp_info['diff_units_label'],
                                 **cp_info['diff_colorbar_opt'])

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

                # Format color bars; the units label goes on the baseline's
                # colorbar only, as the test case shares its scale
                plt.colorbar(img1, ax=ax[1], location='right', pad=cmap_pad,
                             label=cp_info['units_label'], **cp_info['colorbar_opt'])
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
                
                #Set main title for difference plots column
                ax[2].set_title(r"$\mathbf{Test} - \mathbf{Baseline}$",fontsize=10)

                #Write the figure to provided workspace/file:
                fig.savefig(plot_name, bbox_inches='tight', dpi=300)

                #Add plot to website (if enabled):
                adf.add_website_data(plot_name, var, case_name, season=s, plot_type="WACCM",
                                     ext="SeasonalCycle_Mean",category="TEM")

                plt.close()
    print("  ...TEM plots have been generated successfully.")

# Helper functions
##################
