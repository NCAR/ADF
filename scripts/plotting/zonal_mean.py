from pathlib import Path
import numpy as np
import xarray as xr
import plotting_functions as pf
import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

warnings.formatwarning = my_formatwarning

def zonal_mean(adfobj):

    """
    This script plots zonal averages.
    Compare CAM climatologies against
    other climatological data (observations or baseline runs).

    Description of needed inputs from ADF:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
    var_list         -> List of CAM output variables provided by "diag_var_list"
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".
    Notes:
        The script produces plots of 2-D and 3-D variables,
        but needs to determine which type along the way.
        For 3-D variables, the default behavior is to interpolate
        climo files to pressure levels, which requires the hybrid-sigma
        coefficients and surface pressure. That ASSUMES that the climo
        files are using native hybrid-sigma levels rather than being
        transformed to pressure levels.
    """

    #Notify user that script has started:
    print("  Generating zonal mean plots...")

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    var_list = adfobj.diag_var_list
    model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

    #Special ADF variable which contains the output paths for
    #all generated plots and tables:
    plot_locations = adfobj.plot_location

    #CAM simulation variables (this is always assumed to be a list):
    case_names = adfobj.get_cam_info("cam_case_name", required=True)

    # CAUTION:
    # "data" here refers to either obs or a baseline simulation,
    # Until those are both treated the same (via intake-esm or similar)
    # we will do a simple check and switch options as needed:
    if adfobj.get_basic_info("compare_obs"):

        #Extract variable-obs dictionary:
        var_obs_dict = adfobj.var_obs_dict

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("No observations found to plot against, so no zonal-mean maps will be generated.")
            return

    else:
        data_name = adfobj.get_baseline_info("cam_case_name", required=True) # does not get used, is just here as a placemarker
        data_list = [data_name] # gets used as just the name to search for climo files HAS TO BE LIST
        data_loc = adfobj.get_baseline_info("cam_climo_loc", required=True)

    res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"NOTE: Plot type is set to {plot_type}")

    #-----------------------------------------

    #Set data path variables:
    #-----------------------
    mclimo_rg_loc = Path(model_rgrid_loc)
    if not adfobj.compare_obs:
        dclimo_loc  = Path(data_loc)
    #-----------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Loop over variables:
    for var in var_list:

        if adfobj.compare_obs:
            #Check if obs exist for the variable:
            if var in var_obs_dict:
                #Note: In the future these may all be lists, but for
                #now just convert the target_list.
                #Extract target file:
                dclimo_loc = var_obs_dict[var]["obs_file"]
                #Extract target list (eventually will be a list, for now need to convert):
                data_list = [var_obs_dict[var]["obs_name"]]
                #Extract target variable name:
                data_var = var_obs_dict[var]["obs_var"]
            else:
                dmsg = f"No obs found for variable `{var}`, zonal mean plotting skipped."
                adfobj.debug_log(dmsg)
                continue
            #End if
        else:
            #Set "data_var" for consistent use below:
            data_var = var
        #End if

        #Notify user of variable being plotted:
        print("\t - zonal mean plots for {}".format(var))

        # Check res for any variable specific options that need to be used BEFORE going to the plot:
        if var in res:
            vres = res[var]
            #If found then notify user, assuming debug log is enabled:
            adfobj.debug_log(f"zonal_mean: Found variable defaults for {var}")

        else:
            vres = {}
        #End if

        #loop over different data sets to plot model against:
        for data_src in data_list:
            # load data (observational) comparison files
            # (we should explore intake as an alternative to having this kind of repeated code):
            if adfobj.compare_obs:
                #For now, only grab one file (but convert to list for use below)
                oclim_fils = [dclimo_loc]
            else:
                oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))
            #End if
            oclim_ds = _load_dataset(oclim_fils)

            #Loop over model cases:
            for case_idx, case_name in enumerate(case_names):

                #Set output plot location:
                plot_loc = Path(plot_locations[case_idx])

                #Check if plot output directory exists, and if not, then create it:
                if not plot_loc.is_dir():
                    print("    {} not found, making new directory".format(plot_loc))
                    plot_loc.mkdir(parents=True)

                # load re-gridded model files:
                mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))
                mclim_ds = _load_dataset(mclim_fils)

                # stop if data is invalid:
                if (oclim_ds is None) or (mclim_ds is None):
                    warnings.warn(f"invalid data, skipping zonal mean plot of {var}")
                    continue

                #Extract variable of interest
                odata = oclim_ds[data_var].squeeze()  # squeeze in case of degenerate dimensions
                mdata = mclim_ds[var].squeeze()

                # APPLY UNITS TRANSFORMATION IF SPECIFIED:
                # NOTE: looks like our climo files don't have all their metadata
                mdata = mdata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                # update units
                mdata.attrs['units'] = vres.get("new_unit", mdata.attrs.get('units', 'none'))

                # Do the same for the baseline case if need be:
                if not adfobj.compare_obs:
                    odata = odata * vres.get("scale_factor",1) + vres.get("add_offset", 0)
                    # update units
                    odata.attrs['units'] = vres.get("new_unit", odata.attrs.get('units', 'none'))
                # Or for observations
                else:
                    odata = odata * vres.get("obs_scale_factor",1) + vres.get("obs_add_offset", 0)
                    # Note: we are going to assume that the specification ensures the conversion makes the units the same. Doesn't make sense to add a different unit.

                # determine whether it's 2D or 3D
                # 3D triggers search for surface pressure
                has_lat, has_lev = pf.zm_validate_dims(mdata)  # assumes will work for both mdata & odata
                if has_lev:

                    #For now, there is no easy way to use observations with specified pressure levels, so
                    #bail out here:
                    if adfobj.compare_obs:
                        print(f"\t - plot_press_levels currently doesn't work with observations, so skipping case for variabe '{var}'.")
                        continue #Skip variable
                    #End if

                    print("{} has lev dimension.".format(var))
                    # need hyam, hybm, PS, P0 for both datasets
                    if 'hyam' not in mclim_ds:
                        print("!! PROBLEM -- NO hyam, skipping to next case/obs data")
                        print(mclim_ds)
                        continue
                    mhya = mclim_ds['hyam']
                    mhyb = mclim_ds['hybm']
                    if 'time' in mhya.dims:
                        mhya = mhya.isel(time=0).squeeze()
                    if 'time' in mhyb.dims:
                        mhyb = mhyb.isel(time=0).squeeze()
                    if 'P0' in mclim_ds:
                        P0 = mclim_ds['P0']
                    else:
                        P0 = 100000.0  # Pa
                    if 'PS' in mclim_ds:
                        mps = mclim_ds['PS']
                    else:
                        # look for the file (this isn't great b/c we'd have to constantly re-load)
                        mps_files = sorted(list(mclimo_rg_loc.glob("{}_{}_PS_*.nc".format(data_src, case_name))))
                        if len(mps_files) > 0:
                            mps_ds = _load_dataset(mps_files)
                            mps = mps_ds['PS']
                        else:
                            continue  # what else could we do?

                    # We need a way to check whether 'obs' or 'baseline' here:
                    # (and/or a way to specify pressure levels or hybrid levels)
                    ohya = oclim_ds['hyam']
                    ohyb = oclim_ds['hybm']
                    if 'time' in ohya.dims:
                        ohya = ohya.isel(time=0).squeeze()
                    if 'time' in ohyb.dims:
                        ohyb = ohyb.isel(time=0).squeeze()
                    if 'PS' in oclim_ds:
                        ops = oclim_ds['PS']
                    else:
                        # look for the file (this isn't great b/c we'd have to constantly re-load)
                        ops_files = sorted(list(dclimo_loc.glob("{}_PS_*.nc".format(data_src))))
                        if len(ops_files) > 0:
                            ops_ds = _load_dataset(ops_files)
                            ops = ops_ds['PS']
                        else:
                            continue  # what else could we do?
                else:
                    mhya = mhyb = ohya = ohyb = None

                #
                # Seasonal Averages
                # Note: xarray can do seasonal averaging, but depends on having time accessor,
                # which these prototype climo files don't.
                #

                #Create new dictionaries:
                mseasons = {}
                oseasons = {}
                dseasons = {} # hold the differences

                #Loop over season dictionary:
                for s in seasons:
                    mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                    oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')

                    if has_lev:
                        mps_season = mps.sel(time=seasons[s]).mean(dim='time')
                        ops_season = ops.sel(time=seasons[s]).mean(dim='time')
                    else:
                        mps_season = ops_season = None

                    # difference: each entry should be (lat, lon) or (plev, lat, lon)
                    # dseasons[s] = mseasons[s] - oseasons[s]
                    # difference will be calculated in plot_zonal_mean_and_save;
                    # because we can let any pressure-level interpolation happen there
                    # This could be re-visited for efficiency or improved code structure.

                    # time to make plot; here we'd probably loop over whatever plots we want for this variable
                    # I'll just call this one "Zonal_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                    # NOTE: Up to this point, nothing really differs from plot_example, except to deal with getting PS ready,
                    #       so we could have made one script instead of two.
                    #       Merging would make overall timing better because looping twice will double I/O steps.
                    #
                    plot_name = plot_loc / "{}_{}_Zonal_Mean.{}".format(var, s, plot_type)

                    #Remove old plot, if it already exists:
                    if plot_name.is_file():
                        plot_name.unlink()

                    #Create new plot:
                    pf.plot_zonal_mean_and_save(plot_name, mseasons[s], mps_season, mhya, mhyb,
                                                           oseasons[s], ops_season, ohya, ohyb, **vres)

                #End for (seasons loop)
            #End for (case names loop)
        #End for (obs/baseline loop)
    #End for (variables loop)

    #Notify user that script has ended:
    print("  ...Zonal mean plots have been generated successfully.")


#
# Helpers
#
def _load_dataset(fils):
    if len(fils) == 0:
        warnings.warn(f"Input file list is empty.")
        return None
    elif len(fils) > 1:
        return xr.open_mfdataset(fils, combine='by_coords')
    else:
        sfil = str(fils[0])
        return xr.open_dataset(sfil)
