#Import standard modules:
import xarray as xr

def regrid_climo_wrapper(adf):

    """
    This funtion regrids the test cases to the same horizontal
    grid as the observations or baseline climatology.

    Description of needed inputs from ADF:

    case_name        -> Name of CAM case provided by "cam_case_name"
    input_climo_loc  -> Location of CAM climo files provided by "cam_climo_loc"
    output_loc       -> Location to write re-gridded CAM files, specified by "cam_climo_regrid_loc"
    var_list         -> List of CAM output variables provided by "diag_var_list"
    var_defaults     -> Dict that has keys that are variable names and values that are plotting preferences/defaults.
    target_list      -> List of target data sets CAM could be regridded to
    taget_loc        -> Location of target files that CAM will be regridded to
    overwrite_regrid -> Logical to determine if already existing re-gridded
                        files will be overwritten. Specified by "cam_overwrite_climo_regrid"
    """

    #Import necessary modules:
    import plotting_functions as pf

    from pathlib import Path

    # regridding
    # Try just using the xarray method
    # import xesmf as xe  # This package is for regridding, and is just one potential solution.

    # Steps:
    # - load climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - regrid one to the other (probably should be a choice)

    #Notify user that script has started:
    print("\n  Regridding CAM climatologies...")

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    overwrite_regrid = adf.get_basic_info("cam_overwrite_climo_regrid", required=True)
    output_loc       = adf.get_basic_info("cam_climo_regrid_loc", required=True)
    var_list         = adf.diag_var_list
    var_defaults     = adf.variable_defaults

    #CAM simulation variables (these quantities are always lists):
    case_names = adf.get_cam_info("cam_case_name", required=True)
    input_climo_locs = adf.get_cam_info("cam_climo_loc", required=True)

    #Grab case years
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    #Check if land fraction exists
    #in the variable list:
    for var in ["LANDFRAC"]:
        if var in var_list:
            #If so, then move it to the front of variable list so
            #that it can be used to mask
            #other model variables if need be:
            var_idx = var_list.index(var)
            var_list.pop(var_idx)
            var_list.insert(0,var)
        #End if
    #End for

    #Create new variable that potentially stores the re-gridded
    #land fraction dataset:
    lnd_frc_ds = None

    #Regrid target variables (either obs or a baseline run):
    if adf.compare_obs:

        #Set obs name to match baseline (non-obs)
        target_list = ["Obs"]

        #Extract variable-obs dictionary:
        var_obs_dict = adf.var_obs_dict

        #If dictionary is empty, then  there are no observations to regrid to,
        #so quit here:
        if not var_obs_dict:
            print("\t No observations found to regrid to, so no re-gridding will be done.")
            return
        #End if

    else:

        #Extract model baseline variables:
        target_loc = adf.get_baseline_info("cam_climo_loc", required=True)
        target_list = [adf.get_baseline_info("cam_case_name", required=True)]
    #End if

    #Grab baseline years (which may be empty strings if using Obs):
    syear_baseline = adf.climo_yrs["syear_baseline"]
    eyear_baseline = adf.climo_yrs["eyear_baseline"]

    #Set attributes dictionary for climo years to save in the file attributes
    base_climo_yrs_attr = f"{target_list[0]}: {syear_baseline}-{eyear_baseline}"

    #-----------------------------------------

    #Set output/target data path variables:
    #------------------------------------
    rgclimo_loc = Path(output_loc)
    if not adf.compare_obs:
        tclimo_loc  = Path(target_loc)
    #------------------------------------

    #Check if re-gridded directory exists, and if not, then create it:
    if not rgclimo_loc.is_dir():
        print(f"    {rgclimo_loc} not found, making new directory")
        rgclimo_loc.mkdir(parents=True)
    #End if

    #Loop over CAM cases:
    for case_idx, case_name in enumerate(case_names):

        #Notify user of model case being processed:
        print(f"\t Regridding case '{case_name}' :")

        #Set case climo data path:
        mclimo_loc  = Path(input_climo_locs[case_idx])

        #Get climo years for case
        syear = syear_cases[case_idx]
        eyear = eyear_cases[case_idx]

        # probably want to do this one variable at a time:
        for var in var_list:

            if adf.compare_obs:
                #Check if obs exist for the variable:
                if var in var_obs_dict:
                    #Note: In the future these may all be lists, but for
                    #now just convert the target_list.
                    #Extract target file:
                    tclimo_loc = var_obs_dict[var]["obs_file"]
                    #Extract target list (eventually will be a list, for now need to convert):
                    target_list = [var_obs_dict[var]["obs_name"]]
                else:
                    dmsg = f"No obs found for variable `{var}`, regridding skipped."
                    adf.debug_log(dmsg)
                    continue
                #End if
            #End if

            #Notify user of variable being regridded:
            print(f"\t - regridding {var} (known targets: {target_list})")

            #loop over regridding targets:
            for target in target_list:

                #Write to debug log if enabled:
                adf.debug_log(f"regrid_example: regrid target = {target}")

                #Determine regridded variable file name:
                regridded_file_loc = rgclimo_loc / f'{case_name}_{var}_regridded.nc'

                #Check if re-gridded file already exists and over-writing is allowed:
                if regridded_file_loc.is_file() and overwrite_regrid:
                    #If so, then delete current file:
                    regridded_file_loc.unlink()
                #End if

                #Check again if re-gridded file already exists:
                if not regridded_file_loc.is_file():

                    #Create list of regridding target files (we should explore intake as an alternative to having this kind of repeated code)
                    # NOTE: This breaks if you have files from different cases in same directory!
                    if adf.compare_obs:
                        #For now, only grab one file (but convert to list for use below):
                        tclim_fils = [tclimo_loc]
                    else:
                       tclim_fils = sorted(tclimo_loc.glob(f"{target}*_{var}_climo.nc"))
                    #End if

                    #Write to debug log if enabled:
                    adf.debug_log(f"regrid_example: tclim_fils (n={len(tclim_fils)}): {tclim_fils}")

                    if len(tclim_fils) > 1:
                        #Combine all target files together into a single data set:
                        tclim_ds = xr.open_mfdataset(tclim_fils, combine='by_coords')
                    elif len(tclim_fils) == 0:
                        print(f"\t - regridding {var} failed, no file. Continuing to next variable.")
                        continue
                    else:
                        #Open single file as new xarray dataset:
                        tclim_ds = xr.open_dataset(tclim_fils[0])
                    #End if

                    #Generate CAM climatology (climo) file list:
                    mclim_fils = sorted(mclimo_loc.glob(f"{case_name}_{var}_*.nc"))

                    if len(mclim_fils) > 1:
                        #Combine all cam files together into a single data set:
                        mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
                    elif len(mclim_fils) == 0:
                        wmsg = f"\t - Unable to find climo file for '{var}'."
                        wmsg += " Continuing to next variable."
                        print(wmsg)
                        continue
                    else:
                        #Open single file as new xarray dataset:
                        mclim_ds = xr.open_dataset(mclim_fils[0])
                    #End if

                    #Create keyword arguments dictionary for regridding function:
                    regrid_kwargs = {}

                    #Perform regridding of variable:
                    rgdata_interp = _regrid(mclim_ds, var,
                                            regrid_dataset=tclim_ds,
                                            **regrid_kwargs)

                    #Extract defaults for variable:
                    var_default_dict = var_defaults.get(var, {})

                    if 'mask' in var_default_dict:
                        if var_default_dict['mask'].lower() == 'land':
                            #Check if the land fraction has already been regridded
                            #and saved:
                            if lnd_frc_ds:
                                lfrac = lnd_frc_ds['LANDFRAC']
                                # set the bounds of regridded lndfrac to 0 to 1
                                lfrac = xr.where(lfrac>1,1,lfrac)
                                lfrac = xr.where(lfrac<0,0,lfrac)

                                # apply land fraction mask to variable
                                rgdata_interp['LANDFRAC'] = lfrac
                                var_tmp = rgdata_interp[var]
                                var_tmp = pf.mask_land(var_tmp,lfrac)
                                rgdata_interp[var] = var_tmp
                            else:
                                print(f"LANDFRAC not found, unable to apply mask to '{var}'")
                            #End if
                        else:
                            #Currently only a land mask is supported, so print warning here:
                            wmsg = "Currently the only variable mask option is 'land',"
                            wmsg += f"not '{var_default_dict['mask'].lower()}'"
                            print(wmsg)
                        #End if
                    #End if

                    #If the variable is land fraction, then save the dataset for use later:
                    if var == 'LANDFRAC':
                        lnd_frc_ds = rgdata_interp
                    #End if

                    #Finally, write re-gridded data to output file:
                    #Convert the list of Path objects to a list of strings
                    climatology_files_str = [str(path) for path in mclim_fils]
                    climatology_files_str = ', '.join(climatology_files_str)
                    test_attrs_dict = {
                            "adf_user": adf.user,
                            "climo_yrs": f"{case_name}: {syear}-{eyear}",
                            "climatology_files": climatology_files_str,
                        }
                    rgdata_interp = rgdata_interp.assign_attrs(test_attrs_dict)
                    save_to_nc(rgdata_interp, regridded_file_loc)
                    rgdata_interp.close()  # bpm: we are completely done with this data

                else:
                    print("\t Regridded file already exists, so skipping...")
                #End if (file check)
            #End do (target list)
        #End do (variable list)
    #End do (case list)

    #Notify user that script has ended:
    print("  ...CAM climatologies have been regridded successfully.")

#################
#Helper functions
#################

def _regrid(model_dataset, var_name, regrid_dataset=None, regrid_ofrac=False, **kwargs):

    """
    Function that takes a variable from a model xarray
    dataset, regrids it to another dataset's lat/lon
    coordinates (if applicable)
    ----------
    model_dataset -> The xarray dataset which contains the model variable data
    var_name      -> The name of the variable to be regridded/interpolated.

    Optional inputs:

    ps_file        -> NOT APPLICABLE: A NetCDF file containing already re-gridded surface pressure
    regrid_dataset -> The xarray dataset that contains the lat/lon grid that
                      "var_name" will be regridded to.  If not present then
                      only the vertical interpolation will be done.

    kwargs         -> Keyword arguments that contain paths to THE REST IS NOT APPLICABLE: surface pressure
                      and mid-level pressure files, which are necessary for
                      certain types of vertical interpolation.

    This function returns a new xarray dataset that contains the regridded
    model variable.
    """

    #Import ADF-specific functions:
    import numpy as np
    import plotting_functions as pf
    from regrid_se_to_fv import make_se_regridder, regrid_se_data_conservative 

    #Extract keyword arguments:
    if 'ps_file' in kwargs:
        ps_file = kwargs['ps_file']
    else:
        ps_file = None
    #End if

    #Extract variable info from model data (and remove any degenerate dimensions):
    mdata = model_dataset[var_name].squeeze()
    mdat_ofrac = None
    #if regrid_lfrac:
    #    if 'LANDFRAC' in model_dataset:
    #        mdat_lfrac = model_dataset['LANDFRAC'].squeeze()

    #Regrid variable to target dataset (if available):
    if regrid_dataset:

        #Extract grid info from target data:
        if 'time' in regrid_dataset.coords:
            if 'lev' in regrid_dataset.coords:
                tgrid = regrid_dataset.isel(time=0, lev=0).squeeze()
            else:
                tgrid = regrid_dataset.isel(time=0).squeeze()
            #End if
        #End if

        # Hardwiring for now
        con_weight_file = "/glade/work/wwieder/map_ne30pg3_to_fv0.9x1.25_scripgrids_conserve_nomask_c250108.nc"

        fv_t232_file = '/glade/derecho/scratch/wwieder/ctsm5.3.018_SP_f09_t232_mask/run/ctsm5.3.018_SP_f09_t232_mask.clm2.h0.0001-01.nc'
        fv_t232 = xr.open_dataset(fv_t232_file)

        model_dataset[var_name] = model_dataset[var_name].fillna(0)
        model_dataset['landfrac']= model_dataset['landfrac'].fillna(0)
        model_dataset[var_name] = model_dataset[var_name] * model_dataset.landfrac  # weight flux by land frac

        #Regrid model data to match target grid:
        # These two functions come with import regrid_se_to_fv
        regridder = make_se_regridder(weight_file=con_weight_file,
                                      s_data = model_dataset.landmask.isel(time=0),
                                      d_data = fv_t232.landmask,
                                      Method = 'coservative',  # Bug in xesmf needs this without "n"
                                      )
        rgdata = regrid_se_data_conservative(regridder, model_dataset)

        rgdata[var_name] = (rgdata[var_name] / rgdata.landfrac)

        rgdata['lat'] = fv_t232.lat
        rgdata['landmask'] = fv_t232.landmask
        rgdata['landfrac'] = rgdata.landfrac.isel(time=0)

        # calculate area
        area_km2 = np.zeros(shape=(len(rgdata['lat']), len(rgdata['lon'])))
        earth_radius_km = 6.37122e3  # in meters

        yres_degN = np.abs(np.diff(rgdata['lat'].data))  # distances between gridcell centers...
        xres_degE = np.abs(np.diff(rgdata['lon']))  # ...end up with one less element, so...
        yres_degN = np.append(yres_degN, yres_degN[-1])  # shift left (edges <-- centers); assume...
        xres_degE = np.append(xres_degE, xres_degE[-1])  # ...last 2 distances bet. edges are equal

        dy_km = yres_degN * earth_radius_km * np.pi / 180  # distance in m
        phi_rad = rgdata['lat'].data * np.pi / 180  # degrees to radians

        # grid cell area
        for j in range(len(rgdata['lat'])):
            for i in range(len(rgdata['lon'])):
              dx_km = xres_degE[i] * np.cos(phi_rad[j]) * earth_radius_km * np.pi / 180  # distance in m
              area_km2[j,i] = dy_km[j] * dx_km

        rgdata['area'] = xr.DataArray(area_km2,
                                      coords={'lat': rgdata.lat, 'lon': rgdata.lon},
                                      dims=["lat", "lon"])
        rgdata['area'].attrs['units'] = 'km2'
        rgdata['area'].attrs['long_name'] = 'Grid cell area'
    else:
        #Just rename variables:
        rgdata = mdata
    #End if

    #Return dataset:
    return rgdata

#####

def save_to_nc(tosave, outname, attrs=None, proc=None):
    """Saves xarray variable to new netCDF file"""

    xo = tosave  # used to have more stuff here.
    # deal with getting non-nan fill values.
    if isinstance(xo, xr.Dataset):
        enc_dv = {xname: {'_FillValue': None} for xname in xo.data_vars}
    else:
        enc_dv = {}
    #End if
    enc_c = {xname: {'_FillValue': None} for xname in xo.coords}
    enc = {**enc_c, **enc_dv}
    if attrs is not None:
        xo.attrs = attrs
    if proc is not None:
        xo.attrs['Processing_info'] = f"Start from file {origname}. " + proc
    xo.to_netcdf(outname, format='NETCDF4', encoding=enc)

#####

def regrid_data(fromthis, tothis, method=1):
    """Regrid data using various different methods"""

    if method == 1:
        # kludgy: spatial regridding only, seems like can't automatically deal with time
        if 'time' in fromthis.coords:
            result = [fromthis.isel(time=t).interp_like(tothis) for t,time in enumerate(fromthis['time'])]
            result = xr.concat(result, 'time')
            return result
        else:
            return fromthis.interp_like(tothis)
    elif method == 2:
        newlat = tothis['lat']
        newlon = tothis['lon']
        coords = dict(fromthis.coords)
        coords['lat'] = newlat
        coords['lon'] = newlon
        return fromthis.interp(coords)
    elif method == 3:
        newlat = tothis['lat']
        newlon = tothis['lon']
        ds_out = xr.Dataset({'lat': newlat, 'lon': newlon})
        regridder = xe.Regridder(fromthis, ds_out, 'bilinear')
        return regridder(fromthis)
    elif method==4:
        # geocat
        newlat = tothis['lat']
        newlon = tothis['lon']
        result = geocat.comp.linint2(fromthis, newlon, newlat, False)
        result.name = fromthis.name
        return result
    #End if

#####
