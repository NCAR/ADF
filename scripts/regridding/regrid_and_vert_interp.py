"""Driver for horizontal and vertical interpolation.
"""
import xarray as xr

def regrid_and_vert_interp(adf):

    """
    Regrids the test cases to the same horizontal
    grid as the reference climatology and vertically 
    interpolates the test case (and reference if needed) 
    to match a default set of pressure levels (in hPa).

    Parameters
    ----------
    adf
        The ADF object
    

    Notes
    -----
    Default pressure levels:
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50,
    30, 20, 10, 7, 5, 3, 2, 1

    Currently any 3-D observations file needs to have equivalent pressure
    levels in order to work properly, although in the future it is hoped
    to enable the vertical interpolation of observations as well.
    """

    #Import necessary modules:
    import numpy as np
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
    msg = "\n  Regridding CAM climatologies..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    overwrite_regrid = adf.get_basic_info("cam_overwrite_regrid", required=True)
    output_loc       = adf.get_basic_info("cam_regrid_loc", required=True)
    var_list         = adf.diag_var_list
    var_defaults     = adf.variable_defaults

    #CAM simulation variables (these quantities are always lists):
    case_names = adf.get_cam_info("cam_case_name", required=True)
    input_climo_locs = adf.get_cam_info("cam_climo_loc", required=True)

    #Grab case years
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    #Check if mid-level pressure, ocean fraction or land fraction exist
    #in the variable list:
    for var in ["PMID", "OCNFRAC", "LANDFRAC"]:
        if var in var_list:
            #If so, then move them to the front of variable list so
            #that they can be used to mask or vertically interpolate
            #other model variables if need be:
            var_idx = var_list.index(var)
            var_list.pop(var_idx)
            var_list.insert(0,var)
        #End if
    #End for

    #Create new variables that potentially stores the re-gridded
    #ocean/land fraction dataset:
    ocn_frc_ds = None
    tgt_ocn_frc_ds = None

    #Check if surface pressure exists in variable list:
    if "PS" in var_list:
        #If so, then move it to front of variable list so that
        #it can be used to vertically interpolate model variables
        #if need be.  This should be done after PMID so that the order
        #is PS, PMID, other variables:
        ps_idx = var_list.index("PS")
        var_list.pop(ps_idx)
        var_list.insert(0,"PS")
    #End if

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

        #Create empty dictionaries which store the locations of regridded surface
        #pressure and mid-level pressure fields:
        ps_loc_dict = {}
        pmid_loc_dict = {}

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
                regridded_file_loc = rgclimo_loc / f'{target}_{case_name}_{var}_regridded.nc'

                #If surface or mid-level pressure, then save for potential use by other variables:
                if var == "PS":
                    ps_loc_dict[target] = regridded_file_loc
                elif var == "PMID":
                    pmid_loc_dict[target] = regridded_file_loc
                #End if

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
                        print(f"\t    WARNING: regridding {var} failed, no climo file for case '{target}'. Continuing to next variable.")
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
                        #wmsg = f"\t    WARNING: Unable to find climo file for '{var}'."
                        #wmsg += " Continuing to next variable."
                        wmsg= f"\t    WARNING: regridding {var} failed, no climo file for case '{case_name}'. Continuing to next variable."
                        print(wmsg)
                        continue
                    else:
                        #Open single file as new xarray dataset:
                        mclim_ds = xr.open_dataset(mclim_fils[0])
                    #End if

                    #Create keyword arguments dictionary for regridding function:
                    regrid_kwargs = {}

                    #Check if target in relevant pressure variable dictionaries:
                    if target in ps_loc_dict:
                        regrid_kwargs.update({'ps_file': ps_loc_dict[target]})
                    #End if
                    if target in pmid_loc_dict:
                        regrid_kwargs.update({'pmid_file': pmid_loc_dict[target]})
                    #End if

                    #Perform regridding and interpolation of variable:
                    rgdata_interp = _regrid_and_interpolate_levs(mclim_ds, var,
                                                                 regrid_dataset=tclim_ds,
                                                                 **regrid_kwargs)

                    #Extract defaults for variable:
                    var_default_dict = var_defaults.get(var, {})

                    if 'mask' in var_default_dict:
                        if var_default_dict['mask'].lower() == 'ocean':
                            #Check if the ocean fraction has already been regridded
                            #and saved:
                            if ocn_frc_ds:
                                ofrac = ocn_frc_ds['OCNFRAC']
                                # set the bounds of regridded ocnfrac to 0 to 1
                                ofrac = xr.where(ofrac>1,1,ofrac)
                                ofrac = xr.where(ofrac<0,0,ofrac)

                                # apply ocean fraction mask to variable
                                rgdata_interp['OCNFRAC'] = ofrac
                                var_tmp = rgdata_interp[var]
                                var_tmp = pf.mask_land_or_ocean(var_tmp,ofrac)
                                rgdata_interp[var] = var_tmp
                            else:
                                print(f"\t    WARNING: OCNFRAC not found, unable to apply mask to '{var}'")
                            #End if
                        else:
                            #Currently only an ocean mask is supported, so print warning here:
                            wmsg = "\t    WARNING: Currently the only variable mask option is 'ocean',"
                            wmsg += f"not '{var_default_dict['mask'].lower()}'"
                            print(wmsg)
                        #End if
                    #End if

                    #If the variable is ocean fraction, then save the dataset for use later:
                    if var == 'OCNFRAC':
                        ocn_frc_ds = rgdata_interp
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

                    #Now vertically interpolate baseline (target) climatology,
                    #if applicable:

                    #Set interpolated baseline file name:
                    interp_bl_file = rgclimo_loc / f'{target}_{var}_baseline.nc'

                    if not adf.compare_obs and not interp_bl_file.is_file():

                        #Look for a baseline climo file for surface pressure (PS):
                        bl_ps_fil = tclimo_loc / f'{target}_PS_climo.nc'

                        #Also look for a baseline climo file for mid-level pressure (PMID):
                        bl_pmid_fil = tclimo_loc / f'{target}_PMID_climo.nc'

                        #Create new keyword arguments dictionary for regridding function:
                        regrid_kwargs = {}

                        #Check if PS and PMID files exist:
                        if bl_ps_fil.is_file():
                            regrid_kwargs.update({'ps_file': bl_ps_fil})
                        #End if
                        if bl_pmid_fil.is_file():
                            regrid_kwargs.update({'pmid_file': bl_pmid_fil})
                        #End if

                        #Generate vertically-interpolated baseline dataset:
                        tgdata_interp = _regrid_and_interpolate_levs(tclim_ds, var,
                                                                     **regrid_kwargs)

                        if tgdata_interp is None:
                            #Something went wrong during interpolation, so just cycle through
                            #for now:
                            continue
                        #End if

                        #If the variable is ocean fraction, then save the dataset for use later:
                        if var == 'OCNFRAC':
                            tgt_ocn_frc_ds = tgdata_interp
                        #End if

                        if 'mask' in var_default_dict:
                            if var_default_dict['mask'].lower() == 'ocean':
                                #Check if the ocean fraction has already been regridded
                                #and saved:
                                if tgt_ocn_frc_ds:
                                    ofrac = tgt_ocn_frc_ds['OCNFRAC']
                                    # set the bounds of regridded ocnfrac to 0 to 1
                                    ofrac = xr.where(ofrac>1,1,ofrac)
                                    ofrac = xr.where(ofrac<0,0,ofrac)
                                    # mask the land in TS for global means
                                    tgdata_interp['OCNFRAC'] = ofrac
                                    ts_tmp = tgdata_interp[var]
                                    ts_tmp = pf.mask_land_or_ocean(ts_tmp,ofrac)
                                    tgdata_interp[var] = ts_tmp
                                else:
                                    wmsg = "\t    WARNING: OCNFRAC not found in target,"
                                    wmsg += f" unable to apply mask to '{var}'"
                                    print(wmsg)
                                #End if
                            #End if
                        #End if

                        # Convert the list to a string (join with commas or another separator)
                        climatology_files_str = [str(path) for path in tclim_fils]
                        climatology_files_str = ', '.join(climatology_files_str)
                        # Create a dictionary of attributes
                        base_attrs_dict = {
                            "adf_user": adf.user,
                            "climo_yrs": f"{case_name}: {syear}-{eyear}; {base_climo_yrs_attr}",
                            "climatology_files": climatology_files_str,
                        }
                        tgdata_interp = tgdata_interp.assign_attrs(base_attrs_dict)

                        #Write interpolated baseline climatology to file:
                        save_to_nc(tgdata_interp, interp_bl_file)
                    #End if
                else:
                    print("\t    INFO: Regridded file already exists, so skipping...")
                #End if (file check)
            #End do (target list)
        #End do (variable list)
    #End do (case list)

    #Notify user that script has ended:
    print("  ...CAM climatologies have been regridded successfully.")

#################
#Helper functions
#################

def _regrid_and_interpolate_levs(model_dataset, var_name, regrid_dataset=None, **kwargs):

    """
    Function that takes a variable from a model xarray
    dataset, regrids it to another dataset's lat/lon
    coordinates (if applicable), and then interpolates
    it vertically to a set of pre-defined pressure levels.

    Parameters
    ----------
    model_dataset : xarray.Dataset
        The xarray dataset which contains the model variable data
    var_name : str
        The name of the variable to be regridded/interpolated.
    regrid_dataset : xr.Dataset or xr.DataArray, optional
        The xarray object that contains the destination lat/lon grid
        If not present then only vertical interpolation will be performed.
    **kwargs
        Additional optional arguments:
        - `ps_file` : str or Path
            specify surface pressure netCDF file
        - `pmid_file` : str or Path
            specify vertical layer midpoint pressure netCDF file

    Returns
    -------
    xarray.Dataset
        This function returns a new xarray dataset that contains the regridded
        and/or vertically-interpolated model variable.
    """

    #Import ADF-specific functions:
    import plotting_functions as pf

    #Extract keyword arguments:
    if 'ps_file' in kwargs:
        ps_file = kwargs['ps_file']
    else:
        ps_file = None
    #End if
    if 'pmid_file' in kwargs:
        pmid_file = kwargs['pmid_file']
    else:
        pmid_file = None
    #End if

    #Extract variable info from model data (and remove any degenerate dimensions):
    mdata = model_dataset[var_name].squeeze()
    mdat_ofrac = None
    #if regrid_ofrac:
    #    if 'OCNFRAC' in model_dataset:
    #        mdat_ofrac = model_dataset['OCNFRAC'].squeeze()

    #Check if variable has a vertical component:
    if 'lev' in mdata.dims or 'ilev' in mdata.dims:
        has_lev = True

        #If lev exists, then determine what kind of vertical coordinate
        #is being used:
        if 'lev' in mdata.dims:
            lev_attrs = model_dataset['lev'].attrs
        elif 'ilev' in mdata.dims:
            lev_attrs = model_dataset['ilev'].attrs

        #First check if there is a "vert_coord" attribute:
        if 'vert_coord' in lev_attrs:
            vert_coord_type = lev_attrs['vert_coord']
        else:
            #Next check that the "long_name" attribute exists:
            if 'long_name' in lev_attrs:
                #Extract long name:
                lev_long_name = lev_attrs['long_name']

                #Check for "keywords" in the long name:
                if 'hybrid level' in lev_long_name:
                    #Set model to hybrid vertical levels:
                    vert_coord_type = "hybrid"
                elif 'zeta level' in lev_long_name:
                    #Set model to height (z) vertical levels:
                    vert_coord_type = "height"
                else:
                    #Print a warning, and skip variable re-gridding/interpolation:
                    wmsg = "WARNING! Unable to determine the vertical coordinate"
                    wmsg +=f" type from the 'lev' long name, which is:\n'{lev_long_name}'"
                    print(wmsg)
                    return None
                #End if

            else:
                #Print a warning, and assume hybrid levels (for now):
                wmsg = "WARNING!  No long name found for the 'lev' dimension,"
                wmsg += f" so no re-gridding/interpolation will be done."
                print(wmsg)
                return None
            #End if
        #End if

    else:
        has_lev = False
    #End if

    #Check if variable has a vertical levels dimension:
    if has_lev:

        if vert_coord_type == "hybrid":
            # Need hyam, hybm, and P0 for vertical interpolation of hybrid levels:
            if 'lev' in mdata.dims:
                if ('hyam' not in model_dataset) or ('hybm' not in model_dataset):
                    print(f"!! PROBLEM -- NO hyam or hybm for 3-D variable {var_name}, so it will not be re-gridded.")
                    return None #Return None to skip to next variable.
                #End if
                mhya = model_dataset['hyam']
                mhyb = model_dataset['hybm']
            elif 'ilev' in mdata.dims:
                if ('hyai' not in model_dataset) or ('hybi' not in model_dataset):
                    print(f"!! PROBLEM -- NO hyai or hybi for 3-D variable {var_name}, so it will not be re-gridded.")
                    return None #Return None to skip to next variable.
                #End if
                mhya = model_dataset['hyai']
                mhyb = model_dataset['hybi']
            if 'time' in mhya.dims:
                mhya = mhya.isel(time=0).squeeze()
            if 'time' in mhyb.dims:
                mhyb = mhyb.isel(time=0).squeeze()
            if 'P0' in model_dataset:
                P0_tmp = model_dataset['P0']
                if isinstance(P0_tmp, xr.DataArray):
                    #All of these value should be the same,
                    #so just grab the first one:
                    P0 = P0_tmp[0]
                else:
                    #Just rename variable:
                    P0 = P0_tmp
                #End if
            else:
                P0 = 100000.0  # Pa
            #End if

        elif vert_coord_type == "height":
            #Initialize already-regridded PMID logical:
            regridded_pmid = False

            #Need mid-level pressure for vertical interpolation of height levels:
            if 'PMID' in model_dataset:
                mpmid = model_dataset['PMID']
            else:
                #Check if target has an associated surface pressure field:
                if pmid_file:
                    mpmid_ds = xr.open_dataset(pmid_file)
                    mpmid = mpmid_ds['PMID']
                    #This mid-level pressure field has already been regridded:
                    regridded_pmid = True
                else:
                    print(f"!! PROBLEM -- NO PMID for 3-D variable {var_name}, so it will not be re-gridded.")
                    return None
                #End if
            #End if
        #End if (vert_coord_type)

        #It is probably good to try and acquire PS for all vertical coordinate types, so try here:
        regridded_ps = False
        if 'PS' in model_dataset:
            mps = model_dataset['PS']
        else:
            #Check if target has an associated surface pressure field:
            if ps_file:
                mps_ds = xr.open_dataset(ps_file)
                mps = mps_ds['PS']
                #This surface pressure field has already been regridded:
                regridded_ps = True
            else:
                print(f"!! PROBLEM -- NO PS for 3-D variable {var_name}, so it will not be re-gridded.")
                return None
            #End if
        #End if
    #End if (has_lev)

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

        #Regrid model data to match target grid:
        rgdata = regrid_data(mdata, tgrid, method=1)
        if mdat_ofrac:
            rgofrac = regrid_data(mdat_ofrac, tgrid, method=1)
        #Regrid surface pressure if need be:
        if has_lev:
            if not regridded_ps:
                rg_ps = regrid_data(mps, tgrid, method=1)
            else:
                rg_ps = mps
            #End if

            #Also regrid mid-level pressure if need be:
            if vert_coord_type == "height":
                if not regridded_pmid:
                    rg_pmid = regrid_data(mpmid, tgrid, method=1)
                else:
                    rg_pmid = mpmid
                #End if
            #End if
        #End if
    else:
        #Just rename variables:
        rgdata = mdata
        if has_lev:
            rg_ps = mps
            if vert_coord_type == "height":
                rg_pmid = mpmid
            #End if
        #End if
    #End if

    #Vertical interpolation:

    #Interpolate variable to default pressure levels:
    if has_lev:

        if vert_coord_type == "hybrid":
            #Interpolate from hybrid sigma-pressure to the standard pressure levels:
            rgdata_interp = pf.lev_to_plev(rgdata, rg_ps, mhya, mhyb, P0=P0, \
                                           convert_to_mb=True)
        elif vert_coord_type == "height":
            #Interpolate variable using mid-level pressure (PMID):
            rgdata_interp = pf.pmid_to_plev(rgdata, rg_pmid, convert_to_mb=True)
        else:
            #The vertical coordinate type is un-recognized, so print warning and
            #skip vertical interpolation:
            wmsg = f"WARNING! Un-recognized vertical coordinate type: '{vert_coord_type}',"
            wmsg += f" for variable '{var_name}'.  Skipping vertical interpolation."
            print(wmsg)
            #Don't process variable:
            return None
        #End if
    else:
        #Just rename variable:
        rgdata_interp = rgdata
    #End if

    #Convert to xarray dataset:
    rgdata_interp = rgdata_interp.to_dataset()
    if mdat_ofrac:
        rgdata_interp['OCNFRAC'] = rgofrac

    #Add surface pressure to variable if a hybrid (just in case):
    if has_lev:
        rgdata_interp['PS'] = rg_ps

        #Update "vert_coord" attribute for variable "lev":
        rgdata_interp['lev'].attrs.update({"vert_coord": "pressure"})
    #End if

    #Return dataset:
    return rgdata_interp


def save_to_nc(tosave, outname, attrs=None, proc=None):
    """Saves xarray variable to new netCDF file
    
    Parameters
    ----------
    tosave : xarray.Dataset or xarray.DataArray
        data to write to file
    outname : str or Path
        output netCDF file path
    attrs : dict, optional
        attributes dictionary for data
    proc : str, optional
        string to append to "Processing_info" attribute    
    """

    xo = tosave
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


def regrid_data(fromthis, tothis, method=1):
    """Regrid between lat-lon grids using various different methods
    
    Parameters
    ----------
    fromthis : xarray.DataArray
        original data
    tothis : xarray.DataArray
        provides destination grid information (regular lat-lon)
    method : int, optional
        method to use for regridding
        1 - xarray, `interp_like`
        2 - xarray, `interp`
        3 - xESMF, `Regridder()`
        4 - GeoCAT, `linint2` (may be deprecated)

    Returns
    -------
    xarray.DataArray
        Data interpolated to destination grid

    Notes
    -----
    1. xarray's interpolation does not respect longitude's periodicity
    2. xESMF can sometimes malfunction depending on dependencies
    3. GeoCAT `linint2` might be deprecated
    
    A more robust regridding solution is being explored.
    
    """

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