def regrid_example(case_name, input_climo_loc, output_loc,
                   var_list, target_list, target_loc,
                   overwrite_regrid):

    """
    This is an example function showing how to set-up a
    re-gridding method for CAM climatology (climo) files,
    so that they are on the same grid as observations or
    baseline climatologies.

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name"
    input_climo_loc  -> Location of CAM climo files provided by "cam_climo_loc"
    output_loc       -> Location to write re-gridded CAM files, specified by "cam_regrid_loc"
    var_list         -> List of CAM output variables provided by "diag_var_list"
    target_list      -> List of target data sets CAM could be regridded to
    taget_loc        -> Location of target files that CAM will be regridded to
    overwrite_regrid -> Logical to determine if already existing re-gridded
                        files will be overwritten. Specified by "cam_overwrite_regrid"
    """

    #Import necessary modules:
    import xarray as xr
    import numpy as np

    from pathlib import Path

    # regridding
    # Try just using the xarray method
    # import xesmf as xe  # This package is for regridding, and is just one potential solution.

    # Steps:
    # - load climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - regrid one to the other (probably should be a choice)

    #Notify user that script has started:
    print("  Regridding CAM climatologies...")

    #Set input/output data path variables:
    #------------------------------------
    mclimo_loc  = Path(input_climo_loc)
    rgclimo_loc = Path(output_loc)
    tclimo_loc  = Path(target_loc)
    #------------------------------------

    #Check if re-gridded directory exists, and if not, then create it:
    if not rgclimo_loc.is_dir():
        print("    {} not found, making new directory".format(rgclimo_loc))
        rgclimo_loc.mkdir(parents=True)

    # probably want to do this one variable at a time:
    for var in var_list:

        #loop over regridding targets:
        for target in target_list:

            #Determine regridded variable file name:
            regridded_file_loc = rgclimo_loc / '{}_{}_{}_regridded.nc'.format(target, case_name, var)

            #Check if re-gridded file already exists and over-writing is allowed:
            if regridded_file_loc.is_file() and overwrite_regrid:
                #If so, then delete current file:
                regridded_file_loc.unlink()

            #Check again if re-gridded file already exists:
            if not regridded_file_loc.is_file():

                #Create list of regridding target files (we should explore intake as an alternative to having this kind of repeated code)
                tclim_fils = sorted(list(tclimo_loc.glob("{}*_{}_*.nc".format(target, var))))

                if len(tclim_fils) > 1:
                    #Combine all target files together into a single data set:
                    tclim_ds = xr.open_mfdataset(tclim_fils, combine='by_coords')
                else:
                    #Open single file as new xarray dataset:
                    tclim_ds = xr.open_dataset(tclim_fils[0])

                #Generate CAM climatology (climo) file list:
                mclim_fils = sorted(list(mclimo_loc.glob("{}_{}_*.nc".format(case_name, var))))

                if len(mclim_fils) > 1:
                    #Combine all cam files together into a single data set:
                    mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
                else:
                    #Open single file as new xsarray dataset:
                    mclim_ds = xr.open_dataset(mclim_fils[0])

                #Extract variable info from data sets:
                tdata = tclim_ds[var]
                mdata = mclim_ds[var]

                #Regrid model data to match observational data:
                mdata_regrid = regrid_data(mdata, tdata)

                #Set netCDF encoding method (deal with getting non-nan fill values):
                #enc_dv = {xname: {'_FillValue': None, 'zlib': True, 'complevel': 4} for xname in mdata_regrid.data_vars}
                #enc_c  = {xname: {'_FillValue': None} for xname in mdata_regrid.coords}
                #enc    = {**enc_c, **enc_dv}

                #Write re-gridded data to output file:
                mdata_regrid.to_netcdf(regridded_file_loc, format='NETCDF4') #, encoding=enc)

    #Notify user that script has ended:
    print("  ...CAM climatologies have been regridded successfully.")

#################
#Helper functions
#################

def regrid_data(fromthis, tothis):
    """Wrapper function for 'interp_like' xarray method"""
    return fromthis.interp_like(tothis)

