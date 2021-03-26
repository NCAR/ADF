def averaging_example(case_name, input_ts_loc, output_loc, var_list, search=None):

    """
    This is an example function showing
    how to set-up a time-averaging method
    for calculating climatologies from
    CAM time series files.

    Description of function inputs:

    case_name    -> Name of CAM case provided by "cam_case_name"
    input_ts_loc -> Location of CAM time series files provided by "cam_ts_loc"
    output_loc   -> Location to write CAM climo files to, provided by "cam_climo_loc"
    var_list     -> List of CAM output variables provided by "diag_var_list"

    search -> optional; if supplied requires a string used as a template to find the time series files
              using {CASE} and {VARIABLE} and otherwise an arbitrary shell-like globbing pattern:
              example 1: provide the string "{CASE}.*.{VARIABLE}.*.nc" this is the default
              example 2: maybe CASE is not necessary because post-process destroyed the info "post_process_text-{VARIABLE}.nc"
              example 3: order does not matter "{VARIABLE}.{CASE}.*.nc"
              Only CASE and VARIABLE are allowed because they are arguments to the averaging function
    """

    #Import necessary modules:
    import xarray as xr
    from pathlib import Path

    #Notify user that script has started:
    print("  Calculating CAM climatologies...")

    #Create "Path" objects:
    input_location  = Path(input_ts_loc)
    output_location = Path(output_loc)

    #Check that time series input directory actually exists:
    if not input_location.is_dir():
        errmsg = "Time series directory '{}' not found.  Script is exiting.".format(input_ts_loc)
        end_diag_script(errmsg)

    #Check if climo directory exists, and if not, then create it:
    if not output_location.is_dir():
        print("    {} not found, making new directory".format(output_loc))
        output_location.mkdir(parents=True)

    # Time series file search
    if search is None:
        search = "{CASE}*.{VARIABLE}.*.nc"

    #Loop over CAM output variables:
    for var in var_list:

        #Create list of time series files present for variable:
        ts_filenames = search.format(CASE=case_name, VARIABLE=var)
        ts_files = sorted(list(input_location.glob(ts_filenames)))

        #If no files exist, then kill diagnostics script (for now):
        if not ts_files:
             errmsg = "Time series files for variable '{}' not found.  Script is exiting.".format(var)
             end_diag_script(errmsg)

        #Read in files via xarray (xr):
        if len(ts_files) == 1:
            cam_ts_data = xr.open_dataset(ts_files[0], decode_times=True)
        else:
            cam_ts_data = xr.open_mfdataset(ts_files, decode_times=True, combine='by_coords')

        #Average time dimension over time bounds, if bounds exist:
        if 'time_bnds' in cam_ts_data:
            time = cam_ts_data['time']
            time = xr.DataArray(cam_ts_data['time_bnds'].mean(dim='nbnd').values, dims=time.dims, attrs=time.attrs)
            cam_ts_data['time'] = time
            cam_ts_data.assign_coords(time=time)
            cam_ts_data = xr.decode_cf(cam_ts_data)

        #Group time series values by month, and average those months together:
        cam_climo_data = cam_ts_data.groupby('time.month').mean(dim='time')

        #Rename "months" to "time":
        cam_climo_data = cam_climo_data.rename({'month':'time'})

        #Create name of climatology output file (which includes the full path):
        output_file = output_location / "{}_{}_climo.nc".format(case_name,var)

        #Set netCDF encoding method (deal with getting non-nan fill values):
        enc_dv = {xname: {'_FillValue': None, 'zlib': True, 'complevel': 4} for xname in cam_climo_data.data_vars}
        enc_c  = {xname: {'_FillValue': None} for xname in cam_climo_data.coords}
        enc    = {**enc_c, **enc_dv}

        #Output variable climatology to NetCDF-4 file:
        cam_climo_data.to_netcdf(output_file, format='NETCDF4', encoding=enc)

    #Notify user that script has ended:
    print("  ...CAM climatologies have been calculated successfully.")
