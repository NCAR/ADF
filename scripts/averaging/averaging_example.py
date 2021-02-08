def averaging_example(case_name, input_ts_loc, output_loc, var_list, overwrite_climo):

    """
    This is an example function showing
    how to set-up a time-averaging method
    for calculating climatologies from
    CAM time series files.

    Description of function inputs:

    case_name       -> Name of CAM case provided by "cam_case_name"
    input_ts_loc    -> Location of CAM time series files provided by "cam_ts_loc"
    output_loc      -> Location to write CAM climo files to, provided by "cam_climo_loc"
    var_list        -> List of CAM output variables provided by "diag_var_list"
    overwrite_climo -> Logical that determines whether or not the climo files should
                       be overwritten, if they already exist.
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

    #Loop over CAM output variables:
    for var in var_list:

        #Notify users of variable be averaged:
        print("\t \u25B6 climo calculation for {}".format(var))

        #Create name of climatology output file (which includes the full path):
        output_file = output_location / "{}_{}_climo.nc".format(case_name,var)

        # don't make a climo file if it is there and over-writing is turned off:
        if Path(output_file).is_file() and not overwrite_climo:
            print("\t \u274C climo file exists for {}, skipping it.".format(var))
            continue

        #Create list of time series files present for variable:
        ts_filenames = '{}.*{}*.nc'.format(case_name, var)
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
        else:
            # rely on the time dimension being correct
            cam_ts_data = xr.decode_cf(cam_ts_data)

        #Group time series values by month, and average those months together:
        climo_data = cam_ts_data[var].groupby('time.month').mean(dim='time')

        #Add surface pressure (PS) climatologies to each data set:
        if "PS" in cam_ts_data:
            ps_climo = cam_ts_data['PS'].groupby('time.month').mean(dim='time')
            cam_climo_data = xr.merge([climo_data, ps_climo])
        else:
            cam_climo_data = climo_data.to_dataset()

        #Also add some special variables that should be carried along:
        for special in ['hyam', 'hybm', 'P0', 'hyai', 'hybi']:
            if special in cam_ts_data:
                cam_climo_data[special] = cam_ts_data[special]

        #Rename "months" to "time":
        cam_climo_data = cam_climo_data.rename({'month':'time'})

        #Set netCDF encoding method (deal with getting non-nan fill values):
        enc_dv = {xname: {'_FillValue': None, 'zlib': True, 'complevel': 4} for xname in cam_climo_data.data_vars}
        enc_c  = {xname: {'_FillValue': None} for xname in cam_climo_data.coords}
        enc    = {**enc_c, **enc_dv}

        #Output variable climatology to NetCDF-4 file:
        cam_climo_data.to_netcdf(output_file, format='NETCDF4', encoding=enc)

    #Notify user that script has ended:
    print("...CAM climatologies have been calculated successfully.")
