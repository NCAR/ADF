"""
Module to create (monthly) climatology files.
"""

from adf_base import AdfError
import adf_utils as utils

import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

import numpy as np
import xarray as xr  # module-level import so all functions can get to it.


def get_time_slice_by_year(time, startyear, endyear):
    """
    Return slice object inclusive of specified start and end years.

    Parameters
    ----------
    time
        Time coordinate variable, expects xarray `dt` accessor.
    startyear : int
        The year to start the slice
    endyear : int
        The year to end the slice

    Returns
    -------
    slice
        slice object with start and end years specified;
        if `dt` accessor is available values will be time indices

    Notes
    -----
    When the `dt` accessor is not available, instead of indices
    returns `slice(startyear, endyear)` and prints a warning
    since this is unlikely to actually work.
    """
    if not hasattr(time, 'dt'):
        print("Warning: get_time_slice_by_year requires the `time` parameter to be an xarray time coordinate with a dt accessor. Returning generic slice (which will probably fail).")
        return slice(startyear, endyear)
    start_time_index = np.argwhere((time.dt.year >= startyear).values).flatten().min()
    end_time_index = np.argwhere((time.dt.year <= endyear).values).flatten().max()
    return slice(start_time_index, end_time_index+1)



##############
#Main function
##############

def create_climo_files(adf, clobber=False, search=None):
    """
    Orchestrates production of monthly climatology files
    from CAM time series files.

    Parameters
    ----------
    adf
        the ADF object
    clobber : bool, optional
        Overwrite existing climatology files if true.
        Defaults to False (do not delete).
    search : str, optional
        optional string used as a template to find the time series files
        using {CASE} and {VARIABLE} and otherwise an arbitrary shell-like globbing pattern:
        example 1: provide the string "{CASE}.*.{VARIABLE}.*.nc" this is the default
        example 2: maybe CASE is not necessary because post-process destroyed the info "post_process_text-{VARIABLE}.nc"
        example 3: order does not matter "{VARIABLE}.{CASE}.*.nc"
        Only CASE and VARIABLE are allowed because they are arguments to the averaging function

    Notes
    -----
    No return value; produces netCDF files.

    Uses multiprocessing for parallelization.

    Calls local function `process_variable` for calculation.

    Description of needed inputs from ADF:
        case_name    -> Name of CAM case provided by "cam_case_name"
        input_ts_loc -> Location of CAM time series files provided by "cam_ts_loc"
        output_loc   -> Location to write CAM climo files to, provided by "cam_climo_loc"
        var_list     -> List of CAM output variables provided by "diag_var_list"
    """

    #Import necessary modules:
    from pathlib import Path

    #Notify user that script has started:
    msg = "\n  Calculating CAM climatologies..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Set up multiprocessing pool to parallelize writing climo files.
    number_of_cpu = adf.num_procs  # Get number of available processors from the ADF

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    var_list = adf.diag_var_list

    #CAM simulation variables (These quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    input_ts_locs = adf.get_cam_info("cam_ts_loc", required=True)
    output_locs   = adf.get_cam_info("cam_climo_loc", required=True)
    calc_climos   = adf.get_cam_info("calc_cam_climo")
    overwrite     = adf.get_cam_info("cam_overwrite_climo")
    print(f"CHECK ON INPUT")
    print(case_names)
    print(input_ts_locs)


    #Extract simulation years:
    start_year = adf.climo_yrs["syears"]
    end_year   = adf.climo_yrs["eyears"]

    #If variables weren't provided in config file, then make them a list
    #containing only None-type entries:
    if not calc_climos:
        calc_climos = [None]*len(case_names)
    if not overwrite:
        overwrite = [None]*len(case_names)
    #End if

    #Check if a baseline simulation is also being used:
    if not adf.get_basic_info("compare_obs"):
        #Extract CAM baseline variaables:
        baseline_name     = adf.get_baseline_info("cam_case_name", required=True)
        input_ts_baseline = adf.get_baseline_info("cam_ts_loc", required=True)
        output_bl_loc     = adf.get_baseline_info("cam_climo_loc", required=True)
        calc_bl_climos    = adf.get_baseline_info("calc_cam_climo")
        ovr_bl            = adf.get_baseline_info("cam_overwrite_climo")

        #Extract baseline years:
        bl_syr = adf.climo_yrs["syear_baseline"]
        bl_eyr = adf.climo_yrs["eyear_baseline"]

        #Append to case lists:
        case_names.append(baseline_name)
        input_ts_locs.append(input_ts_baseline)
        output_locs.append(output_bl_loc)
        calc_climos.append(calc_bl_climos)
        overwrite.append(ovr_bl)
        start_year.append(bl_syr)
        end_year.append(bl_eyr)
    #-----------------------------------------

    # Check whether averaging interval is supplied
    # -> using only years takes from the beginning of first year to end of second year.
    # -> slice('1991','1998') will get all of [1991,1998].
    # -> slice(None,None) will use all times.


    #Loop over CAM cases:
    for case_idx, case_name in enumerate(case_names):

        #Check if climatology is being calculated.
        #If not then just continue on to the next case:
        if not calc_climos[case_idx]:
            continue

        #Notify user of model case being processed:
        print(f"\n\t Calculating climatologies for case '{case_name}' :")

        is_baseline = False
        if (not adf.get_basic_info("compare_obs")) and (case_name == baseline_name):
            is_baseline = True

        #Create "Path" objects:
        input_location  = Path(input_ts_locs[case_idx])
        output_location = Path(output_locs[case_idx])

        #Whether to overwrite existing climo files
        clobber = overwrite[case_idx]

        #Check that time series input directory actually exists:
        if not input_location.is_dir():
            errmsg = f"Time series directory '{input_ts_locs}' not found.  Script is exiting."
            raise AdfError(errmsg)

        #Check if climo directory exists, and if not, then create it:
        if not output_location.is_dir():
            print(f"\t    {output_location} not found, making new directory")
            output_location.mkdir(parents=True)

        # If we need to allow custom search, could put it into adf.data
        # #Time series file search
        # if search is None:
        #     search = "{CASE}*{HIST_STR}*.{VARIABLE}.*nc"  # NOTE: maybe we should not care about the file extension part at all, but check file type later?

        #Check model year bounds:
        syr, eyr = check_averaging_interval(start_year[case_idx], end_year[case_idx])

        #Loop over CAM output variables:
        list_of_arguments = []
        for var in var_list:
            # Notify user of new climo file:
            print(f"\t - climatology for {var}")

            # Create name of climatology output file (which includes the full path)
            # and check whether it is there (don't do computation if we don't want to overwrite):
            output_file = output_location / f"{case_name}_{var}_climo.nc"
            if (not clobber) and (output_file.is_file()):
                msg = f"\t    INFO: '{var}' file was found "
                msg += "and overwrite is False. Will use existing file."
                print(msg)
                continue
            elif (clobber) and (output_file.is_file()):
                print(f"\t    INFO: Climo file exists for {var}, but clobber is {clobber}, so will OVERWRITE it.")

            #Create list of time series files present for variable:
            # Note that we hard-code for h0 because we only want to make climos of monthly output
            if is_baseline:
                ts_files = adf.data.get_ref_timeseries_file(var)
            else:
                ts_files = adf.data.get_timeseries_file(case_name, var)

            #If no files exist, try to move to next variable. --> Means we can not proceed with this variable,
            # and it'll be problematic later unless there are multiple hist file streams and the variable is in the others
            if not ts_files:
                errmsg = f"\t    WARNING: Time series files for variable '{var}' not found.  Script will continue to next variable.\n"
                errmsg += f"\t      The input location searched was: {input_location}."
                print(errmsg)
                logmsg = f"climo file generation: The input location searched was: {input_location}. The glob pattern was {ts_files}."
                #Write to debug log if enabled:
                adf.debug_log(logmsg)
                #  end_diag_script(errmsg) # Previously we would kill the run here.
                continue

            list_of_arguments.append((adf.user, ts_files, syr, eyr, output_file))

        # Parallelize the computation using multiprocessing pool:
        print(f"  --> Starting Pool with {number_of_cpu} workers for {len(list_of_arguments)} variables.")
        import multiprocessing as mp
        # Use 'spawn' to ensure a fresh memory space for each process
        # Safer on HPC systems than the default 'fork'
        context = mp.get_context('spawn')
        with context.Pool(processes=number_of_cpu) as p:
            results = p.starmap(process_variable, list_of_arguments)
        # Print results to see if any specific variable failed
        for res in results:
            if "Failed" in res:
                print(f"\t    {res}")
        print("  ... multiprocessing pool closed.")
    print("  ...CAM climatologies have been calculated successfully.")


#
# Local functions
#
def process_variable(adf_user, ts_files, syr, eyr, output_file):
    '''
    Compute and save the monthly climatology file.

    Parameters
    ----------
    adf_user
        The user from the ADF object
    ts_files : list
        list of paths to time series files
    syr : str
        start year, with leading zeros if needed
    eyr : str
        end year, with leading zeros if needed
    output_file : str or Path
        file path for output climatology file
    '''
    import xarray as xr
    import numpy as np
    import dask
    import gc
    dask.config.set(scheduler='synchronous') # Disable internal dask multi-threading
    try:
        # Using chunks={} forces xarray to use dask, which handles memory better 
        # than loading everything into RAM at once via open_dataset
        with xr.open_mfdataset(ts_files, decode_times=True, combine='by_coords', chunks={'time': 12}) as ds:
            if 'time_bnds' in ds:
                new_time = ds['time_bnds'].load().mean(dim='nbnd')
                ds = ds.assign_coords(time=new_time.values)
                ds = xr.decode_cf(ds)

            tslice = get_time_slice_by_year(ds.time, int(syr), int(eyr))
            ds_subset = ds.isel(time=tslice)

            climo = ds_subset.groupby('time.month').mean(dim='time')
            climo = climo.rename({'month': 'time'})

            enc_dv = {xname: {'_FillValue': None, 'zlib': True, 'complevel': 4} for xname in climo.data_vars}
            enc_c  = {xname: {'_FillValue': None} for xname in climo.coords}
            enc    = {**enc_c, **enc_dv}

            climo.attrs.update({
                "adf_user": adf_user,
                "climo_yrs": f"{syr}-{eyr}",
                "time_series_files": ", ".join([str(f) for f in ts_files])
            })

            climo.to_netcdf(output_file, format='NETCDF4', encoding=enc)
        return f"Success: {output_file.name}"
    except Exception as e:
        return f"Failed: {output_file.name} with error: {str(e)}"
    finally:
        # Force cleanup of memory
        gc.collect()

def check_averaging_interval(syear_in, eyear_in):
    """
    Parameters
    ----------
    syear_in
        start year, should be convertible to int
    eyear_in
        end year, should be convertible to int

    Returns
    -------
    tuple
        (start_year, end_year) as str with leading zeros included if needed (4-digit)
    """
    #For now, make sure year inputs are integers or None,
    #in order to allow for the zero additions done below:
    if syear_in:
        check_syr = int(syear_in)
    else:
        check_syr = None
    #end if

    if eyear_in:
        check_eyr = int(eyear_in)
    else:
        check_eyr = None

    #Need to add zeros if year values aren't long enough:
    #------------------
    #start year:
    if check_syr:
        assert check_syr >= 0, 'Sorry, values must be positive whole numbers.'
        try:
            syr = f"{check_syr:04d}"
        except:
            errmsg = " 'start_year' values must be positive whole numbers"
            errmsg += f"not '{syear_in}'."
            raise AdfError(errmsg)
    else:
        syr = None
    #End if

    #end year:
    if check_eyr:
        assert check_eyr >= 0, 'Sorry, end_year values must be positive whole numbers.'
        try:
            eyr = f"{check_eyr:04d}"
        except:
            errmsg = " 'end_year' values must be positive whole numbers"
            errmsg += f"not '{eyear_in}'."
            raise AdfError(errmsg)
    else:
        eyr = None
    #End if
    return syr, eyr