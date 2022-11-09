import numpy as np
import xarray as xr
from pathlib import Path
import warnings  # use to warn user about missing files.

try:
    import scipy.stats as stats # for easy linear regression and testing
except ImportError:
    print("Scipy module does not exist in python path, but is needed for amwg_table.")
    print("Please install module, e.g. 'pip install scipy'.")

#Import "special" modules:
try:
    import pandas as pd
except ImportError:
    print("Pandas module does not exist in python path, but is needed for amwg_table.")
    print("Please install module, e.g. 'pip install pandas'.")
    sys.exit(1)


def amwg_table(adf):

    """
    Main function goes through series of steps:
    - load the variable data
    - Determine whether there are spatial dims; if yes, do global average (TODO: regional option)
    - Apply annual average (TODO: add seasonal here)
    - calculates the statistics
      + mean
      + sample size
      + standard deviation
      + standard error of the mean
      + 5/95% confidence interval of the mean
      + linear trend
      + p-value of linear trend
    - puts statistics into a CSV file
    - generates simple HTML that can display the data

    Description of needed inputs from ADF:

    case_names      -> Name(s) of CAM case provided by "cam_case_name"
    input_ts_locs   -> Location(s) of CAM time series files provided by "cam_ts_loc"
    output_loc      -> Location to write AMWG table files to, provided by "cam_diag_plot_loc"
    var_list        -> List of CAM output variables provided by "diag_var_list"

    and if doing a CAM baseline comparison:

    baseline_name     -> Name of CAM baseline case provided by "cam_case_name"
    input_ts_baseline -> Location of CAM baseline time series files provied by "cam_ts_loc"

    """

    #Import necessary modules:
    from adf_base import AdfError
    


    #Additional information:
    #----------------------

    # GOAL: replace the "Tables" set in AMWG
    #       Set Description
    #   1 Tables of ANN, DJF, JJA, global and regional means and RMSE.
    #
    # STRATEGY:
    # I think the right solution is to generate one CSV (or other?) file that
    # contains all of the data.
    # So we need:
    # - a function that would produces the data, and
    # - then call a function that adds the data to a file
    # - another function(module?) that uses the file to produce a "web page"

    # IMPLEMENTATION:
    # - assume that we will have time series of global averages already ... that should be done ahead of time
    # - given a variable or file for a variable (equivalent), we will calculate the all-time, DJF, JJA, MAM, SON
    #   + mean
    #   + standard error of the mean
    #     -- 95% confidence interval for the mean, estimated by:
    #     ---- CI95 = mean + (SE * 1.96)
    #     ---- CI05 = mean - (SE * 1.96)
    #   + standard deviation
    # AMWG also includes the RMSE b/c it is comparing two things, but I will put that off for now.

    # DETAIL: we use python's type hinting as much as possible

    # in future, provide option to do multiple domains
    # They use 4 pre-defined domains:
    domains = {"global": (0, 360, -90, 90),
               "tropics": (0, 360, -20, 20),
               "southern": (0, 360, -90, -20),
               "northern": (0, 360, 20, 90)}

    # and then in time it is DJF JJA ANN

    # within each domain and season
    # the result is just a table of
    # VARIABLE-NAME, RUN VALUE, OBS VALUE, RUN-OBS, RMSE
    #----------------------

    #Notify user that script has started:
    print("\n  Calculating AMWG variable table...")


    #Extract needed quantities from ADF object:
    #-----------------------------------------
    var_list   = adf.diag_var_list

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    output_locs = adf.plot_location

    #CAM simulation variables (these quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    input_ts_locs = adf.get_cam_info("cam_ts_loc", required=True)

    #Check if a baseline simulation is also being used:
    if not adf.get_basic_info("compare_obs"):
        #Extract CAM baseline variaables:
        baseline_name     = adf.get_baseline_info("cam_case_name", required=True)
        input_ts_baseline = adf.get_baseline_info("cam_ts_loc", required=True)

        if "CMIP" in baseline_name:
            print("CMIP files detected, skipping AMWG table (for now)...")

        else:
            #Append to case list:
            case_names.append(baseline_name)
            input_ts_locs.append(input_ts_baseline)

        #Save the baseline to the first case's plots directory:
        output_locs.append(output_locs[0])

    #-----------------------------------------
    #Create (empty) dictionary to use for the
    #residual top of model (RESTOM) radiation calculation:
    restom_dict = {}

    #Loop over CAM cases:
    for case_idx, case_name in enumerate(case_names):

        #Convert output location string to a Path object:
        output_location = Path(output_locs[case_idx])

        #Generate input file path:
        input_location = Path(input_ts_locs[case_idx])

        #Check that time series input directory actually exists:
        if not input_location.is_dir():
            errmsg = f"Time series directory '{input_location}' not found.  Script is exiting."
            raise AdfError(errmsg)
        #Write to debug log if enabled:
        adf.debug_log(f"DEBUG: location of files is {str(input_location)}")
        #Check if analysis directory exists, and if not, then create it:
        if not output_location.is_dir():
            print(f"\t    {output_locs[case_idx]} not found, making new directory")
            output_location.mkdir(parents=True)

        #Create output file name:
        output_csv_file = output_location / f"amwg_table_{case_name}.csv"

        #Given that this is a final, user-facing analysis, go ahead and re-do it every time:
        if Path(output_csv_file).is_file():
            Path.unlink(output_csv_file)

        #Save case name as a new key in the RESTOM dictonary:
        restom_dict[case_name] = {}

        #Loop over CAM output variables:
        for var in var_list:

            #Notify users of variable being added to table:
            print(f"\t - Variable '{var}' being added to table")

            #Create list of time series files present for variable:
            ts_filenames = f'{case_name}.*.{var}.*nc'
            ts_files = sorted(input_location.glob(ts_filenames))

            # If no files exist, try to move to next variable. --> Means we can not proceed with this variable, and it'll be problematic later.
            if not ts_files:
                errmsg = f"Time series files for variable '{var}' not found.  Script will continue to next variable."
                warnings.warn(errmsg)
                continue

            #TEMPORARY:  For now, make sure only one file exists:
            if len(ts_files) != 1:
                errmsg =  "Currently the AMWG table script can only handle one time series file per variable."
                errmsg += f" Multiple files were found for the variable '{var}'"
                raise AdfError(errmsg)

            #Load model data from file:
            data = _load_data(ts_files[0], var)

            #Extract units string, if available:
            if hasattr(data, 'units'):
                unit_str = data.units
            else:
                unit_str = '--'

            #Check if variable has a vertical coordinate:
            if 'lev' in data.coords or 'ilev' in data.coords:
                print(f"\t   Variable '{var}' has a vertical dimension, "+\
                      "which is currently not supported for the AMWG Table. Skipping...")
                #Skip this variable and move to the next variable in var_list:
                continue

            # we should check if we need to do area averaging:
            if len(data.dims) > 1:
                # flags that we have spatial dimensions
                # Note: that could be 'lev' which should trigger different behavior
                # Note: we should be able to handle (lat, lon) or (ncol,) cases, at least
                data = _spatial_average(data)  # changes data "in place"

            #Add necessary data for RESTOM calcs below
            if var == "FLNT":
                restom_dict[case_name][var] = data
                #Copy units for RESTOM as well:
                restom_units = unit_str
            if var == "FSNT":
                restom_dict[case_name][var] = data

            # In order to get correct statistics, average to annual or seasonal
            data = data.groupby('time.year').mean(dim='time') # this should be fast b/c time series should be in memory
                                                              # NOTE: data will now have a 'year' dimension instead of 'time'
            
            # create a dataframe:
            cols = ['variable', 'unit', 'mean', 'sample size', 'standard dev.',
                    'standard error', '95% CI', 'trend', 'trend p-value']
            
            # These get written to our output file: 
            stats_list = _get_row_vals(data)
            row_values = [var, unit_str] + stats_list

            # Format entries:
            dfentries = {c:[row_values[i]] for i,c in enumerate(cols)}

            # Add entries to Pandas structure:
            df = pd.DataFrame(dfentries)

            # Check if the output CSV file exists,
            # if so, then append to it:
            if output_csv_file.is_file():
                df.to_csv(output_csv_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_csv_file, header=cols, index=False)

        #End of var_list loop
        #--------------------

        if "FSNT" and "FLNT" in var_list:
            #RESTOM Calcs
            var = "RESTOM" #RESTOM = FSNT-FLNT
            print(f"\t - Variable '{var}' being added to table")
            data = restom_dict[case_name]["FSNT"] - restom_dict[case_name]["FLNT"]
            # In order to get correct statistics, average to annual or seasonal
            data = data.groupby('time.year').mean(dim='time') # this should be fast b/c time series should be in memory
                                                                # NOTE: data will now have a 'year' dimension instead of 'time'
            # These get written to our output file: 
            stats_list = _get_row_vals(data) 
            row_values = [var, restom_units] + stats_list
            # col (column) values declared above

            # Format entries:
            dfentries = {c:[row_values[i]] for i,c in enumerate(cols)}

            # Add entries to Pandas structure:
            df = pd.DataFrame(dfentries)

            # Check if the output CSV file exists,
            # if so, then append to it:
            if output_csv_file.is_file():
                df.to_csv(output_csv_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_csv_file, header=cols, index=False)
            #End if

        else:
            #Print message to debug log:
            adf.debug_log("RESTOM not calculated because FSNT and/or FLNT variables not in dataset")
        #End if

        # last step is to add table dataframe to website (if enabled):
        table_df = pd.read_csv(output_csv_file)
        adf.add_website_data(table_df, case_name, case_name, plot_type="Tables")

    #End of model case loop
    #----------------------


    #Notify user that script has ended:
    print("  ...AMWG variable table has been generated successfully.")

    #Check if observations are being comapred to, if so skip table comparison...
    if not adf.get_basic_info("compare_obs"):
        if "CMIP" in baseline_name:
            print("CMIP case detected, skipping comparison table...")
        else:
            #Create comparison table for both cases
            print("\n  Making comparison table...")
            _df_comp_table(adf, output_location, case_names)
            print("  ... Comparison table has been generated successfully")
        #End if
    else:
        print(" Comparison table currently doesn't work with obs, so skipping...")
    #End if


##################
# Helper functions
##################

def _load_data(dataloc, varname):
    import xarray as xr
    ds = xr.open_dataset(dataloc)
    return ds[varname]

#####

def _spatial_average(indata):
    import xarray as xr
    import numpy as np
    import warnings
    assert 'lev' not in indata.coords
    assert 'ilev' not in indata.coords
    if 'lat' in indata.coords:
        weights = np.cos(np.deg2rad(indata.lat))
        weights.name = "weights"
    elif 'ncol' in indata.coords:
        warnings.warn("We need a way to get area variable. Using equal weights.")
        weights = xr.DataArray(1.)
        weights.name = "weights"
    else:
        weights = xr.DataArray(1.)
        weights.name = "weights"
    weighted = indata.weighted(weights)
    # we want to average over all non-time dimensions
    avgdims = [dim for dim in indata.dims if dim != 'time']
    return weighted.mean(dim=avgdims)

#####

def _get_row_vals(data):    
    # Now that data is (time,), we can do our simple stats:
    data_stats = data.copy()
 
    if data_stats.max() > 10000000:
        data_stats = data_stats.where(data_stats < 10000000)
    if data_stats.min() < -10000000:    
        data_stats = data_stats.where(data_stats > -10000000)

    data_mean = data_stats.mean()
    #Conditional Formatting depending on type of float
    if np.abs(data_mean) < 1:
        formatter = ".3g"
    else:
        formatter = ".3f"
  
    data_sample = len(data_stats)
    data_std = data_stats.std()
    data_sem = data_std / data_sample
    data_ci = data_sem * 1.96  # https://en.wikipedia.org/wiki/Standard_error
    data_trend = stats.linregress(data_stats.year, data_stats.values)

    stdev = f'{data_std.data.item() : {formatter}}'
    sem = f'{data_sem.data.item() : {formatter}}'
    ci = f'{data_ci.data.item() : {formatter}}'
    slope_int = f'{data_trend.intercept : {formatter}} + {data_trend.slope : {formatter}} t'
    pval = f'{data_trend.pvalue : {formatter}}'

    return [f'{data_mean:{formatter}}', data_sample, stdev, sem, ci, slope_int, pval]

#####

def _df_comp_table(adf, output_location, case_names):
    import pandas as pd

    output_csv_file_comp = output_location / "amwg_table_comp.csv"

    # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    #This will be for single-case for now (case_names[0]),
    #will need to change to loop as multi-case is introduced
    case = output_location/f"amwg_table_{case_names[0]}.csv"
    baseline = output_location/f"amwg_table_{case_names[-1]}.csv"

    #Read in test case and baseline dataframes:
    df_case = pd.read_csv(case)
    df_base = pd.read_csv(baseline)

    #Create a merged dataframe that contains only the variables
    #contained within both the test case and the baseline:
    df_merge = pd.merge(df_case, df_base, how='inner', on=['variable'])

    #Create the "comparison" dataframe:
    df_comp = pd.DataFrame(dtype=object)
    df_comp[['variable','unit','case']] = df_merge[['variable','unit_x','mean_x']]
    df_comp['baseline'] = df_merge[['mean_y']]
    
    diffs = df_comp['case'].values-df_comp['baseline'].values
    df_comp['diff'] = [f'{i:.3g}' if np.abs(i) < 1 else f'{i:.3f}' for i in diffs]

    #Write the comparison dataframe to a new CSV file:
    cols_comp = ['variable', 'unit', 'test', 'control', 'diff']
    df_comp.to_csv(output_csv_file_comp, header=cols_comp, index=False)

    #Add comparison table dataframe to website (if enabled):
    adf.add_website_data(df_comp, "Case Comparison", case_names[0], plot_type="Tables")

##############
#END OF SCRIPT
