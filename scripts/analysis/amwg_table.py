import numpy as np
import xarray as xr
import sys
from pathlib import Path
import warnings  # use to warn user about missing files.

#Import "special" modules:
try:
    import scipy.stats as stats # for easy linear regression and testing
except ImportError:
    print("Scipy module does not exist in python path, but is needed for amwg_table.")
    print("Please install module, e.g. 'pip install scipy'.")
    sys.exit(1)
#End except

try:
    import pandas as pd
except ImportError:
    print("Pandas module does not exist in python path, but is needed for amwg_table.")
    print("Please install module, e.g. 'pip install pandas'.")
    sys.exit(1)
#End except

#Import ADF-specific modules:
import plotting_functions as pf

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
    var_defaults    -> Dict that has keys that are variable names and values that are plotting preferences/defaults.

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
    msg = "\n  Calculating AMWG variable tables..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Extract needed quantities from ADF object:
    #-----------------------------------------
    var_list     = adf.diag_var_list
    var_defaults = adf.variable_defaults

    #Check if ocean or land fraction exist
    #in the variable list:
    for var in ["OCNFRAC", "LANDFRAC"]:
        if var in var_list:
            #If so, then move them to the front of variable list so
            #that they can be used to mask or vertically interpolate
            #other model variables if need be:
            var_idx = var_list.index(var)
            var_list.pop(var_idx)
            var_list.insert(0,var)
        #End if
    #End if

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    output_locs = adf.plot_location

    #CAM simulation variables (these quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    input_locs = adf.ts_locs["test"]
    input_climo_locs = adf.climo_locs["test"]

    #Grab case years
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    #Check if a baseline simulation is also being used:
    if not adf.get_basic_info("compare_obs"):
        #Extract CAM baseline variaables:
        baseline_name     = adf.get_baseline_info("cam_case_name", required=True)
        input_loc = adf.ts_locs["baseline"]
        input_climo_loc = adf.climo_locs["baseline"]
        input_climo_locs.append(input_climo_loc)

        #Grab baseline years (which may be empty strings if using Obs):
        syear_baseline = adf.climo_yrs["syear_baseline"]
        eyear_baseline = adf.climo_yrs["eyear_baseline"]

        syear_cases.append(syear_baseline)
        eyear_cases.append(eyear_baseline)

        #Convert output location string to a Path object:
        output_location = Path(output_locs[0])
        if not input_loc:
            emsg = "\n  User indicates no time series files will be used."
            emsg += " Looking if table already exisits:"
            print(emsg)

            #if ah:
            #for case_idx, case_name in enumerate(case_names):
            #Create output file name:
            output_csv_file = output_location / f"amwg_table_{baseline_name}.csv"
            if Path(output_csv_file).is_file():
                print(f"\t - AMWG table for '{baseline_name}' exists, adding to website.")
                table_df = pd.read_csv(output_csv_file)
                # last step is to add table dataframe to website (if enabled):
                adf.add_website_data(table_df, baseline_name, baseline_name, plot_type="Tables")
            else:
                print(f"\t - AMWG table for '{baseline_name}' does not exist.")
                print('\t  check here:',output_csv_file,"\n")
            input_locs.append(None)
            pass#return
        else:
            input_locs.append(input_loc)

        case_names.append(baseline_name)

        #Save the baseline to the first case's plots directory:
        output_locs.append(output_location)
    else:
        print("AMWG table doesn't currently work with obs, so obs table won't be created.")
    #End if

    #-----------------------------------------
    #Loop over CAM cases:
    #Initialize list of case name csv files for case comparison check later
    csv_list = []
    for case_idx, case_name in enumerate(case_names):
        syear = syear_cases[case_idx]
        eyear = eyear_cases[case_idx]

        #Convert output location string to a Path object:
        output_location = Path(output_locs[case_idx])

        #Create output file name:
        output_csv_file = output_location / f"amwg_table_{case_name}.csv"

        #Given that this is a final, user-facing analysis, go ahead and re-do it every time:
        if Path(output_csv_file).is_file():
            Path.unlink(output_csv_file)
        #End if

        #Create/reset new variable that potentially stores the re-gridded
        #ocean fraction xarray data-array:
        ocn_frc_da = None

        #Notify user that script has started:
        print(f"\n  Calculating AMWG variable table for '{case_name}'...")
    
        #Loop over CAM output variables:
        for var in var_list:
            #Notify users of variable being added to table:
            print(f"\t - Variable '{var}' being added to table")

            is_climo = False # default to time series
            #Generate input file path:
            if input_locs[case_idx]:
                input_location = Path(input_locs[case_idx])

                filenames = f'{case_name}.*.{var}.*nc'
                files = sorted(input_location.glob(filenames))
            else:
                files = None

            # If no files exist, try to move to next variable. --> Means we can not proceed with this variable, and it'll be problematic later.
            if not files:
                # Try for climo files:
                msg = f"\t    INFO: Time series files for variable '{var}' in {case_name} not found.  Checking on climo files."
                print(msg)
                filenames = f'{case_name}_{var}_climo.nc'
                try_input_location = Path(input_climo_locs[case_idx])
                try_files = sorted(try_input_location.glob(filenames))
                if not try_files:
                    errmsg = f"\t    WARNING: Climo files for variable '{var}' not found.  Script will continue to next variable."
                    print(errmsg)
                    continue
                else:
                    print(f"\t         - User supplied climo files for {var}, will make only global mean (no other stats).")
                    files = try_files
                    input_location = try_input_location
                    is_climo = True
            #End if

            #Check that time series input directory actually exists:
            if not input_location.is_dir():
                errmsg = f"amwg_table: Time series directory '{input_location}' not found.  Script is exiting."
                raise AdfError(errmsg)
            #Write to debug log if enabled:
            adf.debug_log(f"DEBUG: location of files is {str(input_location)}")

            #Load model variable data from file:
            ds = pf.load_dataset(files)

            if not is_climo:
                #Average time dimension over time bounds, if bounds exist:
                if 'time_bnds' in ds:
                    time = ds['time']
                    # NOTE: force `load` here b/c if dask & time is cftime, throws a NotImplementedError:
                    time = xr.DataArray(ds['time_bnds'].load().mean(dim='nbnd').values, dims=time.dims, attrs=time.attrs)
                    ds['time'] = time
                    ds.assign_coords(time=time)
                    ds = xr.decode_cf(ds)

            if len(files) > 1:
                # Slice for years 0500 to 0521
                # Slice using only the 4-digit year
                time_slice = slice(str(syear).zfill(4), str(eyear).zfill(4))
                ds = ds.sel(time=time_slice)
                data = ds[var].compute()
            else:
                data = ds[var]

            #Extract units string, if available:
            if hasattr(data, 'units'):
                unit_str = data.units
            else:
                unit_str = '--'

            #Check if variable has a vertical coordinate:
            if 'lev' in data.coords or 'ilev' in data.coords:
                print(f"\t    WARNING: Variable '{var}' has a vertical dimension, "+\
                      "which is currently not supported for the AMWG Table. Skipping...")
                #Skip this variable and move to the next variable in var_list:
                continue
            #End if

            #Extract defaults for variable:
            var_default_dict = var_defaults.get(var, {})

            #Check if variable should be masked:
            if 'mask' in var_default_dict:
                if var_default_dict['mask'].lower() == 'ocean':
                    #Check if the ocean fraction has already been regridded
                    #and saved:
                    if ocn_frc_da is not None:
                        ofrac = ocn_frc_da
                        # set the bounds of regridded ocnfrac to 0 to 1
                        ofrac = xr.where(ofrac>1,1,ofrac)
                        ofrac = xr.where(ofrac<0,0,ofrac)

                        # apply ocean fraction mask to variable
                        data = pf.mask_land_or_ocean(data, ofrac, use_nan=True)
                    else:
                        print(f"\t    WARNING: OCNFRAC not found, unable to apply mask to '{var}'")
                    #End if
                else:
                    #Currently only an ocean mask is supported, so print warning here:
                    wmsg = "\t    INFO: Currently the only variable mask option is 'ocean',"
                    wmsg += f"not '{var_default_dict['mask'].lower()}'"
                    print(wmsg)
                #End if
            #End if

            #If the variable is ocean fraction, then save the dataset for use later:
            if var == 'OCNFRAC':
                ocn_frc_da = data
            #End if

            # we should check if we need to do area averaging:
            if len(data.dims) > 1:
                # flags that we have spatial dimensions
                # Note: that could be 'lev' which should trigger different behavior
                # Note: we should be able to handle (lat, lon) or (ncol,) cases, at least
                data = pf.spatial_average(data)  # changes data "in place"

            if is_climo:
                data = pf.seasonal_mean(data, season="ANN", is_climo=True)
                #Conditional Formatting depending on type of float
                if np.abs(data) < 1:
                    formatter = ".3g"
                else:
                    formatter = ".3f"
                mean_final = f'{data:{formatter}}'

                # create a dataframe:
                cols = ['variable', 'unit', 'mean', 'sample size', 'standard dev.',
                            'standard error', '95% CI', 'trend', 'trend p-value']
                row_values = [var, unit_str] + [mean_final] + ["-","-","-","-","-","-"]
            else:
                # In order to get correct statistics, average to annual or seasonal
                data = pf.annual_mean(data, whole_years=True, time_name='time')

                # create a dataframe:
                cols = ['variable', 'unit', 'mean', 'sample size', 'standard dev.',
                            'standard error', '95% CI', 'trend', 'trend p-value']
                stats_list = _get_row_vals(data)
                row_values = [var, unit_str] + stats_list
            #End if

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

        # Move RESTOM to top of table (if applicable)
        #--------------------------------------------
        try:
            table_df = pd.read_csv(output_csv_file)
            if 'RESTOM' in table_df['variable'].values:
                table_df = pd.concat([table_df[table_df['variable'] == 'RESTOM'], table_df]).reset_index(drop = True)
                table_df = table_df.drop_duplicates()
                table_df.to_csv(output_csv_file, header=cols, index=False)

            # last step is to add table dataframe to website (if enabled):
            adf.add_website_data(table_df, case_name, case_name, plot_type="Tables")
        except FileNotFoundError:
            print(f"\n\tAMWG table for '{case_name}' not created.\n")
        #End try/except

        #Keep track of case csv files for comparison table check later
        csv_list.extend(sorted(output_location.glob(f"amwg_table_{case_name}.csv")))

    #End of model case loop
    #----------------------

    #Start case comparison tables
    #----------------------------
    #Check if observations are being compared to, if so skip table comparison...
    if not adf.get_basic_info("compare_obs"):
        #Check if all tables were created to compare against, if not, skip table comparison...
        if len(csv_list) != len(case_names):
            print("\tNot enough cases to compare, skipping comparison table...")
        else:
            #Create comparison table for both cases
            print("\n  Making comparison table...")
            _df_comp_table(adf, output_location, case_names)
            print("  ... Comparison table has been generated successfully")
        #End if
    else:
        print(" No comparison table will be generated due to running against obs.")
    #End if

    #Notify user that script has ended:
    print("  ...AMWG variable table(s) have been generated successfully.")


##################
# Helper functions
##################

def _get_row_vals(data):
    # Now that data is (time,), we can do our simple stats:

    data_mean = data.data.mean()
    #Conditional Formatting depending on type of float
    if np.abs(data_mean) < 1:
        formatter = ".3g"
    else:
        formatter = ".3f"

    data_sample = len(data)
    data_std = data.std()
    data_sem = data_std / data_sample
    data_ci = data_sem * 1.96  # https://en.wikipedia.org/wiki/Standard_error
    data_trend = stats.linregress(data.year, data.values)

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