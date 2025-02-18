import sys, builtins

from pathlib import Path
import xarray as xr

import warnings # use to warn user about missing files

def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

# "reference data"
# It is often just a "baseline case", 
# but could be some totally external data (reanalysis or observation or other model)
# When it is another simulation, it gets treated like another "case"
# When it is external data expect:
# - "climo" files (12 monthly climos in the file)
# - one variable per "climo"
# - source can differ for each variable, requires label
# - resolution can differ for each variable, requires regridded file(s)
# - the variable name and units in the file may differ from CAM; use defaults.yaml to set conversion
# - there could be multiple instances of a variable from different sources (e.g. different observations)

# NOTE: the last item (multiple instances of a variable) is not allowed in AdfObs.var_obs_dict
#       Since ADF is not able to handle this case, for now it is excluded the AdfData class.

# NOTE: To make the "baseline case" vs "external data" cases as similar as possible,
#       below construct the "baseline case" version to be similar to "external data".
#       - provide a dictionary of (variable: file-path)
#         + For external data, that dictionay is from AdfObs.var_obs_dict,
#           which provides a dict of all the available variables. 
#         + For reference simulation, look for files that match the diag_var_list

# NOTE: There is currently a "base_nickname" allowed from AdfInfo. 
#       Set AdfData.ref_nickname to that.
#       Could be altered from "Obs" to be the data source label.

class AdfData:
    """A class instantiated with an AdfDiag object. 
       Methods provide means to load data. 
       This class does not interact with plotting, 
       just provides access to data locations and loading data.

       A future need is to add some kind of frequency/sampling
       parameters to allow for non-h0 files. 

    """
    def __init__(self, adfobj):
        self.adf = adfobj  # provides quick access to the AdfDiag object
        # paths 
        self.model_rgrid_loc = adfobj.get_basic_info("cam_regrid_loc", required=True)

        # variables (and info for unit transform)
        # use self.adf.diag_var_list and self.adf.self.adf.variable_defaults

        # case names and nicknames
        self.case_names = adfobj.get_cam_info("cam_case_name", required=True)
        self.test_nicknames = adfobj.case_nicknames["test_nicknames"]
        self.base_nickname = adfobj.case_nicknames["base_nickname"]
        self.ref_nickname = self.base_nickname

        # define reference data
        self.set_reference() # specify "ref_labels" -> called "data_list" in zonal_mean (name of data source)

    def set_reference(self):
        """Set attributes for reference (aka baseline) data location, names, and variables."""
        if self.adf.compare_obs:
            self.ref_var_loc = {v: self.adf.var_obs_dict[v]['obs_file'] for v in self.adf.var_obs_dict}
            self.ref_labels = {v: self.adf.var_obs_dict[v]['obs_name'] for v in self.adf.var_obs_dict}
            self.ref_var_nam = {v: self.adf.var_obs_dict[v]['obs_var'] for v in self.adf.var_obs_dict}
            self.ref_case_label = "Obs"
            if not self.adf.var_obs_dict:
                warnings.warn("\t    WARNING: reference is observations, but no observations found to plot against.")
        else:
            self.ref_var_loc = {}
            self.ref_var_nam = {}
            self.ref_labels = {}
            # when using a reference simulation, allow a "special" attribute with the case name:
            self.ref_case_label = self.adf.get_baseline_info("cam_case_name", required=True)
            for v in self.adf.diag_var_list:
                self.ref_var_nam[v] = v
                self.ref_labels[v] = self.adf.get_baseline_info("cam_case_name", required=True)
                f = self.get_reference_climo_file(v)
                if f:
                    self.ref_var_loc[v] = f

    def set_ref_var_loc(self):
        """Set reference climo file locations"""
        for v in self.adf.diag_var_list:
            f = self.get_reference_climo_file(v)
            self.ref_var_loc[v] = f

    
    # Time series files
    #------------------
    # Test case(s)
    def get_timeseries_file(self, case, field):
        """Return list of test time series files"""
        #ts_locs = self.adf.get_cam_info("cam_ts_loc", required=True) # list of paths (could be multiple cases)
        ts_locs = self.adf.ts_locs["test"]
        caseindex = (self.case_names).index(case)
        ts_loc = Path(ts_locs[caseindex])
        ts_filenames = f'{case}.*.{field}.*nc'
        ts_files = sorted(ts_loc.glob(ts_filenames))
        return ts_files

    # Reference case (baseline/obs)
    def get_ref_timeseries_file(self, field):
        """Return list of reference time series files"""
        if self.adf.compare_obs:
            warnings.warn("\t    WARNING: ADF does not currently expect observational time series files.")
            return None
        else:
            #ts_loc = Path(self.adf.get_baseline_info("cam_ts_loc", required=True))
            ts_loc = Path(self.adf.ts_locs["baseline"])
            ts_filenames = f'{self.ref_case_label}.*.{field}.*nc'
            ts_files = sorted(ts_loc.glob(ts_filenames))
            return ts_files

    '''# NOT CURRENTLY USED ANYWHERE
    def load_timeseries_dataset(self, fils, syr, eyr):
        """Return DataSet from time series file(s) and assign time to midpoint of interval"""
        if (len(fils) == 0):
            warnings.warn("\t    WARNING: Input file list is empty.")
            return None
        elif (len(fils) > 1):
            ds = xr.open_mfdataset(fils, decode_times=False)
        else:
            sfil = str(fils[0])
            if not Path(sfil).is_file():
                warnings.warn(f"\t    WARNING: Expecting to find file: {sfil}")
                return None
            ds = xr.open_dataset(sfil, decode_times=False)
        if ds is None:
            warnings.warn(f"\t    WARNING: invalid data on load_dataset")
        # assign time to midpoint of interval (even if it is already)
        if 'time_bnds' in ds:
            t = ds['time_bnds'].mean(dim='nbnd')
            t.attrs = ds['time'].attrs
            ds = ds.assign_coords({'time':t})
        elif 'time_bounds' in ds:
            t = ds['time_bounds'].mean(dim='nbnd')
            t.attrs = ds['time'].attrs
            ds = ds.assign_coords({'time':t})
        else:
            warnings.warn("\t    INFO: Timeseries file does not have time bounds info.")
        print("\n\n",ds,"\n\n")
        """if ds is not None:
            #Extract data subset using provided year bounds:
            #tslice = get_time_slice_by_year(ds.time, int(syr), int(eyr))
            #ds = ds.isel(time=tslice)
            
            #Extract data subset using provided year bounds:
            tslice = self.get_time_slice_by_year(ds.time, int(syr), int(eyr))
            ds = ds.isel(time=tslice)
            #Retrieve the actual time values from the slice
            actual_time_values = ds.time.values

            print("Checking to make sure 'cam_ts_data' is being sliced in the time dimension correctly: ",actual_time_values)"""
        
        return xr.decode_cf(ds)'''


    def load_timeseries_da(self, case, variablename, syr, eyr):
        """Return DataArray from time series file(s).
           Uses defaults file to convert units.
        """
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_timeseries_file(case, variablename)
        return self.load_da(fils, variablename, syr, eyr, type="timeseries", add_offset=add_offset, scale_factor=scale_factor)
    
    def load_reference_timeseries_da(self, field, syr, eyr):
        """Return a DataArray time series to be used as reference 
          (aka baseline) for variable field.
        """
        fils = self.get_ref_timeseries_file(field)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find reference time series file(s), variable: {field}")
            return None
        #Change the variable name from CAM standard to what is
        # listed in variable defaults for this observation field
        if self.adf.compare_obs:
            field = self.ref_var_nam[field]
            add_offset = 0
            scale_factor = 1
        else:
            add_offset, scale_factor = self.get_value_converters(self.ref_case_label, field)

        return self.load_da(fils, field, syr, eyr, type="timeseries", add_offset=add_offset, scale_factor=scale_factor)


    #------------------


    # Climatology files
    #------------------

    # Test case(s)
    def get_climo_file(self, case, variablename):
        """Retrieve the climo file path(s) for variablename for a specific case."""
        #a = self.adf.get_cam_info("cam_climo_loc", required=True) # list of paths (could be multiple cases)
        climo_locs = self.adf.climo_locs["test"]
        caseindex = (self.case_names).index(case) # the entry for specified case
        model_cl_loc = Path(climo_locs[caseindex])
        return sorted(model_cl_loc.glob(f"{case}_{variablename}_climo.nc"))


    def load_climo_da(self, case, variablename, syr, eyr):
        """Return DataArray from climo file"""
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_climo_file(case, variablename)
        return self.load_da(fils, variablename, syr, eyr, add_offset=add_offset, scale_factor=scale_factor)

    '''# NOT USED ANYWHERE CURRENTLY
    def load_climo_dataset(self, case, variablename, syr, eyr):
        """Return Dataset for climo of variablename"""
        fils = self.get_climo_file(case, variablename)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find climo file for variable: {variablename}. Will try to skip.")
            return None
        return self.load_dataset(fils, syr, eyr)'''


    # Reference case (baseline/obs)
    def get_reference_climo_file(self, var):
        """Return a list of files to be used as reference (aka baseline) for variable var."""
        if self.adf.compare_obs:
            fils = self.ref_var_loc.get(var, None)
            return [fils] if fils is not None else None
        #ref_loc = self.adf.get_baseline_info("cam_climo_loc")
        ref_loc = Path(self.adf.climo_locs["baseline"])
        # NOTE: originally had this looking for *_baseline.nc
        fils = sorted(ref_loc.glob(f"{self.ref_case_label}_{var}_climo.nc"))
        if fils:
            return fils
        return []#None
    

    def load_reference_climo_da(self, case, variablename, syr, eyr):
        """Return DataArray from reference (aka baseline) climo file"""
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_reference_climo_file(variablename)
        return self.load_da(fils, variablename, syr, eyr, add_offset=add_offset, scale_factor=scale_factor)

    #------------------

    
    # Regridded files
    #------------------

    # Test case(s)
    def get_regrid_file(self, case, field):
        """Return list of test regridded files"""
        model_rg_loc = Path(self.adf.get_basic_info("cam_regrid_loc", required=True))
        rlbl = self.ref_labels[field]  # rlbl = "reference label" = the name of the reference data that defines target grid
        return sorted(model_rg_loc.glob(f"{rlbl}_{case}_{field}_regridded.nc"))

    '''# NOT USED ANYWHERE CURRENTLY
    def load_regrid_dataset(self, case, field, syr, eyr):
        """Return a data set to be used as reference (aka baseline) for variable field."""
        fils = self.get_regrid_file(case, field)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find regrid file(s) for case: {case}, variable: {field}")
            return None
        return self.load_dataset(fils, syr, eyr)'''

    
    def load_regrid_da(self, case, field, syr, eyr):
        """Return a data array to be used as reference (aka baseline) for variable field."""
        add_offset, scale_factor = self.get_value_converters(case, field)
        fils = self.get_regrid_file(case, field)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find regrid file(s) for case: {case}, variable: {field}")
            return None
        return self.load_da(fils, field, syr, eyr, add_offset=add_offset, scale_factor=scale_factor)


    # Reference case (baseline/obs)
    def get_ref_regrid_file(self, case, field):
        """Return list of reference regridded files"""
        if self.adf.compare_obs:
            obs_loc = self.ref_var_loc.get(field, None)
            if obs_loc:
                fils = [str(obs_loc)]
            else:
                fils = []
        else:
            model_rg_loc = Path(self.adf.get_basic_info("cam_regrid_loc", required=True))
            fils = sorted(model_rg_loc.glob(f"{case}_{field}_baseline.nc"))
        return fils

    '''# NOT USED ANYWHERE CURRENTLY
    def load_reference_regrid_dataset(self, case, field, syr, eyr):
        """Return a data set to be used as reference (aka baseline) for variable field."""
        fils = self.get_ref_regrid_file(case, field)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find regridded file(s) for case: {case}, variable: {field}")
            return None
        return self.load_dataset(fils, syr, eyr)'''

    
    def load_reference_regrid_da(self, case, field, syr, eyr):
        """Return a data array to be used as reference (aka baseline) for variable field."""
        add_offset, scale_factor = self.get_value_converters(case, field)
        fils = self.get_ref_regrid_file(case, field)
        if not fils:
            warnings.warn(f"\t    WARNING: Did not find regridded file(s) for case: {case}, variable: {field}")
            return None
        #Change the variable name from CAM standard to what is
        # listed in variable defaults for this observation field
        if self.adf.compare_obs:
            field = self.ref_var_nam[field]
        return self.load_da(fils, field, syr, eyr, add_offset=add_offset, scale_factor=scale_factor)

    #------------------


    # DataSet and DataArray load
    #---------------------------

    # Load DataSet
    def load_dataset(self, fils):
        """Return xarray DataSet from file(s)"""
        if (len(fils) == 0):
            warnings.warn("\t    WARNING: Input file list is empty.")
            return None
        elif (len(fils) > 1):
            ds = xr.open_mfdataset(fils, combine='by_coords')
        else:
            sfil = str(fils[0])
            if not Path(sfil).is_file():
                warnings.warn(f"\t    WARNING: Expecting to find file: {sfil}")
                return None
            ds = xr.open_dataset(sfil)
        if ds is None:
            warnings.warn(f"\t    WARNING: invalid data on load_dataset")
        """else:
            #Extract data subset using provided year bounds:
            tslice = self.get_time_slice_by_year(ds.time, int(syr), int(eyr))
            ds = ds.isel(time=tslice)
            #Retrieve the actual time values from the slice
            actual_time_values = ds.time.values

            msg = "Checking to make sure dataset is being sliced in the time dimension correctly: ",actual_time_values
            print(msg)
            self.debug_log(msg)"""

        return ds

    # Load DataArray
    def load_da(self, fils, variablename, syr, eyr, **kwargs):
        """Return xarray DataArray from files(s) w/ optional scale factor, offset, and/or new units"""
        ds = self.load_dataset(fils)
        if ds is None:
            warnings.warn(f"\t    WARNING: Load failed for {variablename}")
            return None
        da = (ds[variablename]).squeeze()
        type = kwargs.get('type', None)
        if type == "timeseries":
            #Extract data subset using provided year bounds:
            tslice = self.get_time_slice_by_year(da.time, int(syr), int(eyr))
            da = da.isel(time=tslice)
            #Retrieve the actual time values from the slice
            actual_time_values = da.time.values
            msg = "Checking to make sure dataarray is being sliced in the time dimension correctly: ",actual_time_values
            self.debug_log(msg)
        scale_factor = kwargs.get('scale_factor', 1)
        add_offset = kwargs.get('add_offset', 0)
        da = da * scale_factor + add_offset
        if variablename in self.adf.variable_defaults:
            vres = self.adf.variable_defaults[variablename]
            da.attrs['units'] = vres.get("new_unit", da.attrs.get('units', 'none'))
        else:
            da.attrs['units'] = 'none'
        return da

    # Get variable conversion defaults, if applicable
    def get_value_converters(self, case, variablename):
        """
        Get variable defaults if applicable
        
           - This is to get any scale factors or off-sets

        Returns
        -------
           add_offset - int/float
           scale_factor - int/float
        """
        add_offset = 0
        scale_factor = 1
        res = self.adf.variable_defaults
        if variablename in res:
            vres = res[variablename]
            if variablename in self.ref_labels:
                if (case == self.ref_labels[variablename]) and (self.adf.compare_obs):
                    scale_factor = vres.get("obs_scale_factor",1)
                    add_offset = vres.get("obs_add_offset", 0)
                else:
                    scale_factor = vres.get("scale_factor",1)
                    add_offset = vres.get("add_offset", 0)
        return add_offset, scale_factor

    #------------------

    def get_time_slice_by_year(self, time, startyear, endyear):
        import numpy as np
        if not hasattr(time, 'dt'):
            print("Warning: get_time_slice_by_year requires the `time` parameter to be an xarray time coordinate with a dt accessor. Returning generic slice (which will probably fail).")
            return slice(startyear, endyear)
        start_time_index = np.argwhere((time.dt.year >= startyear).values).flatten().min()
        end_time_index = np.argwhere((time.dt.year <= endyear).values).flatten().max()
        return slice(start_time_index, end_time_index+1)











    '''
    def set_warning_filter(enable=True):
        """Enable or disable filtering of print statements containing 'WARNING'."""
        original_print = builtins.print

        def filtered_print(*args, **kwargs):
            message = " ".join(map(str, args))
            if enable and "WARNING" in message:
                return  # Skip printing warnings
            original_print(*args, **kwargs)

        builtins.print = filtered_print if enable else original_print


    class SuppressWarningsPrint:
        """Context manager to suppress print statements containing specific keywords."""
        
        def __init__(self, suppress=True, keywords=None):
            """
            Parameters:
            - suppress (bool): Whether to suppress matching print statements.
            - keywords (list): List of keywords to filter (default: ["INFO", "WARNING", "ERROR"]).
            """
            self.suppress = suppress
            self.original_print = print  # Store the original print function
            self.keywords = keywords if keywords is not None else ["INFO", "WARNING", "ERROR"]

        def filtered_print(self, *args, **kwargs):
            """Custom print function that filters out messages containing specified keywords."""
            message = " ".join(map(str, args))
            if not self.suppress or not any(keyword in message for keyword in self.keywords):
                self.original_print(*args, **kwargs)

        def __enter__(self):
            """Override the print function globally."""
            if self.suppress:
                sys.modules['builtins'].print = self.filtered_print

        def __exit__(self, exc_type, exc_value, traceback):
            """Restore the original print function."""
            sys.modules['builtins'].print = self.original_print
    '''
