"""Data-access layer: locate and load ADF time series, climo, and regridded files."""
import warnings # use to warn user about missing files
from pathlib import Path

import xarray as xr

import adf_utils as utils
warnings.formatwarning = utils.my_formatwarning

# "reference data"
# It is often just a "baseline case",
# but could be some totally external data (reanalysis or observation or other model)
# When it is another simulation, it gets treated like another "case"
# When it is external data expect:
# - "climo" files (12 monthly climos in the file)
# - one variable per "climo"
# - source can differ for each variable, requires label
# - resolution can differ for each variable, requires regridded file(s)
# - the variable name and units in the file may differ from CAM;
#   use defaults.yaml to set conversion
# - there could be multiple instances of a variable from different sources
#   (e.g. different observations)

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

# NOTE: Standard ADF workflow creates time series files with NCO.
#       Climo files are then generated with create_climo_files.py
#       Since neither of these apply units conversions (add_offset, scale_factor),
#       the methods here default to applying them when loading
#       time series and climo files, using the kwarg apply_scaling.
#       Regridded files are made with regrid_and_vert_interp[_2].py,
#       which uses this module for loading climo files, so will apply
#       scaling.
#       Therefore the default on loading regridded files is to NOT
#       apply scaling.

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
        # set_reference specifies "ref_labels" (called "data_list" in zonal_mean,
        # i.e. the name of the data source)
        self.set_reference()

    def set_reference(self):
        """Set attributes for reference (aka baseline) data location, names, and variables."""
        if self.adf.compare_obs:
            obs = self.adf.var_obs_dict
            self.ref_var_loc = {v: obs[v]['obs_file'] for v in obs}
            self.ref_labels = {v: obs[v]['obs_name'] for v in obs}
            self.ref_var_nam = {v: obs[v]['obs_var'] for v in obs}
            self.ref_case_label = "Obs"
            if not self.adf.var_obs_dict:
                warnings.warn("\t    WARNING: reference is observations, but no "
                              "observations found to plot against.")
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

    # History stream helpers
    #------------------
    @staticmethod
    def _as_hist_str_list(hist_strs):
        """Coerce a hist_str entry (str, list, None, or empty) to a clean list of streams."""
        if not hist_strs:
            return []
        if isinstance(hist_strs, str):
            return [hist_strs]
        return [h for h in hist_strs if h]

    def _hist_strs_for_case(self, case):
        """Ordered history streams configured for a test case (priority order)."""
        test_hist_strs = self.adf.hist_string["test_hist_str"]
        caseindex = (self.case_names).index(case)
        if caseindex < len(test_hist_strs):
            return self._as_hist_str_list(test_hist_strs[caseindex])
        return []

    def _hist_strs_for_reference(self):
        """Ordered history streams configured for the reference/baseline case."""
        return self._as_hist_str_list(self.adf.hist_string["base_hist_str"])


    # Time series files
    #------------------
    # Test case(s)
    def _select_ts_files(self, fils, syr, eyr, field):
        """Narrow time series files to the years wanted, saying so in the log.

        Files are only narrowed when they could not be opened together, so a
        message here means the directory held more than one set for `field`.
        """
        chosen = utils.select_ts_files(fils, syr, eyr)
        if len(chosen) != len(fils):
            msg = f"\t    INFO: '{field}' has {len(fils)} time series files that "
            msg += "cannot be used together, so the "
            msg += f"{len(chosen)} needed for years {syr}-{eyr} were used."
            #Say it on screen as well as in the log: the numbers a user sees
            #change when this happens, and that should not be silent.
            print(msg)
            self.adf.debug_log(
                msg + "  Files used: "
                + ", ".join(str(Path(f).name) for f in chosen))
        #End if
        return chosen

    def get_timeseries_file(self, case, field, hist_str=None):
        """Return list of test time series files.

        If hist_str is given, restrict the search to that history stream
        (time series files are named {case}.{hist_str}.{field}.*.nc).

        The files are narrowed to those needed for the case's climatology
        years, so that a directory holding more than one set for the same
        variable (years 1-20 alongside years 1-40, say) does not produce a
        combined time axis with duplicate times.
        """
        # list of paths (could be multiple cases)
        ts_locs = self.adf.get_cam_info("cam_ts_loc", required=True)
        caseindex = (self.case_names).index(case)
        ts_loc = Path(ts_locs[caseindex])
        if hist_str:
            ts_filenames = f'{case}.{hist_str}.{field}.*nc'
        else:
            ts_filenames = f'{case}.*.{field}.*nc'
        fils = utils.find_ts_files(ts_loc, ts_filenames)
        climo_yrs = self.adf.climo_yrs
        return self._select_ts_files(fils, climo_yrs["syears"][caseindex],
                                     climo_yrs["eyears"][caseindex], field)

    # Reference case (baseline/obs)
    def get_ref_timeseries_file(self, field, hist_str=None):
        """Return list of reference time series files.

        If hist_str is given, restrict the search to that history stream.
        Narrowed to the baseline's climatology years, as for the test cases.
        """
        if self.adf.compare_obs:
            warnings.warn("\t    WARNING: ADF does not currently expect "
                          "observational time series files.")
            return None
        ts_loc = Path(self.adf.get_baseline_info("cam_ts_loc", required=True))
        if hist_str:
            ts_filenames = f'{self.ref_case_label}.{hist_str}.{field}.*nc'
        else:
            ts_filenames = f'{self.ref_case_label}.*.{field}.*nc'
        fils = utils.find_ts_files(ts_loc, ts_filenames)
        climo_yrs = self.adf.climo_yrs
        return self._select_ts_files(fils, climo_yrs["syear_baseline"],
                                     climo_yrs["eyear_baseline"], field)


    def load_timeseries_dataset(self, fils):
        """Return DataSet from time series file(s) and assign time to midpoint of interval"""
        if len(fils) == 0:
            warnings.warn("\t    WARNING: Input file list is empty.")
            return None
        if len(fils) > 1:
            ds = xr.open_mfdataset(fils, decode_times=False)
        else:
            sfil = str(fils[0])
            if not Path(sfil).is_file():
                warnings.warn(f"\t    WARNING: Expecting to find file: {sfil}")
                return None
            ds = xr.open_dataset(sfil, decode_times=False)
        if ds is None:
            warnings.warn("\t    WARNING: invalid data on load_dataset")
            return ds
        # Assign time to the midpoint of the interval each step covers.  The
        # shared helper reads the bounds the file names for itself, so a file
        # calling them something other than 'time_bnds' is handled too, and it
        # hands back the dataset it was given when the file records no bounds:
        fixed = utils.use_time_bounds_midpoint(ds)
        if fixed is ds:
            warnings.warn("\t    INFO: Timeseries file does not have time bounds info.")
        # End if
        return xr.decode_cf(fixed)

    def load_timeseries_da(self, case, variablename):
        """Return DataArray from time series file(s).
           Uses defaults file to convert units.
        """
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_timeseries_file(case, variablename)
        if not fils:
            warnings.warn("\t    WARNING: Did not find case time series file(s), "
                          f"variable: {variablename}")
            return None
        return self.load_da(
            fils,
            variablename,
            use_time_bounds=True,
            add_offset=add_offset,
            scale_factor=scale_factor,
        )

    def load_reference_timeseries_da(self, field, apply_scaling=True):
        """Return a DataArray time series to be used as reference
          (aka baseline) for variable field.

          apply_scaling: bool
            If True, apply add_offset and scale_factor to data (if present).
        """
        fils = self.get_ref_timeseries_file(field)
        if not fils:
            warnings.warn("\t    WARNING: Did not find reference time series file(s), "
                          f"variable: {field}")
            return None
        #Change the variable name from CAM standard to what is
        # listed in variable defaults for this observation field
        if self.adf.compare_obs:
            field = self.ref_var_nam[field]
            add_offset = 0
            scale_factor = 1
        else:
            add_offset, scale_factor = self.get_value_converters(self.ref_case_label, field)

        if not apply_scaling:
            add_offset = 0
            scale_factor = 1

        return self.load_da(
            fils,
            field,
            use_time_bounds=True,
            add_offset=add_offset,
            scale_factor=scale_factor,
        )


    #------------------


    # Climatology files
    #------------------

    # Test case(s)
    def load_climo_ds(self, case, variablename):
        """Return Dataset from climo file; applies scale factor and offset to `variablename`."""
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_climo_file(case, variablename)
        if not fils:
            warnings.warn("\t    WARNING: Did not find climo file for case: "
                          f"{case}, variable: {variablename}")
            return None
        ds = self.load_dataset(fils)
        if ds is None:
            return None
        # xarray arithmetic drops attrs, so carry them across by hand -- otherwise
        # the regridded files lose 'units' and the plotting scripts KeyError on it.
        attrs = ds[variablename].attrs.copy()
        ds[variablename] = ds[variablename] * scale_factor + add_offset
        ds[variablename].attrs = attrs
        if scale_factor != 1 or add_offset != 0:
            new_unit = self.adf.variable_defaults.get(variablename, {}).get("new_unit")
            if new_unit:
                ds[variablename].attrs['units'] = new_unit
            # Stamp on any conversion, not only one that renames the units:
            # TAUX/TAUY are scaled by -1 with no "new_unit", and an unstamped
            # file is indistinguishable from one that was never converted.
            # int, not bool: netCDF4 cannot store a Python bool as an attribute
            ds[variablename].attrs['transformed'] = 1
        return ds

    def load_climo_da(self, case, variablename, apply_scaling=True):
        """Return DataArray from climo file"""
        if not apply_scaling:
            add_offset = 0
            scale_factor = 1
        else:
            add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_climo_file(case, variablename)
        return self.load_da(fils, variablename, add_offset=add_offset, scale_factor=scale_factor)


    def load_climo_file(self, case, variablename):
        """Return Dataset for climo of variablename"""
        fils = self.get_climo_file(case, variablename)
        if not fils:
            warnings.warn("\t    WARNING: Did not find climo file for variable: "
                          f"{variablename}. Will try to skip.")
            return None
        return self.load_dataset(fils)


    def get_climo_file(self, case, variablename):
        """Retrieve the climo file path(s) for variablename for a specific case.

        Climo files created by recent ADF versions include the history stream
        in the name ({case}_{hist_str}_{var}_climo.nc). Search for those first,
        in the case's configured hist_str priority order, then fall back to the
        older stream-less convention ({case}_{var}_climo.nc) so that
        pre-existing climo files still work.
        """
        # list of paths (could be multiple cases)
        a = self.adf.get_cam_info("cam_climo_loc", required=True)
        caseindex = (self.case_names).index(case) # the entry for specified case
        model_cl_loc = Path(a[caseindex])
        for hist_str in self._hist_strs_for_case(case):
            fils = sorted(model_cl_loc.glob(f"{case}_{hist_str}_{variablename}_climo.nc"))
            if fils:
                return fils
        # Fall back to older naming (no hist_str) for pre-existing climo files:
        return sorted(model_cl_loc.glob(f"{case}_{variablename}_climo.nc"))


    # Reference case (baseline/obs)
    def load_reference_climo_ds(self, case, variablename, apply_scaling=True):
        """Return Dataset from reference climo file.

        Applies the scale factor and offset to ``variablename``.
        """
        add_offset, scale_factor = self.get_value_converters(case, variablename)
        fils = self.get_reference_climo_file(variablename)
        if not fils:
            warnings.warn("\t    WARNING: Did not find reference climo file for "
                          f"variable: {variablename}")
            return None
        ds = self.load_dataset(fils)
        if ds is None:
            return None
        vname = self.ref_var_nam[variablename]  # name of variable in the reference data
        # Check if already transformed (via attribute or units)
        new_unit = self.adf.variable_defaults.get(variablename, {}).get('new_unit')
        unit_match = new_unit is not None and ds[vname].attrs.get('units') == new_unit
        if ds[vname].attrs.get('transformed', False) or unit_match:
            apply_scaling = False
        if not apply_scaling:
            add_offset = 0
            scale_factor = 1

        attrs = ds[vname].attrs.copy()
        ds[vname] = ds[vname] * scale_factor + add_offset
        ds[vname].attrs = attrs
        if scale_factor != 1 or add_offset != 0:
            # int, not bool: netCDF4 cannot store a Python bool as an attribute
            ds[vname].attrs['transformed'] = 1
        return ds

    def load_reference_climo_da(self, case, variablename, apply_scaling=True):
        """Return DataArray from reference (aka baseline) climo file"""
        fils = self.get_reference_climo_file(variablename)
        vname = self.ref_var_nam[variablename]
        if not apply_scaling:
            add_offset = 0
            scale_factor = 1
        else:
            add_offset, scale_factor = self.get_value_converters(case, variablename)
        return self.load_da(fils, vname, add_offset=add_offset, scale_factor=scale_factor)

    def get_reference_climo_file(self, var):
        """Return a list of files to be used as reference (aka baseline) for variable var."""
        if self.adf.compare_obs:
            fils = self.ref_var_loc.get(var, None)
            return [fils] if fils is not None else []
        ref_loc = self.adf.get_baseline_info("cam_climo_loc")
        # Prefer stream-aware naming (in priority order), then fall back to the
        # older convention so pre-existing baseline climo files still work:
        for hist_str in self._hist_strs_for_reference():
            fils = sorted(Path(ref_loc).glob(f"{self.ref_case_label}_{hist_str}_{var}_climo.nc"))
            if fils:
                return fils
        # NOTE: originally had this looking for *_baseline.nc
        return sorted(Path(ref_loc).glob(f"{self.ref_case_label}_{var}_climo.nc"))

    #------------------


    # Regridded files
    #------------------

    # Test case(s)
    def get_regrid_file(self, case, field):
        """Return list of test regridded files"""
        model_rg_loc = Path(self.model_rgrid_loc)
        # rlbl = "reference label" = name of the reference data that defines the target grid
        rlbl = self.ref_labels[field]
        return sorted(model_rg_loc.glob(f"{rlbl}_{case}_{field}_regridded.nc"))


    def load_regrid_dataset(self, case, field):
        """Return a data set to be used as reference (aka baseline) for variable field."""
        fils = self.get_regrid_file(case, field)
        if not fils:
            warnings.warn("\t    WARNING: Did not find regrid file(s) for case: "
                          f"{case}, variable: {field}")
            return None
        return self.load_dataset(fils)


    def load_regrid_da(self, case, field, apply_scaling=None):
        """Return a data array of regridded data for case and variable field.

        Parameters
        ----------
        case : str
            Name of the test case.
        field : str
            ADF name of the variable to load.
        apply_scaling : bool, optional
            Whether to apply ``add_offset``/``scale_factor``. The default,
            ``None``, decides from the file itself -- see
            :meth:`_regrid_converters`.
        """
        fils = self.get_regrid_file(case, field)
        if not fils:
            warnings.warn("\t    WARNING: Did not find regrid file(s) for case: "
                          f"{case}, variable: {field}")
            return None
        add_offset, scale_factor = self._regrid_converters(fils, field, case, field,
                                                           apply_scaling)
        return self.load_da(fils, field, add_offset=add_offset, scale_factor=scale_factor)


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
            model_rg_loc = Path(self.model_rgrid_loc)
            fils = sorted(model_rg_loc.glob(f"{case}_{field}_baseline.nc"))
        return fils


    def load_reference_regrid_dataset(self, case, field):
        """Return a data set to be used as reference (aka baseline) for variable field."""
        fils = self.get_ref_regrid_file(case, field)
        if not fils:
            warnings.warn("\t    WARNING: Did not find regridded file(s) for case: "
                          f"{case}, variable: {field}")
            return None
        return self.load_dataset(fils)


    def load_reference_regrid_da(self, case, field, apply_scaling=None):
        """Return a data array to be used as reference (aka baseline) for variable field.

        Parameters
        ----------
        case : str
            Name of the reference (baseline or observational) data source.
        field : str
            ADF name of the variable to load.
        apply_scaling : bool, optional
            Whether to apply ``add_offset``/``scale_factor``. The default,
            ``None``, decides from the file itself -- see
            :meth:`_regrid_converters`. Observation files are never written by
            the regridding stage, so they are always converted here.
        """
        fils = self.get_ref_regrid_file(case, field)
        if not fils:
            warnings.warn("\t    WARNING: Did not find regridded file(s) for case: "
                          f"{case}, variable: {field}")
            return None
        #Change the variable name from CAM standard to what is
        # listed in variable defaults for this observation field
        file_field = self.ref_var_nam[field] if self.adf.compare_obs else field
        add_offset, scale_factor = self._regrid_converters(fils, file_field, case, field,
                                                           apply_scaling)
        return self.load_da(fils, file_field, add_offset=add_offset, scale_factor=scale_factor)

    def _regrid_converters(self, fils, file_field, case, field, apply_scaling):
        """Return the (add_offset, scale_factor) to use for a regridded file.

        The regridding stage applies the variable-defaults conversion when it
        writes, and :meth:`load_da` stamps ``transformed`` on what it wrote, so
        a file this ADF produced needs no further conversion. Two kinds of file
        carry no stamp and still do: observation files, which the regridder
        never touches, and regridded files left in ``cam_regrid_loc`` by an
        older ADF, where the conversion happened at plot time instead. The
        shipped default is ``cam_overwrite_regrid: false``, so those older
        files are reused rather than rewritten -- converting unconditionally
        would double-scale the new ones, and not converting at all would plot
        the old ones in raw units with no error anywhere.
        """
        if apply_scaling is False:
            return 0, 1
        add_offset, scale_factor = self.get_value_converters(case, field)
        if apply_scaling or (scale_factor == 1 and add_offset == 0):
            return add_offset, scale_factor
        ds = self.load_dataset(fils)
        if ds is not None and ds[file_field].attrs.get('transformed', 0):
            return 0, 1
        return add_offset, scale_factor

    #---------------------------
    # DataSet and DataArray load
    #---------------------------
    def load_dataset(self, fils, use_time_bounds=False):
        """Return xarray DataSet from file(s).

        `use_time_bounds` moves the time coordinate to the midpoint of the
        interval each step covers, which is what a time series wants.  It is
        off by default: climatology and regridded files carry a time
        coordinate of month numbers, and turning that into dates would change
        the files the ADF writes and reads back.
        """
        if len(fils) == 0:
            warnings.warn("\t    WARNING: Input file list is empty.")
            return None
        if len(fils) > 1:
            ds = xr.open_mfdataset(fils, combine='by_coords')
        else:
            sfil = str(fils[0])
            if not Path(sfil).is_file():
                warnings.warn(f"\t    WARNING: Expecting to find file: {sfil}")
                return None
            ds = xr.open_dataset(sfil)
        if ds is None:
            warnings.warn("\t    WARNING: invalid data on load_dataset")
            return ds
        if use_time_bounds:
            # Time stamps that name one end of an averaging interval put steps
            # in the wrong year, so use what the file records about it:
            ds = utils.use_time_bounds_midpoint(ds)
        # End if
        return ds

    def load_da(self, fils, variablename, use_time_bounds=False, **kwargs):
        """Return xarray DataArray from file(s) w/ optional scale factor, offset, new units.

        `use_time_bounds` is passed to `load_dataset`; see there.
        """
        ds = self.load_dataset(fils, use_time_bounds=use_time_bounds)
        if ds is None:
            warnings.warn(f"\t    WARNING: Load failed for {variablename}")
            return None
        da = ds[variablename].squeeze()
        scale_factor = kwargs.get('scale_factor', 1)
        add_offset = kwargs.get('add_offset', 0)
        attrs = da.attrs.copy()
        da = da * scale_factor + add_offset
        da.attrs = attrs

        if scale_factor != 1 or add_offset != 0:
            new_unit = self.adf.variable_defaults.get(variablename, {}).get("new_unit")
            if new_unit:
                da.attrs['units'] = new_unit
            # Stamp on any conversion, not only one that renames the units --
            # see load_climo_ds.
            # int, not bool: netCDF4 cannot store a Python bool as an attribute
            da.attrs['transformed'] = 1
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
