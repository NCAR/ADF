"""
Information/Parameter (Info) class for
the Atmospheric Diagnostics Framework (ADF).

This class inherits from the AdfConfig class.
Currently this class does four things:

1.  Initializes an instance of AdfConfig.

2.  Checks for the three, required upper-level
    dictionaries specified in the config file,
    and makes copies where the variables have
    been expanded.

3.  Extract values for "compare_obs", "diag_var_list",
    and "plot_location", and provide properties to
    access these values to the rest of ADF.

4.  Set "num_procs" variable, and provde num_procs
    property to the rest of ADF.

This class also provide methods for extracting
variables from the standard, expanded config
dictionaries.
"""

# ++++++++++++++++++++++++++++++
# Import standard python modules
# ++++++++++++++++++++++++++++++

from pathlib import Path
import copy
import os
import re
import getpass

# +++++++++++++++++++++++++++++++++++++++++++++++++
# import non-standard python modules, including ADF
# +++++++++++++++++++++++++++++++++++++++++++++++++

# pylint: disable=unused-import
import numpy as np
import xarray as xr

# pylint: enable=unused-import

# ADF modules:
import adf_utils as utils
from adf_config import AdfConfig
from adf_base import AdfError


def _warn_on_long_climo_range(syr, eyr):
    """Tell the user when the year range worked out from the files is large."""
    if eyr - syr >= 100:
        msg = f"WARNING: the found climo year range is large: {eyr-syr} years, "
        msg += "this may take a long time!"
        print(msg)
    # End if


# Dimensions that make a history file variable a field the ADF can plot.  A
# history file also holds coefficients, counters and coordinate bookkeeping
# (hyam, date, nsteph, gw ...), none of which are diagnostics, and all of which
# lack either a time dimension or a horizontal one:
_HORIZONTAL_DIMS = {"lat", "lon", "ncol", "lndgrid"}


def variables_in_case_files(case_name, hist_loc, hist_strs, ts_loc, ts_done):
    """
    Report the variables available for one case, for "diag_var_list: all".

    Parameters
    ----------
    case_name : str
        name of the case
    hist_loc : str or Path or None
        directory holding the case's history files
    hist_strs : str or list or None
        the case's configured history stream(s)
    ts_loc : str or Path or None
        directory holding pre-made time series files, when there are any
    ts_done : bool
        whether the time series files are pre-made

    Returns
    -------
    set of str
        variable names found in the case's input files

    Notes
    -----
    Pre-made time series hold one variable per file and carry its name in the
    file name, so they are read from the names rather than by opening every
    file.  History files are read from the first file of each stream, which is
    the same assumption the time series step already makes: that every file in
    a stream holds the same variables.
    """
    found = set()

    # An unconfigured stream leaves the search unanchored, which finds the
    # files whichever stream they are in:
    hist_str_list = utils.as_hist_str_list(hist_strs) or [""]

    if ts_done:
        if not ts_loc:
            return found
        # End if
        # Time series files are named {case}.{hist_str}.{variable}.{dates}.nc,
        # so anchoring on the configured stream keeps a directory holding more
        # than one stream from contributing variables the run will not use:
        for hist_str in hist_str_list:
            pattern = f"{case_name}.{hist_str}.*.nc" if hist_str else f"{case_name}*.nc"
            for fname in utils.find_ts_files(ts_loc, pattern):
                var = utils.ts_var_from_filename(fname)
                if var:
                    found.add(var)
                # End if
            # End for
        # End for
        return found
    # End if

    if not hist_loc:
        return found
    # End if

    for hist_str in hist_str_list:
        pattern = f"*{hist_str}.*.nc" if hist_str else "*.nc"
        hist_files = sorted(Path(hist_loc).glob(pattern))
        if not hist_files:
            continue
        # End if
        with xr.open_dataset(
            hist_files[0], decode_cf=False, decode_times=False
        ) as hist_ds:
            found.update(
                var
                for var in hist_ds.data_vars
                if "time" in hist_ds[var].dims
                and _HORIZONTAL_DIMS.intersection(hist_ds[var].dims)
            )
        # End with
    # End for

    return found


# +++++++++++++++++++
# Define Obs class
# +++++++++++++++++++
# ++++++++++++++++
# Helper functions
# ++++++++++++++++


def _first_ts_file(ts_loc, case_name):
    """The first time series file of `case_name` in `ts_loc`, or None."""
    fils = sorted(Path(ts_loc).glob(f"{case_name}.*.nc"))
    return fils[0] if fils else None


def _on_native_grid(fil):
    """Whether the model output in `fil` is on a native (unstructured) grid.

    Parameters
    ----------
    fil : str, pathlib.Path, or None
        A history or time series file; None when there is none to look at.

    Returns
    -------
    bool
        True for an unstructured grid (CAM "ncol" or CLM "lndgrid");
        False for a lat/lon grid, or when there is no file.
    """
    if fil is None:
        return False
    with xr.open_dataset(fil, decode_times=False) as dset:
        dims = set(dset.dims)
    if dims & {"ncol", "ncols"}:
        print("\t  Looks like this is an atmosphere unstructured grid")
        return True
    if "lndgrid" in dims:
        print("\t  Looks like this is a land unstructured grid")
        return True
    print("\t  Looks like this is a structured lat/lon grid")
    return False


def _model_component(hist_strs):
    """The model component ("atm" or "lnd") named by a case's history streams.

    Parameters
    ----------
    hist_strs : str, list of str, or None
        The case's history stream(s), e.g. "cam.h0a" or ["clm2.h0"].

    Returns
    -------
    str
        "lnd" when the streams are CLM ("clm2.h0", ...), otherwise "atm",
        which is also what a case with no stream given is taken to be.
    """
    if not hist_strs:
        return "atm"
    if isinstance(hist_strs, str):
        hist_strs = [hist_strs]
    if any(str(hstr).startswith("clm") for hstr in hist_strs):
        return "lnd"
    return "atm"


def calc_ts_setting(calc_ts, ts_done):
    """Whether the ADF should make a case's time series from its history files.

    'calc_cam_ts' is the setting, mirroring 'calc_cam_climo'.  'cam_ts_done' is
    its older, inverted name, still accepted so that existing config files
    keep working.

    Parameters
    ----------
    calc_ts : bool or None
        The case's 'calc_cam_ts' entry, or None when it is not given.
    ts_done : bool or None
        The case's 'cam_ts_done' entry, or None when it is not given.

    Returns
    -------
    bool or None
        True to make the time series from the history files (also the default
        when neither entry is given), False to use the pre-made time series in
        'cam_ts_loc'.  None when the two entries contradict each other, which
        the caller reports.
    """
    if calc_ts is None:
        return True if ts_done is None else not ts_done
    if ts_done is not None and bool(ts_done) == bool(calc_ts):
        return None
    return bool(calc_ts)


def _climo_loc_years(climo_loc):
    """Read the years out of a climatology folder named like '.../yrs_1_5/'.

    Parameters
    ----------
    climo_loc : str, pathlib.Path, or None
        A case's 'cam_climo_loc'.

    Returns
    -------
    tuple of int or None
        (start, end) from the innermost 'yrs_<start>_<end>' folder in the path,
        or None if there is none.
    """
    if not climo_loc:
        return None
    for part in reversed(Path(climo_loc).parts):
        match = re.fullmatch(r"yrs_(\d+)_(\d+)", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    # End for
    return None


def _ts_years_span(ts_loc, case_name, hist_strs):
    """First and last whole year held by a case's existing time series files.

    Parameters
    ----------
    ts_loc : str, pathlib.Path, or None
        The case's 'cam_ts_loc'.
    case_name : str
        The case.
    hist_strs : list of str or None
        The case's history streams; None searches every stream.

    Returns
    -------
    tuple of int or None
        (first, last), or None when there are no readable time series files.
    """
    if not ts_loc or not Path(ts_loc).is_dir():
        return None
    years = set()
    for hist_str in hist_strs or [None]:
        pattern = f"{case_name}.{hist_str}.*.nc" if hist_str else f"{case_name}.*.nc"
        for fil in Path(ts_loc).glob(pattern):
            # One file at a time, so one unreadable name does not hide the rest:
            years |= utils.ts_years_covered([fil]) or set()
    # End for
    return (min(years), max(years)) if years else None


def _climo_years(name, climo_syr, climo_eyr, ts_syr, ts_eyr):
    """The climatology years for a case, defaulting to its time series years.

    'climo_start_year'/'climo_end_year' let the climatology cover only part of
    the time series years; a year that is not given, or that falls outside the
    time series years, is replaced by the time series year.
    """
    if climo_syr is None:
        climo_syr = ts_syr
    elif not ts_syr <= int(climo_syr) <= ts_eyr:
        msg = f"\t WARNING: Given climo start year '{climo_syr}' for {name} is not "
        msg += f"within its time series years {ts_syr}-{ts_eyr}, using {ts_syr}"
        print(msg)
        climo_syr = ts_syr
    if climo_eyr is None:
        climo_eyr = ts_eyr
    elif not ts_syr <= int(climo_eyr) <= ts_eyr:
        msg = f"\t WARNING: Given climo end year '{climo_eyr}' for {name} is not "
        msg += f"within its time series years {ts_syr}-{ts_eyr}, using {ts_eyr}"
        print(msg)
        climo_eyr = ts_eyr
    return int(climo_syr), int(climo_eyr)


class AdfInfo(AdfConfig):
    """
    Information/Parameter class, which initializes
    an AdfConfig object and provides additional
    variables and methods to simplify access to the
    standard, expanded config dictionaries.
    """

    def __init__(self, config_file, debug=False):
        """
        Initalize ADF Info object.
        """

        # Initialize Config attributes:
        super().__init__(config_file, debug=debug)

        # Add basic diagnostic info to object:
        self.__basic_info = self.read_config_var("diag_basic_info", required=True)

        # Expand basic info variable strings:
        self.expand_references(self.__basic_info)

        # Add CAM climatology info to object:
        self.__cam_climo_info = self.read_config_var("diag_cam_climo", required=True)

        # Expand CAM climo info variable strings:
        self.expand_references(self.__cam_climo_info)

        # Add CVDP info to object:
        self.__cvdp_info = self.read_config_var("diag_cvdp_info")
        # Expand CVDP climo info variable strings:
        if self.__cvdp_info is not None:
            self.expand_references(self.__cvdp_info)
            cvdp_default_loc = Path("externals/CVDP/")
            self.__cvdp_info.get("cvdp_codebase_loc", cvdp_default_loc)
        # End if

        # Add MDTF info to object:
        self.__mdtf_info = self.read_config_var("diag_mdtf_info")

        if self.__mdtf_info is not None:
            self.expand_references(self.__mdtf_info)
        # End if

        # Get the current system user
        self.__user = getpass.getuser()

        # Check if inputs are of the correct type:
        # -------------------------------------------

        # Use "cam_case_name" as the variable that sets the total number of cases:
        if isinstance(self.get_cam_info("cam_case_name", required=True), list):

            # Extract total number of test cases:
            self.__num_cases = len(self.get_cam_info("cam_case_name"))

        else:
            # Set number of cases to one:
            self.__num_cases = 1
        # End if

        # Loop over all items in config dict:
        for conf_var, conf_val in self.__cam_climo_info.items():
            # Hist_str can be a list for each case, so set it as a nested list here
            if "hist_str" in conf_var:
                self.hist_str_to_list(conf_var, conf_val)
            elif isinstance(conf_val, list):
                # If a list, then make sure it is has the correct number of entries:
                if not len(conf_val) == self.__num_cases:
                    emsg = f"diag_cam_climo config variable '{conf_var}' should have"
                    emsg += (
                        f" {self.__num_cases} entries, instead it has {len(conf_val)}"
                    )
                    self.end_diag_fail(emsg)
            else:
                # If not a list, replicate the scalar value for each case
                self.__cam_climo_info[conf_var] = [conf_val] * self.__num_cases
        # End for

        # Initialize ADF variable list:
        self.__diag_var_list = self.read_config_var("diag_var_list", required=True)

        # A single entry can be given as a plain string in the config file:
        if isinstance(self.__diag_var_list, str):
            self.__diag_var_list = [self.__diag_var_list]
        # End if

        # Replace an "all" entry with the variables the input files actually hold:
        self.__diag_var_list = self._expand_var_list_all(self.__diag_var_list)

        # Initialize list of regions for the regional (land) diagnostics, if any:
        self.__region_list = self.read_config_var("region_list")
        if not self.__region_list:
            self.__region_list = []
        elif isinstance(self.__region_list, str):
            self.__region_list = [self.__region_list]
        # End if

        # Case names:
        case_names = self.get_cam_info("cam_case_name", required=True)

        # Grab test case nickname(s)
        test_nickname_list = self.get_cam_info("case_nickname")

        if test_nickname_list:
            test_nicknames = []  # set to be an empty list
            for i, nickname in enumerate(test_nickname_list):
                if nickname is None:
                    test_nicknames.append(case_names[i])
                else:
                    test_nicknames.append(test_nickname_list[i])
                # End if
            # End for
        else:
            test_nicknames = []  # Re-set to be an empty list
            for case_name in case_names:
                test_nicknames.append(case_name)
            # End for
        # End if

        self.__native_grid = {}

        # Whether the ADF makes each case's time series (see the calc_ts property);
        # the baseline's stays None when comparing against observations:
        self.__calc_ts = []
        self.__calc_ts_baseline = None

        self.__base_hist_str = ""

        # Initialize "compare_obs" variable:
        self.__compare_obs = self.get_basic_info("compare_obs")
        redo_plot = self.get_basic_info("redo_plot")
        plot_type = self.basic_info_dict.get("plot_type", "png")
        print(f"\t NOTE: redo_plot is set to {redo_plot}")
        print(f"\t NOTE: Plot type is set to {plot_type}")

        # Check if a CAM vs AMWG obs comparison is being performed:
        if self.__compare_obs:

            # If so, then set the baseline info to None, to ensure any scripts
            # that check this variable won't crash:
            self.__cam_bl_climo_info = None

            # Set baseline hist string object to None
            self.__base_hist_str = None

            # Also set data name for use below:
            data_name = "Obs"
            base_nickname = "Obs"

            # Set the baseline years to empty strings:
            syear_baseline = ""
            eyear_baseline = ""
            climo_syear_baseline = ""
            climo_eyear_baseline = ""

            # Observations don't set the model component; the test cases do:
            base_comp = None

            # Observations are never on a model's native grid:
            self.__unstruct_base = False
            baseline_mesh_file = None  # self.get_baseline_info("mesh_file")
            self.__baseline_mesh_file = baseline_mesh_file

            # Lat/lon file (land mask for CLM gridding), only used by land:
            self.__baseline_latlon_file = None

            # Check if any a weights file exists if using native grid, OPTIONAL
            baseline_wgts_file = None  # self.get_baseline_info("weights_file")
            self.__baseline_wgts_file = baseline_wgts_file

            baseline_regrid_method = None  # self.get_baseline_info("regrid_method")
            if baseline_regrid_method == "conservative":
                print(
                    "user defined 'conservative', "
                    "but xesmf has a typo, changing to 'coservative'"
                )
                baseline_regrid_method = "coservative"
            self.__baseline_regrid_method = baseline_regrid_method
        else:
            # If not, then assume a CAM vs CAM run and add CAM baseline climatology info to object:
            self.__cam_bl_climo_info = self.read_config_var(
                "diag_cam_baseline_climo", required=True
            )

            # Expand CAM baseline climo info variable strings:
            self.expand_references(self.__cam_bl_climo_info)

            # Set data name to baseline case name:
            data_name = self.get_baseline_info("cam_case_name", required=True)

            # Attempt to grab baseline start_years (not currently required).
            # 'ts_start_year'/'ts_end_year' are accepted as other names for these:
            syear_baseline = self.get_baseline_info("start_year")
            if syear_baseline is None:
                syear_baseline = self.get_baseline_info("ts_start_year")
            eyear_baseline = self.get_baseline_info("end_year")
            if eyear_baseline is None:
                eyear_baseline = self.get_baseline_info("ts_end_year")

            # Optional climatology years, a part of the years above:
            climo_syear_baseline = self.get_baseline_info("climo_start_year")
            climo_eyear_baseline = self.get_baseline_info("climo_end_year")

            # Get climo years for verification or assignment if missing
            baseline_hist_locs = self.get_baseline_info("cam_hist_loc")

            # Read hist_str (component.hist_num, eg cam.h0) from the yaml file
            baseline_hist_str = self.get_baseline_info("hist_str")

            # Model component of the baseline (atm or lnd):
            base_comp = _model_component(baseline_hist_str)

            # Record the stream the user configured straight away, so that it
            # is available whether the years come from history files or from
            # pre-made time series.  Only what was actually given is recorded:
            # with pre-made time series and no stream configured this stays
            # empty, and the searches downstream go on matching any stream.
            if baseline_hist_str and not isinstance(baseline_hist_str, list):
                baseline_hist_str = [baseline_hist_str]
            # End if
            if baseline_hist_str:
                self.__base_hist_str = baseline_hist_str
            # End if

            # Check if any time series files are pre-made
            baseline_calc_ts = self.__resolve_calc_ts(
                [self.get_baseline_info("calc_cam_ts")],
                [self.get_baseline_info("cam_ts_done")],
                [data_name],
            )[0]
            self.__calc_ts_baseline = baseline_calc_ts
            baseline_mesh_file = self.get_baseline_info("mesh_file")
            self.__baseline_mesh_file = baseline_mesh_file

            # Lat/lon file (land mask for CLM gridding), only used by land:
            self.__baseline_latlon_file = self.get_baseline_info("latlon_file")

            # Check if any a weights file exists if using native grid, OPTIONAL
            baseline_wgts_file = self.get_baseline_info("weights_file")
            self.__baseline_wgts_file = baseline_wgts_file

            baseline_regrid_method = self.get_baseline_info("regrid_method")
            if baseline_regrid_method == "conservative":
                print(
                    "user defined 'conservative', but xesmf has a typo, changing to 'coservative'"
                )
                baseline_regrid_method = "coservative"
            self.__baseline_regrid_method = baseline_regrid_method

            # Whether the baseline is on a native grid is read from its data
            # below: a history file, or else a pre-made time series file.
            unstruct = False

            # Check if time series files already exist,
            # if so don't rely on climo years from history location
            if not baseline_calc_ts:
                baseline_hist_locs = None

                # Years and grid from the pre-made time series, or (with none)
                # from the pre-made climatologies:
                syear_baseline, eyear_baseline, unstruct = self.__premade_case_info(
                    data_name,
                    {
                        "ts_loc": self.get_baseline_info("cam_ts_loc"),
                        "hist_str": baseline_hist_str,
                        "climo_loc": self.get_baseline_info("cam_climo_loc"),
                        "calc_climo": self.get_baseline_info("calc_cam_climo"),
                        "section": "diag_cam_baseline_climo",
                    },
                    syear_baseline,
                    eyear_baseline,
                )
            # End if

            # Check if history file path exists:
            elif baseline_hist_locs and any(baseline_hist_locs):
                # History files are found by name, so a stream is needed here
                # even when the user did not give one:
                if not baseline_hist_str:
                    baseline_hist_str = ["cam.h0a"]
                    self.__base_hist_str = baseline_hist_str
                # End if

                # Grab first possible hist string, just looking for years of run
                base_hist_str = baseline_hist_str[0]
                starting_location = Path(baseline_hist_locs)
                print(f"\tChecking history files in '{starting_location}'")

                # Check that the history file location exists and can be read
                hist_problem = utils.describe_dir_problem(starting_location)
                if hist_problem:
                    msg = "Checking history file location:\n"
                    msg += f"\tCannot use the history file location: {hist_problem}."
                    self.debug_log(msg)
                    emsg = f"{data_name} starting_location: {hist_problem}!\n"
                    emsg += "\tCheck the path 'cam_hist_loc' in the "
                    emsg += "'diag_cam_baseline_climo' section of your config file."
                    self.end_diag_fail(emsg)
                file_list = sorted(
                    starting_location.glob("*" + base_hist_str + ".*.nc")
                )

                # Check if there are any history files
                if len(file_list) == 0:
                    msg = "Checking history files:\n"
                    msg += f"\tThere are no history files in '{starting_location}'."
                    self.debug_log(msg)
                    emsg = f"{data_name} starting_location {starting_location}: "
                    emsg += f"No history files found for {base_hist_str}!\n"
                    emsg += "\tTry checking the path 'cam_hist_loc' or the 'hist_str' "
                    emsg += " in 'diag_cam_baseline_climo' "
                    emsg += "section in your config file are correct..."
                    self.end_diag_fail(emsg)

                # Partition string to find exactly where h-number is
                # This cuts the string before and after the `{hist_str}.` sub-string
                # so there will always be three parts:
                # before sub-string, sub-string, and after sub-string
                # Since the last part always includes the time range, grab that with last index (2)
                # NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                base_climo_yrs = [
                    int(str(i).partition(f"{base_hist_str}.")[2][0:4])
                    for i in file_list
                ]
                if not base_climo_yrs:
                    msg = f"\t ERROR: No climo years found in {baseline_hist_locs}, "
                    raise AdfError(msg)

                base_climo_yrs = sorted(np.unique(base_climo_yrs))

                base_found_syr = int(base_climo_yrs[0])
                base_found_eyr = int(base_climo_yrs[-1])

                # Years the existing time series already hold count too: the
                # history files may be only the new years of an extended run.
                ts_span = _ts_years_span(
                    self.get_baseline_info("cam_ts_loc"), data_name, baseline_hist_str
                )
                if ts_span:
                    base_found_syr = min(base_found_syr, ts_span[0])
                    base_found_eyr = max(base_found_eyr, ts_span[1])
                    base_climo_yrs = [base_found_syr, base_found_eyr]
                # End if

                # Check if start or end year is missing. If so then just assume it is the
                # start or end of the entire available model data.
                if syear_baseline is None:
                    msg = f"\t WARNING: No given start year for {data_name}, "
                    msg += f"using first found year: {base_found_syr}"
                    print(msg)
                    syear_baseline = base_found_syr
                if not base_found_syr <= syear_baseline <= base_found_eyr:
                    msg = f"\t WARNING: Given start year '{syear_baseline}' is not in current "
                    msg += f"dataset {data_name}, using first found year: {base_climo_yrs[0]}"
                    print(msg)
                    syear_baseline = base_found_syr

                if eyear_baseline is None:
                    msg = f"\t WARNING: No given end year for {data_name}, "
                    msg += f"using last found year: {base_found_eyr}"
                    print(msg)
                    eyear_baseline = base_found_eyr
                if not base_found_syr <= eyear_baseline <= base_found_eyr:
                    msg = f"\t WARNING: Given end year '{eyear_baseline}' is not in current "
                    msg += f"dataset {data_name}, using last found year: {base_climo_yrs[-1]}"
                    print(msg)
                    eyear_baseline = base_found_eyr

                unstruct = _on_native_grid(file_list[0])
            else:
                # Neither pre-made time series nor a history file location was provided,
                # so years can only come from explicit start_year/end_year config entries.
                if (syear_baseline is None) or (eyear_baseline is None):
                    emsg = f"Unable to determine baseline years for '{data_name}'.\n"
                    emsg += "\tProvide 'cam_ts_loc' (with 'calc_cam_ts: false'), or 'cam_hist_loc', or "
                    emsg += "explicit 'start_year' and 'end_year' in the 'diag_cam_baseline_climo' "
                    emsg += "section of your config file."
                    self.end_diag_fail(emsg)
            # End if
            self.__unstruct_base = unstruct
            self.__native_grid[data_name] = unstruct

            # Grab baseline nickname
            base_nickname = self.get_baseline_info("case_nickname")
            if base_nickname is None:
                base_nickname = data_name

            # Get integer for baseline years for searching climo files
            syear_baseline = int(syear_baseline)
            eyear_baseline = int(eyear_baseline)

            # Climatology years default to the time series years:
            climo_syear_baseline, climo_eyear_baseline = _climo_years(
                data_name,
                climo_syear_baseline,
                climo_eyear_baseline,
                syear_baseline,
                eyear_baseline,
            )

            # Update baseline case name:
            data_name += f"_{syear_baseline}_{eyear_baseline}"
        # End if (compare_obs)

        # Initialize case nicknames:
        self.__test_nicknames = test_nicknames
        self.__base_nickname = base_nickname

        # Save starting and ending years as object variables:
        self.__syear_baseline = syear_baseline
        self.__eyear_baseline = eyear_baseline
        self.__climo_syear_baseline = climo_syear_baseline
        self.__climo_eyear_baseline = climo_eyear_baseline

        # Create plot location variable for potential use by the website generator.
        # Please note that this is also assumed to be the output location for the analyses scripts:
        # -------------------------------------------------------------------------
        self.__plot_location = []  # Must be a list to manage multiple cases

        # Plot directory:
        plot_dir = self.get_basic_info("cam_diag_plot_loc", required=True)

        # Unstructured plotting:
        unstructured_plotting = self.get_basic_info("unstructured_plotting")
        if not unstructured_plotting:
            unstructured_plotting = False
        self.__unstructured_plotting = unstructured_plotting

        # Mesh file
        mesh_file = self.get_basic_info("mesh_file")
        self.__mesh_file = mesh_file

        # Case names:
        case_names = self.get_cam_info("cam_case_name", required=True)

        # Start years (not currently required).
        # 'ts_start_year'/'ts_end_year' are accepted as other names for these:
        syears = self.get_cam_info("start_year")
        if syears is None:
            syears = self.get_cam_info("ts_start_year")

        # End year (not currently rquired):
        eyears = self.get_cam_info("end_year")
        if eyears is None:
            eyears = self.get_cam_info("ts_end_year")

        # Optional climatology years, a part of the years above:
        climo_syears = self.get_cam_info("climo_start_year")
        climo_eyears = self.get_cam_info("climo_end_year")

        # Make lists of None to be iterated over for case_names
        if syears is None:
            syears = [None] * self.__num_cases
        elif not isinstance(syears, list):
            syears = [syears] * self.__num_cases

        if eyears is None:
            eyears = [None] * self.__num_cases
        elif not isinstance(eyears, list):
            eyears = [eyears] * self.__num_cases

        if climo_syears is None:
            climo_syears = [None] * self.__num_cases
        elif not isinstance(climo_syears, list):
            climo_syears = [climo_syears] * self.__num_cases

        if climo_eyears is None:
            climo_eyears = [None] * self.__num_cases
        elif not isinstance(climo_eyears, list):
            climo_eyears = [climo_eyears] * self.__num_cases
        # Extract cam history files location:
        cam_hist_locs = self.get_cam_info("cam_hist_loc")

        # A history file location is not needed by a case running on pre-made
        # time series, so it can be absent altogether or given for only some
        # cases.  Normalize it to one entry per case, as the years above are,
        # so that it can be indexed by case number and set per case below.
        if cam_hist_locs is None:
            cam_hist_locs = [None] * self.__num_cases
        elif not isinstance(cam_hist_locs, list):
            cam_hist_locs = [cam_hist_locs] * self.__num_cases
        # End if

        # Get cleaned nested list of hist_str for test case(s) (component.hist_num, eg cam.h0)
        cam_hist_str = self.__cam_climo_info.get("hist_str", None)

        if not cam_hist_str:
            hist_str = [["cam.h0a"]] * self.__num_cases
        else:
            hist_str = cam_hist_str
        # End if

        # Initialize CAM history string nested list
        self.__hist_str = hist_str

        # Check if using pre-made ts files
        calc_ts = self.__resolve_calc_ts(
            self.get_cam_info("calc_cam_ts"),
            self.get_cam_info("cam_ts_done"),
            case_names,
        )
        self.__calc_ts = calc_ts

        # Check if any a FV file exists if using native grid
        cam_mesh_files = self.get_cam_info("mesh_file")
        if cam_mesh_files is None:
            cam_mesh_files = [None] * len(case_names)
        self.__cam_mesh_files = cam_mesh_files

        # Lat/lon file (land mask for CLM gridding), only used by land:
        cam_latlon_files = self.get_cam_info("latlon_file")
        if cam_latlon_files is None:
            cam_latlon_files = [None] * len(case_names)
        self.__cam_latlon_files = cam_latlon_files

        # Check if any a weights file exists if using native grid, OPTIONAL
        cam_wgts_files = self.get_cam_info("weights_file")
        if cam_wgts_files is None:
            cam_wgts_files = [None] * len(case_names)
        self.__cam_wgts_files = cam_wgts_files

        # The method is checked (and "conservative" mapped to what xesmf
        # needs) by adf_utils.grid_to_latlon, which also takes None:
        cam_regrid_method = self.get_cam_info("regrid_method")
        if cam_regrid_method is None:
            cam_regrid_method = [None] * len(case_names)
        self.__cam_regrid_method = cam_regrid_method

        # Grab case time series file location(s).  Not needed by a case run on
        # pre-made climatologies alone:
        input_ts_locs = self.get_cam_info("cam_ts_loc") or [None] * len(case_names)
        climo_locs = self.get_cam_info("cam_climo_loc") or [None] * len(case_names)
        calc_climos = self.get_cam_info("calc_cam_climo") or [None] * len(case_names)

        # Loop over cases:
        syears_fixed = []
        eyears_fixed = []
        climo_syears_fixed = []
        climo_eyears_fixed = []
        unstructs = []
        case_comps = []
        for case_idx, case_name in enumerate(case_names):
            # Whether the case is on a native grid is read from its data
            # below: a history file, or else a pre-made time series file.
            unstruct = False

            syear = syears[case_idx]
            eyear = eyears[case_idx]

            # Check if time series files exist, if so don't rely on climo years
            if not calc_ts[case_idx]:
                cam_hist_locs[case_idx] = None

                # Years and grid from the pre-made time series, or (with none)
                # from the pre-made climatologies:
                syear, eyear, unstruct = self.__premade_case_info(
                    case_name,
                    {
                        "ts_loc": input_ts_locs[case_idx],
                        "hist_str": self.__hist_str[case_idx],
                        "climo_loc": climo_locs[case_idx],
                        "calc_climo": calc_climos[case_idx],
                        "section": "diag_cam_climo",
                    },
                    syear,
                    eyear,
                )
            # End if

            # Check if history file path exists:
            hist_str_case = hist_str[case_idx]

            # Model component of this case (atm or lnd):
            case_comps.append(_model_component(hist_str_case))

            # This case's own location, not any case's: with several cases,
            # some on pre-made time series and some not, asking whether any of
            # them has one sent a case with none into the search below and
            # tripped over its empty location.
            if cam_hist_locs[case_idx]:
                # Grab first possible hist string, just looking for years of run
                hist_str_use = hist_str_case[0]

                # Get climo years for verification or assignment if missing
                starting_location = Path(cam_hist_locs[case_idx])
                print(f"\tChecking history files in '{starting_location}'")

                file_list = sorted(starting_location.glob("*" + hist_str_use + ".*.nc"))

                # Check that the history file location exists and can be read
                hist_problem = utils.describe_dir_problem(starting_location)
                if hist_problem:
                    msg = "Checking history file location:\n"
                    msg += f"\tCannot use the history file location: {hist_problem}."
                    self.debug_log(msg)
                    emsg = f"{case_name} starting_location: {hist_problem}!\n"
                    emsg += "\tCheck the path 'cam_hist_loc' in the 'diag_cam_climo' "
                    emsg += "section of your config file."
                    self.end_diag_fail(emsg)

                # Check if there are any history files
                file_list = sorted(starting_location.glob("*" + hist_str_use + ".*.nc"))
                if len(file_list) == 0:
                    msg = "Checking history files:\n"
                    msg += f"\tThere are no history files in '{starting_location}'."
                    self.debug_log(msg)
                    emsg = f"{case_name} starting_location {starting_location}: "
                    emsg += f"No history files found for {hist_str}!\n"
                    emsg += "\tTry checking the path 'cam_hist_loc' or the 'hist_str' "
                    emsg += "in 'diag_cam_climo' "
                    emsg += "section in your config file are correct..."
                    self.end_diag_fail(emsg)

                unstruct = _on_native_grid(file_list[0])

                # Partition string to find exactly where h-number is
                # This cuts the string before and after the `{hist_str}.` sub-string
                # so there will always be three parts:
                # before sub-string, sub-string, and after sub-string
                # Since the last part always includes the time range, grab that with last index (2)
                # NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                case_climo_yrs = [
                    int(str(i).partition(f"{hist_str_use}.")[2][0:4]) for i in file_list
                ]
                if not case_climo_yrs:
                    msg = (
                        f"\t ERROR: No climo years found in {cam_hist_locs[case_idx]}, "
                    )
                    raise AdfError(msg)
                case_climo_yrs = sorted(np.unique(case_climo_yrs))

                case_found_syr = int(case_climo_yrs[0])
                case_found_eyr = int(case_climo_yrs[-1])

                # Years the existing time series already hold count too: the
                # history files may be only the new years of an extended run.
                ts_span = _ts_years_span(
                    input_ts_locs[case_idx], case_name, hist_str_case
                )
                if ts_span:
                    case_found_syr = min(case_found_syr, ts_span[0])
                    case_found_eyr = max(case_found_eyr, ts_span[1])
                    case_climo_yrs = [case_found_syr, case_found_eyr]
                # End if

                # Check if start or end year is missing.  If so then just assume it is the
                # start or end of the entire available model data.
                if syear is None:
                    msg = f"\t WARNING: No given start year for {case_name}, "
                    msg += f"using first found year: {case_found_syr}"
                    print(msg)
                    syear = case_found_syr
                if not case_found_syr <= syear <= case_found_eyr:
                    msg = f"\t WARNING: Given start year '{syear}' is not in current dataset "
                    msg += f"{case_name}, using first found year: {case_climo_yrs[0]}\n"
                    print(msg)
                    syear = case_found_syr
                # End if
                if eyear is None:
                    msg = f"\t WARNING: No given end year for {case_name}, "
                    msg += f"using last found year: {case_found_eyr}"
                    print(msg)
                    eyear = case_found_eyr
                if not case_found_syr <= eyear <= case_found_eyr:
                    msg = f"\t WARNING: Given end year '{eyear}' is not in current dataset "
                    msg += f"{case_name}, using last found year: {case_climo_yrs[-1]}\n"
                    print(msg)
                    eyear = case_found_eyr
                # End if
            # End if

            unstructs.append(unstruct)
            self.__native_grid[case_name] = unstruct

            # Update climo year lists in case anything changed
            syear = int(syear)
            eyear = int(eyear)
            syears_fixed.append(syear)
            eyears_fixed.append(eyear)

            # Climatology years default to the time series years:
            climo_syear, climo_eyear = _climo_years(
                case_name, climo_syears[case_idx], climo_eyears[case_idx], syear, eyear
            )
            climo_syears_fixed.append(climo_syear)
            climo_eyears_fixed.append(climo_eyear)

            # Update case name with provided/found years:
            case_name += f"_{syear}_{eyear}"

            # Set the final directory name and save it to plot_location:
            direc_name = f"{case_name}_vs_{data_name}"
            plot_loc = os.path.join(plot_dir, direc_name)
            self.__plot_location.append(plot_loc)

            # If first iteration, then save directory name for use by baseline:
            first_case_dir = ""
            if case_idx == 0:
                first_case_dir = direc_name
            # End if

            # Go ahead and make the diag plot location if it doesn't exist already
            diag_location = Path(plot_loc)
            print(f"\n\tDiagnostic Plot Location: {diag_location}")
            if not diag_location.is_dir():
                print(
                    "\tINFO: Directory not found, making new diagnostic plot location"
                )
                diag_location.mkdir(parents=True)
        # End for

        self.__syears = syears_fixed
        self.__eyears = eyears_fixed
        self.__climo_syears = climo_syears_fixed
        self.__climo_eyears = climo_eyears_fixed

        self.__unstruct_test = unstructs

        # All cases (and the baseline, if a model case) must be the same
        # model component, which sets how the diagnostics are run (atm or lnd):
        all_comps = set(case_comps)
        if base_comp is not None:
            all_comps.add(base_comp)
        if len(all_comps) > 1:
            emsg = "Looks like the model components are not the same:\n"
            emsg += f"\t  - Test case(s): {case_comps}; Baseline case: {base_comp}\n"
            emsg += "\tCheck the 'hist_str' entries in your config file."
            self.end_diag_fail(emsg)
        # End if
        self.__model_component = all_comps.pop()
        print(f"\t NOTE: Model component is set to '{self.__model_component}'")

        # Finally add baseline case (if applicable) for use by the website table
        # generator.  These files will be stored in the same location as the first
        # listed case.
        if not self.compare_obs:
            self.__plot_location.append(os.path.join(plot_dir, first_case_dir))
        # End if
        # -------------------------------------------------------------------------

        # Initialize "num_procs" variable:
        # -----------------------------------------
        temp_num_procs = self.get_basic_info("num_procs")

        if not temp_num_procs:
            # Variable not present, so set to a single processor:
            self.__num_procs = 1
        else:
            # Check if variable is a string and matches keyword:
            if isinstance(temp_num_procs, str) and temp_num_procs.strip() == "*":

                # Set number of processors to total number of CPUs
                # on the node.  Please note that at some point this
                # may need to be replaced with a DASK implementation
                # instead:

                # First try to get CPUs allowed by OS for process to use:
                try:
                    self.__num_procs = len(os.sched_getaffinity(0))
                except AttributeError:
                    # Operating system doesn't support getaffinity, so try
                    # straight CPU number:
                    if os.cpu_count():
                        self.__num_procs = os.cpu_count()
                    else:
                        # Something is weird with this Operating System,
                        # so warn user and then try to run in serial mode:
                        wmsg = "WARNING!!!! ADF unable to determine how"
                        wmsg += " many processors are availble on this system,"
                        wmsg += " so defaulting to a single process/core."
                        print(wmsg)
                        self.__num_procs = 1
                    # End if
                # End except
            else:
                # If anything else, then try to convert to integer:
                try:
                    self.__num_procs = int(temp_num_procs)
                except ValueError:
                    # This variable has been set to something that
                    # can't be converted into an integer, so warn
                    # user and then try to run in serial mode:
                    wmsg = "WARNING!!!!  The 'num_procs' variable"
                    wmsg += f" has been set to '{temp_num_procs}'"
                    wmsg += " which cannot be converted to an integer."
                    wmsg += "\nThe ADF will now default to a single core"
                    wmsg += " and attempt to run."
                    print(wmsg)
                    self.__num_procs = 1
                # End except
            # End if
        # End if
        # Print number of processors being used to debug log (if requested):
        self.debug_log(f"ADF is running with {self.__num_procs} processors.")

    #########

    def hist_str_to_list(self, conf_var, conf_val):
        """
        Normalizes hist_str input into a nested list [ncases][nfiles].
        """
        n = self.__num_cases
        result = None

        # 1. Handle Single String input: "h0" -> [["h0"], ["h0"], ...]
        if isinstance(conf_val, str):
            result = [[conf_val] for _ in range(n)]

        elif isinstance(conf_val, list):
            # 2. Check if it's already a nested list: [["h0"], ["h0"]]
            # We check the first element to see if it's a list.
            if len(conf_val) == n and all(isinstance(i, list) for i in conf_val):
                result = conf_val

            # 3. Check if it's a list of strings matching N cases: ["h0", "h1"]
            elif len(conf_val) == n and all(isinstance(i, str) for i in conf_val):
                result = [[i] for i in conf_val]

            # 4. Otherwise, treat it as a single set of files for ALL cases: ["h0", "h1"]
            else:
                # We wrap the list and multiply it
                result = [conf_val for _ in range(n)]

        if result is None:
            raise ValueError(f"Invalid format for {conf_var}: {conf_val}")

        self.__cam_climo_info[conf_var] = result

    #########

    # Create property needed to return "user" name to user:
    @property
    def user(self):
        """Return the "user" name if requested."""
        return self.__user

    # Create property needed to return "compare_obs" logical to user:
    @property
    def compare_obs(self):
        """Return the "compare_obs" logical to the user if requested."""
        return self.__compare_obs

    # Create property needed to return the number of test cases (num_cases) to user:
    @property
    def num_cases(self):
        """Return the "num_cases" integer value to the user if requested."""
        return self.__num_cases

    @property
    def diag_var_list(self):
        """Return a copy of the "diag_var_list" list to the user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__diag_var_list)

    # Create property needed to return "region_list" list to user:
    @property
    def region_list(self):
        """Return a copy of the "region_list" list to the user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__region_list)

    # Create property needed to return the model component to user:
    @property
    def model_component(self):
        """Return the model component of the cases to the user if requested, ie atm or lnd"""
        return self.__model_component

    # Create property needed to return "basic_info" expanded dictionary to user:
    @property
    def basic_info_dict(self):
        """Return a copy of the "basic_info" list to the user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__basic_info)

    # Create property needed to return "cam_climo_dict" expanded dictionary to user:
    @property
    def cam_climo_dict(self):
        """Return a copy of the "cam_climo_dict" list to the user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__cam_climo_info)

    # Create property needed to return "baseline_climo_dict" expanded dictionary to user:
    @property
    def baseline_climo_dict(self):
        """Return a copy of the "cam_bl_climo_info" list to the user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__cam_bl_climo_info)

    # Create property needed to return "unstructured_plotting" to user:
    @property
    def unstructured_plotting(self):
        """Return the "unstructured_plotting" logical to the user if requested."""
        return self.__unstructured_plotting

    # Create property needed to return the "diag_basic_info" "mesh_file" to user
    # (the diagnostics use each case's own mesh, from "mesh_files", instead):
    @property
    def mesh_file(self):
        """Return the "mesh_file" path from "diag_basic_info" (None if not given)."""
        return self.__mesh_file

    # Create property needed to return the test case and baseline mesh files to user:
    @property
    def mesh_files(self):
        """Return the test case and baseline mesh files to the user if requested.

        "test_mesh_file" is a list with one entry per test case (None for a case
        without one); "baseline_mesh_file" is the baseline's (or None).
        """

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        cam_mesh_files = copy.copy(self.__cam_mesh_files)
        baseline_mesh_file = self.__baseline_mesh_file

        return {
            "test_mesh_file": cam_mesh_files,
            "baseline_mesh_file": baseline_mesh_file,
        }

    # Create property needed to return "num_procs" to user:
    @property
    def num_procs(self):
        """Return the "num_procs" logical to the user if requested."""
        return self.__num_procs

    # Create property needed to return "plot_location" variable to user:
    @property
    def plot_location(self):
        """Return a copy of the '__plot_location' string list to user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable:
        return copy.copy(self.__plot_location)

    # Create property needed to return the lat/lon file to user:
    @property
    def latlon_files(self):
        """Return the test case and baseline lat/lon file (land mask for CLM gridding)
        to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        cam_latlon_files = copy.copy(self.__cam_latlon_files)
        baseline_latlon_file = self.__baseline_latlon_file
        return {
            "test_latlon_file": cam_latlon_files,
            "baseline_latlon_file": baseline_latlon_file,
        }

    # Create property needed to return the weight file dictionary to user:
    @property
    def latlon_wgt_files(self):
        """Return the test case and weight file dictionary to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        cam_wgts_files = copy.copy(self.__cam_wgts_files)
        baseline_wgts_file = self.__baseline_wgts_file

        return {
            "test_wgts_file": cam_wgts_files,
            "baseline_wgts_file": baseline_wgts_file,
        }

    # Create property needed to return the lat/lon regrid method to user:
    @property
    def latlon_regrid_method(self):
        """Return the test case and baseline lat/lon regrid method to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        cam_regrid_method = copy.copy(self.__cam_regrid_method)
        baseline_regrid_method = self.__baseline_regrid_method

        return {
            "test_regrid_method": cam_regrid_method,
            "baseline_regrid_method": baseline_regrid_method,
        }

    # Create property needed to return the unstructured dictionary to user:
    @property
    def unstructs(self):
        """Return the test case and baseline unstructured dictionary to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        unstruct_tests = copy.copy(self.__unstruct_test)
        unstruct_base = self.__unstruct_base

        return {"unstruct_tests": unstruct_tests, "unstruct_base": unstruct_base}

    # Create property needed to return the time series setting to user:
    @property
    def calc_ts(self):
        """Return whether the ADF makes each case's time series ('calc_cam_ts').

        Returns
        -------
        dict
            ``test_calc_ts``: list of bool, one per test case;
            ``baseline_calc_ts``: bool, or None when comparing to observations.
            True means the time series are made from the history files, False
            that pre-made ones are read from 'cam_ts_loc'.
        """
        return {
            "test_calc_ts": copy.copy(self.__calc_ts),
            "baseline_calc_ts": self.__calc_ts_baseline,
        }

    def __resolve_calc_ts(self, calc_ts, ts_done, case_names):
        """Resolve 'calc_cam_ts' (or the older 'cam_ts_done') for each case.

        Parameters
        ----------
        calc_ts : list or None
            The 'calc_cam_ts' entries, one per case, or None when not given.
        ts_done : list or None
            The 'cam_ts_done' entries, one per case, or None when not given.
        case_names : list of str
            The cases, for the messages.

        Returns
        -------
        list of bool
            Whether the ADF makes each case's time series; see
            :func:`calc_ts_setting`.
        """
        calc_ts = calc_ts or [None] * len(case_names)
        ts_done = ts_done or [None] * len(case_names)
        resolved = []
        for case_name, case_calc, case_done in zip(case_names, calc_ts, ts_done):
            setting = calc_ts_setting(case_calc, case_done)
            if setting is None:
                emsg = f"'{case_name}' sets both 'calc_cam_ts: {case_calc}' and"
                emsg += f" 'cam_ts_done: {case_done}', which contradict each other."
                emsg += "  'cam_ts_done' is the older name; keep only 'calc_cam_ts'."
                self.end_diag_fail(emsg)
            # End if
            if case_calc is None and case_done is not None:
                msg = (
                    f"\t NOTE: '{case_name}' uses 'cam_ts_done: {case_done}', the older"
                )
                msg += f" name for 'calc_cam_ts: {not case_done}'; please switch to"
                msg += " 'calc_cam_ts'."
                print(msg)
            # End if
            resolved.append(setting)
        # End for
        return resolved

    def __check_years(self, case_name, syear, eyear, found_syear, found_eyear):
        """Fill in or correct a case's years against the years its files hold.

        Parameters
        ----------
        case_name : str
            The case, for the messages.
        syear, eyear : int or None
            The configured 'start_year'/'end_year', or None.
        found_syear, found_eyear : int
            The first and last year found in the case's files.

        Returns
        -------
        tuple of int
            (syear, eyear): the configured years where they are within the
            found ones, otherwise the found years.
        """
        if syear is None:
            msg = f"\t WARNING: No given start year for {case_name}, "
            msg += f"using first found year: {found_syear}"
            print(msg)
            syear = found_syear
        if not found_syear <= syear <= found_eyear:
            msg = f"\t WARNING: Given start year '{syear}' is not in current dataset "
            msg += f"{case_name}, using first found year: {found_syear}\n"
            print(msg)
            syear = found_syear
        # End if
        if eyear is None:
            msg = f"\t WARNING: No given end year for {case_name}, "
            msg += f"using last found year: {found_eyear}"
            print(msg)
            eyear = found_eyear
        if not found_syear <= eyear <= found_eyear:
            msg = f"\t WARNING: Given end year '{eyear}' is not in current dataset "
            msg += f"{case_name}, using last found year: {found_eyear}\n"
            print(msg)
            eyear = found_eyear
        # End if
        return syear, eyear

    def __premade_case_info(self, case_name, inputs, syear, eyear):
        """Years and grid of a case that does not make its own time series.

        With 'calc_cam_ts: false' the case runs on pre-made files: its time
        series in 'cam_ts_loc' when there are any, otherwise (pre-made
        climatologies only, 'calc_cam_climo: false') nothing but its climo files.
        In that last case the years are the configured 'start_year'/'end_year',
        or else read from a 'yrs_<start>_<end>' climo folder name.

        Parameters
        ----------
        case_name : str
            The case.
        inputs : dict
            The case's ``ts_loc``, ``hist_str``, ``climo_loc`` and
            ``calc_climo`` settings, and its config ``section`` (for messages).
        syear, eyear : int or None
            The configured 'start_year'/'end_year', or None.

        Returns
        -------
        tuple
            (syear, eyear, unstruct), unstruct being whether the case is on a
            native grid.
        """
        ts_loc = inputs["ts_loc"]
        if ts_loc and _first_ts_file(ts_loc, case_name) is not None:
            input_ts_loc = Path(ts_loc)
            print(f"Checking existing time-series files in {input_ts_loc}")
            found_syear, found_eyear = self.get_climo_yrs_from_ts(
                input_ts_loc, case_name, hist_str=inputs["hist_str"]
            )
            syear, eyear = self.__check_years(
                case_name, syear, eyear, found_syear, found_eyear
            )
            return syear, eyear, _on_native_grid(_first_ts_file(ts_loc, case_name))
        # End if

        # No time series, which is only workable on pre-made climatologies:
        where = f"in 'cam_ts_loc' ({ts_loc})" if ts_loc else "('cam_ts_loc' is not set)"
        if inputs["calc_climo"]:
            emsg = f"'{case_name}' has 'calc_cam_ts: false' but no time series files"
            emsg += f" {where}, so its climatologies cannot be made.  Point"
            emsg += " 'cam_ts_loc' at its time series, set 'calc_cam_ts: true' to make"
            emsg += (
                " them, or set 'calc_cam_climo: false' to use pre-made climatologies."
            )
            self.end_diag_fail(emsg)
        # End if
        if syear is None or eyear is None:
            folder_years = _climo_loc_years(inputs["climo_loc"])
            if folder_years is None:
                emsg = (
                    f"'{case_name}' runs on pre-made climatologies only, so its years"
                )
                emsg += " cannot be read from time series or history files.  Give"
                emsg += (
                    f" 'start_year' and 'end_year' in '{inputs['section']}', or keep"
                )
                emsg += " the climatologies in a 'yrs_<start>_<end>' folder."
                self.end_diag_fail(emsg)
            # End if
            msg = f"\t NOTE: '{case_name}' years read from its climatology folder:"
            msg += f" {folder_years[0]}-{folder_years[1]}"
            print(msg)
            syear = folder_years[0] if syear is None else syear
            eyear = folder_years[1] if eyear is None else eyear
        # End if
        climo_loc = inputs["climo_loc"]
        climo_files = (
            # with or without the climatology years in the name:
            sorted(Path(climo_loc).glob(f"{case_name}*_climo*.nc"))
            if climo_loc
            else []
        )
        msg = f"\t NOTE: '{case_name}' runs on pre-made climatologies only (years"
        msg += f" {syear}-{eyear}); diagnostics that need time series will skip it."
        print(msg)
        return syear, eyear, _on_native_grid(climo_files[0] if climo_files else None)

    # Create property needed to return the native grid dictionary to user:
    @property
    def native_grid(self):
        """Return the test case and baseline native grid dictionary to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        native_grid = self.__native_grid

        return native_grid

    '''# Create property needed to return the native grid dictionary to user:
    @property
    def native_grid_gridded(self):
        """Return the test case and baseline native grid dictionary to the user if requested."""

        #Note that copies are needed in order to avoid having a script mistakenly
        #modify these variables, as they are mutable and thus passed by reference:
        native_grid_gridded = self.__native_grid_gridded

        return native_grid_gridded'''

    # Create property needed to return the climo start (syear) and end (eyear) years to user:
    @property
    def climo_yrs(self):
        """Return the climatology "syear" and "eyear" integer values to the user if requested.

        These are 'climo_start_year'/'climo_end_year' when given, otherwise
        the time series years (see `ts_yrs`).
        """
        syears = copy.copy(
            self.__climo_syears
        )  # Send copies so a script doesn't modify the original
        eyears = copy.copy(self.__climo_eyears)
        return {
            "syears": syears,
            "eyears": eyears,
            "syear_baseline": self.__climo_syear_baseline,
            "eyear_baseline": self.__climo_eyear_baseline,
        }

    # Create property needed to return the time series start and end years to user:
    @property
    def ts_yrs(self):
        """Return the time series "syear" and "eyear" integer values to the user if requested.

        These are 'start_year'/'end_year' (or the years found in the data),
        and are the years the time series files and case directories are made for.
        """
        syears = copy.copy(
            self.__syears
        )  # Send copies so a script doesn't modify the original
        eyears = copy.copy(self.__eyears)
        return {
            "syears": syears,
            "eyears": eyears,
            "syear_baseline": self.__syear_baseline,
            "eyear_baseline": self.__eyear_baseline,
        }

    # Create property needed to return the case nicknames to user:
    @property
    def case_nicknames(self):
        """Return the test case and baseline nicknames to the user if requested."""

        # Note that copies are needed in order to avoid having a script mistakenly
        # modify these variables, as they are mutable and thus passed by reference:
        test_nicknames = copy.copy(self.__test_nicknames)
        base_nickname = self.__base_nickname

        return {"test_nicknames": test_nicknames, "base_nickname": base_nickname}

    @property
    def hist_string(self):
        """Return the CAM history string list to the user if requested."""
        cam_hist_strs = copy.copy(self.__hist_str)
        if self.__base_hist_str:
            base_hist_strs = copy.copy(self.__base_hist_str)
        else:
            base_hist_strs = ""
        hist_strs = {"test_hist_str": cam_hist_strs, "base_hist_str": base_hist_strs}
        return hist_strs

    #########

    # Utility function to access expanded 'diag_basic_info' variables:
    def get_basic_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_basic_info' as requested by
        the user.
        """

        return self.read_config_var(
            var_str, conf_dict=self.__basic_info, required=required
        )

    #########

    # Utility function to access expanded 'diag_cam_climo' variables:
    def get_cam_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_climo' as requested by
        the user."""

        return self.read_config_var(
            var_str, conf_dict=self.__cam_climo_info, required=required
        )

    #########

    # Utility function to access expanded 'diag_cam_baseline_climo' variables:
    def get_baseline_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_baseline_climo' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        # Check if the cam baseline dictionary exists:
        if not self.__cam_bl_climo_info:
            # If required, then throw an error:
            if required:
                emsg = "get_baseline_info: Requested variable cannot be found"
                emsg += " because no baseline info exists.\n"
                emsg += (
                    "This is likely because an observational comparison is being done,"
                )
                emsg += " so try adding 'required = False' to the get call."
                self.end_diag_fail(emsg)
            # End if

            # If not required, then return none:
            return None
        # End if

        # If basline dictionary exists, then search for variable like normal:
        return self.read_config_var(
            var_str, conf_dict=self.__cam_bl_climo_info, required=required
        )

    #########

    def _expand_var_list_all(self, var_list):
        """
        Replace an "all" entry in 'diag_var_list' with the variables found in
        the test case input files.

        Parameters
        ----------
        var_list : list
            'diag_var_list' as the user wrote it

        Returns
        -------
        list
            the same list when it holds no "all" entry, otherwise the
            variables found in the input files together with any variables
            the user listed alongside "all".

        Notes
        -----
        Only the test case files are searched.  A baseline case that is
        missing one of the variables is already handled downstream -- the
        same as when the user lists a variable by hand -- and when the
        comparison is against observations there is no baseline to search.
        Variables listed alongside "all" are kept, so a derived variable that
        is in no input file (PRECT, for instance) can still be asked for.
        """
        # The entry is matched without regard to case, so "all", "All" and
        # "ALL" all work:
        keep = [var for var in var_list if str(var).lower() != "all"]
        if len(keep) == len(var_list):
            return var_list
        # End if

        case_names = self.get_cam_info("cam_case_name", required=True)
        hist_locs = self.get_cam_info("cam_hist_loc")
        hist_strs = self.get_cam_info("hist_str")
        ts_locs = self.get_cam_info("cam_ts_loc")
        # This runs before the cases are set up (where a contradictory
        # 'calc_cam_ts'/'cam_ts_done' pair is reported), so read both here:
        calc_ts = self.get_cam_info("calc_cam_ts")
        ts_done = self.get_cam_info("cam_ts_done")

        found = set()
        for case_idx, case_name in enumerate(case_names):
            hist_loc = hist_locs[case_idx] if hist_locs else None
            ts_loc = ts_locs[case_idx] if ts_locs else None
            case_ts_done = (
                calc_ts_setting(
                    calc_ts[case_idx] if calc_ts else None,
                    ts_done[case_idx] if ts_done else None,
                )
                is False
            )

            # Say what is wrong with the directory being searched, rather than
            # letting an unreadable one look like a case holding no variables:
            search_loc = ts_loc if case_ts_done else hist_loc
            loc_name = "cam_ts_loc" if case_ts_done else "cam_hist_loc"
            loc_problem = (
                utils.describe_dir_problem(search_loc)
                if search_loc
                else "was not provided"
            )
            if loc_problem:
                emsg = f"Test case '{case_name}' '{loc_name}' {loc_problem},"
                emsg += " so 'diag_var_list: all' cannot find its variables."
                self.end_diag_fail(emsg)
            # End if

            found.update(
                variables_in_case_files(
                    case_name,
                    hist_loc,
                    hist_strs[case_idx] if hist_strs else None,
                    ts_loc,
                    case_ts_done,
                )
            )
        # End for

        if not found:
            emsg = "No variables were found in the test case input files, so"
            emsg += " 'diag_var_list: all' has nothing to process."
            self.end_diag_fail(emsg)
        # End if

        var_list = sorted(found.union(keep))
        print(f"\t INFO: 'diag_var_list: all' -> {len(var_list)} variables to process.")
        self.debug_log(f"diag_var_list from input files: {var_list}")

        return var_list

    #########

    # Utility function to add a new model variable to the ADF (diag) variable list:
    def add_diag_var(self, var_str):
        """
        Adds a new variable to the ADF variable list
        """
        if var_str not in self.__diag_var_list:
            self.__diag_var_list.append(var_str)
        # End if

    #########

    # Utility function to access expanded 'diag_cvdp_info' variables
    def get_cvdp_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cvdp_info' as requested by
        the user. If 'diag_cvdp_info' is not found then try grabbing the
        variable from the top level of the YAML config file dictionary
        instead.
        """

        return self.read_config_var(
            var_str, conf_dict=self.__cvdp_info, required=required
        )

    #########

    # Utility function to access expanded 'diag_mdtf_info' variables
    def get_mdtf_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_mdtf_info' as requested by
        the user. If 'diag_mdtf_info' is not found then try grabbing the
        variable from the top level of the YAML config file dictionary
        instead.
        """

        return self.read_config_var(
            var_str, conf_dict=self.__mdtf_info, required=required
        )

    #########

    # Utility function to grab climo years from pre-made time series files:
    def get_climo_yrs_from_ts(self, input_ts_loc, case_name, hist_str=None):
        """
        Grab start and end climo years if none are specified in config file
        for pre-made time series file(s)

        Parameters
        ----------
        input_ts_loc
            directory holding the pre-made time series files
        case_name
            name of the case whose files should be searched for
        hist_str : str or list, optional
            history stream(s) configured for this case.  When given, the
            stream is used in the search; when absent the older, looser
            "any h0 stream" search is used instead.

        Return
        ------
          - start year
          - end year

        Notes
        -----
        The period is read from the file names when they carry it, so a
        directory holding more than one set of files for the same variable
        (years 1-20 alongside years 1-40) reports the whole available span
        instead of failing to open the files together.  Names whose dates
        cannot be read fall back to reading the period from the data.
        """

        # Grab variable list
        var_list = self.diag_var_list

        # Create "Path" objects:
        input_location = Path(input_ts_loc)

        # Check that the time series input directory exists and can be read:
        ts_problem = utils.describe_dir_problem(input_location)
        if ts_problem:
            errmsg = (
                f"\t ERROR: Time series directory {ts_problem}.  Script is exiting."
            )
            raise AdfError(errmsg)

        # Normalize the configured history stream(s) into a list:
        if not hist_str:
            hist_strs = []
        elif isinstance(hist_str, str):
            hist_strs = [hist_str]
        else:
            hist_strs = [h for h in hist_str if h]
        # End if

        # Search for first available variable in var_list to get a time series file to read
        # NOTE: it is assumed all the variables have the same dates!
        # Try the configured stream(s) first, then the older, looser search
        # (any h0 stream) so pre-existing directories still work:
        def ts_patterns(var):
            """Search patterns for one variable, most specific first."""
            return [f"{case_name}.{hstr}.{var}.*nc" for hstr in hist_strs] + [
                f"{case_name}*h0*.{var}.*nc"
            ]

        # End def

        # Sweep by pattern rank rather than by variable: every variable is tried
        # against the configured stream before any variable is tried against the
        # looser fallback, so a stray file from another stream cannot outrank
        # the stream the user actually configured.  Within a rank, the flat
        # search comes first for every variable and the recursive one only
        # afterwards -- a missing variable is normal, and recursing per pattern
        # would walk the whole tree once for every pattern tried.
        n_ranks = len(hist_strs) + 1
        ts_files = []
        skipped = var_list
        for rank in range(n_ranks):
            for recursive in (False, True):
                for idx, var in enumerate(var_list):
                    ts_files = utils.find_ts_files(
                        input_location, ts_patterns(var)[rank], recursive=recursive
                    )
                    if ts_files:
                        # Everything ahead of it was tried and missed:
                        skipped = var_list[:idx]
                        break
                    # End if
                # End for
                if ts_files:
                    break
                # End if
            # End for
            if ts_files:
                break
            # End if
        # End for

        # Report only the variables that were genuinely passed over.  Logging
        # inside the sweep would name every variable in diag_var_list whenever
        # the files turned out to be in a nested layout, which is not a problem
        # and not worth dozens of lines of debug log.
        for var in skipped:
            logmsg = "get years for time series:"
            logmsg += f"\n\tVar '{var}' not in dataset, skip to next to try and find climo years..."
            self.debug_log(logmsg)
        # End for

        if not ts_files:
            errmsg = (
                f"\t ERROR: No time series files found in '{input_ts_loc}' for case "
            )
            errmsg += f"'{case_name}' for any variable in 'diag_var_list'.  Script is exiting."
            raise AdfError(errmsg)
        # End if

        # The file names carry the period they hold, so use those rather than
        # opening the files: overlapping sets (years 1-20 alongside years 1-40,
        # left behind when a run is extended and the time series are remade)
        # cannot be opened together, and this runs during ADF setup, so failing
        # here takes down the whole run before any diagnostic starts.
        span = utils.ts_file_span(ts_files)
        if span and len(span[0]) >= 4:
            syr = int(span[0][:4])
            eyr = int(span[1][:4])
            _warn_on_long_climo_range(syr, eyr)
            return syr, eyr
        # End if

        # Read in file(s) -- names whose dates could not be read, so the period
        # has to come from the data itself:
        if len(ts_files) == 1:
            cam_ts_data = xr.open_dataset(ts_files[0], decode_times=True)
        else:
            cam_ts_data = xr.open_mfdataset(
                ts_files, decode_times=True, combine="by_coords"
            )

        # Use the interval each step covers rather than its stamp, so that the
        # years found here are the years the data actually covers:
        cam_ts_data = utils.use_time_bounds_midpoint(cam_ts_data)

        # Extract first and last years from dataset:
        syr = int(cam_ts_data.time[0].dt.year.values)
        eyr = int(cam_ts_data.time[-1].dt.year.values)

        _warn_on_long_climo_range(syr, eyr)

        return syr, eyr


# ++++++++++++++++++++
# End Class definition
# ++++++++++++++++++++
