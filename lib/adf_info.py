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

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

from pathlib import Path
import copy
import os
import getpass
import subprocess

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

# pylint: disable=unused-import
import numpy as np
import xarray as xr
# pylint: enable=unused-import

#ADF modules:
import adf_utils as utils
from adf_config import AdfConfig
from adf_base   import AdfError


def _warn_on_long_climo_range(syr, eyr):
    """Tell the user when the year range worked out from the files is large."""
    if eyr-syr >= 100:
        msg = f"WARNING: the found climo year range is large: {eyr-syr} years, "
        msg += "this may take a long time!"
        print(msg)
    #End if


#+++++++++++++++++++
#Define Obs class
#+++++++++++++++++++
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

        #Initialize Config attributes:
        super().__init__(config_file, debug=debug)

        #Add basic diagnostic info to object:
        self.__basic_info = self.read_config_var('diag_basic_info', required=True)

        #Expand basic info variable strings:
        self.expand_references(self.__basic_info)

        #Add CAM climatology info to object:
        self.__cam_climo_info = self.read_config_var('diag_cam_climo', required=True)

        #Expand CAM climo info variable strings:
        self.expand_references(self.__cam_climo_info)

        # Add CVDP info to object:
        self.__cvdp_info = self.read_config_var("diag_cvdp_info")
        # Expand CVDP climo info variable strings:
        if self.__cvdp_info is not None:
            self.expand_references(self.__cvdp_info)
            cvdp_default_loc = Path("externals/CVDP/")
            self.__cvdp_info.get("cvdp_codebase_loc",cvdp_default_loc)
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

        #Use "cam_case_name" as the variable that sets the total number of cases:
        if isinstance(self.get_cam_info("cam_case_name", required=True), list):

            #Extract total number of test cases:
            self.__num_cases = len(self.get_cam_info("cam_case_name"))

        else:
            #Set number of cases to one:
            self.__num_cases = 1
        #End if

        #Loop over all items in config dict:
        for conf_var, conf_val in self.__cam_climo_info.items():
            # Hist_str can be a list for each case, so set it as a nested list here
            if "hist_str" in conf_var:
                self.hist_str_to_list(conf_var, conf_val)
            elif isinstance(conf_val, list):
                # If a list, then make sure it is has the correct number of entries:
                if not len(conf_val) == self.__num_cases:
                    emsg = f"diag_cam_climo config variable '{conf_var}' should have"
                    emsg += f" {self.__num_cases} entries, instead it has {len(conf_val)}"
                    self.end_diag_fail(emsg)
            else:
                # If not a list, replicate the scalar value for each case
                self.__cam_climo_info[conf_var] = [conf_val] * self.__num_cases
        # End for

        #Initialize ADF variable list:
        self.__diag_var_list = self.read_config_var('diag_var_list', required=True)

        #Case names:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Grab test case nickname(s)
        test_nickname_list = self.get_cam_info('case_nickname')

        if test_nickname_list:
            test_nicknames = [] #set to be an empty list
            for i,nickname in enumerate(test_nickname_list):
                if nickname is None:
                    test_nicknames.append(case_names[i])
                else:
                    test_nicknames.append(test_nickname_list[i])
                #End if
            #End for
        else:
            test_nicknames = [] #Re-set to be an empty list
            for case_name in case_names:
                test_nicknames.append(case_name)
            #End for
        #End if

        self.__base_hist_str = ""

        #Initialize "compare_obs" variable:
        self.__compare_obs = self.get_basic_info('compare_obs')

        #Check if a CAM vs AMWG obs comparison is being performed:
        if self.__compare_obs:

            #If so, then set the baseline info to None, to ensure any scripts
            #that check this variable won't crash:
            self.__cam_bl_climo_info = None

            # Set baseline hist string object to None
            self.__base_hist_str = None

            #Also set data name for use below:
            data_name = "Obs"
            base_nickname = "Obs"

            #Set the baseline years to empty strings:
            syear_baseline = ""
            eyear_baseline = ""
        else:
            #If not, then assume a CAM vs CAM run and add CAM baseline climatology info to object:
            self.__cam_bl_climo_info = self.read_config_var('diag_cam_baseline_climo',
                                                            required=True)

            #Expand CAM baseline climo info variable strings:
            self.expand_references(self.__cam_bl_climo_info)

            #Set data name to baseline case name:
            data_name = self.get_baseline_info('cam_case_name', required=True)

            #Attempt to grab baseline start_years (not currently required):
            syear_baseline = self.get_baseline_info('start_year')
            eyear_baseline = self.get_baseline_info('end_year')

            #Get climo years for verification or assignment if missing
            baseline_hist_locs = self.get_baseline_info('cam_hist_loc')

            # Read hist_str (component.hist_num, eg cam.h0) from the yaml file
            baseline_hist_str = self.get_baseline_info("hist_str")

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

            #Check if any time series files are pre-made
            baseline_ts_done   = self.get_baseline_info("cam_ts_done")

            #Check if time series files already exist,
            #if so don't rely on climo years from history location
            if baseline_ts_done:
                baseline_hist_locs = None

                #Grab baseline time series file location
                input_ts_baseline = self.get_baseline_info("cam_ts_loc", required=True)
                input_ts_loc = Path(input_ts_baseline)

                #Get years from pre-made timeseries file(s)
                found_syear_baseline, found_eyear_baseline = self.get_climo_yrs_from_ts(
                    input_ts_loc, data_name, hist_str=baseline_hist_str)

                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
                if syear_baseline is None:
                    msg = f"\t WARNING: No given start year for {data_name}, "
                    msg += f"using first found year: {found_syear_baseline}"
                    print(msg)
                    syear_baseline = found_syear_baseline
                if not found_syear_baseline <= syear_baseline <= found_eyear_baseline:
                    msg = f"\t WARNING: Given start year '{syear_baseline}' is not in current "
                    msg += f"dataset {data_name}, using first found year: {found_syear_baseline}"
                    print(msg)
                    syear_baseline = found_syear_baseline

                if eyear_baseline is None:
                    msg = f"\t WARNING: No given end year for {data_name}, "
                    msg += f"using last found year: {found_eyear_baseline}"
                    print(msg)
                    eyear_baseline = found_eyear_baseline
                if not found_syear_baseline <= eyear_baseline <= found_eyear_baseline:
                    msg = f"\t WARNING: Given end year '{eyear_baseline}' is not in current "
                    msg += f"dataset {data_name}, using last found year: {found_eyear_baseline}"
                    print(msg)
                    eyear_baseline = found_eyear_baseline
            # End if

            # Check if history file path exists:
            elif baseline_hist_locs and any(baseline_hist_locs):
                # History files are found by name, so a stream is needed here
                # even when the user did not give one:
                if not baseline_hist_str:
                    baseline_hist_str = ["cam.h0a"]
                    self.__base_hist_str = baseline_hist_str
                # End if

                #Grab first possible hist string, just looking for years of run
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
                file_list = sorted(starting_location.glob("*" + base_hist_str + ".*.nc"))

                #Check if there are any history files
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
                #Since the last part always includes the time range, grab that with last index (2)
                #NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                base_climo_yrs = [int(str(i).partition(f"{base_hist_str}.")[2][0:4]) for i in file_list]
                if not base_climo_yrs:
                    msg = f"\t ERROR: No climo years found in {baseline_hist_locs}, "
                    raise AdfError(msg)

                base_climo_yrs = sorted(np.unique(base_climo_yrs))

                base_found_syr = int(base_climo_yrs[0])
                base_found_eyr = int(base_climo_yrs[-1])

                #Check if start or end year is missing. If so then just assume it is the
                #start or end of the entire available model data.
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

            else:
                #Neither pre-made time series nor a history file location was provided,
                #so years can only come from explicit start_year/end_year config entries.
                if (syear_baseline is None) or (eyear_baseline is None):
                    emsg = f"Unable to determine baseline years for '{data_name}'.\n"
                    emsg += "\tProvide 'cam_ts_loc' (with 'cam_ts_done'), or 'cam_hist_loc', or "
                    emsg += "explicit 'start_year' and 'end_year' in the 'diag_cam_baseline_climo' "
                    emsg += "section of your config file."
                    self.end_diag_fail(emsg)
            #End if

            #Grab baseline nickname
            base_nickname = self.get_baseline_info('case_nickname')
            if base_nickname is None:
                base_nickname = data_name

            #Get integer for baseline years for searching climo files
            syear_baseline = int(syear_baseline)
            eyear_baseline = int(eyear_baseline)

            #Update baseline case name:
            data_name += f"_{syear_baseline}_{eyear_baseline}"
        #End if (compare_obs)

        #Initialize case nicknames:
        self.__test_nicknames = test_nicknames
        self.__base_nickname = base_nickname

        #Save starting and ending years as object variables:
        self.__syear_baseline = syear_baseline
        self.__eyear_baseline = eyear_baseline

        #Create plot location variable for potential use by the website generator.
        #Please note that this is also assumed to be the output location for the analyses scripts:
        #-------------------------------------------------------------------------
        self.__plot_location = [] #Must be a list to manage multiple cases

        #Plot directory:
        plot_dir = self.get_basic_info('cam_diag_plot_loc', required=True)

        #Case names:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Start years (not currently required):
        syears = self.get_cam_info('start_year')

        #End year (not currently rquired):
        eyears = self.get_cam_info('end_year')

        #Make lists of None to be iterated over for case_names
        if syears is None:
            syears = [None] * self.__num_cases
        elif not isinstance(syears, list):
            syears = [syears] * self.__num_cases

        if eyears is None:
            eyears = [None] * self.__num_cases
        elif not isinstance(eyears, list):
            eyears = [eyears] * self.__num_cases
        #Extract cam history files location:
        cam_hist_locs = self.get_cam_info('cam_hist_loc')

        # A history file location is not needed by a case running on pre-made
        # time series, so it can be absent altogether or given for only some
        # cases.  Normalize it to one entry per case, as the years above are,
        # so that it can be indexed by case number and set per case below.
        if cam_hist_locs is None:
            cam_hist_locs = [None] * self.__num_cases
        elif not isinstance(cam_hist_locs, list):
            cam_hist_locs = [cam_hist_locs] * self.__num_cases
        # End if

        #Get cleaned nested list of hist_str for test case(s) (component.hist_num, eg cam.h0)
        cam_hist_str = self.__cam_climo_info.get('hist_str', None)

        if not cam_hist_str:
            hist_str = [['cam.h0a']]*self.__num_cases
        else:
            hist_str = cam_hist_str
        #End if

        #Initialize CAM history string nested list
        self.__hist_str = hist_str

        #Check if using pre-made ts files
        cam_ts_done   = self.get_cam_info("cam_ts_done")

        #Grab case time series file location(s)
        input_ts_locs = self.get_cam_info("cam_ts_loc", required=True)

        #Loop over cases:
        syears_fixed = []
        eyears_fixed = []
        for case_idx, case_name in enumerate(case_names):

            syear = syears[case_idx]
            eyear = eyears[case_idx]

            #Check if time series files exist, if so don't rely on climo years
            if cam_ts_done[case_idx]:
                cam_hist_locs[case_idx] = None

                #Grab case time series file location
                input_ts_loc = Path(input_ts_locs[case_idx])
                print(f"Checking existing time-series files in {input_ts_loc}")

                #Get years from pre-made timeseries file(s)
                found_syear, found_eyear = self.get_climo_yrs_from_ts(
                    input_ts_loc, case_name, hist_str=self.__hist_str[case_idx])

                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
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
                #End if
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
                #End if
            #End if

            #Check if history file path exists:
            hist_str_case = hist_str[case_idx]
            # This case's own location, not any case's: with several cases,
            # some on pre-made time series and some not, asking whether any of
            # them has one sent a case with none into the search below and
            # tripped over its empty location.
            if cam_hist_locs[case_idx]:
                #Grab first possible hist string, just looking for years of run
                hist_str_use = hist_str_case[0]

                #Get climo years for verification or assignment if missing
                starting_location = Path(cam_hist_locs[case_idx])
                print(f"\tChecking history files in '{starting_location}'")

                file_list = sorted(starting_location.glob('*'+hist_str_use+'.*.nc'))

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

                #Check if there are any history files
                file_list = sorted(starting_location.glob('*'+hist_str_use+'.*.nc'))
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

                #Partition string to find exactly where h-number is
                #This cuts the string before and after the `{hist_str}.` sub-string
                # so there will always be three parts:
                # before sub-string, sub-string, and after sub-string
                #Since the last part always includes the time range, grab that with last index (2)
                #NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                case_climo_yrs = [int(str(i).partition(f"{hist_str_use}.")[2][0:4]) for i in file_list]
                if not case_climo_yrs:
                    msg = f"\t ERROR: No climo years found in {cam_hist_locs[case_idx]}, "
                    raise AdfError(msg)
                case_climo_yrs = sorted(np.unique(case_climo_yrs))

                case_found_syr = int(case_climo_yrs[0])
                case_found_eyr = int(case_climo_yrs[-1])

                #Check if start or end year is missing.  If so then just assume it is the
                #start or end of the entire available model data.
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
                #End if
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
                #End if
            #End if

            #Update climo year lists in case anything changed
            syear = int(syear)
            eyear = int(eyear)
            syears_fixed.append(syear)
            eyears_fixed.append(eyear)

            #Update case name with provided/found years:
            case_name += f"_{syear}_{eyear}"

            #Set the final directory name and save it to plot_location:
            direc_name = f"{case_name}_vs_{data_name}"
            plot_loc = os.path.join(plot_dir, direc_name)
            self.__plot_location.append(plot_loc)

            #If first iteration, then save directory name for use by baseline:
            first_case_dir = ''
            if case_idx == 0:
                first_case_dir = direc_name
            #End if

            #Go ahead and make the diag plot location if it doesn't exist already
            diag_location = Path(plot_loc)
            print(f"\n\tDiagnostic Plot Location: {diag_location}")
            if not diag_location.is_dir():
                print("\tINFO: Directory not found, making new diagnostic plot location")
                diag_location.mkdir(parents=True)
        #End for

        self.__syears = syears_fixed
        self.__eyears = eyears_fixed

        #Finally add baseline case (if applicable) for use by the website table
        #generator.  These files will be stored in the same location as the first
        #listed case.
        if not self.compare_obs:
            self.__plot_location.append(os.path.join(plot_dir, first_case_dir))
        #End if
        #-------------------------------------------------------------------------

        #Initialize "num_procs" variable:
        #-----------------------------------------
        temp_num_procs = self.get_basic_info('num_procs')

        if not temp_num_procs:
            #Variable not present, so set to a single processor:
            self.__num_procs = 1
        else:
            #Check if variable is a string and matches keyword:
            if isinstance(temp_num_procs, str) and \
               temp_num_procs.strip() == "*":

                #Set number of processors to total number of CPUs
                #on the node.  Please note that at some point this
                #may need to be replaced with a DASK implementation
                #instead:

                #First try to get CPUs allowed by OS for process to use:
                try:
                    self.__num_procs = len(os.sched_getaffinity(0))
                except AttributeError:
                    #Operating system doesn't support getaffinity, so try
                    #straight CPU number:
                    if os.cpu_count():
                        self.__num_procs = os.cpu_count()
                    else:
                        #Something is weird with this Operating System,
                        #so warn user and then try to run in serial mode:
                        wmsg = "WARNING!!!! ADF unable to determine how"
                        wmsg += " many processors are availble on this system,"
                        wmsg += " so defaulting to a single process/core."
                        print(wmsg)
                        self.__num_procs = 1
                    #End if
                #End except
            else:
                #If anything else, then try to convert to integer:
                try:
                    self.__num_procs = int(temp_num_procs)
                except ValueError:
                    #This variable has been set to something that
                    #can't be converted into an integer, so warn
                    #user and then try to run in serial mode:
                    wmsg = "WARNING!!!!  The 'num_procs' variable"
                    wmsg += f" has been set to '{temp_num_procs}'"
                    wmsg += " which cannot be converted to an integer."
                    wmsg += "\nThe ADF will now default to a single core"
                    wmsg += " and attempt to run."
                    print(wmsg)
                    self.__num_procs = 1
                #End except
            #End if
        #End if
        #Print number of processors being used to debug log (if requested):
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

    # Create property needed to return "diag_var_list" list to user:
    @property
    def diag_var_list(self):
        """Return a copy of the "diag_var_list" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__diag_var_list)

    # Create property needed to return "basic_info" expanded dictionary to user:
    @property
    def basic_info_dict(self):
        """Return a copy of the "basic_info" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__basic_info)

    # Create property needed to return "basic_info" expanded dictionary to user:
    @property
    def cam_climo_dict(self):
        """Return a copy of the "cam_climo_dict" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__cam_climo_info)

    # Create property needed to return "basic_info" expanded dictionary to user:
    @property
    def baseline_climo_dict(self):
        """Return a copy of the "cam_bl_climo_info" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__cam_bl_climo_info)

    # Create property needed to return "num_procs" to user:
    @property
    def num_procs(self):
        """Return the "num_procs" logical to the user if requested."""
        return self.__num_procs

    # Create property needed to return "plot_location" variable to user:
    @property
    def plot_location(self):
        """Return a copy of the '__plot_location' string list to user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable:
        return copy.copy(self.__plot_location)

    # Create property needed to return the climo start (syear) and end (eyear) years to user:
    @property
    def climo_yrs(self):
        """Return the "syear" and "eyear" integer values to the user if requested."""
        syears = copy.copy(self.__syears) #Send copies so a script doesn't modify the original
        eyears = copy.copy(self.__eyears)
        return {"syears":syears,"eyears":eyears,
                "syear_baseline":self.__syear_baseline, "eyear_baseline":self.__eyear_baseline}

    # Create property needed to return the case nicknames to user:
    @property
    def case_nicknames(self):
        """Return the test case and baseline nicknames to the user if requested."""

        #Note that copies are needed in order to avoid having a script mistakenly
        #modify these variables, as they are mutable and thus passed by reference:
        test_nicknames = copy.copy(self.__test_nicknames)
        base_nickname = self.__base_nickname

        return {"test_nicknames":test_nicknames,"base_nickname":base_nickname}

    @property
    def hist_string(self):
        """ Return the CAM history string list to the user if requested."""
        cam_hist_strs = copy.copy(self.__hist_str)
        if self.__base_hist_str:
            base_hist_strs = copy.copy(self.__base_hist_str)
        else:
            base_hist_strs = ""
        hist_strs = {"test_hist_str":cam_hist_strs, "base_hist_str":base_hist_strs}
        return hist_strs


    #########

    #Utility function to access expanded 'diag_basic_info' variables:
    def get_basic_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_basic_info' as requested by
        the user.
        """

        return self.read_config_var(var_str,
                                    conf_dict=self.__basic_info,
                                    required=required)

    #########

    #Utility function to access expanded 'diag_cam_climo' variables:
    def get_cam_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_climo' as requested by
        the user.  """

        return self.read_config_var(var_str,
                                    conf_dict=self.__cam_climo_info,
                                    required=required)

    #########

    #Utility function to access expanded 'diag_cam_baseline_climo' variables:
    def get_baseline_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_baseline_climo' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        #Check if the cam baseline dictionary exists:
        if not self.__cam_bl_climo_info:
            #If required, then throw an error:
            if required:
                emsg = "get_baseline_info: Requested variable cannot be found"
                emsg += " because no baseline info exists.\n"
                emsg += "This is likely because an observational comparison is being done,"
                emsg += " so try adding 'required = False' to the get call."
                self.end_diag_fail(emsg)
            #End if

            #If not required, then return none:
            return None
        #End if

        #If basline dictionary exists, then search for variable like normal:
        return self.read_config_var(var_str,
                                    conf_dict=self.__cam_bl_climo_info,
                                    required=required)

    #########

    #Utility function to add a new model variable to the ADF (diag) variable list:
    def add_diag_var(self, var_str):
        """
        Adds a new variable to the ADF variable list
        """
        if var_str not in self.__diag_var_list:
            self.__diag_var_list.append(var_str)
        #End if

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

        #Grab variable list
        var_list = self.diag_var_list

        #Create "Path" objects:
        input_location  = Path(input_ts_loc)

        # Check that the time series input directory exists and can be read:
        ts_problem = utils.describe_dir_problem(input_location)
        if ts_problem:
            errmsg = (
                f"\t ERROR: Time series directory {ts_problem}.  Script is exiting."
            )
            raise AdfError(errmsg)

        #Normalize the configured history stream(s) into a list:
        if not hist_str:
            hist_strs = []
        elif isinstance(hist_str, str):
            hist_strs = [hist_str]
        else:
            hist_strs = [h for h in hist_str if h]
        #End if

        # Search for first available variable in var_list to get a time series file to read
        # NOTE: it is assumed all the variables have the same dates!
        # Try the configured stream(s) first, then the older, looser search
        # (any h0 stream) so pre-existing directories still work:
        def ts_patterns(var):
            """Search patterns for one variable, most specific first."""
            return ([f"{case_name}.{hstr}.{var}.*nc" for hstr in hist_strs]
                    + [f"{case_name}*h0*.{var}.*nc"])
        #End def

        #Sweep by pattern rank rather than by variable: every variable is tried
        #against the configured stream before any variable is tried against the
        #looser fallback, so a stray file from another stream cannot outrank
        #the stream the user actually configured.  Within a rank, the flat
        #search comes first for every variable and the recursive one only
        #afterwards -- a missing variable is normal, and recursing per pattern
        #would walk the whole tree once for every pattern tried.
        n_ranks = len(hist_strs) + 1
        ts_files = []
        skipped = var_list
        for rank in range(n_ranks):
            for recursive in (False, True):
                for idx, var in enumerate(var_list):
                    ts_files = utils.find_ts_files(input_location,
                                                   ts_patterns(var)[rank],
                                                   recursive=recursive)
                    if ts_files:
                        #Everything ahead of it was tried and missed:
                        skipped = var_list[:idx]
                        break
                    #End if
                #End for
                if ts_files:
                    break
                #End if
            #End for
            if ts_files:
                break
            #End if
        #End for

        #Report only the variables that were genuinely passed over.  Logging
        #inside the sweep would name every variable in diag_var_list whenever
        #the files turned out to be in a nested layout, which is not a problem
        #and not worth dozens of lines of debug log.
        for var in skipped:
            logmsg = "get years for time series:"
            logmsg += f"\n\tVar '{var}' not in dataset, skip to next to try and find climo years..."
            self.debug_log(logmsg)
        #End for

        if not ts_files:
            errmsg = f"\t ERROR: No time series files found in '{input_ts_loc}' for case "
            errmsg += f"'{case_name}' for any variable in 'diag_var_list'.  Script is exiting."
            raise AdfError(errmsg)
        #End if

        #The file names carry the period they hold, so use those rather than
        #opening the files: overlapping sets (years 1-20 alongside years 1-40,
        #left behind when a run is extended and the time series are remade)
        #cannot be opened together, and this runs during ADF setup, so failing
        #here takes down the whole run before any diagnostic starts.
        span = utils.ts_file_span(ts_files)
        if span and len(span[0]) >= 4:
            syr = int(span[0][:4])
            eyr = int(span[1][:4])
            _warn_on_long_climo_range(syr, eyr)
            return syr, eyr
        #End if

        #Read in file(s) -- names whose dates could not be read, so the period
        #has to come from the data itself:
        if len(ts_files) == 1:
            cam_ts_data = xr.open_dataset(ts_files[0], decode_times=True)
        else:
            cam_ts_data = xr.open_mfdataset(ts_files, decode_times=True, combine='by_coords')

        # Use the interval each step covers rather than its stamp, so that the
        # years found here are the years the data actually covers:
        cam_ts_data = utils.use_time_bounds_midpoint(cam_ts_data)

        #Extract first and last years from dataset:
        syr = int(cam_ts_data.time[0].dt.year.values)
        eyr = int(cam_ts_data.time[-1].dt.year.values)

        _warn_on_long_climo_range(syr, eyr)

        return syr, eyr


#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
