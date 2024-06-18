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

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

# pylint: disable=unused-import
import numpy as np
import xarray as xr
# pylint: enable=unused-import

#ADF modules:
from adf_config import AdfConfig
from adf_base   import AdfError

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
        # End if

        # Add MDTF info to object:
        self.__mdtf_info = self.read_config_var("diag_mdtf_info")

        if self.__mdtf_info is not None:
            self.expand_references(self.__mdtf_info)
        # End if

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
                #If not a list, then convert it to one:
                self.__cam_climo_info[conf_var] = [conf_val]
            #End if
        #End for
        #-------------------------------------------

        #Read hist_str (component.hist_num) from the yaml file, or set to default
        hist_str = self.__cam_climo_info('hist_str')
        #If hist_str is not present, then default to 'cam.h0':
        if not hist_str:
            hist_str = [['cam.h0']]*self.__num_cases
        #End if
        self.__hist_str = hist_str

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

        #Initialize "compare_obs" variable:
        self.__compare_obs = self.get_basic_info('compare_obs')

        #Check if a CAM vs AMWG obs comparison is being performed:
        if self.__compare_obs:

            #If so, then set the baseline info to None, to ensure any scripts
            #that check this variable won't crash:
            self.__cam_bl_climo_info = None

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
                found_syear_baseline, found_eyear_baseline = self.get_climo_yrs_from_ts(input_ts_loc, data_name)
                found_yr_range = np.arange(found_syear_baseline,found_eyear_baseline,1)

                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
                if syear_baseline is None:
                    msg = f"No given start year for {data_name}, "
                    msg += f"using first found year: {found_syear_baseline}"
                    print(msg)
                    syear_baseline = found_syear_baseline
                if syear_baseline not in found_yr_range:
                    msg = f"Given start year '{syear_baseline}' is not in current dataset "
                    msg += f"{data_name}, using first found year: {found_syear_baseline}\n"
                    print(msg)
                    syear_baseline = found_syear_baseline

                if eyear_baseline is None:
                    msg = f"No given end year for {data_name}, "
                    msg += f"using last found year: {found_eyear_baseline}"
                    print(msg)
                    eyear_baseline = found_eyear_baseline
                if eyear_baseline not in found_yr_range:
                    msg = f"Given end year '{eyear_baseline}' is not in current dataset "
                    msg += f"{data_name}, using first found year: {found_eyear_baseline}\n"
                    print(msg)
                    eyear_baseline = found_eyear_baseline
            # End if

            # Check if history file path exists:
            if any(baseline_hist_locs):
                hist_str = baseline_hist_str[0]
                starting_location = Path(baseline_hist_locs)
                file_list = sorted(starting_location.glob("*" + hist_str + ".*.nc"))
                # Partition string to find exactly where h-number is
                # This cuts the string before and after the `{hist_str}.` sub-string
                # so there will always be three parts:
                # before sub-string, sub-string, and after sub-string
                #Since the last part always includes the time range, grab that with last index (2)
                #NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                base_climo_yrs = [int(str(i).partition(f"{hist_str}.")[2][0:4]) for i in file_list]
                base_climo_yrs = sorted(np.unique(base_climo_yrs))

                base_found_syr = int(base_climo_yrs[0])
                base_found_eyr = int(base_climo_yrs[-1])

                #Check if start or end year is missing. If so then just assume it is the
                #start or end of the entire available model data.
                if syear_baseline is None:
                    msg = f"No given start year for {data_name}, "
                    msg += f"using first found year: {base_found_syr}"
                    print(msg)
                    syear_baseline = base_found_syr
                if syear_baseline not in base_climo_yrs:
                    msg = f"Given start year '{syear_baseline}' is not in current dataset "
                    msg += f"{data_name}, using first found year: {base_climo_yrs[0]}\n"
                    print(msg)
                    syear_baseline = base_found_syr

                if eyear_baseline is None:
                    msg = f"No given end year for {data_name}, "
                    msg += f"using last found year: {base_found_eyr}"
                    print(msg)
                    eyear_baseline = base_found_eyr
                if eyear_baseline not in base_climo_yrs:
                    msg = f"Given end year '{eyear_baseline}' is not in current dataset "
                    msg += f"{data_name}, using last found year: {base_climo_yrs[-1]}\n"
                    print(msg)
                    eyear_baseline = base_found_eyr

                #Grab baseline nickname
                base_nickname = self.get_baseline_info('case_nickname')
                if base_nickname is None:
                    base_nickname = data_name
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
            syears = [None]*len(case_names)
        #End if
        if eyears is None:
            eyears = [None]*len(case_names)
        #End if

        #Extract cam history files location:
        cam_hist_locs = self.get_cam_info('cam_hist_loc')

        # Read hist_str (component.hist_num, eg cam.h0) from the yaml file
        cam_hist_str = self.__hist_str

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

                #Get years from pre-made timeseries file(s)
                found_syear, found_eyear = self.get_climo_yrs_from_ts(input_ts_loc, case_name)
                found_yr_range = np.arange(found_syear,found_eyear,1)

                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
                if syear is None:
                    msg = f"No given start year for {case_name}, "
                    msg += f"using first found year: {found_syear}"
                    print(msg)
                    syear = found_syear
                if syear not in found_yr_range:
                    msg = f"Given start year '{syear}' is not in current dataset "
                    msg += f"{case_name}, using first found year: {found_syear}\n"
                    print(msg)
                    syear = found_syear
                #End if
                if eyear is None:
                    msg = f"No given end year for {case_name}, "
                    msg += f"using last found year: {found_eyear}"
                    print(msg)
                    eyear = found_eyear
                if eyear not in found_yr_range:
                    msg = f"Given end year '{eyear}' is not in current dataset "
                    msg += f"{case_name}, using last found year: {found_eyear}\n"
                    print(msg)
                    eyear = found_eyear
                #End if
            #End if

            #Check if history file path exists:
            hist_str_case = cam_hist_str[case_idx]
            if any(cam_hist_locs):
                hist_str = hist_str_case[0]

                #Get climo years for verification or assignment if missing
                starting_location = Path(cam_hist_locs[case_idx])
                file_list = sorted(starting_location.glob('*'+hist_str+'.*.nc'))
                #Partition string to find exactly where h-number is
                #This cuts the string before and after the `{hist_str}.` sub-string
                # so there will always be three parts:
                # before sub-string, sub-string, and after sub-string
                #Since the last part always includes the time range, grab that with last index (2)
                #NOTE: this is based off the current CAM file name structure in the form:
                #  $CASE.cam.h#.YYYY<other date info>.nc
                case_climo_yrs = [int(str(i).partition(f"{hist_str}.")[2][0:4]) for i in file_list]
                case_climo_yrs = sorted(np.unique(case_climo_yrs))

                case_found_syr = int(case_climo_yrs[0])
                case_found_eyr = int(case_climo_yrs[-1])

                #Check if start or end year is missing.  If so then just assume it is the
                #start or end of the entire available model data.
                if syear is None:
                    msg = f"No given start year for {case_name}, "
                    msg += f"using first found year: {case_found_syr}"
                    print(msg)
                    syear = case_found_syr
                if syear not in case_climo_yrs:
                    msg = f"Given start year '{syear}' is not in current dataset "
                    msg += f"{case_name}, using first found year: {case_climo_yrs[0]}\n"
                    print(msg)
                    syear = case_found_syr
                #End if
                if eyear is None:
                    msg = f"No given end year for {case_name}, "
                    msg += f"using last found year: {case_found_eyr}"
                    print(msg)
                    eyear = case_found_eyr
                if eyear not in case_climo_yrs:
                    msg = f"Given end year '{eyear}' is not in current dataset "
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
            if not diag_location.is_dir():
                print(f"\t    {diag_location} not found, making new directory")
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
        # -----------------------------------------

    #########
    def hist_str_to_list(self, conf_var, conf_val):
        """
        Make hist_str a nested list [ncases,nfiles] of the given value(s)
        """
        if isinstance(conf_val, list):
            hist_str = conf_val
        else:  # one case, one hist str
            hist_str = [
                conf_val
            ]
        self.__cam_climo_info[conf_var] = [hist_str]
        # -----------------------------------------

    #########

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
        """ Return the history string name to the user if requested."""
        return self.__hist_str

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
    def get_climo_yrs_from_ts(self, input_ts_loc, case_name):
        """
        Grab start and end climo years if none are specified in config file
        for pre-made time series file(s)

        Return
        ------
          - start year
          - end year
        """

        #Grab variable list
        var_list = self.diag_var_list

        #Create "Path" objects:
        input_location  = Path(input_ts_loc)

        #Check that time series input directory actually exists:
        if not input_location.is_dir():
            errmsg = f"Time series directory '{input_ts_loc}' not found.  Script is exiting."
            raise AdfError(errmsg)

        # Search for first variable in var_list to get a time series file to read
        # NOTE: it is assumed all the variables have the same dates!
        # Also, it is assumed that only h0 files should be climo-ed.
        ts_files = sorted(input_location.glob(f"{case_name}*h0*.{var_list[0]}.*nc"))

        #Read in file(s)
        if len(ts_files) == 1:
            cam_ts_data = xr.open_dataset(ts_files[0], decode_times=True)
        else:
            cam_ts_data = xr.open_mfdataset(ts_files, decode_times=True, combine='by_coords')

        #Average time dimension over time bounds, if bounds exist:
        if 'time_bnds' in cam_ts_data:
            time = cam_ts_data['time']
            #NOTE: force `load` here b/c if dask & time is cftime,
            #throws a NotImplementedError:
            time = xr.DataArray(cam_ts_data['time_bnds'].load().mean(dim='nbnd').values, 
                                dims=time.dims, attrs=time.attrs)
            cam_ts_data['time'] = time
            cam_ts_data.assign_coords(time=time)
            cam_ts_data = xr.decode_cf(cam_ts_data)

        #Extract first and last years from dataset:
        syr = int(cam_ts_data.time[0].dt.year.values)
        eyr = int(cam_ts_data.time[-1].dt.year.values)

        if eyr-syr >= 100:
            msg = f"WARNING: the found climo year range is large: {eyr-syr} years, "
            msg += "this may take a long time!"
            print(msg)

        return syr, eyr

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
