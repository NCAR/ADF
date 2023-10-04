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
import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

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

        #Check if inputs are of the correct type:
        #-------------------------------------------

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
            if isinstance(conf_val, list):
                #If a list, then make sure it is has the correct number of entries:
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
        hist_str = self.get_basic_info('hist_str')
        #If hist_str is not present, then default to 'cam.h0':
        if not hist_str:
            hist_str = 'cam.h0'
        #End if

        #Initialize ADF variable list:
        self.__diag_var_list = self.read_config_var('diag_var_list', required=True)

        #Case names:
        case_names = self.get_cam_info('cam_case_name', required=True)

        print(case_names)

        #Grab test case nickname(s)
        test_nicknames = self.get_cam_info('case_nickname')
        if test_nicknames is None:
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
            syear_baseline = int(f"{str(syear_baseline).zfill(4)}")
            eyear_baseline = self.get_baseline_info('end_year')
            eyear_baseline = int(f"{str(eyear_baseline).zfill(4)}")

            #Get climo years for verification or assignment if missing
            baseline_hist_locs = self.get_baseline_info('cam_hist_loc')

            #Check if history file path exists:
            if baseline_hist_locs:

                starting_location = Path(baseline_hist_locs)
                files_list = sorted(starting_location.glob('*'+hist_str+'.*.nc'))
                base_climo_yrs = sorted(np.unique([i.stem[-7:-3] for i in files_list]))

                #Check if start or end year is missing.  If so then just assume it is the
                #start or end of the entire available model data.
                if syear_baseline is None:
                    print(f"No given start year for {data_name}, using first found year...")
                    syear_baseline = int(base_climo_yrs[0])
                elif str(syear_baseline) not in base_climo_yrs:
                    print(f"Given start year '{syear_baseline}' is not in current dataset {data_name}, using first found year:",base_climo_yrs[0],"\n")
                    syear_baseline = int(base_climo_yrs[0])
                #End if
                if eyear_baseline is None:
                    print(f"No given end year for {data_name}, using last found year...")
                    eyear_baseline = int(base_climo_yrs[-1])
                elif str(eyear_baseline) not in base_climo_yrs:
                    print(f"Given end year '{eyear_baseline}' is not in current dataset {data_name}, using last found year:",base_climo_yrs[-1],"\n")
                    eyear_baseline = int(base_climo_yrs[-1])
                #End if

                #Grab baseline nickname
                base_nickname = self.get_baseline_info('case_nickname')
                if base_nickname == None:
                    base_nickname = data_name

            else:
                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
                if syear_baseline is None or eyear_baseline is None:
                    emsg = "Missing starting year ('start_year') and final year ('end_year') "
                    emsg += "entries in the 'diag_cam_baseline_climo' config section.\n"
                    emsg += "These are required if the ADF is running "
                    emsg += "directly from time series files for the basline case."
                    raise AdfError(emsg)
                #End if
            #End if

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
        cam_hist_locs = self.get_cam_info('cam_hist_loc',required=False)


        #Loop over cases:
        syears_fixed = []
        eyears_fixed = []
        for case_idx, case_name in enumerate(case_names):

            syear = int(f"{str(syears[case_idx]).zfill(4)}")
            syears_fixed.append(syear)
            eyear = int(f"{str(eyears[case_idx]).zfill(4)}")
            eyears_fixed.append(eyear)

            #Check if history file path exists:

            if any(cam_hist_locs):
                #Get climo years for verification or assignment if missing
                starting_location = Path(cam_hist_locs[case_idx])
                files_list = sorted(starting_location.glob('*'+hist_str+'.*.nc'))
                case_climo_yrs = sorted(np.unique([i.stem[-7:-3] for i in files_list]))

                #Check if start or end year is missing.  If so then just assume it is the
                #start or end of the entire available model data.
                if syear is None:
                    print(f"No given start year for {case_name}, using first found year...")
                    syear = int(case_climo_yrs[0])
                elif str(syear) not in case_climo_yrs:
                    print(f"Given start year '{syear}' is not in current dataset {case_name}, using first found year:",case_climo_yrs[0],"\n")
                    syear = int(case_climo_yrs[0])
                #End if
                if eyear is None:
                    print(f"No given end year for {case_name}, using last found year...")
                    eyear = int(case_climo_yrs[-1])
                elif str(eyear) not in case_climo_yrs:
                    print(f"Given end year '{eyear}' is not in current dataset {case_name}, using last found year:",case_climo_yrs[-1],"\n")
                    eyear = int(case_climo_yrs[-1])
                #End if
            else:
                #History file path isn't needed if user is running ADF directly on time series.
                #So make sure start and end year are specified:
                if syears is None or eyears is None:
                    emsg = "Missing starting year ('start_year') and final year ('end_year') "
                    emsg += "entries in the 'diag_cam_climo' config section.\n"
                    emsg += "These are required if the ADF is running "
                    emsg += "directly from time series files for the test case(s)."
                    raise AdfError(emsg)
                #End if
            #End if

            #Update case name with provided/found years:
            case_name += f"_{syear}_{eyear}"

            #Set the final directory name and save it to plot_location:
            direc_name = f"{case_name}_vs_{data_name}"
            self.__plot_location.append(os.path.join(plot_dir, direc_name))

            #If first iteration, then save directory name for use by baseline:
            if case_idx == 0:
                first_case_dir = direc_name
            #End if

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
        #-----------------------------------------

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

    # Create property needed to return the climo start (syear) and end (eyear) years to user:
    @property
    def case_nicknames(self):
        """Return the test case and baseline nicknames to the user if requested."""

        #Note that copies are needed in order to avoid having a script mistakenly
        #modify these variables, as they are mutable and thus passed by reference:
        test_nicknames = copy.copy(self.__test_nicknames)
        base_nickname = self.__base_nickname

        return {"test_nicknames":test_nicknames,"base_nickname":base_nickname}

    #########

    #Utility function to access expanded 'diag_basic_info' variables:
    def get_basic_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_basic_info' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        return self.read_config_var(var_str,
                                    conf_dict=self.__basic_info,
                                    required=required)

    #########

    #Utility function to access expanded 'diag_cam_climo' variables:
    def get_cam_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_climo' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required. (DRB: This statement contradicts the default value of required=False)
        """

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

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
