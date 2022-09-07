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

        #Read history file number from the yaml file
        hist_num = self.get_basic_info('hist_num')

        #If hist_num is not present, then default to 'h0':
        if not hist_num:
            hist_num = 'h0'
        #End if

        hist_str = '*.cam.'+hist_num

        #Initialize ADF variable list:
        self.__diag_var_list = self.read_config_var('diag_var_list', required=True)

        #Initialize "compare_obs" variable:
        self.__compare_obs = self.get_basic_info('compare_obs')

        #Check if a CAM vs AMWG obs comparison is being performed:
        if self.__compare_obs:

            #If so, then set the baseline info to None, to ensure any scripts
            #that check this variable won't crash:
            self.__cam_bl_climo_info = None

            #Also set data name for use below:
            data_name = "Obs"

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

            #if (syear_baseline and eyear_baseline) != "None":
            #    data_name += f"_{syear_baseline}_{eyear_baseline}"
            if syear_baseline and eyear_baseline == None:
                print("No given climo years for baseline...")
                baseline_hist_locs = self.get_baseline_info('cam_hist_loc',
                                                    required=True)
                starting_location = Path(baseline_hist_locs)
                files_list = sorted(starting_location.glob(hist_str+'.*.nc'))
                base_climo_yrs = sorted(np.unique([i.stem[-7:-3] for i in files_list]))
                syear_baseline = int(min(base_climo_yrs))
                eyear_baseline = int(max(base_climo_yrs))
                data_name += f"_{syear_baseline}_{eyear_baseline}"

            else:
                data_name += f"_{syear_baseline}_{eyear_baseline}"
        #End if
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
        if syears and eyears == None:
            syears = [None]*len(case_names)
            eyears = [None]*len(case_names)

        #Loop over cases:
        cam_hist_locs = self.get_cam_info('cam_hist_loc',
                                                  required=True)

        for case_idx, case_name in enumerate(case_names):

            if syears[case_idx] and eyears[case_idx] == None:
                print("No given climo years for case...")
                starting_location = Path(cam_hist_locs[case_idx])
                files_list = sorted(starting_location.glob(hist_str+'.*.nc'))
                case_climo_yrs = sorted(np.unique([i.stem[-7:-3] for i in files_list]))
                syear = int(min(case_climo_yrs))
                eyear = int(max(case_climo_yrs))
                case_name += f"_{syear}_{eyear}"

            else:
                case_name += f"_{syears[case_idx]}_{eyears[case_idx]}"
            #End if

            #Set the final directory name and save it to plot_location:
            direc_name = f"{case_name}_vs_{data_name}"
            self.__plot_location.append(os.path.join(plot_dir, direc_name))

            #If first iteration, then save directory name for use by baseline:
            if case_idx == 0:
                first_case_dir = direc_name
            #End if

        #End for

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
        then it must be required.
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

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++