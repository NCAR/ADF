"""
Observations (obs) class for the Atmospheric
Diagnostics Framework (ADF).
This class inherits from the AdfConfig class.

Currently this class does four things:

1.  Initializes an instance of AdfConfig.

2.  Sets the "variable_defaults" ADF variable.

3.  Determines if this is a model vs obs ADF run

4.  If so, then creates a dictionary of what
    observational dataset is associated with
    each requested variable, along with any
    relevant observational meta-data.

This class also provide methods for extracting
the observational data and meta-data for use
in various scripts.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import copy
import os

from pathlib import Path

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

import yaml

#ADF modules:
from adf_config import AdfConfig

#+++++++++++++++++++
#Define Obs class
#+++++++++++++++++++

class AdfObs(AdfConfig):

    """
    Observations class, which reads in
    config (YAML) files and provides
    a mechanism to process and retreive
    relevant config variables.
    """

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF Obs object.
        """

        #Initialize Config attributes:
        super().__init__(config_file, debug=debug)

        #Determine local directory:
        _adf_lib_dir = Path(__file__).parent

        #Read in "basic_info" config dictionary:
        _basic_info = self.read_config_var('diag_basic_info', required=True)

        #Expand basic info variable strings:
        self.expand_references(_basic_info)

        #Determin if variable defaults will be used:
        self.__use_defaults = self.read_config_var('use_defaults', conf_dict=_basic_info)

        # Check whether user wants to use defaults:
        #-----------------------------------------
        if self.__use_defaults:
            #Determine whether to use adf defaults or custom:
            _defaults_file = self.read_config_var('custom_defaults', conf_dict=_basic_info)
            if _defaults_file is None:
                _defaults_file = _adf_lib_dir/'adf_variable_defaults.yaml'

            #Open YAML file:
            with open(_defaults_file, encoding='UTF-8') as dfil:
                self.__variable_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)
        else:
            #Set variable_defaults to empty dictionary:
            self.__variable_defaults = {}
        #-----------------------------------------

        #Initialize ADF variable list:
        self.__diag_var_list = self.read_config_var('diag_var_list', required=True)

        #Initialize "compare_obs" variable:
        self.__compare_obs = self.read_config_var('compare_obs', conf_dict=_basic_info)

        #Initialize "num_procs" variable:
        #-----------------------------------------
        temp_num_procs = self.read_config_var('num_procs', conf_dict=_basic_info)

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

        #Initialize observations dictionary:
        self.__var_obs_dict = {}

        #If this is not a model vs obs run, then stop here:
        if not self.__compare_obs:
            return

        #Extract the "obs_data_loc" default observational data location:
        obs_data_loc = self.read_config_var("obs_data_loc", conf_dict=_basic_info)

        #Check that a variable defaults file exists (as it is currently needed to extract obs data):
        if not self.__variable_defaults:
            #Determine whether to use adf defaults or custom:
            _defaults_file = self.read_config_var('custom_defaults', conf_dict=_basic_info)
            if _defaults_file is None:
                _defaults_file = _adf_lib_dir/'adf_variable_defaults.yaml'

            #Open YAML file (but don't assign to object):
            with open(_defaults_file, encoding='UTF-8') as nfil:
                _variable_defaults = yaml.load(nfil, Loader=yaml.SafeLoader)
        else:
            #Set local variable to stored variable defaults dictionary:
            _variable_defaults = self.__variable_defaults
        #End if

        #Loop over variable list:
        for var in self.__diag_var_list:

            #Check if variable is in defaults dictionary:
            if var in _variable_defaults:
                #Extract variable sub-dictionary:
                default_var_dict = _variable_defaults[var]

                #Check if an observations file is specified:
                if "obs_file" in default_var_dict:
                    #Set found variable:
                    found = False

                    #Extract path/filename:
                    obs_file_path = Path(default_var_dict["obs_file"])

                    #Check if file exists:
                    if not obs_file_path.is_file():
                        #If not, then check if it is in "obs_data_loc"
                        if obs_data_loc:
                            obs_file_path = Path(obs_data_loc)/obs_file_path

                            if obs_file_path.is_file():
                                found = True

                    else:
                        #File was found:
                        found = True
                    #End if

                    #If found, then set observations dataset and variable names:
                    if found:
                        #Check if observations dataset name is specified:
                        if "obs_name" in default_var_dict:
                            obs_name = default_var_dict["obs_name"]
                        else:
                            #If not, then just use obs file name:
                            obs_name = obs_file_path.name

                        #Check if observations variable name is specified:
                        if "obs_var_name" in default_var_dict:
                            #If so, then set obs_var_name variable:
                            obs_var_name = default_var_dict["obs_var_name"]
                        else:
                            #Assume observation variable name is the same ad model variable:
                            obs_var_name = var
                        #End if

                        #Add variable to observations dictionary:
                        self.__var_obs_dict[var] = \
                            {"obs_file" : obs_file_path,
                             "obs_name" : obs_name,
                             "obs_var" : obs_var_name}

                    else:
                        #If not found, then print to log and skip variable:
                        msg = f'''Unable to find obs file '{default_var_dict["obs_file"]}' '''
                        msg += f"for variable '{var}'."
                        self.debug_log(msg)
                        continue
                    #End if

                else:
                    #No observation file was specified, so print
                    #to log and skip variable:
                    self.debug_log(f"No observations file was listed for variable '{var}'.")
                    continue
            else:
                #Variable not in defaults file, so print to log and skip variable:
                msg = f"Variable '{var}' not found in variable defaults file: `{_defaults_file}`"
                self.debug_log(msg)
            #End if
        #End for (var)

        #If variable dictionary is still empty, then print warning to screen:
        if not self.__var_obs_dict:
            wmsg = "!!!!WARNING!!!!\n"
            wmsg += "No observations found for any variables, but this is a model vs obs run!\n"
            wmsg += "ADF will still calculate time series and climatologies if requested,"
            wmsg += " but will stop there.\n"
            wmsg += "If this result is unexpected, then run with '--debug'"
            wmsg += " and check the log for messages.\n"
            wmsg += "!!!!!!!!!!!!!!!\n"
            print(wmsg)

    #########

    # Create property needed to return "variable_defaults" variable to user:
    @property
    def variable_defaults(self):
        """Return a copy of the '__variable_defaults' dictionary to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__variable_defaults)

    # Create property needed to return "use_defaults" variable to user:
    @property
    def use_defaults(self):
        """Return the '__use_defaults' logical to the user if requested."""
        return self.__use_defaults

    # Create property needed to return "compare_obs" logical to user:
    @property
    def compare_obs(self):
        """Return the "compare_obs" logical to the user if requested."""
        return self.__compare_obs

    # Create property needed to return "diag_var_list" list to user:
    @property
    def diag_var_list(self):
        """Return a copy of the "diag_var_list" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__diag_var_list)

    # Create property needed to return "var_obs_dict" dictionary to user:
    @property
    def var_obs_dict(self):
        """Return a copy of the "var_obs_dict" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__var_obs_dict)

    # Create property needed to return "num_procs" to user:
    @property
    def num_procs(self):
        """Return the "num_procs" logical to the user if requested."""
        return self.__num_procs

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
