"""
Observations (obs) class for the Atmospheric
Diagnostics Framework (ADF).
This class inherits from the AdfInfo class.

Currently this class does three things:

1.  Initializes an instance of AdfInfo.

2.  Sets the "variable_defaults" ADF variable.

3.  Checks whether any requested variable is supposed
    to have a land or ocean mask, and if so then
    adds land and ocean fractions to the variable
    list.

4.  If a model vs obs run, then creates a
    dictionary of what observational dataset
    is associated with each requested variable,
    along with any relevant observational meta-data.

This class also provide methods for extracting
the observational data and meta-data for use
in various scripts.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import copy

from pathlib import Path

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

import yaml

#ADF modules:
from adf_info import AdfInfo

#+++++++++++++++++++
#Define Obs class
#+++++++++++++++++++

class AdfObs(AdfInfo):

    """
    Observations class, which initializes
    an AdfInfo object and provides
    additional variables and methods
    needed for managing observational data.
    """

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF Obs object.
        """

        #Initialize Config attributes:
        super().__init__(config_file, debug=debug)

        #Determine local directory:
        _adf_lib_dir = Path(__file__).parent

        # Check whether user wants to use defaults:
        #-----------------------------------------
        #Determine whether to use adf defaults or custom:
        _defaults_file = self.get_basic_info('defaults_file')
        if _defaults_file is None:
            _defaults_file = _adf_lib_dir/'adf_variable_defaults.yaml'
        else:
            print(f"\n\t Not using ADF default variables yaml file, instead using {_defaults_file}\n")
        #End if

        #Open YAML file:
        with open(_defaults_file, encoding='UTF-8') as dfil:
            self.__variable_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)

        _variable_defaults = self.__variable_defaults
        #-----------------------------------------

        #Check if land or ocean mask is requested, and if so then add OCNFRAC
        #to the variable list.  Note that this setting, and the defaults_file
        #code above, should probably be moved to AdfInfo, or somewhere else
        #farther down in the ADF inheritance chain:
        #----------------------------------------
        if self.__variable_defaults:
            #Variable defaults exist, so check if any want a land or ocean mask:
            for var in self.diag_var_list:
                #Check if any variable wants a land or ocean mask:
                if var in self.__variable_defaults:
                    if 'mask' in self.__variable_defaults[var]:
                        #Variable needs a mask, so add "OCNFRAC" to
                        #the variable list:
                        self.add_diag_var('OCNFRAC')
                        break
                   #End if
                #End if
            #End for
        #End if
        #-----------------------------------------

        #Initialize observations dictionary:
        self.__var_obs_dict = {}

        #If this is not a model vs obs run, then stop here:
        if not self.compare_obs:
            return
        #End if

        #Extract the "obs_data_loc" default observational data location:
        obs_data_loc = self.get_basic_info("obs_data_loc")
        print(obs_data_loc)
        #Loop over variable list:
        for var in self.diag_var_list:

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
        #End if

    #########

    # Create property needed to return "variable_defaults" variable to user:
    @property
    def variable_defaults(self):
        """Return a copy of the '__variable_defaults' dictionary to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__variable_defaults)

    # Create property needed to return "var_obs_dict" dictionary to user:
    @property
    def var_obs_dict(self):
        """Return a copy of the "var_obs_dict" list to the user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable, as it is mutable and thus passed by reference:
        return copy.copy(self.__var_obs_dict)

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
