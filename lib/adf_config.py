"""
Config class for the Atmospheric
Diagnostics Framework (ADF).
This class inherits from the AdfBase class.

Currently this class does three things:

1.  Initializes an instance of AdfBase.

2.  Reads in a config (YAML) file.

3.  Expands any keywords into their relevant variable values.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import os.path
import re
import copy
import os
import subprocess

#+++++++++++++++++++++++++++++++++++++++++++++++++
#import non-standard python modules, including ADF
#+++++++++++++++++++++++++++++++++++++++++++++++++

import yaml
from adf_base import AdfBase

#+++++++++++++++++++
#Define config class
#+++++++++++++++++++

class AdfConfig(AdfBase):

    """
    Config class, which reads in
    config (YAML) files and provides
    a mechanism to process and retreive
    relevant config variables.
    """

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF Config object.
        """

        #Initialize Base attributes:
        super().__init__(debug=debug)

        #Expand any environmental user name variables in the path:
        config_file = os.path.expanduser(config_file)

        #Check that YAML file actually exists:
        if not os.path.exists(config_file):
            emsg = f"File '{config_file}' not found. Please provide full path."
            raise FileNotFoundError(emsg)

        #Open YAML file:
        with open(config_file, encoding='UTF-8') as nfil:
            #Load YAML file:
            self.__config_dict = yaml.load(nfil, Loader=yaml.SafeLoader)

        #Create search dictionary for variable expansion:
        self.__search_dict = self.__create_search_dict(self.__config_dict)

        #Create YAML self-reference keyword regex:
        self.__kword_pattern = re.compile(r'\$\{[a-z_\.\d]+\}')

    #########

    def __create_search_dict(self, config_dict, sub_dict=None):

        """
        Recursive function that creates a non-hierarchical
        dictionary for use in global key/value searches.

        Please note that PyYAML doesn't allow for multiple
        variables with the same name in the same dictionary,
        so no need to check here.
        """

        #Create empty dictionary:
        config_search_dict = {}

        #Loop over all top-level config variables:
        for key, value in config_dict.items():

            #Check if value is a string, integer, or another dict:
            if isinstance(value, (str, int)):

                #Check if sub dictionary is present:
                if sub_dict:
                    #Create new key with sub dictionary prefix:
                    key = sub_dict+"."+key

                #Add key/value to search dict:
                config_search_dict[key] = str(value)

            #Check if value is a dictionary instead:
            elif isinstance(value, dict):
                #Currently this routine only handles one level of
                #nested dictionaries, so throw an error if one has
                #gone beyond that:
                if sub_dict:
                    ermsg = "ADF currently only allows for a single nested dict"
                    ermsg += f" in the config (YAML) file.\n  Variable '{value}' is nested too far."
                    self.end_diag_fail(ermsg)

                #Apply routine to sub dictionary:
                sub_config_search_dict = self.__create_search_dict(value,
                                                                 sub_dict = key)

                #Append sub-dict search dictionary to top-level dictionary:
                config_search_dict.update(sub_config_search_dict)

        #Return search dictionary:
        return config_search_dict

    #########

    def __expand_yaml_var_ref(self, var_val):

        """
        Recursive function to replace all keywords with their
        associated values from the provided dictionary.
        """

        #If variable value is not a string, then convert it:
        if not isinstance(var_val, str):
            var_val = str(var_val)

        #Look for keyword using provided regular expression:
        kword_match = self.__kword_pattern.search(var_val)

        #Continue if at least one match is found:
        if kword_match:

            #Copy input variable value string,
            #which is needed for generating
            #proper error messages:
            new_var_val = var_val

            #Start while loop:
            another_match = True

            while another_match:

                #Extract match string:
                kword_match_str = kword_match.group(0)

                #Remove special characters ("${" and "}"):
                kword_match_str = kword_match_str[2:-1]

                #Initalize kword_match_str_key value:
                kword_match_str_key = kword_match_str

                #Check if period (".") is in string,
                #If so, then the keyword will be used directly,
                #otherwise do the following:
                #--------------------------
                if kword_match_str.find(".") == -1:

                    #Initalize match counter:
                    match_count = 0

                    #Loop through search dictionary keys:
                    for key in self.__search_dict:

                        #Attempt to find period string index:
                        pidx = key.find(".")

                        #Compare kword to text on the right side of period
                        #or at start of string if no period exists:
                        if kword_match_str == key[pidx+1:]:
                            #Set match string to full key string:
                            kword_match_str_key = key

                            #Add one to counter:
                            match_count += 1

                    #If more than one match, then throw an error:
                    if match_count > 1:
                        ermsg = f"More than one variable matches keyword '{var_val}'"
                        ermsg += "\nPlease use '${section.variable}' keyword method to specify"
                        ermsg += " which variable you want to use."
                        self.end_diag_fail(ermsg)
                #--------------------------

                #Throw an error if keyword not in dictionary:
                if kword_match_str_key not in self.__search_dict.keys():
                    ermsg = f"ERROR: Variable '{kword_match_str}'"
                    ermsg += " not found in config (YAML) file."
                    self.end_diag_fail(ermsg)

                #Extract keyword value from config dictionary:
                kword_val = self.__search_dict[kword_match_str_key]

                #Expand keyword if found:
                final_kword_val = self.__expand_yaml_var_ref(kword_val)

                #Substitute keyword with final kword_val:
                new_var_val = new_var_val[:kword_match.start()] + final_kword_val + \
                                                                new_var_val[kword_match.end():]

                #Search the string again for keyword values:
                kword_match = self.__kword_pattern.search(new_var_val)

                #End loop if no other matches found:
                if not kword_match:
                    another_match = False

            #End while loop

            #Pass back final value:
            return new_var_val
        #End if

        #Return the string un-modified:
        return var_val

    #########

    def expand_references(self, config_dict):

        """
        Replace keyword (${var} or ${dict.var}) entries
        in the YAML (config) dictionary that reference
        other YAML dictionary variables/keys with the
        values of those variables.

        Currently this function will always convert the
        referenced variable to a string.
        """

        #copy YAML config dictionary:
        config_dict_copy = copy.copy(config_dict)

        #Loop through dictionary:
        for key, value in config_dict_copy.items():

            #Skip non-strings (as they won't contain a keyword):
            if not isinstance(value, str):
                continue

            #expand any keywords to their full values:
            new_value = self.__expand_yaml_var_ref(value)

            #Set config variable to new, expanded value:
            config_dict[key] = new_value

    #########

    def read_config_var(self, varname, conf_dict=None, required=False):

        """
        Checks if variable/list/dictionary exists in
        configure dictionary,and if so returns it.

        Please note that in order to protect the values in the original
        YAML-defined config dictionary, only copies of the variable
        values are returned to the user.
        """

        #Check if the config dictionary has been specified:
        if isinstance(conf_dict, dict):
            var_dict = conf_dict
        elif isinstance(conf_dict, type(None)):
            var_dict = self.__config_dict
        else:
            emsg = "Supplied 'conf_dict' variable should be a dictionary,"
            emsg += f" not type '{type(conf_dict)}'"
            raise TypeError(emsg)
        #End if

        if varname == "ALL":
            return copy.deepcopy(var_dict)
        #End if

        #Check that variable name exists in dictionary:
        if varname not in var_dict.keys():
            #If variable is required, then throw an error:
            if required:
                emsg = f"Required variable '{varname}' not found in config file."
                emsg +=" Please see 'config_cam_baseline_example.yaml'."
                raise KeyError(emsg)
            #End if

            #If not, then just return None:
            return None
        #End if

        #Extract variable from dictionary:
        var = var_dict[varname]

        #If variable is required, then check that
        #configure variable is not empty (None):
        if required and var is None:
            emsg = f"Required variable '{varname}' has not been set to a value."
            emsg += " Please see 'config_cam_baseline_example.yaml'."
            raise ValueError(emsg)
        #End if

        #return a copy of the variable/list/dictionary,
        #this is done so that scripts can modify the copy
        #without worrying about modifying the actual
        #config variables dictionary:
        return copy.deepcopy(var)

    def config_dict(self):
        
        """
        Return a copy of the entire config dictionary.
        """

        config_dict = self.__config_dict
        return copy.copy(config_dict)

    def get_git_info(self):

        """
        Gather currnet Git info during ADF run.

        Returns:
        --------
        info : dict
            Dictionary with the following keys:
            - branch: Current Git branch name.
            - commit: Current commit hash.
            - remote_url: URL of the remote repository.
            - repo_name: Name of the repository.
            - is_dirty: Boolean indicating if there are uncommitted changes.
        """

        #Initialize empty dictionary:
        info = {}

        try:
            # Current branch
            branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                    stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
            info['branch'] = branch

            # Commit hash
            commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                    stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
            info['commit'] = commit

            # Remote URL
            remote_url = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                      stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
            info['remote_url'] = remote_url

            # Repo name
            info['repo_name'] = os.path.splitext(os.path.basename(remote_url))[0]

            # Status
            status = subprocess.run(['git', 'status', '--short'],
                                    stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
            info['is_dirty'] = bool(status)

        except subprocess.CalledProcessError as e:
            print("Git command failed:", e)
            info = None

        return info
    #########

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++