"""
Collection of python unit tests
for the "AdfConfig" class.
"""

#+++++++++++++++++++++++
#Import required modules
#+++++++++++++++++++++++

import unittest
import sys
import os
import os.path

#Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)
_TEST_FILES_DIR = os.path.join(_CURRDIR, "test_files")

#Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

#Import AdfConfig class and AdfError
from adf_config import AdfConfig
from adf_base import AdfError

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Main AdfBase testing routine, used when script is run directly
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class AdfConfigTestRoutine(unittest.TestCase):

    """
    Runs all of the unit tests
    for the AdfConfig class.  Ideally
    this set of tests will provide
    complete code coverage for AdfConfig.
    """
    def test_AdfConfig_create(self):

        """
        Check that the AdfConfig class can
        be initialized properly.
        """

        #Use example config file:
        baseline_example_file = os.path.join(_ADF_LIB_DIR, os.pardir, "config_cam_baseline_example.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(baseline_example_file)

        #Assert that new object is of the "AdfConfig" class:
        self.assertIsInstance(adf_test, AdfConfig)

        #Also check that "read_config_var" works as expected:
        basic_diag_dict = adf_test.read_config_var("diag_basic_info")

        cam_case_name_val = adf_test.read_config_var("cam_case_name", conf_dict=basic_diag_dict)

        self.assertEqual(cam_case_name_val, "new_best.came-run")

    #####

    def test_AdfConfig_missing_file(self):

        """
        Check that AdfConfig throws the
        proper error when no config file
        is found.
        """

        #Set error message:
        ermsg = "File 'not_real.yaml' not found. Please provide full path."

        #Expect a FileNotFound error:
        with self.assertRaises(FileNotFoundError) as err:

            #Try and create AdfConfig object with non-existent file:
            adf_test = AdfConfig("not_real.yaml")

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_double_nested_config_var(self):

        """
        Check that AdfConfig throws the
        proper error when there is a
        doubly-nested variable present
        in the config (YAML) file.
        """

        #Use double-nested var config file:
        unset_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_double_nested.yaml")

        #Set error message:
        ermsg = "ADF currently only allows for a single nested dict"
        ermsg += " in the config (YAML) file.\n  Variable '{'double_nested_var': 'bad_val'}' is nested too far."

        #Expect an ADF error:
        with self.assertRaises(AdfError) as err:

            #Try and create AdfConfig object with doubly-nested config variable:
            adf_test = AdfConfig(unset_example_file)

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_read_config_bad_conf_dict(self):

        """
        Check that the "read_config_var"
        method throws the correct error
        when a non-dictionary is passed
        to "conf_dict".
        """

        #Use example config file:
        baseline_example_file = os.path.join(_ADF_LIB_DIR, os.pardir, "config_cam_baseline_example.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(baseline_example_file)

        #Set error message:
        ermsg = "Supplied 'conf_dict' variable should be a dictionary, not type '<class 'str'>'"

        #Expect a Type error:
        with self.assertRaises(TypeError) as err:

            #Try to read variable with bad "conf_dict" type:
            _ = adf_test.read_config_var("diag_basic_info", conf_dict="hello")

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_read_config_missing_var(self):

        """
        Check that the "read_config_var"
        method returns None when a
        non-required variable is requested that
        doesn't exist in the config dictionary.
        """

        #Use example config file:
        baseline_example_file = os.path.join(_ADF_LIB_DIR, os.pardir, "config_cam_baseline_example.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(baseline_example_file)

        #Try to read non-existing variable:
        conf_val = adf_test.read_config_var("hello")

        #Check that provided value is "None":
        self.assertEqual(conf_val, None)

    #####

    def test_AdfConfig_read_config_missing_required_var(self):

        """
        Check that the "read_config_var"
        method throws the correct error
        when a variable is requested that
        doesn't exist in the config dictionary,
        and is required.
        """

        #Use example config file:
        baseline_example_file = os.path.join(_ADF_LIB_DIR, os.pardir, "config_cam_baseline_example.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(baseline_example_file)

        #Set error message:
        #Note that for some reason a KeyError adds exra quotes,
        #hence the extra string quotes here
        ermsg = '''"Required variable 'hello' not found in config file. Please see 'config_cam_baseline_example.yaml'."'''

        #Expect a Key error:
        with self.assertRaises(KeyError) as err:

            #Try to read non-existing variable:
            _ = adf_test.read_config_var("hello", required=True)

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_read_config_unset_var(self):

        """
        Check that the "read_config_var"
        method returns None when a
        non-required variable is requested that
        exists but hasn't been set to a value.
        """

        #Use unset var config file:
        unset_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_unset_var.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(unset_example_file)

        #Try to read non-existing variable:
        conf_val = adf_test.read_config_var("bad_var")

        #Check that provided value is "None":
        self.assertEqual(conf_val, None)

    #####

    def test_AdfConfig_read_config_required_unset_var(self):

        """
        Check that the "read_config_var"
        method throws the correct error
        when a variable is requested that
        exists but hasn't been set to a value
        """

        #Use unset var config file:
        unset_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_unset_var.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(unset_example_file)

        #Set error message:
        ermsg = "Required variable 'bad_var' has not been set to a value. Please see 'config_cam_baseline_example.yaml'."

        #Expect a Value error:
        with self.assertRaises(ValueError) as err:

            #Try to read non-existing variable:
            _ = adf_test.read_config_var("bad_var", required=True)

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_expand_references(self):

        """
        Check that the AdfConfig class can
        properly expand variables using keywords
        """

        #Use example config file:
        keyword_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_keywords.yaml")

        #Create AdfConfig object:
        adf_test = AdfConfig(keyword_example_file)

        #Check that variables match pre-expansion:
        test_dict = adf_test.read_config_var("good_dict")
        test_dict_two = adf_test.read_config_var("good_dict_two")

        test_var = adf_test.read_config_var("good_var", conf_dict=test_dict)
        test_var_two = adf_test.read_config_var("good_var", conf_dict=test_dict_two)

        self.assertEqual(test_var, "It says ${test_var} and ${another_var}.")
        self.assertEqual(test_var_two, "${good_dict.good_var}")

        #Now expand variable references and check results:
        adf_test.expand_references(test_dict)
        adf_test.expand_references(test_dict_two)

        test_var_expanded = adf_test.read_config_var("good_var", conf_dict=test_dict)
        test_var_two_expanded = adf_test.read_config_var("good_var", conf_dict=test_dict_two)

        self.assertEqual(test_var_expanded, "It says yay! and 5.")
        self.assertEqual(test_var_two_expanded, "It says yay! and 5.")

    #####

    def test_AdfConfig_expand_references_non_specific_var(self):

       """
       Check that expand_references throws
       the correct error when a variable
       is used in a keyword that is defined
       in multiple different locations
       """

       #Use example config file:
       keyword_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_keywords.yaml")

       #Create AdfConfig object:
       adf_test = AdfConfig(keyword_example_file)

       #Check that variable matches pre-expansion:
       test_dict = adf_test.read_config_var("bad_dict")

       test_var = adf_test.read_config_var("bad_var", conf_dict=test_dict)

       self.assertEqual(test_var, "${good_var}")

       #Set error message:
       ermsg = "More than one variable matches keyword '${good_var}'"
       ermsg += "\nPlease use '${section.variable}' keyword method to specify"
       ermsg += " which variable you want to use."

       #Expect an ADF error:
       with self.assertRaises(AdfError) as err:

           #Now check for failure when variable reference is expanded:
           adf_test.expand_references(test_dict)

       #Check that error message matches what's expected:
       self.assertEqual(ermsg, str(err.exception))

    #####

    def test_AdfConfig_expand_references_non_existent_var(self):

       """
       Check that expand_references throws
       the correct error when a variable
       is used in a keyword that doesn't
       actually exist in the config file.
       """

       #Use example config file:
       keyword_example_file = os.path.join(_TEST_FILES_DIR, "config_cam_keywords.yaml")

       #Create AdfConfig object:
       adf_test = AdfConfig(keyword_example_file)

       #Check that variable matches pre-expansion:
       test_dict = adf_test.read_config_var("bad_dict_two")

       test_var = adf_test.read_config_var("bad_var_two", conf_dict=test_dict)

       self.assertEqual(test_var, "${no_var}")

       #Set error message:
       ermsg = f"ERROR: Variable 'no_var'"
       ermsg += " not found in config (YAML) file."

       #Expect an ADF error:
       with self.assertRaises(AdfError) as err:

           #Now check for failure when variable reference is expanded:
           adf_test.expand_references(test_dict)

       #Check that error message matches what's expected:
       self.assertEqual(ermsg, str(err.exception))


#++++++++++++++++++++++++++++++++++++++++++++++++
#Run unit tests if this script is called directly
#++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    unittest.main()

