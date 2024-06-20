"""
Collection of python unit tests
for the "AdfBase" class.
"""

#+++++++++++++++++++++++
#Import required modules
#+++++++++++++++++++++++

import unittest
import sys
import os
import os.path
import logging
import glob

#Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

#Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

#Import AdfBase class
from adf_base import AdfBase
from adf_base import AdfError

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Main AdfBase testing routine, used when script is run directly
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class AdfBaseTestRoutine(unittest.TestCase):

    """
    Runs all of the unit tests
    for the AdfBase class.  Ideally
    this set of tests will provide
    complete code coverage for AdfBase.
    """

    #Set-up unit tests:
    def tearDown(self):

        """
        Remove log files (if they exist).
        """

        debug_list = glob.glob("ADF_debug*")

        #for dfile in debug_list:
        #    #Remove log file if it exists:
        #    if os.path.exists(dfile):
        #        os.remove(dfile)


        #Close all log streams:
        logging.shutdown()

    def test_AdfBase_create(self):

        """
        Check that the Adfbase class can
        be initialized properly.
        """

        #Create AdfBase object:
        adf_test = AdfBase()

        #Assert that new object is of the "AdfBase" class:
        self.assertIsInstance(adf_test, AdfBase)

    def test_AdfBase_debug_create(self):

        """
        Check that the Adfbase class can
        be initialized properly when the
        debug flag is set, and that a
        debug log file is created.
        """

        #Create AdfBase object with debug setting:
        adf_test = AdfBase(debug=True)

        #Grab debug log name
        debug_fname = adf_test.debug_fname

        #Assert that new object is of the "AdfBase" class:
        self.assertIsInstance(adf_test, AdfBase)

        #Assert that ADF debug log file exists in local directory:
        self.assertTrue(os.path.exists(debug_fname))

    def test_AdfBase_bad_debug(self):

        """
        Check that the Adfbase class
        throws the proper error if a bad
        value is passed-in via the "debug" variable.
        """

        #Set error message:
        ermsg = "'debug' must be a boolean type (True or False)"

        #Expect a Type error:
        with self.assertRaises(TypeError) as typerr:

            #Create AdfBase object with bad debug setting:
            adf_test = AdfBase(debug=5)

        #Check that error message matches what's expected:
        self.assertEqual(ermsg, str(typerr.exception))


    def test_AdfBase_debug_nothing(self):

        """
        Check that using the "debug_log" method
        without debugging enabled does nothing.
        """

        #Create AdfBase object with no debug setting:
        adf_test = AdfBase()

        #Call "debug_log" method:
        adf_test.debug_log("test")

        #Grab debug log name
        debug_fname = adf_test.debug_fname

        #Check that no log file exists:
        self.assertFalse(os.path.exists(debug_fname))

    def test_AdfBase_debug_write(self):

        """
        Check that using the "debug_log" method
        with debugging enabled properly writes
        a message to the debug log file.
        """

        #Create AdfBase object with debug setting:
        adf_test = AdfBase(debug=True)

        #Grab debug log name
        debug_fname = adf_test.debug_fname

        #Call "debug_log" method:
        adf_test.debug_log("test")

        print(debug_fname)

        #Check that debug log exists:
        self.assertTrue(os.path.exists(debug_fname))

        #If debug log exists, then open file:
        if os.path.exists(debug_fname):

            #Open log file:
            with open(debug_fname) as logfil:

                #Extract file contents:
                log_text = logfil.read()

                #Check that log text matches what was written:
                self.assertEqual("DEBUG:ADF:test\n",log_text)

    def test_AdfBase_script_end_fail(self):

        """
        Check that using "end_diag_fail" raises
        the correct exception and error message
        """

        #Create AdfBase object:
        adf_test = AdfBase()

        #Expect a Type error:
        with self.assertRaises(AdfError) as adferr:

            #Call "end_diag_fail" method:
            adf_test.end_diag_fail("test")

        #Check that error message matches what's expected:
        self.assertEqual("test", str(adferr.exception))

#++++++++++++++++++++++++++++++++++++++++++++++++
#Run unit tests if this script is called directly
#++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    unittest.main()

