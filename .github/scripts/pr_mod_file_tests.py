#!/usr/bin/env python

"""
Script name:  pr_mod_file_tests.py

Goal:  To generate a list of files modified in the associated
       Github Pull Request (PR), using the PyGithub interface,
       and then to run tests on those files when appropriate.

Note:  This version currently limit the tests to a subset of files,
       in order to avoid running pylint on non-core python source files.

Written by:  Jesse Nusbaumer <nusbaume@ucar.edu> - November, 2020
"""

#+++++++++++++++++++++
#Import needed modules
#+++++++++++++++++++++

import sys
import os
import subprocess
import shlex
import argparse

from stat import S_ISREG
from github import Github

#Local scripts:
from pylint_threshold_test import pylint_check

#################

class PrModTestFail(ValueError):
    """Class used to handle file test failures
    (i.e., raise test failures without backtrace)"""

#################
#HELPER FUNCTIONS
#################

def _file_is_python(filename):

    """
    Checks whether a given file
    is a python script or
    python source code.
    """

    #Initialize return logical:
    is_python = False

    #Extract status of provided file:
    file_stat = os.stat(filename)

    #Check if it is a "regular" file:
    if S_ISREG(file_stat.st_mode):

        #Next, check if file ends in ".py":
        file_ext = os.path.splitext(filename)[1]

        if file_ext.strip() == ".py":
            #Assume file is python:
            is_python = True
        else:
            #If no ".py" extension exists, then
            #open the file and look for a shabang
            #that contains the word "python".
            with open(filename, "r") as mod_file:
                #Loop over lines in file:
                for line in mod_file:

                    #Ignore blank lines:
                    if line.strip():

                        #Check that first non-blank
                        #line is a shabang:
                        if line[0:2] == '#!':
                            #If so, then check that the word
                            #"python" is also present:
                            if line.find("python") != -1:
                                #If the word exists, then assume
                                #it is a python file:
                                is_python = True

                        #Exit loop, as only the first non-blank
                        #line should be examined:
                        break

    #Return file type result:
    return is_python

#++++++++++++++++++++++++++++++
#Input Argument parser function
#++++++++++++++++++++++++++++++

def parse_arguments():

    """
    Parses command-line input arguments using the argparse
    python module and outputs the final argument object.
    """

    #Create parser object:
    parser = argparse.ArgumentParser(description='Generate list of all files modified by pull request.')

    #Add input arguments to be parsed:
    parser.add_argument('--access_token', metavar='<GITHUB_TOKEN>', action='store', type=str,
                        help="access token used to access GitHub API")

    parser.add_argument('--pr_num', metavar='<PR_NUMBER>', action='store', type=int,
                        help="pull request number")

    parser.add_argument('--rcfile', metavar='<pylintrc file path>', action='store', type=str,
                       help="location of pylintrc file (full path)")

    parser.add_argument('--pylint_level', metavar='<number>', action='store', type=float,
                        required=False, help="pylint score that file(s) must exceed")

    #Parse Argument inputs
    args = parser.parse_args()
    return args

#############
#MAIN PROGRAM
#############

def _main_prog():

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    #++++++++++++
    #Begin script
    #++++++++++++

    print("Generating list of modified files...")

    #This should eventually be passed in via a command-line
    #argument, and include everything inside the "lib" directory -JN:
    testable_files = ["lib/adf_base.py",
                      "lib/adf_config.py",
                      "lib/adf_diag.py"]

    #+++++++++++++++++++++++
    #Read in input arguments
    #+++++++++++++++++++++++

    args = parse_arguments()

    #Add argument values to variables:
    token = args.access_token
    pr_num = args.pr_num
    rcfile = args.rcfile
    pylev = args.pylint_level

    #++++++++++++++++++++++++++++++++
    #Log-in to github API using token
    #++++++++++++++++++++++++++++++++

    ghub = Github(token)

    #++++++++++++++++++++
    #Open ESCOMP/CAM repo
    #++++++++++++++++++++

    #Official CAM repo:
    cam_repo = ghub.get_repo("NCAR/CAM_diagnostics")

    #++++++++++++++++++++++++++++++++++++++++++
    #Open Pull Request which triggered workflow
    #++++++++++++++++++++++++++++++++++++++++++

    pull_req = cam_repo.get_pull(pr_num)

    #++++++++++++++++++++++++++++++
    #Extract list of modified files
    #++++++++++++++++++++++++++++++

    #Create empty list to store python files:
    pyfiles = list()

    #Extract Github file objects:
    file_obj_list = pull_req.get_files()

    for file_obj in file_obj_list:

        #Check if file exists. If not,
        #then it was likely deleted in the
        #PR itself, so don't check its file type:
        if os.path.exists(file_obj.filename):

            #Check if it is a python file:
            if _file_is_python(file_obj.filename):
                #If so, then add to python list:
                pyfiles.append(file_obj.filename)

    #++++++++++++++++++++++++++++++++++++++++++++
    #Check if any python files are being modified:
    #++++++++++++++++++++++++++++++++++++++++++++
    if pyfiles in testable_files:

        testable_files = ["lib/adf_base.py",
                          "lib/adf_config.py",
                          "lib/adf_diag.py"]

        #Notify users of python files that will
        #be tested:
        print("The following modified python files will be tested:")
        for pyfile in pyfiles:
            if pyfile in testable_files:
                print(pyfile)
            else:
                continue

        #+++++++++++++++++++++++++
        #Run pylint threshold test
        #+++++++++++++++++++++++++

        lint_msgs = pylint_check(pyfiles, rcfile,
                                 threshold=pylev)

        #++++++++++++++++++
        #Check test results
        #++++++++++++++++++

        #If pylint check lists are non-empty, then
        #a test has failed, and an exception should
        #be raised with the relevant pytlint info:
        if lint_msgs:
            #Print pylint results for failed tests to screen:
            print("+++++++++++PYLINT FAILURE MESSAGES+++++++++++++")
            for lmsg in lint_msgs:
                print(lmsg)
            print("+++++++++++++++++++++++++++++++++++++++++++++++")

            #Raise test failure exception:
            fail_msg = "One or more files are below allowed pylint " \
                       "score of {}.\nPlease see pylint message(s) " \
                       "above for possible fixes."
            raise PrModTestFail(fail_msg)
        else:
            #All tests have passed, so exit normally:
            print("All pylint tests passed!")
            sys.exit(0)

    #If no python files exist in PR, then exit script:
    else:
        print("No python files present in PR, so there is nothing to test.")
        sys.exit(0)

#############################################

#Run the main script program:
if __name__ == "__main__":
    _main_prog()
