"""
Base class for the Atmospheric
Diagnostics Framework (ADF).  All
other ADF classes inherit from this
class.

Currently this class only does two things:

1.  Creates a debug logger, if requested.

2.  Defines an ADF-specific function to end
    the diagnostics program, if need be.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import logging
from datetime import datetime

#+++++++++++++++++++++++++
# ADF Error-handling class
#+++++++++++++++++++++++++

class AdfError(RuntimeError):
    """Class used to handle ADF value errors
    (e.g., log user errors without backtrace)"""

#+++++++++++++++++
#Define base class
#+++++++++++++++++

class AdfBase:

    """
    Base class for the ADF
    """

    def __init__(self, debug = False):

        """
        Initalize CAM diagnostics object.
        """

        # Check that debug is in fact a boolean,
        # in order to avoid accidental boolean evaluation:
        if not isinstance(debug, bool):
            raise TypeError("'debug' must be a boolean type (True or False)")

        self.__debug_fname = ''

        # Create debug log, if requested:
        if debug:
            # Get the current date and time
            current_timestamp = datetime.now()
            # Format the datetime object to a string without microseconds
            dt_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            ext = f'{str(dt_str).replace(" ","-")}'
            debug_fname = f"ADF_debug_{ext}.log"
            self.__debug_fname = debug_fname
            logging.basicConfig(filename=debug_fname, level=logging.DEBUG)
            self.__debug_log = logging.getLogger("ADF")
        else:
            self.__debug_log = None

        

    #########

    # Create property needed to return the name of the debug log file (debug_fname) to user:
    @property
    def debug_fname(self):
        """Return the "debug_fname" string to the user."""
        return self.__debug_fname

    def debug_log(self, msg: str):

        """
        Write message to debug log, if enabled.
        """

        #If debug log exists, then write message to log:
        if self.__debug_log:
            self.__debug_log.debug(msg)

    #########

    def end_diag_fail(self, msg: str):

        """
        Prints message to log and screen,
        and then exits program with an
        ADF-specific error.
        """

        #Print message to log, if applicable:
        self.debug_log(msg)

        print("\n")
        raise AdfError(msg)

#++++++++++++++++++++
#End Class definition
#++++++++++++++++++++
