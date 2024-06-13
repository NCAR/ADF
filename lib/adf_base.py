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

        # Create debug log, if requested:
        if debug:
            from datetime import datetime
            # Get the current date and time
            current_timestamp = datetime.now()
            ext = f'{str(current_timestamp).replace(" ","-")}'
            logging.basicConfig(filename=f"ADF_debug_{ext}.log", level=logging.DEBUG)
            self.__debug_log = logging.getLogger("ADF")
        else:
            self.__debug_log = None

    #########

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
