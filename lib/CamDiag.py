"""
Location of the "cam_diag" object, which
is used to store all relevant data and
info needed for generating CAM
diagnostics, including info
on the averaging, regridding, and
plotting methods themselves.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import sys
import os.path

#Check if "PyYAML" is present in python path:
try:
    import yaml
except ImportError:
    print("PyYAML module does not exist in python path.")
    print("Please install module, e.g. 'pip install pyyaml'.")
    sys.exit(1)

#+++++++++++++++++++++++++++++
#Add Cam diagnostics 'scripts'
#directories to Python path
#+++++++++++++++++++++++++++++

#Determine local directory path:
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

#Add "scripts" directory to path:
_DIAG_SCRIPTS_PATH = os.path.join(_LOCAL_PATH,os.pardir,"scripts")

#Check that "scripts" directory actually exists:
if not os.path.isdir(_DIAG_SCRIPTS_PATH):
    #If not, then raise error:
    raise FileNotFoundError("'{}' directory not found. Has 'CamDiag.py' been moved?".format(_DIAG_SCRIPTS_PATH))

#Walk through all sub-directories in "scripts" directory:
for root, dirs, files in os.walk(_DIAG_SCRIPTS_PATH):
    #Add all sub-directories to python path:
    for dirname in dirs:
        sys.path.append(os.path.join(root,dirname))

#################
#Helper functions
#################

#++++++++++++++++++++++++++++++++
#Script message and exit function
#++++++++++++++++++++++++++++++++

def end_diag_script(msg):

    """
    Prints message, and then exits script.
    """

    print("\n")
    print(msg)
    sys.exit(1)

def read_namelist_obj(namelist_obj, varname):

    """
    Checks if variable/list/dictionary exists in
    namelist object,and if so returns it.
    """

    try:
        var = namelist_obj[varname]
    except:
       raise KeyError("'{}' not found in namelist file.  Please see 'example_namelist.yaml'.".format(varname))

    #return variable/list/dictionary:
    return var

######################################
#Main CAM diagnostics class (cam_diag)
######################################

class CamDiag:

    """
    Main CAM diagnostics object.

    This object is initalized using
    a CAM diagnostics namelist (YAML) file,
    which specifies various user inputs,
    including CAM history file names and
    locations, years being analyzed,
    types of averaging, regridding,
    and other post-processing options being
    used, and the type of plots that will
    be created.

    This object also contains various methods
    used to actually generate the plots and
    post-processed data.
    """

    def __init__(self, namelist_file):

        """
        Initalize CAM diagnostics object.
        """

        #Check that YAML file actually exists:
        if not os.path.exists(namelist_file):
            raise FileNotFoundError("'{}' file not found.".format(namelist_file))

        #Open YAML file:
        with open(namelist_file) as nfil:
            #Load YAML file:
            namelist = yaml.load(nfil, Loader=yaml.SafeLoader)

        #Add basic diagnostic info to object:
        self.__basic_info = read_namelist_obj(namelist, 'diag_basic_info')

        #Check if CAM climatology files will be calculated:
        try:
            if namelist['diag_cam_climo']['calc_cam_climo']:
                self.__cam_climo_info = read_namelist_obj(namelist, 'diag_cam_climo')
        except:
            raise KeyError("'calc_cam_climo' in 'diag_cam_climo' not found in namelist file.  Please see 'example_namelist.yaml'.")

        #Check if CAM baseline climatology files will be calculated:
        if not self.__basic_info['compare_obs']:
            try:
                if namelist['diag_cam_baseline_climo']['calc_bl_cam_climo']:
                    self.__cam_bl_climo_info = read_namelist_obj(namelist, 'diag_cam_baseline_climo')
            except:
                raise KeyError("'calc_bl_cam_climo' in 'diag_cam_baseline_climo' not found in namelist file.  Please see 'example_namelist.yaml'.")

        #Add averaging script name:
        self.__averaging_script = read_namelist_obj(namelist, 'averaging_script')

        #Add regridding script name:
        self.__regridding_script = read_namelist_obj(namelist, 'regridding_script')

        #Add plotting script names:
        self.__plot_scripts = read_namelist_obj(namelist, 'plotting_scripts')

        #Add CAM variable list:
        self.__diag_var_list = read_namelist_obj(namelist, 'diag_var_list')

    def create_plots(self):

        """
        Generate CAM diagnositc plots.
        """

        #Notify user that plot creation is starting:
        print("Creating CAM diagnostic plots.")

        #How to actually run plotting scripts:
        #for plot_script in self.__plot_scripts:
           #Add '()' to script name:
        #   plot_func = plot_script+'()'
        #   eval(plot_func)


###############
