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
import glob
import subprocess
import importlib

from pathlib import Path

#Check if "PyYAML" is present in python path:
try:
    import yaml
except ImportError:
    print("PyYAML module does not exist in python path.")
    print("Please install module, e.g. 'pip install pyyaml'.")
    sys.exit(1)

#Check if "xarray" is present in python path:
try:
    import xarray as xr
except ImportError:
    print("Xarray module does not exist in python path.")
    print("Please install module, e.g. 'pip install xarray'.")
    sys.exit(1)

#Check if "numpy" is present in python path:
try:
    import numpy as np
except ImportError:
    print("Numpy module does not exist in python path.")
    print("Please install module, e.g. 'pip install numpy'.")
    sys.exit(1)

#Check if "matplolib" is present in python path:
try:
    import matplotlib as mpl
except ImportError:
    print("Matplotlib module does not exist in python path.")
    print("Please install module, e.g. 'pip install matplotlib'.")
    sys.exit(1)

#Check if "cartopy" is present in python path:
try:
    import cartopy.crs as ccrs
except ImportError:
    print("Cartopy module does not exist in python path.")
    print("Please install module, e.g. 'pip install Cartopy'.")
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

#####

def read_config_obj(config_obj, varname):

    """
    Checks if variable/list/dictionary exists in
    configure object,and if so returns it.
    """

    #Attempt to read in YAML config variable:
    try:
        var = config_obj[varname]
    except:
       raise KeyError("'{}' not found in config file.  Please see 'config_example.yaml'.".format(varname))

    #Check that configure variable is not empty (None):
    if var is None:
        raise NameError("'{}' has not been set to a value. Please see 'config_example.yaml'.".format(varname))

    #return variable/list/dictionary:
    return var

######################################
#Main CAM diagnostics class (cam_diag)
######################################

class CamDiag:

    """
    Main CAM diagnostics object.

    This object is initalized using
    a CAM diagnostics configure (YAML) file,
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

    def __init__(self, config_file):

        """
        Initalize CAM diagnostics object.
        """

        #Expand any environmental user name variables in the path:
        config_file = os.path.expanduser(config_file)

        #Check that YAML file actually exists:
        if not os.path.exists(config_file):
            raise FileNotFoundError("'{}' file not found.".format(config_file))

        #Open YAML file:
        with open(config_file) as nfil:
            #Load YAML file:
            config = yaml.load(nfil, Loader=yaml.SafeLoader)

        #Add basic diagnostic info to object:
        self.__basic_info = read_config_obj(config, 'diag_basic_info')

        #Add CAM climatology info to object:
        self.__cam_climo_info = read_config_obj(config, 'diag_cam_climo')

        #Check if CAM baseline climatology files will be calculated:
        if not self.__basic_info['compare_obs']:
            try:
                if config['diag_cam_baseline_climo']['calc_cam_climo']:
                    #If so, then add CAM baseline climatology info to object:
                    self.__cam_bl_climo_info = read_config_obj(config, 'diag_cam_baseline_climo')
            except:
                raise KeyError("'calc_bl_cam_climo' in 'diag_cam_baseline_climo' not found in config file.  Please see 'config_example.yaml'.")

        #Add averaging script names:
        self.__time_averaging_scripts = read_config_obj(config, 'time_averaging_scripts')

        #Add regridding script names:
        self.__regridding_scripts = read_config_obj(config, 'regridding_scripts')

        #Add analysis script names:
        self.__analysis_scripts = read_config_obj(config, 'analysis_scripts')

        #Add plotting script names:
        self.__plotting_scripts = read_config_obj(config, 'plotting_scripts')

        #Add CAM variable list:
        self.__diag_var_list = read_config_obj(config, 'diag_var_list')

        #Add CAM observation type list (filename prefix for observation files):
        self.__obs_type_list = read_config_obj(config, 'obs_type_list')

    # Create property needed to return "compare_obs" logical to user:
    @property
    def compare_obs(self):
        """Return the "compare_obs" logical to user if requested"""
        return self.__basic_info['compare_obs']

    #########

    def create_time_series(self, baseline=False):

        """
        Generate time series versions of the CAM history file data.
        """

        #Check if baseline time-series files are being created:
        if baseline:
            #Then use the CAM baseline climo dictionary
            #and case name:
            cam_climo_dict = self.__cam_bl_climo_info
            case_name = self.__basic_info['cam_baseline_case_name']
        else:
            #If not, then just extract the standard CAM climo dictionary
            #and case name::
            cam_climo_dict = self.__cam_climo_info
            case_name = self.__basic_info['cam_case_name']

        #Check if climatologies are being calculated:
        if cam_climo_dict['calc_cam_climo']:

            #Notify user that script has started:
            print("  Generating CAM time series files...")

            #Extract cam time series directory:
            ts_dir = cam_climo_dict['cam_ts_loc']

            #Extract start and end year values:
            start_year = int(cam_climo_dict['start_year'])
            end_year   = int(cam_climo_dict['end_year'])

            #Extract cam variable list:
            var_list = self.__diag_var_list

            #Create path object for the CAM history file(s) location:
            starting_location = Path(cam_climo_dict['cam_hist_loc'])

            #Create empty list:
            files_list = list()

            #Check if history files actually exist. If not then kill script:
            if not starting_location.glob('*.cam.h0.*.nc'):
                msg = "No CAM history (h0) files found in '{}'.  Script is ending here."
                msg = msg.format(starting_location)
                end_diag_script(msg)

            #Loop over start and end years:
            for year in range(start_year, end_year+1):
                #Add files to main file list:
                for fname in starting_location.glob('*.cam.h0.*{}-*.nc'.format(year)):
                    files_list.append(fname)

            #Create ordered list of CAM history files:
            hist_files = sorted(files_list)

            #Check if time series directory exists, and if not, then create it:
            if not os.path.isdir(ts_dir):
                print("    {} not found, making new directory".format(ts_dir))
                os.mkdir(ts_dir)

            #Loop over CAM history variables:
            for var in var_list:

                #Create full path name:
                ts_outfil_str = ts_dir + os.sep + case_name + \
                              ".ncrcat."+var+".nc"

                #Check if files already exist in time series directory:
                ts_file_list = glob.glob(ts_outfil_str)

                #If files exist, then check if over-writing is allowed:
                if ts_file_list:
                    if not cam_climo_dict['cam_overwrite_ts']:
                        #If not, then simply skip this variable:
                        continue

                #Notify user of new time series file:
                print("\t \u231B time series for {}".format(var))

                #Run "ncrcat" command to generate time series file:
                cmd = ["ncrcat", "-O", "-4", "-h", "-v", f"{var},hyam,hybm,hyai,hybi,P0,PS"] + hist_files + ["-o", ts_outfil_str]
                subprocess.run(cmd)

            #Notify user that script has ended:
            print("  ...CAM time series file generation has finished successfully.")

        else:
            #If not, then notify user that time series generation is skipped.
            print("  Climatology files are not being generated, so neither will time series files.")

    #########

    def create_climo(self, baseline=False):

        """
        Temporally average CAM time series data
        in order to generate CAM climatologies.

        The actual averaging is done using the
        scripts listed under "time_averaging_scripts"
        as specified in the config file.  This is done
        so that the user can specify the precise kinds
        of averaging that are done (e.g. weighted vs.
        non-weighted averaging).
        """

        #Check if CAM Baselines are being calculated:
        if baseline:
            #If so, then use the CAM baseline climo dictionary
            #case name, and output location:
            cam_climo_dict = self.__cam_bl_climo_info
            case_name  = self.__basic_info['cam_baseline_case_name']
            output_loc = self.__basic_info['cam_baseline_climo_loc']
        else:
            #If not, then just extract the standard CAM climo dictionary
            #case name, and output location:
            cam_climo_dict = self.__cam_climo_info
            case_name = self.__basic_info['cam_case_name']
            output_loc = self.__basic_info['cam_climo_loc']

        #Check if users wants climatologies to be calculated:
        if cam_climo_dict['calc_cam_climo']:

            #If so, then extract names of time-averaging scripts:
            avg_func_names = self.__time_averaging_scripts

            #Extract necessary variables from CAM configure dictionary:
            input_ts_loc    = cam_climo_dict['cam_ts_loc']
            overwrite_climo = cam_climo_dict['cam_overwrite_climo']
            var_list        = self.__diag_var_list

            #Loop over all averaging script names:
            for avg_func_name in avg_func_names:

                #Add file suffix to script name (to help with the file search):
                avg_script = avg_func_name+'.py'

                #Create full path to averaging script:
                avg_script_path = os.path.join(os.path.join(_DIAG_SCRIPTS_PATH,"averaging"),avg_script)

                #Check that file exists in "scripts/averaging" directory:
                if not os.path.exists(avg_script_path):
                    msg = "Time averaging file '{}' is missing. Script is ending here.".format(avg_script_path)
                    end_diag_script(msg)

                #Create averaging script import statement:
                avg_func_import_statement = "from {} import {}".format(avg_func_name, avg_func_name)

                #Run averaging script import statement:
                exec(avg_func_import_statement)

                #Create actual function call:
                avg_func = avg_func_name+'(case_name, input_ts_loc, output_loc, var_list, overwrite_climo)'

                #Evaluate (run) averaging script function:
                eval(avg_func)

        else:
            #If not, then notify user that climo file generation is skipped.
            print("  No climatology files were requested by user, so averaging will be skipped.")

    #########

    def regrid_climo(self):

        """
        Re-grid CAM climatology files to observations
        or baseline climatologies, in order to allow
        for direct comparisons.

        The actual regridding is done using the
        scripts listed under "regridding_scripts"
        as specified in the config file.  This is done
        so that the user can specify the precise kinds
        of re-gridding that are done (e.g. bilinear vs.
        nearest-neighbor regridding).
        """

        #Check if comparison is being made to observations:
        if self.compare_obs:
            #Set regridding target to observations:
            target_list = self.__obs_type_list
            target_loc  = self.__basic_info['obs_climo_loc']
        else:
            #Assume a CAM vs. CAM comparison is being run,
            #so set target to baseline climatologies:
            target_list = [self.__basic_info['cam_baseline_case_name']]
            target_loc  = self.__basic_info['cam_baseline_climo_loc']

        #Extract remaining required info from configure dictionaries:
        case_name        = self.__basic_info['cam_case_name']
        input_climo_loc  = self.__basic_info['cam_climo_loc']
        output_loc       = self.__basic_info['cam_regrid_loc']
        overwrite_regrid = self.__basic_info['cam_overwrite_regrid']
        var_list         = self.__diag_var_list

        #Extract names of re-gridding scripts:
        regrid_func_names = self.__regridding_scripts

        #Loop over all re-gridding script names:
        for regrid_func_name in regrid_func_names:

            #Add file suffix to script name (to help with the file search):
            regrid_script = regrid_func_name+'.py'

            #Create full path to regridding script:
            regrid_script_path = os.path.join(os.path.join(_DIAG_SCRIPTS_PATH,"regridding"),
                                              regrid_script)

            #Check that file exists in "scripts/regridding" directory:
            if not os.path.exists(regrid_script_path):
                msg = "Regridding file '{}' is missing. Script is ending here.".format(regrid_script_path)
                end_diag_script(msg)

            #Create regridding script import statement:
            regrid_func_import_statement = "from {} import {}".format(regrid_func_name, regrid_func_name)

            #Run regridding script import statement:
            exec(regrid_func_import_statement)

            #Create actual function call:
            regrid_func = regrid_func_name+\
            '(case_name, input_climo_loc, output_loc, var_list, target_list, target_loc, overwrite_regrid)'

            #Evaluate (run) averaging script function:
            eval(regrid_func)

    #########

    def perform_analyses(self, baseline=False):

        """
        Performs statistical and other analyses as specified by the
        user.  This currently only includes the AMWG table generation.

        This method also assumes that the analysis scripts require model
        inputs in a time series format.
        """

        #Extract names of plotting scripts:
        anly_func_names = self.__analysis_scripts

        #If no scripts are listed, then exit routine:
        if not anly_func_names:
            print("Nothing listed under 'analysis_scripts', so no plots will be made.")
            return

        #Check if CAM Baselines are being calculated:
        if baseline:
            #If so, then use the CAM baseline climo dictionary
            #case name, and output location:
            cam_climo_dict = self.__cam_bl_climo_info
            case_name  = self.__basic_info['cam_baseline_case_name']
        else:
            #If not, then just extract the standard CAM climo dictionary
            #case name, and output location:
            cam_climo_dict = self.__cam_climo_info
            case_name = self.__basic_info['cam_case_name']


        #Extract necessary variables from CAM configure dictionary:
        input_ts_loc    = cam_climo_dict['cam_ts_loc']
        output_loc      = self.__basic_info['cam_diag_plot_loc']
        write_html      = self.__basic_info['create_html']
        var_list        = self.__diag_var_list

        #Loop over all averaging script names:
        for anly_func_name in anly_func_names:

            #Add file suffix to script name (to help with the file search):
            anly_script = anly_func_name+'.py'

            #Create full path to averaging script:
            anly_script_path = os.path.join(os.path.join(_DIAG_SCRIPTS_PATH,"analysis"), anly_script)

            #Check that file exists in "scripts/analysis" directory:
            if not os.path.exists(anly_script_path):
                msg = "Analysis script file '{}' is missing. Script is ending here.".format(anly_script_path)
                end_diag_script(msg)

            #Create averaging script import statement:
            anly_func_import_statement = "from {} import {}".format(anly_func_name, anly_func_name)

            #Run averaging script import statement:
            exec(anly_func_import_statement)

            #Create actual function call:
            anly_func = anly_func_name+'(case_name, input_ts_loc, output_loc, var_list, write_html)'

            #Evaluate (run) averaging script function:
            eval(anly_func)

    #########

    def create_plots(self):

        """
        Generate CAM diagnositc plots.

        The actual plotting is done using the
        scripts listed under "plotting_scripts"
        as specified in the config file.  This is done
        so that the user can add their own plotting
        script(s) without having to modify the
        main CAM diagnostics routines.
        """

        #Extract names of plotting scripts:
        plot_func_names = self.__plotting_scripts

        #If no scripts are listed, then exit routine:
        if not plot_func_names:
            print("Nothing listed under 'plotting_scripts', so no plots will be made.")
            return

        #Extract required input variables:
        case_name       = self.__basic_info['cam_case_name']
        model_rgrid_loc = self.__basic_info['cam_regrid_loc']
        plot_location   = self.__basic_info['cam_diag_plot_loc']
        var_list        = self.__diag_var_list

        #Set "data" variables, which depend on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
            data_loc  = self.__basic_info['obs_climo_loc']
            data_list = self.__obs_type_list
        else:
            data_name = self.__basic_info['cam_baseline_case_name']
            data_loc  = self.__basic_info['cam_baseline_climo_loc']
            data_list = [data_name]

        #Loop over all re-gridding script names:
        for plot_func_name in plot_func_names:

            #Add file suffix to script name (to help with the file search):
            plot_script = plot_func_name+'.py'

            #Create full path to plotting scripts:
            plot_script_path = os.path.join(os.path.join(_DIAG_SCRIPTS_PATH,"plotting"),
                                            plot_script)

            #Check that file exists in "scripts/plotting" directory:
            if not os.path.exists(plot_script_path):
                msg = "Plotting file '{}' is missing. Script is ending here.".format(plot_script_path)
                end_diag_script(msg)

            #Create regridding script import statement:
            plot_func_import_statement = "from {} import {}".format(plot_func_name, plot_func_name)

            #Run regridding script import statement:
            exec(plot_func_import_statement)

            #Create actual function call:
            plot_func = plot_func_name+\
            '(case_name, model_rgrid_loc, data_name, data_loc, var_list, data_list, plot_location)'

            #Evaluate (run) plotting script function:
            eval(plot_func)

###############
