"""
Location of the "AdfDiag" object, which
is used to store all relevant data and
info needed for generating CAM/ADF
diagnostics, including info
on the averaging, regridding, and
plotting methods themselves.
"""

#++++++++++++++++++++++++++++++
#Import standard python modules
#++++++++++++++++++++++++++++++

import sys
import os
import os.path
import glob
import ast
import subprocess
import importlib

from pathlib import Path
from typing import Optional

#Check if "PyYAML" is present in python path:
# pylint: disable=unused-import
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

# pylint: enable=unused-import

#+++++++++++++++++++++++++++++
#Add ADF diagnostics 'scripts'
#directories to Python path
#+++++++++++++++++++++++++++++

#Determine local directory path:
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

#Add "scripts" directory to path:
_DIAG_SCRIPTS_PATH = os.path.join(_LOCAL_PATH,os.pardir,"scripts")

#Check that "scripts" directory actually exists:
if not os.path.isdir(_DIAG_SCRIPTS_PATH):
    #If not, then raise error:
    ermsg = f"'{_DIAG_SCRIPTS_PATH}' directory not found. Has 'AdfDiag.py' been moved?"
    raise FileNotFoundError(ermsg)

#Walk through all sub-directories in "scripts" directory:
for root, dirs, files in os.walk(_DIAG_SCRIPTS_PATH):
    #Add all sub-directories to python path:
    for dirname in dirs:
        sys.path.append(os.path.join(root,dirname))

#+++++++++++++++++++++++++++++

#Finally, import needed ADF modules:
from adf_config import AdfConfig
from adf_base   import AdfError

#################
#Helper functions
#################

def construct_index_info(page_dict, fnam, opf):

    """
    Helper function for generating web pages.
    d : dictionary for the index page information
    fnam : the image filename, img.stem  --> then decompose the img file's parts.
    opf: outputfile for the image
    """
    vname, plot_desc = fnam[0:fnam.index("_")], fnam[fnam.index("_")+1:]
    if 'ANN' in plot_desc:
        temporal = 'ANN'
    elif 'DJF' in plot_desc:
        temporal = 'DJF'
    elif 'JJA' in plot_desc:
        temporal = 'JJA'
    elif 'MAM' in plot_desc:
        temporal = 'MAM'
    elif 'SON' in plot_desc:
        temporal = 'SON'
    else:
        temporal = 'NoInfo'
    plot_type = plot_desc.replace(temporal+"_", "")
    if vname not in page_dict:
        page_dict[vname] = {}
    if plot_type not in page_dict[vname]:
        page_dict[vname][plot_type] = {}
    page_dict[vname][plot_type][temporal] = opf

######################################
#Main ADF diagnostics class (AdfDiag)
######################################

class AdfDiag(AdfConfig):

    """
    Main ADF diagnostics object.

    This object is initalized using
    an ADF diagnostics configure (YAML) file,
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

    def __init__(self, config_file, debug=False):

        """
        Initalize ADF diagnostics object.
        """

        #Initialize Config/Base attributes:
        super().__init__(config_file, debug=debug)

        #Add basic diagnostic info to object:
        self.__basic_info = self.read_config_var('diag_basic_info', required=True)

        #Expand basic info variable strings:
        self.expand_references(self.__basic_info)

        #Add CAM climatology info to object:
        self.__cam_climo_info = self.read_config_var('diag_cam_climo', required=True)

        #Expand CAM climo info variable strings:
        self.expand_references(self.__cam_climo_info)

        #Check if a CAM vs AMWG obs comparison is being performed:
        if self.read_config_var('compare_obs', conf_dict=self.__basic_info):

            #Finally, set the baseline info to None, to ensure any scripts
            #that check this variable won't crash:
            self.__cam_bl_climo_info = None

        else:
            #If not, then assume a CAM vs CAM run and add CAM baseline climatology info to object:
            self.__cam_bl_climo_info = self.read_config_var('diag_cam_baseline_climo',
                                                            required=True)

            #Expand CAM baseline climo info variable strings:
            self.expand_references(self.__cam_bl_climo_info)

        #Add averaging script names:
        self.__time_averaging_scripts = self.read_config_var('time_averaging_scripts')

        #Add regridding script names:
        self.__regridding_scripts = self.read_config_var('regridding_scripts')

        #Add analysis script names:
        self.__analysis_scripts = self.read_config_var('analysis_scripts')

        #Add plotting script names:
        self.__plotting_scripts = self.read_config_var('plotting_scripts')

        #Add CAM variable list:
        self.__diag_var_list = self.read_config_var('diag_var_list', required=True)

        #Add CAM observation type list (filename prefix for observation files):
        self.obs_type_list = self.read_config_var('obs_type_list')

        #Create plot location variable for potential use by the website generator.
        #Please note that this variable is only set if "create_plots" or
        #is called:
        self.__plot_location = ""

    # Create property needed to return "compare_obs" logical to user:
    @property
    def compare_obs(self):
        """Return the "compare_obs" logical to user if requested."""
        return self.read_config_var('compare_obs', conf_dict=self.__basic_info)

    # Create property needed to return "create_html" logical to user:
    @property
    def create_html(self):
        """Return the "create_html" logical to user if requested."""
        return self.read_config_var('create_html', conf_dict=self.__basic_info)

    # Create property needed to return "diag_var_list" list to user:
    @property
    def diag_var_list(self):
        """Return the "diag_var_list"  list to user if requested."""
        return self.__diag_var_list

    # Create property needed to return "plot_location" variable to user:
    @property
    def plot_location(self):
        """Return the '__plot_location' string to user if requested."""
        return self.__plot_location

    #########
    #Variable extraction functions
    #########

    def get_basic_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_basic_info' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        return self.read_config_var(var_str,
                                    conf_dict=self.__basic_info,
                                    required=required)

    #########

    def get_cam_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_climo' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        return self.read_config_var(var_str,
                                    conf_dict=self.__cam_climo_info,
                                    required=required)

    #########

    def get_baseline_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cam_baseline_climo' as requested by
        the user.  This function assumes that if the user is requesting it,
        then it must be required.
        """

        #Check if the cam baseline dictionary exists:
        if not self.__cam_bl_climo_info:
            #If required, then throw an error:
            if required:
                emsg = "get_baseline_info: Requested variable cannot be found"
                emsg += " because no baseline info exists.\n"
                emsg += "This is likely because an observational comparison is being done,"
                emsg += " so try adding 'required = False' to the get call."
                raise AdfError(emsg)

            #If not required, then return none:
            return None

        #If basline dictionary exists, then search for variable like normal:
        return self.read_config_var(var_str,
                                    conf_dict=self.__cam_bl_climo_info,
                                    required=required)

    #########
    #Script-running functions
    #########

    def __diag_scripts_caller(self, scripts_dir: str, func_names: list,
                              default_kwargs: Optional[dict] = None,
                              log_section: Optional[str] = None):

        """
        Parse a list of scripts as provided by the config file,
        and call them as functions while passing in the correct inputs.

        scripts_dir    : string, sub-directory under "scripts" where scripts are located
        func_names     : list of function/scripts (either string or dictionary):
        default_kwargs : optional list of default keyword arguments for the scripts if
                         none are specified by the config file
        log_section    : optional variable that specifies where the log entries are coming from.
                         Note:  Is it better to just make a child log instead?
        """

        #Loop over all averaging script names:
        for func_name in func_names:

            #Check if func_name is a dictonary,
            #this implies that the function has user-defined inputs:
            if isinstance(func_name, dict):
                emsg = "Function dictionary must be of the form: "
                emsg += "{function_name : {kwargs:{...}, module:'xxxx'}}"
                assert len(func_name) == 1, emsg
                has_opt = True
                opt = func_name[list(func_name.keys())[0]]  # un-nests the dictionary
                # not ideal, but change to a str representation; iteration will continue ok:
                func_name = list(func_name.keys())[0]
            elif isinstance(func_name, str):
                has_opt = False
            else:
                raise TypeError("Provided script must either be a string or a dictionary.")

            func_script = func_name + '.py'  # default behavior: Add file suffix to script name
            if has_opt:
                if 'module' in opt:
                    func_script = opt['module']

            #Create full path to function script:
            func_script_path = \
                os.path.join(os.path.join(_DIAG_SCRIPTS_PATH, scripts_dir), func_script)

           #Check that file exists in specified directory:
            if not os.path.exists(func_script_path):
                emsg = f"Script file '{func_script_path}' is missing. Diagnostics are ending here."
                self.end_diag_fail(emsg)

            if func_script_path not in sys.path:
                #Add script path to debug log if requested:
                if log_section:
                    dmsg = f"{log_section}: Inserting to sys.path: {func_script_path}"
                    self.debug_log(dmsg)
                else:
                    dmsg = f"diag_scripts_caller: Inserting to sys.path: {func_script_path}"
                    self.debug_log(dmsg)

                #Add script to python path:
                sys.path.insert(0, func_script_path)

            # NOTE: when we move to making this into a proper package,
            #       this path-checking stuff should be removed and dealt with on the package-level.

            # Arguments; check if user has specified custom arguments
            func_kwargs = default_kwargs
            if has_opt:
                if 'kwargs' in opt:
                    func_kwargs = opt['kwargs']

            #Add function calls debug log if requested:
            if log_section:
                dmsg = f"{log_section}: \n \t func_name = {func_name}\n "
                dmsg += f"\t func_kwargs = {func_kwargs}"
                self.debug_log(dmsg)
            else:
                dmsg = f"diag_scripts_caller: \n \t func_name = {func_name}\n "
                dmsg += f"\t func_kwargs = {func_kwargs}"
                self.debug_log(dmsg)


            #Call function
            self.__function_caller(func_name,
                                   func_kwargs=func_kwargs,
                                   module_name=func_name)

    #########

    # pylint: disable=no-self-use

    def __function_caller(self, func_name: str,
                          func_kwargs: Optional[dict] = None, module_name=None):

        """
        Call a function with given arguments.

        func_name : string, name of the function to call
        func_kwargs : [optional] dict, the keyword arguments to pass to the function
        module_name : [optional] string, the name of the module where func_name is defined;
                      if not provided, assume func_name.py

        return : the output of func_name(self, **func_kwargs)
        """

        if module_name is None:
            module_name = func_name #+'.py'

        # note: when we use importlib, specify the module name without the ".py" extension.
        module = importlib.import_module(module_name)
        if hasattr(module, func_name) and callable(getattr(module, func_name)):
            func = getattr(module, func_name)
        else:
            emsg = f"Function '{func_name}' cannot be found in module '{module_name}.py'."
            raise AdfError(emsg)

        #Run function and return result:
        if func_kwargs:
            return func(self, **func_kwargs)
        else:
            return func(self)
        #End if

    # pylint: enable=no-self-use

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
        else:
            #If not, then just extract the standard CAM climo dictionary
            #and case name::
            cam_climo_dict = self.__cam_climo_info

        #Extract case name:
        case_name = self.read_config_var('cam_case_name',
                                         conf_dict=cam_climo_dict,
                                         required=True)

        #Check if climatologies are being calculated:
        if self.read_config_var('calc_cam_climo', conf_dict=cam_climo_dict):
            # Skip history file stuff if time series are pre-computed:
            if self.read_config_var('cam_ts_done', conf_dict=cam_climo_dict):
                # skip time series generation, and just make the climo
                emsg = "  Configuration file indicates time series files have been pre-computed,"
                emsg += " will rely on those files only."
                print(emsg)
                return

            #Notify user that script has started:
            print("  Generating CAM time series files...")

            #Extract cam time series directory:
            ts_dir = cam_climo_dict['cam_ts_loc']

            #Extract start and end year values:
            # pylint: disable=raise-missing-from
            try:
                start_year = int(cam_climo_dict['start_year'])
            except TypeError:
                if cam_climo_dict['start_year'] is None:
                    start_year = "*"
                else:
                    emsg = "start_year needs to be a year-like value or None, "
                    emsg += f"got '{cam_climo_dict['start_year']}'"
                    raise IOError(emsg)
                #End if
            #End try

            try:
                end_year   = int(cam_climo_dict['end_year'])
            except TypeError:
                if cam_climo_dict['end_year'] is None:
                    end_year = "*"
                else:
                    emsg = "end_year needs to be a year-like value or None, "
                    emsg += f"got '{cam_climo_dict['end_year']}'"
                    raise IOError(emsg)
                #End if
            #End try
            # pylint: enable=raise-missing-from


            #Extract cam variable list:
            var_list = self.__diag_var_list

            #Create path object for the CAM history file(s) location:
            cam_hist_loc = self.read_config_var('cam_hist_loc',
                                                conf_dict=cam_climo_dict,
                                                required=True)
            starting_location = Path(cam_hist_loc)

            #Check that path actually exists:
            if not starting_location.is_dir():
                if baseline:
                    emsg = f"Provided baseline 'cam_hist_loc' directory '{starting_location}' "
                    emsg += "not found.  Script is ending here."
                else:
                    emsg = "Provided 'cam_hist_loc' directory '{starting_location}' not found."
                    emsg += " Script is ending here."
                #End if

                self.end_diag_fail(emsg)
            #End if

            #Check if history files actually exist. If not then kill script:
            if not list(starting_location.glob('*.cam.h0.*.nc')):
                emsg = f"No CAM history (h0) files found in '{starting_location}'."
                emsg += " Script is ending here."
                self.end_diag_fail(emsg)
            #End if

            # NOTE: We need to have the half-empty cases covered, too. (*, end) & (start, *)
            if start_year == end_year == "*":
                files_list = sorted(list(starting_location.glob('*.cam.h0.*.nc')))
            else:
                #Create empty list:
                files_list = list()

                #Loop over start and end years:
                for year in range(start_year, end_year+1):
                    #Add files to main file list:
                    for fname in starting_location.glob(f'*.cam.h0.*{year}-*.nc'):
                        files_list.append(fname)

            #Create ordered list of CAM history files:
            hist_files = sorted(files_list)

            # Check if time series directory exists, and if not, then create it:
            # Use pathlib to create parent directories, if necessary.
            Path(ts_dir).mkdir(parents=True, exist_ok=True)

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
                cmd = ["ncrcat", "-O", "-4", "-h", "-v", f"{var},hyam,hybm,hyai,hybi,PS"] + \
                      hist_files + ["-o", ts_outfil_str]
                subprocess.run(cmd, check=True)

            #Notify user that script has ended:
            print("  ...CAM time series file generation has finished successfully.")

        else:
            #If not, then notify user that time series generation is skipped.
            print("  Climatology files are not being generated, so neither will time series files.")

    #########

    def create_climo(self):

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

        #Check if a user wants any climatologies to be calculated:
        if self.get_cam_info('calc_cam_climo') or \
           self.get_baseline_info('calc_cam_climo'):


            #If so, then extract names of time-averaging scripts:
            avg_func_names = self.__time_averaging_scripts  # this is a list of script names
                                                            # _OR_
                                                            # a **list** of dictionaries with
                                                            # script names as keys that hold
                                                            # args(list), kwargs(dict), and
                                                            # module(str)

            if not avg_func_names:
                emsg = "No time_averaging_scripts provided for calculating"
                emsg += " climatologies, but climatologies were requested.\n"
                emsg += "Please either provide a valid averaging script,"
                emsg += " or skip the calculation of climatologies."
                raise AdfError(emsg)

            #Run the listed scripts:
            self.__diag_scripts_caller("averaging", avg_func_names,
                                       log_section = "create_climo")

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

        #Extract names of re-gridding scripts:
        regrid_func_names = self.__regridding_scripts # this is a list of script names
                                                      # _OR_
                                                      # a **list** of dictionaries with
                                                      # script names as keys that hold
                                                      # kwargs(dict) and module(str)

        if not regrid_func_names or all([func_names is None for func_names in regrid_func_names]):
            print("No regridding options provided, continue.")
            return
            # NOTE: if no regridding options provided, we should skip it, but
            #       do we need to still copy (symlink?) files into the regrid directory?

        #Run the listed scripts:
        self.__diag_scripts_caller("regridding", regrid_func_names,
                                   log_section = "regrid_climo")

    #########

    def perform_analyses(self):

        """
        Performs statistical and other analyses as specified by the
        user.  This currently only includes the AMWG table generation.

        This method also assumes that the analysis scripts require model
        inputs in a time series format.
        """

        #Extract names of plotting scripts:
        anly_func_names = self.__analysis_scripts  # this is a list of script names
                                                   # _OR_
                                                   # a **list** of dictionaries with
                                                   # script names as keys that hold
                                                   # args(list), kwargs(dict), and module(str)

        #If no scripts are listed, then exit routine:
        if not anly_func_names:
            print("Nothing listed under 'analysis_scripts', exiting 'perform_analyses' method.")
            return

        #Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)

        #Set "plot_location" variable, if it doesn't exist already, and save value in diag object.
        #Please note that this is also assumed to be the output location for the analyses scripts.
        if not self.__plot_location:

            #Plot directory:
            plot_dir = self.get_basic_info('cam_diag_plot_loc', required=True)

            #Case name:
            case_name = self.get_cam_info('cam_case_name', required=True)

            #Start year (not currently required):
            syear = self.get_cam_info('start_year')

            #End year (not currently rquired):
            eyear = self.get_cam_info('end_year')

            if syear and eyear:
                self.__plot_location = os.path.join(plot_dir,
                                               f"{case_name}_vs_{data_name}_{syear}_{eyear}")
            else:
                self.__plot_location = os.path.join(plot_dir, f"{case_name}_vs_{data_name}")

        #Run the listed scripts:
        self.__diag_scripts_caller("analysis", anly_func_names,
                                   log_section = "perform_analyses")

    #########

    def create_plots(self):

        """
        Generate ADF diagnostic plots.

        The actual plotting is done using the
        scripts listed under "plotting_scripts"
        as specified in the config file.  This is done
        so that the user can add their own plotting
        script(s) without having to modify the
        main ADF diagnostics routines.
        """

        #Extract names of plotting scripts:
        plot_func_names = self.__plotting_scripts  # this is a list of script names
                                                   # _OR_
                                                   # a **list** of dictionaries with
                                                   # script names as keys that hold
                                                   # args(list), kwargs(dict), and module(str)


        #If no scripts are listed, then exit routine:
        if not plot_func_names:
            print("Nothing listed under 'plotting_scripts', so no plots will be made.")
            return

        #Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.get_baseline_info('cam_case_name', required=True)

        #Set "plot_location" variable, if it doesn't exist already, and save value in diag object:
        if not self.__plot_location:

            #Plot directory:
            plot_dir = self.get_basic_info('cam_diag_plot_loc', required=True)

            #Case name:
            case_name = self.get_cam_info('cam_case_name', required=True)

            #Start year (not currently required):
            syear = self.get_cam_info('start_year')

            #End year (not currently rquired):
            eyear = self.get_cam_info_('end_year')

            if syear and eyear:
                self.__plot_loction = os.path.join(plot_dir,
                                               f"{case_name}_vs_{data_name}_{syear}_{eyear}")
            else:
                self.__plot_loction = os.path.join(plot_dir, f"{case_name}_vs_{data_name}")

        #Run the listed scripts:
        self.__diag_scripts_caller("plotting", plot_func_names,
                                   log_section = "create_plots")

    #########

    def create_website(self):

        """
        Generate webpages to display diagnostic results.
        """

        #import needed standard modules:
        import shutil
        from collections import OrderedDict

        #Import "special" modules:
        try:
            import jinja2
        except ImportError:
            print("Jinja2 module does not exist in python path, but is needed for website.")
            print("Please install module, e.g. 'pip install Jinja2'.")
            sys.exit(1)

        #Notify user that script has started:
        print("  Generating Diagnostics webpages...")

        #Check where the relevant plots are located:
        if self.__plot_location:
            plot_location = self.__plot_location
        else:
            plot_location  = self.read_config_var('cam_diag_plot_loc',
                                                  conf_dict=self.__basic_info,
                                                  required=True)
        #End if

        #Extract needed variables from yaml file:
        case_name = self.read_config_var('cam_case_name',
                                         conf_dict=self.__cam_climo_info,
                                         required=True)

        var_list = self.__diag_var_list

        #Set name of comparison data, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.read_config_var('cam_case_name',
                                             conf_dict=self.__cam_bl_climo_info,
                                             required=True)
        #End if

        #Set preferred order of seasons:
        season_order = ["ANN", "DJF", "MAM", "JJA", "SON"]

        #Set preferred order of plot types:
        plot_type_order = ["LatLon", "Zonal"]

        #Create new path object from user-specified plot directory path:
        plot_path = Path(plot_location)

        #Create the directory where the website will be built:
        website_dir = plot_path / "website"
        website_dir.mkdir(exist_ok=True)

        #Create a directory that will hold just the html files for individual images:
        img_pages_dir = website_dir / "html_img"
        img_pages_dir.mkdir(exist_ok=True)

        #Create a directory that will hold copies of the actual images:
        assets_dir = website_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        #Specify where CSS files will be stored:
        css_files_dir = website_dir / "templates"
        css_files_dir.mkdir(exist_ok=True)

        #Set path to Jinja2 template files:
        jinja_template_dir = Path(_LOCAL_PATH, 'website_templates')

        #Copy CSS files over to output directory:
        for css_file in jinja_template_dir.glob('*.css'):
            shutil.copyfile(css_file, css_files_dir / css_file.name)

        #Copy images into the website image dictionary:
        for img in plot_path.glob("*.png"):
            idest = assets_dir / img.name
            shutil.copyfile(img, idest) # store image in assets


        mean_html_info = OrderedDict()  # this is going to hold the data for building the mean
                                        # plots provisional structure:
                                        # key = variable_name
                                        # values -> dict w/ keys being "TYPE" of plots
                                        # w/ values being dict w/ keys being TEMPORAL sampling,
                                        # values being the URL

        #Create the jinja Environment object:
        jinenv = jinja2.Environment(loader=jinja2.FileSystemLoader(jinja_template_dir))

        #Create alphabetically-sorted variable list:
        var_list_alpha = sorted(var_list)

        #Loop over variables:
        for var in var_list_alpha:
            #Loop over plot type:
            for ptype in plot_type_order:
                #Loop over seasons:
                for season in season_order:
                    #Create the data that will be fed into the template:
                    for img in assets_dir.glob(f"{var}_{season}_{ptype}_*.png"):
                        alt_text  = img.stem #Extract image file name text

                        #Create output file (don't worry about analysis type for now):
                        outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'
                        # Hacky - how to get the relative path in a better way?:
                        img_data = [os.pardir+os.sep+assets_dir.name+os.sep+img.name, alt_text]
                        title = f"Variable: {var}"              #Create title
                        tmpl = jinenv.get_template('template.html')  #Set template
                        rndr = tmpl.render(title=title, value=img_data, case1=case_name,
                                           case2=data_name) #The template rendered

                        #Open HTML file:
                        with open(outputfile,'w') as ofil:
                            ofil.write(rndr)
                        #End with

                        #Initialize Ordered Dictionary for variable:
                        if var not in mean_html_info:
                            mean_html_info[var] = OrderedDict()

                        #Initialize Ordered Dictionary for plot type:
                        if ptype not in mean_html_info[var]:
                            mean_html_info[var][ptype] = OrderedDict()

                        mean_html_info[var][ptype][season] = outputfile.name

        #Construct mean_diag.html
        mean_title = "AMP Diagnostic Plots"
        mean_tmpl = jinenv.get_template('template_mean_diag.html')
        mean_rndr = mean_tmpl.render(title=mean_title,
                        case1=case_name,
                        case2=data_name,
                        mydata=mean_html_info)

        #Write mean diagnostic plots HTML file:
        outputfile = img_pages_dir / "mean_diag.html"
        with open(outputfile,'w') as ofil:
            ofil.write(mean_rndr)
        #End with

        #Search for AMWG Table HTML files:
        table_html_files = list(plot_path.glob("amwg_table_*.html"))

        #Determine if any AMWG tables were generated:
        if table_html_files:

            #Set Table HTML generation logical to "TRUE":
            gen_table_html = True

            #Create a directory that will hold table html files:
            table_pages_dir = website_dir / "html_table"
            table_pages_dir.mkdir(exist_ok=True)

            #Move all table html files to new directory:
            for table_html in table_html_files:
                shutil.move(table_html, table_pages_dir / table_html.name)

            #Construct dictionary needed for HTML page:
            amwg_tables = OrderedDict()

            #Loop over cases:
            for case in [case_name, data_name]:

                #Search for case name in moved HTML files:
                table_htmls = table_pages_dir.glob(f"amwg_table_{case}.html")

                #Check if file exists:
                if table_htmls:

                    #Initialize loop counter:
                    count = 0

                    #Loop over globbed files:
                    for table_html in table_htmls:

                        #Create relative path for HTML file:
                        amwg_tables[case] = table_html.name

                        #Update counter:
                        count += 1

                        #If counter greater than one, then throw an error:
                        if count > 1:
                            emsg = f"More than one AMWG table is associated with case '{case}'."
                            emsg += "\nNot sure what is going on, "
                            emsg += "so website generation will end here."
                            self.end_diag_fail(emsg)


            #Construct mean_table.html
            mean_title = "AMP Diagnostic Tables:"
            mean_tmpl = jinenv.get_template('template_mean_table.html')
            mean_rndr = mean_tmpl.render(title=mean_title,
                            amwg_tables=amwg_tables)

            #Write mean diagnostic tables HTML file:
            outputfile = table_pages_dir / "mean_table.html"
            with open(outputfile,'w') as ofil:
                ofil.write(mean_rndr)
            #End with

        else:
            #No Tables exist, so no link will be added to main page:
            gen_table_html = False

        #Construct index.html
        index_title = "AMP Diagnostics Prototype"
        index_tmpl = jinenv.get_template('template_index.html')
        index_rndr = index_tmpl.render(title=index_title,
                         case1=case_name,
                         case2=data_name,
                         gen_table_html=gen_table_html)

        #Write Mean diagnostics HTML file:
        outputfile = website_dir / "index.html"
        with open(outputfile,'w') as ofil:
            ofil.write(index_rndr)
        #End with

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

###############
