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
import os
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
print(100*"=")
print(sys.path)
print(100*"=")
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

#####

def construct_index_info(d, fnam, opf):

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
    if vname not in d:
        d[vname] = {}
    if plot_type not in d[vname]:
        d[vname][plot_type] = {}
    d[vname][plot_type][temporal] = opf

#####

def function_caller(func_name: str, func_args: list, func_kwargs: dict = {}, module_name=None):
    """Call a function with given arguments.
    
    func_name : string, name of the function to call
    func_args : list, the arguments to pass to the function
    func_kwargs : [optional] dict, the keyword arguments to pass to the function
    module_name : [optional] string, the name of the module where func_name is defined; if not provided, assume func_name.py
    
    return : the output of func_name(*func_args, **func_kwargs)
    """
    if module_name is None:
        module_name = func_name #+'.py'
    # note: when we use importlib, specify the module name without the ".py" extension.
    module = importlib.import_module(func_name)
    if hasattr(module, func_name) and callable(getattr(module, func_name)):
        func = getattr(module, func_name)
    return func(*func_args, **func_kwargs)



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
        """Return the "compare_obs" logical to user if requested."""
        return self.__basic_info['compare_obs']

    # Create property needed to return "create_html" logical to user:
    @property
    def create_html(self):
        """Return the "create_html" logical to user if requested."""
        return self.__basic_info['create_html']

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
            # Skip history file stuff if time series are pre-computed:
            if ('cam_ts_done' in cam_climo_dict) and (cam_climo_dict['cam_ts_done'] == True):
                # skip time series generation, and just make the climo
                print("  Configuration file indicates time series files have been pre-computed, will rely on those files only.")
                return None

            #Notify user that script has started:
            print("  Generating CAM time series files...")

            #Extract cam time series directory:
            ts_dir = cam_climo_dict['cam_ts_loc']

            #Extract start and end year values:
            try:
                start_year = int(cam_climo_dict['start_year'])
            except TypeError:
                if cam_climo_dict['start_year'] is None:
                    start_year = "*"
            else:
                raise IOError("start_year needs to be a year-like value or None, got {}".format(cam_climo_dict['start_year']))

            try:
                end_year   = int(cam_climo_dict['end_year'])
            except TypeError:
                if cam_climo_dict['end_year'] is None:
                    end_year = "*"
            else:
                raise IOError("end_year needs to be a year-like value or None, got {}".format(cam_climo_dict['end_year']))

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

            # NOTE: We need to have the half-empty cases covered, too. (*, end) & (start, *)
            if start_year == end_year == "*":
                files_list = sorted(list(starting_location.glob('*.cam.h0.*.nc')))
            else:
                #Loop over start and end years:
                for year in range(start_year, end_year+1):
                    #Add files to main file list:
                    for fname in starting_location.glob('*.cam.h0.*{}-*.nc'.format(year)):
                        files_list.append(fname)

            #Create ordered list of CAM history files:
            hist_files = sorted(files_list)

            # Check if time series directory exists, and if not, then create it:
            # Use pathlib to create parent directories.
            # the check isn't necessary b/c we can use exist_ok=True kwarg.
            # if not os.path.isdir(ts_dir):
            #     print("    {} not found, making new directory".format(ts_dir))
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
                cmd = ["ncrcat", "-O", "-4", "-h", "-v", f"{var},hyam,hybm,hyai,hybi,PS"] + hist_files + ["-o", ts_outfil_str]
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
            avg_func_names = self.__time_averaging_scripts  # this is a list of script names 
                                                            # _OR_ 
                                                            # a **list** of dictionaries with script names as keys that hold args(list), kwargs(dict), and module(str)
            
            #Extract necessary variables from CAM configure dictionary:
            input_ts_loc    = cam_climo_dict['cam_ts_loc']
            overwrite_climo = cam_climo_dict['cam_overwrite_climo']
            var_list        = self.__diag_var_list

            #Loop over all averaging script names:
            for avg_func_name in avg_func_names:
                # option to specify the module's name:
                if isinstance(avg_func_name, dict):
                    assert len(avg_func_name) == 1  # must just be {function_name : {args:[...], <kwargs:{...}>, <module:'xxxx'>}}
                    has_opt = True
                    opt = avg_func_name[list(avg_func_name.keys())[0]]  # un-nests the dictionary
                    avg_func_name = list(avg_func_name.keys())[0]  # not ideal, but change to a str representation; iteration will continue ok
                else:
                    has_opt = False

                avg_script =  avg_func_name + '.py'  # default behavior: Add file suffix to script name
                if has_opt:
                    if 'module' in opt:
                        avg_script = opt['module']

                #Create full path to averaging script:
                avg_script_path = os.path.join(os.path.join(_DIAG_SCRIPTS_PATH,"averaging"),avg_script)

                #Check that file exists in "scripts/averaging" directory:
                if not os.path.exists(avg_script_path):
                    msg = "Time averaging file '{}' is missing. Script is ending here.".format(avg_script_path)
                    end_diag_script(msg)
                if avg_script_path not in sys.path:
                    # print(f"[INFO] Inserting to sys.path: {avg_script_path}")
                    sys.path.insert(0, avg_script_path)
                # NOTE: when we move to making this into a proper package, this path-checking stuff should be removed and dealt with on the package-level.
                    
                # Arguments; check if user has specified custom arguments
                avg_func_args = [case_name, input_ts_loc, output_loc, var_list]
                avg_func_kwargs = {"clobber":overwrite_climo}
                if has_opt:
                    if ('args' in opt):
                        # RULES: it has to be a list of strings, and then we will take whatever of those are in locals
                        assert isinstance(opt['args'], list)
                        assert all(isinstance(item, str) for item in opt['args'])
                        avg_func_args = list()  # start over
                        for variableToCheck in opt['args']:
                            if variableToCheck in locals():
                                avg_func_args.append(locals()[variableToCheck])
                            else:
                                print("{} is not available".format(variableToCheck))
                    if 'kwargs' in opt:
                        avg_func_kwargs = opt['kwargs']
                # print(f"DEBUG: \n \t avg_func_name = {avg_func_name}\n \t avg_func_args= {avg_func_args}\n \t avg_kwargs= {avg_func_kwargs}")
                function_caller(avg_func_name, avg_func_args, func_kwargs=avg_func_kwargs, module_name=avg_func_name)
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
        if all([xx is None for xx in regrid_func_names]):
            print("No regridding options provided, continue.")
            return
            # NOTE: if no regridding options provided, we should skip it, but
            #       do we need to still copy (symlink?) files into the regrid directory?
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

            regrid_func_args = [case_name, input_climo_loc, output_loc, var_list, target_list, target_loc, overwrite_regrid]
            function_caller(regrid_func_name, regrid_func_args, func_kwargs={}, module_name=regrid_func_name+'.py')

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
            print("Nothing listed under 'analysis_scripts', exiting 'perform_analyses' method.")
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

            anly_func_args = [case_name, input_ts_loc, output_loc, var_list, write_html]
            function_caller(anly_func_name, anly_func_args, func_kwargs={}, module_name=anly_func_name+'.py')

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

            plot_func_args = [case_name, model_rgrid_loc, data_name, data_loc, var_list, data_list, plot_location]
            # print(f"[INFO ]PLOTTING ARGS: {plot_func_args}")
            function_caller(plot_func_name, plot_func_args, func_kwargs={}, module_name=plot_func_name+'.py')

    #########

    def create_webpage(self):

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

        #Extract needed variables from yaml file:
        plot_location  = self.__basic_info['cam_diag_plot_loc']
        case_name = self.__basic_info['cam_case_name']
        var_list = self.__diag_var_list

        #Set name of comparison data, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            data_name = self.__basic_info['cam_baseline_case_name']

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

        #Specify where the images will be:
        img_source_dir = plot_path / f"{case_name}_vs_{data_name}"

        #Specify where CSS files will be stored:
        css_files_dir = website_dir / "templates"
        css_files_dir.mkdir(exist_ok=True)

        #Set path to Jinja2 template files:
        jinja_template_dir = Path(_LOCAL_PATH, 'website_templates')

        #Copy CSS files over to output directory:
        for css_file in jinja_template_dir.glob('*.css'):
            shutil.copyfile(css_file, css_files_dir / css_file.name)

        #Copy images into the website image dictionary:
        for img in img_source_dir.glob("*.png"):
            idest = assets_dir / img.name
            shutil.copyfile(img, idest) # store image in assets


        mean_html_info = OrderedDict()  # this is going to hold the data for building the mean
                                        # plots provisional structure:
                                        # key = variable_name
                                        # values -> dict w/ keys being "TYPE" of plots
                                        # w/ values being dict w/ keys being TEMPORAL sampling, values being the URL

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
                        img_info  = alt_text.split("_") #Split file name into relevant sub-strings
                        anyl_type = img_info[3] #Extract analysis type

                        #Create output file (don't worry about analysis type for now):
                        outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'
                        img_data = [os.pardir+os.sep+assets_dir.name+os.sep+img.name, alt_text]  # Hacky - how to get the relative path in a better way?
                        title = f"Variable: {var}"              #Create title
                        tmpl = jinenv.get_template('template.html')  #Set template
                        rndr = tmpl.render(title=title, value=img_data, case1=case_name, case2=data_name) #The template rendered

                        #Open HTML file:
                        with open(outputfile,'w') as f: f.write(rndr)

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
        with open(outputfile,'w') as f: f.write(mean_rndr)

        #Search for AMWG Table HTML files:
        table_html_files = plot_path.glob("amwg_table_*.html")

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
                            msg = "More than one AMWG table is associated with case '{}'.".format(case)
                            msg += "\nNot sure what is going on, so website generation will end here."
                            end_diag_script(msg)


            #Construct mean_table.html
            mean_title = "AMP Diagnostic Tables:"
            mean_tmpl = jinenv.get_template('template_mean_table.html')
            mean_rndr = mean_tmpl.render(title=mean_title,
                            amwg_tables=amwg_tables)

            #Write mean diagnostic tables HTML file:
            outputfile = table_pages_dir / "mean_table.html"
            with open(outputfile,'w') as f: f.write(mean_rndr)

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
        with open(outputfile,'w') as f: f.write(index_rndr)

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

###############
