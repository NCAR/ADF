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
import subprocess
import multiprocessing as mp

import importlib
import copy

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

#Finally, import needed ADF module:
from adf_obs import AdfObs

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

class AdfDiag(AdfObs):

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

        #Add CVDP info to object:
        self.__cvdp_info = self.read_config_var('diag_cvdp_info')

        #Expand CVDP climo info variable strings:
        if self.__cvdp_info is not None:
            self.expand_references(self.__cvdp_info)
        #End if

        #Check if inputs are of the correct type.
        #Ideally this sort of checking should be done
        #in its own class that AdfDiag inherits from:
        #-------------------------------------------

        #Use "cam_case_name" as the variable that sets the total number of cases:
        if isinstance(self.get_cam_info("cam_case_name", required=True), list):

            #Extract total number of test cases:
            num_cases = len(self.get_cam_info("cam_case_name"))

        else:
            #Set number of cases to one:
            num_cases = 1
        #End if

        #Loop over all items in config dict:
        for conf_var, conf_val in self.__cam_climo_info.items():
            if isinstance(conf_val, list):
                #If a list, then make sure it is has the correct number of entries:
                if not len(conf_val) == num_cases:
                    emsg = f"diag_cam_climo config variable '{conf_var}' should have"
                    emsg += f" {num_cases} entries, instead it has {len(conf_val)}"
                    self.end_diag_fail(emsg)
            else:
                #If not a list, then convert it to one:
                self.__cam_climo_info[conf_var] = [conf_val]
            #End if
        #End for

        #-------------------------------------------

        #Check if a CAM vs AMWG obs comparison is being performed:
        if self.compare_obs:

            #Finally, set the baseline info to None, to ensure any scripts
            #that check this variable won't crash:
            self.__cam_bl_climo_info = None

        else:
            #If not, then assume a CAM vs CAM run and add CAM baseline climatology info to object:
            self.__cam_bl_climo_info = self.read_config_var('diag_cam_baseline_climo',
                                                            required=True)

            #Expand CAM baseline climo info variable strings:
            self.expand_references(self.__cam_bl_climo_info)
        #End if

        #Add averaging script names:
        self.__time_averaging_scripts = self.read_config_var('time_averaging_scripts')

        #Add regridding script names:
        self.__regridding_scripts = self.read_config_var('regridding_scripts')

        #Add analysis script names:
        self.__analysis_scripts = self.read_config_var('analysis_scripts')

        #Add plotting script names:
        self.__plotting_scripts = self.read_config_var('plotting_scripts')

        #Create plot location variable for potential use by the website generator.
        #Please note that this variable is only set if "create_plots" or "peform_analyses"
        #is called:
        self.__plot_location = [] #Must be a list to manage multiple cases

    #####

    # Create property needed to return "create_html" logical to user:
    @property
    def create_html(self):
        """Return the "create_html" logical to user if requested."""
        return self.get_basic_info('create_html')

    # Create property needed to return "plot_location" variable to user:
    @property
    def plot_location(self):
        """Return a copy of the '__plot_location' string list to user if requested."""
        #Note that a copy is needed in order to avoid having a script mistakenly
        #modify this variable:
        return copy.copy(self.__plot_location)

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

    def get_cvdp_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cvdp_info' as requested by
        the user. If 'diag_cvdp_info' is not found then try grabbing the
        variable from the top level of the YAML config file dictionary
        instead.
        """

        return self.read_config_var(var_str,
                                    conf_dict=self.__cvdp_info,
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
                self.end_diag_fail(emsg)
            #End if

            #If not required, then return none:
            return None
        #End if

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
            self.end_diag_fail(emsg)

        #If kwargs are present, then run function with kwargs and return result:
        if func_kwargs:
            return func(self, **func_kwargs)

        #Otherwise just run function as-is, and return result:
        return func(self)

    #########

    def create_time_series(self, baseline=False):

        """
        Generate time series versions of the CAM history file data.
        """

        global call_ncrcat
        def call_ncrcat(cmd):
            '''this is an internal function to `create_time_series`
            It just wraps the subprocess.call() function, so it can be
            used with the multiprocessing Pool that is constructed below.
            It is declared as global to avoid AttributeError.
            '''
            return subprocess.run(cmd, shell=False)

        #Check if baseline time-series files are being created:
        if baseline:
            #Then use the CAM baseline climo dictionary
            #and case name:
            cam_climo_dict = self.__cam_bl_climo_info
        else:
            #If not, then just extract the standard CAM climo dictionary
            #and case name::
            cam_climo_dict = self.__cam_climo_info

        #Notify user that script has started:
        print("\n  Generating CAM time series files...")

        #Extract case name(s):
        case_names = self.read_config_var('cam_case_name',
                                         conf_dict=cam_climo_dict,
                                         required=True)

        #Check if case_name is actually a list of cases:
        if isinstance(case_names, list):
            #If so, then read in needed variables directly:
            cam_ts_done   = self.read_config_var('cam_ts_done', conf_dict=cam_climo_dict)
            start_years   = self.read_config_var('start_year', conf_dict=cam_climo_dict)
            end_years     = self.read_config_var('end_year', conf_dict=cam_climo_dict)
            cam_hist_locs = self.read_config_var('cam_hist_loc', conf_dict=cam_climo_dict,
                                                  required=True)
            ts_dir        = self.read_config_var('cam_ts_loc', conf_dict=cam_climo_dict,
                                                  required=True)
            overwrite_ts  = self.read_config_var('cam_overwrite_ts', conf_dict=cam_climo_dict)

            #If variables weren't provided in config file, then make them a list
            #containing only None-type entries:
            if not cam_ts_done:
                cam_ts_done = [None]*len(case_names)
            if not overwrite_ts:
                overwrite_ts = [None]*len(case_names)
            if not start_years:
                start_years = [None]*len(case_names)
            if not end_years:
                end_years = [None]*len(case_names)
            #End if

            #Also rename case name list:
            case_name_list = case_names
        else:
            #If not, then read in variables and convert to lists:
            cam_ts_done   = [self.read_config_var('cam_ts_done', conf_dict=cam_climo_dict)]
            start_years   = [self.read_config_var('start_year', conf_dict=cam_climo_dict)]
            end_years     = [self.read_config_var('end_year', conf_dict=cam_climo_dict)]
            cam_hist_locs = [self.read_config_var('cam_hist_loc', conf_dict=cam_climo_dict,
                                                  required=True)]
            ts_dir        = [self.read_config_var('cam_ts_loc', conf_dict=cam_climo_dict,
                                                   required=True)]
            overwrite_ts  = [self.read_config_var('cam_overwrite_ts', conf_dict=cam_climo_dict)]

            #Also convert  case_names to list:
            case_name_list = [case_names]
        #End if

        #Loop over cases:
        for case_idx, case_name in enumerate(case_name_list):

            #Check if particular case should be processed:
            if cam_ts_done[case_idx]:
                emsg = " Configuration file indicates time series files have been pre-computed"
                emsg += f" for case '{case_name}'.  Will rely on those files directly."
                print(emsg)
                continue
            #End if

            print(f"\t Processing time series for case '{case_name}' :")

            #Extract start and end year values:
            try:
                start_year = int(start_years[case_idx])
            except TypeError:
                if start_years[case_idx] is None:
                    start_year = "*"
                else:
                    emsg = "start_year needs to be a year-like value or None, "
                    emsg += f"got '{start_years[case_idx]}'"
                    self.end_diag_fail(emsg)
                #End if
            #End try

            try:
                end_year   = int(end_years[case_idx])
            except TypeError:
                if end_years[case_idx] is None:
                    end_year = "*"
                else:
                    emsg = "end_year needs to be a year-like value or None, "
                    emsg += f"got '{end_years[case_idx]}'"
                    self.end_diag_fail(emsg)
                #End if
            #End try

            #Create path object for the CAM history file(s) location:
            starting_location = Path(cam_hist_locs[case_idx])

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
                files_list = sorted(starting_location.glob('*.cam.h0.*.nc'))
            else:
                #Create empty list:
                files_list = []

                #For now make sure both year values are present:
                if start_year == "*" or end_year == "*":
                    emsg = "Must set both start_year and end_year, "
                    emsg = "or remove them both from the config file."
                    self.end_diag_fail(emsg)
                #End if

                #Loop over start and end years:
                for year in range(start_year, end_year+1):
                    #Add files to main file list:
                    for fname in starting_location.glob(f'*.cam.h0.*{year}-*.nc'):
                        files_list.append(fname)
                    #End for
                #End for
            #End if

            #Create ordered list of CAM history files:
            hist_files = sorted(files_list)

            #Open an xarray dataset from the first model history file:
            hist_file_ds = xr.open_dataset(hist_files[0], decode_cf=False, decode_times=False)

            #Get a list of data variables in the 1st hist file:
            hist_file_var_list = list(hist_file_ds.data_vars)
            #Note: could use `open_mfdataset`, but that can become very slow;
            #      This approach effectively assumes that all files contain the same variables.

            #Check what kind of vertical coordinate (if any) is being used for this model run:
            #------------------------
            if 'lev' in hist_file_ds:
                #Extract vertical level attributes:
                lev_attrs = hist_file_ds['lev'].attrs

                #First check if there is a "vert_coord" attribute:
                if 'vert_coord' in lev_attrs:
                    vert_coord_type = lev_attrs['vert_coord']
                else:
                    #Next check that the "long_name" attribute exists:
                    if 'long_name' in lev_attrs:
                        #Extract long name:
                        lev_long_name = lev_attrs['long_name']

                        #Check for "keywords" in the long name:
                        if 'hybrid level' in lev_long_name:
                            #Set model to hybrid vertical levels:
                            vert_coord_type = "hybrid"
                        elif 'zeta level' in lev_long_name:
                            #Set model to height (z) vertical levels:
                            vert_coord_type = "height"
                        else:
                            #Print a warning, and assume that no vertical
                            #level information is needed.
                            wmsg = "WARNING! Unable to determine the vertical coordinate"
                            wmsg +=f" type from the 'lev' long name, which is:\n'{lev_long_name}'."
                            wmsg += "\nNo additional vertical coordinate information will be"
                            wmsg += " transferred beyond the 'lev' dimension itself."
                            print(wmsg)

                            vert_coord_type = None
                        #End if
                    else:
                        #Print a warning, and assume hybrid levels (for now):
                        wmsg = "WARNING!  No long name found for the 'lev' dimension,"
                        wmsg += " so no additional vertical coordinate information will be"
                        wmsg += " transferred beyond the 'lev' dimension itself."
                        print(wmsg)

                        vert_coord_type = None
                    #End if (long name)
                #End if (vert_coord)
            else:
                #No level dimension found, so assume there is no vertical coordinate:
                vert_coord_type = None
            #End if (lev existence)
            #------------------------

            #Check if time series directory exists, and if not, then create it:
            #Use pathlib to create parent directories, if necessary.
            Path(ts_dir[case_idx]).mkdir(parents=True, exist_ok=True)

            #INPUT NAME TEMPLATE: $CASE.$scomp.[$type.][$string.]$date[$ending]
            first_file_split = str(hist_files[0]).split(".")
            if first_file_split[-1] == "nc":
                time_string_start = first_file_split[-2].replace("-","")
            else:
                time_string_start = first_file_split[-1].replace("-","")
            last_file_split = str(hist_files[-1]).split(".")
            if last_file_split[-1] == "nc":
                time_string_finish = last_file_split[-2].replace("-","")
            else:
                time_string_finish = last_file_split[-1].replace("-","")
            time_string = "-".join([time_string_start, time_string_finish])

            #Loop over CAM history variables:
            list_of_commands = []
            for var in self.diag_var_list:
                if var not in hist_file_var_list:
                    msg = f"WARNING: {var} is not in the file {hist_files[0]}."
                    msg += " No time series will be generated."
                    print(msg)
                    continue

                #Check if variable has a "lev" dimension according to first file:
                if 'lev' in hist_file_ds[var].dims:
                    has_lev = True
                else:
                    has_lev = False
                #End if

                #Create full path name,  file name template:
                #$cam_case_name.h0.$variable.YYYYMM-YYYYMM.nc

                ts_outfil_str = ts_dir[case_idx] + os.sep + \
                ".".join([case_name, "h0", var, time_string, "nc" ])

                #Check if files already exist in time series directory:
                ts_file_list = glob.glob(ts_outfil_str)

                #If files exist, then check if over-writing is allowed:
                if ts_file_list:
                    if not overwrite_ts[case_idx]:
                        #If not, then simply skip this variable:
                        continue

                #Notify user of new time series file:
                print(f"\t - time series for {var}")

                #Determine "ncrcat" command to generate time series file:
                if has_lev and vert_coord_type:
                    if vert_coord_type == "hybrid":
                        cmd = ["ncrcat", "-O", "-4", "-h", "-v",
                               f"{var},hyam,hybm,hyai,hybi,PS"] + \
                               hist_files + ["-o", ts_outfil_str]
                    elif vert_coord_type == "height":
                        #Adding PMID here works, but significantly increases
                        #the storage (disk usage) requirements of the ADF.
                        #This can be alleviated in the future by figuring out
                        #a way to determine all of the regridding targets at
                        #the start of the ADF run, and then regridding a single
                        #PMID file to each one of those targets separately. -JN
                        cmd = ["ncrcat", "-O", "-4", "-h", "-v", f"{var},PMID,PS"] + \
                                hist_files + ["-o", ts_outfil_str]
                    #End if
                else:
                    #No vertical coordinate (or no coordinate meta-data),
                    #so no additional variables needed:
                    cmd = ["ncrcat", "-O", "-4", "-h", "-v", f"{var}"] + \
                           hist_files + ["-o", ts_outfil_str]
                #End if

                #Add to command list for use in multi-processing pool:
                list_of_commands.append(cmd)

            #End variable loop

            #Now run the "ncrcat" subprocesses in parallel:
            with mp.Pool(processes=self.num_procs) as p:
                result = p.map(call_ncrcat, list_of_commands)
            #End with

        #End cases loop

        #Notify user that script has ended:
        print("  ...CAM time series file generation has finished successfully.")

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
                self.end_diag_fail(emsg)

            #Run the listed scripts:
            self.__diag_scripts_caller("averaging", avg_func_names,
                                       log_section = "create_climo")

        else:
            #If not, then notify user that climo file generation is skipped.
            print("\n  No climatology files were requested by user, so averaging will be skipped.")

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

        if not regrid_func_names or all(func_names is None for func_names in regrid_func_names):
            print("\n  No regridding options provided, continue.")
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
            print("\n  Nothing listed under 'analysis_scripts', exiting 'perform_analyses' method.")
            return
        #End if

        #Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            #Set data_name to basline case:
            data_name = self.get_baseline_info('cam_case_name', required=True)

            #Attempt to grab baseline start_years (not currently required):
            syear_baseline = self.get_baseline_info('start_year')
            eyear_baseline = self.get_baseline_info('end_year')

            #If years exist, then add them to the data_name string:
            if syear_baseline and eyear_baseline:
                data_name += f"_{syear_baseline}_{eyear_baseline}"
            #End if
        #End if

        #Set "plot_location" variable, if it doesn't exist already, and save value in diag object.
        #Please note that this is also assumed to be the output location for the analyses scripts:
        if not self.__plot_location:

            #Plot directory:
            plot_dir = self.get_basic_info('cam_diag_plot_loc', required=True)

            #Case names:
            case_names = self.get_cam_info('cam_case_name', required=True)

            #Start years (not currently required):
            syears = self.get_cam_info('start_year')

            #End year (not currently rquired):
            eyears = self.get_cam_info('end_year')

            #Loop over cases:
            for case_idx, case_name in enumerate(case_names):

                #Set case name if start and end year are present:
                if syears[case_idx] and eyears[case_idx]:
                    case_name += f"_{syears[case_idx]}_{eyears[case_idx]}"
                #End if

                #Set the final directory name and save it to plot_location:
                direc_name = f"{case_name}_vs_{data_name}"
                self.__plot_location.append(os.path.join(plot_dir, direc_name))
            #End for
        #End if

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
            print("\n  Nothing listed under 'plotting_scripts', so no plots will be made.")
            return
        #End if

        #Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            #Set data_name to basline case:
            data_name = self.get_baseline_info('cam_case_name', required=True)

            #Attempt to grab baseline start_years (not currently required):
            syear_baseline = self.get_baseline_info('start_year')
            eyear_baseline = self.get_baseline_info('end_year')

            #If years exist, then add them to the data_name string:
            if syear_baseline and eyear_baseline:
                data_name += f"_{syear_baseline}_{eyear_baseline}"
            #End if
        #End if

        #Set "plot_location" variable, if it doesn't exist already, and save value in diag object:
        if not self.__plot_location:

            #Plot directory:
            plot_dir = self.get_basic_info('cam_diag_plot_loc', required=True)

            #Case names:
            case_names = self.get_cam_info('cam_case_name', required=True)

            #Start years (not currently required):
            syears = self.get_cam_info('start_year')

            #End year (not currently rquired):
            eyears = self.get_cam_info('end_year')

            #Loop over cases:
            for case_idx, case_name in enumerate(case_names):

                #Set case name if start and end year are present:
                if syears[case_idx] and eyears[case_idx]:
                    case_name += f"_{syears[case_idx]}_{eyears[case_idx]}"
                #End if

                #Set the final directory name and save it to plot_location:
                direc_name = f"{case_name}_vs_{data_name}"
                self.__plot_location.append(os.path.join(plot_dir, direc_name))
            #End for
        #End if

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
        #End except

        #Notify user that script has started:
        print("\n  Generating Diagnostics webpages...")

        #Check where the relevant plots are located:
        if self.__plot_location:
            plot_location = self.__plot_location
        else:
            plot_location.append(self.get_basic_info('cam_diag_plot_loc', required=True))
        #End if

        #If there is more than one plot location, then create new website directory:
        if len(plot_location) > 1:
            main_site_path = Path(self.get_basic_info('cam_diag_plot_loc', required=True))
            main_site_path = main_site_path / "main_website"
            main_site_path.mkdir(exist_ok=True)
            case_sites = OrderedDict()
        else:
            main_site_path = "" #Set main_site_path to blank value
        #End if

        #Extract needed variables from yaml file:
        case_names = self.read_config_var('cam_case_name',
                                         conf_dict=self.__cam_climo_info,
                                         required=True)

        #Extract variable list:
        var_list = self.diag_var_list

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

        # Variable categories
        var_cat_dict = {
            'Clouds': {'ACTNI', 'ACTNL', 'ACTREI', 'ACTREL', 'ADRAIN', 'ADSNOW',
                       'AREI', 'AREL', 'CCN3', 'CDNUMC', 'CLDHGH', 'CLDICE',
                       'CLDLIQ', 'CLDLOW', 'CLDMED', 'CLDTOT', 'CLOUD', 'CONCLD',
                       'EVAPPREC', 'EVAPSNOW', 'FCTI', 'FCTL', 'FICE', 'FREQI',
                       'FREQL', 'FREQR', 'FREQS', 'MPDQ', 'PRECC', 'PRECL',
                       'PRECSC', 'PRECSL', 'PRECT', 'TGCLDIWP', 'TGCLDLWP'},
            'Deep Convection': {'CAPE', 'CMFMC_DP', 'FREQZM', 'ZMDQ', 'ZMDT'},
            'COSP': {'CLDTOT_ISCCP', 'CLIMODIS', 'CLTMODIS', 'CLWMODIS',
                     'FISCCP1_COSP', 'ICE_ICLD_VISTAU', 'IWPMODIS',
                     'LIQ_ICLD_VISTAU', 'LWPMODIS', 'MEANCLDALB_ISCCP',
                     'MEANPTOP_ISCCP', 'MEANTAU_ISCCP', 'MEANTB_ISCCP',
                     'MEANTBCLR_ISCCP', 'PCTMODIS', 'REFFCLIMODIS', 'REFFCLWMODIS',
                     'SNOW_ICLD_VISTAU', 'TAUTMODIS', 'TAUWMODIS',
                     'TOT_CLD_VISTAU', 'TOT_ICLD_VISTAU'},
            'Budget': {'DCQ', 'DQCORE', 'DTCORE', 'MPDICE', 'MPDLIQ', 'PTEQ'},
            'Radiation': {'FLNS', 'FLNSC', 'FLNT', 'FLNTC', 'FLUT', 'FSDS',
                          'FSDSC', 'FSNS', 'FSNSC', 'FSNT', 'FSNTC', 'FSNTOA',
                          'LHFLX', 'LWCF', 'QRL', 'QRS', 'SHFLX', 'SWCF'},
            'State': {'OMEGA', 'OMEGA500', 'PINT', 'PMID', 'PS', 'PSL', 'Q',
                      'RELHUM', 'T', 'U', 'V', 'Z3', 'Wind'},
            'Surface': {'PBLH', 'QFLX', 'TAUX', 'TAUY', 'TREFHT', 'U10',
                        'Surface_Wind_Stress'},
            'GW': {'QTGW', 'UGTW_TOTAL', 'UTGWORO', 'VGTW_TOTAL', 'VTGWORO'},
            'CLUBB': {'RVMTEND_CLUBB', 'STEND_CLUBB', 'WPRTP_CLUBB', 'WPTHLP_CLUBB'}
        }

        #Set preferred order of plot types:
        plot_type_order = ["LatLon", 
                           "LatLon_Vector", "Zonal", 
                           "NHPolar", "SHPolar",
                           "TaylorDiag"]
        plot_type_web = ["html_img/mean_diag_LatLon.html",
                         "html_img/mean_diag_LatLon_Vector.html","html_img/mean_diag_Zonal.html",
                         "html_img/mean_diag_NHPolar.html","html_img/mean_diag_SHPolar.html",
                         "html_img/mean_diag_TaylorDiag.html",]
        plot_type_html = dict(zip(plot_type_order, plot_type_web))
        main_title = "CAM Diagnostics"

        #Check if any variables are associated with specific vector quantities,
        #and if so then add the vectors to the website variable list.
        for var in var_list:
            if var in self.variable_defaults:
                vect_name = self.variable_defaults[var].get("vector_name", None)
                if vect_name and (vect_name not in var_list):
                    var_list.append(vect_name)
                #End if
            #End if
        #End for

        #Extract pressure levels being plotted:
        pres_levs = self.get_basic_info("plot_press_levels")

        if pres_levs:
            #Create pressure-level variable dictionary:
            pres_levs_var_dict = {}

            #Now add variables on pressure levels, if applicable.
            #Please note that this method is not particularly
            #efficient as most of these variables won't actually exist:
            for var in var_list:
                #Find variable category:
                category = next((cat for cat, varz in var_cat_dict.items() if var in varz), None)

                #Add variable with pressure levels:
                #Please note that this method is not particularly
                #efficient as most of these variables won't actually exist:
                for pres in pres_levs:
                    if category:
                        if category in pres_levs_var_dict:
                            pres_levs_var_dict[category].append(f"{var}_{pres}hpa")
                        else:
                            pres_levs_var_dict[category] = [f"{var}_{pres}hpa"]
                        #End if
                    else:
                        if "none" in pres_levs_var_dict:
                            pres_levs_var_dict["none"].append(f"{var}_{pres}hpa")
                        else:
                            pres_levs_var_dict["none"] = [f"{var}_{pres}hpa"]
                        #End if
                    #End if
                #End for
            #End for

            #Now loop over pressure variable dictionary:
            for category, pres_var_names in pres_levs_var_dict.items():
                #Add pressure-level variable to category if applicable:
                if category in var_cat_dict:
                    var_cat_dict[category].update(pres_var_names)
                #End if

                #Add pressure-level variable to variable list:
                var_list.extend(pres_var_names)

            #End for
        #End if

        # add fake "cam" variable to variable list in order to find Taylor diagram plots:
        var_list.append('cam')

        #Set path to Jinja2 template files:
        jinja_template_dir = Path(_LOCAL_PATH, 'website_templates')

        #Create the jinja Environment object:
        jinenv = jinja2.Environment(loader=jinja2.FileSystemLoader(jinja_template_dir))

        #Create alphabetically-sorted variable list:
        var_list_alpha = sorted(var_list)

        #Loop over model cases:
        for case_idx, case_name in enumerate(case_names):

            #Create new path object from user-specified plot directory path:
            plot_path = Path(plot_location[case_idx])

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

            #Copy CSS files over to output directory:
            for css_file in jinja_template_dir.glob('*.css'):
                shutil.copyfile(css_file, css_files_dir / css_file.name)
            #End for

            #Copy images into the website image dictionary:
            for img in plot_path.glob("*.png"):
                idest = assets_dir / img.name
                shutil.copyfile(img, idest) # store image in assets
            #End for

            #Loop over plot type:
            for ptype in plot_type_order:
                # this is going to hold the data for building the mean
                # plots provisional structure:
                # key = variable_name
                # values -> dict w/ keys being "TYPE" of plots
                # w/ values being dict w/ keys being TEMPORAL sampling,
                # values being the URL
                mean_html_info = OrderedDict()

                for var in var_list_alpha:
                    #Loop over seasons:
                    for season in season_order:

                        #Create the data that will be fed into the template:
                        for img in assets_dir.glob(f"{var}_{season}_{ptype}_Mean*.png"):

                            #Create output file (don't worry about analysis type for now):
                            outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'

                            # Search through all categories and see
                            # which one the current variable is part of
                            category = next((cat for cat, varz \
                                             in var_cat_dict.items() if var in varz), None)
                            if not category:
                                category = 'No category yet'
                            #End if

                            if category not in mean_html_info:
                                mean_html_info[category] = OrderedDict()

                            #Initialize Ordered Dictionary for variable:
                            if var not in mean_html_info[category]:
                                mean_html_info[category][var] = OrderedDict()

                            #Initialize Ordered Dictionary for plot type:
                            if ptype not in mean_html_info[category][var]:
                                mean_html_info[category][var][ptype] = OrderedDict()

                            #Initialize Ordered Dictionary for season:
                            if season not in mean_html_info[category][var][ptype]:
                                mean_html_info[category][var][ptype][season] = OrderedDict()

                            mean_html_info[category][var][ptype][season] = outputfile.name

                #Loop over variables:
                for var in var_list_alpha:
                    #Loop over seasons:
                    for season in season_order:
                        #Create the data that will be fed into the template:
                        for img in assets_dir.glob(f"{var}_{season}_{ptype}_Mean*.png"):
                            alt_text  = img.stem #Extract image file name text

                            #Create output file (don't worry about analysis type for now):
                            outputfile = img_pages_dir / f'plot_page_{var}_{season}_{ptype}.html'
                            # Hacky - how to get the relative path in a better way?:
                            img_data = [os.pardir+os.sep+assets_dir.name+os.sep+img.name, alt_text]

                            #Create titles
                            var_title = f"Variable: {var}"
                            season_title = f"Season: {season}"
                            plottype_title = f"Plot: {ptype}"
                            tmpl = jinenv.get_template('template.html')  #Set template
                            rndr = tmpl.render(title=main_title,
                                               var_title=var_title,
                                               season_title=season_title,
                                               plottype_title=plottype_title,
                                               imgs=img_data,
                                               case1=case_name,
                                               case2=data_name,
                                               mydata=mean_html_info,
                                               plot_types=plot_type_html) #The template rendered

                            #Open HTML file:
                            with open(outputfile, 'w', encoding='utf-8') as ofil:
                                ofil.write(rndr)
                            #End with

                            #Construct individual plot type mean_diag html files
                            mean_tmpl = jinenv.get_template(f'template_mean_diag_{ptype}.html')
                            mean_rndr = mean_tmpl.render(title=main_title,
                                            case1=case_name,
                                            case2=data_name,
                                            mydata=mean_html_info,
                                            plot_types=plot_type_html)

                            #Write mean diagnostic plots HTML file:
                            outputfile = img_pages_dir / f"mean_diag_{ptype}.html"
                            with open(outputfile,'w', encoding='utf-8') as ofil:
                                ofil.write(mean_rndr)
                            #End with
                        #End for (assests loop)
                    #End for (seasons loop)

            #Grab AMWG Table HTML files:
            table_html_files = list(plot_path.glob(f"amwg_table_{case_name}*.html"))

            #Grab the comparison table and move it to website dir
            comp_table_html_file = list(plot_path.glob("*comp.html"))

            #Also grab baseline/obs tables, which are always stored in the first case directory:
            if case_idx == 0:
                data_table_html_files = list(plot_path.glob(f"amwg_table_{data_name}*.html"))
            #End if

            #Determine if any AMWG tables were generated:
            if table_html_files:

                #Set Table HTML generation logical to "TRUE":
                gen_table_html = True

                #Create a directory that will hold table html files:
                table_pages_dir = website_dir / "html_table"
                table_pages_dir.mkdir(exist_ok=True)

                #Move all case table html files to new directory:
                for table_html in table_html_files:
                    shutil.move(table_html, table_pages_dir / table_html.name)
                #End for

                #copy all data table html files as well:
                for data_table_html in data_table_html_files:
                    shutil.copy2(data_table_html, table_pages_dir / data_table_html.name)
                #End for

                #Construct dictionary needed for HTML page:
                amwg_tables = OrderedDict()

                for case in [case_name, data_name]:

                    #Search for case name in moved HTML files:
                    table_htmls = sorted(table_pages_dir.glob(f"amwg_table_{case}.html"))

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
                                emsg += "\nso website generation will end here."
                                self.end_diag_fail(emsg)
                            #End if
                        #End for (table html file loop)
                    #End if (table html file exists check)
                #End for (case vs data)

                #Check if comp table exists
                #(if not, then obs are being compared and comp table is not created)
                if comp_table_html_file:
                    #Move the comparison table html file to new directory
                    for comp_table in comp_table_html_file:
                        shutil.move(comp_table, table_pages_dir / comp_table.name)
                        #Add comparison table to website dictionary
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                        #This will be for single-case for now,
                        #will need to think how to change as multi-case is introduced
                        amwg_tables["Case Comparison"] = comp_table.name
                        # * * * * * * * * * * * * * * * * * * * * * * * * * * * *
                    #End for

                # need this to grab the locations of the amwg tables...
                amwg_table_data = [str(table_pages_dir / table_html.name), ""]

                #Construct mean_table.html
                mean_tmpl = jinenv.get_template('template_mean_table.html')
                mean_rndr = mean_tmpl.render(title=main_title,
                                value=amwg_table_data,
                                case1=case_name,
                                case2=data_name,
                                amwg_tables=amwg_tables,
                                plot_types=plot_type_html,
                                )

                #Write mean diagnostic tables HTML file:
                outputfile = table_pages_dir / "mean_table.html"
                with open(outputfile, 'w', encoding='utf-8') as ofil:
                    ofil.write(mean_rndr)
                #End with
            else:
                #No Tables exist, so no link will be added to main page:
                gen_table_html = False
            #End if

            #Construct index.html
            #index_title = "AMP Diagnostics Prototype"
            index_tmpl = jinenv.get_template('template_index.html')
            index_rndr = index_tmpl.render(title=main_title,
                             case1=case_name,
                             case2=data_name,
                             gen_table_html=gen_table_html,
                             plot_types=plot_type_html,
                             )

            #Write Mean diagnostics HTML file:
            outputfile = website_dir / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(index_rndr)
            #End with

            #If this is a multi-case instance, then copy website to "main" directory:
            if main_site_path:
                shutil.copytree(website_dir, main_site_path / case_name)
                #Also add path to case_sites dictionary:
                case_sites[case_name] = os.path.join(os.curdir, case_name, "index.html")
                #Finally, if first case, then also copy templates directory for CSS files:
                if case_idx == 0:
                    shutil.copytree(css_files_dir, main_site_path / "templates")
                #End if
            #End if
        #End for (model case loop)

        #Create multi-case site, if needed:
        if main_site_path:
            main_title = "ADF Diagnostics"
            main_tmpl = jinenv.get_template('template_multi_case_index.html')
            main_rndr = main_tmpl.render(title=main_title,
                            case_sites=case_sites,
                            )
            #Write multi-case main HTML file:
            outputfile = main_site_path / "index.html"
            with open(outputfile, 'w', encoding='utf-8') as ofil:
                ofil.write(main_rndr)
            #End with
        #End if

        #Notify user that script has finishedd:
        print("  ...Webpages have been generated successfully.")

    #########

    def setup_run_cvdp(self):

        """
        Create CVDP directory tree, generate namelist file and
        edit driver.ncl needed to run CVDP. Submit CVDP diagnostics.

        """

        #import needed standard modules:
        import shutil

        #Case names:
        case_names = self.get_cam_info('cam_case_name', required=True)

        #Start years (not currently required):
        syears = self.get_cam_info('start_year')

        #End year (not currently rquired):
        eyears = self.get_cam_info('end_year')

        #Timeseries locations:
        cam_ts_loc = self.get_cam_info('cam_ts_loc')

        #set CVDP directory, recursively copy cvdp codebase to the CVDP directory
        if len(case_names) > 1:
            cvdp_dir = self.get_cvdp_info('cvdp_loc', required=True)+case_names[0]+'_multi_case'
        else:
            cvdp_dir = self.get_cvdp_info('cvdp_loc', required=True)+case_names[0]
        #end if
        if not os.path.isdir(cvdp_dir):
            shutil.copytree(self.get_cvdp_info('cvdp_codebase_loc', required=True),cvdp_dir)
        #End if

        #check to see if there is a CAM baseline case. If there is, read in relevant information.
        if not self.get_basic_info('compare_obs'):
            case_name_baseline = self.get_baseline_info('cam_case_name')
            syears_baseline = self.get_baseline_info('start_year')
            eyears_baseline = self.get_baseline_info('end_year')
            baseline_ts_loc = self.get_baseline_info('cam_ts_loc')
        #End if

        #Loop over cases to create individual text array to be written to namelist file.
        row_list = []
        for case_idx, case_name in enumerate(case_names):
            row = [case_name,' | ',str(cam_ts_loc[case_idx]),os.sep,' | ',
                   str(syears[case_idx]),' | ',str(eyears[case_idx])]
            row_list.append("".join(row))
        #End for

        #Create new namelist file. If CAM baseline case present add it to list,
        #namelist file must end in a blank line.
        with open(os.path.join(cvdp_dir, "namelist"), 'w', encoding='utf-8') as fnml:
            for rowtext in row_list:
                fnml.write(rowtext)
            #End for
            fnml.write('\n\n')
            if "baseline_ts_loc" in locals():
                rowb = [case_name_baseline,' | ',str(baseline_ts_loc),os.sep,' | ',
                        str(syears_baseline),' | ',str(eyears_baseline)]
                rowtextb = "".join(rowb)
                fnml.write(rowtextb)
                fnml.write('\n\n')
            #End if
        #End with

        #modify driver.ncl to set the proper output directory, webpage title, and location
        #of CVDP NCL scripts, set modular = True (to run multiple CVDP scripts at once),
        #and modify the modular_list to exclude all scripts focused solely on non-atmospheric
        #variables, and set tar_output to True if cvdp_tar: true
        with open(os.path.join(cvdp_dir, "driver.ncl"), 'r', encoding='utf-8') as f_in, \
             open(os.path.join(cvdp_dir, f"driver.{case_names[0]}.ncl"), 'w', \
                               encoding='utf-8') as f_out:
            for line in f_in:
                if '  outdir  ' in line:
                    line = '  outdir = "'+cvdp_dir+'/output/"'
                if '  webpage_title  ' in line:
                    line = '  webpage_title = "ADF/CVDP Comparison"'
                if 'directory path of CVDP NCL scripts' in line:
                    line = '  zp = "'+cvdp_dir+'/ncl_scripts/"'
                if '  modular = ' in line:
                    line = '  modular = "True"'
                if '  modular_list = ' in line:
                    line = '  modular_list = "'
                    line += 'psl.nam_nao,psl.pna_npo,tas.trends_timeseries,snd.trends,'
                    line += 'psl.trends,amo,pdo,sst.indices,pr.trends_timeseries,'
                    line += 'psl.sam_psa,sst.mean_stddev,'
                    line += 'psl.mean_stddev,pr.mean_stddev,sst.trends_timeseries,'
                    line += 'tas.mean_stddev,ipo"'
                if self.get_cvdp_info('cvdp_tar'):
                    if '  tar_output  ' in line:
                        line = '  tar_output = "True"'
                    #End if
                #End if
                f_out.write(line)
            #End for
        #End with

        #Submit the CVDP driver script in background mode, send output to cvdp.out file
        with open(os.path.join(cvdp_dir,'cvdp.out'), 'w', encoding='utf-8') as subout:
            _ = subprocess.Popen([f'cd {cvdp_dir}; ncl -Q '+ \
                                  os.path.join(cvdp_dir,f'driver.{case_names[0]}.ncl')],
                                  shell=True, stdout=subout, close_fds=True)
        #End with

        print('   ')
        print('CVDP is running in background. ADF continuing.')
        print(f'CVDP terminal output is located in {cvdp_dir}/cvdp.out')
        if self.get_cvdp_info('cvdp_tar'):
            print('CVDP graphical and netCDF file output can be found here:' + \
                  f' {cvdp_dir}/output/cvdp.tar')
            print('Open index.html (within cvdp.tar file) in web browser to view CVDP results.')
        else:
            print(f'CVDP graphical and netCDF file output can be found here: {cvdp_dir}/output/')
            print(f'Open {cvdp_dir}/output/index.html file in web browser to view CVDP results.')
        #End if
        print('For CVDP information visit: https://www.cesm.ucar.edu/working_groups/CVC/cvdp/')
        print('   ')

###############
