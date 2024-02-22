"""
Location of the "AdfDiag" object, which
is used to store all relevant data and
info needed for generating CAM/ADF
diagnostics, including info
on the averaging, regridding, and
plotting methods themselves.
"""

# ++++++++++++++++++++++++++++++
# Import standard python modules
# ++++++++++++++++++++++++++++++

import sys
import os
import os.path
import glob
import subprocess
import multiprocessing as mp
import copy

import importlib

from pathlib import Path
from typing import Optional

# Check if "PyYAML" is present in python path:
# pylint: disable=unused-import
try:
    import yaml
except ImportError:
    print("PyYAML module does not exist in python path.")
    print("Please install module, e.g. 'pip install pyyaml'.")
    sys.exit(1)

# Check if "xarray" is present in python path:
try:
    import xarray as xr
except ImportError:
    print("Xarray module does not exist in python path.")
    print("Please install module, e.g. 'pip install xarray'.")
    sys.exit(1)

# Check if "numpy" is present in python path:
try:
    import numpy as np
except ImportError:
    print("Numpy module does not exist in python path.")
    print("Please install module, e.g. 'pip install numpy'.")
    sys.exit(1)

# Check if "matplolib" is present in python path:
try:
    import matplotlib as mpl
except ImportError:
    print("Matplotlib module does not exist in python path.")
    print("Please install module, e.g. 'pip install matplotlib'.")
    sys.exit(1)

# Check if "cartopy" is present in python path:
try:
    import cartopy.crs as ccrs
except ImportError:
    print("Cartopy module does not exist in python path.")
    print("Please install module, e.g. 'pip install Cartopy'.")
    sys.exit(1)

# pylint: enable=unused-import

# +++++++++++++++++++++++++++++
# Add ADF diagnostics 'scripts'
# directories to Python path
# +++++++++++++++++++++++++++++

# Determine local directory path:
_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

# Add "scripts" directory to path:
_DIAG_SCRIPTS_PATH = os.path.join(_LOCAL_PATH, os.pardir, "scripts")

# Check that "scripts" directory actually exists:
if not os.path.isdir(_DIAG_SCRIPTS_PATH):
    # If not, then raise error:
    ermsg = f"'{_DIAG_SCRIPTS_PATH}' directory not found. Has 'AdfDiag.py' been moved?"
    raise FileNotFoundError(ermsg)

# Walk through all sub-directories in "scripts" directory:
for root, dirs, files in os.walk(_DIAG_SCRIPTS_PATH):
    # Add all sub-directories to python path:
    for dirname in dirs:
        sys.path.append(os.path.join(root, dirname))

# +++++++++++++++++++++++++++++

# Finally, import needed ADF module:
from adf_web import AdfWeb


#################
# Helper functions
#################


def construct_index_info(page_dict, fnam, opf):
    """
    Helper function for generating web pages.
    d : dictionary for the index page information
    fnam : the image filename, img.stem  --> then decompose the img file's parts.
    opf: outputfile for the image
    """
    vname, plot_desc = fnam[0 : fnam.index("_")], fnam[fnam.index("_") + 1 :]
    if "ANN" in plot_desc:
        temporal = "ANN"
    elif "DJF" in plot_desc:
        temporal = "DJF"
    elif "JJA" in plot_desc:
        temporal = "JJA"
    elif "MAM" in plot_desc:
        temporal = "MAM"
    elif "SON" in plot_desc:
        temporal = "SON"
    else:
        temporal = "NoInfo"
    plot_type = plot_desc.replace(temporal + "_", "")
    if vname not in page_dict:
        page_dict[vname] = {}
    if plot_type not in page_dict[vname]:
        page_dict[vname][plot_type] = {}
    page_dict[vname][plot_type][temporal] = opf


######################################
# Main ADF diagnostics class (AdfDiag)
######################################


class AdfDiag(AdfWeb):

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

        # Initialize Config/Base attributes:
        super().__init__(config_file, debug=debug)

        # Add CVDP info to object:
        self.__cvdp_info = self.read_config_var("diag_cvdp_info")

        # Expand CVDP climo info variable strings:
        if self.__cvdp_info is not None:
            self.expand_references(self.__cvdp_info)
        # End if

        # Add averaging script names:
        self.__time_averaging_scripts = self.read_config_var("time_averaging_scripts")

        # Add regridding script names:
        self.__regridding_scripts = self.read_config_var("regridding_scripts")

        # Add analysis script names:
        self.__analysis_scripts = self.read_config_var("analysis_scripts")

        # Add plotting script names:
        self.__plotting_scripts = self.read_config_var("plotting_scripts")

    # Create property needed to return "plotting_scripts" variable to user:
    @property
    def plotting_scripts(self):
        """Return a copy of the '__plotting_scripts' string list to user if requested."""
        # Note that a copy is needed in order to avoid having a script mistakenly
        # modify this variable:
        return copy.copy(self.__plotting_scripts)

    #########
    # Variable extraction functions
    #########

    def get_cvdp_info(self, var_str, required=False):
        """
        Return the config variable from 'diag_cvdp_info' as requested by
        the user. If 'diag_cvdp_info' is not found then try grabbing the
        variable from the top level of the YAML config file dictionary
        instead.
        """

        return self.read_config_var(
            var_str, conf_dict=self.__cvdp_info, required=required
        )

    #########
    # Script-running functions
    #########

    def __diag_scripts_caller(
        self,
        scripts_dir: str,
        func_names: list,
        default_kwargs: Optional[dict] = None,
        log_section: Optional[str] = None,
    ):
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

        # Loop over all averaging script names:
        for func_name in func_names:
            # Check if func_name is a dictonary,
            # this implies that the function has user-defined inputs:
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
                raise TypeError(
                    "Provided script must either be a string or a dictionary."
                )

            func_script = (
                func_name + ".py"
            )  # default behavior: Add file suffix to script name
            if has_opt:
                if "module" in opt:
                    func_script = opt["module"]

            # Create full path to function script:
            func_script_path = os.path.join(
                os.path.join(_DIAG_SCRIPTS_PATH, scripts_dir), func_script
            )

            # Check that file exists in specified directory:
            if not os.path.exists(func_script_path):
                emsg = f"Script file '{func_script_path}' is missing. Diagnostics are ending here."
                self.end_diag_fail(emsg)

            if func_script_path not in sys.path:
                # Add script path to debug log if requested:
                if log_section:
                    dmsg = f"{log_section}: Inserting to sys.path: {func_script_path}"
                    self.debug_log(dmsg)
                else:
                    dmsg = f"diag_scripts_caller: Inserting to sys.path: {func_script_path}"
                    self.debug_log(dmsg)

                # Add script to python path:
                sys.path.insert(0, func_script_path)

            # NOTE: when we move to making this into a proper package,
            #       this path-checking stuff should be removed and dealt with on the package-level.

            # Arguments; check if user has specified custom arguments
            func_kwargs = default_kwargs
            if has_opt:
                if "kwargs" in opt:
                    func_kwargs = opt["kwargs"]

            # Add function calls debug log if requested:
            if log_section:
                dmsg = f"{log_section}: \n \t func_name = {func_name}\n "
                dmsg += f"\t func_kwargs = {func_kwargs}"
                self.debug_log(dmsg)
            else:
                dmsg = f"diag_scripts_caller: \n \t func_name = {func_name}\n "
                dmsg += f"\t func_kwargs = {func_kwargs}"
                self.debug_log(dmsg)

            # Call function
            self.__function_caller(
                func_name, func_kwargs=func_kwargs, module_name=func_name
            )

    #########

    def __function_caller(
        self, func_name: str, func_kwargs: Optional[dict] = None, module_name=None
    ):
        """
        Call a function with given arguments.

        func_name : string, name of the function to call
        func_kwargs : [optional] dict, the keyword arguments to pass to the function
        module_name : [optional] string, the name of the module where func_name is defined;
                      if not provided, assume func_name.py

        return : the output of func_name(self, **func_kwargs)
        """

        if module_name is None:
            module_name = func_name  # +'.py'

        # note: when we use importlib, specify the module name without the ".py" extension.
        module = importlib.import_module(module_name)
        if hasattr(module, func_name) and callable(getattr(module, func_name)):
            func = getattr(module, func_name)
        else:
            emsg = (
                f"Function '{func_name}' cannot be found in module '{module_name}.py'."
            )
            self.end_diag_fail(emsg)

        # If kwargs are present, then run function with kwargs and return result:
        if func_kwargs:
            return func(self, **func_kwargs)

        # Otherwise just run function as-is, and return result:
        return func(self)

    #########

    def create_time_series(self, baseline=False):
        """
        Generate time series versions of the CAM history file data.
        """

        global call_ncrcat

        def call_ncrcat(cmd):
            """this is an internal function to `create_time_series`
            It just wraps the subprocess.call() function, so it can be
            used with the multiprocessing Pool that is constructed below.
            It is declared as global to avoid AttributeError.
            """
            return subprocess.run(cmd, shell=False)

        # End def

        # Notify user that script has started:
        print("\n  Generating CAM time series files...")

        # Check if baseline time-series files are being created:
        if baseline:
            # Use baseline settings, while converting them all
            # to lists:
            case_names = [self.get_baseline_info("cam_case_name", required=True)]
            cam_ts_done = [self.get_baseline_info("cam_ts_done")]
            cam_hist_locs = [self.get_baseline_info("cam_hist_loc")]
            ts_dir = [self.get_baseline_info("cam_ts_loc", required=True)]
            overwrite_ts = [self.get_baseline_info("cam_overwrite_ts")]
            start_years = [self.climo_yrs["syear_baseline"]]
            end_years = [self.climo_yrs["eyear_baseline"]]
        else:
            # Use test case settings, which are already lists:
            case_names = self.get_cam_info("cam_case_name", required=True)
            cam_ts_done = self.get_cam_info("cam_ts_done")
            cam_hist_locs = self.get_cam_info("cam_hist_loc")
            ts_dir = self.get_cam_info("cam_ts_loc", required=True)
            overwrite_ts = self.get_cam_info("cam_overwrite_ts")
            start_years = self.climo_yrs["syears"]
            end_years = self.climo_yrs["eyears"]
        # End if

        # Read hist_str (component.hist_num) from the yaml file, or set to default
        hist_str = self.get_basic_info("hist_str")
        # If hist_str is not present, then default to 'cam.h0':
        if not hist_str:
            hist_str = "cam.h0"
        # End if

        # get info about variable defaults
        res = self.variable_defaults

        # Loop over cases:
        for case_idx, case_name in enumerate(case_names):
            # Check if particular case should be processed:
            if cam_ts_done[case_idx]:
                emsg = " Configuration file indicates time series files have been pre-computed"
                emsg += f" for case '{case_name}'.  Will rely on those files directly."
                print(emsg)
                continue
            # End if

            print(f"\t Processing time series for case '{case_name}' :")

            # Extract start and end year values:
            start_year = start_years[case_idx]
            end_year = end_years[case_idx]

            # Create path object for the CAM history file(s) location:
            starting_location = Path(cam_hist_locs[case_idx])

            # Check that path actually exists:
            if not starting_location.is_dir():
                if baseline:
                    emsg = f"Provided baseline 'cam_hist_loc' directory '{starting_location}' "
                    emsg += "not found.  Script is ending here."
                else:
                    emsg = "Provided 'cam_hist_loc' directory '{starting_location}' not found."
                    emsg += " Script is ending here."
                # End if

                self.end_diag_fail(emsg)
            # End if

            # Check if history files actually exist. If not then kill script:
            if not list(starting_location.glob("*" + hist_str + ".*.nc")):
                emsg = (
                    f"No history *{hist_str}.*.nc files found in '{starting_location}'."
                )
                emsg += " Script is ending here."
                self.end_diag_fail(emsg)
            # End if

            # Create empty list:
            files_list = []

            # Loop over start and end years:
            for year in range(start_year, end_year + 1):
                # Add files to main file list:
                for fname in starting_location.glob(
                    f"*{hist_str}.*{str(year).zfill(4)}*.nc"
                ):
                    files_list.append(fname)
                # End for
            # End for

            # Create ordered list of CAM history files:
            hist_files = sorted(files_list)

            # Open an xarray dataset from the first model history file:
            hist_file_ds = xr.open_dataset(
                hist_files[0], decode_cf=False, decode_times=False
            )

            # Get a list of data variables in the 1st hist file:
            hist_file_var_list = list(hist_file_ds.data_vars)
            # Note: could use `open_mfdataset`, but that can become very slow;
            #      This approach effectively assumes that all files contain the same variables.


            # Check what kind of vertical coordinate (if any) is being used for this model run:
            # ------------------------
            if "lev" in hist_file_ds:
                # Extract vertical level attributes:
                lev_attrs = hist_file_ds["lev"].attrs

                # First check if there is a "vert_coord" attribute:
                if "vert_coord" in lev_attrs:
                    vert_coord_type = lev_attrs["vert_coord"]
                else:
                    # Next check that the "long_name" attribute exists:
                    if "long_name" in lev_attrs:
                        # Extract long name:
                        lev_long_name = lev_attrs["long_name"]

                        # Check for "keywords" in the long name:
                        if "hybrid level" in lev_long_name:
                            # Set model to hybrid vertical levels:
                            vert_coord_type = "hybrid"
                        elif "zeta level" in lev_long_name:
                            # Set model to height (z) vertical levels:
                            vert_coord_type = "height"
                        else:
                            # Print a warning, and assume that no vertical
                            # level information is needed.
                            wmsg = (
                                "WARNING! Unable to determine the vertical coordinate"
                            )
                            wmsg += f" type from the 'lev' long name, which is:\n'{lev_long_name}'."
                            wmsg += "\nNo additional vertical coordinate information will be"
                            wmsg += " transferred beyond the 'lev' dimension itself."
                            print(wmsg)

                            vert_coord_type = None
                        # End if
                    else:
                        # Print a warning, and assume hybrid levels (for now):
                        wmsg = "WARNING!  No long name found for the 'lev' dimension,"
                        wmsg += (
                            " so no additional vertical coordinate information will be"
                        )
                        wmsg += " transferred beyond the 'lev' dimension itself."
                        print(wmsg)

                        vert_coord_type = None
                    # End if (long name)
                # End if (vert_coord)
            else:
                # No level dimension found, so assume there is no vertical coordinate:
                vert_coord_type = None
            # End if (lev existence)
            # ------------------------

            # Check if time series directory exists, and if not, then create it:
            # Use pathlib to create parent directories, if necessary.
            Path(ts_dir[case_idx]).mkdir(parents=True, exist_ok=True)

            # INPUT NAME TEMPLATE: $CASE.$scomp.[$type.][$string.]$date[$ending]
            first_file_split = str(hist_files[0]).split(".")
            if first_file_split[-1] == "nc":
                time_string_start = first_file_split[-2].replace("-", "")
            else:
                time_string_start = first_file_split[-1].replace("-", "")
            last_file_split = str(hist_files[-1]).split(".")
            if last_file_split[-1] == "nc":
                time_string_finish = last_file_split[-2].replace("-", "")
            else:
                time_string_finish = last_file_split[-1].replace("-", "")
            time_string = "-".join([time_string_start, time_string_finish])

            # Loop over CAM history variables:
            list_of_commands = []
            vars_to_derive = []
            # create copy of var list that can be modified for derivable variables
            diag_var_list = self.diag_var_list
            for var in diag_var_list:
                if var not in hist_file_var_list:
                    vres = res.get(var, {})
                    if "derivable_from" in vres:
                        constit_list = vres["derivable_from"]
                        for constit in constit_list:
                            if constit not in diag_var_list:
                                diag_var_list.append(constit)
                        vars_to_derive.append(var)
                        continue
                    else:
                        msg = f"WARNING: {var} is not in the file {hist_files[0]}."
                        msg += " No time series will be generated."
                        print(msg)
                        continue

                # Check if variable has a "lev" dimension according to first file:
                has_lev = bool("lev" in hist_file_ds[var].dims)

                # Create full path name, file name template:
                # $cam_case_name.$hist_str.$variable.YYYYMM-YYYYMM.nc

                ts_outfil_str = (
                    ts_dir[case_idx]
                    + os.sep
                    + ".".join([case_name, hist_str, var, time_string, "nc"])
                )

                # Check if files already exist in time series directory:
                ts_file_list = glob.glob(ts_outfil_str)

                # If files exist, then check if over-writing is allowed:
                if ts_file_list:
                    if not overwrite_ts[case_idx]:
                        # If not, then simply skip this variable:
                        continue

                # Notify user of new time series file:
                print(f"\t - time series for {var}")

                # Variable list starts with just the variable
                ncrcat_var_list = f"{var}"

                # Determine "ncrcat" command to generate time series file:
                if "date" in hist_file_ds[var].dims:
                    ncrcat_var_list = ncrcat_var_list + ",date"
                if "datesec" in hist_file_ds[var].dims:
                    ncrcat_var_list = ncrcat_var_list + ",datesec"

                if has_lev and vert_coord_type:
                    # For now, only add these variables if using CAM:
                    if "cam" in hist_str:
                        # PS might be in a different history file. If so, continue without error.
                        ncrcat_var_list = ncrcat_var_list + ",hyam,hybm,hyai,hybi"

                        if "PS" in hist_file_var_list:
                            ncrcat_var_list = ncrcat_var_list + ",PS"
                            print("Adding PS to file")
                        else:
                            wmsg = "WARNING: PS not found in history file."
                            wmsg += " It might be needed at some point."
                            print(wmsg)
                        # End if

                        if vert_coord_type == "height":
                            # Adding PMID here works, but significantly increases
                            # the storage (disk usage) requirements of the ADF.
                            # This can be alleviated in the future by figuring out
                            # a way to determine all of the regridding targets at
                            # the start of the ADF run, and then regridding a single
                            # PMID file to each one of those targets separately. -JN
                            if "PMID" in hist_file_var_list:
                                ncrcat_var_list = ncrcat_var_list + ",PMID"
                                print("Adding PMID to file")
                            else:
                                wmsg = "WARNING: PMID not found in history file."
                                wmsg += " It might be needed at some point."
                                print(wmsg)
                            # End if PMID
                        # End if height
                    # End if cam
                # End if has_lev
                #print(ncrcat_var_list)
                #print(hist_files)

                cmd = (
                    ["ncrcat", "-O", "-4", "-h", "--no_cll_mth", "-v", ncrcat_var_list]
                    + hist_files
                    + ["-o", ts_outfil_str]
                )

                # Add to command list for use in multi-processing pool:
                list_of_commands.append(cmd)
            print(list_of_commands)
            # End variable loop
            print("here?")
            print(ts_outfil_str)
            """
            # Now run the "ncrcat" subprocesses in parallel:
            with mp.Pool(processes=self.num_procs) as mpool:
                _ = mpool.map(call_ncrcat, list_of_commands)
            """
            #call_ncrcat(list_of_commands)
            list_of_commands =["ncrcat", "-O", "-4", "-h", "--no_cll_mth", "-v", "SNOWDP", "/glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/ctsm51d159_f45_GSWP3_bgccrop_1850pAD/lnd/hist/ctsm51d159_f45_GSWP3_bgccrop_1850pAD.clm2.h0.0001-01.nc", "/glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/ctsm51d159_f45_GSWP3_bgccrop_1850pAD/lnd/hist/ctsm51d159_f45_GSWP3_bgccrop_1850pAD.clm2.h0.0001-02.nc", "-o", ts_outfil_str]
            
            #list_of_commands =["ncrcat", "-O", "-4", "-h", "--no_cll_mth", "-v", "SNOWDP",
            #"/glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/ctsm51d159_f45_GSWP3_bgccrop_1850pSASU/lnd/hist/ctsm51d159_f45_GSWP3_bgccrop_1850pSASU.clm2.h0.0001-01.nc /glade/campaign/cesm/development/cross-wg/diagnostic_framework/CESM_output_for_testing/ctsm51d159_f45_GSWP3_bgccrop_1850pSASU/lnd/hist/ctsm51d159_f45_GSWP3_bgccrop_1850pSASU.clm2.h0.0001-02.nc"]# "-o", "go_to_hell.nc"]
            #print(list_of_commands)
            #hist_files2 = []
            #for i in hist_files:
            #    hist_files2.append(str(i))
            #list_of_commands.extend(hist_files2)
            #print(list_of_commands)
            #list_of_commands.append("-o")
            #list_of_commands.append(ts_outfil_str)
            subprocess.run(list_of_commands, shell=False)

            if vars_to_derive:
                self.derive_variables(
                    vars_to_derive=vars_to_derive, ts_dir=ts_dir[case_idx]
                )
            # End with

        # End cases loop

        # Notify user that script has ended:
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

        # Extract climatology calculation config options:
        calc_climo = self.get_cam_info("calc_cam_climo")

        # Check if climo calculation config option is a list:
        if isinstance(calc_climo, list):
            # If so, then check if any of the entries are "True":
            calc_climo = any(calc_climo)
        # End if

        # Next check if a baseline simulation is being used
        # and no other model cases need climatologies calculated:
        if not self.compare_obs and not calc_climo:
            calc_bl_climo = self.get_baseline_info("calc_cam_climo")

            # Check if baseline climo calculation config option is a list,
            # although it really never should be:
            if isinstance(calc_bl_climo, list):
                # If so, then check if any of the entries are "True":
                calc_bl_climo = any(calc_bl_climo)
            # End if
        else:
            # Just set to False:
            calc_bl_climo = False
        # End if

        # Check if a user wants any climatologies to be calculated:
        if calc_climo or calc_bl_climo:
            # If so, then extract names of time-averaging scripts:
            avg_func_names = (
                self.__time_averaging_scripts
            )  # this is a list of script names
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

            # Run the listed scripts:
            self.__diag_scripts_caller(
                "averaging", avg_func_names, log_section="create_climo"
            )

        else:
            # If not, then notify user that climo file generation is skipped.
            print(
                "\n  No climatology files were requested by user, so averaging will be skipped."
            )

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

        # Extract names of re-gridding scripts:
        regrid_func_names = self.__regridding_scripts  # this is a list of script names
        # _OR_
        # a **list** of dictionaries with
        # script names as keys that hold
        # kwargs(dict) and module(str)

        if not regrid_func_names or all(
            func_names is None for func_names in regrid_func_names
        ):
            print("\n  No regridding options provided, continue.")
            return
            # NOTE: if no regridding options provided, we should skip it, but
            #       do we need to still copy (symlink?) files into the regrid directory?

        # Run the listed scripts:
        self.__diag_scripts_caller(
            "regridding", regrid_func_names, log_section="regrid_climo"
        )

    #########

    def perform_analyses(self):
        """
        Performs statistical and other analyses as specified by the
        user.  This currently only includes the AMWG table generation.

        This method also assumes that the analysis scripts require model
        inputs in a time series format.
        """

        # Extract names of plotting scripts:
        anly_func_names = self.__analysis_scripts  # this is a list of script names
        # _OR_
        # a **list** of dictionaries with
        # script names as keys that hold
        # args(list), kwargs(dict), and module(str)

        # If no scripts are listed, then exit routine:
        if not anly_func_names:
            print(
                "\n  Nothing listed under 'analysis_scripts', exiting 'perform_analyses' method."
            )
            return
        # End if

        # Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            # Set data_name to basline case:
            data_name = self.get_baseline_info("cam_case_name", required=True)

            # Attempt to grab baseline start_years (not currently required):
            syear_baseline = self.climo_yrs["syear_baseline"]
            eyear_baseline = self.climo_yrs["eyear_baseline"]

            # If years exist, then add them to the data_name string:
            if syear_baseline and eyear_baseline:
                data_name += f"_{syear_baseline}_{eyear_baseline}"
            # End if
        # End if

        # Run the listed scripts:
        self.__diag_scripts_caller(
            "analysis", anly_func_names, log_section="perform_analyses"
        )

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

        # Extract names of plotting scripts:
        plot_func_names = self.__plotting_scripts  # this is a list of script names
        # _OR_
        # a **list** of dictionaries with
        # script names as keys that hold
        # args(list), kwargs(dict), and module(str)

        # If no scripts are listed, then exit routine:
        if not plot_func_names:
            print(
                "\n  Nothing listed under 'plotting_scripts', so no plots will be made."
            )
            return
        # End if

        # Set "data_name" variable, which depends on "compare_obs":
        if self.compare_obs:
            data_name = "obs"
        else:
            # Set data_name to basline case:
            data_name = self.get_baseline_info("cam_case_name", required=True)

            # Attempt to grab baseline start_years (not currently required):
            syear_baseline = self.climo_yrs["syear_baseline"]
            eyear_baseline = self.climo_yrs["eyear_baseline"]

            # If years exist, then add them to the data_name string:
            if syear_baseline and eyear_baseline:
                data_name += f"_{syear_baseline}_{eyear_baseline}"
            # End if
        # End if

        # Run the listed scripts:
        self.__diag_scripts_caller(
            "plotting", plot_func_names, log_section="create_plots"
        )

    #########

    def setup_run_cvdp(self):
        """
        Create CVDP directory tree, generate namelist file and
        edit driver.ncl needed to run CVDP. Submit CVDP diagnostics.

        """

        # import needed standard modules:
        import shutil

        # Case names:
        case_names = self.get_cam_info("cam_case_name", required=True)

        # Start years (not currently required):
        syears = self.climo_yrs["syears"]

        # End year (not currently rquired):
        eyears = self.climo_yrs["eyears"]

        # Timeseries locations:
        cam_ts_loc = self.get_cam_info("cam_ts_loc")

        # set CVDP directory, recursively copy cvdp codebase to the CVDP directory
        if len(case_names) > 1:
            cvdp_dir = (
                self.get_cvdp_info("cvdp_loc", required=True)
                + case_names[0]
                + "_multi_case"
            )
        else:
            cvdp_dir = self.get_cvdp_info("cvdp_loc", required=True) + case_names[0]
        # end if
        if not os.path.isdir(cvdp_dir):
            shutil.copytree(
                self.get_cvdp_info("cvdp_codebase_loc", required=True), cvdp_dir
            )
        # End if

        # check to see if there is a CAM baseline case. If there is, read in relevant information.
        if not self.get_basic_info("compare_obs"):
            case_name_baseline = self.get_baseline_info("cam_case_name")
            syears_baseline = self.climo_yrs["syear_baseline"]
            eyears_baseline = self.climo_yrs["eyear_baseline"]
            baseline_ts_loc = self.get_baseline_info("cam_ts_loc")
        # End if

        # Loop over cases to create individual text array to be written to namelist file.
        row_list = []
        for case_idx, case_name in enumerate(case_names):
            row = [
                case_name,
                " | ",
                str(cam_ts_loc[case_idx]),
                os.sep,
                " | ",
                str(syears[case_idx]),
                " | ",
                str(eyears[case_idx]),
            ]
            row_list.append("".join(row))
        # End for

        # Create new namelist file. If CAM baseline case present add it to list,
        # namelist file must end in a blank line.
        with open(os.path.join(cvdp_dir, "namelist"), "w", encoding="utf-8") as fnml:
            for rowtext in row_list:
                fnml.write(rowtext)
            # End for
            fnml.write("\n\n")
            if "baseline_ts_loc" in locals():
                rowb = [
                    case_name_baseline,
                    " | ",
                    str(baseline_ts_loc),
                    os.sep,
                    " | ",
                    str(syears_baseline),
                    " | ",
                    str(eyears_baseline),
                ]
                rowtextb = "".join(rowb)
                fnml.write(rowtextb)
                fnml.write("\n\n")
            # End if
        # End with

        # modify driver.ncl to set the proper output directory, webpage title, and location
        # of CVDP NCL scripts, set modular = True (to run multiple CVDP scripts at once),
        # and modify the modular_list to exclude all scripts focused solely on non-atmospheric
        # variables, and set tar_output to True if cvdp_tar: true
        with open(
            os.path.join(cvdp_dir, "driver.ncl"), "r", encoding="utf-8"
        ) as f_in, open(
            os.path.join(cvdp_dir, f"driver.{case_names[0]}.ncl"), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                if "  outdir  " in line:
                    line = '  outdir = "' + cvdp_dir + '/output/"'
                if "  webpage_title  " in line:
                    line = '  webpage_title = "ADF/CVDP Comparison"'
                if "directory path of CVDP NCL scripts" in line:
                    line = '  zp = "' + cvdp_dir + '/ncl_scripts/"'
                if "  modular = " in line:
                    line = '  modular = "True"'
                if "  modular_list = " in line:
                    line = '  modular_list = "'
                    line += "psl.nam_nao,psl.pna_npo,tas.trends_timeseries,snd.trends,"
                    line += "psl.trends,amo,pdo,sst.indices,pr.trends_timeseries,"
                    line += "psl.sam_psa,sst.mean_stddev,"
                    line += "psl.mean_stddev,pr.mean_stddev,sst.trends_timeseries,"
                    line += 'tas.mean_stddev,ipo"'
                if self.get_cvdp_info("cvdp_tar"):
                    if "  tar_output  " in line:
                        line = '  tar_output = "True"'
                    # End if
                # End if
                f_out.write(line)
            # End for
        # End with

        # Submit the CVDP driver script in background mode, send output to cvdp.out file
        with open(os.path.join(cvdp_dir, "cvdp.out"), "w", encoding="utf-8") as subout:
            _ = subprocess.Popen(
                [
                    f"cd {cvdp_dir}; ncl -Q "
                    + os.path.join(cvdp_dir, f"driver.{case_names[0]}.ncl")
                ],
                shell=True,
                stdout=subout,
                close_fds=True,
            )
        # End with

        print("   ")
        print("CVDP is running in background. ADF continuing.")
        print(f"CVDP terminal output is located in {cvdp_dir}/cvdp.out")
        if self.get_cvdp_info("cvdp_tar"):
            print(
                "CVDP graphical and netCDF file output can be found here:"
                + f" {cvdp_dir}/output/cvdp.tar"
            )
            print(
                "Open index.html (within cvdp.tar file) in web browser to view CVDP results."
            )
        else:
            print(
                f"CVDP graphical and netCDF file output can be found here: {cvdp_dir}/output/"
            )
            print(
                f"Open {cvdp_dir}/output/index.html file in web browser to view CVDP results."
            )
        # End if
        print(
            "For CVDP information visit: https://www.cesm.ucar.edu/working_groups/CVC/cvdp/"
        )
        print("   ")

    #########

    def derive_variables(self, vars_to_derive=None, ts_dir=None, overwrite=None):
        """
        Derive variables acccording to steps given here.  Since derivations will depend on the
        variable, each variable to derive will need its own set of steps below.

        Caution: this method assumes that there will be one time series file per variable

        If the file for the derived variable exists, the kwarg `overwrite` determines
        whether to overwrite the file (true) or exit with a warning message.
        """

        for var in vars_to_derive:
            if var == "PRECT":
                # PRECT can be found by simply adding PRECL and PRECC
                # grab file names for the PRECL and PRECC files from the case ts directory
                if glob.glob(os.path.join(ts_dir, "*PRECC*")) and glob.glob(
                    os.path.join(ts_dir, "*PRECL*")
                ):
                    constit_files = sorted(glob.glob(os.path.join(ts_dir, "*PREC*")))
                else:
                    ermsg = "PRECC and PRECL were not both present; PRECT cannot be calculated."
                    ermsg += " Please remove PRECT from diag_var_list or find the relevant CAM files."
                    raise FileNotFoundError(ermsg)
                # create new file name for PRECT
                prect_file = constit_files[0].replace("PRECC", "PRECT")
                if Path(prect_file).is_file():
                    if overwrite:
                        Path(prect_file).unlink()
                    else:
                        print(
                            f"[{__name__}] Warning: PRECT file was found and overwrite is False. Will use existing file."
                        )
                        continue
                # append PRECC to the file containing PRECL
                os.system(f"ncks -A -v PRECC {constit_files[0]} {constit_files[1]}")
                # create new file with the sum of PRECC and PRECL
                os.system(
                    f"ncap2 -s 'PRECT=(PRECC+PRECL)' {constit_files[1]} {prect_file}"
                )
            if var == "RESTOM":
                # RESTOM = FSNT-FLNT
                # Have to be more precise than with PRECT because FSNTOA, FSTNC, etc are valid variables
                if glob.glob(os.path.join(ts_dir, "*.FSNT.*")) and glob.glob(
                    os.path.join(ts_dir, "*.FLNT.*")
                ):
                    input_files = [
                        sorted(glob.glob(os.path.join(ts_dir, f"*.{v}.*")))
                        for v in ["FLNT", "FSNT"]
                    ]
                    constit_files = []
                    for elem in input_files:
                        constit_files += elem
                else:
                    ermsg = "FSNT and FLNT were not both present; RESTOM cannot be calculated."
                    ermsg += " Please remove RESTOM from diag_var_list or find the relevant CAM files."
                    raise FileNotFoundError(ermsg)
                # create new file name for RESTOM
                derived_file = constit_files[0].replace("FLNT", "RESTOM")
                if Path(derived_file).is_file():
                    if overwrite:
                        Path(derived_file).unlink()
                    else:
                        print(
                            f"[{__name__}] Warning: RESTOM file was found and overwrite is False. Will use existing file."
                        )
                        continue
                # append FSNT to the file containing FLNT
                os.system(f"ncks -A -v FLNT {constit_files[0]} {constit_files[1]}")
                # create new file with the difference of FLNT and FSNT
                os.system(
                    f"ncap2 -s 'RESTOM=(FSNT-FLNT)' {constit_files[1]} {derived_file}"
                )
