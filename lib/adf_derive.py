import glob
import os
from pathlib import Path
import xarray as xr


def check_derive(self, res, var, case_name, diag_var_list, constit_dict, hist_file_ds, hist0):
    """
    For incoming variable, look for list of constituents if available
     - as a list in variable defaults file

     If the variable does not have the argument `derivable_from` or `derivable_from_cam_chem`,
     then it will be assumed not to be a derivable variable, just missing from history file

     If the variable does have the argument `derivable_from` or `derivable_from_cam_chem`,
     first check cam-chem, then regular cam.

    Arguments
    ---------
        self: AdfDiag
            - ADF object
        res: dict
            - variable defaults dictionary from yaml file
        var: str
            - derived variable name
        case_name: str
            - model case
        diag_var_list: list
            - list of variables for diagnostics
            NOTE: this is user supplied, but gets modified here for constituents
        constit_dict: dict
            - dictionary of derived variables as keys and list of constituents as values
        hist_file_ds: xarray.DataSet
            - history file dataset for checking if constituents are available
        hist0: str
            - history number for case
    
    Returns
    -------
        constit_list: list
           - list of declared consituents from the variable defaults yaml file
           - empty list:
             * if missing `derived_from` argument(s)
             * if `derived_from` argument(s) exist but not declared
        
        diag_var_list: list
           - updated list (if applicable) of ADF variables for time series creation
    """

    # Aerosol Calcs
    #--------------

    # Always make sure PMID is made if aerosols are desired in config file
    # Since there's no requirement for `aerosol_zonal_list`, allow it to be absent:
    azl = res.get("aerosol_zonal_list", [])
    if azl:
        if "PMID" not in diag_var_list:
            if any(item in azl for item in diag_var_list):
                diag_var_list += ["PMID"]
        if "T" not in diag_var_list:
            if any(item in azl for item in diag_var_list):
                diag_var_list += ["T"]
    # End aerosol calcs

    # Set error messages for printing/debugging
    # Derived variable, but missing constituent list
    constit_errmsg = f"create time series for {case_name}:"
    constit_errmsg += f"\n Can't create time series for {var}. \n\tThis variable"
    constit_errmsg += " is flagged for derivation, but is missing list of constiuents."
    constit_errmsg += "\n\tPlease add list of constituents to 'derivable_from' "
    constit_errmsg += f"for {var} in variable defaults yaml file."

    # No time series creation
    exit_msg = f"WARNING: {var} is not in the file {hist0} and can't be derived."
    exit_msg += "\n\t  ** No time series will be generated. **\n"

    # Initialiaze list for constituents
    # NOTE: This is if the variable is NOT derivable but needs
    #       an empty list as a check later
    constit_list = []

    try_cam_constits = True
    # Try finding info from variable defaults yaml file
    try:
        vres = res[var]
    except KeyError:
        print(exit_msg)
        self.debug_log(exit_msg)
        return diag_var_list, constit_dict

    # Check first if variable is potentially part of a CAM-CHEM run
    if "derivable_from_cam_chem" in vres:
        constit_list = vres["derivable_from_cam_chem"]

        if constit_list:
            if all(item in hist_file_ds.data_vars for item in constit_list):
                # Set check to look for regular CAM constituents in variable defaults
                try_cam_constits = False
                msg = f"derive time series for {case_name}:"
                msg += "\n\tLooks like this a CAM-CHEM run, "
                msg += f"checking constituents for '{var}'"
                self.debug_log(msg)
        else:
            self.debug_log(constit_errmsg)
        # End if
    # End if
    
    # If not CAM-CHEM, check regular CAM runs
    if try_cam_constits:
        if "derivable_from" in vres:
            constit_list = vres["derivable_from"]
        else:
            # Missing variable or missing derivable_from argument
            der_from_msg = f"derive time series for {case_name}:"
            der_from_msg += f"\n Can't create time series for {var}.\n\tEither "
            der_from_msg += "the variable is missing from CAM output or it is a "
            der_from_msg += "derived quantity and is missing the 'derivable_from' "
            der_from_msg += "config argument.\n\tPlease add variable to CAM run "
            der_from_msg += "or set appropriate argument in variable "
            der_from_msg += "defaults yaml file."
            self.debug_log(der_from_msg)
        # End if
    # End if

    # Log if this variable can be derived but is missing list of constituents
    if isinstance(constit_list, list) and not constit_list:
        self.debug_log(constit_errmsg)

    # Check if any constituents were found
    if constit_list:
        # Add variable and constituent list to dictionary
        constit_dict[var] = constit_list

        # Add constituents to ADF diag variable list for time series generation
        for constit in constit_list:
            if constit not in diag_var_list:
                diag_var_list.append(constit)
    else:
        print(exit_msg)
        self.debug_log(exit_msg)
    # End if

    return diag_var_list, constit_dict

########

def derive_variable(self, case_name, var, res=None, ts_dir=None,
                         constit_list=None, overwrite=None):
    """
    Derive variables acccording to steps given here.  Since derivations will depend on the
    variable, each variable to derive will need its own set of steps below.

    Caution: this method assumes that there will be one time series file per variable

    If the file for the derived variable exists, the kwarg `overwrite` determines
    whether to overwrite the file (true) or exit with a warning message.

    """

    # Loop through derived variables
    print(f"\t - deriving time series for {var}")

    # Grab all required time series files for derived variable
    constit_files = []
    for constit in constit_list:
        # Check if the constituent file is present, if so add it to list
        if glob.glob(os.path.join(ts_dir, f"*.{constit}.*.nc")):
            constit_files.append(glob.glob(os.path.join(ts_dir, f"*.{constit}.*"))[0])
    # End for

    # Check if all the necessary constituent files were found
    if len(constit_files) != len(constit_list):
        ermsg = f"\t   ** Not all constituent files present; {var} cannot be calculated. **\n"
        ermsg += f"\t     Please remove {var} from 'diag_var_list' or find the "
        ermsg += "relevant CAM files.\n"
        print(ermsg)
        if constit_files:
            # Add what's missing to debug log
            dmsg = f"derived time series for {case_name}:"
            dmsg += f"\n\tneeded constituents for derivation of "
            dmsg += f"{var}:\n\t\t- {constit_list}\n\tfound constituent file(s) in "
            dmsg += f"{Path(constit_files[0]).parent}:\n\t\t"
            dmsg += f"- {[Path(f).parts[-1] for f in constit_files if Path(f).is_file()]}"
            self.debug_log(dmsg)
        else:
            dmsg = f"derived time series for {case_name}:"
            dmsg += f"\n\tneeded constituents for derivation of "
            dmsg += f"{var}:\n\t\t- {constit_list}\n"
            dmsg += f"\tNo constituent(s) found in history files"
            self.debug_log(dmsg)
        # End if
    else:
        # Open a new dataset with all the constituent files/variables
        ds = self.data.load_dataset(constit_files)
        if not ds:
            dmsg = f"derived time series for {case_name}:"
            dmsg += f"\n\tNo files to open."
            self.debug_log(dmsg)
            return

        # Grab attributes from first constituent file to be used in derived variable
        attrs = ds[constit_list[0]].attrs

        # create new file name for derived variable
        derived_file = constit_files[0].replace(constit_list[0], var)

        # Check if clobber is true for file
        if Path(derived_file).is_file():
            if overwrite:
                Path(derived_file).unlink()
            else:
                msg = f"[{__name__}] Warning: '{var}' file was found "
                msg += "and overwrite is False. Will use existing file."
                print(msg)

        #NOTE: this will need to be changed when derived equations are more complex! - JR
        if var == "RESTOM":
            der_val = ds["FSNT"]-ds["FLNT"]
        else:
            # Loop through all constituents and sum
            der_val = 0
            for v in constit_list:
                der_val += ds[v]

        # Set derived variable name and add to dataset
        der_val.name = var
        ds[var] = der_val

        # Aerosol Calculations
        #----------------------------------------------------------------------------------
        # These will be multiplied by rho (density of dry air)

        # User-defined defaults might not include aerosol zonal list
        azl = res.get("aerosol_zonal_list", [])
        if var in azl:
            # Check if PMID is in file:
            ds_pmid = self.data.load_dataset(glob.glob(os.path.join(ts_dir, "*.PMID.*"))[0])
            if not ds_pmid:
                errmsg = "Missing necessary files for dry air density (rho) "
                errmsg += "calculation.\nPlease make sure 'PMID' is in the CAM "
                errmsg += "run for aerosol calculations"
                print(errmsg)
                dmsg = "derived time series:"
                dmsg += f"\n\t missing 'PMID' in {ts_dir}, can't make time series for {var} "
                self.debug_log(dmsg)

            # Check if T is in file:
            ds_t = self.data.load_dataset(glob.glob(os.path.join(ts_dir, "*.T.*"))[0])
            if not ds_t:
                errmsg = "Missing necessary files for dry air density (rho) "
                errmsg += "calculation.\nPlease make sure 'T' is in the CAM "
                errmsg += "run for aerosol calculations"
                print(errmsg)

                dmsg = "derived time series:"
                dmsg += f"\n\t missing 'T' in {ts_dir}, can't make time series for {var} "
                self.debug_log(dmsg)

            # Multiply aerosol by dry air density (rho): (P/Rd*T)
            ds[var] = ds[var]*(ds_pmid["PMID"]/(res["Rgas"]*ds_t["T"]))

            # Sulfate conversion factor
            if var == "SO4":
                ds[var] = ds[var]*(96./115.)
        #----------------------------------------------------------------------------------

        # Drop all constituents from final saved dataset
        # These are not necessary because they have their own time series files
        ds_final = ds.drop_vars(constit_list)
        # Copy attributes from constituent file to derived variable
        ds_final[var].attrs = attrs
        ds_final.to_netcdf(derived_file, unlimited_dims='time', mode='w')
    # End if (all the necessary constituent files exist)
########