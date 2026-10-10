from pathlib import Path
import numpy as np
import xarray as xr

import adf_utils as utils


def check_derive(
    self, res, var, case_name, diag_var_list, constit_dict, hist_file_ds, hist0
):
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
    # --------------

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
    exit_msg += "\t     ** No time series will be generated. **"

    # Initialiaze list for constituents
    # NOTE: This is if the variable is NOT derivable but needs
    #       an empty list as a check later
    constit_list = []

    # Initialize tracking for error messages
    constit_errmsg_written = False

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
            # Only write error if we haven't already written it
            self.debug_log(constit_errmsg)
            constit_errmsg_written = True
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
            constit_errmsg_written = True
        # End if
    # End if

    # Log if this variable can be derived but is missing list of constituents
    # (only if not already logged above)
    if not constit_list and not constit_errmsg_written:
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


def find_constit(ts_dir, case_name, constit, hist_str=None, *, syr=None, eyr=None):
    """
    Locate a constituent's time series file(s) for one case and stream.

    Most specific first.  The history stream matters: derivation runs once per
    configured stream, and a case with two of them holds two files per
    constituent whose dates are identical, which cannot be combined.  The case
    name matters because find_ts_files falls back to a recursive search, so an
    unanchored pattern can reach another case's files when several cases share
    a time series tree.  The looser patterns are kept as fallbacks so
    directories that do not follow the naming still work.

    Parameters
    ----------
    ts_dir : str or Path
        directory holding the time series files
    case_name : str
        name of the case being processed
    constit : str
        variable name to search for
    hist_str : str, optional
        history stream being processed, e.g. "cam.h0a"
    syr, eyr : int, optional
        first and last year being processed.  When given, only the files
        needed to cover them are returned, so a directory holding more than
        one set of files for the same constituent (years 1-20 alongside years
        1-40, say) can still be used.

    Returns
    -------
    list of Path
        Matching files, sorted; empty if nothing matches.
    """
    patterns = []
    if hist_str:
        patterns.append(f"{case_name}.{hist_str}.{constit}.*.nc")
    # End if
    patterns += [f"{case_name}.*.{constit}.*.nc", f"*.{constit}.*.nc"]

    for pattern in patterns:
        found = utils.find_ts_files(ts_dir, pattern)
        if found:
            return utils.select_ts_files(found, syr, eyr)
        # End if
    # End for
    return []


def derive_variable(
    self,
    case_name,
    var,
    res=None,
    ts_dir=None,
    constit_list=None,
    overwrite=None,
    hist_str=None,
    *,
    syr=None,
    eyr=None,
):
    """
    Derive variables acccording to steps given here.  Since derivations will depend on the
    variable, each variable to derive will need its own set of steps below.

    The years being processed (`syr`, `eyr`) narrow each constituent's files
    to the ones needed, so a directory that holds two sets for the same
    constituent -- years 1-20 alongside years 1-40, which happens when a run
    is extended and the time series are remade over a longer period -- can
    still be used.

    A constituent may be split across several consecutive time series files,
    which is what GenTS produces when 'gents_slice_years' is set.  All of a
    constituent's files are opened together and the derived variable is
    written as a single file spanning them, which is what the built-in back
    end produces.  Constituents whose files cover *overlapping* periods are
    refused, because the combined time axis would contain duplicates.

    Only ADF/GenTS names, `{case}.{stream}.{variable}.{start}-{end}.nc`, are
    understood.  A chunked archive named some other way (a CMIP `_`-separated
    name, say) cannot have its dates read, and is refused rather than combined
    on a guess.

    Derivation runs once per configured history stream, so `hist_str` has to
    be passed for a case with more than one: without it the search matches
    every stream's copy of a constituent, and those cover the same dates and
    cannot be combined.

    If the file for the derived variable exists, the kwarg `overwrite` determines
    whether to overwrite the file (true) or exit with a warning message.

    """

    # Loop through derived variables
    print(f"\t - deriving time series for {var}")

    # Grab all required time series files for derived variable.  A constituent
    # can contribute more than one file, so key them by constituent rather than
    # counting files:
    constit_matches = {}
    for constit in constit_list:
        # Check if the constituent file(s) are present, if so add them to the dict
        matches = find_constit(ts_dir, case_name, constit, hist_str, syr=syr, eyr=eyr)
        if not matches:
            continue
        if utils.ts_files_overlap(matches):
            wmsg = f"\t   ** Time series files for constituent '{constit}' cover "
            wmsg += f"overlapping or unrecognized periods, so {var} cannot be "
            wmsg += "calculated. **\n"
            wmsg += "\t     Please leave only one set of time series files per "
            wmsg += "variable in the directory.\n"
            print(wmsg)
            continue
        # End if
        constit_matches[constit] = [str(f) for f in matches]
    # End for

    # Flattened, in constituent order, for opening as one dataset:
    constit_files = [
        f for constit in constit_list for f in constit_matches.get(constit, [])
    ]

    # Check if all the necessary constituent files were found
    if len(constit_matches) != len(constit_list):
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
            dmsg += (
                f"- {[Path(f).parts[-1] for f in constit_files if Path(f).is_file()]}"
            )
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

        # create new file name for derived variable.  The derived file holds
        # every constituent chunk that was just opened, so its name has to
        # advertise that whole span, not the span of the first chunk alone.
        # With one file per constituent the span is that file's own dates, so
        # the name is unchanged from what ADF has always written.
        first_files = constit_matches[constit_list[0]]
        # Substitute whole dot-separated tokens rather than doing a plain
        # string replace: a case name that happens to contain the constituent
        # name (say case 'bFSNTtest' with constituent 'FSNT') would otherwise
        # be rewritten too, producing a file nothing downstream can find.
        tokens = Path(first_files[0]).stem.split(".")
        span = utils.ts_file_span(first_files)
        for idx, token in enumerate(tokens):
            if token == constit_list[0]:
                tokens[idx] = var
                break
            # End if
        # End for
        if span:
            tokens[-1] = f"{span[0]}-{span[1]}"
        # End if
        derived_file = str(Path(first_files[0]).parent / (".".join(tokens) + ".nc"))

        # Check if clobber is true for file
        if Path(derived_file).is_file():
            if overwrite:
                Path(derived_file).unlink()
            else:
                msg = f"[{__name__}] Warning: '{var}' file was found "
                msg += "and overwrite is False. Will use existing file."
                print(msg)

        # NOTE: this will need to be changed when derived equations are more complex! - JR
        if var == "RESTOM":
            der_val = ds["FSNT"] - ds["FLNT"]
            der_long_name = "Net radiative flux at top of model (FSNT - FLNT)"
        else:
            # Loop through all constituents and sum
            der_val = 0
            for v in constit_list:
                der_val += ds[v]
            der_long_name = "Sum of " + ", ".join(constit_list)

        # Set derived variable name and add to dataset
        der_val.name = var
        ds[var] = der_val

        # Aerosol Calculations
        # ----------------------------------------------------------------------------------
        # These will be multiplied by rho (density of dry air)

        # User-defined defaults might not include aerosol zonal list
        azl = res.get("aerosol_zonal_list", [])
        if var in azl:
            # PMID and T are chunked the same way the constituents are, so pass
            # the whole list: taking [0] would multiply a full-span variable by
            # a single chunk, which time-axis alignment turns silently into NaN.
            # Check if PMID is in file:
            ds_pmid = self.data.load_dataset(
                find_constit(ts_dir, case_name, "PMID", hist_str, syr=syr, eyr=eyr)
            )
            if not ds_pmid:
                errmsg = "Missing necessary files for dry air density (rho) "
                errmsg += "calculation.\nPlease make sure 'PMID' is in the CAM "
                errmsg += "run for aerosol calculations"
                print(errmsg)
                dmsg = "derived time series:"
                dmsg += f"\n\t missing 'PMID' in {ts_dir}, can't make time series for {var} "
                self.debug_log(dmsg)
                return

            # Check if T is in file:
            ds_t = self.data.load_dataset(
                find_constit(ts_dir, case_name, "T", hist_str, syr=syr, eyr=eyr)
            )
            if not ds_t:
                errmsg = "Missing necessary files for dry air density (rho) "
                errmsg += "calculation.\nPlease make sure 'T' is in the CAM "
                errmsg += "run for aerosol calculations"
                print(errmsg)

                dmsg = "derived time series:"
                dmsg += (
                    f"\n\t missing 'T' in {ts_dir}, can't make time series for {var} "
                )
                self.debug_log(dmsg)
                return

            # PMID and T have to cover the same times as the constituents.
            # xarray aligns on time with an outer join, so a mismatch does not
            # raise -- it fills with NaN, and a wholly disjoint pair would write
            # an all-NaN field that looks like a real one.  Check before
            # multiplying rather than shipping that.
            for aux_name, aux_ds in (("PMID", ds_pmid), ("T", ds_t)):
                shared = np.intersect1d(ds[var]["time"].values, aux_ds["time"].values)
                if len(shared) == len(ds[var]["time"]):
                    continue
                # End if
                errmsg = f"\t   ** '{aux_name}' covers {len(shared)} of the "
                errmsg += f"{len(ds[var]['time'])} times needed for {var}, so the "
                errmsg += "dry air density calculation would be incomplete. **\n"
                errmsg += f"\t     Please check the '{aux_name}' time series files "
                errmsg += f"in {ts_dir}.\n"
                print(errmsg)
                dmsg = f"derived time series for {case_name}:"
                dmsg += f"\n\t '{aux_name}' time axis does not cover {var}; "
                dmsg += "skipping derivation."
                self.debug_log(dmsg)
                return
            # End for

            # Multiply aerosol by dry air density (rho): (P/Rd*T)
            ds[var] = ds[var] * (ds_pmid["PMID"] / (res["Rgas"] * ds_t["T"]))

            # Sulfate conversion factor
            if var == "SO4":
                ds[var] = ds[var] * (96.0 / 115.0)

            # Multiplying by density turned a mixing ratio into a concentration:
            der_long_name += ", times dry air density"
            attrs = {**attrs, "units": "kg/m3"}
        # ----------------------------------------------------------------------------------

        # Drop all constituents from final saved dataset
        # These are not necessary because they have their own time series files
        ds_final = ds.drop_vars(constit_list)
        # Copy attributes from constituent file to derived variable, but not its
        # name: the constituent's 'long_name' describes the constituent, not the
        # variable just derived from it.
        ds_final[var].attrs = {
            **attrs,
            "long_name": res.get(var, {}).get("long_name", der_long_name),
        }
        # open_mfdataset leaves time bounds as a *chunked* datetime array, and
        # xarray refuses to encode one of those when the units it inherits from
        # 'time' come without a dtype.  They are tiny, so just load them.
        # ponytail: load, not re-encode; revisit if a bounds variable is ever large
        for tvar in [
            v
            for v in ds_final.variables
            if ds_final[v].dtype.kind in "OM" and ds_final[v].chunks
        ]:
            ds_final[tvar] = ds_final[tvar].load()
        ds_final.to_netcdf(derived_file, unlimited_dims="time", mode="w")
    # End if (all the necessary constituent files exist)


########
