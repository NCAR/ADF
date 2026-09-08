"""
GenTS interface for the Atmospheric Diagnostics Framework (ADF).

Provides an alternative back end for the time series generation step, so a run
can build its time series with GenTS (https://github.com/AgentOxygen/GenTS) --
the tool the CESM project is standardizing on -- instead of ADF's built-in
"ncrcat" path.  Selected with 'ts_tool: gents' in the 'diag_basic_info' section
of the config file.

The two back ends are interchangeable.  Pointing GenTS at the history file
directory itself makes its output path template collapse to the output
directory, so it writes

    $case.$hist_str.$variable.$dates.nc

flat into 'cam_ts_loc' -- which is the naming ADF already looks for.  No
downstream ADF code needs to know which back end produced the files.

There is one real difference in file *content*.  ADF's ncrcat path copies
hyam/hybm/hyai/hybi and PS into every 3-D variable's file.  GenTS copies any
variable that is not time-varying into every file of a group, so hyam, hybm,
hyai, hybi, P0 and gw all still ride along -- but PS is time-varying, so GenTS
gives it its own file instead.  ADF's regridding already knows how to pick PS
up from a separate file, but only when PS is in 'diag_var_list', so this module
checks for that up front rather than letting every 3-D variable silently drop
out of the regridding step later on.
"""

# ++++++++++++++++++++++++++++++
# Import standard python modules
# ++++++++++++++++++++++++++++++

import sys
from pathlib import Path

import xarray as xr

# ADF modules:
from adf_base import AdfError
from adf_file_utils import describe_dir_problem
from adf_derive import check_derive, derive_variable

# ++++++++++++++++++++++++++++++


def _import_gents():
    """
    Import GenTS, or fail with an actionable message.

    GenTS is an optional dependency: it needs newer netCDF4/numpy than the
    pinned ADF environment provides, so it is deliberately not part of
    'env/conda_environment.yaml' and is only imported when actually requested.
    """
    try:
        from gents.hfcollection import HFCollection
        from gents.timeseries import TSCollection
    except ImportError as err:
        emsg = "'ts_tool: gents' requires the GenTS package, which was not found.\n"
        emsg += "\tInstall it with: pip install gents\n"
        emsg += (
            "\t(GenTS needs python>=3.10, numpy>=2.0 and netCDF4>=1.7.0, which may be\n"
        )
        emsg += (
            "\t newer than the pinned ADF environment, so it may need its own env.\n"
        )
        emsg += "\t Afterwards check numpy against numba, which the climatology step uses:\n"
        emsg += "\t numba caps the numpy it accepts and the cap moves with each numba\n"
        emsg += (
            "\t release, so a numpy your numba rejects makes every climatology fail.)\n"
        )
        emsg += "\tOr set 'ts_tool: adf' to use the built-in ncrcat path instead."
        raise AdfError(emsg) from err
    return HFCollection, TSCollection


def _uses_model_levels(hist_file, variables):
    """Return True if any of 'variables' is on model levels in 'hist_file'."""
    with xr.open_dataset(hist_file, decode_cf=False, decode_times=False) as ds:
        for var in variables:
            if var in ds and ({"lev", "ilev"} & set(ds[var].dims)):
                return True
    return False


def _expand_derived_vars(adf, case_name, hist_files):
    """
    Work out which variables GenTS actually needs to produce.

    Any variable in 'diag_var_list' that is not in the history files may still
    be derivable from constituents (e.g. RESTOM from FSNT and FLNT).  Those
    constituents have to be part of the GenTS request, otherwise there would be
    nothing to derive from afterwards.  Reuses ADF's existing 'check_derive' so
    the two back ends agree on what is derivable.

    Returns (variables_to_generate, constit_dict).
    """
    res = adf.variable_defaults
    diag_var_list = adf.diag_var_list
    constit_dict = {}

    with xr.open_dataset(
        hist_files[0], decode_cf=False, decode_times=False
    ) as hist_file_ds:
        hist_file_var_list = list(hist_file_ds.data_vars)
        # Iterate over a copy: check_derive extends the list with constituents.
        for var in list(diag_var_list):
            if var not in hist_file_var_list:
                print(
                    f"\t     {var} not in history file, will try to derive if possible"
                )
                diag_var_list, constit_dict = check_derive(
                    adf,
                    res,
                    var,
                    case_name,
                    diag_var_list,
                    constit_dict,
                    hist_file_ds,
                    hist_files[0],
                )
            # End if
        # End for
    # End with

    # Only ask GenTS for variables that are actually in the history files; the
    # derived ones are built afterwards from their constituents.
    wanted = [v for v in diag_var_list if v in hist_file_var_list]
    return wanted, constit_dict


def _restrict_to_vars(tsc, variables):
    """Keep only the GenTS orders whose primary variable was requested."""
    #'primary_var' is a key of GenTS's own order dictionaries rather than part
    # of its documented surface, so it is worth pinning down which GenTS version
    # a failure came from: a rename raises KeyError here.
    wanted = set(variables)
    return tsc.copy(ts_orders=[o for o in tsc if o["primary_var"] in wanted])


def create_time_series_gents(adf, baseline=False):
    """
    Generate ADF time series files using GenTS.

    Drop-in alternative to :meth:`AdfDiag.create_time_series`; see this
    module's docstring for how the two back ends line up.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object holding the configuration information.
    baseline : bool, optional
        If ``True``, generate the baseline case's time series; otherwise
        generate the test cases'.  Default is ``False``.

    Returns
    -------
    None
        Writes time series files into each case's ``cam_ts_loc``.

    Raises
    ------
    AdfError
        If GenTS is not installed, if ``gents_compression`` was given without
        ``gents_compression_level``, if a history file directory is missing or
        cannot be read, if ``cam_ts_loc`` cannot be written in while there are
        files to write, or if ``PS`` is absent from ``diag_var_list`` while
        model-level variables are being diagnosed.

    Notes
    -----
    Uses ``adf.get_basic_info``, ``adf.get_ts_case_config``,
    ``adf.diag_var_list``, ``adf.variable_defaults``, ``adf.num_procs``,
    ``adf.user``, ``adf.end_diag_fail`` and
    ``adf.derive_from_premade_ts``, plus ``check_derive`` and
    ``derive_variable`` from :mod:`adf_derive` and ``describe_dir_problem``
    from :mod:`adf_file_utils`.
    """

    HFCollection, TSCollection = _import_gents()

    # Notify user that script has started:
    msg = "\n  Calculating CAM time series with GenTS..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # GenTS draws progress bars on stdout.  Useful interactively, but they turn
    # a redirected ADF log into thousands of columns of bar characters, so only
    # ask for them when stdout is actually a terminal.
    show_progress = sys.stdout.isatty()

    # GenTS options (all optional; absent means "behave like the ADF back end").
    # These are flat keys rather than a 'gents_options' sub-dictionary because
    # ADF's config reader allows only one level of nesting (adf_config.py:111).
    all_vars = bool(adf.get_basic_info("gents_all_vars"))
    nested_layout = bool(adf.get_basic_info("gents_nested_layout"))
    slice_years = adf.get_basic_info("gents_slice_years")
    compression_alg = adf.get_basic_info("gents_compression")
    compression_level = adf.get_basic_info("gents_compression_level")

    # GenTS requires a level whenever an algorithm is given, so check here
    # rather than letting it fail part-way through generation:
    if compression_alg and compression_level is None:
        emsg = f"'gents_compression' is set to '{compression_alg}', but"
        emsg += " 'gents_compression_level' was not given.  Please provide a"
        emsg += " compression level (0-9), or remove 'gents_compression'."
        adf.end_diag_fail(emsg)
    # End if

    # Same config the built-in back end uses, so the two cannot drift apart:
    cfg = adf.get_ts_case_config(baseline=baseline)
    case_type_string = cfg["case_type_string"]

    # Loop over cases:
    for case_idx, case_name in enumerate(cfg["case_names"]):

        ts_dir = cfg["ts_dirs"][case_idx]

        print(f"\n  Generating CAM time series files for '{case_name}'...")
        print(f"\n    Writing time series files to {ts_dir}")

        start_year = cfg["start_years"][case_idx]
        end_year = cfg["end_years"][case_idx]

        #Check if particular case should be processed:
        if cfg["cam_ts_done"][case_idx]:
            emsg = "\tNOTE: Configuration file indicates time series files have been "
            emsg += f"pre-computed for case '{case_name}'.  Will rely on those files directly."
            print(emsg)
            # Pre-made time series still need their derived variables; see
            # AdfDiag.derive_from_premade_ts (issue #431):
            adf.derive_from_premade_ts(
                case_name,
                ts_dir,
                adf.variable_defaults,
                cfg["hist_str_list"][case_idx],
                syr=start_year,
                eyr=end_year,
            )
            continue
        # End if

        #Create path object for the CAM history file(s) location:
        starting_location = Path(cfg["cam_hist_locs"][case_idx])

        # Check that the path exists and can be read:
        hist_problem = describe_dir_problem(starting_location)
        if hist_problem:
            emsg = f"Provided {case_type_string} 'cam_hist_loc' directory"
            emsg += f" {hist_problem}.  Script is ending here."
            adf.end_diag_fail(emsg)
        # End if

        # Check if time series directory exists, and if not, then create it:
        Path(ts_dir).mkdir(parents=True, exist_ok=True)

        for hist_str in cfg["hist_str_list"][case_idx]:

            print(
                f"\t Processing time series for {case_type_string} {case_name}, "
                f"{hist_str} files:"
            )

            hist_files = sorted(starting_location.glob(f"*{hist_str}.*.nc"))
            if not hist_files:
                emsg = (
                    f"No history *{hist_str}.*.nc files found in '{starting_location}'."
                )
                emsg += " Script is ending here."
                adf.end_diag_fail(emsg)
            # End if

            # Work out the variable list before handing over to GenTS, so that
            # constituents of derived variables are generated too:
            if all_vars:
                wanted_vars = None
                _, constit_dict = _expand_derived_vars(adf, case_name, hist_files)
            else:
                wanted_vars, constit_dict = _expand_derived_vars(
                    adf, case_name, hist_files
                )
            # End if

            # GenTS writes PS to its own file rather than copying it into each
            # 3-D variable's file, so ADF needs it requested explicitly:
            check_vars = wanted_vars if wanted_vars is not None else adf.diag_var_list
            if "PS" not in adf.diag_var_list and _uses_model_levels(
                hist_files[0], check_vars
            ):
                emsg = "GenTS writes 'PS' to its own time series file instead of copying it\n"
                emsg += "\tinto every 3-D variable's file the way ADF's built-in back end does.\n"
                emsg += "\tADF needs 'PS' to interpolate model levels, so please add 'PS' to\n"
                emsg += "\t'diag_var_list' when using 'ts_tool: gents'."
                adf.end_diag_fail(emsg)
            # End if

            # Build the history file collection.  Filter on path first (free),
            # then on year (needs file metadata, which GenTS pulls on demand):
            hfc = HFCollection(str(starting_location), num_processes=adf.num_procs)
            hfc = hfc.include(f"*{hist_str}.*.nc")
            # Pull metadata explicitly so the progress bar can be silenced;
            # include_years would otherwise trigger it with its own default.
            # NOTE: pull_metadata mutates hfc in place and returns None, unlike
            # the include/slice calls around it, so it is deliberately not
            # reassigned here.
            hfc.pull_metadata(show_progress=show_progress)
            hfc = hfc.include_years(start_year, end_year)

            # One file per variable spanning the requested years, matching what
            # ADF's ncrcat back end produces, unless the user asked for slices.
            # Note GenTS defaults start_year to 0, which misaligns the slice
            # windows for anything that does not start at year 0.
            span = (end_year - start_year) + 1
            hfc = hfc.slice_groups(
                slice_size_years=slice_years or span, start_year=start_year
            )

            # Build the time series orders.  Strip any trailing separator:
            # GenTS joins the output directory to a path that already has a
            # leading separator.
            tsc = TSCollection(
                hfc, str(ts_dir).rstrip("/"), num_processes=adf.num_procs
            )

            if wanted_vars is not None:
                tsc = _restrict_to_vars(tsc, wanted_vars)
            # End if

            if not len(tsc):
                wmsg = (
                    f"\t    WARNING: GenTS found nothing to generate for '{hist_str}'."
                )
                print(wmsg)
                continue
            # End if

            if nested_layout:
                # The layout the CESM project archives: <component>/proc/tseries/<freq>/
                tsc = tsc.apply_path_swap("/hist/", "/proc/tseries/")
                tsc = tsc.append_timestep_dirs()
            # End if

            if cfg["overwrite_ts"][case_idx]:
                tsc = tsc.apply_overwrite("*")
            # End if

            if compression_alg:
                tsc = tsc.apply_compression(
                    level=compression_level, alg=compression_alg, path_glob="*"
                )
            # End if

            # Same provenance attributes the ncrcat back end adds via ncatted
            # (GenTS records its own version in 'gents_version'):
            tsc = tsc.add_attrs(
                {"adf_user": adf.user, "hist_file_locs": str(starting_location)}
            )

            # Only now is a write actually required, and only for the files
            # GenTS still has to make.  Checking before this point would fail a
            # run that writes nothing, which is how a read-only directory of
            # finished time series works today:
            ts_problem = describe_dir_problem(ts_dir, need_write=True)
            if ts_problem:
                emsg = f"Provided {case_type_string} 'cam_ts_loc' directory"
                emsg += f" {ts_problem}.\n\tSet 'cam_ts_loc' to a directory you own,"
                emsg += " or set 'cam_ts_done: true' to read the time series that are"
                emsg += " already there."
                adf.end_diag_fail(emsg)
            # End if

            print(f"\t - generating {len(tsc)} time series file(s) with GenTS")
            tsc.create_directories()
            tsc.execute(show_progress=show_progress)

            # Finally, run through the derived variables if applicable:
            if constit_dict:
                res = adf.variable_defaults
                for der_var, constit_list in constit_dict.items():
                    derive_variable(
                        adf,
                        case_name,
                        der_var,
                        res,
                        ts_dir,
                        constit_list,
                        hist_str=hist_str,
                        syr=start_year,
                        eyr=end_year,
                    )
                # End for
            # End if
        # End for hist_str
    # End cases loop

    print("  ...CAM time series file generation has finished successfully.")
