"""Driver for horizontal and vertical interpolation.
"""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

import adf_utils as utils


# Default pressure levels (hPa) for vertical interpolation.  A model whose top
# is above 1 hPa, or a run that wants a different set, overrides these with
# 'interp_press_levels' in the config file -- see _interp_levels.
DEFAULT_PLEVS = [
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50,
    30, 20, 10, 7, 5, 3, 2, 1
]
# ndarray, not a list: geocat's interp_hybrid_to_pressure and utils.vert_remap
# both index it as an array (.size / .shape).
DEFAULT_PLEVS_Pa = np.array(DEFAULT_PLEVS, dtype=float) * 100.0

# Horizontal dimension of native (unstructured, e.g. spectral-element) CAM
# output.  See _to_latlon for how such data is handled.
# Native grid dimensions: CAM's "ncol" and CLM's "lndgrid":
UNSTRUCTURED_DIMS = ("ncol", "lndgrid")


def _interp_levels(adf):
    """The pressure levels to interpolate onto, in Pa.

    'interp_press_levels' (hPa, in diag_basic_info) replaces DEFAULT_PLEVS.
    The levels the config asks to *plot* are checked against them here: a
    'plot_press_levels' entry that was never interpolated to cannot be plotted,
    and saying so once at the start of regridding beats a plotting script
    finding an empty selection later on.

    Parameters
    ----------
    adf : AdfDiag
        The ADF object, for the config entries and for end_diag_fail.

    Returns
    -------
    numpy.ndarray
        The interpolation levels, in Pa, largest pressure first.
    """
    levels_pa = DEFAULT_PLEVS_Pa
    try:
        levels_pa = utils.pressure_levels_pa(
            adf.get_basic_info("interp_press_levels"), DEFAULT_PLEVS
        )
    except ValueError as err:
        # end_diag_fail raises AdfError; starting from the default means the
        # lines below cannot run on an unbound name if that ever changes.
        adf.end_diag_fail(f"Bad 'interp_press_levels' in the config file: {err}")

    levels_hpa = levels_pa / 100.0
    if not np.array_equal(levels_hpa, np.array(DEFAULT_PLEVS, dtype=float)):
        print(
            f"\t INFO: Interpolating onto {len(levels_hpa)} pressure levels "
            f"(hPa): {[float(lev) for lev in levels_hpa]}"
        )

    plot_levels = adf.get_basic_info("plot_press_levels") or []
    if isinstance(plot_levels, (int, float)) and not isinstance(plot_levels, bool):
        plot_levels = [plot_levels]
    try:
        missing = [lev for lev in plot_levels if float(lev) not in set(levels_hpa)]
    except (TypeError, ValueError):
        # Whatever 'plot_press_levels' holds, it is the plotting scripts' to
        # complain about; this check is a convenience and must not end the run.
        missing = []
    if missing:
        print(
            f"\t WARNING: 'plot_press_levels' asks for {missing} hPa, which "
            "is not among the levels being interpolated to, so those levels "
            "cannot be plotted.  Add them to 'interp_press_levels'."
        )
    return levels_pa

def regrid_and_vert_interp(adf):
    """
    Regrids the test cases to the same horizontal
    grid as the reference climatology and vertically
    interpolates the test case (and reference if needed)
    to match a default set of pressure levels (in hPa).
    """
    msg = "\n  Regridding CAM climatologies..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    overwrite_regrid = adf.get_basic_info("cam_overwrite_regrid", required=True)
    output_loc = Path(adf.get_basic_info("cam_regrid_loc", required=True))
    output_loc.mkdir(parents=True, exist_ok=True)
    var_list = adf.diag_var_list
    var_defaults = adf.variable_defaults
    levels_pa = _interp_levels(adf)

    # Cases whose pressure source has been reported; see _announce_pressure_source.
    announced = set()
    pressure_names = _pressure_field_names(adf)

    case_names = adf.get_cam_info("cam_case_name", required=True)
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    # Move critical variables to the front of the list
    for var in ["PMID", "OCNFRAC", "LANDFRAC", "PS"]:
        if var in var_list:
            var_list.insert(0, var_list.pop(var_list.index(var)))

    # The reference does not depend on the test cases, so do it once, up front:
    if not adf.compare_obs:
        _write_reference_files(
            adf, var_list, var_defaults, output_loc, overwrite_regrid, levels_pa
        )

    for case_idx, case_name in enumerate(case_names):
        # print(f"\t Regridding case '{case_name}':")
        syear = syear_cases[case_idx]
        eyear = eyear_cases[case_idx]

        for var in var_list:
            if var in adf.data.ref_var_nam:
                target_name = adf.data.ref_labels[var]
            elif var in pressure_names:
                # The pressure field the ADF added for itself.  It has no
                # observational counterpart and needs none: the interpolation
                # reads it straight from the climo file.  Saying "ERROR" here
                # would report a problem the run does not have.
                adf.debug_log(
                    f"No reference data for the pressure field '{var}';"
                    " not regridded, which is expected."
                )
                continue
            else:
                print(f"\t ERROR: No reference data available for {var}.")
                continue

            # Named after the case's climatology years, as its climo file is:
            regridded_file_loc = output_loc / utils.year_tagged_names(
                f'{target_name}_{case_name}_{var}_regridded', syear, eyear)[0]

            if regridded_file_loc.is_file() and not overwrite_regrid:
                print(f"\t INFO: Regridded file already exists, skipping: {regridded_file_loc}")
                continue
            
            if regridded_file_loc.is_file() and overwrite_regrid:
                regridded_file_loc.unlink()


            model_ds = adf.data.load_climo_ds(case_name, var)
            ref_ds = adf.data.load_reference_climo_ds(adf.data.ref_case_label, var)
            if not ref_ds:
                # The reference only supplies the target grid (the vertical
                # target is the fixed pressure levels), and its regridded file
                # is on that same grid -- so a baseline whose climo files are
                # gone but whose regridded files are still here will do:
                ref_ds = adf.data.load_reference_regrid_dataset(
                    adf.data.ref_case_label, var
                )
                if ref_ds:
                    adf.debug_log(
                        f"regrid: no reference climo for '{var}'; using the "
                        "reference's regridded file for the target grid"
                    )
            if not ref_ds:
                print(f"\t ERROR: Missing reference data for {var}. Skipping.")
                continue
            if not model_ds:
                print(f"\t ERROR: Missing model data for {var}. Skipping.")
                continue

            # Native-grid data that is not being plotted natively goes onto
            # its lat/lon grid first; from here on it is handled exactly like
            # any other lat/lon case:
            ref_ds = _to_latlon(adf, ref_ds, var)
            model_ds = _to_latlon(adf, model_ds, var, case_name=case_name, save=True)

            model_da = model_ds[var].squeeze()
            original_attrs = model_da.attrs.copy()

            # A lat/lon case against a reference on its native grid (plotted
            # natively): the case goes onto the reference's mesh, so the two
            # can be compared there.
            ref_mesh = None
            if _is_unstructured(ref_ds) and not _is_unstructured(model_da):
                ref_mesh = adf.mesh_files["baseline_mesh_file"]
                if not ref_mesh:
                    print(
                        f"\t WARNING: The reference for '{var}' is on a native grid "
                        f"with no 'mesh_file', so '{case_name}' cannot be put onto "
                        "it.  Skipping."
                    )
                    continue

            if _is_unstructured(model_da) and not _same_native_grid(model_da, ref_ds):
                # Plotting natively needs the test case and the reference on
                # one mesh.  A lat/lon reference (a lat/lon baseline or
                # observations) is put onto this case's mesh; the case itself
                # then passes through below, as for a reference on its mesh.
                # A reference on another mesh cannot be compared:
                if _is_unstructured(ref_ds) or not _write_reference_on_mesh(
                    adf, case_idx, case_name, var, target_name, model_ds,
                    output_loc, overwrite_regrid, (syear, eyear),
                ):
                    print(
                        f"\t WARNING: '{case_name}' is on a native grid but the "
                        f"reference for '{var}' is not on that same grid, so it can "
                        "not be plotted natively.  Skipping."
                    )
                    continue

            # --- Horizontal Regridding ---
            regridded_da = _handle_horizontal_regridding(
                model_da, ref_ds, output_loc, mesh_file=ref_mesh
            )
            regridded_da.attrs.update(original_attrs)
            # --- Vertical Interpolation ---
            # pass the Dataset: the hyam/hybm fallback check needs the other
            # variables.  Pass the DataArray too, so a 2-D field is not treated
            # as a model-level one just because its file carries hybrid coefficients.
            vert_type = _determine_vertical_coord_type(model_ds, regridded_da)
            ps_da = None
            pres_da = None
            pressure_note = None
            if vert_type in ('hybrid', 'height'):
                # Prefer the model's own pressure field; fall back to PS + hybrid
                # coefficients. Either way it has to land on the target grid.
                lev_dim = 'lev' if 'lev' in model_ds.dims else 'ilev'
                pres_source = _find_pressure_field(model_ds, adf, lev_dim, case=case_name)
                if pres_source is not None:
                    pres_name = pres_source.name
                    original_pres_attrs = pres_source.attrs.copy()
                    pres_da = _handle_horizontal_regridding(
                        pres_source, ref_ds, output_loc, mesh_file=ref_mesh
                    )
                    pres_da.attrs.update(original_pres_attrs)
                    pres_da = _pressure_in_pa(pres_da, name=pres_name)
                    _announce_pressure_source(announced, case_name, lev_dim, pres_name)
                    pressure_note = _pressure_note(pres_name)
                elif vert_type == 'hybrid':
                    ps_regridded_path = output_loc / utils.year_tagged_names(
                        f'{target_name}_{case_name}_PS_regridded', syear, eyear)[0]
                    if ps_regridded_path.exists():
                        ps_da = xr.open_dataset(ps_regridded_path)['PS']
                    else:
                        ps_da_source = _find_surface_pressure(model_ds, adf, case=case_name)
                        if ps_da_source is None:
                            print(f"\t    WARNING: No PS available, unable to interpolate '{var}'")
                            continue
                        original_ps_attrs = ps_da_source.attrs.copy()
                        ps_da = _handle_horizontal_regridding(
                            ps_da_source, ref_ds, output_loc, mesh_file=ref_mesh
                        )
                        ps_da.attrs.update(original_ps_attrs)
                    ps_da = _pressure_in_pa(ps_da, name="PS")
                    # Announced here, not before the branch: the fallback can
                    # come up empty too, and that path has already skipped the
                    # variable with a warning of its own.
                    _announce_pressure_source(announced, case_name, lev_dim, None)
                    pressure_note = _pressure_note(None)
                else:
                    print(f"\t    WARNING: No PMID/PINT available, unable to interpolate '{var}'")
                    continue
            interp_da = _handle_vertical_interpolation(
                regridded_da,
                vert_type,
                model_ds,
                ps_da=ps_da,
                pres_da=pres_da,
                levels_pa=levels_pa,
            )
            interp_da.attrs.update(original_attrs)
            # --- Masking ---
            var_default_dict = var_defaults.get(var, {})
            if (
                "mask" in var_default_dict
                and var_default_dict["mask"].lower() == "ocean"
            ):
                ocn_frac_da = _get_ocean_fraction(
                    adf, case_name, target_name, output_loc, ref_ds, (syear, eyear),
                    mesh_file=ref_mesh,
                )
                if ocn_frac_da is None:
                    # Writing the field unmasked is worse than not writing it:
                    # every plot and statistic downstream would be taken over
                    # land as well as ocean, and look perfectly plausible.
                    warnings.warn(
                        f"\t    WARNING: '{var}' is defined with 'mask: ocean', but no "
                        f"OCNFRAC is available for case '{case_name}'.  Skipping '{var}' "
                        "rather than writing it unmasked -- add 'OCNFRAC' to "
                        "'diag_var_list' to get this variable."
                    )
                    continue
                interp_da = _apply_ocean_mask(interp_da, ocn_frac_da)

            # --- Save to file ---
            final_ds = interp_da.to_dataset(name=var)
            # On a mesh, the cells' area (and land fraction) come from the
            # native data: the case's own, or the reference's when the case
            # was put onto the reference's mesh.
            _keep_native_area(final_ds, ref_ds if ref_mesh else model_ds)

            test_attrs_dict = {
                "adf_user": adf.user,
                # "YYYY-YYYY", as in the file name and the climo files:
                "climo_yrs": utils.years_tag(syear, eyear) or f"{syear}-{eyear}",
                "climo_case": case_name,
                "climatology_files": str(adf.data.get_climo_file(case_name, var)),
            }
            if pressure_note:
                # Provenance: this file may be reused by a later run, which will
                # not repeat the choice or say anything about it.
                test_attrs_dict["vert_interp_pressure"] = pressure_note
            final_ds = final_ds.assign_attrs(test_attrs_dict)
            save_to_nc(final_ds, regridded_file_loc)

    print("  ...CAM climatologies have been regridded successfully.")


def _pressure_field_names(adf):
    """The names the ADF may have added to the variable list for itself.

    The pressure fields are the only variables the ADF requests on a user's
    behalf here, and they are the only ones allowed to have no reference
    counterpart, so name them exactly rather than inferring it from a variable
    default that means something else.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object, for the 'pressure_field_names' config entry.

    Returns
    -------
    set of str
        Names to treat as the ADF's own; empty when the model's pressure field
        has been turned off in the config.
    """
    names = adf.get_basic_info("pressure_field_names")
    return {
        name
        for name in (utils.pressure_field_name(dim, names) for dim in ("lev", "ilev"))
        if name
    }


def _pressure_note(pres_name):
    """How the pressure for vertical interpolation was obtained, for the file.

    A regridded file outlives the run that made it: a later run reuses it
    without repeating the choice, or saying anything about it, so the answer
    belongs in the file as well as in the log.

    Parameters
    ----------
    pres_name : str or None
        Name of the model's own pressure field, or ``None`` when pressure was
        rebuilt from PS and the hybrid coefficients.

    Returns
    -------
    str
        A short description, for the "vert_interp_pressure" attribute.
    """
    if pres_name:
        return f"{pres_name} (the model's own pressure field)"
    return "reconstructed from PS and the hybrid coefficients"


def _announce_pressure_source(announced, label, level_dim, pres_name):
    """Say which pressure the vertical interpolation is using, once per case.

    Which pressure a run used is not a detail: the model's own field is the
    pressure the model actually had, while PS and the hybrid coefficients only
    reproduce it for a pure hybrid-sigma coordinate -- not for the dry-mass
    coordinate of recent CAM/WACCM.  Two runs of the same case can therefore
    differ for no reason visible in the output, so say which one this is.

    Call this only once the source is settled.  A variable whose pressure could
    not be worked out at all has already been skipped with a warning naming it,
    and announcing a source for it would contradict that warning.

    Parameters
    ----------
    announced : set
        Case/dimension pairs already announced; added to in place.
    label : str
        The case (or reference) name, so each one is announced once.
    level_dim : str
        The vertical dimension, ``'lev'`` or ``'ilev'``: a run can hold fields
        on both, with a pressure field for one and not the other.
    pres_name : str or None
        Name of the model's own pressure field, or ``None`` when pressure was
        rebuilt from PS and the hybrid coefficients.  A name is passed rather
        than the field itself because regridding drops it.
    """
    key = (label, level_dim)
    if key in announced:
        return
    announced.add(key)
    if pres_name:
        print(
            f"\t INFO: '{label}' ({level_dim}): vertical interpolation is using "
            f"the model's own pressure field, '{pres_name}'."
        )
    else:
        print(
            f"\t INFO: '{label}' ({level_dim}): no pressure field was found, so "
            "pressure is being reconstructed from PS and the hybrid "
            "coefficients.  That reproduces the model's pressure only for a "
            "pure hybrid-sigma coordinate; for the dry-mass coordinate of "
            "recent CAM/WACCM it is an approximation.  Add the model's "
            "pressure field to the run to avoid it."
        )


def _pressure_in_pa(pres_da, name="PS"):
    """Return a pressure field in Pascals; see adf_utils.pressure_in_pa."""
    return utils.pressure_in_pa(pres_da, name=name)


def _find_surface_pressure(dset, adf, case=None):
    """Surface pressure for hybrid-level interpolation, on the grid of `dset`.

    ADF climo files for a 3-D variable carry PS alongside hyam/hybm, so prefer
    that; fall back to a standalone PS climo file, which only exists when PS is
    itself in diag_var_list. Returns None if neither is available.

    Pass `case` for a test case; omit it for the reference.
    """
    if 'PS' in dset:
        return dset['PS'].squeeze()
    if case is None:
        # get_reference_climo_file avoids load_reference_climo_da's ref_var_nam
        # lookup, which raises KeyError when PS is not in diag_var_list.
        fils = adf.data.get_reference_climo_file('PS')
        ps_ds = adf.data.load_dataset(fils) if fils else None
    else:
        ps_ds = adf.data.load_climo_ds(case, 'PS')
    ps_ds = _to_latlon(adf, ps_ds, 'PS', case_name=case)
    if ps_ds is None or 'PS' not in ps_ds:
        return None
    return ps_ds['PS'].squeeze()


def _find_pressure_field(dset, adf, level_dim, case=None):
    """The model's own 3-D pressure field, on the grid of `dset`.

    PMID for data on layer midpoints, PINT for data on interfaces -- or
    whatever the model calls them, resolved by adf_utils.find_pressure_field.
    This is preferred over reconstructing pressure from PS and the hybrid
    coefficients: it is what the model actually used, and it is the only option
    for vertical coordinates that are not hybrid-sigma. Returns None when no
    pressure field is available, in which case the caller falls back to
    PS + hyam/hybm.

    Note this deliberately does not look at "*_PMID_regridded.nc". PMID is a 3-D
    field, so if it is in diag_var_list then that file has already been
    interpolated onto the output pressure levels and is useless as a source
    pressure. The climo file is the honest source.

    Pass `case` for a test case; omit it for the reference.
    """
    names = adf.get_basic_info("pressure_field_names")
    found = utils.find_pressure_field(dset, level_dim, names)
    if found:
        return dset[found].squeeze()
    name = utils.pressure_field_name(level_dim, names)
    if name is None:
        return None
    if case is None:
        # get_reference_climo_file avoids load_reference_climo_da's ref_var_nam
        # lookup, which raises KeyError when the field is not in diag_var_list.
        fils = adf.data.get_reference_climo_file(name)
        pres_ds = adf.data.load_dataset(fils) if fils else None
    else:
        pres_ds = adf.data.load_climo_ds(case, name)
    pres_ds = _to_latlon(adf, pres_ds, name, case_name=case)
    if pres_ds is None or name not in pres_ds:
        return None
    return pres_ds[name].squeeze()


def _interp_with_pressure_field(da, pres_da, levels_pa=None):
    """Interpolate `da` onto `levels_pa` using an explicit 3-D pressure field.

    `levels_pa` defaults to DEFAULT_PLEVS_Pa.
    """
    if levels_pa is None:
        levels_pa = DEFAULT_PLEVS_Pa
    level_dim = 'lev' if 'lev' in da.dims else 'ilev'

    # utils.pmid_to_plev stacks on a dimension literally named "lev", so
    # interface data has to be renamed on the way in. The output lands on
    # pressure levels named "lev" either way.
    if level_dim != 'lev':
        da = da.rename({level_dim: 'lev'})
        pres_da = pres_da.rename({level_dim: 'lev'})

    # vert_remap pairs the two stacked arrays column by column, so the pressure
    # field has to carry exactly the same dimensions in the same order.
    pres_da = pres_da.broadcast_like(da).transpose(*da.dims)

    out = utils.pmid_to_plev(da, pres_da, new_levels=levels_pa, convert_to_mb=True)

    # vert_remap interpolates with np.interp, which clamps to the end values
    # instead of returning NaN outside the source range. The hybrid path (geocat)
    # returns NaN there, so mask below-ground and above-model-top levels to keep
    # the two paths comparable -- otherwise a 1000 hPa level over Tibet quietly
    # reports the lowest model level's value.
    lev_pa = out['lev'] * 100.0
    in_range = (lev_pa >= pres_da.min(dim='lev')) & (lev_pa <= pres_da.max(dim='lev'))
    out = out.where(in_range)

    # pmid_to_plev stacks and unstacks, which leaves 'lev' leading. Put the
    # dimensions back in the source order so a file written through this path
    # looks like one written through the hybrid path.
    return out.transpose(*da.dims)


def _write_reference_files(
    adf, var_list, var_defaults, output_loc, overwrite, levels_pa=None
):
    """Write the reference climo on the target grid as {base}_{var}_baseline.nc.

    The reference defines the target grid, so it needs no horizontal regridding,
    but 3-D fields still have to land on the same pressure levels as the test
    cases. 2-D fields are written as a pass-through, because the plotting
    scripts read this file for every variable, not just the 3-D ones.
    """
    base = adf.data.ref_case_label
    announced = set()
    syear = adf.climo_yrs["syear_baseline"]
    eyear = adf.climo_yrs["eyear_baseline"]

    pressure_names = _pressure_field_names(adf)

    for var in var_list:
        if var in pressure_names and var not in adf.data.ref_var_nam:
            # The pressure field, added to the variable list after AdfData read
            # it, so it is not part of the reference bookkeeping.  Interpolating
            # it onto pressure levels would be circular anyway --
            # _find_pressure_field reads the climo file.  Without this,
            # load_reference_climo_ds raises KeyError as soon as the baseline
            # has a climo file of its own for it.
            adf.debug_log(
                f"The pressure field '{var}' is not a reference variable;"
                " not regridded for the baseline."
            )
            continue

        # Named after the baseline's climatology years:
        baseline_file = output_loc / utils.year_tagged_names(f'{base}_{var}_baseline', syear, eyear)[0]
        if baseline_file.is_file() and not overwrite:
            print(f"\t INFO: Baseline file already exists, skipping: {baseline_file}")
            continue

        ref_ds = adf.data.load_reference_climo_ds(base, var)
        if ref_ds is None:
            print(f"\t ERROR: Missing reference data for {var}. Skipping.")
            continue
        # A native-grid baseline that is not being plotted natively goes onto
        # its lat/lon grid, which then becomes the target grid for the cases:
        ref_ds = _to_latlon(adf, ref_ds, var, save=True)
        ref_da = ref_ds[adf.data.ref_var_nam[var]].squeeze()
        original_attrs = ref_da.attrs.copy()

        # --- Vertical Interpolation (no horizontal regrid: this IS the target grid) ---
        vert_type = _determine_vertical_coord_type(ref_ds, ref_da)
        ps_da = None
        pres_da = None
        pressure_note = None
        if vert_type in ('hybrid', 'height'):
            # No horizontal regrid needed: the reference already defines the target grid.
            lev_dim = 'lev' if 'lev' in ref_ds.dims else 'ilev'
            pres_da = _find_pressure_field(ref_ds, adf, lev_dim)
            if pres_da is not None:
                pres_name = pres_da.name
                pres_da = _pressure_in_pa(pres_da, name=pres_name)
                _announce_pressure_source(announced, base, lev_dim, pres_name)
                pressure_note = _pressure_note(pres_name)
            elif vert_type == 'hybrid':
                ps_da = _find_surface_pressure(ref_ds, adf)
                if ps_da is None:
                    print(f"\t    WARNING: No baseline PS, unable to interpolate '{var}'")
                    continue
                ps_da = _pressure_in_pa(ps_da, name="PS")
                _announce_pressure_source(announced, base, lev_dim, None)
                pressure_note = _pressure_note(None)
            else:
                print(f"\t    WARNING: No baseline PMID/PINT, unable to interpolate '{var}'")
                continue
        interp_da = _handle_vertical_interpolation(
            ref_da, vert_type, ref_ds, ps_da=ps_da, pres_da=pres_da, levels_pa=levels_pa
        )
        interp_da.attrs.update(original_attrs)

        # --- Masking ---
        var_default_dict = var_defaults.get(var, {})
        if 'mask' in var_default_dict and var_default_dict['mask'].lower() == 'ocean':
            # var_list is sorted so OCNFRAC is written before anything that needs it:
            ocn_frac_path = output_loc / utils.year_tagged_names(
                f'{base}_OCNFRAC_baseline', syear, eyear)[0]
            if ocn_frac_path.exists():
                ocn_frac_da = xr.open_dataset(ocn_frac_path)['OCNFRAC']
                interp_da = _apply_ocean_mask(interp_da, ocn_frac_da)
            else:
                print(f"\t    WARNING: OCNFRAC not found, unable to apply mask to '{var}'")

        final_ds = interp_da.to_dataset(name=var)
        _keep_native_area(final_ds, ref_ds)
        baseline_attrs = {
            "adf_user": adf.user,
            "climo_yrs": utils.years_tag(syear, eyear) or f"{syear}-{eyear}",
            "climo_case": base,
            "climatology_files": str(adf.data.get_reference_climo_file(var)),
        }
        if pressure_note:
            baseline_attrs["vert_interp_pressure"] = pressure_note
        final_ds = final_ds.assign_attrs(baseline_attrs)
        save_to_nc(final_ds, baseline_file)


def _write_reference_on_mesh(
    adf, case_idx, case_name, var, target_name, model_ds, output_loc, overwrite, years
):
    """Write the lat/lon reference for `var` onto a native test case's mesh.

    With 'unstructured_plotting' a native case is compared with its reference
    on its own mesh.  When the reference is on a lat/lon grid (a lat/lon
    baseline, or observations), its file as the plots would read it -- the
    baseline file written by `_write_reference_files`, already on the
    pressure levels, or the observation file -- is remapped onto the mesh and
    written as {ref}_{case}_{var}_reference_on_mesh_YYYY-YYYY.nc (the case's
    years), on the case's native dimension ("ncol" or "lndgrid").  The
    variable keeps its name and attributes, so it is converted when plotted
    exactly as the lat/lon file would be (see AdfData._regrid_converters).
    The case's cell areas (and land fraction) go with it, so both are
    averaged over the same cells.

    The remap uses ESMF conservative weights, made once per lat/lon grid and
    mesh and kept in the "regrid_weights" directory, and falls back to
    uxarray's bilinear remap when ESMF_RegridWeightGen is not available.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object.
    case_idx : int
        Index of the test case.
    case_name : str
        Name of the test case.
    var : str
        ADF name of the variable.
    target_name : str
        Label of the reference data set.
    model_ds : xarray.Dataset
        The case's climo, on its native grid.
    output_loc : pathlib.Path
        The regrid output directory.
    overwrite : bool
        Rewrite an existing file.
    years : tuple
        The case's climatology (first, last) years, which name the file.

    Returns
    -------
    bool
        True when the reference is on the mesh (written now or before).
    """
    out_file = output_loc / utils.year_tagged_names(
        f"{target_name}_{case_name}_{var}_reference_on_mesh", *years)[0]
    if out_file.is_file() and not overwrite:
        print(f"\t INFO: Reference on the mesh already exists, skipping: {out_file}")
        return True
    mesh_file = adf.mesh_files["test_mesh_file"][case_idx]
    if not mesh_file:
        print(
            f"\t WARNING: '{case_name}' has no 'mesh_file', so the reference "
            f"for '{var}' cannot be put onto its grid."
        )
        return False
    fils = adf.data.get_ref_regrid_file(adf.data.ref_case_label, var)
    if not fils:
        return False
    ref_ds = xr.open_dataset(fils[0])
    file_field = adf.data.ref_var_nam[var] if adf.compare_obs else var
    if file_field not in ref_ds:
        return False
    ref_da = ref_ds[file_field]
    if not ("lat" in ref_da.dims and "lon" in ref_da.dims):
        print(
            f"\t WARNING: The reference for '{var}' has no lat/lon dimensions "
            f"({', '.join(ref_da.dims)}), so it cannot be put onto the mesh."
        )
        return False

    native_dim = next(dim for dim in UNSTRUCTURED_DIMS if dim in model_ds.dims)
    out = _on_native_dim(_latlon_to_mesh(ref_da, mesh_file, output_loc), ref_da, native_dim)
    final_ds = out.to_dataset()
    # The case's cell areas (and land fraction), which do not change with
    # time: taken at one time, so they do not have to line up with the
    # reference's time axis (one observation file has a single time, 0):
    for name in ("area", "landfrac", "landmask"):
        if name in model_ds:
            fld = model_ds[name]
            final_ds[name] = fld.isel(time=0, drop=True) if "time" in fld.dims else fld
    final_ds = final_ds.assign_attrs(
        dict(
            ref_ds.attrs,
            adf_user=adf.user,
            reference_file=str(fils[0]),
            mesh_file=str(mesh_file),
            remapped_for_case=case_name,
        )
    )
    save_to_nc(final_ds, out_file)
    return True


def _on_native_dim(on_mesh, source_da, native_dim):
    """A remapped field (uxarray, on "n_face") as a plain DataArray on `native_dim`.

    The ADF writes and reads native files on the model's own dimension ("ncol"
    or "lndgrid"); the name, attributes and other coordinates of `source_da`
    are kept, with the remap method added.
    """
    other = [d for d in on_mesh.dims if d != "n_face"]
    return xr.DataArray(
        np.asarray(on_mesh.values),
        dims=other + [native_dim],
        coords={d: source_da[d] for d in other if d in source_da.coords},
        name=source_da.name,
        attrs=dict(source_da.attrs, remap_method=on_mesh.attrs.get("remap_method", "")),
    )


def _latlon_to_mesh(ref_da, mesh_file, output_loc):
    """Remap a lat/lon field onto a mesh: ESMF conservative, else uxarray.

    The weights are made once for each lat/lon grid and mesh and kept in
    `output_loc`/regrid_weights.
    """
    lat = ref_da["lat"].values
    lon = ref_da["lon"].values
    grid_id = hashlib.md5(lat.tobytes() + lon.tobytes()).hexdigest()[:8]
    weights_dir = Path(output_loc) / "regrid_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_file = weights_dir / (
        f"weights_latlon_{len(lat)}_{len(lon)}_{grid_id}_to_{Path(mesh_file).stem}_conserve.nc"
    )
    try:
        if not weights_file.is_file():
            scrip = weights_dir / f"scrip_latlon_{len(lat)}_{len(lon)}_{grid_id}.nc"
            utils.write_latlon_scrip(lat, lon, scrip)
            utils.make_mesh_weights(scrip, mesh_file, weights_file)
        return utils.apply_mesh_weights(ref_da, weights_file, mesh_file)
    except (OSError, utils.AdfError) as err:
        # e.g. no ESMF_RegridWeightGen on the PATH
        print(
            "\t    NOTE: ESMF weights for the mesh are not available "
            f"({str(err).splitlines()[0]}); using uxarray's bilinear remap."
        )
    try:
        return utils.latlon_to_mesh(ref_da, mesh_file, "bilinear")
    except (TypeError, ValueError):
        # uxarray 2025.6.0's bilinear remap fails on some grids (GPCP 2.5 deg)
        return utils.latlon_to_mesh(ref_da, mesh_file, "inverse_distance_weighted")


def _handle_horizontal_regridding(
    source_da, target_grid, regrid_loc, method='conservative', mesh_file=None
):
    """
    Performs horizontal regridding using xesmf.
    Manages and reuses regridding weight files.

    A lat/lon field whose target is on a native grid (a native reference
    plotted natively) is put onto that grid's mesh, `mesh_file`, instead (see
    `_latlon_to_mesh`).

    Parameters
    ----------
    source_da : xarray.DataArray
        The DataArray to regrid.
    target_grid : xarray.Dataset
        A dataset defining the target grid.
    regrid_loc : pathlib.Path
        The regrid output directory; weight files go in a subdirectory of it.
    method : str, optional
        Regridding method. Defaults to 'conservative'.
    mesh_file : str, optional
        The target's mesh, for a target on a native grid.

    Returns
    -------
    xarray.DataArray
        The regridded DataArray.
    """

    # Native-grid data being plotted natively stays on its mesh.  The main
    # loop has already checked that the target is that same mesh.
    if _is_unstructured(source_da):
        return source_da
    if _is_unstructured(target_grid):
        native_dim = next(dim for dim in UNSTRUCTURED_DIMS if dim in target_grid.dims)
        on_mesh = _latlon_to_mesh(source_da, mesh_file, regrid_loc)
        return _on_native_dim(on_mesh, source_da, native_dim)

    # Generate a unique name for the weights file
    source_grid_type = "unstructured" if "ncol" in source_da.dims else "structured"
    target_grid_type = "unstructured" if "ncol" in target_grid.dims else "structured"

    # A simple naming convention for weight files.
    source_grid_desc = f"{source_grid_type}_{len(source_da.lat)}_{len(source_da.lon)}" if source_grid_type == "structured" else f"{source_grid_type}_{len(source_da.ncol)}"
    target_grid_desc = f"{target_grid_type}_{len(target_grid.lat)}_{len(target_grid.lon)}" if target_grid_type == "structured" else f"{target_grid_type}_{len(target_grid.ncol)}"

    if target_grid_type == "structured":
        target_grid = utils.create_clean_grid(target_grid)
    source_grid = utils.create_clean_grid(source_da) if source_grid_type == "structured" else source_da

    regrid_weights_dir = Path(regrid_loc) / "regrid_weights"
    regrid_weights_dir.mkdir(parents=True, exist_ok=True)
    weights_file = regrid_weights_dir / f"weights_{source_grid_desc}_to_{target_grid_desc}_{method}.nc"
    if weights_file.exists():
        # xesmf can accept a path to a weights file
        regridder = xe.Regridder(source_grid, target_grid, method, weights=str(weights_file))
    else:
        regridder = xe.Regridder(source_grid, target_grid, method)
        regridder.to_netcdf(weights_file)
    return regridder(source_da)


def _determine_vertical_coord_type(dset, da=None):
    """
    Determines the type of vertical coordinate in a dataset.

    Parameters
    ----------
    dset : xarray.Dataset
        The dataset to inspect.  The whole dataset is needed because the
        hyam/hybm fallback check looks at the other variables in the file.
    da : xarray.DataArray, optional
        The variable actually being interpolated.  When given, a variable with
        no vertical dimension of its own reports 'none' even if the dataset it
        came from carries one.  This matters for time series written by GenTS,
        which copies every non-time-varying variable -- ``lev``, ``ilev``,
        ``hyam``, ``hybm`` -- into *every* file, so a 2-D field such as TS
        arrives in a dataset that has a ``lev`` dimension it does not use.

    Returns
    -------
    str
        The vertical coordinate type: 'hybrid', 'height', 'pressure', or 'none'.
    """

    #The variable's own dimensions decide whether it needs interpolating:
    if da is not None and not ({'lev', 'ilev'} & set(da.dims)):
        return 'none'
    #End if

    if 'lev' in dset.dims or 'ilev' in dset.dims:
        lev_coord_name = 'lev' if 'lev' in dset.dims else 'ilev'
        lev_attrs = dset[lev_coord_name].attrs

        if 'vert_coord' in lev_attrs:
            return lev_attrs['vert_coord']

        if 'long_name' in lev_attrs:
            lev_long_name = lev_attrs['long_name']
            if 'hybrid level' in lev_long_name:
                return "hybrid"
            if 'pressure level' in lev_long_name:
                return "pressure"
            if 'zeta level' in lev_long_name:
                return "height"

        # If no specific metadata is found, make an educated guess.
        # This part might need refinement based on expected data conventions.
        if 'hyam' in dset or 'hyai' in dset:
            return "hybrid"

        print(f"\t WARNING: Vertical coordinate type for '{lev_coord_name}' could not be determined. Assuming 'pressure'.")
        return "pressure"

    return 'none'


def _handle_vertical_interpolation(
    da, vert_type, source_ds, ps_da=None, pres_da=None, levels_pa=None
):
    """
    Performs vertical interpolation to a set of pressure levels.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to interpolate.
    vert_type : str
        The vertical coordinate type ('hybrid', 'height', 'pressure').
    source_ds : xarray.Dataset
        The source dataset containing auxiliary variables (e.g., hyam, hybm).
    ps_da : xarray.DataArray, optional
        Surface pressure (Pa), used to rebuild pressure from hybrid coefficients
        when no explicit pressure field is available.
    pres_da : xarray.DataArray, optional
        The model's own 3-D pressure field (Pa) -- PMID on midpoints, PINT on
        interfaces -- already on the same grid as `da`. Takes precedence over
        `ps_da`: it is the pressure the model used, rather than a reconstruction,
        and it is the only option for non-hybrid vertical coordinates.
    levels_pa : numpy.ndarray, optional
        The pressure levels to interpolate onto, in Pa.  Defaults to
        ``DEFAULT_PLEVS_Pa``.

    Returns
    -------
    xarray.DataArray
        The vertically interpolated DataArray.
    """
    if vert_type == 'none':
        return da

    if levels_pa is None:
        levels_pa = DEFAULT_PLEVS_Pa

    # An explicit pressure field wins whenever we have one.
    if pres_da is not None:
        return _interp_with_pressure_field(da, pres_da, levels_pa)

    if vert_type == "hybrid":
        if ps_da is None:
            raise ValueError("Surface pressure ('PS') is required for hybrid vertical interpolation.")
        
        lev_coord_name = 'lev' if 'lev' in source_ds.dims else 'ilev'
        hyam_name = 'hyam' if lev_coord_name == 'lev' else 'hyai'
        hybm_name = 'hybm' if lev_coord_name == 'lev' else 'hybi'

        if hyam_name not in source_ds or hybm_name not in source_ds:
            raise ValueError(f"Hybrid coefficients ('{hyam_name}', '{hybm_name}') not found in dataset.")

        hyam = source_ds[hyam_name]
        hybm = source_ds[hybm_name]
        
        if 'time' in hyam.dims:
            hyam = hyam.isel(time=0).squeeze()
        if 'time' in hybm.dims:
            hybm = hybm.isel(time=0).squeeze()

        p0 = source_ds.get('P0', 100000.0)
        if isinstance(p0, xr.DataArray):
            p0 = p0.values[0]

        # hot fix for lev attributes
        da[lev_coord_name].attrs["axis"] = "Z"
        da[lev_coord_name].attrs["positive"] = "down" # standard for pressure/hybrid
        da[lev_coord_name].attrs["standard_name"] = "atmosphere_hybrid_sigma_pressure_coordinate"

        return utils.lev_to_plev(
            da, ps_da, hyam, hybm, P0=p0, convert_to_mb=True, new_levels=levels_pa
        )

    elif vert_type == "height":
        # Reaching here means _find_pressure_field came up empty, and a height
        # coordinate cannot be converted without one.
        raise ValueError("'PMID' (or 'PINT') is required for height vertical "
                         "interpolation, and neither was found.")

    elif vert_type == "pressure":
        return utils.plev_to_plev(da, new_levels=levels_pa, convert_to_mb=True)

    else:
        raise ValueError(f"Unknown vertical coordinate type: '{vert_type}'")

def _get_ocean_fraction(
    adf, case_name, target_name, output_loc, ref_ds, years=(None, None), mesh_file=None
):
    """Return the case's OCNFRAC on the target grid, or ``None``.

    The already-regridded file is used when it exists.  In ``compare_obs`` mode
    it never does: OCNFRAC has no observational counterpart, so the main loop
    skips it.  The mask only needs the *model's* ocean fraction on the reference
    grid, though, so it is regridded here on the fly instead.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object.
    case_name : str
        Name of the test case.
    target_name : str
        Label of the reference data set the case is being regridded onto.
    output_loc : pathlib.Path
        The regrid output directory.
    ref_ds : xarray.Dataset
        Reference dataset defining the target grid.
    years : tuple, optional
        The case's climatology (first, last) years, which name its files.
    mesh_file : str, optional
        The reference's mesh, when `ref_ds` is on a native grid (see
        `_handle_horizontal_regridding`).

    Returns
    -------
    xarray.DataArray or None
        Ocean fraction on the target grid, or ``None`` if the case has no
        OCNFRAC climatology.
    """
    ocn_frac_path = output_loc / utils.year_tagged_names(
        f"{target_name}_{case_name}_OCNFRAC_regridded", *years)[0]
    if ocn_frac_path.exists():
        return xr.open_dataset(ocn_frac_path)["OCNFRAC"]
    ocn_frac_ds = adf.data.load_climo_ds(case_name, "OCNFRAC")
    ocn_frac_ds = _to_latlon(adf, ocn_frac_ds, "OCNFRAC", case_name=case_name)
    if ocn_frac_ds is None or "OCNFRAC" not in ocn_frac_ds:
        return None
    return _handle_horizontal_regridding(
        ocn_frac_ds["OCNFRAC"].squeeze(), ref_ds, output_loc, mesh_file=mesh_file
    )


def _apply_ocean_mask(da, ocn_frac_da):
    """
    Applies an ocean mask to a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to mask.
    ocn_frac_da : xarray.DataArray
        The ocean fraction DataArray.

    Returns
    -------
    xarray.DataArray
        The masked DataArray.
    """
    # Ensure ocean fraction is between 0 and 1
    ocn_frac_da = ocn_frac_da.clip(0, 1)
    
    # Apply the mask
    return utils.mask_land_or_ocean(da, ocn_frac_da)

def _is_unstructured(obj):
    """Whether `obj` (a Dataset or DataArray) is on a native unstructured grid."""
    dims = getattr(obj, "dims", ())
    return any(dim in dims for dim in UNSTRUCTURED_DIMS)


def _same_native_grid(da, target_grid):
    """Whether `target_grid` is on the same native unstructured grid as `da`.

    Parameters
    ----------
    da : xarray.DataArray
        Data on a native grid.
    target_grid : xarray.Dataset
        The reference data.

    Returns
    -------
    bool
        True when both have the same unstructured dimension, of the same size.
    """
    return any(
        dim in da.dims and target_grid.sizes.get(dim) == da.sizes[dim]
        for dim in UNSTRUCTURED_DIMS
    )


def _keep_native_area(final_ds, source_ds):
    """Copy the grid-cell area into an output file that stays on its native grid.

    The native (uxarray) plotting weights its averages by the model's cell
    area (and, for land, its land fraction; see adf_utils.cell_weights), which
    it reads from the regridded and baseline files.  On a lat/lon
    grid the plots weight by latitude instead, and the source grid's area would
    be wrong after regridding anyway, so it is only kept for native data.

    Parameters
    ----------
    final_ds : xarray.Dataset
        The dataset about to be written; updated in place.
    source_ds : xarray.Dataset
        The climo dataset it was made from.
    """
    if _is_unstructured(final_ds):
        for name in ("area", "landfrac", "landmask"):
            if name in source_ds:
                final_ds[name] = source_ds[name]


def _to_latlon(adf, dset, var, case_name=None, save=False):
    """Put native-grid climo data onto its lat/lon grid, unless plotting natively.

    Usually the climo files of a native-grid case are on lat/lon already: the
    ADF grids its time series (into a "gridded" directory) and builds the
    climatologies from those.  Climo files made from native time series --
    pre-made ones, say -- arrive here still on the native grid, and are gridded
    with the case's weights file, so that the rest of this script only ever
    sees lat/lon data and treats it exactly like any other lat/lon case.

    With 'unstructured_plotting', or for data that is not on a native grid,
    `dset` is returned unchanged.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object.
    dset : xarray.Dataset or None
        The climo dataset.
    var : str
        The variable to grid.
    case_name : str, optional
        The test case; omit it for the baseline.
    save : bool, optional
        Also write the gridded data to "gridded/{case}_{var}_gridded_climo.nc"
        under the case's climo directory.

    Returns
    -------
    xarray.Dataset or None
        The data on a lat/lon grid (or `dset` unchanged, see above).
    """
    # Check the data first: lat/lon data needs nothing else from `adf`.
    if not _is_unstructured(dset) or adf.unstructured_plotting:
        return dset

    if case_name is None:
        label = adf.data.ref_case_label
        wgt_file = adf.latlon_wgt_files["baseline_wgts_file"]
        method = adf.latlon_regrid_method["baseline_regrid_method"]
        latlon_file = adf.latlon_files["baseline_latlon_file"]
        climo_loc = adf.get_baseline_info("cam_climo_loc", required=True)
        years = (adf.climo_yrs["syear_baseline"], adf.climo_yrs["eyear_baseline"])
    else:
        case_idx = adf.get_cam_info("cam_case_name", required=True).index(case_name)
        label = case_name
        wgt_file = adf.latlon_wgt_files["test_wgts_file"][case_idx]
        method = adf.latlon_regrid_method["test_regrid_method"][case_idx]
        latlon_file = adf.latlon_files["test_latlon_file"][case_idx]
        climo_loc = adf.get_cam_info("cam_climo_loc", required=True)[case_idx]
        years = (adf.climo_yrs["syears"][case_idx], adf.climo_yrs["eyears"][case_idx])

    if not wgt_file:
        # Every variable of the case would fail the same way, so stop here:
        adf.end_diag_fail(
            f"'{label}' is on a native grid, but has no 'weights_file' to grid it "
            "to lat/lon.  Add one, or set 'unstructured_plotting: true'."
        )

    adf.debug_log(f"regrid: gridding native '{var}' for '{label}' to lat/lon")
    gridded = utils.grid_to_latlon(
        dset,
        var,
        wgt_file=wgt_file,
        method=method,
        comp=adf.model_component,
        latlon_file=latlon_file,
    )
    # (grid_to_latlon carries hyam, hybm, P0 and the rest over, which the
    # vertical interpolation needs.)

    if save:
        gridded_loc = Path(climo_loc) / "gridded"
        gridded_loc.mkdir(parents=True, exist_ok=True)
        save_to_nc(
            gridded,
            gridded_loc / utils.year_tagged_names(f"{label}_{var}_gridded_climo", *years)[0],
        )
    return gridded


def save_to_nc(tosave, outname, attrs=None, proc=None):
    """Saves xarray variable to new netCDF file
    
    Parameters
    ----------
    tosave : xarray.Dataset or xarray.DataArray
        data to write to file
    outname : str or Path
        output netCDF file path
    attrs : dict, optional
        attributes dictionary for data
    proc : str, optional
        string to append to "Processing_info" attribute    
    """

    xo = tosave
    # deal with getting non-nan fill values.  An encoding given to to_netcdf
    # replaces the variable's own, so keep any time units it had (see
    # adf_diag.save_to_nc):
    def _enc(xname):
        kept = {key: val for key, val in xo[xname].encoding.items()
                if key in ('units', 'calendar')}
        return {'_FillValue': None, **kept}

    if isinstance(xo, xr.Dataset):
        enc_dv = {xname: _enc(xname) for xname in xo.data_vars}
    else:
        enc_dv = {}
    #End if
    enc_c = {xname: _enc(xname) for xname in xo.coords}
    enc = {**enc_c, **enc_dv}
    if attrs is not None:
        xo.attrs = attrs
    if proc is not None:
        origname = tosave.attrs.get('climatology_files', 'unknown')
        xo.attrs['Processing_info'] = f"Start from file {origname}. " + proc
    xo.to_netcdf(outname, format='NETCDF4', encoding=enc)
