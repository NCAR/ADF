"""Driver for horizontal and vertical interpolation.
"""
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

import adf_utils as utils


# Default pressure levels for vertical interpolation
DEFAULT_PLEVS = [
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50,
    30, 20, 10, 7, 5, 3, 2, 1
]
# ndarray, not a list: geocat's interp_hybrid_to_pressure and utils.vert_remap
# both index it as an array (.size / .shape).
DEFAULT_PLEVS_Pa = np.array(DEFAULT_PLEVS, dtype=float) * 100.0

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

    case_names = adf.get_cam_info("cam_case_name", required=True)
    syear_cases = adf.climo_yrs["syears"]
    eyear_cases = adf.climo_yrs["eyears"]

    # Move critical variables to the front of the list
    for var in ["PMID", "OCNFRAC", "LANDFRAC", "PS"]:
        if var in var_list:
            var_list.insert(0, var_list.pop(var_list.index(var)))

    # The reference does not depend on the test cases, so do it once, up front:
    if not adf.compare_obs:
        _write_reference_files(adf, var_list, var_defaults, output_loc, overwrite_regrid)

    for case_idx, case_name in enumerate(case_names):
        # print(f"\t Regridding case '{case_name}':")
        syear = syear_cases[case_idx]
        eyear = eyear_cases[case_idx]

        for var in var_list:
            if var in adf.data.ref_var_nam:
                target_name = adf.data.ref_labels[var]
            else:
                print(f"\t ERROR: No reference data available for {var}.")
                continue

            regridded_file_loc = output_loc / f'{target_name}_{case_name}_{var}_regridded.nc'

            if regridded_file_loc.is_file() and not overwrite_regrid:
                print(f"\t INFO: Regridded file already exists, skipping: {regridded_file_loc}")
                continue
            
            if regridded_file_loc.is_file() and overwrite_regrid:
                regridded_file_loc.unlink()


            model_ds = adf.data.load_climo_ds(case_name, var)
            ref_ds = adf.data.load_reference_climo_ds(adf.data.ref_case_label, var)
            if not ref_ds:
                print(f"\t ERROR: Missing reference data for {var}. Skipping.")
                continue
            if not model_ds:
                print(f"\t ERROR: Missing model data for {var}. Skipping.")
                continue

            model_da = model_ds[var].squeeze()
            original_attrs = model_da.attrs.copy()

            # --- Horizontal Regridding ---
            regridded_da = _handle_horizontal_regridding(model_da, ref_ds, output_loc)
            regridded_da.attrs.update(original_attrs)
            # --- Vertical Interpolation ---
            # pass the Dataset: the hyam/hybm fallback check needs the other
            # variables.  Pass the DataArray too, so a 2-D field is not treated
            # as a model-level one just because its file carries hybrid coefficients.
            vert_type = _determine_vertical_coord_type(model_ds, regridded_da)
            ps_da = None
            pres_da = None
            if vert_type in ('hybrid', 'height'):
                # Prefer the model's own pressure field; fall back to PS + hybrid
                # coefficients. Either way it has to land on the target grid.
                lev_dim = 'lev' if 'lev' in model_ds.dims else 'ilev'
                pres_source = _find_pressure_field(model_ds, adf, lev_dim, case=case_name)
                if pres_source is not None:
                    original_pres_attrs = pres_source.attrs.copy()
                    pres_da = _handle_horizontal_regridding(pres_source, ref_ds, output_loc)
                    pres_da.attrs.update(original_pres_attrs)
                    pres_da = _pressure_in_pa(pres_da, name=('PMID' if lev_dim == 'lev' else 'PINT'))
                elif vert_type == 'hybrid':
                    ps_regridded_path = output_loc / f'{target_name}_{case_name}_PS_regridded.nc'
                    if ps_regridded_path.exists():
                        ps_da = xr.open_dataset(ps_regridded_path)['PS']
                    else:
                        ps_da_source = _find_surface_pressure(model_ds, adf, case=case_name)
                        if ps_da_source is None:
                            print(f"\t    WARNING: No PS available, unable to interpolate '{var}'")
                            continue
                        original_ps_attrs = ps_da_source.attrs.copy()
                        ps_da = _handle_horizontal_regridding(ps_da_source, ref_ds, output_loc)
                        ps_da.attrs.update(original_ps_attrs)
                    ps_da = _pressure_in_pa(ps_da, name="PS")
                else:
                    print(f"\t    WARNING: No PMID/PINT available, unable to interpolate '{var}'")
                    continue
            interp_da = _handle_vertical_interpolation(regridded_da, vert_type, model_ds,
                                                      ps_da=ps_da, pres_da=pres_da)
            interp_da.attrs.update(original_attrs)
            # --- Masking ---
            var_default_dict = var_defaults.get(var, {})
            if 'mask' in var_default_dict and var_default_dict['mask'].lower() == 'ocean':
                ocn_frac_regridded_path = output_loc / f'{target_name}_{case_name}_OCNFRAC_regridded.nc'
                if ocn_frac_regridded_path.exists():
                    ocn_frac_da = xr.open_dataset(ocn_frac_regridded_path)['OCNFRAC']
                    interp_da = _apply_ocean_mask(interp_da, ocn_frac_da)
                else:
                     print(f"\t    WARNING: OCNFRAC not found, unable to apply mask to '{var}'")

            # --- Save to file ---
            final_ds = interp_da.to_dataset(name=var)

            test_attrs_dict = {
                "adf_user": adf.user,
                "climo_yrs": f"{case_name}: {syear}-{eyear}",
                "climatology_files": str(adf.data.get_climo_file(case_name, var)),
            }
            final_ds = final_ds.assign_attrs(test_attrs_dict)
            save_to_nc(final_ds, regridded_file_loc)

    print("  ...CAM climatologies have been regridded successfully.")

def _pressure_in_pa(pres_da, name="PS"):
    """Return a pressure field in Pascals.

    The interpolation routines need Pa, but a pressure read back from a
    "*_regridded.nc" file has already had the variable defaults applied, and
    those convert pressures to hPa. Feeding hPa in silently squeezes the whole
    model column into a few hPa, so every target level below it comes out NaN --
    the troposphere disappears without an error anywhere.
    """
    units = str(pres_da.attrs.get('units', '')).strip().lower()
    if units in ('hpa', 'mb', 'millibar', 'millibars'):
        scaled = pres_da * 100.0
    elif units in ('pa', 'pascal', 'pascals'):
        return pres_da
    else:
        # No usable units attribute: tropospheric pressure in Pa is ~1e4-1e5,
        # in hPa ~1e2-1e3.
        if float(pres_da.max()) > 2000.0:
            return pres_da
        print(f"\t    WARNING: {name} has no units attribute and looks like hPa; "
              "converting to Pa for vertical interpolation.")
        scaled = pres_da * 100.0
    scaled.attrs = dict(pres_da.attrs)
    scaled.attrs['units'] = 'Pa'
    return scaled


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
    if ps_ds is None or 'PS' not in ps_ds:
        return None
    return ps_ds['PS'].squeeze()


def _find_pressure_field(dset, adf, level_dim, case=None):
    """The model's own 3-D pressure field, on the grid of `dset`.

    PMID for data on layer midpoints, PINT for data on interfaces. This is
    preferred over reconstructing pressure from PS and the hybrid coefficients:
    it is what the model actually used, and it is the only option for vertical
    coordinates that are not hybrid-sigma. Returns None when neither is
    available, in which case the caller falls back to PS + hyam/hybm.

    Note this deliberately does not look at "*_PMID_regridded.nc". PMID is a 3-D
    field, so if it is in diag_var_list then that file has already been
    interpolated onto the output pressure levels and is useless as a source
    pressure. The climo file is the honest source.

    Pass `case` for a test case; omit it for the reference.
    """
    name = 'PMID' if level_dim == 'lev' else 'PINT'
    if name in dset:
        return dset[name].squeeze()
    if case is None:
        # get_reference_climo_file avoids load_reference_climo_da's ref_var_nam
        # lookup, which raises KeyError when the field is not in diag_var_list.
        fils = adf.data.get_reference_climo_file(name)
        pres_ds = adf.data.load_dataset(fils) if fils else None
    else:
        pres_ds = adf.data.load_climo_ds(case, name)
    if pres_ds is None or name not in pres_ds:
        return None
    return pres_ds[name].squeeze()


def _interp_with_pressure_field(da, pres_da):
    """Interpolate `da` to DEFAULT_PLEVS using an explicit 3-D pressure field."""
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

    out = utils.pmid_to_plev(da, pres_da, new_levels=DEFAULT_PLEVS_Pa,
                            convert_to_mb=True)

    # vert_remap interpolates with np.interp, which clamps to the end values
    # instead of returning NaN outside the source range. The hybrid path (geocat)
    # returns NaN there, so mask below-ground and above-model-top levels to keep
    # the two paths comparable -- otherwise a 1000 hPa level over Tibet quietly
    # reports the lowest model level's value.
    lev_pa = out['lev'] * 100.0
    in_range = (lev_pa >= pres_da.min(dim='lev')) & (lev_pa <= pres_da.max(dim='lev'))
    return out.where(in_range)


def _write_reference_files(adf, var_list, var_defaults, output_loc, overwrite):
    """Write the reference climo on the target grid as {base}_{var}_baseline.nc.

    The reference defines the target grid, so it needs no horizontal regridding,
    but 3-D fields still have to land on the same pressure levels as the test
    cases. 2-D fields are written as a pass-through, because the plotting
    scripts read this file for every variable, not just the 3-D ones.
    """
    base = adf.data.ref_case_label
    syear = adf.climo_yrs["syear_baseline"]
    eyear = adf.climo_yrs["eyear_baseline"]

    for var in var_list:
        baseline_file = output_loc / f'{base}_{var}_baseline.nc'
        if baseline_file.is_file() and not overwrite:
            print(f"\t INFO: Baseline file already exists, skipping: {baseline_file}")
            continue

        ref_ds = adf.data.load_reference_climo_ds(base, var)
        if ref_ds is None:
            print(f"\t ERROR: Missing reference data for {var}. Skipping.")
            continue
        ref_da = ref_ds[adf.data.ref_var_nam[var]].squeeze()
        original_attrs = ref_da.attrs.copy()

        # --- Vertical Interpolation (no horizontal regrid: this IS the target grid) ---
        vert_type = _determine_vertical_coord_type(ref_ds, ref_da)
        ps_da = None
        pres_da = None
        if vert_type in ('hybrid', 'height'):
            # No horizontal regrid needed: the reference already defines the target grid.
            lev_dim = 'lev' if 'lev' in ref_ds.dims else 'ilev'
            pres_da = _find_pressure_field(ref_ds, adf, lev_dim)
            if pres_da is not None:
                pres_da = _pressure_in_pa(pres_da,
                                          name=('PMID' if lev_dim == 'lev' else 'PINT'))
            elif vert_type == 'hybrid':
                ps_da = _find_surface_pressure(ref_ds, adf)
                if ps_da is None:
                    print(f"\t    WARNING: No baseline PS, unable to interpolate '{var}'")
                    continue
                ps_da = _pressure_in_pa(ps_da, name="PS")
            else:
                print(f"\t    WARNING: No baseline PMID/PINT, unable to interpolate '{var}'")
                continue
        interp_da = _handle_vertical_interpolation(ref_da, vert_type, ref_ds,
                                                  ps_da=ps_da, pres_da=pres_da)
        interp_da.attrs.update(original_attrs)

        # --- Masking ---
        var_default_dict = var_defaults.get(var, {})
        if 'mask' in var_default_dict and var_default_dict['mask'].lower() == 'ocean':
            # var_list is sorted so OCNFRAC is written before anything that needs it:
            ocn_frac_path = output_loc / f'{base}_OCNFRAC_baseline.nc'
            if ocn_frac_path.exists():
                ocn_frac_da = xr.open_dataset(ocn_frac_path)['OCNFRAC']
                interp_da = _apply_ocean_mask(interp_da, ocn_frac_da)
            else:
                print(f"\t    WARNING: OCNFRAC not found, unable to apply mask to '{var}'")

        final_ds = interp_da.to_dataset(name=var)
        final_ds = final_ds.assign_attrs({
            "adf_user": adf.user,
            "climo_yrs": f"{base}: {syear}-{eyear}",
            "climatology_files": str(adf.data.get_reference_climo_file(var)),
        })
        save_to_nc(final_ds, baseline_file)


def _handle_horizontal_regridding(source_da, target_grid, regrid_loc, method='conservative'):
    """
    Performs horizontal regridding using xesmf.
    Manages and reuses regridding weight files.

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

    Returns
    -------
    xarray.DataArray
        The regridded DataArray.
    """

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

def _handle_vertical_interpolation(da, vert_type, source_ds, ps_da=None, pres_da=None):
    """
    Performs vertical interpolation to default pressure levels.

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

    Returns
    -------
    xarray.DataArray
        The vertically interpolated DataArray.
    """
    if vert_type == 'none':
        return da

    # An explicit pressure field wins whenever we have one.
    if pres_da is not None:
        return _interp_with_pressure_field(da, pres_da)

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

        return utils.lev_to_plev(da, ps_da, hyam, hybm, P0=p0, convert_to_mb=True, new_levels=DEFAULT_PLEVS_Pa)

    elif vert_type == "height":
        # Reaching here means _find_pressure_field came up empty, and a height
        # coordinate cannot be converted without one.
        raise ValueError("'PMID' (or 'PINT') is required for height vertical "
                         "interpolation, and neither was found.")

    elif vert_type == "pressure":
        return utils.plev_to_plev(da, new_levels=DEFAULT_PLEVS_Pa, convert_to_mb=True)

    else:
        raise ValueError(f"Unknown vertical coordinate type: '{vert_type}'")

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
    # deal with getting non-nan fill values.
    if isinstance(xo, xr.Dataset):
        enc_dv = {xname: {'_FillValue': None} for xname in xo.data_vars}
    else:
        enc_dv = {}
    #End if
    enc_c = {xname: {'_FillValue': None} for xname in xo.coords}
    enc = {**enc_c, **enc_dv}
    if attrs is not None:
        xo.attrs = attrs
    if proc is not None:
        origname = tosave.attrs.get('climatology_files', 'unknown')
        xo.attrs['Processing_info'] = f"Start from file {origname}. " + proc
    xo.to_netcdf(outname, format='NETCDF4', encoding=enc)
