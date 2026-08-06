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
            # pass the Dataset: the hyam/hybm fallback check needs the other variables
            vert_type = _determine_vertical_coord_type(model_ds)
            ps_da = None
            if vert_type == 'hybrid':
                # For hybrid, we need surface pressure on the target grid.
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
            interp_da = _handle_vertical_interpolation(regridded_da, vert_type, model_ds, ps_da=ps_da)
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
        vert_type = _determine_vertical_coord_type(ref_ds)
        ps_da = None
        if vert_type == 'hybrid':
            # No horizontal regrid needed: the reference already defines the target grid.
            ps_da = _find_surface_pressure(ref_ds, adf)
            if ps_da is None:
                print(f"\t    WARNING: No baseline PS, unable to interpolate '{var}'")
                continue
        interp_da = _handle_vertical_interpolation(ref_da, vert_type, ref_ds, ps_da=ps_da)
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
        target_grid = _create_clean_grid(target_grid)
    source_grid = _create_clean_grid(source_da) if source_grid_type == "structured" else source_da

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


def _create_clean_grid(da):
    """
    Creates a minimal, CF-compliant xarray Dataset for xesmf from a DataArray.
    Adapted from regrid_and_vert_interp_2.py
    """
    if isinstance(da, xr.DataArray):
        ds = da.to_dataset()
    else:
        ds = da

    # Extract raw values
    lat_centers = ds.lat.values.astype(np.float64)
    lon_centers = ds.lon.values.astype(np.float64)

    if np.any(np.isnan(lat_centers)) or np.any(np.isinf(lat_centers)):
        print("ERROR: Found NaNs or Infs in latitude centers!")
        lat_centers = np.nan_to_num(lat_centers, nan=0.0, posinf=90.0, neginf=-90.0)

    # NOTE: do NOT nudge the centers off the poles. These become the output
    # coordinate, and a target grid at -89.999999 instead of -90 will not align
    # with the reference in xarray arithmetic, silently dropping both polar rows
    # from every difference field.

    # Build basic Dataset
    clean_ds = xr.Dataset(
        coords={
            "lat": (["lat"], lat_centers, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": (["lon"], lon_centers, {"units": "degrees_east", "standard_name": "longitude"}),
        }
    )

    # Add Bounds as vertices if they exist
    # Check for various possible bounds names
    lat_bnds_names = ['lat_bnds', 'lat_bounds', 'latitude_bnds', 'latitude_bounds']
    lon_bnds_names = ['lon_bnds', 'lon_bounds', 'longitude_bnds', 'longitude_bounds']
    
    lat_bnds = None
    lon_bnds = None
    
    for name in lat_bnds_names:
        if name in ds:
            lat_bnds = ds[name]
            break
    
    for name in lon_bnds_names:
        if name in ds:
            lon_bnds = ds[name]
            break
    
    if lat_bnds is not None and lon_bnds is not None:
        lat_v = np.append(lat_bnds.values[:, 0], lat_bnds.values[-1, 1])
        lon_v = np.append(lon_bnds.values[:, 0], lon_bnds.values[-1, 1])
    else:
        # CAM history/climo files carry no bounds, but conservative regridding needs
        # them, so infer cell edges as the midpoints between centers:
        lat_v = _edges_from_centers(lat_centers)
        lon_v = _edges_from_centers(lon_centers)

    # Cell edges may be extrapolated past the poles; clamp them to the valid range.
    # This is the bounds array, not the centers, so it does not affect the output grid.
    lat_v = np.clip(lat_v, -90.0, 90.0).astype(np.float64)

    # xesmf looks for 'lat_b' and 'lon_b' in the dataset for conservative regridding
    clean_ds["lat_b"] = (["lat_f"], lat_v, {"units": "degrees_north"})
    clean_ds["lon_b"] = (["lon_f"], lon_v.astype(np.float64), {"units": "degrees_east"})
    return clean_ds


def _edges_from_centers(centers):
    """Cell edges midway between centers, with the outer two extrapolated."""
    mids = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate([[2 * centers[0] - mids[0]], mids, [2 * centers[-1] - mids[-1]]])


def _determine_vertical_coord_type(dset):
    """
    Determines the type of vertical coordinate in a dataset.

    Parameters
    ----------
    dset : xarray.Dataset
        The dataset to inspect.

    Returns
    -------
    str
        The vertical coordinate type: 'hybrid', 'height', 'pressure', or 'none'.
    """

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

def _handle_vertical_interpolation(da, vert_type, source_ds, ps_da=None):
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
        Surface pressure DataArray, required for hybrid coordinates.

    Returns
    -------
    xarray.DataArray
        The vertically interpolated DataArray.
    """
    if vert_type == 'none':
        return da

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
        pmid = source_ds.get('PMID')
        if pmid is None:
            raise ValueError("'PMID' is required for height vertical interpolation.")
        return utils.pmid_to_plev(da, pmid, convert_to_mb=True, new_levels=DEFAULT_PLEVS_Pa)

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
