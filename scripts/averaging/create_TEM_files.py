"""
Module to create Transformed Eulerian Mean (TEM) diagnostic files.
"""

import xarray as xr
import numpy as np
from scipy import integrate
from numpy import ma
from datetime import date
from pathlib import Path
from glob import glob
from itertools import chain

import adf_utils as utils


#TEM variables written to (and plotted from) the TEM files.
#NOTE: TZM is not calculated here; tem.py derives it from THZM and PMID.
TEM_VAR_LIST = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","DELF",
                "UTENDVTEM","UTENDWTEM"]

#The zonal-mean fields calc_tem needs from the TEM history stream:
TEM_INPUT_VARS = ("Uzm", "Vzm", "Wzm", "THzm", "UVzm", "UWzm", "VTHzm")

#The TEM fields carried in the observation file, under their (lower-case) CAM
#names, mapped to the ADF names. There is no THZM or VZM: the ERA5 TEM file
#has neither.
OBS_TEM_VARS = {"uzm": "UZM", "epfy": "EPFY", "epfz": "EPFZ", "vtem": "VTEM",
                "wtem": "WTEM", "psitem": "PSITEM", "utendepfd": "DELF",
                "utendvtem": "UTENDVTEM", "utendwtem": "UTENDWTEM"}


def create_TEM_files(adf):
    """
    Calculate the TEM variables and write one netCDF file per case.

    Reads the zonal mean fields listed in TEM_INPUT_VARS from the history
    stream named by 'tem_hist_str' (default 'h4a') and writes
    '<case>.TEMdiag_<start>-<end>.nc' into 'cam_tem_loc'. The zonal mean surface
    and mid-level pressure (PS, PMID) are also copied in from the case's main
    history stream, if available, so that tem.py can convert potential
    temperature to temperature with the model's own pressure.

    A baseline simulation is appended to the list of cases and processed the
    same way. Observations already hold TEM diagnostics, so they are only
    renamed and written out.

    A case that cannot be processed is reported and skipped.
    """

    #Notify user that script has started:
    msg = "\n  Generating CAM TEM diagnostics files..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Special ADF variables
    #CAM simulation variables (these quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    base_name     = adf.get_baseline_info("cam_case_name")

    #Grab history files locations
    cam_hist_locs = adf.get_cam_info("cam_hist_loc", required=True)
    #Main history stream for each case (where PS and PMID are found):
    cam_hist_strs = [_first_stream(h, "cam.h0a") for h in adf.hist_string["test_hist_str"]]

    #Extract test case years
    start_years   = adf.climo_yrs["syears"]
    end_years     = adf.climo_yrs["eyears"]

    res = adf.variable_defaults # will be dict of variable-specific plot preferences

    #Grab TEM diagnostics options
    #----------------------------
    #Extract TEM file save locations
    tem_base_loc = adf.get_baseline_info("cam_tem_loc")
    tem_case_locs = adf.get_cam_info("cam_tem_loc")

    #If path not specified, skip TEM calculation:
    if tem_case_locs is None:
        print("\t 'cam_tem_loc' not found in 'diag_cam_climo', so no TEM files/diagnostics will be generated.")
        return
    #End if
    tem_locs = [_ensure_dir(loc) for loc in tem_case_locs]

    #One TEM history stream and one clobber setting per case, defaulting to h4a:
    ncases = len(case_names)
    hist_nums = [_first_stream(h, "h4a")
                 for h in _per_case(adf.get_cam_info("tem_hist_str"), "h4a", ncases)]
    overwrite_tem_cases = _per_case(adf.get_cam_info("overwrite_tem"), False, ncases)

    #Check if comparing to observations
    if adf.get_basic_info("compare_obs"):
        #Save Obs TEM file to first test case location
        _write_obs_tem_file(adf, TEM_VAR_LIST, res, tem_locs[0])
    elif tem_base_loc:
        cam_hist_locs.append(adf.get_baseline_info("cam_hist_loc", required=True))
        cam_hist_strs.append(_first_stream(adf.hist_string["base_hist_str"], "cam.h0a"))
        case_names.append(base_name)
        start_years.append(adf.climo_yrs["syear_baseline"])
        end_years.append(adf.climo_yrs["eyear_baseline"])
        tem_locs.append(_ensure_dir(tem_base_loc))
        overwrite_tem_cases.append(adf.get_baseline_info("overwrite_tem", False))
        hist_nums.append(_first_stream(adf.get_baseline_info("tem_hist_str"), "h4a"))
    else:
        print("\t 'cam_tem_loc' not found in 'diag_cam_baseline_climo', so no baseline files/diagnostics will be generated.")
    #End if (check for obs)

    #Loop over cases:
    skipped = []
    for case_idx, case_name in enumerate(case_names):

        print(f"\t Processing TEM for case '{case_name}' :")

        #Extract start and end year values:
        start_year = start_years[case_idx]
        end_year   = end_years[case_idx]

        #Create path object for the CAM history file(s) location:
        starting_location = Path(cam_hist_locs[case_idx])

        #Check that path actually exists:
        if not starting_location.is_dir():
            print(f"\t    WARNING: 'cam_hist_loc' directory '{starting_location}' "
                  f"not found, skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        #Check if history files actually exist:
        hist_str = hist_nums[case_idx]
        if not list(starting_location.glob(f"*{hist_str}.*.nc")):
            print(f"\t    WARNING: no CAM history {hist_str} files in "
                  f"'{starting_location}', skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        #Main history stream, for PS and PMID:
        hist0_str = cam_hist_strs[case_idx]

        #Get full path and file for file name
        output_loc_idx = tem_locs[case_idx]

        #Set case file name
        tem_fil = output_loc_idx / f'{case_name}.TEMdiag_{start_year}-{end_year}.nc'

        #Get current case tem over-write boolean
        overwrite_tem = overwrite_tem_cases[case_idx]

        #If the file is already there and clobber is off, there is nothing to do:
        if tem_fil.is_file() and not overwrite_tem:
            print("\t    INFO: Found TEM file and clobber is False, so moving to next case.")
            continue
        #End if
        if tem_fil.is_file():
            print("\t    INFO: Found TEM file but clobber is True, so over-writing file.")
        #End if

        #Glob each set of years (leading zeros on the year just in case)
        hist_files = _glob_years(starting_location, hist_str, start_year, end_year)
        hist0_files = _glob_years(starting_location, hist0_str, start_year, end_year)

        ds = xr.open_mfdataset(hist_files, decode_times=True, combine='by_coords')

        #Newer CESM versions call the zonal mean latitude 'zmlat' rather than
        #'zalat'; settle on 'zalat' so the rest of the ADF only sees one name:
        ds = _normalize_zonal_lat(ds)
        if 'zalat' not in ds.dims:
            print(f"\t    WARNING: neither 'zalat' nor 'zmlat' found in the {hist_str} "
                  f"files, skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        #Settle the vertical grid once, before the per-time loop, rather than
        #assuming the zonal-mean fields are on layer midpoints.
        ds, lev_name = harmonize_tem_levels(ds)
        if lev_name is None:
            print(f"\t    WARNING: skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        #calc_tem works one time step at a time:
        dstem0 = xr.concat([calc_tem(ds.isel(time=idx).squeeze(), lev_name)
                            for idx in range(ds.sizes["time"])], dim="time")

        #Use the midpoint of each averaging interval as the time stamp:
        dstem0 = utils.use_time_bounds_midpoint(dstem0)

        #Update the attributes
        dstem0.attrs = ds.attrs
        dstem0.attrs['created'] = str(date.today())
        dstem0[lev_name] = ds[lev_name]

        #Hybrid coefficients are time-invariant; take the pair that belongs to
        #the grid actually used, and drop the time dimension open_mfdataset
        #gives them when it concatenates the history files.
        for coef in ('hyam', 'hybm') if lev_name == 'lev' else ('hyai', 'hybi'):
            if coef in ds:
                dstem0[coef] = ds[coef].isel(time=0, drop=True) \
                               if 'time' in ds[coef].dims else ds[coef]
            #End if
        #End for

        #Add zonal mean PS and PMID, used by tem.py to derive TZM:
        if lev_name == 'lev':
            pres = _zonal_mean_pressure(hist0_files, dstem0)
            for pvar, pdata in pres.items():
                dstem0[pvar] = pdata
            #End for
            if 'PMID' not in pres:
                print(f"\t    WARNING: no PMID found in the {hist0_str} files for "
                      f"'{case_name}', so TZM will be derived with the hybrid "
                      "'lev' coordinate instead.")
            #End if
        #End if

        # write output to a netcdf file
        dstem0.to_netcdf(tem_fil, unlimited_dims='time', mode='w')

    #End for (case loop)

    #Notify user that script has ended, naming anything that did not get made
    #so a partial run is not reported as a complete one:
    if skipped:
        print(f"  ...TEM variables calculated, except for: {', '.join(skipped)}.")
    else:
        print("  ...TEM variables have been calculated successfully.")
    #End if

#########

# Helper functions
##################

def _per_case(value, default, ncases):
    """
    Return a config entry as one value per case: `value` unchanged if it is
    already a list, otherwise `ncases` copies of `value` (or of `default` when
    `value` is None).
    """
    if value is None:
        return [default] * ncases
    return value if isinstance(value, list) else [value] * ncases

#########

def _first_stream(hist_str, default):
    """
    Return the first history stream named by a 'hist_str' config entry, which
    can be a string, a (possibly nested) list, or None.
    """
    while isinstance(hist_str, list) and hist_str:
        hist_str = hist_str[0]
    return hist_str if hist_str else default

#########

def _ensure_dir(path):
    """
    Return `path` as a Path, creating the directory if it does not exist.
    """
    path = Path(path)
    if not path.is_dir():
        print(f"    {path} not found, making new directory")
        path.mkdir(parents=True)
    #End if
    return path

#########

def _glob_years(location, hist_str, start_year, end_year):
    """
    Return the sorted history files for stream `hist_str` covering the years
    `start_year` through `end_year`.
    """
    return sorted(chain.from_iterable(
        glob(f"{location}/*{hist_str}.{yr:04d}*.nc")
        for yr in range(int(start_year), int(end_year) + 1)))

#########

def _normalize_zonal_lat(ds):
    """
    Rename the zonal mean latitude 'zmlat' to 'zalat', if needed.

    If both are present, 'zalat' is kept and 'zmlat' is dropped.
    """
    if 'zmlat' in ds.dims or 'zmlat' in ds.variables:
        if 'zalat' in ds.dims:
            ds = ds.drop_vars('zmlat')
        else:
            ds = ds.rename({'zmlat': 'zalat'})
        #End if
    #End if
    return ds

#########

def _zonal_mean_pressure(hist0_files, dstem):
    """
    Return the zonal mean PS and PMID, interpolated onto the TEM latitudes.

    Parameters
    ----------
    hist0_files : list
        History files from the case's main stream.
    dstem : xarray.Dataset
        TEM dataset, used for its 'zalat' coordinate and time stamps.

    Returns
    -------
    dict
        {name: DataArray} for whichever of PS and PMID could be found.
    """
    if not hist0_files:
        return {}
    #End if

    ds_h0 = xr.open_mfdataset(hist0_files, decode_times=True, combine='by_coords')
    ds_h0 = utils.use_time_bounds_midpoint(ds_h0)

    target_lat = xr.DataArray(dstem['zalat'].values, dims="zalat",
                              coords={"zalat": dstem['zalat'].values})
    out = {}
    for pvar in ("PS", "PMID"):
        if pvar not in ds_h0:
            continue
        #End if
        #Take the zonal mean first, then interpolate onto the TEM latitudes
        #(interpolating onto a target named 'zalat' renames the dimension):
        pdata = ds_h0[pvar].mean(dim="lon")
        pdata = pdata.interp(lat=target_lat, method="linear",
                             kwargs={"fill_value": "extrapolate"})
        #Match the TEM time stamps (both are interval midpoints):
        pdata = pdata.assign_coords(time=dstem['time'].values) \
                if pdata.sizes['time'] == dstem.sizes['time'] else pdata
        out[pvar] = pdata.compute()
    #End for
    return out

#########

def _write_obs_tem_file(adf, var_list, res, output_loc):
    """
    Write 'Obs.TEMdiag.nc' from pre-computed observational TEM diagnostics.

    Nothing is calculated here. The observational file already holds TEM
    diagnostics; only the variable and coordinate names differ from what the
    rest of the ADF expects.
    """
    print("\t Processing TEM for observations :")

    #Not every TEM variable has an observational counterpart, so only ask for
    #the ones that declare a file:
    obs_data_loc = adf.get_basic_info("obs_data_loc")
    tem_obs_fils = []
    for var in var_list:
        obs_file = res.get(var, {}).get("obs_file")
        if obs_file is None:
            continue
        #End if
        obs_file_path = Path(obs_file)
        if not obs_file_path.is_file() and obs_data_loc:
            obs_file_path = Path(obs_data_loc) / obs_file_path
        #End if
        #Several TEM variables usually share one file:
        if obs_file_path not in tem_obs_fils:
            tem_obs_fils.append(obs_file_path)
        #End if
    #End for

    if not tem_obs_fils:
        print("\t    WARNING: no observation files found for any TEM variable, "
              "so no observational TEM file will be written.")
        return
    #End if

    ds_obs = xr.open_mfdataset(tem_obs_fils, decode_times=True, combine='by_coords')
    missing = [v for v in OBS_TEM_VARS if v not in ds_obs]
    if missing:
        print(f"\t    WARNING: observation TEM data has no {', '.join(missing)}.")
    #End if

    #The observation file uses the lower-case CAM names; the TEM files the ADF
    #writes, and the plotting script, use the upper-case ADF ones.
    ds_base = xr.Dataset(
        {adf_name: xr.Variable(('time', 'lev', 'zalat'), ds_obs[obs_name].data,
                               attrs=ds_obs[obs_name].attrs)
         for obs_name, adf_name in OBS_TEM_VARS.items() if obs_name in ds_obs},
        coords={'lev': ds_obs.level.values,
                'zalat': ds_obs.lat.values,
                'time': ds_obs.time.values},
        attrs={**ds_obs.attrs, 'created': str(date.today())})

    print(f"\t NOTE: Observation TEM file being saved to '{output_loc}'")
    ds_base.to_netcdf(output_loc / 'Obs.TEMdiag.nc', unlimited_dims='time', mode='w')

#########

def harmonize_tem_levels(ds):
    """
    Put the TEM input fields on a single vertical grid, and report which one.

    Returns the dataset and the name of its vertical dimension ('lev' or
    'ilev'), or None when the fields cannot be put on a common grid.

    CAM may write the zonal mean fields on layer midpoints ('lev') or on layer
    interfaces ('ilev'), and a history stream can carry a mix of the two. When
    they all agree the grid is left alone. When they disagree the midpoint grid
    is used and the interface fields are interpolated onto it.
    """
    present = {v: utils.vertical_dim(ds[v]) for v in TEM_INPUT_VARS if v in ds}
    missing = [v for v in TEM_INPUT_VARS if v not in ds]
    if missing:
        print(f"\t    WARNING: TEM input fields missing from history files: {', '.join(missing)}.")
        return ds, None
    #End if
    found = {dim for dim in present.values() if dim is not None}

    if not found:
        print("\t    WARNING: none of the TEM input fields have a 'lev' or "
              "'ilev' dimension.")
        return ds, None
    if len(found) == 1:
        return ds, found.pop()
    #End if

    #Mixed grids: interpolate the interface fields onto the midpoints.
    if "lev" not in ds.coords:
        print("\t    WARNING: TEM inputs are on mixed vertical grids but the "
              "history files have no 'lev' coordinate to interpolate onto.")
        return ds, None
    #End if
    mixed = sorted(v for v, dim in present.items() if dim == "ilev")
    print("\t    INFO: TEM inputs are on mixed vertical grids; interpolating "
          f"{', '.join(mixed)} from 'ilev' onto 'lev'.")
    target = xr.DataArray(ds["lev"].values, dims="lev",
                          coords={"lev": ds["lev"].values})
    out = ds.copy()
    for var in mixed:
        out[var] = ds[var].interp(ilev=target, method="linear",
                                  kwargs={"fill_value": "extrapolate"})
    #End for
    return out, "lev"

#########

def calc_tem(ds, lev_name="lev"):
    """
    # calc_tem() function to calculate TEM diagnostics on CAM/WACCM output
    # This assumes the data have already been organized into zonal mean fluxes
    # Uzm, THzm, VTHzm, Vzm, UVzm, UWzm, Wzm
    # note that calculations are performed on model interface levels, which is ok
    # in the stratosphere but not in the troposphere.  If interested in tropospheric
    # TEM diagnostics, make sure input fields have been interpolated to true pressure levels.

    # The outputs are 2-dimensional (`lev_name`, zalat) DataArrays for one time step.
    # Iterate over time to process more than one time step.

    # The values of the `lev_name` coordinate are taken to be pressure in hPa.

    # Uses the TEM recipe in Appendix A of Gerber, E. P. and Manzini, E.:
    # The Dynamics and Variability Model Intercomparison Project (DynVarMIP) for CMIP6:
    # assessing the stratosphere–troposphere system, Geosci. Model Dev., 9, 3413–3425,
    # https://doi.org/10.5194/gmd-9-3413-2016, 2016 and the corrigendum.

    # pdf available here: https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016.pdf,
    # https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016-corrigendum.pdf

    # Output (Table A1 of above reference, plus zonal means):
    # UZM       Zonal-Mean zonal wind [m s-1]
    # VZM       Zonal-Mean meridional wind [m s-1]
    # THZM      Zonal-Mean potential temperature [K]
    # EPFY      northward component of the Eliassen–Palm flux [m3 s−2]
    # EPFZ      upward component of the Eliassen–Palm flux [m3 s−2]
    # VTEM      Transformed Eulerian mean northward wind [m s−1]
    # WTEM      Transformed Eulerian mean upward wind [m s−1]
    # PSITEM    Transformed Eulerian mean mass stream function [kg s−1]
    # DELF      tendency of eastward wind due to Eliassen–Palm flux divergence [m s−2]
    #           (called 'utendepfd' in DynVarMIP and the observations)
    # UTENDVTEM tendency of eastward wind due to TEM northward wind advection and the Coriolis term [m s−2]
    # UTENDWTEM tendency of eastward wind due to TEM upward wind advection [m s−2]

    # Based upon python code developed by Isla Simpson 25 Feb 2021
    # Initial coding of stand alone function: Dan Marsh 16 Dec 2022
    """

    # constants for TEM calculations
    p0 = 101325.
    a = 6.371e6
    om = 7.29212e-5
    H = 7000.
    g0 = 9.80665

    nlat = ds['zalat'].size
    nlev = ds[lev_name].size

    latrad = np.radians(ds.zalat)
    coslat = np.cos(latrad)
    coslat2d = np.tile(coslat,(nlev,1))

    pre = ds[lev_name]*100. # pressure levels in Pascals
    f = 2.*om*np.sin(latrad[:])
    f2d = np.tile(f,(nlev,1))

    # change missing values to NaNs
    uzm = ds['Uzm']
    uzm.values = ma.masked_greater_equal(uzm, 1e33)
    vzm = ds['Vzm']
    vzm.values = ma.masked_greater_equal(vzm, 1e33)
    wzm = ds['Wzm']
    wzm.values = ma.masked_greater_equal(wzm, 1e33)
    thzm = ds['THzm']
    thzm.values = ma.masked_greater_equal(thzm, 1e33)

    uvzm = ds['UVzm']
    uvzm.values = ma.masked_greater_equal(uvzm, 1e33)
    uwzm = ds['UWzm']
    uwzm.values = ma.masked_greater_equal(uwzm, 1e33)
    vthzm = ds['VTHzm']
    vthzm.values = ma.masked_greater_equal(vthzm, 1e33)

    # convert w terms from m/s to Pa/s
    wzm  = -1.*wzm*pre/H
    uwzm = -1.*uwzm*pre/H

    # compute the latitudinal gradient of U
    dudphi = (1./(a*coslat2d))*np.gradient(uzm*coslat2d,
                                latrad,
                                axis=1)

    # compute the vertical gradient of theta and u
    dthdp = np.gradient(thzm,
                        pre,
                        axis=0)

    dudp = np.gradient(uzm,
                       pre,
                       axis=0)

    # compute eddy streamfunction and its vertical gradient
    psieddy = vthzm/dthdp
    dpsidp = np.gradient(psieddy,
                         pre,
                         axis=0)

    # (1/acos(phii))**d(psi*cosphi/dphi) for getting w*
    dpsidy = (1./(a*coslat2d)) \
           * np.gradient(psieddy*coslat2d,
                         latrad,
                         axis=1)

    # TEM vertical velocity (Eq A7 of dynvarmip)
    wtem = wzm+dpsidy

    # utendwtem (Eq A10 of dynvarmip)
    utendwtem = -1.*wtem*dudp

    # vtem (Eq A6 of dynvarmip)
    vtem = vzm-dpsidp

    # utendvtem (Eq A9 of dynvarmip)
    utendvtem = vtem*(f2d - dudphi)

    # calculate E-P fluxes
    epfy = a*coslat2d*(dudp*psieddy - uvzm) # Eq A2
    epfz = a*coslat2d*((f2d-dudphi)*psieddy - uwzm) # Eq A3

    # calculate E-P flux divergence and zonal wind tendency
    # due to resolved waves (Eq A5)
    depfydphi = (1./(a*coslat2d)) \
              * np.gradient(epfy*coslat2d,
                            latrad,
                            axis=1)

    depfzdp = np.gradient(epfz,
                          pre,
                          axis=0)

    delf = (depfydphi + depfzdp)/(a*coslat2d)
    delf = xr.DataArray(delf, coords = ds.Uzm.coords, name='delf')

    # TEM stream function, Eq A8
    topvzm = np.zeros([1,nlat])
    vzmwithzero = np.concatenate((topvzm, vzm), axis=0)
    prewithzero = np.concatenate((np.zeros([1]), pre))
    intv = integrate.cumulative_trapezoid(vzmwithzero,prewithzero,axis=0)
    psitem = (2*np.pi*a*coslat2d/g0)*(intv - psieddy)

    # final scaling of E-P fluxes and divergence to transform to log-pressure
    epfy = epfy*pre/p0      # A13
    epfz = -1.*(H/p0)*epfz  # A14
    wtem = -1.*(H/pre)*wtem # A16

    #
    # add long name and unit attributes to TEM diagnostics
    uzm.attrs['long_name'] = 'Zonal-Mean zonal wind'
    uzm.attrs['units'] = 'm/s'

    vzm.attrs['long_name'] = 'Zonal-Mean meridional wind'
    vzm.attrs['units'] = 'm/s'

    thzm.attrs['long_name'] = 'Zonal-Mean potential temperature'
    thzm.attrs['units'] = 'K'

    epfy.attrs['long_name'] = 'northward component of E-P flux'
    epfy.attrs['units'] = 'm3/s2'

    epfz.attrs['long_name'] = 'upward component of E-P flux'
    epfz.attrs['units'] = 'm3/s2'

    vtem.attrs['long_name'] = 'Transformed Eulerian mean northward wind'
    vtem.attrs['units'] = 'm/s'

    wtem.attrs['long_name'] = 'Transformed Eulerian mean upward wind'
    wtem.attrs['units'] = 'm/s'

    psitem.attrs['long_name'] = 'Transformed Eulerian mean mass stream function'
    psitem.attrs['units'] = 'kg/s'

    delf.attrs['long_name'] = 'tendency of eastward wind due to Eliassen-Palm flux divergence'
    delf.attrs['units'] = 'm/s2'

    utendvtem.attrs['long_name'] = 'tendency of eastward wind due to TEM northward wind advection and the coriolis term'
    utendvtem.attrs['units'] = 'm/s2'

    utendwtem.attrs['long_name'] = 'tendency of eastward wind due to TEM upward wind advection'
    utendwtem.attrs['units'] = 'm/s2'

    epfy.values = np.float32(epfy.values)
    epfz.values = np.float32(epfz.values)
    wtem.values = np.float32(wtem.values)
    psitem.values = np.float32(psitem.values)
    delf.values = np.float32(delf.values)
    utendvtem.values = np.float32(utendvtem.values)
    utendwtem.values = np.float32(utendwtem.values)

    dstem = xr.Dataset(data_vars=dict(date = ds.date,
                                      datesec = ds.datesec,
                                      UZM = uzm,
                                      VZM = vzm,
                                      THZM = thzm,
                                      EPFY = epfy,
                                      EPFZ = epfz,
                                      VTEM = vtem,
                                      WTEM = wtem,
                                      PSITEM = psitem,
                                      DELF = delf,
                                      UTENDVTEM = utendvtem,
                                      UTENDWTEM = utendwtem
                                      ))

    #Carry the time bounds through under whichever name the history files use:
    for name in ('time_bnds', 'time_bounds'):
        if name in ds:
            dstem['time_bnds'] = ds[name]
            break
        #End if
    #End for

    return dstem
