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


def _per_case(value, default, ncases):
    """
    Return a config entry as one value per case.

    Parameters
    ----------
    value : list, scalar or None
        The value as it came out of the config file.
    default : any
        Used for every case when `value` is None.
    ncases : int
        Number of cases.

    Returns
    -------
    list
        `value` unchanged if it was already a list, otherwise `ncases` copies
        of `value` or of `default`.
    """
    if value is None:
        return [default] * ncases
    return value if isinstance(value, list) else [value] * ncases


def _first_stream(hist_str, default="h4"):
    """
    Return the first history stream named by a 'hist_str' config entry.

    Parameters
    ----------
    hist_str : str, list or None
        A history stream entry. The ADF normalizes the test case entries to a
        nested list ([ncases][nstreams]); a baseline entry arrives as whatever
        the config file held, so both are unwrapped.
    default : str, optional
        Returned when `hist_str` is empty or None.

    Returns
    -------
    str
        The stream name, for example 'h4' or 'cam.h4a'.
    """
    while isinstance(hist_str, list) and hist_str:
        hist_str = hist_str[0]
    return hist_str if hist_str else default


def _ensure_dir(path):
    """
    Return `path` as a Path, creating the directory if it does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory to create.

    Returns
    -------
    pathlib.Path
        The same directory, which now exists.
    """
    path = Path(path)
    if not path.is_dir():
        print(f"    {path} not found, making new directory")
        path.mkdir(parents=True)
    #End if
    return path


def _write_obs_tem_file(adf, var_list, res, output_loc):
    """
    Write 'Obs.TEMdiag.nc' from pre-computed observational TEM diagnostics.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object, used for the 'obs_data_loc' setting.
    var_list : list of str
        TEM variables being diagnosed, used to find the observation files.
    res : dict
        The variable defaults, which name each variable's 'obs_file'.
    output_loc : pathlib.Path
        Directory to write the file into.

    Returns
    -------
    None
        Does not return a value, writes a file.

    Notes
    -----
    Nothing is calculated here. The observational file already holds TEM
    diagnostics; only the variable and coordinate names differ from what the
    rest of the ADF expects. Variables with no observational counterpart are
    left out, and any that are missing are reported.
    """
    print("\t Processing TEM for observations :")

    #Not every TEM variable has an observational counterpart -- the ERA5 TEM
    #file has no potential temperature, for one -- so only ask for the ones
    #that declare a file.
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

    ds_obs = xr.open_mfdataset(tem_obs_fils, combine="nested")
    missing = [v for v in OBS_TEM_VARS if v not in ds_obs]
    if missing:
        print(f"\t    WARNING: observation TEM data has no {', '.join(missing)}.")
    #End if

    #The observation file uses the lower-case CAM names; the TEM files the ADF
    #writes, and the plotting script, use the upper-case ones.
    ds_base = xr.Dataset(
        {name.upper(): xr.Variable(('time', 'lev', 'zalat'), ds_obs[name].data)
         for name in OBS_TEM_VARS if name in ds_obs},
        coords={'lev': ds_obs.level.values,
                'zalat': ds_obs.lat.values,
                'time': ds_obs.time.values},
        attrs={**ds_obs.attrs, 'created': str(date.today())})

    print(f"\t NOTE: Observation TEM file being saved to '{output_loc}'")
    ds_base.to_netcdf(output_loc / 'Obs.TEMdiag.nc', unlimited_dims='time', mode='w')


def create_TEM_files(adf):
    """
    Calculate the TEM diagnostics and write one netCDF file per case.

    Parameters
    ----------
    adf : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    None
        Does not return a value, writes files.

    Notes
    -----
    Directly uses adf for the following:
    get_cam_info, get_baseline_info, get_basic_info, climo_yrs,
    variable_defaults, plotting_scripts, var_obs_dict

    Reads the zonal mean fields listed in TEM_INPUT_VARS from the history
    stream named by 'tem_hist_str' (default 'h4') and writes
    '<case>.TEMdiag_<start>-<end>.nc' into 'cam_tem_loc'. An existing file is
    left alone unless 'overwrite_tem' is set.

    A baseline simulation is appended to the list of cases and processed the
    same way. Observations are different: the observational file already holds
    TEM diagnostics, so it only needs renaming, which _write_obs_tem_file does.

    A case that cannot be processed is reported and skipped, and the closing
    message names any that were, so a partial run is not reported as a
    complete one.
    """

    #Notify user that script has started:
    msg = "\n  Generating CAM TEM diagnostics files..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Special ADF variables
    #CAM simulation variables (these quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    base_name     = adf.get_baseline_info("cam_case_name")

    #Where the CAM history files live
    cam_hist_locs = adf.get_cam_info("cam_hist_loc", required=True)

    #Extract test case years
    start_years   = adf.climo_yrs["syears"]
    end_years     = adf.climo_yrs["eyears"]

    res = adf.variable_defaults # will be dict of variable-specific plot preferences

    if "qbo" in adf.plotting_scripts:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM",
                    "PSITEM","UTENDEPFD","UTENDVTEM","UTENDWTEM"]
    else:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","UTENDEPFD"]

    #Grab TEM diagnostics options
    #----------------------------
    #Extract TEM file save locations
    tem_base_loc = adf.get_baseline_info("cam_tem_loc")
    tem_case_locs = adf.get_cam_info("cam_tem_loc")

    #Without an output location there is nowhere to put the files, and the
    #case loop below indexes tem_locs, so there is nothing to carry on with.
    if tem_case_locs is None:
        print("\t 'cam_tem_loc' not found in 'diag_cam_climo', so no TEM files/diagnostics will be generated.")
        return
    #End if
    tem_locs = [_ensure_dir(loc) for loc in tem_case_locs]

    #One TEM history stream and one clobber setting per case, defaulting to h4:
    ncases = len(case_names)
    hist_nums = [_first_stream(h)
                 for h in _per_case(adf.get_cam_info("tem_hist_str"), "h4", ncases)]
    overwrite_tem_cases = _per_case(adf.get_cam_info("overwrite_tem"), False, ncases)

    #Observations need no TEM calculation -- the observational file already
    #holds TEM diagnostics -- so they are written out directly and the loop
    #below still makes the test cases' files. A baseline simulation, by
    #contrast, is processed exactly like a test case, so it is appended to the
    #case lists rather than handled separately.
    if adf.get_basic_info("compare_obs"):
        _write_obs_tem_file(adf, var_list, res, _ensure_dir(tem_locs[0]))
    elif tem_base_loc:
        cam_hist_locs.append(adf.get_baseline_info("cam_hist_loc", required=True))
        case_names.append(base_name)
        start_years.append(adf.climo_yrs["syear_baseline"])
        end_years.append(adf.climo_yrs["eyear_baseline"])
        tem_locs.append(_ensure_dir(tem_base_loc))
        overwrite_tem_cases.append(adf.get_baseline_info("overwrite_tem", False))
        hist_nums.append(_first_stream(adf.get_baseline_info("tem_hist_str")))
    else:
        print("\t 'cam_tem_loc' not found in 'diag_cam_baseline_climo', so no baseline files/diagnostics will be generated.")

    #Loop over cases:
    skipped = []
    for case_idx, case_name in enumerate(case_names):

        print(f"\t Processing TEM for case '{case_name}' :")

        #Extract start and end year values:
        start_year = start_years[case_idx]
        end_year   = end_years[case_idx]

        #Create path object for the CAM history file(s) location:
        starting_location = Path(cam_hist_locs[case_idx])

        #A case the ADF cannot process is reported and skipped; the remaining
        #cases still get their TEM files.
        if not starting_location.is_dir():
            print(f"\t    WARNING: 'cam_hist_loc' directory '{starting_location}' "
                  f"not found, skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        hist_str = hist_nums[case_idx]
        if not list(starting_location.glob(f"*{hist_str}.*.nc")):
            print(f"\t    WARNING: no CAM history {hist_str} files in "
                  f"'{starting_location}', skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        output_loc_idx = _ensure_dir(tem_locs[case_idx])

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

        hist_files = sorted(chain.from_iterable(
            #leading zeros on the year just in case
            glob(f"{starting_location}/*{hist_str}.{yr:04d}*.nc")
            for yr in range(int(start_year), int(end_year) + 1)))

        ds = xr.open_mfdataset(hist_files)

        #Settle the vertical grid once, before the per-time loop, rather than
        #assuming the zonal-mean fields are on layer midpoints.
        ds, lev_name = harmonize_tem_levels(ds)
        if lev_name is None:
            print(f"\t    WARNING: skipping TEM for '{case_name}'.")
            skipped.append(case_name)
            continue
        #End if

        #calc_tem works one time step at a time:
        dstem0 = xr.concat([calc_tem(ds.squeeze().isel(time=idx), lev_name)
                            for idx in range(ds.sizes['time'])], 'time')

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

        # write output to a netcdf file
        dstem0.to_netcdf(tem_fil, unlimited_dims='time', mode='w')

    #Notify user that script has ended, naming anything that did not get made
    #so a partial run is not reported as a complete one:
    if skipped:
        print(f"  ...TEM variables calculated, except for: {', '.join(skipped)}.")
    else:
        print("  ...TEM variables have been calculated successfully.")
    #End if




#The zonal-mean fields calc_tem needs from the history stream:
TEM_INPUT_VARS = ("Uzm", "Vzm", "Wzm", "THzm", "UVzm", "UWzm", "VTHzm")

#The TEM fields carried in the observation file, under their CAM names. There is
#no THZM or VZM: the ERA5 TEM file has neither.
OBS_TEM_VARS = ("uzm", "epfy", "epfz", "vtem", "wtem", "psitem",
                "utendepfd", "utendvtem", "utendwtem")


def harmonize_tem_levels(ds):
    """
    Put the TEM input fields on a single vertical grid, and report which one.

    Parameters
    ----------
    ds : xarray.Dataset
        History data holding the fields listed in TEM_INPUT_VARS.

    Returns
    -------
    xarray.Dataset
        The dataset with those fields on one vertical grid.
    str or None
        Name of that vertical dimension, or None when the fields cannot be put
        on a common grid. The caller reports that and skips the case.

    Notes
    -----
    CAM may write the zonal mean fields on layer midpoints ('lev') or on layer
    interfaces ('ilev'), and a history stream can carry a mix of the two. When
    they all agree the grid is left alone, so an interface-only stream stays on
    interfaces and goes with PINT. When they disagree the midpoint grid is used
    as the common one and the interface fields are interpolated onto it, so the
    result goes with PMID. See adf_utils.pressure_field_name.
    """
    present = {v: utils.vertical_dim(ds[v]) for v in TEM_INPUT_VARS if v in ds}
    found = {dim for dim in present.values() if dim is not None}

    if not found:
        print("\t    WARNING: none of the TEM input fields have a 'lev' or "
              "'ilev' dimension.")
        return ds, None
    if len(found) == 1:
        return ds, found.pop()
    #End if

    #Mixed grids: interpolate the interface fields onto the midpoints. A mix
    #means something is already on 'lev', so the dimension is there, but the
    #file need not carry the matching coordinate values to interpolate onto.
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


def calc_tem(ds, lev_name="lev"):
    """
    Calculate TEM diagnostics for one time step of CAM/WACCM output.

    Parameters
    ----------
    ds : xarray.Dataset
        One time step, holding the zonal mean fields listed in TEM_INPUT_VARS
        (Uzm, Vzm, Wzm, THzm, UVzm, UWzm, VTHzm) with dimensions
        (`lev_name`, zalat), plus `date` and `datesec`.
    lev_name : str, optional
        Name of the vertical dimension the input fields are on, 'lev' for layer
        midpoints or 'ilev' for interfaces. Use harmonize_tem_levels to settle
        this, as CAM writes the zonal mean stream either way. The values of
        that coordinate are taken to be pressure in hPa.

    Returns
    -------
    xarray.Dataset
        UZM, VZM, THZM, EPFY, EPFZ, VTEM, WTEM, PSITEM, UTENDEPFD, UTENDVTEM
        and UTENDWTEM, on the same vertical grid as the input, along with
        `date`, `datesec` and the time bounds if the input carried them.

    Notes
    -----
    Iterate over time to process more than one time step.

    On interfaces the calculation is valid in the stratosphere but not in the
    troposphere. For tropospheric TEM diagnostics, interpolate the input fields
    to true pressure levels first.

    Follows the TEM recipe in Appendix A of Gerber, E. P. and Manzini, E.: The
    Dynamics and Variability Model Intercomparison Project (DynVarMIP) for
    CMIP6: assessing the stratosphere-troposphere system, Geosci. Model Dev.,
    9, 3413-3425, https://doi.org/10.5194/gmd-9-3413-2016, 2016, and its
    corrigendum:
    https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016.pdf
    https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016-corrigendum.pdf

    Table A1 of that appendix gives the momentum budget variables produced here
    (2-D monthly or daily zonal means):

    ==========  ================================================================
    Name        Long name [unit]
    ==========  ================================================================
    epfy        northward component of the Eliassen-Palm flux [m3 s-2]
    epfz        upward component of the Eliassen-Palm flux [m3 s-2]
    vtem        Transformed Eulerian mean northward wind [m s-1]
    wtem        Transformed Eulerian mean upward wind [m s-1]
    psitem      Transformed Eulerian mean mass stream function [kg s-1]
    utendepfd   tendency of eastward wind due to Eliassen-Palm flux divergence
                [m s-2]
    utendvtem   tendency of eastward wind due to TEM northward wind advection
                and the Coriolis term [m s-2]
    utendwtem   tendency of eastward wind due to TEM upward wind advection
                [m s-2]
    ==========  ================================================================

    Based on python code developed by Isla Simpson (25 Feb 2021); initial
    coding of the stand alone function by Dan Marsh (16 Dec 2022).
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

    utendepfd = (depfydphi + depfzdp)/(a*coslat2d)
    utendepfd = xr.DataArray(utendepfd, coords = ds.Uzm.coords, name='utendepfd')

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

    utendepfd.attrs['long_name'] = 'tendency of eastward wind due to Eliassen-Palm flux divergence'
    utendepfd.attrs['units'] = 'm/s2'

    utendvtem.attrs['long_name'] = 'tendency of eastward wind due to TEM northward wind advection and the coriolis term'
    utendvtem.attrs['units'] = 'm/s2'

    utendwtem.attrs['long_name'] = 'tendency of eastward wind due to TEM upward wind advection'
    utendwtem.attrs['units'] = 'm/s2'

    epfy.values = np.float32(epfy.values)
    epfz.values = np.float32(epfz.values)
    wtem.values = np.float32(wtem.values)
    psitem.values = np.float32(psitem.values)
    utendepfd.values = np.float32(utendepfd.values)
    utendvtem.values = np.float32(utendvtem.values)
    utendwtem.values = np.float32(utendwtem.values)

    #Carry the time bounds through under whichever name the history files use:
    time_bounds_name = None
    for name in ('time_bnds', 'time_bounds'):
        if name in ds:
            time_bounds_name = name
            break

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
                                      UTENDEPFD = utendepfd,
                                      UTENDVTEM = utendvtem,
                                      UTENDWTEM = utendwtem
                                      ))

    if time_bounds_name is not None:
        dstem['time_bnds'] = ds[time_bounds_name]

    return dstem
