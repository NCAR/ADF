import xarray as xr
import numpy as np
from scipy import integrate
from numpy import ma
from datetime import date
from pathlib import Path
from glob import glob
from itertools import chain
#import metpy.calc.thermo as thermo
#from metpy.units import units


def create_TEM_files(adf):
    """
    Calculate the TEM variables and create new netCDF files

    """

    #Notify user that script has started:
    msg = "\n  Generating CAM TEM diagnostics files..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    #Special ADF variables
    #CAM simulation variables (these quantities are always lists):
    case_names    = adf.get_cam_info("cam_case_name", required=True)
    base_name     = adf.get_baseline_info("cam_case_name")

    #Grab h4 history files locations
    cam_hist_locs = adf.get_cam_info("cam_hist_loc", required=True)
    cam_hist_strs = adf.hist_string["test_hist_str"]

    #Extract test case years
    start_years   = adf.climo_yrs["syears"]
    end_years     = adf.climo_yrs["eyears"]

    res = adf.variable_defaults # will be dict of variable-specific plot preferences

    if "qbo" in adf.plotting_scripts:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM",
                    "PSITEM","UTENDEPFD","UTENDVTEM","UTENDWTEM"]
    else:
        var_list = ["UZM","THZM","EPFY","EPFZ","VTEM","WTEM","PSITEM","UTENDEPFD"]

    tem_locs = []
    
    #Grab TEM diagnostics options
    #----------------------------
    #Extract TEM file save locations
    tem_base_loc = adf.get_baseline_info("cam_tem_loc")
    tem_case_locs = adf.get_cam_info("cam_tem_loc")

    #If path not specified, skip TEM calculation?
    if tem_case_locs is None:
        print("\t 'cam_tem_loc' not found in 'diag_cam_climo', so no TEM files/diagnostics will be generated.")
        pass
    else:
        for tem_case_loc in tem_case_locs:
            tem_case_loc = Path(tem_case_loc)
            #Check if TEM directory exists, and if not, then create it:
            if not tem_case_loc.is_dir():
                print(f"    {tem_case_loc} not found, making new directory")
                tem_case_loc.mkdir(parents=True)
            #End if
            tem_locs.append(tem_case_loc)
        #End for

    #Set default to h4
    hist_nums = adf.get_cam_info("tem_hist_str")[0]
    if hist_nums is None:
        hist_nums = ["h4a"]*len(case_names)

    #Get test case(s) tem over-write boolean and force to list if not by default
    overwrite_tem_cases = adf.get_cam_info("overwrite_tem")

    #If overwrite argument is missing, then default to False:
    if overwrite_tem_cases is None:
        overwrite_tem_cases = [False]*len(case_names)

    #If overwrite argument is a scalar,
    #then convert it into a list:
    if not isinstance(overwrite_tem_cases, list): overwrite_tem_cases = [overwrite_tem_cases]

    #Check if comparing to observations
    if adf.get_basic_info("compare_obs"):
        var_obs_dict = adf.var_obs_dict

        #If dictionary is empty, then there are no observations, so quit here:
        if not var_obs_dict:
            print("No observations found to plot against, so no obs-based TEM plot will be generated.")
            pass
        
        print(f"\t Processing TEM for observations :")

        base_name = "Obs"

        #Save Obs TEM file to first test case location
        output_loc_idx = tem_locs[0]

        #Check if re-gridded directory exists, and if not, then create it:
        if not output_loc_idx.is_dir():
            print(f"    {output_loc_idx} not found, making new directory")
            output_loc_idx.mkdir(parents=True)
        #End if

        print(f"\t NOTE: Observation TEM file being saved to '{output_loc_idx}'")

        #Set baseline file name as full path
        tem_fil = output_loc_idx / f'{base_name}.TEMdiag.nc'

        #Group all TEM observation files together
        tem_obs_fils = []
        for var in var_list:
            if var in res:
                #Gather from variable defaults file
                obs_file_path = Path(res[var]["obs_file"])
                if not obs_file_path.is_file():
                    obs_data_loc = adf.get_basic_info("obs_data_loc")
                    obs_file_path = Path(obs_data_loc)/obs_file_path
                #It's likely multiple TEM vars will come from one file, so check
                #to see if it already exists from other vars.
                if obs_file_path not in tem_obs_fils:
                    tem_obs_fils.append(obs_file_path)

        #ds = xr.open_mfdataset(tem_obs_fils,combine="nested")
        ds = xr.open_mfdataset(tem_obs_fils,decode_times=True, combine='by_coords')
        if "zalat" in ds.dims:
            zm_name = "zalat"
        if "zmlat" in ds.dims:
            zm_name = "zmlat"
        start_year = str(ds.time[0].values)[0:4]
        end_year = str(ds.time[-1].values)[0:4]

        #Update the attributes
        ds.attrs['created'] = str(date.today())
        ds['lev']=ds['level']
        ds[zm_name]=ds['lat']

        #Make a copy of obs data so we don't do anything bad
        ds_obs = ds.copy()
        ds_base = xr.Dataset({'uzm': xr.Variable(('time', 'lev', zm_name), ds_obs.uzm.data),
                                'epfy': xr.Variable(('time', 'lev', zm_name), ds_obs.epfy.data),
                                'epfz': xr.Variable(('time', 'lev', zm_name), ds_obs.epfz.data),
                                'vtem': xr.Variable(('time', 'lev', zm_name), ds_obs.vtem.data),
                                'wtem': xr.Variable(('time', 'lev', zm_name), ds_obs.wtem.data),
                                'psitem': xr.Variable(('time', 'lev', zm_name), ds_obs.psitem.data),
                                'utendepfd': xr.Variable(('time', 'lev', zm_name), ds_obs.utendepfd.data),
                                'utendvtem': xr.Variable(('time', 'lev', zm_name), ds_obs.utendvtem.data),
                                'utendwtem': xr.Variable(('time', 'lev', zm_name), ds_obs.utendwtem.data),
                                'lev': xr.Variable('lev', ds_obs.level.values),
                                'zmlat': xr.Variable(zm_name, ds_obs.lat.values),
                                'time': xr.Variable('time', ds_obs.time.values)
                    })

        # write output to a netcdf file
        ds_base.to_netcdf(tem_fil, unlimited_dims='time', mode='w')

    else:
        if tem_base_loc:
            cam_hist_locs.append(adf.get_baseline_info("cam_hist_loc", required=True))
            cam_hist_strs.append(adf.hist_string["base_hist_str"])

            #Set default to h4
            hist_num = adf.get_baseline_info("tem_hist_str")
            if hist_num is None:
                hist_num = "h4a"

            #Extract baseline years (which may be empty strings if using Obs):
            syear_baseline = adf.climo_yrs["syear_baseline"]
            eyear_baseline = adf.climo_yrs["eyear_baseline"]

            case_names.append(base_name)
            start_years.append(syear_baseline)
            end_years.append(eyear_baseline)
           
            tem_base_loc = Path(tem_base_loc)
            #Check if TEM directory exists, and if not, then create it:
            if not tem_base_loc.is_dir():
                print(f"    {tem_base_loc} not found, making new directory")
                tem_base_loc.mkdir(parents=True)
            #End if

            tem_locs.append(tem_base_loc)
            overwrite_tem_cases.append(adf.get_baseline_info("overwrite_tem", False))

            hist_nums.append(hist_num)
        else:
            print("\t 'cam_tem_loc' not found in 'diag_cam_baseline_climo', so no baseline files/diagnostics will be generated.")

    #End if (check for obs)

    #Loop over cases:
    for case_idx, case_name in enumerate(case_names):

        print(f"\t Processing TEM for case '{case_name}' :")

        #Extract start and end year values:
        start_year = start_years[case_idx]
        end_year   = end_years[case_idx]

        #Create path object for the CAM history file(s) location:
        starting_location = Path(cam_hist_locs[case_idx])

        #Check that path actually exists:
        if not starting_location.is_dir():
            emsg = f"Provided 'cam_hist_loc' directory '{starting_location}' not found."
            emsg += " Script is ending here."
            return
        #End if

        #Check if history files actually exist. If not then kill script:
        
        hist_str = f"{hist_nums[case_idx]}"
        if not list(starting_location.glob("*"+hist_str+'.*.nc')):
            emsg = f"No CAM history {hist_str} files found in '{starting_location}'."
            emsg += " Script is ending here."
            adf.end_diag_fail(emsg)
        #End if

        hist0_str = cam_hist_strs[case_idx]

        #Get full path and file for file name
        output_loc_idx = tem_locs[case_idx]

        #Check if re-gridded directory exists, and if not, then create it:
        if not output_loc_idx.is_dir():
            print(f"    {output_loc_idx} not found, making new directory")
            output_loc_idx.mkdir(parents=True)
        #End if

        #Set case file name
        tem_fil = output_loc_idx / f'{case_name}.TEMdiag_{start_year}-{end_year}.nc'

        #Get current case tem over-write boolean
        overwrite_tem = overwrite_tem_cases[case_idx]

        #If files exist, then check if over-writing is allowed:
        if (tem_fil.is_file()) and (not overwrite_tem):
            print(f"\t    INFO: Found TEM file and clobber is False, so moving to next case.")
            pass
        else:
            if tem_fil.is_file():
                print(f"\t    INFO: Found TEM file but clobber is True, so over-writing file.")

            #Glob each set of years
            #NOTE: This will make a nested list
            hist_files = []
            hist0_files = []
            for yr in np.arange(int(start_year),int(end_year)+1):
                #Grab all leading zeros for climo year just in case
                yr = f"{str(yr).zfill(4)}"
                hist_files.append(glob(f"{starting_location}/*{hist_str}.{yr}*.nc"))
                hist0_files.append(glob(f"{starting_location}/*{hist0_str[0]}.{yr}*.nc"))

            #Flatten list of lists to 1d list
            hist_files = sorted(list(chain.from_iterable(hist_files)))
            ds = xr.open_mfdataset(hist_files,decode_times=True, combine='by_coords')
            
            hist0_files = sorted(list(chain.from_iterable(hist0_files)))
            ds_h0 = xr.open_mfdataset(hist0_files,decode_times=True, combine='by_coords')
            if "zalat" in ds_h0.dims:
                zm_name = "zalat"
            if "zmlat" in ds_h0.dims:
                zm_name = "zmlat"
            #ds_h0 = ds_h0.rename({'lat': zm_name})

            #Average time dimension over time bounds, if bounds exist:
            if 'time_bnds' in ds_h0:
                time_bounds_name = 'time_bnds'
            elif 'time_bounds' in ds_h0:
                time_bounds_name = 'time_bounds'
            else:
                time_bounds_name = None

            if time_bounds_name:
                time = ds_h0['time']
                #NOTE: force `load` here b/c if dask & time is cftime,
                #throws a NotImplementedError:

                time = xr.DataArray(ds_h0[time_bounds_name].load().mean(dim='nbnd').values,
                                    dims=time.dims, attrs=time.attrs)
                ds_h0['time'] = time
                ds_h0.assign_coords(time=time)
                ds_h0 = xr.decode_cf(ds_h0)

            #iterate over the times in a dataset
            for idx,_ in enumerate(ds.time.values):
                if idx == 0:
                    dstem0 = calc_tem(ds.squeeze().isel(time=idx))
                else:
                    dstem = calc_tem(ds.squeeze().isel(time=idx))
                    if "zalat" in dstem.dims:
                        zm_name = "zalat"
                    if "zmlat" in dstem.dims:
                        zm_name = "zmlat"
                    dstem0 = xr.concat([dstem0, dstem],'time')
                #End if
            #End if

            #Average time dimension over time bounds, if bounds exist:
            if 'time_bnds' in ds:
                time_bounds_name = 'time_bnds'
            elif 'time_bounds' in ds:
                time_bounds_name = 'time_bounds'
            else:
                time_bounds_name = None

            if time_bounds_name:
                time = ds['time']
                #NOTE: force `load` here b/c if dask & time is cftime,
                #throws a NotImplementedError:

                time = xr.DataArray(ds[time_bounds_name].load().mean(dim='nbnd').values,
                                    dims=time.dims, attrs=time.attrs)
                dstem0['time'] = time
                dstem0.assign_coords(time=time)
                dstem0 = xr.decode_cf(dstem0)

            # Step 1: Your standard latitudes
            za_lats = dstem[zm_name].values

            # Step 2: Interpolate ds2 to standard latitudes
            # List of possible coordinate names
            possible_lat_names = ['zalat', 'zmlat']

            # Find the one that exists in the dataset
            for name in possible_lat_names:
                if name in ds_h0.coords:
                    lat_coord_name = name
                    break
            else:
                raise ValueError("No known latitude coordinate found in dataset.")

            # Interpolate using dynamic coordinate name
            ds_h0_lats = ds_h0.interp({lat_coord_name: za_lats})

            #ds_h0_lats = ds_h0.interp(zalat=za_lats)

            zonal_mean_PS = ds_h0_lats['PS'].mean(dim='lon').compute()
            zonal_mean_PMID = ds_h0_lats['PMID'].mean(dim='lon').compute()

            #Update the attributes
            dstem0.attrs = ds.attrs
            dstem0.attrs['created'] = str(date.today())
            dstem0['lev']=ds['lev']
            dstem0['PS'] = zonal_mean_PS
            dstem0['PMID'] = zonal_mean_PMID

            dstem0.to_netcdf(tem_fil, unlimited_dims='time', mode='w')

        #End if (file creation or over-write file)
    #Notify user that script has ended:
    print("  ...TEM variables have been calculated successfully.")




def calc_tem(ds):
    """
    # calc_tem() function to calculate TEM diagnostics on CAM/WACCM output
    # This assumes the data have already been organized into zonal mean fluxes
    # Uzm, THzm, VTHzm, Vzm, UVzm, UWzm, Wzm
    # note that calculations are performed on model interface levels, which is ok
    # in the stratosphere but not in the troposphere.  If interested in tropospheric
    # TEM diagnostics, make sure input fields have been interpolated to true pressure levels.

    # The code follows the 'TEM recipe' from Appendix A of Gerber, E. P. and Manzini, E.:
    # The Dynamics and Variability Model Intercomparison Project (DynVarMIP) for CMIP6:
    # assessing the stratosphere–troposphere system, Geosci. Model Dev., 9, 3413–3425,
    # https://doi.org/10.5194/gmd-9-3413-2016, 2016 and the corrigendum.

    # pdf available here: https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016.pdf,
    # https://gmd.copernicus.org/articles/9/3413/2016/gmd-9-3413-2016-corrigendum.pdf

    # Output from post-processing function

    # Table A1. Momentum budget variable list (2-D monthly / daily zonal means, YZT).

    # Name      Long name [unit]

    # epfy      northward component of the Eliassen–Palm flux [m3 s−2]
    # epfz      upward component of the Eliassen–Palm flux [m3 s−2]
    # vtem      Transformed Eulerian mean northward wind [m s−1]
    # wtem      Transformed Eulerian mean upward wind [m s−1]
    # psitem    Transformed Eulerian mean mass stream function [kg s−1]
    # utendepfd tendency of eastward wind due to Eliassen–Palm flux divergence [m s−2]
    # utendvtem tendency of eastward wind due to TEM northward wind advection and the Coriolis term [m s−2]
    # utendwtem tendency of eastward wind due to TEM upward wind advection [m s−2]

    # this utility based on python code developed by Isla Simpson 25 Feb 2021
    # initial coding of stand alone function by Dan Marsh 16 Dec 2022

    # NOTE: function expects an xarray dataset with dataarrays of dimension (nlev,nlat)
    # to process more than one timestep iterate over time.
    """

    # constants for TEM calculations
    p0 = 101325.
    a = 6.371e6
    om = 7.29212e-5
    H = 7000.
    g0 = 9.80665
    if "zalat" in ds.dims:
        zm_name = "zalat"
    if "zmlat" in ds.dims:
        zm_name = "zmlat"
    nlat = ds[zm_name].size
    nlev = ds['lev'].size

    latrad = np.radians(ds[zm_name])
    coslat = np.cos(latrad)
    coslat2d = np.tile(coslat,(nlev,1))

    pre = ds['lev']*100. # pressure levels in Pascals
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
    intv = integrate.cumtrapz(vzmwithzero,prewithzero,axis=0)
    psitem = (2*np.pi*a*coslat2d/g0)*(intv - psieddy)

    # final scaling of E-P fluxes and divergence to transform to log-pressure
    epfy = epfy*pre/p0      # A13
    epfz = -1.*(H/p0)*epfz  # A14
    wtem = -1.*(H/pre)*wtem # A16

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

    #Average time dimension over time bounds, if bounds exist:
    if 'time_bnds' in ds:
        time_bounds_name = 'time_bnds'
    elif 'time_bounds' in ds:
        time_bounds_name = 'time_bounds'

    dstem = xr.Dataset(data_vars=dict(date = ds.date,
                                      datesec = ds.datesec,
                                      time_bnds = time_bounds_name,
                                      hybm=ds.hybm,
                                      hyam=ds.hyam,
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

    return dstem