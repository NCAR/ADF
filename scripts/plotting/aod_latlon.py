"""Module for AOD-specific plotting functionality"""

from pathlib import Path
import numpy as np
import xarray as xr
import xesmf as xe


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

import plotting_functions as pf

from dataclasses import dataclass, field

@dataclass
class AODPlotConfig:
    """Configuration for AOD plots."""
    seasons: list = ('DJF', 'MAM', 'JJA', 'SON')
    season_names: dict = field(default_factory=lambda: {
        'DJF': 'Dec-Jan-Feb',
        'MAM': 'Mar-Apr-May',
        'JJA': 'Jun-Jul-Aug',
        'SON': 'Sep-Oct-Nov'
    })
    obs_sources: list = ('TERRA MODIS', 'MERRA2')
    var_name: str = 'AODVISdn'


def aod_latlon(adfobj):
    """Generate AOD comparison plots."""
    config = AODPlotConfig()
    
    # Load observations
    obs_data = load_observations(adfobj)
    if not obs_data:
        return
        
    # Process model data
    model_data = process_model_cases(adfobj, config.var_name, obs_data)
    if not model_data:
        return
        
    # Generate plots
    for obs_source, obs_dataset in obs_data.items():
        for season in config.seasons:
            create_aod_panel(adfobj, model_data, obs_dataset, 
                           season, obs_source)
            

def load_observations(adfobj):
    """Load MERRA2 and MODIS observation datasets.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object containing configuration
        
    Returns
    -------
    dict
        Dictionary of observation datasets keyed by source name
    """
    obs_dir = adfobj.get_basic_info("obs_data_loc")
    obs_files = {
        'TERRA MODIS': 'MOD08_M3_192x288_AOD_2001-2020_climo.nc',
        'MERRA2': 'MERRA2_192x288_AOD_2001-2020_climo.nc'
    }
    
    obs_data = {}
    for source, filename in obs_files.items():
        ds = load_obs_data(obs_dir, filename)
        if ds is None:
            print(f"\t  WARNING: AOD Panel plots not made, missing {source} file")
            return None
            
        # Extract correct variable based on source
        if source == 'MERRA2':
            ds = ds['TOTEXTTAU']
        else:  # MODIS
            ds = ds['AOD_550_Dark_Target_Deep_Blue_Combined_Mean_Mean']
            
        # Calculate seasonal means
        ds_seasonal = monthly_to_seasonal(ds, obs=True)
        obs_data[source] = ds_seasonal
        
    return obs_data


def load_obs_data(obs_dir, file_name):
    """Load and prepare observational dataset."""
    file_path = Path(obs_dir) / file_name
    if not file_path.is_file():
        return None
        
    ds = xr.open_dataset(file_path)
    # Round coordinates for consistency
    ds['lon'] = ds['lon'].round(5)
    ds['lat'] = ds['lat'].round(5)
    return ds


def process_model_cases(adfobj, var, obs_data):
    """Process model cases and regrid if necessary.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object containing configuration
    var : str
        Variable name to process
    obs_data : dict
        Dictionary of observation datasets with their grids
        
    Returns
    -------
    list
        List of processed model datasets, one per case
    """
    # Get case information
    cases = adfobj.get_cam_info('cam_case_name', required=True)
    if not adfobj.compare_obs:
        cases = cases + [adfobj.data.ref_case_label] # ref case added to cases

    # Get reference grid from first observation dataset
    ref_obs = next(iter(obs_data.values()))
    
    # Process each case
    processed_data = []
    for case_name in cases:
        print(f"[process_model_cases] {case_name = }")
        # Load and process model data
        case_data = process_model_data(adfobj, case_name, var, ref_obs)
        if case_data is not None:
            processed_data.append((case_data, case_name))
            
    return processed_data if processed_data else None


def process_model_data(adfobj, case_name, var, obs_shape):
    """Process model data and check grid compatibility."""
    if case_name == adfobj.data.ref_case_label:
        ds_case = adfobj.data.load_reference_climo_da(case_name, var)
    else:
        ds_case = adfobj.data.load_climo_da(case_name, var)
    if ds_case is None:
        print(f"\t WARNING: No climo file for {case_name} variable {var}")
        return None
        
    ds_case['lon'] = ds_case['lon'].round(5)
    ds_case['lat'] = ds_case['lat'].round(5)
    
    # Check grid compatibility
    needs_regrid = check_grid_compatibility(ds_case, obs_shape)
    if needs_regrid:
        ds_case = regrid_to_obs(ds_case, obs_shape)
        
    return monthly_to_seasonal(ds_case)


def check_grid_compatibility(model_arr, obs_arr):
    """Check if model grid matches observation grid.
    
    Parameters
    ----------
    model_arr : xarray.DataArray
        Model data array with lat/lon coordinates
    obs_arr : xarray.DataArray
        Observation data array with lat/lon coordinates
        
    Returns
    -------
    bool
        True if grids don't match and regridding is needed
    """
    test_lons = model_arr.lon
    test_lats = model_arr.lat
    obs_lons = obs_arr.lon
    obs_lats = obs_arr.lat

    # Check if shapes match first
    if obs_lons.shape != test_lons.shape:
        return True
        
    # Check exact coordinate matches
    try:
        xr.testing.assert_equal(test_lons, obs_lons)
        xr.testing.assert_equal(test_lats, obs_lats)
        return False
    except AssertionError:
        return True
    
def create_aod_panel(adfobj, data_sets, obs_dataset, season, obs_name):
    """Create AOD panel plot with differences and percent differences."""
    plot_data = []
    plot_titles = []
    plot_params = []
    case_names = []
    types = []
    
    # Get plot parameters from configuration
    plot_config = get_plot_params(adfobj)
    
    for case_data, case_name in data_sets:
        # Calculate differences
        diff = calculate_differences(case_data, obs_dataset, season)
        plot_data.append(diff)
        plot_titles.append(make_plot_config(diff, case_name, obs_name, season, "Diff"))
        plot_params.append(plot_config['default'])
        case_names.append(case_name)
        types.append("Diff")
        
        # Calculate percent differences
        pdiff = calculate_percent_diff(case_data, obs_dataset, season)
        plot_data.append(pdiff)
        plot_titles.append(make_plot_config(pdiff, case_name, obs_name, season, "Percent Diff"))
        plot_params.append(plot_config['relerr'])
        case_names.append(case_name)
        types.append("Percent Diff")
        
    return aod_panel_latlon(adfobj, plot_titles, plot_params, plot_data, 
                           season, obs_name, case_names, len(data_sets), 
                           types, symmetric=True)


def validate_obs_data(merra_data, modis_data):
    """Validate observation datasets."""
    if merra_data is None or modis_data is None:
        raise ValueError("Missing observation data")
    
    if not np.array_equal(merra_data.lat, modis_data.lat):
        raise ValueError("Observation grids do not match")


def regrid_to_obs(model_arr, obs_arr):
    """Regrid model data to match observation grid using bilinear interpolation.
    
    Parameters
    ----------
    model_arr : xarray.DataArray
        Model data array to be regridded
    obs_arr : xarray.DataArray
        Observation data array with target grid
        
    Returns
    -------
    xarray.DataArray
        Regridded model data, or None if grids already match
    """
    # Create target grid specification
    ds_out = xr.Dataset({
        "lat": (["lat"], obs_arr.lat.values, {"units": "degrees_north"}),
        "lon": (["lon"], obs_arr.lon.values, {"units": "degrees_east"})
    })
    
    # Perform regridding
    regridder = xe.Regridder(model_arr, ds_out, "bilinear", periodic=True)
    model_regrid = regridder(model_arr, keep_attrs=True)
    
    return model_regrid

def calculate_differences(case_data, obs_data, season):
    """Calculate differences between case and observation data for a given season.
    
    Parameters
    ----------
    case_data : xarray.DataArray
        Model case data
    obs_data : xarray.DataArray
        Observation data
    season : str
        Season to calculate difference for
        
    Returns
    -------
    xarray.DataArray
        Difference between case and observation data
    """
    return case_data.sel(season=season) - obs_data.sel(season=season)


def calculate_percent_diff(case_data, obs_data, season):
    """Calculate percent difference between case and observation data.
    
    Parameters
    ----------
    case_data : xarray.DataArray
        Model case data
    obs_data : xarray.DataArray
        Observation data 
    season : str
        Season to calculate difference for
        
    Returns
    -------
    xarray.DataArray
        Percent difference, clipped to [-100, 100]
    """
    diff = calculate_differences(case_data, obs_data, season)
    pdiff = 100 * diff / obs_data.sel(season=season)
    return np.clip(pdiff, -100, 100)


def make_plot_config(data, case_name, obs_name, season, plot_type):
    """Create plot configuration dictionary.
    
    Parameters
    ----------
    data : xarray.DataArray
        Data to plot
    case_name : str
        Name of case being plotted
    obs_name : str  
        Name of observation dataset
    season : str
        Season being plotted
    plot_type : str
        Type of plot ('Diff' or 'Percent Diff')
        
    Returns
    -------
    dict
        Plot configuration including data and metadata
    """
    config = AODPlotConfig()
    return {
        'data': data,
        'title': f'{case_name} - {obs_name}\nAOD 550 nm - {config.season_names[season]}',
        'case_name': case_name,
        'plot_type': plot_type,
        'season': season
    }


def get_plot_params(adfobj):
    """Get AOD plot parameters from ADF configuration."""
    res = adfobj.variable_defaults
    res_aod_diags = res.get("aod_diags", {})
    return {
        'default': res_aod_diags.get("plot_params", {}),
        'relerr': res_aod_diags.get("plot_params_relerr", {})
    }

### refactored aod_panel_latlon:
def aod_panel_latlon(adfobj, plot_titles, plot_params, data, season, obs_name, case_names, case_num, types, symmetric=False):
    """Create AOD panel plot with model vs observation differences.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object containing configuration
    plot_titles : list
        List of titles for each panel
    plot_params : list
        List of plotting parameters for each panel
    data : list
        List of xarray DataArrays to plot
    season : str
        Current season being plotted
    obs_name : str
        Name of observation dataset
    case_names : list
        List of case names
    case_num : int
        Number of cases
    types : list
        List of plot types ('Diff' or 'Percent Diff')
    symmetric : bool, optional
        Whether to use symmetric colormap, by default False
    """
    # Get plot configuration
    file_type = adfobj.read_config_var("diag_basic_info").get('plot_type', 'png')
    plot_dir = adfobj.plot_location[0]
    plotfile = Path(plot_dir) / f'AOD_diff_{obs_name.replace(" ","_")}_{season}_LatLon_Mean.{file_type}'

    # Check if plot should be regenerated
    if plotfile.is_file() and not adfobj.get_basic_info('redo_plot'):
        adfobj.add_website_data(plotfile, f'AOD_diff_{obs_name.replace(" ","_")}', None,
                               season=season, multi_case=True, plot_type="LatLon", 
                               category="4-Panel AOD Diags")
        return

    # Create figure and axes
    fig = plt.figure(figsize=(7*case_num, 10))
    gs = mpl.gridspec.GridSpec(2*case_num, int(3*case_num), wspace=0.5, hspace=0.0)
    gs.tight_layout(fig)
    
    axs = []
    for i in range(case_num):
        start = i * 3
        end = (i + 1) * 3
        axs.append(plt.subplot(gs[0:case_num, start:end], projection=ccrs.PlateCarree()))
        axs.append(plt.subplot(gs[case_num:, start:end], projection=ccrs.PlateCarree()))

    # Generate each panel
    for i, field in enumerate(data):
        # Create individual plot
        ind_fig, ind_ax = plt.subplots(1, 1, figsize=((7*case_num)/2, 10/2),
                                      subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Prepare data
        field_values = field.values[:,:]
        lon_values = field.lon.values
        lat_values = field.lat.values
        field_values, lon_values = add_cyclic_point(field_values, coord=lon_values)
        lon_mesh, lat_mesh = np.meshgrid(lon_values, lat_values)
        field_mean = np.nanmean(field_values)

        # Set plot parameters
        plot_param = plot_params[i]
        levels = np.linspace(plot_param['range_min'], plot_param['range_max'],
                           plot_param['nlevel'], endpoint=True)
        if 'augment_levels' in plot_param:
            levels = sorted(np.append(levels, np.array(plot_param['augment_levels'])))

        plot_config = plot_titles[i]
        title = f"{plot_config['title']} Mean {field_mean:.2g}"

        # Create plots
        cmap_option = (plot_param.get('colormap', plt.cm.bwr) if symmetric 
                      else plot_param.get('colormap', plt.cm.turbo))
        extend_option = 'both' if symmetric else 'max'
        
        for ax, is_panel in [(axs[i], True), (ind_ax, False)]:
            print(f"DEBUGGING: {type(ax) = }, {is_panel = } //  {type(lon_mesh) = }, {lon_mesh.shape = } // {type(lat_mesh) = }, {lat_mesh.shape = } // {field_values.shape = }")
            img = ax.contourf(lon_mesh, lat_mesh, field_values,
                            levels, cmap=cmap_option, extend=extend_option,
                            transform=ccrs.PlateCarree())
            ax.set_facecolor('gray')
            ax.coastlines()
            ax.set_title(title, fontsize=10)

            cbar = plt.colorbar(img, orientation='horizontal', pad=0.05)
            if 'ticks' in plot_param:
                cbar.set_ticks(plot_param['ticks'])
            if 'tick_labels' in plot_param:
                cbar.ax.set_xticklabels(plot_param['tick_labels'])
            cbar.ax.tick_params(labelsize=6)

        # Save individual plot
        pbase = f'AOD_{case_names[i]}_vs_{obs_name.replace(" ","_")}_{types[i].replace(" ","_")}'
        ind_plotfile = Path(plot_dir) / f'{pbase}_{season}_LatLon_Mean.{file_type}'
        ind_fig.savefig(ind_plotfile, bbox_inches='tight', dpi=300)
        plt.close(ind_fig)

    # Save panel plot
    fig.savefig(plotfile, bbox_inches='tight', dpi=300)
    adfobj.add_website_data(plotfile, f'AOD_diff_{obs_name.replace(" ","_")}', None,
                           season=season, multi_case=True, plot_type="LatLon", 
                           category="4-Panel AOD Diags")
    plt.close(fig)


def monthly_to_seasonal(ds, obs=False):
    """Convert monthly data to seasonal means.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input data with monthly time dimension
    obs : bool, optional
        Whether input is observation data, by default False
        
    Returns
    -------
    xarray.DataArray
        Data array with new season dimension
    """
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    dataarrays = []
    
    if obs and isinstance(ds, xr.Dataset):
        # Handle observation dataset with multiple variables
        for varname in ds.data_vars:
            if '_n' not in varname:  # Skip count variables
                var_data = ds[varname]
                for s in seasons:
                    dataarrays.append(pf.seasonal_mean(var_data, season=s, is_climo=True))
    else:
        # Handle single DataArray
        for s in seasons:
            dataarrays.append(pf.seasonal_mean(ds, season=s, is_climo=True))

    # Combine seasonal means
    ds_seasonal = xr.concat(dataarrays, dim='season')
    ds_seasonal['season'] = seasons
    ds_seasonal = ds_seasonal.transpose('lat', 'lon', 'season')

    return ds_seasonal