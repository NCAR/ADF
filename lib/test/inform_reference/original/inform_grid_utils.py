import pathlib as path
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation  


def grid_flight(cesm: xr.open_dataset, cesm_dat: xr.open_dataset, df: pd.DataFrame) -> dict:    
    
    # Step 1: Identify Variables Automatically
    lat_var = next((var for var in df.columns if 'GGLAT' in var), None)
    lon_var = next((var for var in df.columns if 'GGLON' in var), None)
    alt_var = next((var for var in df.columns if 'GGALT' in var or 'PSXC' in var), None)

    # Compute pressure altitude (palt) from CESM hybrid coordinates
    p0 = cesm.P0  # Reference pressure
    ps = cesm.PS  # Surface pressure [=] Pa
    hyai = cesm.hyai  # Hybrid A coefficient at layer interface
    hybi = cesm.hybi  # Hybrid B coefficient at layer interface
    
    P_dummy = p0 * hyai
    midP = (P_dummy + (hybi * ps))  # [=] Pa
    palt = midP * 0.01  # Convert to hPa
    
    df_vars = [col for col in df.columns if col.lower() != 'time']
    
    if not lat_var or not lon_var or not alt_var:
        raise ValueError("Missing essential latitude, longitude, or altitude variables.")
    
    # Step 2: Create a 3D grid based on CESM & flight data
    # Compute midpoints
    lat_mp = (cesm.lat[:-1] + cesm.lat[1:]) / 2
    lon_mp = (cesm.lon[:-1] + cesm.lon[1:]) / 2
    # Find lat & lon bounds based on aircraft min/max values
    lat_bounds = [int(np.abs(lat_mp - np.min(df[lat_var])).argmin()) - 1, 
                  int(np.abs(lat_mp - np.max(df[lat_var])).argmin()) + 1]
    lon_bounds = [int(np.abs(lon_mp - np.min(df[lon_var])).argmin()) - 1, 
                  int(np.abs(lon_mp - np.max(df[lon_var])).argmin()) + 1]

    # Ensure bounds are within valid range
    lat_bounds = [max(0, lat_bounds[0]), min(len(lat_mp) - 1, lat_bounds[1])]
    lon_bounds = [max(0, lon_bounds[0]), min(len(lon_mp) - 1, lon_bounds[1])]
    
    # Select the subset of `palt` corresponding to the lat/lon bounds
    palt_subset = palt.isel(lat=slice(lat_bounds[0], lat_bounds[1] + 1),
                            lon=slice(lon_bounds[0], lon_bounds[1] + 1))

    # Extract altitude values based on aircraft altitude range
    min_alt, max_alt = np.min(df[alt_var]), np.max(df[alt_var])

    # Find altitude bounds dynamically for each lat-lon grid cell
    alt_indices = []
    for lat_idx in range(lat_bounds[0], lat_bounds[1] + 1):
        for lon_idx in range(lon_bounds[0], lon_bounds[1] + 1):
            local_palt = palt.isel(lat=lat_idx, lon=lon_idx).values  # Get 1D alt profile for this grid cell

            min_alt_idx = np.abs(local_palt - min_alt).argmin()
            max_alt_idx = np.abs(local_palt - max_alt).argmin()

            alt_indices.append((min_alt_idx, max_alt_idx))

    # Determine overall altitude bounds from all lat-lon cells
    min_alt_bound = min(i[0] for i in alt_indices)
    max_alt_bound = max(i[1] for i in alt_indices)
    
    # Ensure altitude bounds are valid
    alt_bounds = [max(0, min_alt_bound - 1), min(palt.shape[0] - 1, max_alt_bound + 1)]
    # Define bounds dictionary
    bounds = {'lat': lat_bounds, 'lon': lon_bounds, 'palt': alt_bounds}

    # Create grid with the correct shape
    grid_shape = (alt_bounds[1] - alt_bounds[0] + 1, 
                  lat_bounds[1] - lat_bounds[0] + 1, 
                  lon_bounds[1] - lon_bounds[0] + 1)
        
        # return np.zeros(grid_shape), bounds
    grid = np.zeros(grid_shape)

    # Step 3: Match aircraft times with CESM times
    da = xr.DataArray(cesm_dat.time, dims="time")
    cesm_times = pd.to_datetime([pd.Timestamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
                        for dt in da.values])    
    aircraft_times = pd.to_datetime(df['Time'])
    
    times = np.array(cesm_times[np.isin(cesm_times, aircraft_times)])
    
    # Step 4: Initialize grid arrays
    mean_lat, mean_lon, mean_alt = np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
    
    mean_values = {var: np.zeros_like(grid, dtype=float) for var in df_vars}
    mean_lat, mean_lon, mean_alt = np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)
    
    # Generate latitude, longitude grid values
    lats = np.array(lat_mp[bounds['lat'][0]:bounds['lat'][1] + 1])
    lons = np.array(lon_mp[bounds['lon'][0]:bounds['lon'][1] + 1])
    
    # Select the region of interest in `palt`
    palt_subset = palt.sel(lat=slice(lats.min(), lats.max()), lon=slice(lons.min(), lons.max()))
    # Adjust time range
    new_time = times[-1] + np.timedelta64(30, 'm')
    minus_time = times[0] - np.timedelta64(30, 'm')
    times = np.append(minus_time, np.append(times, new_time))
    
    mid_time = []
    for t in range(0, len(times) - 1):
        time_start, time_end = times[t], times[t + 1]
    
        # Select aircraft data within the time interval
        air_time_indices = (df['Time'] > time_start) & (df['Time'] <= time_end)
        sliced_df_time = df[air_time_indices]
    
        if not sliced_df_time.empty:
            # Digitize lat & lon into grid bins
            lat_bins = np.digitize(sliced_df_time['GGLAT'], lats) - 1
            lon_bins = np.digitize(sliced_df_time['GGLON'], lons) - 1
    
            # Ensure valid lat/lon indices
            valid_mask = (lat_bins >= 0) & (lat_bins < len(lats) - 1) & \
                         (lon_bins >= 0) & (lon_bins < len(lons) - 1)
    
            if valid_mask.any():
                # Filter valid rows
                sliced_df = sliced_df_time.loc[valid_mask].copy()
                sliced_df['lat_bin'] = lat_bins[valid_mask]
                sliced_df['lon_bin'] = lon_bins[valid_mask]
                # Compute altitude bins dynamically based on (lat_bin, lon_bin)
                alt_bins = np.full(len(sliced_df), -1, dtype=int)  # Initialize invalid bins

                for i, (lat_idx, lon_idx, psxc) in enumerate(zip(sliced_df['lat_bin'], sliced_df['lon_bin'], sliced_df['PSXC'])):
                    # Extract altitude levels for this (lat_idx, lon_idx) from `palt`
                    local_alt_profile = palt_subset.isel(lat=lat_idx, lon=lon_idx).values  # Keep natural order
                    # Ensure the pressure profile is in the correct shape
                    local_alt_profile = local_alt_profile.squeeze()
                
                    # Use np.searchsorted instead of np.digitize to find the correct bin
                    alt_bins[i] = np.searchsorted(local_alt_profile, psxc, side='right') - 1

                # Assign altitude bins
                sliced_df['alt_bin'] = alt_bins
    
                # Ensure valid altitude indices
                valid_alt_mask = (alt_bins >= 0) & (alt_bins < palt.shape[0] - 1)
                sliced_df = sliced_df[valid_alt_mask]
    
                # Group by grid cells and compute means                
                grouped = sliced_df.groupby(['alt_bin', 'lat_bin', 'lon_bin'])
    
                for var in df_vars:
                    grouped_mean = grouped[var].mean()
                    if not grouped_mean.empty:  # ✅ Skip empty bins
                        mean_values[var][tuple(zip(*grouped_mean.index.to_numpy()))] = grouped_mean.values
    
                        mean_lat[tuple(zip(*grouped[lat_var].mean().index.to_numpy()))] = grouped[lat_var].mean().values
                        mean_lon[tuple(zip(*grouped[lon_var].mean().index.to_numpy()))] = grouped[lon_var].mean().values
                        mean_alt[tuple(zip(*grouped[alt_var].mean().index.to_numpy()))] = grouped[alt_var].mean().values
    
                # Compute mid-time for each grid cell
                grouped_times = grouped['Time'].agg(lambda x: x.min() + (x.max() - x.min()) / 2)
                mid_time.extend(grouped_times.values)
    
    # Convert mid_time to NumPy array
    mid_time = np.array(mid_time, dtype='datetime64[ns]')
    
    # Filter valid data points
    mean_t = mean_values['ATX']
    valid_indices = np.argwhere(mean_t != 0)  # (N, 3) array of (alt, lat, lon)
    
    alt_indices, lat_indices, lon_indices = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
    
    selected_time = mid_time[:len(valid_indices)]
    
    grid_dict = {
        'Time': selected_time,
        'Latitude': mean_lat[mean_t != 0],
        'Longitude': mean_lon[mean_t != 0],
        'Altitude': mean_alt[mean_t != 0],
    }
    grid_dict.update({var: mean_values[var][mean_t != 0] for var in df_vars})
    
    # Sort the data by time
    sorted_indices = np.argsort(selected_time)
    grid_dict = {key: np.array(value)[sorted_indices] for key, value in grid_dict.items()}
    grid_dict.update({'long_names': df.attrs['long_names']})
    
    # Confirm data integrity
    if all(len(v) > 0 for v in grid_dict.values()):
        print("✅ Grid dictionary successfully populated with data!")
    else:
        print("⚠️ Warning: Some entries in grid_dict are empty!")
    
    return grid_dict, grid, bounds


def plot_3d_track(grid_data,df):

    # Create a figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(grid_data['GGLON'], grid_data['GGLAT'], grid_data['PSXC'], c=grid_data['ATX'], cmap='viridis', marker='^',label='grid-mean values',s=100)
    # Invert the Z-axis
    ax.scatter(df['GGLON'], df['GGLAT'],df['PSXC'], c=df['ATX'], label='3D Flight track',s=12)
    ax.invert_zaxis()

    ax.set_xlabel('latitude (deg)')
    ax.set_ylabel('longitude (deg)')
    ax.set_zlabel('pressure alt (hPa)') 

    ax.legend()
    # # Color bar to show the mapping of color to the fourth dimension
    plt.colorbar(sc, label='Mean Temperature (°C)')

    # Animation function to rotate the view
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)

    # Create animation
    ani = FuncAnimation(fig, rotate, frames=np.arange(-180, 360, 20), interval=100)

    # Show the animation in Jupyter Notebook
    from IPython.display import HTML
    HTML(ani.to_jshtml())

    plt.show()

def write_nc(grid_data, filename="test_grid_data.nc"):
    """
    Automatically creates and saves a NetCDF file from the given grid data dictionary.
    
    :param grid_data: Dictionary containing time series data with "Time" and corresponding variables.
    :param filename: Name of the NetCDF file to be saved.
    """

    # Extract headers dynamically (excluding "Time")
    headers = [key for key in grid_data.keys() if key.lower() != "time"]

    # Create the xarray dataset dynamically
    ds = xr.Dataset(
        {var: (["time"], grid_data[var]) for var in headers},  # Assign all variables dynamically
        coords={"time": grid_data["Time"]},  # Set "Time" as the coordinate
    )
    # # Save to a NetCDF file
    ds.to_netcdf("test_grid_data.nc")
    print("NetCDF file 'grid_data.nc' saved successfully!")