import pandas as pd
import numpy as np
import inform_utils as inform
import glob
import xarray as xr
import datetime
from scipy.spatial import cKDTree
from datetime import time
from tqdm.notebook import tqdm, trange  
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

def assign_flight_type(df):
    """
    Assigns flight type ('level' or 'profile') to each row of the input DataFrame based on stable altitude blocks 
    and gaps between these blocks. The function uses rolling standard deviation of altitude to identify level legs
    and combines consecutive blocks of stable altitude with a specified time gap threshold. Additionally, it labels 
    flight segments as "level" for level legs and "profile" for the aircraft vertical profile.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame with at least the following columns:
        - 'Time' (timestamp)
        - 'GGALT' (altitude in meters)

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with a new column 'flight_type', where each row is assigned a flight type:
        - 'level' for stable altitude periods
        - 'profile' for gaps between stable altitude blocks

    Example:
    --------
    df = pd.read_csv('flight_data.csv')  # Assuming the CSV contains relevant columns
    df_with_flight_types = assign_flight_type(df)
    """

    #-----------------------------------------
    #----- Find profiles and level legs ------
    #-----------------------------------------
    
    # Define a time gap threshold to combine blocks (e.g., 120 seconds)
    time_gap_threshold = pd.Timedelta(seconds=120)
    
    # Compute rolling standard deviation of altitude to smooth noise
    df['rolling_std'] = df['GGALT'].rolling(window=10, center=True).std()
    
    # Identify where altitude remains stable within the threshold
    df['stable'] = df['rolling_std'] < 3  # You can adjust the threshold (meters)
    
    # Assign unique block IDs when stability changes
    df['block_id'] = (df['stable'] != df['stable'].shift()).cumsum()
    
    # Group by block_id and filter for long-duration stable blocks
    block_info = df[df['stable']].groupby('block_id').agg(
        start_time=('Time', 'first'),
        end_time=('Time', 'last'),
        lower_bound=('GGALT', 'min'),  # Minimum altitude (lower bound)
        upper_bound=('GGALT', 'max'),  # Maximum altitude (upper bound)
        duration=('Time', lambda x: x.max() - x.min())
    )
    
    # Filter out short-duration blocks
    valid_blocks = block_info[block_info['duration'] > pd.Timedelta(seconds=150)] ## EDIT?
    
    # Sort the blocks by start time
    valid_blocks = valid_blocks.sort_values(by='start_time')
    
    # Define a time gap threshold to combine blocks (e.g., 120 seconds)
    time_gap_threshold = pd.Timedelta(seconds=120)
    
    # Combine consecutive blocks that are less than the threshold apart
    combined_blocks = []
    previous_block = valid_blocks.iloc[0]
    
    for idx, current_block in valid_blocks.iloc[1:].iterrows():
        # Check if the gap between the end time of the previous block and start time of the current block is below the threshold
        if current_block['start_time'] - previous_block['end_time'] <= time_gap_threshold:
            # Extend the previous block's end time to the current block's end time
            previous_block['end_time'] = current_block['end_time']
        else:
            # If the gap is too large, append the previous block and update to the current block
            combined_blocks.append(previous_block)
            previous_block = current_block
    
    # Add the last block after the loop
    combined_blocks.append(previous_block)
    
    # Convert combined blocks back to DataFrame
    combined_blocks_df = pd.DataFrame(combined_blocks)
    
    # --- Identify and Label "Profiles" between "Level" (Stable) sections ---
    
    # Create a new column 'flight_type' to categorize the blocks as "level" or "profile"
    combined_blocks_df['flight_type'] = 'level'  # By default, label as 'level'
    
    # Now identify the gaps between "level" blocks and label as "profile"
    profile_blocks = []
    for i in range(len(combined_blocks_df) - 1):
        end_time_current = combined_blocks_df.iloc[i]['end_time']
        start_time_next = combined_blocks_df.iloc[i + 1]['start_time']
        
        # If there's a gap between two 'level' blocks, label the gap as 'profile'
        if start_time_next - end_time_current > time_gap_threshold:
            # Assign 'profile' to the gap between two level blocks and calculate duration
            profile_duration = start_time_next - end_time_current  # Duration of the profile block
            
            profile_blocks.append({
                'start_time': end_time_current,
                'end_time': start_time_next,
                'flight_type': 'profile',
                'duration': profile_duration
            })
    
    # Convert 'profile_blocks' to DataFrame
    profile_blocks_df = pd.DataFrame(profile_blocks)
    
    # Append profile blocks to the original combined blocks DataFrame
    combined_blocks_with_profiles = pd.concat([combined_blocks_df, profile_blocks_df], ignore_index=True)
    
    # Sort again by time
    combined_blocks_with_profiles = combined_blocks_with_profiles.sort_values(by='start_time')
    
    # Check for "Profile" after the Last Level Block
    last_end_time = combined_blocks_with_profiles.iloc[-1]['end_time']
    last_time_in_data = df['Time'].max()
    
    if last_time_in_data - last_end_time > time_gap_threshold:
        # If the gap is greater than the threshold, consider it a "profile" block
        profile_block = pd.DataFrame([{
            'start_time': last_end_time,
            'end_time': last_time_in_data,
            'flight_type': 'profile',
        }])
    
        # Concatenate the new profile block to the existing DataFrame
        combined_blocks_with_profiles = pd.concat([combined_blocks_with_profiles, profile_block], ignore_index=True)
    
    # Check for "Profile" before the First Level Block, used for takeoff/landing
    first_start_time = combined_blocks_with_profiles.iloc[0]['start_time']
    first_time_in_data = df['Time'].min()
    
    if first_start_time - first_time_in_data > time_gap_threshold:
        profile_block_before_first = pd.DataFrame([{
            'start_time': first_time_in_data,
            'end_time': first_start_time,
            'flight_type': 'profile',
        }])
    
        combined_blocks_with_profiles = pd.concat([profile_block_before_first, combined_blocks_with_profiles], ignore_index=True)
    
    # List of columns to remove
    columns_to_remove = ['rolling_std','stable','block_id']
    # Drop the specified columns from df2
    df = df.drop(columns=columns_to_remove)
    
    # Add new column "flight_type" as either "level" or "profile"
    for _, row in combined_blocks_with_profiles.iterrows():
        flight_type = row['flight_type']
        # Find rows in df2 where the time is between start_time and end_time
        mask = (df['Time'] >= row['start_time']) & (df['Time'] <= row['end_time'])
        df.loc[mask, 'flight_type'] = flight_type
    # Assign the flight_type to the first few rows that fall before the first start_time in df1
    df.loc[df['Time'] < first_start_time, 'flight_type'] = df.iloc[0]['flight_type']

    #------------------------------
    #----- Find cloud layers ------
    #------------------------------
    # Ensure 'Time' is in datetime format
    df = df.copy()  # Avoid modifying original DataFrame
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Find best match for column names dynamically
    plwc_col = next((col for col in df.columns if 'PLWCD' in col), None) or \
           next((col for col in df.columns if 'PLWC' in col), None)
    concd_col = next((col for col in df.columns if 'CONCD' in col), None)
    # Add check if there are any cloudy periods
    if not plwc_col or not concd_col:
        print("Required columns not found. Skipping cloud detection.")
        final_cloud_blocks = pd.DataFrame(columns=['start_time', 'end_time', 'lower_bound', 'upper_bound', 'duration', 'Location'])
        df['cloud_status'] = 'Out-of-cloud'
        df['Location'] = 'Free'
    else:
        df['blocked'] = (df[plwc_col] > 0.001) & (df[concd_col] > 10)
        if not df['blocked'].any():
            print("No valid cloud blocks found. Skipping cloud layer logic.")
            final_cloud_blocks = pd.DataFrame(columns=['start_time', 'end_time', 'lower_bound', 'upper_bound', 'duration', 'Location'])
            df['cloud_status'] = 'Out-of-cloud'
            df['Location'] = 'Free'
        else:
            df['block_id'] = (df['blocked'] != df['blocked'].shift()).cumsum()
            block_info = df[df['blocked']].groupby('block_id').agg(
                start_time=('Time', 'first'),
                end_time=('Time', 'last'),
                lower_bound=('GGALT', 'min'),
                upper_bound=('GGALT', 'max'),
            )

            # Calculate duration directly by subtracting start_time from end_time
            block_info['duration'] = block_info['end_time'] - block_info['start_time']
            
            # Filter out short-duration blocks
            min_vertical = 30  # Adjust as needed (100 meters in your case)
            valid_blocks = block_info[(block_info['upper_bound'] - block_info['lower_bound']) > min_vertical].reset_index(drop=True)
            
            # Define the altitude difference and time gap thresholds
            altitude_gap_threshold = 200  # Increased altitude gap threshold
            # time_gap_threshold = pd.Timedelta(minutes=20)  # Time gap threshold for merging
            
            # Sort the valid blocks by their start time to process them in sequence
            valid_blocks = valid_blocks.sort_values(by='lower_bound')
            
            # Initialize a list to store combined blocks
            combined_blocks = []
            previous_block = valid_blocks.iloc[0].to_dict()
            
            # Iterate through the blocks and merge those that are within the thresholds
            for idx, current_block in valid_blocks.iloc[1:].iterrows():
                # Calculate the altitude gap between the current block's lower bound and the previous block's upper bound
                altitude_gap = abs(current_block['lower_bound'] - previous_block['upper_bound'])
                
                # Calculate the time gap between the current block's start time and the previous block's end time
                time_gap = current_block['start_time'] - previous_block['end_time']
                # Check if the altitude gap is within the threshold or if the time gap is within the allowed range for smaller altitudes
                if altitude_gap <= altitude_gap_threshold :
                    # If both criteria are met, merge the blocks
                    previous_block['end_time'] = max(previous_block['end_time'], current_block['end_time'])  # Get the latest end time
                    previous_block['start_time'] = min(previous_block['start_time'], current_block['start_time'])  # Get the earliest start time
                    previous_block['upper_bound'] = max(previous_block['upper_bound'], current_block['upper_bound'])  # Update upper bound
                    previous_block['lower_bound'] = min(previous_block['lower_bound'], current_block['lower_bound'])  # Update lower bound
                    
                    # Recalculate the duration for the merged block
                    previous_block['duration'] = previous_block['end_time'] - previous_block['start_time']
                else:
                    # If the blocks are far apart, save the previous block and move to the next one
                    combined_blocks.append(previous_block)
                    previous_block = current_block.to_dict()
            
            # Add the last block after the loop
            combined_blocks.append(previous_block)
            
            # Convert the merged blocks back into a DataFrame
            combined_blocks_df = pd.DataFrame(combined_blocks)
            
            # Second check for merging adjacent blocks in combined_blocks_df
            final_combined_blocks = []
            previous_block = combined_blocks_df.iloc[0].to_dict()
            
            # Apply additional check for merging based on both time and altitude gap
            for idx, current_block in combined_blocks_df.iloc[1:].iterrows():
                # Calculate the altitude gap and time gap
                altitude_gap = abs(current_block['lower_bound'] - previous_block['upper_bound'])
                time_gap = current_block['start_time'] - previous_block['end_time']
                
                # Check for overlap in the altitude ranges
                overlap_check = (current_block['lower_bound'] >= previous_block['lower_bound']) and (current_block['lower_bound'] <= previous_block['upper_bound'])
            
                # Check if both the altitude gap, time gap, or overlap condition is met
                if altitude_gap <= altitude_gap_threshold or overlap_check:
                    # Merge the blocks
                    previous_block['end_time'] = max(previous_block['end_time'], current_block['end_time'])
                    previous_block['start_time'] = min(previous_block['start_time'], current_block['start_time'])
                    previous_block['upper_bound'] = max(previous_block['upper_bound'], current_block['upper_bound'])
                    previous_block['lower_bound'] = min(previous_block['lower_bound'], current_block['lower_bound'])
                    
                    # Recalculate the duration for the merged block
                    previous_block['duration'] = previous_block['end_time'] - previous_block['start_time']
                else:
                    # Save the previous block and move to the next one
                    final_combined_blocks.append(previous_block)
                    previous_block = current_block.to_dict()
            
            # Add the last block after the loop
            final_combined_blocks.append(previous_block)
            
            # Convert the final combined blocks back into a DataFrame
            final_cloud_blocks = pd.DataFrame(final_combined_blocks)

            # Add 'cloud_status' based on whether altitude and time fall within any blocked region (in the cloud or out of cloud)
            df['cloud_status'] = 'Out-of-cloud'  # Default label
            # Loop through each block and label altitudes as "In-cloud" if they fall within the block's range
            for _, block in final_cloud_blocks.iterrows():
                # Create a mask that checks both altitude and time conditions
                mask = (
                    (df['GGALT'] >= block['lower_bound']) & (df['GGALT'] <= block['upper_bound']) &
                    (df['Time'] >= block['start_time']) & (df['Time'] <= block['end_time'])
                )
                
                # Apply the 'In-cloud' label where the mask is True
                df.loc[mask, 'cloud_status'] = 'In-cloud'
            
            # List of columns to remove
            columns_to_remove = ['blocked','block_id']
            # Drop the specified columns from df2
            df = df.drop(columns=columns_to_remove)
        
            df['Location'] = 'Free'
            
            # Find the minimum in-cloud altitude
            min_ic_alt = np.min(final_cloud_blocks['lower_bound'])-5
            mask = df.GGALT < min_ic_alt
            # Define 
            df.loc[mask, 'Location'] = 'BL'
        
            # Update the Location column based on the GGALT and cloud status
            df.loc[df['GGALT'] < min_ic_alt, 'Location'] = 'BL'
        
            # --- Add Location to final_cloud_blocks DataFrame ---
            # Add 'Location' based on the minimum in-cloud altitude
            final_cloud_blocks['Location'] = final_cloud_blocks['lower_bound'].apply(
                lambda x: 'BL' if x < min_ic_alt else 'Free'
            )
                # Add 'Location' based on the minimum in-cloud altitude
            combined_blocks_with_profiles['Location'] = combined_blocks_with_profiles['lower_bound'].apply(
                lambda x: 'BL' if x < min_ic_alt else 'Free'
            )

    # Sort the dataframe by Time for continuous time grouping
    df = df.sort_values(by='Time')
    # Remove rows where 'flight_type' is NaN
    df = df.dropna(subset=['flight_type'])
    # Create a new column 'block_id' to group continuous time periods based on flight_type, cloud_status, and Location
    df['block_id'] = (df['flight_type'] != df['flight_type'].shift()) | \
                      (df['cloud_status'] != df['cloud_status'].shift()) | \
                      (df['Location'] != df['Location'].shift())
    df['block_id'] = df['block_id'].cumsum()

    Final_ds = {'DataFrame': df,
                'flight_blocks': combined_blocks_with_profiles,
                'Cloud_blocks': final_cloud_blocks
               }
    
    return Final_ds

def block_flight(df):
    """
    Segments a flight dataset into different flight block categories based on cloud status, location, and flight type.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing flight data with at least the following columns:
        - 'block_id' (int): Identifies different flight segments.
        - 'Location' (str): Can be 'BL' (Boundary Layer) or 'Free' airspace.
        - 'flight_type' (str): Can be 'level' or 'profile'.
        - 'cloud_status' (str): Either 'In-cloud' or 'Out-of-cloud'.
        - 'GGALT' (float): Altitude dbata, used for filtering profile segments.
        - 'Time' (datetime): Used to filter level flight segments.

    Returns:
    --------
    Flight_blocks : dict
        A dictionary containing categorized flight data:
        - 'Level BL': List of DataFrames for level flight in the boundary layer.
        - 'In-Cloud Profiles': List of DataFrames for in-cloud profile flights with altitude variation > 30m.
        - 'In-Cloud Level FT': List of DataFrames for level flights in free airspace within clouds.
        - 'Out-of-cloud Level FT': List of DataFrames for level flights in free airspace, lasting at least 3 minutes.

    Notes:
    ------
    - The function removes the first and last 'Level BL' periods to exclude takeoff/landing effects.
    - Only level flights lasting more than 180 seconds are included in 'Out-of-cloud Level FT'.
    - Profile flights are only included if their altitude change is greater than 30 meters.
    """
    # ---------- Level BL periods (KEEP ALL) ----------
    out_of_cloud_bl = df[(df['Location'] == 'BL') & (df['flight_type'] == 'level')]
    bl_ids = sorted(out_of_cloud_bl['block_id'].unique())
    bl_blocks_ds = [df[df['block_id'] == i] for i in bl_ids]
    
    # Find In-Cloud profile periods
    in_cloud_prof = df[(df['cloud_status'] == 'In-cloud') & (df['flight_type'] == 'profile')]
    ic_prof_ids = sorted(in_cloud_prof['block_id'].unique())
    ic_pro_blocks_ds = [df[df['block_id'] == i] for i in ic_prof_ids if df[df['block_id'] == i]['GGALT'].max() - df[df['block_id'] == i]['GGALT'].min() > 30]
    
    # Find level FT periods out-of-cloud
    level_ft = df[(df['cloud_status'] == 'Out-of-cloud') & (df['flight_type'] == 'level') & (df['Location'] == 'Free')]
    out_of_cloud_ft_ids = sorted(level_ft['block_id'].unique())
    level_ft_out_blocks_ds = [df[df['block_id'] == i] for i in out_of_cloud_ft_ids if df[df['block_id'] == i]['Time'].iloc[-1] - df[df['block_id'] == i]['Time'].iloc[0] > pd.Timedelta(seconds=180)]

    # Find level FT periods in-cloud
    level_ft_ic = df[(df['cloud_status'] == 'In-cloud') & (df['flight_type'] == 'level') & (df['Location'] == 'Free')]
    in_cloud_ft_ids = sorted(level_ft_ic['block_id'].unique())
    level_ft_ic_blocks_ds = [df[df['block_id'] == i] for i in in_cloud_ft_ids]
    
    # Save blocks of flight as dictionary for output
    Flight_blocks = {
        'Level BL': bl_blocks_ds,
        'In-Cloud Profiles': ic_pro_blocks_ds,
        'In-Cloud Level FT': level_ft_ic_blocks_ds,
        'Out-of-cloud Level FT': level_ft_out_blocks_ds
    }

    return Flight_blocks

def assign_cloud_type_HCR(flight_blocks, dir, idx: int = 0):
    """
    Assigns cloud echo type classifications from HCR (HIAPER-Cloud Radar) data 
    to flight data blocks within the global Flight_blocks variable.

    Parameters:
    -----------
    dir : str
        Base directory path where the RF (Research Flight) subfolders are located.
    idx : int, optional
        Index of the flight number (used to construct the folder name as 'RF{idx}'), 
        by default 0.

    Returns:
    --------
    dict
        Updated Flight_blocks dictionary with an added 'Echo_Type' column in each block, 
        indicating the radar-derived cloud classification at each time step.

    Notes:
    ------
    - Requires global Flight_blocks to be defined externally.
    - Each time in the flight data is matched with HCR timestamps to assign echo types.
    - Echo type values are pulled from the 'HCR_ECHO_TYPE_1D' variable in netCDF files.
    """

    dir_fold = dir + f"RF{idx+1:02d}" + '/'
    flight_paths = inform.find_flight_fnames(dir_fold)

    hcr_time = []
    echo_type_1D = []
    
    for file in flight_paths:
        nc = inform.open_nc(file)
        hcr_time.extend(np.array(nc.time))
        echo_type_1D.append(np.array(nc.HCR_ECHO_TYPE_1D))
    
    echo_type_1D = np.concatenate(echo_type_1D)
    
    for val in flight_blocks:
        block_type = flight_blocks[val]
        for i in range(len(block_type)):
            # block = block_type[i]
            block = block_type[i].copy()  # Make a copy to avoid SettingWithCopyWarning
            # Extract start and end time from the block
            start_time = block['Time'].iloc[0]
            end_time = block['Time'].iloc[-1]
            
            # Convert start_time and end_time to the same format as hcr_time if necessary
            # Assuming hcr_time is a list of datetime objects, convert if needed
            if isinstance(start_time, str):  # If start_time is in string format, convert it
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):  # If end_time is in string format, convert it
                end_time = pd.to_datetime(end_time)
    
            hcr_time_array = np.array(hcr_time)
            echo_column = np.full(len(block), np.nan)
    
            for j in range(len(block)):
                time_point = pd.to_datetime(block['Time'].iloc[j])
                match_indices = np.where(hcr_time_array == time_point)[0]
                if len(match_indices) > 0:
                    echo_column[j] = echo_type_1D[match_indices[0]]
    
            block.loc[:, 'Echo_Type'] = echo_column  # <- Use loc for safe assignment
            block_type[i] = block
    
        flight_blocks[val] = block_type
        
    return flight_blocks


# High-Level function
def VAP_process_flight_data(df,i):
    """
    High-Level Function for Processing Flight Data in Value Added Products.

    This function serves as the main entry point for processing flight data. It first calls 
    `assign_flight_type` to assign flight types (e.g., 'level' or 'profile') to different segments of the flight 
    based on altitude stability and time gaps. After flight types are assigned, it proceeds to categorize the data 
    into different flight blocks (e.g., level flight in boundary layer, in-cloud profile flight, etc.) by calling 
    the `block_flight` function.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing flight data with at least the following columns:
        - 'Time' (datetime): Time of each flight record.
        - 'GGALT' (float): Altitude of the aircraft.
        - 'PLWCD_' (float): Cloud Droplet Probe LWC.
        - 'CONCD_' (float): Cloud Droplet Probe Number Concentration.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'DataFrame': A modified DataFrame with assigned flight types, cloud status, and location.
        - 'flight_blocks': A dictionary of flight blocks categorized by flight type and cloud status.
        - 'cloud_blocks': A dataframe of flight blocks including blocks of aircraft data inside cloud layers.

    Notes:
    ------
    - The `assign_flight_type` function is responsible for determining whether the flight segments are 'level' or 'profile'.
    - The `block_flight` function segments the flight data based on the assigned flight types and cloud status into specific blocks (e.g., 'Level BL', 'In-Cloud Profiles', etc.).
    - The function ensures proper labeling of different flight segments for further analysis, including cloud status and location (e.g., boundary layer or free airspace).
    """
    # Function to assign flight type "Level" and "Profile" when in/out of cloud
    dict_flight_type = assign_flight_type(df)

    # Extract dataframe that has been modified from the assign_flight_type function
    df_mod = dict_flight_type['DataFrame']
    # Plot time series of aircraft defined flight blocks
    # plot_block_ts(dict_flight_type,i)

    # Run block flight function to return list of Dataframes of "blocked" flight data
    flight_blocks = block_flight(df_mod)
    # Function to assign cloud type from the HCR data
    # flight_block_comp = assign_cloud_type_HCR(flight_blocks,dir,i)
    
    # Plot time series of HCR defined cloud types  
    # plot_hcr_cloud_type(df_mod,flight_block_comp,i)
    return flight_blocks

def select_ERA5_4flight(df, campaign, dat_type="aircraft"):
    # Define function to filter ERA5 files based on time
    def get_matching_files(pattern, start_dt, end_dt):
        file_list = glob.glob(pattern)
        selected = []
        for file in file_list:
            time_strs = file.split('.')[-2].split('_')
            file_start = datetime.datetime.strptime(time_strs[0], "%Y%m%d%H")
            file_end = datetime.datetime.strptime(time_strs[1], "%Y%m%d%H")
            if file_start <= end_dt and file_end >= start_dt:
                selected.append(file)
        return selected
    
    filepath_sfc = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"
    filepath_pl = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    # Extract the times of the research flight
    month, year = df.Time[0].month, df.Time[0].year
    day_start,day_end = df.Time[0].day, df.Time.iloc[-1].day
    start_hour, end_hour = df.Time[0].hour, df.Time.iloc[-1].hour
    
    # Select the latitude/longitude box to reduce size of era5 data
    if campaign == 'SOCRATES':
        lat_max, lat_min = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())
    elif campaign == 'CSET':
        lat_min, lat_max = np.floor(df.GGLAT.min()), np.ceil(df.GGLAT.max())
        
    # ---- build lat/lon selection from the flight track ----
    # Aircraft → 0..360 to match ERA5
    lon0 = ((df.GGLON.to_numpy(dtype=float) % 360.0) + 360.0) % 360.0
    lat0 = df.GGLAT.to_numpy(dtype=float)
    
    # Robust bounds (with a small pad for ERA5 0.25° grid)
    pad = 0.5
    lon_min = float(np.floor(np.nanmin(lon0) - pad))
    lon_max = float(np.ceil (np.nanmax(lon0) + pad))
    # keep inside ERA5 domain
    lon_min = max(0.0, lon_min)
    lon_max = min(359.999, lon_max)
    
    lat_min = float(np.floor(np.nanmin(lat0) - pad))
    lat_max = float(np.ceil (np.nanmax(lat0) + pad))
    
    # ERA5 latitude is usually descending (90 → -90): use slice(max, min)
    lat_slice = slice(lat_max, lat_min)
    
    # ERA5 longitudes are ascending (0 → 360): use slice(min, max)
    lon_slice = slice(lon_min, lon_max)
    
    print("Selecting ERA5 box:",
      f"lon {lon_min}→{lon_max} (0–360), lat {lat_max}→{lat_min} (descending)")

    # Make the yearmonth string for file selection
    dir_date = f"{year}{month:02d}"
    
    # Flight start and end times
    start_dt = datetime.datetime(year, month, day_start, start_hour) 
    end_dt = datetime.datetime(year, month, day_end, end_hour)+datetime.timedelta(hours=1)

    # ---- apply the SAME slices to every dataset you open ----
    ds_sp  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_sp.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_sst  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_sstk.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t2m  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_2t.*.nc", start_dt, end_dt),
                                combine='by_coords').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_u10  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_10u.*.nc", start_dt, end_dt),
                                combine='by_coords')[['VAR_10U']].sel(latitude=lat_slice, 
                                                                      longitude=lon_slice,time=slice(start_dt,end_dt))
    ds_v10  = xr.open_mfdataset(get_matching_files(f"{filepath_sfc}{dir_date}/*_10v.*.nc", start_dt, end_dt),
                                combine='by_coords')[['VAR_10V']].sel(latitude=lat_slice, longitude=lon_slice, 
                                                                      time=slice(start_dt,end_dt))
    
    ds_w    = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_w.*.nc",  start_dt, end_dt),
                                combine='nested', concat_dim='time')
    w_700   = ds_w['W'].sel(level=700).sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    
    # ds_rh700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_r.*.nc", start_dt, end_dt),
    #                              combine='nested', concat_dim='time')[['R']].sel(level=700).drop_vars('level', errors='ignore')
    # rh      = ds_rh700['R'].rename('RH').sortby('time').sel(latitude=lat_slice, longitude=lon_slice, 
    # time=slice(start_dt,end_dt))
    ds_q700  = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_q.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['Q']].sel(level=700).drop_vars('level', errors='ignore')
    q       = ds_q700['Q'].rename('Q').sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_u700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_u.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['U']].sel(level=700).drop_vars('level', errors='ignore')
    ds_u700 = ds_u700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_v700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_v.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['V']].sel(level=700).drop_vars('level', errors='ignore')
    ds_v700 = ds_v700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    
    ds_t    = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=800).drop_vars('level', errors='ignore')
    ds_t    = ds_t.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t700 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=700).drop_vars('level', errors='ignore')
    ds_t700 = ds_t700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
    ds_t850 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                                 combine='nested', concat_dim='time')[['T']].sel(level=850).drop_vars('level', errors='ignore')
    ds_t850 = ds_t700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))

    ws = np.sqrt(ds_u10.VAR_10U**2 + ds_v10.VAR_10V**2)
    wind_dir = (270 - np.degrees(np.arctan2(ds_v10.VAR_10V, ds_u10.VAR_10U))) % 360
    
    # Calculate RH from specfic humidity
    # Murphy & Koop (2005) saturation vapor pressure (Pa)
    def es_MK_water(T):
        # ln(esw [Pa]) valid ~123–332 K
        return xr.ufuncs.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T
                             + xr.ufuncs.tanh(0.0415*(T-218.8)) * (53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T))
    
    def es_MK_ice(T):
        # ln(esi [Pa]) valid ~110–273 K
        return xr.ufuncs.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
    
    esw = es_MK_water(ds_t700.T)
    esi = es_MK_ice(ds_t700.T)
    # Choose water above freezing, ice at/below (adjust threshold if you prefer 273.16)
    es = xr.where(ds_t700.T > 273.15, esw, esi)
    
    # Saturation specific humidity and RH
    eps = 0.622
    qsat = (eps * es) / (70000 - (1.0 - eps) * es)
    
    # Avoid division issues extremely near saturation/low p
    qsat = qsat.clip(min=1e-12)
    
    RH = (q / qsat) * 100.0

    # Calcualte wind shear (SFC - 700mb)
    ws700 = np.sqrt(ds_u700.U**2 + ds_v700.V**2)
    wind_shear = ws700-ws
    
    # Calculate M-value
    Rd = 287
    Cp = 1005   
    theta_sfc = ds_t2m.VAR_2T*(101325/ds_sp.SP)**(Rd/Cp)
    theta_800 = ds_t*(1013.25/800)**(Rd/Cp)
    
    M = theta_sfc.T - theta_800.T
    M = M.transpose("time", "latitude", "longitude")
    
    dt = ds_t2m.VAR_2T - ds_sst.SSTK
    
    # Constants
    Re = 6.371e6  # Earth radius in meters
    deg2rad = np.pi / 180
    phi = np.deg2rad(ds_sst.SSTK['latitude'])
    # meters per 1° at this latitude
    m_per_deg_lon = Re * np.cos(phi) * deg2rad
    m_per_deg_lat = Re * deg2rad

    if dat_type == "dropsonde": # Necessary to calculate delta x/y for dropsonde which only returns one column
        # --- SST gradients & Tadv (K/day) ---
        ds_sst2 = ds_sst.sortby(["latitude", "longitude"]).unify_chunks().chunk({"latitude": -1, "longitude": -1, "time": -1})

        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst2.SSTK.differentiate("longitude") / m_per_deg_lon   # K/m
        dT_dy = ds_sst2.SSTK.differentiate('latitude') / m_per_deg_lat   # K/m

    elif dat_type == "aircraft":
        # gradients in K/m  (NOTE the division by meters-per-degree)
        dT_dx = ds_sst.SSTK.differentiate("longitude") / m_per_deg_lon   # K/m
        dT_dy = ds_sst.SSTK.differentiate('latitude') / m_per_deg_lat   # K/m
 
    # advection: K/s -> K/day
    Tadv = -(ds_u10['VAR_10U'] * dT_dx + ds_v10['VAR_10V'] * dT_dy) * 86400.0
    # Convert to K/day
    Tadv = Tadv.rename("Tadv")

    # Calculate EIS following Wood and Bretherton (2006, J. Climate)
    cp = 1004.     # specific heat at constant pressure for dry air (J / kg / K)
    Rd = 287.         # gas constant for dry air (J / kg / K)
    kappa = Rd / cp
    Lhvap = 2.5e6    # Latent heat of vaporization (J / kg)
    g = 9.81 # m/s^2
    cp = 1004 # J/K/kg
    Lv = 2.5e6 # J/kg
    
    Rv = 461 # J/K/kg;
    Ra = 287 # J/K/kg
    
    def get_qsat(T,p):
        Tcel = T-273.15
        es=6.11*10**(7.5*Tcel/(Tcel+273.15))
        return 0.622*es/p
    
    # Calculate lower tropospheric stability (LTS)
    theta_700 = ds_t700.T*(1013.25/700)**kappa
    LTS = theta_700 - theta_sfc
    
    # T850 = (ds_t2m.VAR_2T+ds_t700.T)/2
    T850 = ds_t850.T
    
    Gammam = (g/cp*(1.0 - (1.0 + Lhvap*get_qsat(T850,850) / Rd / T850) /
                 (1.0 + Lhvap**2 * get_qsat(T850,850)/ cp/Rv/T850**2)))
    
    # Assume exponential decrease of pressure with scale height given by surface temperature
    z700 = (Rd * ds_t2m.VAR_2T / g) * np.log(1000 / 700)
    # Assume 80% relative humidity to compute LCL, appropriate for marine boundary layer
    Tadj = Tadj = ds_t2m.VAR_2T-55.  # in Kelvin
    LCL = cp/g*(Tadj - (1/Tadj - np.log(0.8)/2840.)**(-1))    
    EIS = LTS - Gammam*(z700 - LCL)

    # Convert w700 (m/s) to pa/s
    omega700 = -(70000 / (Rd * ds_t700.T)) * g * w_700
    
    # Merge the dataset variables used later
    ds = {
    'deltaT': dt,
    'Tadv': Tadv,
    'M': M,
    'omega700': omega700,
    'SST': ds_sst.SSTK,
    'WS': ws,
    'Wind_shear': wind_shear,
    'RH700': RH,
    'EIS': EIS
     }

    return ds

def wrap180(lon):
    # Map any longitude to [-180, 180)
    return (lon + 180.0) % 360.0 - 180.0

def nearest_time_indices(era5_times_ns, flight_times_ns):
    # era5_times_ns: 1D int64 nanoseconds, sorted
    # flight_times_ns: 1D int64 nanoseconds
    idx_right = np.searchsorted(era5_times_ns, flight_times_ns, side="left")
    idx_left  = np.clip(idx_right - 1, 0, len(era5_times_ns) - 1)
    idx_right = np.clip(idx_right,       0, len(era5_times_ns) - 1)
    choose_right = np.abs(era5_times_ns[idx_right] - flight_times_ns) < np.abs(era5_times_ns[idx_left] - flight_times_ns)
    return np.where(choose_right, idx_right, idx_left)


def collocate_ERA5_dat(ds, blocks):
    """
    Collocate ERA5 fields onto flight blocks.
    Assumes ds variables have dims ('time','latitude','longitude') and longitude is 0..360 ascending.
    """

    # Ensure Dataset and time sorted
    ds = xr.Dataset(ds).sortby('time')

    # Coords (ERA5 already 0..360)
    lat_vals = ds['latitude'].values
    lon_vals = ds['longitude'].values
    ny, nx   = lat_vals.size, lon_vals.size

    # KDTree over regular lat-lon grid (0..360 frame)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    tree = cKDTree(np.column_stack((lat_grid.ravel(), lon_grid.ravel())))

    # ERA5 times (ns, sorted)
    t_era_ns = ds['time'].values.astype('datetime64[ns]').view('int64')

    # Pull arrays once (T,Y,X) as NumPy
    def arr3(name):
        return ds[name].transpose('time','latitude','longitude').compute().values

    arr = {
        'ERA5_SST':   arr3('SST'),
        'M':          arr3('M'),
        'omega700':       arr3('omega700'),
        'deltaT':     arr3('deltaT'),
        'Wind_sp':    arr3('WS'),
        'Wind_shear': arr3('Wind_shear'),
        'Tadv':       arr3('Tadv'),
        'RH700':      arr3('RH700'),
        'EIS':        arr3('EIS'),
    }

    # Loop blocks
    for key, blist in blocks.items():
        for i, block in enumerate(blist):
            b = block.copy().dropna(subset=['GGLAT','GGLON'])
            if len(b) == 0:
                blist[i] = b
                continue

            # Flight coords/time (convert lon to 0..360)
            flt_lat  = b['GGLAT'].to_numpy(dtype=float)
            flt_lon  = ((b['GGLON'].to_numpy(dtype=float) % 360.0) + 360.0) % 360.0
            flt_t_ns = b['Time'].to_numpy('datetime64[ns]').view('int64')

            # Nearest time index per sample
            ti = nearest_time_indices(t_era_ns, flt_t_ns)  # (N,)

            # Nearest gridpoint (lat, lon) in 0..360 frame
            _, flat_idx = tree.query(np.column_stack((flt_lat, flt_lon)))  # (N,)
            yi, xi = np.unravel_index(flat_idx, (ny, nx))                  # (N,), (N,)

            # Gather all variables
            for out_name, A in arr.items():
                b[out_name] = A[ti, yi, xi]

            blist[i] = b
        blocks[key] = blist

    return blocks

# Function that writes each dictionary "Flight Blocks" as a netCDF
def write_RF_nc(fblks_cr, rf, campaign):
    combined = []
    if isinstance(fblks_cr, dict):
        for label, df_list in fblks_cr.items():
            for i, df in enumerate(df_list):
                df = df.copy()
                df["flight"] = rf
                df["block_label"] = label
                df["block_index"] = i
                combined.append(df)

        df_all = pd.concat(combined, ignore_index=True)
        df_all = df_all.set_index(["block_label", "block_index", "Time"])
        ds = df_all.reset_index().to_xarray()
        today = datetime.date.today().strftime("%Y%m%d")        
        # campaign prefix, no spaces, underscore separator
        campaign_str = str(campaign).upper().replace(" ", "")
        rf_str = str(rf).replace(" ", "_")
        name = f"{campaign_str}_{rf_str}_{today}.nc"

        ds.to_netcdf(name)
        print(f"Wrote {name}")


def cloud_regime_old(fblks):
    # Iterate through blocks
    for val in fblks:
        block_type = fblks[val]
        for i in range(len(block_type)):
            block = block_type[i].copy()
            block['cloud_regime'] = pd.Series('Undetermined', index=block.index, dtype='object')
            condition_cum = ((block['M'] > -7) & (block['Wind_shear'] < 6)) | ((block['M'] > -7) & (block['Wind_sp'] > 10))
            condition_strcu = ((block['M'] <= -10) & (block['Wind_sp'] < 10)) | ((block['M'] <= -10) | (block['Wind_shear'] > 6))
            # Assign values based on conditions
            block.loc[condition_cum, 'cloud_regime'] = 'Open-Cell Cu'
            block.loc[condition_strcu, 'cloud_regime'] = 'Stratiform'
    
            block_type[i] = block
    
        fblks[val] = block_type

    return fblks

def cloud_regime(fblks, campaign):
    """
    Assign cloud_regime per block.

    - SOCRATES:
        Stratocumulus (cond_strat) if ANY of:
            1) M < -9  and Wind_sp < 9
            2) M < -10 and EIS > 7
            3) M < -10 and Wind_shear < 9

        Open-Cell (cond_open) if ANY of:
            1) M >= -7 and Wind_sp >= 9
            2) M >= -8 and EIS < 9
            3) M >= -8 and Wind_shear > 6

    - CSET:
        Stratocumulus (cond_strat) if ANY of:
            1) M < -10 and SST < 295
            2) M < -11 and Tadv < 0

        Open-Cell (cond_open) if ANY of:
            1) M >= -10 and SST >= 296
            2) M >= -10 and Tadv >= -4

    Tie policy:
        - Start everything as 'Undetermined'
        - Assign 'Stratocumulus' only where cond_strat is True
          AND cond_open is False
        - Assign 'Open-Cell' only where cond_open is True
          AND cond_strat is False
        → Points that satisfy both or neither stay 'Undetermined'.
    """

    # Normalize campaign string a bit for safety
    campaign = str(campaign).upper()

    for val in fblks:
        block_list = fblks[val]

        for i in range(len(block_list)):
            block = block_list[i].copy()

            # Default: unknown / in-between regime
            block['cloud_regime'] = pd.Series('Undetermined',
                                              index=block.index,
                                              dtype='object')

            # =====================================================
            # SOCRATES rules
            # =====================================================
            if campaign == 'SOCRATES':
                # Required / optional fields
                M   = block['M']
                WS  = block.get('Wind_sp',
                                pd.Series(np.nan, index=block.index))
                EIS = block.get('EIS',
                                pd.Series(np.nan, index=block.index))
                WSH = block.get('Wind_shear',
                                pd.Series(np.nan, index=block.index))

                # Stratocumulus-favoring conditions
                cond_strat = (
                    ((M < -9)  & (WS  < 9)) |
                    ((M < -10) & (EIS > 7)) |
                    ((M < -10) & (WSH < 9))
                )

                # Open-cell-favoring conditions
                cond_open = (
                    ((M >= -7) & (WS  >= 9)) |
                    ((M >= -8) & (EIS < 9))  |
                    ((M >= -8) & (WSH > 6))
                )

                # Apply labels, with "Undetermined" winning ties
                block.loc[cond_strat & ~cond_open, 'cloud_regime'] = 'Stratocumulus'
                block.loc[cond_open  & ~cond_strat, 'cloud_regime'] = 'Open-Cell'

            # =====================================================
            # CSET rules
            # =====================================================
            elif campaign == 'CSET':
                M    = block['M']
                SST  = block.get('ERA5_SST',
                                 block.get('sst',
                                           pd.Series(np.nan, index=block.index)))
                Tadv = block.get('Tadv',
                                 pd.Series(np.nan, index=block.index))

                # Stratocumulus-favoring conditions
                cond_strat = (
                    ((M < -10) & (SST < 296)) |
                    ((M < -11) & (Tadv < 0))
                )

                # Open-cell-favoring conditions
                cond_open = (
                    ((M >= -10) & (SST >= 296)) |
                    ((M >= -10) & (Tadv >= -3))
                )

                # Apply labels, with "Undetermined" winning ties
                block.loc[cond_strat & ~cond_open, 'cloud_regime'] = 'Stratocumulus'
                block.loc[cond_open  & ~cond_strat, 'cloud_regime'] = 'Open-Cell'

            # Write back modified block
            block_list[i] = block

        # Update this entry in fblks
        fblks[val] = block_list

    return fblks

def plot_block_ts(dict,idx):

    # Assuming the four DataFrames are already created
    # profile_in_cloud, level_in_cloud, level_out_cloud_bl, level_out_cloud_fr
    # Dict = assign_flight_type(df)
    df = dict['DataFrame']
    # print(df)
    blocks = dict['flight_blocks']
    incloud = dict['Cloud_blocks']
    # Creating the four DataFrames based on flight_type, cloud status, and Location
    profile_in_cloud = df[(df['flight_type'] == 'profile') & (df['cloud_status'] == 'In-cloud')]
    level_in_cloud = df[(df['flight_type'] == 'level') & (df['cloud_status'] == 'In-cloud') & (df['Location'] != 'BL')]
    level_out_cloud_bl = df[(df['flight_type'] == 'level') & (df['cloud_status'] != 'In-cloud') & (df['Location'] == 'BL')]
    level_out_cloud_fr = df[(df['flight_type'] == 'level') & (df['cloud_status'] != 'In-cloud') & (df['Location'] == 'Free')]
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.7, 1])  # First subplot is 3x the width of the second

    # First subplot (larger)
    ax1 = fig.add_subplot(gs[0])  # Assigning first subplot

    # Plot flight altitude for each DataFrame in different colors
    ax1.plot(df['Time'], df['GGALT'], color='k', linewidth=3)

    # Plot In-cloud, Level In-cloud, and other data
    ax1.scatter(profile_in_cloud['Time'], profile_in_cloud['GGALT'], color='red', label='Profile In-cloud', marker='s', s=1, zorder=2)
    ax1.scatter(level_in_cloud['Time'], level_in_cloud['GGALT'], color='blue', label='Level In-cloud', marker='s', s=1, zorder=2)
    ax1.scatter(level_out_cloud_bl['Time'], level_out_cloud_bl['GGALT'], color='green', label='Level Out-of-cloud (BL)', marker='s', s=1, zorder=2)
    ax1.scatter(level_out_cloud_fr['Time'], level_out_cloud_fr['GGALT'], color='purple', label='Level Out-of-cloud (Free)', marker='s', s=1, zorder=2)

    # Set x-axis limits
    start_limit = df['Time'].min()
    end_limit = df['Time'].max()
    ax1.set_xlim([start_limit, end_limit])

    # Format x-axis to show only hours, minutes, and seconds
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))  # Adjust format as needed
    fig.autofmt_xdate()

    # Shade blocks according to their type (level or profile)
    for _, row in blocks.iterrows():
        start_time, end_time, flight_type = row['start_time'], row['end_time'], row['flight_type']
        if flight_type == 'level':
            ax1.axvspan(start_time, end_time, color='blue', alpha=0.3, label="Level leg" if 'Level leg' not in ax1.get_legend_handles_labels()[1] else None)
        elif flight_type == 'profile':
            ax1.axvspan(start_time, end_time, color='goldenrod', alpha=0.5, label="Profiling" if 'Profiling' not in ax1.get_legend_handles_labels()[1] else None)

    # Labels and title
    ax1.set_xlabel('Time UTC (MM-dd HH:mm)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Altitude Time Series separating into "Level legs" and "Profiles"')
    ax1.set_ylim(-200, np.max(df['GGALT'] + 600))
    ax1.legend(loc='upper right', ncol=6, markerscale=6,fontsize=8)
    ax1.grid(True)

    # Second subplot (smaller)
    ax2 = fig.add_subplot(gs[1])  # Assigning second subplot

    # Find best match for column names dynamically
    plwc_col = next((col for col in df.columns if 'PLWCD' in col), None) or \
           next((col for col in df.columns if 'PLWC' in col), None)
    concd_col = next((col for col in df.columns if 'CONCD' in col), None)

    # Scatter plot for concentration and altitude
    ax2.scatter(df[concd_col], df.GGALT, color='b', label='CDP', alpha=0.5, marker='^', s=4)

    # Log scale for x-axis
    ax2.set_xscale('log')
    ax2.set_xlabel('Conc (#/cm3)')
    ax2.set_ylabel('Altitude (meters)')
    ax2.grid(True)
    ax2.set_xlim(.01, 1000)

    # Add second x-axis on top
    ax2_top = ax2.twiny()
    ax2_top.plot(df[plwc_col], df.GGALT, color='orange', alpha=.7, linestyle='--', label='CDP LWC')
    ax2_top.set_xlabel('g/m3')
    ax2_top.set_xscale('log')
    ax2_top.set_xlim(0.0001, 10)
    ax2.set_ylim(-200, np.max(df['GGALT'] + 600))

    # Shade blocked altitude regions in ax2
    for i, row in incloud.iterrows():
        ax2.fill_betweenx(
            y=[row['lower_bound'], row['upper_bound']],  # Altitude range for shading
            x1=0.01,  # Left bound (min x-value)
            x2=1000,  # Right bound (max x-value)
            color='red', alpha=0.3, label="Cloud layer" if 'Cloud layer' not in ax2.get_legend_handles_labels()[1] else None
        )

    # Merge legends from both axes
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_top.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')  # Merge legends from both axes

    # Reorder the legends if needed
    all_lines = lines + lines2
    all_labels = labels + labels2
    if len(all_labels) >= 3:
        new_order = [0, 2, 1]  # Modify based on the desired order
        all_lines = [all_lines[i] for i in new_order]
        all_labels = [all_labels[i] for i in new_order]

    # Apply reordered legend
    ax2.legend(all_lines, all_labels, loc='upper right', markerscale=3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.12)  # Increase spacing between subplots

    # Save the figure
    rf_id = f"RF_{idx+1:02d}"
    filename = f"CSET_Altitude_Flight_Cloud_type{rf_id}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

   
def plot_hcr_cloud_type(df,Flight_blocks,idx):
    
    # Initialize figure and gridspec for plotting
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])  # Main altitude plot
    ax2 = fig.add_subplot(gs[1, 0])  # Echo Type plot
    
    tick_values = [14, 16, 18, 25, 30, 32, 34, 36, 38]
    
    # Update to use the new colormap interface
    spectral_cmap = plt.colormaps['Set1']  # Access colormap directly
    tick_to_color = {tick: spectral_cmap(i / len(tick_values)) for i, tick in enumerate(tick_values)}  # Map each tick to a color
       
    # Plot flight altitude for each DataFrame in different colors
    ax1.plot(df['Time'], df['GGALT'], color='k', linewidth=3)
    
    start_limit = df['Time'].min()
    end_limit = df['Time'].max()
    ax1.set_xlim([start_limit, end_limit])
    
    for val in Flight_blocks:
        block_type = Flight_blocks[val]
        for i in range(len(block_type)):
            block = block_type[i]
            start_time = block['Time'].iloc[0]
            end_time = block['Time'].iloc[-1]
            mean_echo_type = np.nanmean(block['Echo_Type'])
        
            # Find the closest tick value and corresponding color
            closest_tick = tick_values[np.argmin(np.ceil(np.abs(np.array(tick_values) - mean_echo_type)))]
            color = tick_to_color[closest_tick]
        
            # Shade region on ax1
            ax1.axvspan(start_time, end_time, color=color, alpha=0.8)
    
            # Plot scatter on ax2
            ax2.scatter(
                block.Time,
                np.zeros(len(block.Time)),
                c=block.Echo_Type,
                cmap='Set1',
                marker='s',
                s=4,
                vmin=min(tick_values),
                vmax=max(tick_values)
            )
    
    # start_time = pd.to_datetime("2018-01-16 01:30:00")
    # end_time = pd.to_datetime("2018-01-16 03:00:00")
    # ax1.set_xlim(start_time, end_time)
    ax1.set_ylabel('Altitude (m)')
    # Clean up ax2 to make it look like a color strip
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_yticks([])
    ax2.set_xlabel('Time UTC (MM-dd HH:mm)')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate()
    # Set x-axis limits
    
    ax2.set_xlim([start_limit, end_limit])
    
    # Define the tick values (bin labels)
    tick_values = [14, 16, 18, 25, 30, 32, 34, 36, 38]
    tick_labels = [
        "stratiform low",
        "stratiform mid",
        "stratiform high",
        "mixed",
        "convective",
        "conv. elevated",
        "conv. shallow",
        "conv. mid",
        "conv. deep"
    ]
    # We need to define edges for each bin; to get N blocks, we need N+1 boundaries
    bounds = list(range(len(tick_values) + 1))  # e.g., 0, 1, 2, ..., 9
    
    # Create a colormap with N colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(tick_values)))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Add vertical colorbar on the right
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=[ax1, ax2],
        orientation='vertical',
        ticks=np.arange(len(tick_values)) + 0.5,  # Tick in the center of each block
        pad=0.012
    )
    # Set category labels instead of numbers
    cbar.ax.set_yticklabels(tick_labels)
    # Set colorbar labels and title
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label("HCR Echo Type")
    # Turn ticks inside for all three axes
    for ax in [ax1, ax2]:
        ax.tick_params(direction='in', which='both', top=True, right=True)
    # # Show the plot
    # plt.show()

    # Save the figure
    rf_id = f"RF_{idx+1:02d}"
    # filename = f"SOCRATES_Altitude_Flight_HCR_Cloud_Echo_{rf_id}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

#==========================================================
# Functions for compositing dropsondes
#==========================================================

def collocate_ERA5_sonde(ds, df):
    """
    Collocate ERA5 gridded fields with dropsonde observations using
    nearest-neighbor matching in space and time.

    For each dropsonde observation, the function:
        1. Finds the nearest ERA5 grid point using a KDTree built from
           the ERA5 latitude/longitude grid.
        2. Finds the nearest ERA5 time step to the sonde observation time.
        3. Extracts ERA5 variables at that grid point and time.
        4. Appends those values as new columns to the dropsonde dataframe.

    Longitude matching is performed in a [-180°, 180°) coordinate system
    to avoid issues near the dateline.

    Parameters
    ----------
    ds : xarray.Dataset or dict-like
        ERA5 dataset containing the variables to collocate. Expected
        dimensions are ('time', 'latitude', 'longitude').

        Required variables:
            SST
            M
            omega700
            deltaT
            WS
            Wind_shear
            Tadv
            RH700
            EIS

    df : pandas.DataFrame
        Dropsonde dataframe containing observation coordinates and time.

        Required columns:
            GGLAT   : latitude (degrees)
            GGLON   : longitude (degrees)
            Time    : observation time (datetime-like)

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with additional columns containing collocated ERA5
        values at the nearest grid point and time:

            ERA5_SST
            M
            Omega700
            deltaT
            Wind_sp
            Wind_shear
            Tadv
            RH700
            EIS
    Notes
    -----
    - Spatial matching uses Euclidean distance in lat/lon space via
      scipy.spatial.cKDTree.
    - Temporal matching uses nearest neighbor in ERA5 time.
    - Rows with missing latitude, longitude, or time receive NaN values
      for all collocated variables.
    - ERA5 variables are loaded into memory as NumPy arrays for fast
      indexing.
    """
    
    # ---------------- core collocation block ----------------
    ds = xr.Dataset(ds)
    
    # ERA5 coords
    lat_vals = ds['latitude'].values.astype(float)
    lon_vals_ds = ds['longitude'].values.astype(float)
    
    # Build KDTree in [-180,180) to avoid wrap issues
    lon_vals_wrapped = wrap180(lon_vals_ds)
    lon_grid, lat_grid = np.meshgrid(lon_vals_wrapped, lat_vals, indexing="xy")  # (nx,ny) if xy; use ij below
    # use ij orientation for unravel consistency:
    YY, XX = np.meshgrid(lat_vals, lon_vals_wrapped, indexing="ij")  # (ny,nx)
    tree = cKDTree(np.c_[YY.ravel(), XX.ravel()])
    ny, nx = YY.shape  # (lat, lon)
    
    # ERA5 time (sorted)
    t_era = ds['time'].values.astype('datetime64[ns]')
    t_era_ns = t_era.view('int64')
    
    # Preload arrays into NumPy (T,Y,X) for fast indexing
    def arr3(name):
        return ds[name].transpose('time', 'latitude', 'longitude').compute().values
    
    arr = {
        'ERA5_SST':     arr3('SST'),
        'M':            arr3('M'),
        'Omega700':     arr3('omega700'),
        'deltaT':       arr3('deltaT'),
        'Wind_sp':      arr3('WS'),
        'Wind_shear':   arr3('Wind_shear'),
        'Tadv':         arr3('Tadv'),
        'RH700':        arr3('RH700'),
        'EIS':          arr3('EIS'),
    }
    
    # Flight coords/time
    flt_lat = df['GGLAT'].to_numpy(float)
    flt_lon = wrap180(df['GGLON'].to_numpy(float))  # match KDTree frame
    flt_t   = pd.to_datetime(df['Time'].values).to_numpy('datetime64[ns]')
    flt_t_ns = flt_t.view('int64')
    
    N = len(df)
    
    # Mask rows we can sample (ignore NaNs)
    valid = np.isfinite(flt_lat) & np.isfinite(flt_lon) & np.isfinite(flt_t_ns)
    
    # If nothing valid, just make the output columns full NaN and return df as-is
    if not np.any(valid):
        for out_name in arr.keys():
            df[out_name] = np.nan
    else:
        # Nearest time indices for valid rows only
        ti_valid = nearest_time_indices(t_era_ns, flt_t_ns[valid])  # (M,)
    
        # Nearest gridpoint for valid rows
        _, flat_idx = tree.query(np.c_[flt_lat[valid], flt_lon[valid]])  # (M,)
        yi, xi = np.unravel_index(flat_idx, (ny, nx))                    # (M,), (M,)
    
        # Gather each variable and scatter back into full-length columns (NaN elsewhere)
        for out_name, A in arr.items():   # A: (T,ny,nx)
            vals_valid = A[ti_valid, yi, xi]              # (M,)
            out_full = np.full(N, np.nan, dtype=float)    # default NaN
            out_full[valid] = vals_valid
            df[out_name] = out_full

    return df

def cloud_regime_sonde(df, campaign, min_valid=5):
    """
    Assign a single cloud_regime label to an entire dropsonde dataframe
    based on block-mean (NaN-excluded) cloud controlling factors.

    Parameters
    ----------
    df : pandas.DataFrame
        Dropsonde dataframe containing ERA5_SST, M, RH700, Tadv, Wind_sp,
        Wind_shear, EIS, etc.
    campaign : str
        'SOCRATES' or 'CSET'
    min_valid : int
        Minimum required valid samples when computing mean()

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with new column 'cloud_regime' filled with a single label
    """

    def mean_if_enough(x, nmin=min_valid):
        """Return mean(x) if enough valid samples exist, else NaN."""
        vals = x.to_numpy(dtype=float)
        return float(np.nanmean(vals)) if np.isfinite(vals).sum() >= nmin else np.nan

    # --- compute block means ---
    M_mean          = mean_if_enough(df.get("M"))
    RH700_mean      = mean_if_enough(df.get("RH700"))
    SST_mean        = mean_if_enough(df.get("ERA5_SST"))
    Tadv_mean       = mean_if_enough(df.get("Tadv"))
    Wind_sp_mean    = mean_if_enough(df.get("Wind_sp"))
    Wind_shear_mean = mean_if_enough(df.get("Wind_shear"))
    EIS_mean        = mean_if_enough(df.get("EIS"))

    # default label
    label = "Unknown"

    # ===========================
    #     SOCRATES RULE SET
    # ===========================
    if campaign.upper() == "SOCRATES":

        # --- Open-cell cumulus ---
        cond_open = (
            # 1) M >= -7 & WS >= 9
            (np.isfinite(M_mean) and np.isfinite(Wind_sp_mean) and
             M_mean >= -7 and Wind_sp_mean >= 9)
            or
            # 2) M >= -8 & EIS < 9
            (np.isfinite(M_mean) and np.isfinite(EIS_mean) and
             M_mean >= -8 and EIS_mean < 9)
            or
            # 3) M >= -8 & wshear > 6
            (np.isfinite(M_mean) and np.isfinite(Wind_shear_mean) and
             M_mean >= -8 and Wind_shear_mean > 6)
        )

        # --- Stratocumulus ---
        cond_strat = (
            # 1) M < -9 & WS < 9
            (np.isfinite(M_mean) and np.isfinite(Wind_sp_mean) and
             M_mean < -9 and Wind_sp_mean < 9)
            or
            # 2) M < -10 & EIS > 7
            (np.isfinite(M_mean) and np.isfinite(EIS_mean) and
             M_mean < -10 and EIS_mean > 7)
            or
            # 3) M < -10 & wshear < 9
            (np.isfinite(M_mean) and np.isfinite(Wind_shear_mean) and
             M_mean < -10 and Wind_shear_mean < 9)
        )

        if cond_open:
            label = "Open-Cell"
        if cond_strat:
            # Stratocumulus overrides if both are true
            label = "Stratocumulus"

    # ===========================
    #        CSET RULE SET
    # ===========================
    elif campaign.upper() == "CSET":

        cond_strat = (
            (np.isfinite(M_mean) and np.isfinite(SST_mean)  and M_mean < -10 and SST_mean < 295)
            or
            (np.isfinite(M_mean) and np.isfinite(Tadv_mean) and M_mean < -10 and Tadv_mean < 0)
        )

        cond_opencu = (
            (np.isfinite(M_mean) and np.isfinite(SST_mean)  and M_mean >= -10 and SST_mean >= 296)
            or
            (np.isfinite(M_mean) and np.isfinite(Tadv_mean) and M_mean >= -4)
        )

        if cond_strat:
            label = "Stratocumulus"
        if cond_opencu:
            label = "Open-Cell"

    # add a single label to the entire df
    df = df.copy()
    df["cloud_regime"] = label
    return df

