from pathlib import Path
import numpy as np
import yaml
import xarray as xr
import uxarray as ux
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings  # use to warn user about missing files.

def my_formatwarning(msg, *args, **kwargs):
    """custom warning"""
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = my_formatwarning


def regional_climatology(adfobj):

    """
    load climo file, subset for each region and each var
    Make a combined plot, save it, add it to website.

    NOTES (from Meg): There are still a lot of to-do's with this script! 
    - convert region defintion netCDF file to a yml, read that in instead 
    - increase number of variables that have a climo plotted; i've just 
      added two, but left room for more in the subplots 
    - check that all varaibles have climo files; likely to break otherwise
    - add option so that this works with a structured grid too     # ...existing code...
    if found:
        #Check if observations dataset name is specified:
        if "obs_name" in default_var_dict:
            obs_name = default_var_dict["obs_name"]
        else:
            obs_name = obs_file_path.name
    
        if "obs_var_name" in default_var_dict:
            obs_var_name = default_var_dict["obs_var_name"]
        else:
            obs_var_name = field
    
        # Use the resolved obs_file_path, not the original string
        obs_data[field] = xr.open_mfdataset([str(obs_file_path)], combine="by_coords")
        plot_obs[field] = True
    # ...existing code...
    - make sure that climo's are being plotted with the preferred units 
    - add in observations (need to regrid/area weight)
    - need to figure out how to display the figures on the website

    """

    #Notify user that script has started:
    print("\n  --- Generating regional climatology plots... ---")

    # Gather ADF configurations
    # plot_loc = adfobj.get_basic_info('cam_diag_plot_loc')
    # plot_type = adfobj.read_config_var("diag_basic_info").get("plot_type", "png")
    plot_locations = adfobj.plot_location
    plot_type = adfobj.get_basic_info('plot_type')
    if not plot_type:
        plot_type = 'png'
    #res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    unstruct_plotting = adfobj.unstructured_plotting
    print(f"\t     unstruct_plotting", unstruct_plotting)

    case_nickname = adfobj.get_cam_info('case_nickname') 
    base_nickname = adfobj.get_baseline_info('case_nickname') 

    region_list = adfobj.region_list
    #TODO, make it easier for users decide on these?
    regional_climo_var_list = ['TSA','PREC','ELAI',
                               'FSDS','FLDS','SNOWDP','ASA',
                               'FSH','QRUNOFF_TO_COUPLER','ET','FCTR',
                               'GPP','TWS','FCEV','FAREA_BURNED',
                               ]

    ## Open observations YML here? 

    ## Read regions from yml file:
    ymlFilename = 'lib/regions_lnd.yaml'
    with open(ymlFilename, 'r') as file:
        regions = yaml.safe_load(file)

    # Extract variables:
    baseline_name        = adfobj.get_baseline_info("cam_case_name", required=True)
    input_climo_baseline = Path(adfobj.get_baseline_info("cam_climo_loc", required=True))
    # TODO hard wired for single case name:
    case_name        = adfobj.get_cam_info("cam_case_name", required=True)[0]
    input_climo_case = Path(adfobj.get_cam_info("cam_climo_loc", required=True)[0]) 

    # Get grid file 
    mesh_file = adfobj.mesh_files["baseline_mesh_file"]
    uxgrid = ux.open_grid(mesh_file)

    # Set keywords
    kwargs = {}
    kwargs["mesh_file"] = mesh_file
    kwargs["unstructured_plotting"] = unstruct_plotting

    #Determine local directory:
    _adf_lib_dir = adfobj.get_basic_info("obs_data_loc")

    #Determine whether to use adf defaults or custom:
    _defaults_file = adfobj.get_basic_info('defaults_file')
    if _defaults_file is None:
        _defaults_file = _adf_lib_dir/'adf_variable_defaults.yaml'
    else:
        print(f"\n\t Not using ADF default variables yaml file, instead using {_defaults_file}\n")
    #End if

    #Open YAML file:
    with open(_defaults_file, encoding='UTF-8') as dfil:
        adfobj.__variable_defaults = yaml.load(dfil, Loader=yaml.SafeLoader)

    _variable_defaults = adfobj.__variable_defaults

    #-----------------------------------------
    #Extract the "obs_data_loc" default observational data location:
    obs_data_loc = adfobj.get_basic_info("obs_data_loc")

    base_data = {}
    case_data = {}
    obs_data = {}
    obs_name = {}
    obs_var_name = {}
    plot_obs = {}

    var_obs_dict = adfobj.var_obs_dict
    
    # First, load all variable data once (instead of inside nested loops)
    for field in regional_climo_var_list:
        # Load the global climatology for this variable
        # TODO unit conversions are not handled consistently here
        base_data[field] = adfobj.data.load_reference_climo_da(baseline_name, field, **kwargs)
        case_data[field] = adfobj.data.load_climo_da(case_name, field, **kwargs)

        if type(base_data[field]) is type(None):
            print('Missing file for ', field)
            continue
        else:
            # get area and landfrac for base and case climo datasets
            mdataset      = adfobj.data.load_climo_dataset(case_name, field, **kwargs) 
            area_c        = mdataset.area.isel(time=0) # drop time dimension to avoid confusion
            landfrac_c    = mdataset.landfrac.isel(time=0)
            # Redundant, but we'll do this for consistency:
            # TODO, won't handle loadling the basecase this way
            #area_b = adfobj.data.load_reference_climo_da(baseline_name, 'area', **kwargs)
            #landfrac_b = adfobj.data.load_reference_climo_da(baseline_name, 'landfrac', **kwargs)

            mdataset_base   = adfobj.data.load_reference_climo_dataset(baseline_name, field, **kwargs)
            area_b          = mdataset_base.area.isel(time=0)
            landfrac_b      = mdataset_base.landfrac.isel(time=0)
 
            # calculate weights 
            # WW: 1) should actual weight calculation be done after subsetting to region?
            #     2) Does this work as intended for different resolutions?
            # wgt = area * landfrac # / (area * landfrac).sum()  

        #-----------------------------------------
        # Now, check if observations are to be plotted for this variable
        plot_obs[field] = False 
        if field in _variable_defaults:
            # Extract variable-obs dictionary
            default_var_dict = _variable_defaults[field]

            #Check if an observations file is specified:
            if "obs_file" in default_var_dict:
                #Set found variable:
                found = False

                #Extract path/filename:
                obs_file_path = Path(default_var_dict["obs_file"])

                #Check if file exists:
                if not obs_file_path.is_file():
                    #If not, then check if it is in "obs_data_loc"
                    if obs_data_loc:
                        obs_file_path = Path(obs_data_loc)/obs_file_path

                        if obs_file_path.is_file():
                            found = True

                else:
                    #File was found:
                    found = True
                #End if

                #If found, then set observations dataset and variable names:
                if found:
                    #Check if observations dataset name is specified:
                    if "obs_name" in default_var_dict:
                        obs_name[field] = default_var_dict["obs_name"]
                    else:
                        #If not, then just use obs file name:
                        obs_name[field] = obs_file_path.name

                    #Check if observations variable name is specified:
                    if "obs_var_name" in default_var_dict:
                        #If so, then set obs_var_name variable:
                        obs_var_name[field] = default_var_dict["obs_var_name"]
                    else:
                        #Assume observation variable name is the same as model variable:
                        obs_var_name[field] = field
                    #End if
                    #Finally read in the obs!
                    obs_data[field] = xr.open_mfdataset([default_var_dict["obs_file"]], combine="by_coords")
                    plot_obs[field] = True
                    # Special handling for some variables:, NOT A GOOD HACK! 
                    # TODO: improve this!
                    if (field == 'ASA') and ('BRDALB' in obs_data[field].variables):
                        obs_data[field]['BRDALB'] = obs_data[field]['BRDALB'].swap_dims({'lsmlat':'lat','lsmlon':'lon'})

                else:
                    #If not found, then print to log and skip variable:
                    msg = f'''Unable to find obs file '{default_var_dict["obs_file"]}' '''
                    msg += f"for variable '{field}'."
                    adfobj.debug_log(msg)
                    continue
                # End if

            else:
                #No observation file was specified, so print to log and skip variable:
                adfobj.debug_log(f"No observations file was listed for variable '{field}'.")
                continue
        else:
            #Variable not in defaults file, so print to log and skip variable:
            msg = f"Variable '{field}' not found in variable defaults file: `{_defaults_file}`"
            adfobj.debug_log(msg)
        # End if
        # End of observation loading
    #-----------------------------------------

    #-----------------------------------------
    # Loop over regions for selected variable 
    for iReg in range(len(region_list)):
        print(f"\n\t - Plotting regional climatology for: {region_list[iReg]}") 
        # regionDS_thisRg = regionDS.isel(region=region_indexList[iReg])
        box_west, box_east, box_south, box_north, region_category = get_region_boundaries(regions, region_list[iReg])
        ## Set up figure 
        ## TODO: Make the plot size/number of subplots resopnsive to number of fields specified 
        fig,axs = plt.subplots(4,4, figsize=(18,12))
        axs = axs.ravel()
        
        plt_counter = 1
        for field in regional_climo_var_list:
            mdataset = adfobj.data.load_climo_dataset(case_name, field, **kwargs) 

            if type(base_data[field]) is type(None):
                continue
            else:
                if unstruct_plotting == True:
                    # uxarray output is time*nface, sum over nface
                    base_var,wgt_sub  = getRegion_uxarray(uxgrid, base_data, field, area_b, landfrac_b,
                                                        box_west, box_east, 
                                                        box_south, box_north)
                    base_var_wgtd = np.sum(base_var * wgt_sub, axis=-1) # WW not needed?/ np.sum(wgt_sub)
                
                    case_var,wgt_sub = getRegion_uxarray(uxgrid, case_data, field, area_c, landfrac_c,
                                                        box_west, box_east, 
                                                        box_south, box_north)
                    case_var_wgtd = np.sum(case_var * wgt_sub, axis=-1) #/ np.sum(wgt_sub)
            
                else:  # regular lat/lon grid
                    # xarray output is time*lat*lon, sum over lat/lon
                    base_var, wgt_sub = getRegion_xarray(base_data[field], field,
                                            box_west, box_east, 
                                            box_south, box_north,
                                            area_b, landfrac_b)
                    base_var_wgtd = np.sum(base_var * wgt_sub, axis=(1,2))

                    case_var, wgt_sub = getRegion_xarray(case_data[field], field,
                                            box_west, box_east, 
                                            box_south, box_north,
                                            area_c, landfrac_c)
                    case_var_wgtd = np.sum(case_var * wgt_sub, axis=(1,2))                

                # Read in observations, if available
                if plot_obs[field] == True:
                    # obs output is time*lat*lon, sum over lat/lon
                    obs_var, wgt_sub = getRegion_xarray(obs_data[field], field,
                                            box_west, box_east, 
                                            box_south, box_north,
                                            obs_var_name=obs_var_name[field])
                    obs_var_wgtd = np.sum(obs_var * wgt_sub, axis=(1,2)) #/ np.sum(wgt_sub) 

            ## Plot the map:
            if plt_counter==1:
                ## Define region in first subplot
                fig.delaxes(axs[0])
                
                transform  = ccrs.PlateCarree()
                projection = ccrs.PlateCarree()
                base_var_mask = base_var.isel(time=0)

                if unstruct_plotting == True:
                    base_var_mask[np.isfinite(base_var_mask)]=1
                    collection = base_var_mask.to_polycollection()
                    
                    collection.set_transform(transform)
                    collection.set_cmap('rainbow_r')
                    collection.set_antialiased(False)
                    map_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())
                    
                    map_ax.coastlines()
                    map_ax.add_collection(collection)
                elif unstruct_plotting == False:
                    base_var_mask = base_var_mask.copy()
                    base_var_mask.values[np.isfinite(base_var_mask.values)] = 1
                    map_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())
                    map_ax.coastlines()
                    
                    #print('debug mask.lon')
                    #print(base_var_mask.lon)
                    #print('debug mask.lat')
                    #print(base_var_mask.lat)

                    # Plot using pcolormesh for structured grids
                    im = map_ax.pcolormesh(base_var_mask.lon, base_var_mask.lat, 
                                        base_var_mask.values,
                                        transform=transform, 
                                        cmap='rainbow_r',
                                        shading='auto')

                # Add map extent selection
                if region_list[iReg]=='N Hemisphere Land':
                    map_ax.set_extent([-180, 179, -3, 90],crs=ccrs.PlateCarree())
                elif region_list[iReg]=='Global':
                    map_ax.set_extent([-180, 179, -89, 90],crs=ccrs.PlateCarree())
                elif region_list[iReg]=='S Hemisphere Land':
                    map_ax.set_extent([-180, 179, -89, 3],crs=ccrs.PlateCarree())
                elif region_list[iReg]=='Polar':
                    map_ax.set_extent([-180, 179, 45, 90],crs=ccrs.PlateCarree())
                else: 
                    if ((box_south >= 30) & (box_east<=-5) ):
                        map_ax.set_extent([-180, -5, 30, 90],crs=ccrs.PlateCarree())
                    elif ((box_south >= 30) & (box_east>=-5) ):
                        map_ax.set_extent([-5, 179, 30, 90],crs=ccrs.PlateCarree())
                    elif ((box_south <= 30) & (box_south >= -30) & 
                        (box_east<=-5) ):
                        map_ax.set_extent([-180, -5, -30, 30],crs=ccrs.PlateCarree())
                    elif ((box_south <= 30) & (box_south >= -30) & 
                        (box_east>=-5) ):
                        map_ax.set_extent([-5, 179, -30, 30],crs=ccrs.PlateCarree())
                    elif ((box_south <= -30) & (box_south >= -60) &
                        (box_east>=-5) ):
                        map_ax.set_extent([-5, 179, -89, -30],crs=ccrs.PlateCarree())
                    elif ((box_south <= -30) & (box_south >= -60) &
                        (box_east<=-5) ):
                        map_ax.set_extent([-180, -5, -89, -30],crs=ccrs.PlateCarree())
                    elif ((box_south <= -60)):
                        map_ax.set_extent([-180, 179, -89, -60],crs=ccrs.PlateCarree())
                # End if for plotting map extent
                    
            ## Plot the climatology: 
            if type(base_data[field]) is type(None):
                print('Missing file for ', field)
                continue
            else:
                axs[plt_counter].plot(np.arange(12)+1, case_var_wgtd,
                                      label=case_nickname, linewidth=2)
                axs[plt_counter].plot(np.arange(12)+1, base_var_wgtd,
                                      label=base_nickname, linewidth=2)
                if plot_obs[field] == True:
                    axs[plt_counter].plot(np.arange(12)+1, obs_var_wgtd,
                                          label=obs_name[field], color='black', linewidth=2)
                axs[plt_counter].set_title(field)
                axs[plt_counter].set_ylabel(base_data[field].units)
                axs[plt_counter].set_xticks(np.arange(1, 13, 2))
                axs[plt_counter].legend()
                

            plt_counter = plt_counter+1

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Save out figure
        plot_loc = Path(plot_locations[0]) / f'{region_list[iReg]}_plot_RegionalClimo_Mean.{plot_type}'

        # Check redo_plot. If set to True: remove old plots, if they already exist:
        if (not redo_plot) and plot_loc.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_loc}' exists and clobber is false.")
            adfobj.add_website_data(plot_loc, region_list[iReg], None, season=None, multi_case=True, 
                                     category=region_category, non_season=True, plot_type = "RegionalClimo")

            #Continue to next iteration:
            return
        elif (redo_plot):
            if plot_loc.is_file():
                plot_loc.unlink()    

        fig.savefig(plot_loc, bbox_inches='tight', facecolor='white')
        plt.close() 

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc, region_list[iReg], None, season=None, multi_case=True, 
                                non_season=True, category=region_category, plot_type = "RegionalClimo")

    return 

print("\n  --- Regional climatology plots generated successfully! ---")

def getRegion_uxarray(gridDS, varDS, varName, area, landfrac, BOX_W, BOX_E, BOX_S, BOX_N): 
    # Method 2: Filter mesh nodes based on coordinates
    node_lons = gridDS.face_lon
    node_lats = gridDS.face_lat
    
    # Create a boolean mask for nodes within your domain
    in_domain = ((node_lons >= BOX_W) & (node_lons <= BOX_E) & 
                 (node_lats >= BOX_S) & (node_lats <= BOX_N))
    
    # Get the indices of nodes within your domain
    node_indices = np.where(in_domain)[0]
    
    # Subset the dataset using these node indices
    domain_subset = varDS[varName].isel(n_face=node_indices)
    area_subset = area.isel(n_face=node_indices)
    landfrac_subset = landfrac.isel(n_face=node_indices)
    wgt_subset = area_subset * landfrac_subset  / (area_subset* landfrac_subset).sum() 

    return domain_subset,wgt_subset

def getRegion_xarray(varDS, varName,
                     BOX_W, BOX_E, BOX_S, BOX_N,
                     area=None, landfrac=None,
                     obs_var_name=None):
    # Assumes regular lat/lon grid in xarray Dataset
    # Assumes varDS has 'lon' and 'lat' coordinates w/ lon in [0,360]
    # Convert BOX_W and BOX_E to [0,360] if necessary
    # Also assumes global weights have already been calculated & masked appropriately 
    if (BOX_W == -180) & (BOX_E == 180):
        BOX_W, BOX_E = 0, 360  # Special case for global domain
    if BOX_W < 0: BOX_W = BOX_W + 360
    if BOX_E < 0: BOX_E = BOX_E + 360   

    if varName not in varDS:
        varName = obs_var_name

    # TODO is there a less brittle way to do this?
    if (area is not None) and (landfrac is not None):
        weight = area * landfrac
    elif ('weight' in varDS) and ('datamask' in varDS):
        weight = varDS['weight'] * varDS['datamask']
    elif ('weight' in varDS) and ('LANDFRAC' in varDS): 
        #used for MODIS albedo product
        weight = varDS['weight'] * varDS['LANDFRAC']
    elif 'area' in varDS and 'landfrac' in varDS:
        weight = varDS['area'] * varDS['landfrac']
    elif 'area' in varDS and 'landmask' in varDS:
        weight = varDS['area'] * varDS['landmask']
        # Fluxnet data also has a datamask
        if 'datamask' in varDS:
            weight = weight * varDS['datamask']  
    else:
        raise ValueError("No valid weight, area, or landmask found in {varName} dataset.")
    
    # check we have a data array for the variable
    if isinstance(varDS, xr.Dataset):
        varDS = varDS[varName]

    # Subset the dataarray using the specified box 
    if BOX_W < BOX_E:
        domain_subset = varDS.sel(lat=slice(BOX_S, BOX_N),
                                  lon=slice(BOX_W, BOX_E))
        weight_subset = weight.sel(lat=slice(BOX_S, BOX_N),
                                   lon=slice(BOX_W, BOX_E)) 

    else:
        # Use boolean indexing to select the region
        # The parentheses are important due to operator precedence
        west_of_0 = varDS.lon >= BOX_W
        east_of_0 = varDS.lon <= BOX_E
        domain_subset = varDS.sel(lat=slice(BOX_S, BOX_N),
                                  lon=(west_of_0 | east_of_0))
        weight_subset = weight.sel(lat=slice(BOX_S, BOX_N),
                                   lon=(west_of_0 | east_of_0))

    wgt_subset = weight_subset / weight_subset.sum() 
    weight_subset = weight.sel
    return domain_subset,wgt_subset

def get_region_boundaries(regions, region_name):
    """Get the boundaries of a specific region."""
    if region_name not in regions:
        raise ValueError(f"Region '{region_name}' not found in regions dictionary")
    
    region = regions[region_name]
    south, north = region['lat_bounds']
    west, east = region['lon_bounds']
    region_category = region['region_category'] if 'region_category' in region else None  
    
    return west, east, south, north, region_category
