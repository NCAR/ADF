from pathlib import Path
import numpy as np
import yaml
import xarray as xr
import uxarray as ux
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import plotting_functions as pf
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
    - add option so that this works with a structured grid too 
    - make sure that climo's are being plotted with the preferred units 
    - add in observations (need to regrid/area weight)
    - need to figure out how to display the figures on the website

    """

    #Notify user that script has started:
    print("\n  Generating global mean time series plots...")

    # Gather ADF configurations
    # plot_loc = adfobj.get_basic_info('cam_diag_plot_loc')
    # plot_type = adfobj.read_config_var("diag_basic_info").get("plot_type", "png")
    plot_locations = adfobj.plot_location
    plot_type = adfobj.get_basic_info('plot_type')
    if not plot_type:
        plot_type = 'png'
    # res = adfobj.variable_defaults # will be dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    unstruct_plotting = adfobj.unstructured_plotting
    print("unstruct_plotting", unstruct_plotting)

    case_nickname = adfobj.get_cam_info('case_nickname') 
    base_nickname = adfobj.get_baseline_info('case_nickname') 

    region_list = adfobj.region_list
    regional_climo_var_list = ['GPP','ELAI','TSA','PREC','RAIN','SNOW', 'TOTRUNOFF',
                               'QOVER', 'QDRAI','QRGWL','QSNOFRZ','QSNOMELT',
                               'QSNWCPICE','ALBD']

    # ## Open file containing regions of interest 
    # nc_reg_file = '/glade/campaign/cgd/tss/people/oleson/FROM_LMWG/diag/lnd_diag4.2/code/resources/region_definitions.nc'
    # regionDS    = xr.open_dataset(nc_reg_file)
    # region_names = [str(item).split('b')[1] for item in regionDS.PTITSTR.values]

    ## Open observations YML here? 

    ## Read regions from yml file:
    ymlFilename = 'lib/regions_lnd.yaml'
    with open(ymlFilename, 'r') as file:
        regions = yaml.safe_load(file)

    # # I want to get the indices that match the reqeusted regions now...
    # region_indexList = []
    # cleaned_candidates = [s.strip("'\"") for s in region_names]
    # cleaned_candidates = [s.strip(" ") for s in cleaned_candidates]

    # # Fix some region names I've broken
    # cleaned_candidates = rename_region(cleaned_candidates, 'Western Si', 'Western Siberia')
    # cleaned_candidates = rename_region(cleaned_candidates, 'Eastern Si', 'Eastern Siberia')
    # cleaned_candidates = rename_region(cleaned_candidates, 'Ara', 'Arabian Peninsula')
    # cleaned_candidates = rename_region(cleaned_candidates, 'Sahara and Ara', 'Sahara and Arabia')
    # cleaned_candidates = rename_region(cleaned_candidates, 'Ti', 'Tibetan Plateau')

    # for iReg in region_list: 
    #     match_indices = [i for i, region in enumerate(cleaned_candidates) if iReg == region]
    #     region_indexList = np.append(region_indexList, match_indices)
    # region_indexList =region_indexList.astype('int')
    

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

    base_data = {}
    case_data = {}
    # First, load all variable data once (instead of inside nested loops)
    for field in regional_climo_var_list:
        # Load the global climatology for this variable
        base_data[field] = adfobj.data.load_reference_climo_da(baseline_name, field, **kwargs)
        case_data[field] = adfobj.data.load_climo_da(case_name, field, **kwargs)

        if type(base_data[field]) is type(None):
            print('Missing file for ', field)
            continue
        else:
            mdataset         = adfobj.data.load_climo_dataset(case_name, field, **kwargs) 
            area             = mdataset.area
            landfrac         = mdataset.landfrac
            # calculate weights
            wgt = area * landfrac / (area * landfrac).sum()  

    # Loop over regions for selected variable 
    for iReg in range(len(region_list)): 
        # regionDS_thisRg = regionDS.isel(region=region_indexList[iReg])
        box_west, box_east, box_south, box_north = get_region_boundaries(regions, region_list[iReg])
        ## Set up figure 
        # fig,axs = plt.subplots(4,5, figsize=(15,10))
        ## TODO: Make the plot size/number of subplots resopnsive to number of fields specified 
        fig,axs = plt.subplots(4,4, figsize=(18,12))
        axs = axs.ravel()
        
        plt_counter = 1
        for field in regional_climo_var_list:
            mdataset         = adfobj.data.load_climo_dataset(case_name, field, **kwargs) 

            if type(base_data[field]) is type(None):
                continue
            else:
                # TODO: handle regular gridded case
                base_var,wgt_sub  = getRegion_uxarray(uxgrid, base_data, field, wgt,
                                        box_west, box_east, 
                                        box_south, box_north)
                base_var_wgtd = np.sum(base_var * wgt_sub, axis=-1) / np.sum(wgt_sub)
            
                case_var,wgt_sub = getRegion_uxarray(uxgrid, case_data, field, wgt,
                                        box_west, box_east, 
                                        box_south, box_north)
                case_var_wgtd = np.sum(case_var * wgt_sub, axis=-1) / np.sum(wgt_sub)

            ## Plot the map:
            if plt_counter==1:
                ## Define region in first subplot
                fig.delaxes(axs[0])
                
                transform  = ccrs.PlateCarree()
                projection = ccrs.PlateCarree()
                base_var_mask = base_var.isel(time=0)
                base_var_mask[np.isfinite(base_var_mask)]=1
                collection = base_var_mask.to_polycollection()
                
                collection.set_transform(transform)
                collection.set_cmap('rainbow_r')
                collection.set_antialiased(False)
                map_ax = fig.add_subplot(4, 4, 1, projection=ccrs.PlateCarree())
                
                map_ax.coastlines()
                map_ax.add_collection(collection)
                map_ax.set_global()
                # Add map extent selection
                if region_list[iReg]=='N Hemisphere Land':
                    map_ax.set_extent([-180, 179, -3, 90],crs=ccrs.PlateCarree())
                elif region_list[iReg]=='Global':
                    map_ax.set_extent([-180, 179, -89, 90],crs=ccrs.PlateCarree())
                elif region_list[iReg]=='S Hemisphere Land':
                    map_ax.set_extent([-180, 179, -89, 3],crs=ccrs.PlateCarree())
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

                    
            ## Plot the timeseries 
            if type(base_data[field]) is type(None):
                # print('Missing file for ', field)
                continue
            else:
                axs[plt_counter].plot(np.arange(12)+1, case_var_wgtd,label=case_nickname)
                axs[plt_counter].plot(np.arange(12)+1, base_var_wgtd,label=base_nickname)
                axs[plt_counter].set_title(field)
                axs[plt_counter].set_ylabel(base_data[field].units)
                axs[plt_counter].legend()
                

            plt_counter = plt_counter+1

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # Save out figure
        # fileFriendlyRegionName = 
        plot_loc  = Path(plot_locations[0]) / f'RegionalClimo_{region_list[iReg]}_RegionalClimo_Mean.{plot_type}'
        #Set path for variance figures:
        # plot_loc = Path(plot_locations[0]) / f'RegionalClimo_{region_list[iReg]}.{plot_type}'
        # print(plot_loc)
        # plot_name = plot_loc+'RegionalClimo_'+region_names[iReg]+'.png'

#        Check redo_plot. If set to True: remove old plots, if they already exist:
        if (not redo_plot) and plot_loc.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_loc}' exists and clobber is false.")
            adfobj.add_website_data(plot_loc, "RegionalClimo", None, season=region_list[iReg], multi_case=True, non_season=True, plot_type = "RegionalClimo")

            #Continue to next iteration:
            return
        elif (redo_plot):
            if plot_loc.is_file():
                plot_loc.unlink()    

        fig.savefig(plot_loc, bbox_inches='tight', facecolor='white')
        plt.close() 

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_loc, "RegionalClimo", None, season=region_list[iReg], multi_case=True, non_season=True, plot_type = "RegionalClimo")

    return 

def getRegion_uxarray(gridDS, varDS, varName, wgt, BOX_W, BOX_E, BOX_S, BOX_N): 
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
    wgt_subset    = wgt.isel(n_face=node_indices)
    # area_subset   = varDS['area'].isel(n_face=node_indices)
    # lf_subset     = varDS['landfrac'].isel(n_face=node_indices)
    
    return domain_subset,wgt_subset

def get_region_boundaries(regions, region_name):
    """Get the boundaries of a specific region."""
    if region_name not in regions:
        raise ValueError(f"Region '{region_name}' not found in regions dictionary")
    
    region = regions[region_name]
    south, north = region['lat_bounds']
    west, east = region['lon_bounds']
    
    return west, east, south, north
# def rename_region(DS, searchStr, replaceStr):
#     iReplace = np.where(np.asarray(DS)==searchStr)[0]
#     if len(iReplace)==1:
#         DS[int(iReplace)] = replaceStr
#     elif len(iReplace>1): 
#         # This happens with Tibetan Plateau; there are two defined
#         # Indices 31 and 35 
#         # Same values, but Box_W and Box_E are swapped.. going to keep the first
#         DS[int(iReplace[0])] = replaceStr
#         # print('Found more than one match for ',searchStr)
#         # print(iReplace)
        
#     return DS