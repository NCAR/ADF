import numpy as np
import xarray as xr
from pathlib import Path

from datetime import datetime
import numpy as np
import itertools
import pandas as pd

# Import necessary ADF modules:
from adf_base import AdfError

def depositions_table(adfobj, Climate=None,**kwargs):
    '''
    Calculate depositions table for all the availabe species

        HCN_DRYDEP (Tg/yr), HCN_WETDEP_elevated (Tg/yr)
        


    List of variable names and descriptions for clarity
    ---------------------------------------------------
        - ListVars: list of all available variables from given history file



    MODIFICATION HISTORY:
        Behrooz Roozitalab, 24, APR, 2026: VERSION 1.00
        - Initial version, based on emissions_table.py

    '''

    #Notify user that script has started:
    msg = "\n   Calculating depositions table..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")


    # Check which type of tables to be created, default to 'False'
    if Climate is None:
        Climate = False


    # Inputs
    #-------
    # Variable defaults info
    res = adfobj.variable_defaults # dict of variable-specific plot preferences
    bres = res['budget_tables']



    # For the case that outputs are saved for a specific region.
    # i.e., when using fincllonlat in user_nl_cam
    ext1_SE = bres['ext1_SE']


    ### NOT WORKING FOR NOW
    # To calculate the budgets only for a region
    # Lat/Lon extent
    limit = bres['limit']
    regional = bres['regional']

    # Dictionary for Molecular weights. Keys must be consistent with variable name
    # For aerosols, the MW is used only for chemical loss, chemical production, and elevated emission calculations
    # For SO4, we report everything in terms of Sulfur, so we use Sulfur MW here
    MW = bres['MW']


                
    # Avogadro's Number
    AVO = float(bres['AVO'])
    # gravity
    gr = float(bres['gr'])
    # Mw air
    Mwair = float(bres['Mwair'])

    # The variables in the list below must be aerosols - do not add AOD and DAOD
    # no need to change this list, unless for a specific need!
    AEROSOLS = bres['AEROSOLS']

    # Start gathering case, path, and data info
    #-----------------------------------------

    # CAM simulation variables (these quantities are always lists):
    case_names = adfobj.get_cam_info('cam_case_name', required=True)

    # Grab all case nickname(s)
    test_nicknames_list = adfobj.case_nicknames["test_nicknames"]
    nicknames_list = test_nicknames_list
    # Grab climo years
    start_years = adfobj.climo_yrs["syears"]
    end_years = adfobj.climo_yrs["eyears"]

    #Grab history strings:
    hist_strs = adfobj.hist_string["test_hist_str"]

    # Grab history file locations from config yaml file
    hist_locs = adfobj.get_cam_info("cam_hist_loc", required=True)
  
    # Check if this is test model vs baseline model
    # If so, update test case(s) lists created above
    if not adfobj.compare_obs:
        # Get baseline case info
        case_names += [adfobj.get_baseline_info("cam_case_name")]
        nicknames_list += [adfobj.case_nicknames["base_nickname"]]

        # Grab climo years
        start_years += [adfobj.climo_yrs["syear_baseline"]]
        end_years += [adfobj.climo_yrs["eyear_baseline"]]

        # Get history file info
        hist_strs += [adfobj.hist_string["base_hist_str"]]
        hist_locs += [adfobj.get_baseline_info("cam_hist_loc")]
    # End if
    
    # Check to ensure number of case names matches number history file locations.
    # If not, exit script
    if len(hist_locs) != len(case_names):
        errmsg = "Error: number of cases does not match number of history file locations. Script is exiting."
        raise AdfError(errmsg)

    # Initialize nicknames dictionary
    #nicknames = {}

    # Filter the list to include only strings that are possible h0 strings
    # - Search for either h0 or h0a
    substrings = {"cam.h0","cam.h0a","cam.hm"}
    case_hist_strs = []
    for cam_case_str in hist_strs:
        # Check each possible h0 string
        for string in cam_case_str:
            if string in substrings:
                case_hist_strs.append(string)
                break

    # Create path object for the CAM history file(s) location:
    data_dirs = []
    for case_idx,case in enumerate(nicknames_list):

        print(f"\t Looking for history location: {hist_locs[case_idx]}")


        #Check that history file input directory actually exists:
        if (hist_locs[case_idx] is None) or (not Path(hist_locs[case_idx]).is_dir()):
            errmsg = f"History files directory '{hist_locs[case_idx]}' not found.  Script is exiting."
            raise AdfError(errmsg)

        #Write to debug log if enabled:
        adfobj.debug_log(f"DEBUG: location of history files is {str(hist_locs[case_idx])}")
        # Update list for found directories
        data_dirs.append(hist_locs[case_idx])

    # End gathering case, path, and data info
    #-----------------------------------------
    # Periods of Interest
    # -------------------
    # choose the period of interest. Plots will be averaged within this period
    durations = {}
    num_yrs = {}

    # Main function
    #--------------
    # Set dictionary of components for each case
    Dic_scn_var_comp = {}
    areas = {}
    trops = {}
    insides = {}
    for i,case in enumerate(nicknames_list):

        start_year = start_years[i]
        end_year = end_years[i] + 1
        start_date = f"{start_year:04d}-1-1"
        end_date = f"{end_year:04d}-1-1"
        
        # Create time periods
        start_period = datetime.strptime(start_date, "%Y-%m-%d")
        end_period = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculated duration of time period in seconds?
        if Climate:
            # Use annual mean file
            durations[case] = 365*86400
            num_yrs[case]=1
    
        else:
            # Calculate the duration
            durations[case] = (end_period-start_period).days*86400 #+365*86400
            # Get number of years for calculations
            num_yrs[case] = (int(end_year)-int(start_year)) #+1


        # Calculated duration of time period in seconds?
        #durations[case] = (end_period-start_period).days*86400 #+365*86400


        # Get number of years for calculations
        #num_yrs[case] = (int(end_year)-int(start_year)) #+1

        # Get currenty history file directory
        data_dir = data_dirs[i]

        # Get all files, lats, lons, and area weights for current case
        Files,Lats,Lons,areas[case],ext1_SE = Get_files(adfobj,data_dir,start_year,end_year,case_hist_strs[i],area=True)
        # find the name of all the variables in the file.
        # this will help the code to work for the variables that are not in the files (assingn 0s)
        tmp_file = xr.open_dataset(Path(data_dir) / Files[0])
        ListVars = list(tmp_file.variables)


        #list of all the variables to be caculated.                    
        VARIABLES=[]
        for v in ListVars:
            if v[:3]=='WD_':
                    VARIABLES.append(v[3:])
            elif v[:3]=='DF_':
                    VARIABLES.append(v[3:])   
         
        VARIABLES+=['dst_a1','dst_a2','dst_a3','ncl_a1','ncl_a2','ncl_a3','pom_a1','pom_a4','bc_a1',
        'bc_a4','so4_a1','so4_a2','so4_a3','so4_a5','soa1_a1','soa2_a1','soa3_a1','soa4_a1',
        'soa5_a1','soa1_a2','soa2_a2','soa3_a2','soa4_a2','soa5_a2']
        VARIABLES=set(VARIABLES)



        
        
        # Set up and fill dictionaries for components for current cases
        dic_SE = set_dic_SE(ListVars,ext1_SE,VARIABLES)
        dic_SE = fill_dic_SE_Dep(adfobj, dic_SE, VARIABLES, ListVars, ext1_SE, AEROSOLS, MW, AVO, gr, Mwair)

        text = f'\n\t Calculating values for {case}'
        print(text)
        print("\t " + "-" * (len(text) - 2))



        # Gather dictionary data for current case
        if Climate:
            # NOTE: The calculations based on annual mean files
            Files_str= [str(f) for f in Files]

            #Special ADF variable which contains the output paths for
            #all generated plots and tables for each case:
            output_locs = adfobj.plot_location
            #Convert output location string to a Path object:
            output_location = Path(output_locs[0])

            cmd = ["ncra", "-O", *Files_str, f"{output_location}/{case}_ANN.nc"]
            subprocess.run(cmd, check=True)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                print("Error: 'ncra' command not found. Please install NCO utilities.")
            except subprocess.CalledProcessError as e:
                print(f"NCO Error (Exit Code {e.returncode}): {e.stderr}")

            #os.sys(f"ncra {Files} {output_location}/{case}_ANN.nc")
            File_mean=[f"{case}_ANN.nc"]

            Dic_scn_var_comp[case]= make_Dic_scn_var_comp_2D(adfobj, VARIABLES, output_location, dic_SE, File_mean, ext1_SE, AEROSOLS)
        else:
            print(f'Using raw files instead of annual mean files.')

            # NOTE: The calculations can take a long time...
            Dic_scn_var_comp[case] = make_Dic_scn_var_comp_2D(adfobj, VARIABLES, data_dir, dic_SE, Files, ext1_SE, AEROSOLS)

 
        # Gather dictionary data for current case
        # NOTE: The calculations can take a long time...
        #Dic_scn_var_comp[case] = make_Dic_scn_var_comp_2D(adfobj, VARIABLES, data_dir, dic_SE, Files, ext1_SE, AEROSOLS)        
        # Regional refinement
        # NOTE: This function 'Inside_SE' is unavailable at the moment! - JR 10/2024
        if regional:
            #inside = Inside_SE_region(current_lat,current_lon,dir_shapefile)
            inside = Inside_SE(Lats,Lons,limit)
        else:
            if len(np.shape(areas[case])) == 1:
                inside = np.full((len(Lons)),True)
            else:
                inside = np.full((len(Lats),len(Lons)),True)


        insides[case] = inside

    # Make and save tables
    table_kwargs = {"adfobj":adfobj,
                    "Dic_scn_var_comp":Dic_scn_var_comp,
                    "areas":areas,
                    "case_names":case_names,
                    "nicknames":nicknames_list,
                    "durations":durations,
                    "insides":insides,
                    "num_yrs":num_yrs,
                    "AEROSOLS":AEROSOLS}

    # Create the budget tables
    #-------------------------
    if len(VARIABLES) > 0:
        print("\tMaking table for all depositions")
        make_table_Dep(vars=VARIABLES, chem_type='all', **table_kwargs)

#######

##################
# Helper functions
##################

def list_files(adfobj, directory, start_year ,end_year, h_case):

    """
    This function extracts the files in the directory that are within the chosen dates
    and history number.
    """

    # History file year range
    yrs = np.arange(int(start_year), int(end_year))

    all_filenames = []
    for i in yrs:
        all_filenames.append(sorted(Path(directory).glob(f'*.{h_case}.{i:04d}*')))

    # Flattening the list of lists
    filenames = list(itertools.chain.from_iterable(sorted(all_filenames)))
    if len(filenames)==0:
        msg = f"chem/aerosol tables, 'list_files':"
        msg += f"\n\t - Directory '{directory}' has no outputs."
        adfobj.debug_log(msg)

    return filenames
#####


def Get_files(adfobj, data_dir, start_year, end_year, h_case, **kwargs):

    """
    This function retrieves the files, latitude, and longitude information
    in all the directories within the chosen dates.
    """
    ext1_SE = kwargs.pop('ext1_SE','')
    area = kwargs.pop('area',False)

    Earth_rad=6.371e6 # Earth Radius in meters

    current_files = list_files(adfobj, data_dir, start_year, end_year,h_case)
    # get the Lat and Lons for each case
    tmp_file = xr.open_dataset(Path(data_dir) / current_files[0])
    lon = tmp_file['lon'+ext1_SE].data
    lon[lon > 180.] -= 360 # shift longitude from 0-360˚ to -180-180˚
    lat = tmp_file['lat'+ext1_SE].data

    if area == True:
        try:
            tmp_area = tmp_file['area'+ext1_SE].data
            Earth_area = 4 * np.pi * Earth_rad**(2)

            areas = tmp_area*Earth_area/np.nansum(tmp_area)
        except KeyError:

            dlon = np.abs(lon[1]-lon[0])
            dlat = np.abs(lat[1]-lat[0])

            lon2d,lat2d = np.meshgrid(lon,lat)
            #area=np.zeros_like(lat2d)

            dy = Earth_rad*dlat*np.pi/180
            dx = Earth_rad*np.cos(lat2d*np.pi/180)*dlon*np.pi/180

            tmp_area = dx*dy
            areas = tmp_area
    # End if

    # Variables to return
    return current_files,lat,lon,areas,ext1_SE
#####

def set_dic_SE(ListVars, ext1_SE,variables):
    """
    Initialize dictionary to house all the relevant tabel data
    """

    # Initialize dictionary
    #----------------------
    dic_SE={}

    # Chemistry
    #----------
    dic_SE['U']={'U'+ext1_SE:1}
    dic_SE['O3']={'O3'+ext1_SE:1e9} # covert to ppb for Tropopause calculation

    # # Aerosols
    # #---------


    dic_SE['DUST']={'dst_a1'+ext1_SE:1,
                    'dst_a2'+ext1_SE:1,
                    'dst_a3'+ext1_SE:1}

    dic_SE['SALT']={'ncl_a1'+ext1_SE:1,
                    'ncl_a2'+ext1_SE:1,
                    'ncl_a3'+ext1_SE:1}

    dic_SE['POM']={'pom_a1'+ext1_SE:1,
                   'pom_a4'+ext1_SE:1}

    dic_SE['BC']={'bc_a1'+ext1_SE:1,
                  'bc_a4'+ext1_SE:1}


    dic_SE['SO4']={'so4_a1'+ext1_SE:1,
                   'so4_a2'+ext1_SE:1,
                   'so4_a3'+ext1_SE:1,
                   'so4_a5'+ext1_SE:1}

    # FOR SOA, first check if the integrated bins are included
    if (('soa_a1'+ext1_SE in ListVars ) & ('soa_a1'+ext1_SE in ListVars )):
        dic_SE['SOA'] = {'soa_a1'+ext1_SE:1,
                       'soa_a2'+ext1_SE:1}
    else:
        dic_SE['SOA'] = {'soa1_a1'+ext1_SE:1,
                   'soa2_a1'+ext1_SE:1,
                   'soa3_a1'+ext1_SE:1,
                   'soa4_a1'+ext1_SE:1,
                   'soa5_a1'+ext1_SE:1,
                   'soa1_a2'+ext1_SE:1,
                   'soa2_a2'+ext1_SE:1,
                   'soa3_a2'+ext1_SE:1,
                   'soa4_a2'+ext1_SE:1,
                   'soa5_a2'+ext1_SE:1}

    dic_SE['DMS']={'DMS'+ext1_SE:1}


    # automatic generation of dic_SE
    for var in variables:
        if var not in dic_SE.keys():
            dic_SE[var]={var+ext1_SE:1}

        # consider for OASISS DMS separately
        if var=='DMS':
            dic_SE['DMS_OASISS']={'DMS_OASISS'+ext1_SE:1}
    # End if

    return dic_SE
#####

def fill_dic_SE_Dep(adfobj, dic_SE, variables, ListVars, ext1_SE, AEROSOLS, MW, AVO, gr, Mwair):
    """
    Function for dealing with conversion factors for different components and filling the main data
    dictionary 'dic_SE'

    Input dictionary and return updated dictionary 'dic_SE'

    Arguments
    ---------
        variables : list
          - list of main variables?
        ListVars : list
          - list of ???????

    Returns
    -------
        dic_SE : dict
          - full dictionary of derived variables
    
    Some conversion factors need density or Layer's pressure, that will be accounted for when reading the files.
    We convert everying to kg/m2/s or kg/m2 or kg/s, so that final Tg/yr or Tg results are consistent
    """

    # Logging info message
    msg = f"chem/aerosol tables: 'fill_dic_SE_Emis'"

    for var in variables:



        dic_SE[var+'_DDF']={}
        dic_SE[var+'_WDF']={}

        if var in AEROSOLS:
            dic_SE[var+'_DDFC']={}
            dic_SE[var+'_WDFC']={}



        # Grab the variable keys 
        var_keys = dic_SE[var].keys()

        for key in var_keys:
            msg += f"\n\t Creating component of {var}: {key}"

     

            
            if var in AEROSOLS:

            

                # for DDF:
                # original unit: [kg/m2/s]
                if key+'DDF' in ListVars:  
                    #if var=='SO4':
                    if var in ['SO4','so4_a1','so4_a2','so4_a3','so4_a5']:
                        
                        dic_SE[var+'_DDF'][key+ext1_SE+'DDF']=32.066/115.11        
                    else:
                        dic_SE[var+'_DDF'][key+ext1_SE+'DDF']=1        
    
                else:
                    dic_SE[var+'_DDF']['PS'+ext1_SE]=0.  
            

                # for SFWET:
                # original unit: [kg/m2/s]
                if key+'SFWET' in ListVars:
                    #if var=='SO4':
                    if var in ['SO4','so4_a1','so4_a2','so4_a3','so4_a5']:
                        
                        dic_SE[var+'_WDF'][key+ext1_SE+'SFWET']=32.066/115.11        
                    else:
                        dic_SE[var+'_WDF'][key+ext1_SE+'SFWET']=1        
    
                else:               
                    dic_SE[var+'_WDF']['PS'+ext1_SE]=0.  
            


            

                # for DDF in cloud water:
                # original unit: [kg/m2/s]
                cloud_key=key[:-2]+'c'+key[-1]
                if cloud_key+ext1_SE+'DDF' in ListVars: 
                    #if var=='SO4':                                
                    if var in ['SO4','so4_a1','so4_a2','so4_a3','so4_a5']:
                    
                        dic_SE[var+'_DDFC'][cloud_key+ext1_SE+'DDF']=32.066/115.11        
                    else:
                        dic_SE[var+'_DDFC'][cloud_key+ext1_SE+'DDF']=1                            
                else:
                    dic_SE[var+'_DDFC']['PS'+ext1_SE]=0.  
    
                    

                # for SFWET in cloud water:
                # original unit: [kg/m2/s]
                if cloud_key+ext1_SE+'SFWET' in ListVars:   
                    #if var=='SO4':
                    if var in ['SO4','so4_a1','so4_a2','so4_a3','so4_a5']:
                        
                        dic_SE[var+'_WDFC'][cloud_key+ext1_SE+'SFWET']=32.066/115.11        
                    else:
                        dic_SE[var+'_WDFC'][cloud_key+ext1_SE+'SFWET']=1                            
                else:
                    dic_SE[var+'_WDFC']['PS'+ext1_SE]=0.                  
    

            else:

                
                # for DF:
                # original unit: [kg/m2/s]
                if 'DF_'+key in ListVars:
                    dic_SE[var+'_DDF']['DF_'+key+ext1_SE]=1
                else:
                    dic_SE[var+'_DDF']['PS'+ext1_SE]=0.

                # for WD:
                # original unit: [kg/m2/s]
                if 'WD_'+key in ListVars:
                    dic_SE[var+'_WDF']['WD_'+key+ext1_SE]=1
                else:
                    dic_SE[var+'_WDF']['PS'+ext1_SE]=0.
                

        # End for
    # End for
    
    # Write to log
    adfobj.debug_log(msg)

    return dic_SE
#####


def make_Dic_scn_var_comp_2D(adfobj, variables, current_dir, dic_SE, current_files, ext1_SE, AEROSOLS): 
    """
    This function retrieves the files, latitude, and longitude information
    in all the directories within the chosen dates.

    current_dir: list
      - showing the directories to look for files. always end with '/'

    current_files: list 
      - List of CAM history files

    start_year: string
      - Starting year

    end_year: string
      - Ending year

    kwargs
    ------
    ext1_SE: string
      - specify if the files are for only a region, which changes to variable names.
        ex: if you saved files for a only a box region ($LL_lat$,$LL_lon$,$UR_lat$,$UR_lon$),
            the 'lat' variable will be saved as: 'lat_$LL_lon$e_to_$UR_lon$e_$LL_lat$n_to_$UR_lat$n'
            for instance: 'lat_65e_to_91e_20n_to_32n'

    Returns
    ------- 

        Dic_scn_var_comp:
          - full dictionary of all variables and components for current case

    NOTE: The LNO is lightning NOx, which should be reported explicitly rather as CO_LNO, O3_LNO, ...
    """

    # Set lists to gather necessary variables for logging
    missing_vars_tot = []
    needed_vars_tot = []

    # Initialize final component dictionary
    Dic_var_comp={}

    for current_var in variables:
        if current_var in AEROSOLS: # AEROSOLS
            components=[current_var+'_DDF',current_var+'_WDF', 
                        current_var+'_DDFC',current_var+'_WDFC']  

        else: # CHEMS

            components=[current_var+'_DDF',current_var+'_WDF']  
                    
            # End if
        # End if
        msg = f"emissions table: 'make_Dic_scn_var_comp_2D'"
        msg += f"\n\t Current CAM variable: {current_var}"
        msg += f"\n\t   Derived components for CAM variable {current_var}: {components}"
        adfobj.debug_log(msg)

        Dic_comp={}
        Dic_comp,missing_vars,needed_vars=SEbudget(adfobj,dic_SE,current_dir,current_files,components,ext1_SE)

        for comp in components:
            # Write details to log file
            msg += f"\n\t\t   calculate derived component: {comp} for main variable, {current_var}"
            adfobj.debug_log(msg)

            # Gather info for debugging
            for var_m in missing_vars:
                if var_m not in missing_vars_tot:
                    missing_vars_tot.append(var_m)
            for var_n in needed_vars:
                if var_n not in needed_vars_tot:
                    needed_vars_tot.append(var_n)
        # End for
    # End for
        # Set dictionary for key of current variable with dictionary values of all
        # necessary constituents for calculating the current variable
        Dic_var_comp[current_var] = Dic_comp
    Dic_scn_var_comp = Dic_var_comp


    msg = f"depositions table:"
    msg += f"\n\t - needed variables for budget {needed_vars_tot}"
    adfobj.debug_log(msg)

    return Dic_scn_var_comp
#####


def SEbudget(adfobj,dic_SE,data_dir,files,vars,ext1_SE,**kwargs):
    """
    Function used for getting the data for the budget calculation. This is the
    chunk of code that takes the longest by far.

    Example:
    ** This is for both chemistry and aerosol calculations

    dic_SE: dictionary specyfing what variables to get. For example,
            for precipitation you can define SE as:
                dic_SE['PRECT']={'PRECC'+ext1_SE:8.64e7,'PRECL'+ext1_SE:8.64e7}
                - It means to sum the file variables "PRECC" and "PRECL"
                    for my arbitrary desired variable named "PRECT"

                - It also has the option to apply conversion factors.
                    For instance, PRECL and PRECC are in m/s. 8.64e7 is used to convernt m/s to mm/day


    data_dir: string of the directory that contains the files. always end with '/'

    files: list of the files to be read

    var: string showing the variable to be extracted.
     -> this will be the individual componnent, ie O3_CHMP, SOA_WDF, etc.
    """

    # gas constanct
    Rgas=287.04 #[J/K/Kg]=8.314/0.028965

    # Set lists to gather necessary variables for logging
    missing_vars = []
    needed_vars = []
    Dic_all_data={}

#    all_data=[]
    for file in range(len(files)):

        ds=xr.open_dataset(Path(data_dir) / files[file])

        # Calculate these just once
        if file==0:
            mock_2d=np.zeros_like(np.array(ds['PS'+ext1_SE].isel(time=0)))
            mock_3d=np.zeros_like(np.array(ds['U'+ext1_SE].isel(time=0)))
            
            if 'ncol' in list(ds.dims.keys()):
                SE=True
            else:
                SE=False
        try:
            delP=np.array(ds['PDELDRY'+ext1_SE].isel(time=0))
        except:
            hyai=np.array(ds['hyai'])
            hybi=np.array(ds['hybi'])

            try:
                PS=np.array(ds['PSDRY'+ext1_SE].isel(time=0))
            except:
                PS=np.array(ds['PS'+ext1_SE].isel(time=0))
            # End try/except
            P0=1e5
            if SE:
                Plevel=np.zeros((len(hyai),len(PS)))
            else:
                Plevel=np.zeros((len(hyai),len(PS),len(PS[0])))
                
            for i in range(len(Plevel)):
                Plevel[i]=hyai[i]*P0+hybi[i]*PS

            delP=Plevel[1:]-Plevel[:-1]

        for var in vars:
            if file == 0:
                    Dic_all_data[var]=[]

       # Star gathering of variable data
            if var=='TROP_P':
                data=np.array(ds['TROP_P'+ext1_SE].isel(time=0))/100
            elif var== 'Pressure':
                try:
                    data=np.array(ds['PMID'+ext1_SE].isel(time=0))/100
                except:
                    hyam=np.array(ds['hyam'])
                    hybm=np.array(ds['hybm'])
                    try:
                        PS=np.array(ds['PSDRY'+ext1_SE].isel(time=0))
                    except:
                        PS=np.array(ds['PS'+ext1_SE].isel(time=0))
                    P0=1e5
                    data=np.zeros_like(np.array(ds['U'+ext1_SE].isel(time=0)))
                    for i in range(len(data)):
                        data[i]=hyam[i]*P0+hybm[i]*PS
                    data=data/100
            else:
                data=[]
                for i in dic_SE[var].keys():
                    if file == 0:
                        msg = f"emissions table: 'SEbudget'"
                        msg += f"\n\t\t   ** variable(s) needed for derived var {var}: {dic_SE[var].keys()}"
                        msg += f"\n\t\t     - constituent for derived var {var}: {i}"
                        adfobj.debug_log(msg)
                        if i not in needed_vars:
                            needed_vars.append(i)
                    if ((i!='PS'+ext1_SE) and (i!='U'+ext1_SE) ) :
                        data.append(np.array(ds[i].isel(time=0))*dic_SE[var][i])
                    else:
                        if i=='PS'+ext1_SE:
                            data.append(mock_2d)
                        else:
                            data.append(mock_3d)
                        if file == 0:
                            if var not in missing_vars:
                                if var!='U': # This is to avoid confusion between U variable or U mock!
                                    missing_vars.append(var)
                                    msg += f"\n\t\t     - no variable was found for var {var}: {i}"
    
                # End if
    
                # Get total summed data for this history file data
                data=np.sum(data,axis=0)

            if ('CHML' in var) or ('CHMP' in var) :
                Temp=np.array(ds['T'+ext1_SE].isel(time=0))
                try:
                    Pres=np.array(ds['PMID'+ext1_SE].isel(time=0))
                except:
                    hyam=np.array(ds['hyam'])
                    hybm=np.array(ds['hybm'])
                    try:
                        PS=np.array(ds['PSDRY'+ext1_SE].isel(time=0))
                    except:
                        PS=np.array(ds['PS'+ext1_SE].isel(time=0))
                    P0=1e5
                    Pres=np.zeros_like(np.array(ds['U'+ext1_SE].isel(time=0)))
                    for i in range(len(Pres)):
                        Pres[i]=hyam[i]*P0+hybm[i]*PS
                rho= Pres/(Rgas*Temp)
                data=data*delP/rho
            elif ('BURDEN' in var):
                data=data*delP
            else:
                data=data
        # End if
            # Add data to list
            Dic_all_data[var].append(data)
        ds.close()
    for var in vars: # Take mean
         Dic_all_data[var]=np.nanmean(Dic_all_data[var],axis=0)


    return Dic_all_data,missing_vars,needed_vars
#####


def calc_budget_data_Dep(current_var, Dic_scn_var_comp, area, inside, num_yrs, duration, AEROSOLS):
    """
    Function to run through desired table values for calculations for the table entries
    """

    # Initialize full data dictionary for current table type
    chem_dict = {}

    # Update variable marker if neccessary
    if current_var in ['SO4','so4_a1','so4_a2','so4_a3','so4_a5']:
        specifier=' S'
    else:
        specifier=''

    
    if current_var in AEROSOLS:

       # Dry Deposition Flux
        try:
            spc_ddfa=Dic_scn_var_comp[current_var][current_var+'_DDF'] 
            spc_ddfc=Dic_scn_var_comp[current_var][current_var+'_DDFC']
            spc_ddf=spc_ddfa +spc_ddfc
        except:
            spc_ddf = 0
        tmp_ddf=spc_ddf
        ddf=np.ma.masked_where(inside==False,tmp_ddf*area)  #convert Kg/m2/s to Tg/yr
        DDF = np.ma.sum(ddf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_DRYDEP (Tg{specifier}/yr)"] = np.round(DDF,5)
        
        # Wet deposition
        try:
            spc_wdfa=Dic_scn_var_comp[current_var][current_var+'_WDF'] 
            spc_wdfc=Dic_scn_var_comp[current_var][current_var+'_WDFC']
            spc_wdf=spc_wdfa +spc_wdfc           
        except:
            spc_wdf = 0
        tmp_wdf=spc_wdf
        wdf=np.ma.masked_where(inside==False,tmp_wdf*area)  #convert Kg/m2/s to Tg/yr
        WDF = -1 * np.ma.sum(wdf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_WETDEP (Tg{specifier}/yr)"] = np.round(WDF,5)

        # Total Deposition
        TDEP = DDF + WDF
        chem_dict[f"{current_var}_TDEP (Tg/yr)"] = np.round(TDEP,5)

        
    else:
    
        # Dry Deposition Flux
        #print("Dry Deposition Flux")
        try:
            spc_ddf=Dic_scn_var_comp[current_var][current_var+'_DDF']
        except:
            spc_ddf = 0
        tmp_ddf=spc_ddf
        ddf=np.ma.masked_where(inside==False,tmp_ddf*area)  #convert Kg/m2/s to Tg/yr
        DDF = np.ma.sum(ddf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_DRYDEP (Tg/yr)"] = np.round(DDF,5)

        # Wet Deposition Flux
        #print("Wet Deposition Flux")
        try:
            spc_wdf=Dic_scn_var_comp[current_var][current_var+'_WDF']
        except:
            spc_wdf = 0
        tmp_wdf=spc_wdf
        wdf=np.ma.masked_where(inside==False,tmp_wdf*area)  #convert Kg/m2/s to Tg/yr
        WDF = -1*np.ma.sum(wdf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_WETDEP (Tg/yr)"] = np.round(WDF,5)



        
    return chem_dict
#####


def make_table_Dep(adfobj, vars, chem_type, Dic_scn_var_comp, areas, case_names, nicknames, durations, insides, num_yrs, AEROSOLS):
    """
    Create CSV table for all emissions, if applicable

    Table includes column values of variable, case(s), difference (if applicable)

    If this is a single model vs model run: 4 columns
        first column: variables names, 
        second column: test case variable values
        third column: baseline case variable values
        final column: difference of test and baseline.
    If this is a model vs obs run: 2 columns
        first column: variables names, 
        second column: test case variable values
    """
    # Initialize an empty dictionary to store DataFrames
    dfs = {}

    #Special ADF variable which contains the output paths for
    #all generated plots and tables for each case:
    output_locs = adfobj.plot_location

    #Convert output location string to a Path object:
    output_location = Path(output_locs[0])

    # Loop over model cases

    for i,case in enumerate(nicknames):

        nickname = case

        # Collect row data in a list of dictionaries
        #durations[case]
        rows = []
        for current_var in vars:
            chem_dict = calc_budget_data_Dep(current_var, Dic_scn_var_comp[case], areas[case], insides[case],
                                         num_yrs[case], durations[case], AEROSOLS)
            # Loop through table variables
            for key, val in chem_dict.items():
                if val != 0:  # Skip variables with a value of 0
                    print(f"\t - Variable '{key}' being added to table")
                    rows.append({'variable': key, nickname: np.round(val, 3)})                   
                else:
                    msg = f"chem/aerosol depositions table:"
                    msg += f"\n\t - Variable '{key}' has value of 0, will not add to table"
                    adfobj.debug_log(msg)
                # End if
            # End for
        # End for

        # Create the DataFrame for the current case
        table_df = pd.DataFrame(rows)


        # Store the DataFrame in the dictionary
        dfs[nickname] = table_df
        
    # End for

    # Merge the DataFrames on the 'variable' column
    if len(case_names) == 2:
        
        table_df = pd.merge(dfs[nicknames[0]], dfs[nicknames[1]], on='variable')

        # Calculate the differences between case columns
        table_df['difference'] = table_df[nicknames[0]] - table_df[nicknames[1]]

    #Create output file name:
    output_csv_file = output_location / f'ADF_amwg_{chem_type}_depositions_table.csv'
    # Save table to CSV and add table dataframe to website (if enabled)
    table_df.to_csv(output_csv_file, index=False)
    adfobj.add_website_data(table_df, chem_type, case_names[0], plot_type="Tables")

#####
