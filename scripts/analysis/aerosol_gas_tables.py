import numpy as np
import xarray as xr
import sys
from pathlib import Path
import warnings  # use to warn user about missing files.

from datetime import datetime
import numpy as np
import itertools

try:
    import pandas as pd
except ImportError:
    print("Pandas module does not exist in python path, but is needed for amwg_table.")
    print("Please install module, e.g. 'pip install pandas'.")
    sys.exit(1)
#End except

# Import necessary ADF modules:
from adf_base import AdfError

def aerosol_gas_tables(adfobj):
    '''
    Calculate aerosol and gaseous budget tables

    Default set of variables: change in lib/adf_variable_defaults.yaml
    -------------------------
    GAS_VARIABLES: ['CH4','CH3CCL3', 'CO', 'O3', 'ISOP', 'MTERP', 'CH3OH', 'CH3COCH3']
    AEROSOL_VARIABLES: ['AOD','SOA', 'SALT', 'DUST', 'POM', 'BC', 'SO4']

    Default output for tables:

        Gases:
        ------
        CH4_BURDEN (Tg), CH4_CHEM_LOSS (Tg/yr), CH4_LIFETIME (years)
        
        CH3CCL3_BURDEN (Tg), CH3CCL3_CHEM_LOSS (Tg/yr), CH3CCL3_LIFETIME (days)
        
        CO_EMIS (Tg/yr), CO_BURDEN (Tg), CO_CHEM_LOSS (Tg/yr), CO_CHEM_PROD (Tg/yr), CO_DRYDEP (Tg/yr)
        CO_TDEP (Tg/yr), CO_LIFETIME (days), CO_TEND (Tg/yr)
        
        O3_BURDEN (Tg), O3_CHEM_LOSS (Tg/yr), O3_CHEM_PROD (Tg/yr), O3_DRYDEP (Tg/yr), O3_TDEP (Tg/yr)
        O3_LIFETIME (days), O3_TEND (Tg/yr), O3_STE (Tg/yr)
        
        LNOx_PROD (Tg N/yr)
        
        ISOP_EMIS (Tg/yr), ISOP_BURDEN (Tg)
        
        Monoterpene_EMIS (Tg/yr), Monoterpene_BURDEN (Tg)
        
        Methanol_EMIS (Tg/yr), Methanol_BURDEN (Tg), Methanol_DRYDEP (Tg/yr), Methanol_WETDEP (Tg/yr), Methanol_TDEP (Tg/yr)
        
        Acetone_EMIS (Tg/yr), Acetone_BURDEN (Tg), Acetone_DRYDEP (Tg/yr), Acetone_WETDEP (Tg/yr), Acetone_TDEP (Tg/yr)



        Aerosols:
        ---------
        AOD_mean
        
        SOA_BURDEN (Tg), SOA_CHEM_LOSS (Tg/yr), SOA_DRYDEP (Tg/yr), SOA_WETDEP (Tg/yr), SOA_GAEX (Tg/yr), SOA_LIFETIME (days)
        
        SALT_EMIS (Tg/yr), SALT_BURDEN (Tg), SALT_DRYDEP (Tg/yr), SALT_WETDEP (Tg/yr), SALT_LIFETIME (days)
        
        DUST_EMIS (Tg/yr), DUST_BURDEN (Tg), DUST_DRYDEP (Tg/yr), DUST_WETDEP (Tg/yr), DUST_LIFETIME (days)
        
        POM_EMIS (Tg/yr), POM_BURDEN (Tg), POM_DRYDEP (Tg/yr), POM_WETDEP (Tg/yr), POM_LIFETIME (days)
        
        BC_EMIS (Tg/yr), BC_BURDEN (Tg), BC_DRYDEP (Tg/yr), BC_WETDEP (Tg/yr), BC_LIFETIME (days)
        
        SO4_EMIS_elevated (Tg S/yr), SO4_BURDEN (Tg S), SO4_DRYDEP (Tg S/yr), SO4_WETDEP (Tg S/yr), SO4_GAEX (Tg S/yr)
        SO4_LIFETIME (days), SO4_AQUEOUS (Tg S/yr), SO4_NUCLEATION (Tg S/yr)


    List of variable names and descriptions for clarity
    ---------------------------------------------------
        - ListVars: list of all available variables from given history file
        - GAS_VARIABLES: list fo necessary CAM gaseous variables
        - AEROSOL_VARIABLES: list fo necessary CAM aerosol variables
        - AEROSOLS: list of necessary aerosols for computations


    MODIFICATION HISTORY:
        Behrooz Roozitalab, 02, NOV, 2022: VERSION 1.00
        - Initial version

        Justin Richling, 27 Nov, 2023
        - updated to fit to ADF and check with old AMWG chem/aerosol tables
        - fixed:
            * added difference bewtween cases column to tables

        Behrooz Roozitalab, 8 Aug, 2024
        - fixed:
            * lifetime inconsitencies
            * Removed redundant calculations to improve the speed
            * Verified the results against the NCL script.
    '''


    #Notify user that script has started:
    print("\n  Calculating chemistry/aerosol budget tables...")

    # Inputs
    #-------
    # Variable defaults info
    res = adfobj.variable_defaults # dict of variable-specific plot preferences
    bres = res['budget_tables']

    # list of the gaseous variables to be caculated.
    GAS_VARIABLES = bres['GAS_VARIABLES']

    # list of the aerosol variables to be caculated.
    AEROSOL_VARIABLES = bres['AEROSOL_VARIABLES']

    #list of all the variables to be caculated.
    VARIABLES = GAS_VARIABLES + AEROSOL_VARIABLES

    # For the case that outputs are saved for a specific region.
    # i.e., when using fincllonlat in user_nl_cam
    ext1_SE = bres['ext1_SE']

    # Tropospheric Values
    # -------------------
    # if True, calculate only Tropospheric values
    # if False, all layers
    # tropopause is defiend as o3>150ppb. If needed, change accordingly.
    Tropospheric = bres['Tropospheric']

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
    nicknames = {}

    # Filter the list to include only strings that are possible h0 strings
    # - Search for either h0 or h0a
    substrings = {"cam.h0","cam.h0a"}
    case_hist_strs = []
    print("hist_strs",hist_strs,"\n")
    for cam_case_str in hist_strs:
        # Check each possible h0 string
        for string in cam_case_str:
            if string in substrings:
                case_hist_strs.append(string)
                break

    # Create path object for the CAM history file(s) location:
    data_dirs = []
    for case_idx,case in enumerate(case_names):
        print(f"\t Looking for history location: {hist_locs[case_idx]}")
        nicknames[case] = nicknames_list[case_idx]

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
    for i,case in enumerate(case_names):
        start_year = start_years[i]
        end_year = end_years[i]
        start_date = f"{start_year}-1-1"
        end_date = f"{end_year}-1-1"
        
        # Create time periods
        start_period = datetime.strptime(start_date, "%Y-%m-%d")
        end_period = datetime.strptime(end_date, "%Y-%m-%d")

        # Calculated duration of time period in seconds?
        durations[case_names[i]] = (end_period-start_period).days*86400+365*86400

        # Get number of years for calculations
        num_yrs[case_names[i]] = (int(end_year)-int(start_year))+1

        # Get currenty history file directory
        data_dir = data_dirs[i]

        # Get all files, lats, lons, and area weights for current case
        Files,Lats,Lons,areas[case],ext1_SE = Get_files(adfobj,data_dir,start_year,end_year,case_hist_strs[i],area=True)

        # find the name of all the variables in the file.
        # this will help the code to work for the variables that are not in the files (assingn 0s)
        tmp_file = xr.open_dataset(Path(data_dir) / Files[0])
        ListVars = list(tmp_file.variables)

        # Set up and fill dictionaries for components for current cases
        dic_SE = set_dic_SE(ListVars,ext1_SE)
        dic_SE = fill_dic_SE(adfobj, dic_SE, VARIABLES, ListVars, ext1_SE, AEROSOLS, MW, AVO, gr, Mwair)

        # Make dictionary of all data for current case
        # NOTE: The calculations can take a long time...
        #print(f'\t Calculating values for {case}',"\n",len(f'\t Calculating values for {case}')*'-','\n')
        text = f'\n\t Calculating values for {case}'
        print(text)
        print("\n\t " + "-" * (len(text) - 1))

        # Gather dictionary data for current case
        Dic_crit, Dic_scn_var_comp[case] = make_Dic_scn_var_comp(adfobj, VARIABLES, data_dir, dic_SE, Files, ext1_SE, AEROSOLS)

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

        # Set critical threshold
        current_crit = Dic_crit[0]

        if Tropospheric:
            trop = np.where(current_crit>150,np.nan,current_crit)
            #strat=np.where(current_crit>150,current_crit,np.nan)
        else:
            trop=current_crit
        trops[case] = trop
        insides[case] = inside

    # Make and save tables
    table_kwargs = {"adfobj":adfobj,
                    "Dic_scn_var_comp":Dic_scn_var_comp,
                    "areas":areas,
                    "trops":trops,
                    "case_names":case_names,
                    "nicknames":nicknames,
                    "durations":durations,
                    "insides":insides,
                    "num_yrs":num_yrs,
                    "AEROSOLS":AEROSOLS}

    # Create the budget tables
    #-------------------------
    # Aerosols
    if len(AEROSOL_VARIABLES) > 0:
        print("\tMaking table for aerosols")
        make_table(vars=AEROSOL_VARIABLES, chem_type='aerosols', **table_kwargs)
    # Gases
    if len(GAS_VARIABLES) > 0:
        print("\tMaking table for gases")
        make_table(vars=GAS_VARIABLES, chem_type='gases', **table_kwargs)
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
    yrs = np.arange(int(start_year), int(end_year)+1)

    all_filenames = []
    for i in yrs:
        all_filenames.append(sorted(Path(directory).glob(f'*.{h_case}.{i}-*')))

    # Flattening the list of lists
    filenames = list(itertools.chain.from_iterable(sorted(all_filenames)))
    if len(filenames)==0:
        #sys.exit(" Directory has no outputs ")
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
            try:
                tmp_area = tmp_file['AREA'+ext1_SE].isel(time=0).data
                Earth_area = 4 * np.pi * Earth_rad**(2)
                areas = tmp_area*Earth_area/np.nansum(tmp_area)
            except:
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

def set_dic_SE(ListVars, ext1_SE):
    """
    Initialize dictionary to house all the relevant tabel data
    """

    # Initialize dictionary
    #----------------------
    dic_SE={}

    # Chemistry
    #----------
    dic_SE['O3']={'O3'+ext1_SE:1e9} # covert to ppb for Tropopause calculation
    dic_SE['CH4']={'CH4'+ext1_SE:1}
    dic_SE['CO']={'CO'+ext1_SE:1}

    dic_SE['ISOP']={'ISOP'+ext1_SE:1}
    dic_SE['MTERP']={'MTERP'+ext1_SE:1}
    dic_SE['CH3OH']={'CH3OH'+ext1_SE:1}
    dic_SE['CH3COCH3']={'CH3COCH3'+ext1_SE:1}
    dic_SE['CH3CCL3']={'CH3CCL3'+ext1_SE:1}


    # Aerosols
    #---------

    dic_SE['DAOD']={'AODDUSTdn'+ext1_SE:1}
    dic_SE['AOD']={'AODVISdn'+ext1_SE:1}

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
    # End if

    return dic_SE
#####

def fill_dic_SE(adfobj, dic_SE, variables, ListVars, ext1_SE, AEROSOLS, MW, AVO, gr, Mwair):
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
    msg = f"chem/aerosol tables: 'fill_dic_SE'"

    for var in variables:

        if 'AOD' in var:
            dic_SE[var+'_AOD']={}
        else:
            dic_SE[var+'_BURDEN']={}
            dic_SE[var+'_CHML']={}
            dic_SE[var+'_CHMP']={}

            dic_SE[var+'_SF']={}
            dic_SE[var+'_CLXF']={}

            dic_SE[var+'_DDF']={}
            dic_SE[var+'_WDF']={}

            if var in AEROSOLS:
                dic_SE[var+'_GAEX']={}
                dic_SE[var+'_DDFC']={}
                dic_SE[var+'_WDFC']={}
            else:
                dic_SE[var+'_TEND']={}
                dic_SE[var+'_LNO']={}
            # End if

            # We have nucleation and aqueous chemistry for sulfate.
            if var=='SO4':
                dic_SE[var+'_NUCL']={}
                dic_SE[var+'_AQS']={}
        # End if

        # Grab the variable keys 
        var_keys = dic_SE[var].keys()

        for key in var_keys:
            msg += f"\n\t Creating component of {var}: {key}"

            # for CHML and CHMP:
            # original unit : [molec/cm3/s]
            # following Tilmes code to convert to [kg/m2/s]
            # conversion: Mw*rho*delP*1e3/Avo/gr
            # rho and delP will be applied when reading the files in SEbudget function.

            # for AOD and DAOD:
            if 'AOD' in var:
                if key in ListVars:
                    dic_SE[var+'_AOD'][key+ext1_SE]=1 
                else:
                    dic_SE[var+'_AOD']['PS'+ext1_SE]=0.
                # End if
                continue # AOD doesn't need any other budget calculations
            # End if

            # for CHML and CHMP:
            # original unit : [molec/cm3/s]
            # following Tilmes code to convert to [kg/m2/s]
            # conversion: Mw*rho*delP*1e3/Avo/gr
            # rho and delP will be applied when reading the files in SEbudget function.
            if key=='O3'+ext1_SE:
                # for O3, we should not include fast cycling reactions
                # As a result, we use below diagnostics in the model if available. If not, we use CHML and CHMP
                if ((key+'_Loss' in ListVars) & (key+'_Prod' in ListVars)) :
                    dic_SE[var+'_CHML'][key+'_Loss'+ext1_SE]=MW[var]*1e3/AVO/gr
                    dic_SE[var+'_CHMP'][key+'_Prod'+ext1_SE]=MW[var]*1e3/AVO/gr
                else:
                    if key+'_CHML' in ListVars:
                        dic_SE[var+'_CHML'][key+'_CHML'+ext1_SE]=MW[var]*1e3/AVO/gr
                    else:
                        dic_SE[var+'_CHML']['U'+ext1_SE]=0
                    # End if

                    if key+'_CHMP' in ListVars:
                        dic_SE[var+'_CHMP'][key+'_CHMP'+ext1_SE]=MW[var]*1e3/AVO/gr
                    else:
                        dic_SE[var+'_CHMP']['U'+ext1_SE]=0
                    # End if
                # End if
            else:
                if key+'_CHML' in ListVars:
                    dic_SE[var+'_CHML'][key+'_CHML'+ext1_SE]=MW[var]*1e3/AVO/gr
                else:
                    dic_SE[var+'_CHML']['U'+ext1_SE]=0
                # End if

                if key+'_CHMP' in ListVars:
                    dic_SE[var+'_CHMP'][key+'_CHMP'+ext1_SE]=MW[var]*1e3/AVO/gr
                else:
                    dic_SE[var+'_CHMP']['U'+ext1_SE]=0
                # End if
            # End if


            # for SF:
            # original unit: [kg/m2/s]
            if 'SF'+key in ListVars:
                if var=='SO4':
                    dic_SE[var+'_SF']['SF'+key+ext1_SE]=32.066/115.11
                else:
                    dic_SE[var+'_SF']['SF'+key+ext1_SE]=1
                # End if
            elif key+'SF' in ListVars:
                dic_SE[var+'_SF'][key+ext1_SE+'SF']=1
            else:
                dic_SE[var+'_SF']['PS'+ext1_SE]=0.
            # End if


            # for CLXF:
            # original unit: [molec/cm2/s]
            # conversion: Mw*10/Avo
            if key+'_CLXF' in ListVars:
                dic_SE[var+'_CLXF'][key+'_CLXF'+ext1_SE]=MW[var]*10/AVO  # convert [molec/cm2/s] to [kg/m2/s]
            else:
                dic_SE[var+'_CLXF']['PS'+ext1_SE]=0.
            # End if

            # Aerosols
            if var in AEROSOLS:
                # for each species:
                # original unit : [kg/kg]  in dry air
                # convert to [kg/m2]
                # conversion: delP/gr
                # delP will be applied when reading the files in SEbudget function.
                if key in ListVars:
                    if var=='SO4': # For SO4, we report all the budget calculation for Sulfur.
                        dic_SE[var+'_BURDEN'][key+ext1_SE]=(32.066/115.11)/gr
                    else:
                        dic_SE[var+'_BURDEN'][key+ext1_SE]=1/gr
                    # End if
                else:
                    dic_SE[var+'_BURDEN']['U'+ext1_SE]=0
                # End if


                # for DDF:
                # original unit: [kg/m2/s]
                if key+'DDF' in ListVars:
                    if var=='SO4':
                        dic_SE[var+'_DDF'][key+ext1_SE+'DDF']=32.066/115.11
                    else:
                        dic_SE[var+'_DDF'][key+ext1_SE+'DDF']=1
                    # End if
                else:
                    dic_SE[var+'_DDF']['PS'+ext1_SE]=0.
                # End if


                # for SFWET:
                # original unit: [kg/m2/s]
                if key+'SFWET' in ListVars:
                    if var=='SO4':
                        dic_SE[var+'_WDF'][key+ext1_SE+'SFWET']=32.066/115.11
                    else:
                        dic_SE[var+'_WDF'][key+ext1_SE+'SFWET']=1
                    # End if
                else:
                    dic_SE[var+'_WDF']['PS'+ext1_SE]=0.
                # End if


                # for sfgaex1:
                # original unit: [kg/m2/s]
                if key+'_sfgaex1' in ListVars:
                    if var=='SO4':
                        dic_SE[var+'_GAEX'][key+ext1_SE+'_sfgaex1']=32.066/115.11
                    else:
                        dic_SE[var+'_GAEX'][key+ext1_SE+'_sfgaex1']=1
                    # End if
                else:
                    dic_SE[var+'_GAEX']['PS'+ext1_SE]=0.
                # End if


                # for DDF in cloud water:
                # original unit: [kg/m2/s]
                cloud_key=key[:-2]+'c'+key[-1]
                if cloud_key+ext1_SE+'DDF' in ListVars:
                    if var=='SO4':
                        dic_SE[var+'_DDFC'][cloud_key+ext1_SE+'DDF']=32.066/115.11
                    else:
                        dic_SE[var+'_DDFC'][cloud_key+ext1_SE+'DDF']=1
                    # End if
                else:
                    dic_SE[var+'_DDFC']['PS'+ext1_SE]=0.
                # End if

                # for SFWET in cloud water:
                # original unit: [kg/m2/s]
                if cloud_key+ext1_SE+'SFWET' in ListVars:
                    if var=='SO4':
                        dic_SE[var+'_WDFC'][cloud_key+ext1_SE+'SFWET']=32.066/115.11
                    else:
                        dic_SE[var+'_WDFC'][cloud_key+ext1_SE+'SFWET']=1
                    # End if
                else:
                    dic_SE[var+'_WDFC']['PS'+ext1_SE]=0.
                # End if

                if var=='SO4':
                    # for Nucleation :
                    # original unit: [kg/m2/s]
                    if key+ext1_SE+'_sfnnuc1' in ListVars:
                        dic_SE[var+'_NUCL'][key+ext1_SE+'_sfnnuc1']=32.066/115.11
                    else:
                        dic_SE[var+'_NUCL']['PS'+ext1_SE]=0.
                    # End if

                    # for Aqueous phase :
                    # original unit: [kg/m2/s]
                    if (('AQSO4_H2O2'+ext1_SE in ListVars) & ('AQSO4_O3'+ext1_SE in ListVars)) :
                            dic_SE[var+'_AQS']['AQSO4_H2O2'+ext1_SE]=32.066/115.11
                            dic_SE[var+'_AQS']['AQSO4_O3'+ext1_SE]=32.066/115.11
                    else:
                        # original unit: [kg/m2/s]
                        if cloud_key+'AQSO4'+ext1_SE in ListVars:
                            dic_SE[var+'_AQS'][cloud_key+'AQSO4'+ext1_SE]=32.066/115.11
                        else:
                            dic_SE[var+'_AQS']['PS'+ext1_SE]=0.
                        # End if

                        if cloud_key+'AQH2SO4'+ext1_SE in ListVars:
                            dic_SE[var+'_AQS'][cloud_key+'AQH2SO4'+ext1_SE]=32.066/115.11
                        else:
                            dic_SE[var+'_AQS']['PS'+ext1_SE]=0.
                        # End if
                    # End if
                # End if

            else: # Gases
                # for each species:
                # original unit : [mole/mole]  in dry air
                # convert to [kg/m2]
                # conversion: Mw*delP/Mwair/gr     Mwair=28.97 gr/mole
                # delP will be applied when reading the files in SEbudget function.
                if key in ListVars:
                    dic_SE[var+'_BURDEN'][key+ext1_SE]=MW[var]/Mwair/gr
                else:
                    dic_SE[var+'_BURDEN']['U'+ext1_SE]=0
                # End if

                # for DF:
                # original unit: [kg/m2/s]
                if 'DF_'+key in ListVars:
                    dic_SE[var+'_DDF']['DF_'+key+ext1_SE]=1
                else:
                    dic_SE[var+'_DDF']['PS'+ext1_SE]=0.
                # End if

                # for WD:
                # original unit: [kg/m2/s]
                if 'WD_'+key in ListVars:
                    dic_SE[var+'_WDF']['WD_'+key+ext1_SE]=1
                else:
                    dic_SE[var+'_WDF']['PS'+ext1_SE]=0.
                # End if

                # for Chem tendency:
                # original unit: [kg/s]
                # conversion: not needed
                if 'D'+key+'CHM' in ListVars:
                    dic_SE[var+'_TEND']['D'+key+'CHM'+ext1_SE]=1  # convert [kg/s] to [kg/s]
                else:
                    dic_SE[var+'_TEND']['U'+ext1_SE]=0
                # End if

                # for Lightning NO production: (always in gas)
                # original unit: [Tg N/Yr]
                # conversion: not needed
                if 'LNO_COL_PROD' in ListVars:
                    dic_SE[var+'_LNO']['LNO_COL_PROD'+ext1_SE]=1  # convert [Tg N/yr] to [Tg N /yr]
                else:
                    dic_SE[var+'_LNO']['PS'+ext1_SE]=0
                # End if
            # End if (aerosols or gases)
        # End for
    # End for
    
    # Write to log
    adfobj.debug_log(msg)

    return dic_SE
#####


def make_Dic_scn_var_comp(adfobj, variables, current_dir, dic_SE, current_files, ext1_SE, AEROSOLS):
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
        Dic_crit:
          - dictionary for critical values for current case
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
        if 'AOD' in current_var:
            components=[current_var+'_AOD']
        else:
            if current_var in AEROSOLS: # AEROSOLS

                # Components are: burden, chemical loss, chemical prod, dry deposition,
                #                 surface emissions, elevated emissions, wet deposition, gas-aerosol exchange
                components=[current_var+'_BURDEN',current_var+'_CHML',current_var+'_CHMP',
                            current_var+'_DDF',current_var+'_WDF', current_var+'_SF', current_var+'_CLXF',
                            current_var+'_DDFC',current_var+'_WDFC']

                if current_var=='SO4':
                    # For SULF we also have AQS, nucleation, and strat-trop gas exchange
                    components.append(current_var+'_AQS')
                    components.append(current_var+'_NUCL')
                    components.append(current_var+'_GAEX')
                    components.remove(current_var+'_CHMP')

                    #components.append(current_var+'_CLXF') # BRT -  CLXF is added above.
                if current_var == "SOA":
                    components.append(current_var+'_GAEX')
            #End if - AEROSOLS

            else: # CHEMS
                # Components are: burden, chemical loss, chemical prod, dry/wet deposition,
                #                 surface emissions, elevated emissions, chemical tendency
                # I always add Lightning NOx production when calculating O3 budget.

                components=[current_var+'_BURDEN',current_var+'_CHML',current_var+'_CHMP',
                            current_var+'_DDF',current_var+'_WDF', current_var+'_SF', current_var+'_CLXF',
                            current_var+'_TEND']

                if current_var =="O3":
                        components.append(current_var+'_LNO')
            # End if
        # End if
        msg = f"chem/aerosol tables: 'make_Dic_scn_var_comp'"
        msg += f"\n\t Current CAM variable: {current_var}"
        msg += f"\n\t   Derived components for CAM variable {current_var}: {components}"
        #adfobj.debug_log(msg)
        Dic_comp={}
        for comp in components:
            # Write details to log file
            msg += f"\n\t\t   calculate derived component: {comp} for main variable, {current_var}"
            adfobj.debug_log(msg)

            # Get component values
            current_data,missing_vars,needed_vars = SEbudget(adfobj,dic_SE,current_dir,current_files,comp,ext1_SE)

            # Gather info for debugging
            for var_m in missing_vars:
                if var_m not in missing_vars_tot:
                    missing_vars_tot.append(var_m)
            for var_n in needed_vars:
                if var_n not in needed_vars_tot:
                    needed_vars_tot.append(var_n)
        # End for
    # End for
    #TODO: check this section to see if it can't be better run
            # Set dictionary for component
            Dic_comp[comp] = current_data
        # Set dictionary for key of current variable with dictionary values of all
        # necessary constituents for calculating the current variable
        Dic_var_comp[current_var] = Dic_comp
    Dic_scn_var_comp = Dic_var_comp

    # Critical threshholds, just run this once
    # this is for finding tropospheric values
    current_crit=SEbudget(adfobj,dic_SE,current_dir,current_files,'O3',ext1_SE)
    Dic_crit=current_crit

    # Log info to logging file
    msg = f"chem/aerosol tables:"
    msg += f"\n\t - potential missing variables from budget? {missing_vars_tot}"
    adfobj.debug_log(msg)

    msg = f"chem/aerosol tables:"
    msg += f"\n\t - needed variables for budget {needed_vars_tot}"
    adfobj.debug_log(msg)

    return Dic_crit,Dic_scn_var_comp
#####


def SEbudget(adfobj,dic_SE,data_dir,files,var,ext1_SE,**kwargs):
    """
    Function used for getting the data for the budget calculation. This is the
    chunk of code that takes the longest by far.

    Example:
    ~70/75 mins per case for 9 years
    ** This is for both chemistry and aeorosl calculations

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

    all_data=[]
    for file in range(len(files)):

        ds=xr.open_dataset(Path(data_dir) / files[file])

        # Calculate these just once
        if file==0:
            mock_2d=np.zeros_like(np.array(ds['PS'+ext1_SE].isel(time=0)))
            mock_3d=np.zeros_like(np.array(ds['U'+ext1_SE].isel(time=0)))

        # Star gathering of variable data
        data=[]
        for i in dic_SE[var].keys():
            if i not in needed_vars:
                needed_vars.append(i)
            if file == 0:
                msg = f"chem/aerosol tables: 'SEbudget'"
                msg += f"\n\t\t   ** variable(s) needed for derived var {var}: {dic_SE[var].keys()}"
                msg += f"\n\t\t     - constituent for derived var {var}: {i}"
                adfobj.debug_log(msg)

            if ((i!='PS'+ext1_SE) and (i!='U'+ext1_SE) ) :
                data.append(np.array(ds[i].isel(time=0))*dic_SE[var][i])
            else:
                if i=='PS'+ext1_SE:
                    data.append(mock_2d)
                else:
                    data.append(mock_3d)
                # End if

                if var not in missing_vars:
                    missing_vars.append(var)
            # End if

        # Get total summed data for this history file data
        data=np.sum(data,axis=0)

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
            Plevel=np.zeros_like(np.array(ds['U'+ext1_SE]))

            for i in range(len(Plevel)):
                Plevel[i]=hyai[i]*P0+hybi[i]*PS

            delP=Plevel[1:]-Plevel[:-1]
        # End try/except

        if ('CHML' in var) or ('CHMP' in var) :
            Temp=np.array(ds['T'+ext1_SE].isel(time=0))
            Pres=np.array(ds['PMID'+ext1_SE].isel(time=0))
            rho= Pres/(Rgas*Temp)
            data=data*delP/rho
        elif ('BURDEN' in var):
            data=data*delP
        else:
            data=data
        # End if

        # Add data to list
        all_data.append(data)

    # Take mean
    all_data=np.nanmean(all_data,axis=0)

    return all_data,missing_vars,needed_vars
#####


def calc_budget_data(current_var, Dic_scn_var_comp, area, trop, inside, num_yrs, duration, AEROSOLS):
    """
    Function to run through desired table values for calculations for the table entries
    """

    # Initialize full data dictionary for current table type
    chem_dict = {}

    # Update variable marker if neccessary
    if current_var == 'SO4':
        specifier = ' S'
    else:
        specifier = ''

    # Calculate values for given variable
    if 'AOD' in current_var:
        # Burden
        spc_burd = Dic_scn_var_comp[current_var][current_var+'_AOD']
        burden = np.ma.masked_where(inside==False,spc_burd)  #convert Kg/m2 to Tg
        BURDEN = np.ma.sum(burden*area)/np.ma.sum(area)
        chem_dict[f"{current_var}_mean"] = np.round(BURDEN,5)
    else:
        # Surface Emissions
        spc_sf = Dic_scn_var_comp[current_var][current_var+'_SF']
        tmp_sf = spc_sf
        sf = np.ma.masked_where(inside==False,tmp_sf*area)  #convert Kg/m2/s to Tg/yr
        SF = np.ma.sum(sf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_EMIS (Tg{specifier}/yr)"] = np.round(SF,5)

        # Elevated Emissions
        spc_clxf = Dic_scn_var_comp[current_var][current_var+'_CLXF']
        tmp_clxf = spc_clxf
        clxf = np.ma.masked_where(inside==False,tmp_clxf*area)  #convert Kg/m2/s to Tg/yr
        CLXF = np.ma.sum(clxf*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_EMIS_elevated (Tg{specifier}/yr)"] = np.round(CLXF,5)

        # Burden
        spc_burd = Dic_scn_var_comp[current_var][current_var+'_BURDEN']
        spc_burd = np.where(np.isnan(trop),np.nan,spc_burd)
        tmp_burden = np.nansum(spc_burd*area,axis=0)
        burden = np.ma.masked_where(inside==False,tmp_burden)  #convert Kg/m2 to Tg
        BURDEN = np.ma.sum(burden*1e-9)
        chem_dict[f"{current_var}_BURDEN (Tg{specifier})"] = np.round(BURDEN,5)

        # Chemical Loss
        spc_chml = Dic_scn_var_comp[current_var][current_var+'_CHML']
        spc_chml = np.where(np.isnan(trop),np.nan,spc_chml)
        tmp_chml = np.nansum(spc_chml*area,axis=0)
        chml = np.ma.masked_where(inside==False,tmp_chml)  #convert Kg/m2/s to Tg/yr
        CHML = np.ma.sum(chml*duration*1e-9)/num_yrs
        chem_dict[f"{current_var}_CHEM_LOSS (Tg{specifier}/yr)"] = np.round(CHML,5)

        # Chemical Production
        if current_var == 'SO4': # chemical production is basically the elevated emissions.
                               # We have removed it for SO4 budget. and put 0 here, so, we don't report it
            chem_dict[f"{current_var}_CHEM_PROD (Tg{specifier}/yr)"] = 0
        else:
            spc_chmp = Dic_scn_var_comp[current_var][current_var+'_CHMP']
            spc_chmp = np.where(np.isnan(trop),np.nan,spc_chmp)
            tmp_chmp = np.nansum(spc_chmp*area,axis=0)
            chmp = np.ma.masked_where(inside==False,tmp_chmp)  #convert Kg/m2/s to Tg/yr
            CHMP = np.ma.sum(chmp*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_CHEM_PROD (Tg{specifier}/yr)"] = np.round(CHMP,5)
        # End if

        # Aerosol calculations
        #---------------------
        if current_var in AEROSOLS:

           # Dry Deposition Flux
            spc_ddfa = Dic_scn_var_comp[current_var][current_var+'_DDF']
            spc_ddfc = Dic_scn_var_comp[current_var][current_var+'_DDFC']
            spc_ddf = spc_ddfa +spc_ddfc
            tmp_ddf = spc_ddf
            ddf = np.ma.masked_where(inside==False,tmp_ddf*area)  #convert Kg/m2/s to Tg/yr
            DDF = np.ma.sum(ddf*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_DRYDEP (Tg{specifier}/yr)"] = np.round(DDF,5)

            # Wet deposition
            spc_wdfa = Dic_scn_var_comp[current_var][current_var+'_WDF']
            spc_wdfc = Dic_scn_var_comp[current_var][current_var+'_WDFC']
            spc_wdf = spc_wdfa +spc_wdfc
            tmp_wdf = spc_wdf
            wdf = np.ma.masked_where(inside==False,tmp_wdf*area)  #convert Kg/m2/s to Tg/yr
            WDF = np.ma.sum(wdf*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_WETDEP (Tg{specifier}/yr)"] = np.round(WDF,5)

            if current_var in ["SOA",'SO4']:
                # gas-aerosol Exchange
                spc_gaex = Dic_scn_var_comp[current_var][current_var+'_GAEX']
                tmp_gaex = spc_gaex
                gaex = np.ma.masked_where(inside==False,tmp_gaex*area)  #convert Kg/m2/s to Tg/yr
                GAEX = np.ma.sum(gaex*duration*1e-9)/num_yrs
                chem_dict[f"{current_var}_GAEX (Tg{specifier}/yr)"] = np.round(GAEX,5)

            # LifeTime = Burden/(loss+deposition)
            LT = BURDEN/(CHML+DDF-WDF)* duration/86400/num_yrs # days
            chem_dict[f"{current_var}_LIFETIME (days)"] = np.round(LT,5)

            if current_var == 'SO4':
                # Aqueous Chemistry
                spc_aqs = Dic_scn_var_comp[current_var][current_var+'_AQS']
                tmp_aqs = spc_aqs
                aqs = np.ma.masked_where(inside==False,tmp_aqs*area)  #convert Kg/m2/s to Tg/yr
                AQS = np.ma.sum(aqs*duration*1e-9)/num_yrs
                chem_dict[f"{current_var}_AQUEOUS (Tg{specifier}/yr)"] = np.round(AQS,5)

                # Nucleation
                spc_nucl = Dic_scn_var_comp[current_var][current_var+'_NUCL']
                tmp_nucl = spc_nucl
                nucl = np.ma.masked_where(inside==False,tmp_nucl*area)  #convert Kg/m2/s to Tg/yr
                NUCL = np.ma.sum(nucl*duration*1e-9)/num_yrs
                chem_dict[f"{current_var}_NUCLEATION (Tg{specifier}/yr)"] = np.round(NUCL,5)

        # Gaseous calculations
        #---------------------
        else:
            # Dry Deposition Flux
            spc_ddf = Dic_scn_var_comp[current_var][current_var+'_DDF']
            tmp_ddf = spc_ddf
            ddf = np.ma.masked_where(inside==False,tmp_ddf*area)  #convert Kg/m2/s to Tg/yr
            DDF = np.ma.sum(ddf*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_DRYDEP (Tg/yr)"] = np.round(DDF,5)

            # Wet Deposition Flux
            spc_wdf = Dic_scn_var_comp[current_var][current_var+'_WDF']
            tmp_wdf = spc_wdf
            wdf = np.ma.masked_where(inside==False,tmp_wdf*area)  #convert Kg/m2/s to Tg/yr
            WDF = -1*np.ma.sum(wdf*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_WETDEP (Tg/yr)"] = np.round(WDF,5)

            # Total Deposition
            TDEP = DDF - WDF
            chem_dict[f"{current_var}_TDEP (Tg/yr)"] = np.round(TDEP,5)

            # LifeTime = Burden/(loss+deposition)
            if current_var == "CH4":
                LT = BURDEN/(CHML+DDF-WDF) # years
                chem_dict[f"{current_var}_LIFETIME (years)"] = np.round(LT,5)
            else:
                if (CHML+DDF-WDF) > 0:
                    if CHML != 0:
                        LT = BURDEN/(CHML+DDF-WDF)*duration/86400/num_yrs # days
                        chem_dict[f"{current_var}_LIFETIME (days)"] = np.round(LT,5)
                    else:
                        # do not report lifetime if chemical loss (for gases) is not included in the model outputs
                        # and put 0 here, so, we don't report it
                        chem_dict[f"{current_var}_LIFETIME (days)"] = 0
                    # End if
                # End if
            # End if

            #NET = CHMP-CHML
            # Chemical Tendency
            spc_tnd = Dic_scn_var_comp[current_var][current_var+'_TEND']
            spc_tnd = np.where(np.isnan(trop),np.nan,spc_tnd)
            tmp_tnd = np.nansum(spc_tnd,axis=0)
            tnd = np.ma.masked_where(inside==False,tmp_tnd)  #convert Kg/s to Tg/yr
            TND = np.ma.sum(tnd*duration*1e-9)/num_yrs
            chem_dict[f"{current_var}_TEND (Tg/yr)"] = np.round(TND,5)

            # O3 dependent calculations
            if current_var == "O3":
                # Stratospheric-Tropospheric Exchange
                STE = DDF-TND
                chem_dict[f"{current_var}_STE (Tg/yr)"] = np.round(STE,5)

                # Lightning NOX production
                spc_lno = Dic_scn_var_comp[current_var][current_var+'_LNO']
                tmp_lno = np.ma.masked_where(inside==False,spc_lno)
                LNO = np.ma.sum(tmp_lno)
                chem_dict[f"{current_var}_LNO (Tg N/yr)"] = np.round(LNO,5)
        # End if (aerosol or gas)
    return chem_dict
#####


def make_table(adfobj, vars, chem_type, Dic_scn_var_comp, areas, trops, case_names, nicknames, durations, insides, num_yrs, AEROSOLS):
    """
    Create CSV table for aeorosols and gases, if applicable

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
    for case in case_names:
        nickname = nicknames[case]

        # Collect row data in a list of dictionaries
        durations[case]
        rows = []
        for current_var in vars:
            chem_dict = calc_budget_data(current_var, Dic_scn_var_comp[case], areas[case], trops[case], insides[case],
                                         num_yrs[case], durations[case], AEROSOLS)

            # Loop through table variables
            for key, val in chem_dict.items():
                if val != 0:  # Skip variables with a value of 0
                    print(f"\t - Variable '{key}' being added to table")
                    rows.append({'variable': key, nickname: np.round(val, 3)})
                else:
                    msg = f"chem/aerosol tables:"
                    msg += f"\n\t - Variable '{key}' has value of 0, will not add to table"
                    adfobj.debug_log(msg)
                # End if
            # End for
        # End for

        # Create the DataFrame for the current case
        table_df = pd.DataFrame(rows)

        if chem_type == 'gases':
            # Replace compound names directly in the DataFrame
            replacements = {
                'MTERP': 'Monoterpene',
                'CH3OH': 'Methanol',
                'CH3COCH3': 'Acetone',
                'O3_LNO': 'LNOx_PROD'
            }
            table_df['variable'] = table_df['variable'].replace(replacements, regex=True)
        # End if

        # Store the DataFrame in the dictionary
        dfs[nickname] = table_df
    # End for

    # Merge the DataFrames on the 'variable' column
    if len(case_names) == 2:
        table_df = pd.merge(dfs[nicknames[case_names[0]]], dfs[nicknames[case_names[1]]], on='variable')

        # Calculate the differences between case columns
        table_df['difference'] = table_df[nicknames[case_names[0]]] - table_df[nicknames[case_names[1]]]

    #Create output file name:
    output_csv_file = output_location / f'ADF_amwg_{chem_type}_table.csv'

    # Save table to CSV and add table dataframe to website (if enabled)
    table_df.to_csv(output_csv_file, index=False)
    adfobj.add_website_data(table_df, chem_type, case, plot_type="Tables")
#####