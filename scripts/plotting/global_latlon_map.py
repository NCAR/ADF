"""
Generate global maps of 2-D fields

Functions
---------
global_latlon_map(adfobj)
    use ADF object to make maps
my_formatwarning(msg, *args, **kwargs)
    format warning messages
    (private method)
plot_file_op
    Check on status of output plot file.
"""
# Import standard modules:
from pathlib import Path
import numpy as np
import warnings

# Import local modules:
import plotting_functions as pf
from .aod_latlon import aod_latlon

# Format warning messages:
def my_formatwarning(msg, *args, **kwargs):
    """Issue `msg` as warning."""
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

#########

def global_latlon_map(adfobj):
    """
    This script/function is designed to generate global
    2-D lat/lon maps of model fields with continental overlays.

    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    Returns
    -------
    Does not return a value; produces plots and saves files.

    Notes
    -----

    It uses the AdfDiag object's methods to get necessary information.
    Makes use of AdfDiag's data sub-class.
    Explicitly accesses:
    adfobj.diag_var_list
        List of variables
    adfobj.plot_location
        output plot path
    adfobj.climo_yrs
        start and end climo years of the case(s), `syears` & `eyears`
        start and end climo years of the reference, `syear_baseline` & `eyear_baseline`
    adfobj.variable_defaults 
        dict of variable-specific plot preferences
    adfobj.read_config_var
        dict of basic info, `diag_basic_info`
        Then use to check `plot_type`
    adfobj.debug_log
        Issues debug message
    adfobj.add_website_data
        Communicates information to the website generator
    adfobj.compare_obs
        Logical to determine if comparing to observations

        
    The `plotting_functions` module is needed for:
    pf.get_central_longitude()
        determine central longitude for global plots
    pf.lat_lon_validate_dims()
        makes sure latitude and longitude are valid
    pf.seasonal_mean()
        calculate seasonal mean
    pf.plot_map_and_save()
        send information to make the plot and save the file
    pf.zm_validate_dims()
        Checks on pressure level dimension
    """

    msg = "\n  Generating lat/lon maps..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Get configuration
    config = get_plot_config(adfobj)
    
    # Process regular variables
    for var in adfobj.diag_var_list:
        process_variable(adfobj, var, **config)
        
    # Handle AOD special case
    if "AODVISdn" in adfobj.diag_var_list:
        print("\tRunning AOD panel diagnostics against MERRA and MODIS...")
        aod_latlon(adfobj)
        
    print("  ...lat/lon maps have been generated successfully.")


def process_variable(adfobj, var, seasons, pres_levs, plot_type, redo_plot)
        vres = adfobj.variable_defaults.get(var, {})
        web_category = vres.get("category", None)

        # For global maps, also set the central longitude:
        # can be specified in adfobj basic info as 'central_longitude' or supplied as a number,
        # otherwise defaults to 180
        vres['central_longitude'] = pf.get_central_longitude(adfobj)

        # Load reference data
        odata = load_reference_data(adfobj, var)
        if odata is None:
            continue

        #Loop over model cases:
        for case_idx, case_name in enumerate(adfobj.data.case_names):
            process_case(adfobj, case_name, case_idx, var, odata, 
                        seasons, pres_levs, plot_type, redo_plot,
                        vres, web_category)



def load_reference_data(adfobj, var):
    """Load and validate reference data."""
    if not adfobj.compare_obs:
        base_name = adfobj.data.ref_case_label
    else:
        if var not in adfobj.data.ref_var_nam:
            dmsg = f"\t    WARNING: No obs data found for variable `{var}`, global lat/lon mean plotting skipped."
            adfobj.debug_log(dmsg)
            print(dmsg)                return None
        base_name = adfobj.data.ref_labels[var]

    odata = adfobj.data.load_reference_regrid_da(base_name, var)
    if odata is None:
        print(f"\t    WARNING: No reference data found for {var}")
        return None

    o_has_dims = pf.validate_dims(odata, ["lat", "lon", "lev"])
    if (not o_has_dims['has_lat']) or (not o_has_dims['has_lon']):
        print(f"\t    WARNING: Reference data missing lat/lon dimensions")
        return None
        
    return odata


def process_case(adfobj, case_name, case_idx, var, odata, seasons, 
                pres_levs, plot_type, redo_plot, vres, web_category):
    """Process individual case data and generate plots."""
    plot_loc = Path(adfobj.plot_location[case_idx])
    plot_loc.mkdir(parents=True, exist_ok=True)

    mdata = adfobj.data.load_regrid_da(case_name, var)
    if mdata is None:
        return

    has_dims = pf.validate_dims(mdata, ["lat", "lon", "lev"])
    if not pf.lat_lon_validate_dims(mdata):
        print(f"\t    WARNING: Model data missing lat/lon dimensions")
        return

    # Check pressure levels if 3D data
    if has_dims['has_lev'] and not pres_levs:
        print(f"\t    WARNING: 3D variable found but no pressure levels specified")
        return

    process_plots(adfobj, mdata, odata, case_name, case_idx,
                 var, seasons, pres_levs, plot_loc, plot_type,
                 redo_plot, vres, web_category, has_dims)


def get_plot_config(adfobj):
    """Get plotting configuration from ADF object."""
    return {
        'seasons': {
            "ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]
        },
        'plot_type': adfobj.read_config_var("diag_basic_info").get('plot_type', 'png'),
        'redo_plot': adfobj.get_basic_info('redo_plot'),
        'pres_levs': adfobj.get_basic_info("plot_press_levels")
    }


def process_seasonal_data(mdata, odata, season, weight_season=True):
    """Helper function to calculate seasonal means and differences."""
    if weight_season:
        mseason = pf.seasonal_mean(mdata, season=season, is_climo=True)
        oseason = pf.seasonal_mean(odata, season=season, is_climo=True)
    else:
        mseason = mdata.sel(time=seasons[s]).mean(dim='time')
        oseason = odata.sel(time=seasons[s]).mean(dim='time')
    
    # Calculate differences
    dseason = mseason - oseason
    
    # Calculate percent change
    pseason = (mseason - oseason) / np.abs(oseason) * 100.0
    pseason = pseason.where(np.isfinite(pseason), np.nan)
    
    return mseason, oseason, dseason, pseason


def plot_file_op(adfobj, plot_name, var, case_name, season, web_category, redo_plot, plot_type):
    """Check if output plot needs to be made or remade.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object that contains all the configuration information

    plot_name : Path
        path of the output plot

    var : str
        name of variable

    case_name : str
        case name
    
    season : str
        season being plotted

    web_category : str
        the category for this variable

    redo_plot : bool
        whether to overwrite existing plot with this file name

    plot_type : str
        the file type for the output plot

    Returns
    -------
    int, None
        Returns 1 if existing file is removed or no existing file.
        Returns None if file exists and redo_plot is False

    Notes
    -----
    The long list of parameters is because add_website_data is called
    when the file exists and will not be overwritten.
    
    """
    # Check redo_plot. If set to True: remove old plot, if it already exists:
    if plot_name.is_file():
        if redo_plot:
            plot_name.unlink()
            return True
        else:
            #Add already-existing plot to website (if enabled):
            adfobj.add_website_data(plot_name, var, case_name, category=web_category,
                                    season=season, plot_type=plot_type)
            return False  # False tells caller that file exists and not to overwrite
    else:
        return True


def process_plots(adfobj, mdata, odata, case_name, case_idx, var, seasons, 
                 pres_levs, plot_loc, plot_type, redo_plot, vres, web_category, has_dims):
    """Process and generate plots for different seasons and pressure levels.
    
    Parameters
    ----------
    adfobj : AdfDiag
        The diagnostics object containing configuration
    mdata : xarray.DataArray  
        Model data
    odata : xarray.DataArray
        Reference/observation data
    case_name : str
        Name of current case
    case_idx : int
        Index of current case
    var : str
        Variable name
    seasons : dict
        Dictionary of season definitions
    pres_levs : list
        Pressure levels to plot
    plot_loc : Path
        Output plot directory
    plot_type : str
        Plot file type (e.g. 'png')
    redo_plot : bool
        Whether to regenerate existing plots
    vres : dict
        Variable-specific plot settings
    web_category : str
        Category for website organization
    has_dims : dict
        Dictionary indicating which dimensions exist in data
        
    Returns
    -------
    None
    """
    # Get case nickname and years
    case_nickname = adfobj.data.test_nicknames[case_idx]
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]
    
    # Check if files exist and build doplot dict
    doplot = check_existing_plots(adfobj, var, plot_loc, plot_type, 
                                case_name, seasons, pres_levs, 
                                has_dims, web_category, redo_plot)
    
    if not any(value is None for value in doplot.values()):
        print(f"\t    INFO: All plots exist for {var}. Redo is {redo_plot}. Existing plots added to website data.")
        return

    # Initialize seasonal data dictionaries
    mseasons = {}
    oseasons = {}
    dseasons = {} 
    pseasons = {}

    if not has_dims['has_lev']:
        # Process 2D data
        process_2d_plots(adfobj, mdata, odata, case_name, case_nickname,
                        var, seasons, plot_loc, plot_type, doplot,
                        mseasons, oseasons, dseasons, pseasons,
                        syear_cases[case_idx], eyear_cases[case_idx],
                        syear_baseline, eyear_baseline,
                        web_category, vres)
    else:
        # Process 3D data with pressure levels
        process_3d_plots(adfobj, mdata, odata, case_name, case_nickname, 
                        var, seasons, pres_levs, plot_loc, plot_type, doplot,
                        mseasons, oseasons, dseasons, pseasons,
                        syear_cases[case_idx], eyear_cases[case_idx],
                        syear_baseline, eyear_baseline,
                        web_category, vres)

def check_existing_plots(adfobj, var, plot_loc, plot_type, case_name, 
                        seasons, pres_levs, has_dims, web_category, redo_plot):
    """Check which plots need to be generated."""
    doplot = {}
    
    if not has_dims['has_lev']:
        for s in seasons:
            plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"
            doplot[plot_name] = plot_file_op(adfobj, plot_name, var, 
                                           case_name, s, web_category, 
                                           redo_plot, "LatLon")
    else:
        for pres in pres_levs:
            for s in seasons:
                plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"
                doplot[plot_name] = plot_file_op(adfobj, plot_name, 
                                               f"{var}_{pres}hpa",
                                               case_name, s, web_category, 
                                               redo_plot, "LatLon")
    return doplot

def process_2d_plots(adfobj, mdata, odata, case_name, case_nickname,
                    var, seasons, plot_loc, plot_type, doplot,
                    mseasons, oseasons, dseasons, pseasons,
                    syear_case, eyear_case, syear_baseline, eyear_baseline,
                    web_category, vres):
    """Process and generate 2D plots."""
    for s in seasons:
        plot_name = plot_loc / f"{var}_{s}_LatLon_Mean.{plot_type}"
        if doplot[plot_name] is None:
            continue
            
        # Calculate seasonal means and differences
        mseasons[s], oseasons[s], dseasons[s], pseasons[s] = \
            process_seasonal_data(mdata, odata, s)

        # Generate plot
        pf.plot_map_and_save(plot_name, case_nickname, adfobj.data.ref_nickname,
                            [syear_case, eyear_case],
                            [syear_baseline, eyear_baseline],
                            mseasons[s], oseasons[s], dseasons[s], pseasons[s],
                            obs=adfobj.compare_obs, **vres)

        # Add to website
        adfobj.add_website_data(plot_name, var, case_name, 
                               category=web_category,
                               season=s, plot_type="LatLon")

def process_3d_plots(adfobj, mdata, odata, case_name, case_nickname,
                    var, seasons, pres_levs, plot_loc, plot_type, doplot,
                    mseasons, oseasons, dseasons, pseasons, 
                    syear_case, eyear_case, syear_baseline, eyear_baseline,
                    web_category, vres):
    """Process and generate 3D plots with pressure levels."""
    for pres in pres_levs:
        # Validate pressure level exists
        if (not (pres in mdata['lev'])) or (not (pres in odata['lev'])):
            print(f"\t    WARNING: plot_press_levels value '{pres}' not present " 
                  f"in {var} [test: {(pres in mdata['lev'])}, "
                  f"ref: {pres in odata['lev']}], so skipping.")
            continue

        for s in seasons:
            plot_name = plot_loc / f"{var}_{pres}hpa_{s}_LatLon_Mean.{plot_type}"
            if doplot[plot_name] is None:
                continue

            # Calculate seasonal means and differences
            mseasons[s], oseasons[s], dseasons[s], pseasons[s] = \
                process_seasonal_data(mdata, odata, s)

            # Generate plot
            pf.plot_map_and_save(plot_name, case_nickname, adfobj.data.ref_nickname,
                                [syear_case, eyear_case],
                                [syear_baseline, eyear_baseline],
                                mseasons[s].sel(lev=pres), 
                                oseasons[s].sel(lev=pres),
                                dseasons[s].sel(lev=pres),
                                pseasons[s].sel(lev=pres),
                                obs=adfobj.compare_obs, **vres)

            # Add to website
            adfobj.add_website_data(plot_name, f"{var}_{pres}hpa",
                                   case_name, category=web_category,
                                   season=s, plot_type="LatLon")