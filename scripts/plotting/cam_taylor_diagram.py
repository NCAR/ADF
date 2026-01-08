'''
Module: cam_taylor_diagram

Provides a Taylor diagram following the AMWG package. Uses spatial information only.

This module, for better or worse, provides both the computation and plotting functionality.
It depends on an ADF instance to obtain the `climo` files.
It is designed to have one "reference" case (could be observations) and arbitrary test cases.
When multiple test cases are provided, they are plotted with different colors.

NOTE: THIS IS A DRAFT REFACTORING TO ALLOW OBSERVATIONS (.b)
'''
#
# --- imports and configuration ---
#
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geocat.comp as gc  # use geocat's interpolation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import adf_utils as utils

import warnings  # use to warn user about missing files.
warnings.formatwarning = utils.my_formatwarning

#
# --- Main Function Shares Name with Module: cam_taylor_diagram ---
#
def cam_taylor_diagram(adfobj):
    """Create Taylor diagrams for specified configuration."""
    msg = "\n  Generating Taylor Diagrams..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")

    # Extract needed quantities from ADF object:
    # -----------------------------------------
    # Case names:
    # NOTE: "baseline" == "reference" == "observations" will be called `base`
    #       test case(s) == case(s) to be diagnosed  will be called `case` (assumes a list)
    case_names = adfobj.get_cam_info('cam_case_name', required=True)  # Loop over these

    #Grab all case nickname(s)
    test_nicknames = adfobj.case_nicknames["test_nicknames"]

    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]

    # ADF variable which contains the output path for plots and tables:
    plot_location = adfobj.plot_location
    if not plot_location:
        plot_location = adfobj.get_basic_info("cam_diag_plot_loc")
    if isinstance(plot_location, list):
        for pl in plot_location:
            plpth = Path(pl)
            #Check if plot output directory exists, and if not, then create it:
            if not plpth.is_dir():
                print(f"\t    {pl} not found, making new directory")
                plpth.mkdir(parents=True)
        if len(plot_location) == 1:
            plot_loc = Path(plot_location[0])
        else:
            print(f"Ambiguous plotting location since all cases go on same plot. Will put them in first location: {plot_location[0]}")
            plot_loc = Path(plot_location[0])
    else:
        plot_loc = Path(plot_location)


    # reference data set(s) -- if comparing with obs, these are dicts.
    data_name = adfobj.data.ref_case_label
    data_loc = adfobj.data.ref_data_loc
    base_nickname = adfobj.data.ref_nickname

    #Extract baseline years (which may be empty strings if using Obs):
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]

    res = adfobj.variable_defaults # dict of variable-specific plot preferences
    # or an empty dictionary if use_defaults was not specified in YAML.

    #Set plot file type:
    # -- this should be set in basic_info_dict, but is not required
    # -- So check for it, and default to png
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get('plot_type', 'png')
    print(f"\t NOTE: Plot type is set to {plot_type}")

    #Check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info('redo_plot')
    print(f"\t NOTE: redo_plot is set to {redo_plot}")

    # Check for required variables
    taylor_var_set = {'U', 'PSL', 'SWCF', 'LWCF', 'LANDFRAC', 'TREFHT', 'TAUX', 'RELHUM', 'T'}
    available_vars = set(adfobj.diag_var_list)
    missing_vars = taylor_var_set - available_vars
    # Check for precipitation (Needs PRECT OR both PRECL and PRECC)
    has_prect = 'PRECT' in available_vars
    has_precl_precc = {'PRECL', 'PRECC'}.issubset(available_vars)
    if missing_vars or not (has_prect or has_precl_precc):
        print("\tTaylor Diagrams skipped due to missing variables:")
        if missing_vars:
            print(f"\t - Missing: {', '.join(sorted(missing_vars))}")
        if not (has_prect or has_precl_precc):
            if not has_prect:
                print("\t - Missing: PRECT (Alternative PRECL + PRECC also incomplete)")
        print("\n\tFull requirement: U, PSL, SWCF, LWCF, LANDFRAC, TREFHT, TAUX, RELHUM, T,")
        print("\tAND (PRECT OR both PRECL & PRECC)")
        return


    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    # TAYLOR PLOT VARIABLES:
    var_list = ['PSL', 'SWCF', 'LWCF',
                'TropicalLandPrecip', 'TropicalOceanPrecip',
                'Land2mTemperature', 'EquatorialPacificStress',
                'U300', 'ColumnRelativeHumidity', 'ColumnTemperature']

    case_colors = [mpl.cm.tab20(i) for i, case in enumerate(case_names)] # change color for each case
    #
    # LOOP OVER SEASON
    #
    for s in seasons.items():
        plot_name = plot_loc / f"TaylorDiag_{s}_Special_Mean.{plot_type}"
        print(f"\t - Plotting Taylor Diagram, {s}")

        # Check redo_plot. If set to True: remove old plot, if it already exists:
        if (not redo_plot) and plot_name.is_file():
            #Add already-existing plot to website (if enabled):
            adfobj.debug_log(f"'{plot_name}' exists and clobber is false.")
            adfobj.add_website_data(plot_name, "TaylorDiag", None, season=s, multi_case=True)

            #Continue to next iteration:
            continue
        elif (redo_plot) and plot_name.is_file():
            plot_name.unlink()

        # hold the data in a DataFrame for each case
        # variable | correlation | stddev ratio | bias
        df_template = pd.DataFrame(index=var_list, columns=['corr', 'ratio', 'bias'])
        result_by_case = {cname: df_template.copy() for cname in case_names}
        #
        # LOOP OVER VARIABLES
        #
        for v in var_list:
            # Load reference data (already regridded to target grid)
            ref_data = _retrieve(adfobj, v, data_name)
            if ref_data is None:
                print(f"\t WARNING: No regridded reference data for {v} in {data_name}, skipping.")
                continue
            # ASSUMING `time` is 1-12, get the current season:
            ref_data = ref_data.sel(time=s).mean(dim='time')

            for casenumber, case in enumerate(case_names):
                # Load test case data regridded to match reference grid
                case_data = _retrieve(adfobj, v, case)
                if case_data is None:
                    print(f"\t WARNING: No regridded data for {v} in {case}, skipping.")
                    continue
                # ASSUMING `time` is 1-12, get the current season:
                case_data = case_data.sel(time=s).mean(dim='time')
                # Now compute stats (grids are aligned)
                result_by_case[case].loc[v] = taylor_stats_single(case_data, ref_data)
        #
        # -- PLOTTING (one per season) --
        #
        fig, ax = taylor_plot_setup(title=f"Taylor Diagram - {s}",
                                    baseline=f"Baseline: {base_nickname}  yrs: {syear_baseline}-{eyear_baseline}")

        for i, case in enumerate(case_names):
            ax = plot_taylor_data(ax, result_by_case[case], case_color=case_colors[i], use_bias=True)

        ax = taylor_plot_finalize(ax, test_nicknames, case_colors, syear_cases, eyear_cases, needs_bias_labels=True)
        # add text with variable names:
        txtstrs = [f"{i+1} - {v}" for i, v in enumerate(var_list)]
        fig.text(0.9, 0.9, "\n".join(txtstrs), va='top')
        fig.savefig(plot_name, bbox_inches='tight')
        adfobj.debug_log(f"\t Taylor Diagram: completed {s}. \n\t File: {plot_name}")

        #Add plot to website (if enabled):
        adfobj.add_website_data(plot_name, "TaylorDiag", None, season=s, multi_case=True)

    #Notify user that script has ended:
    print("  ...Taylor Diagrams have been generated successfully.")

    return

#
# --- Local Functions ---
#

# --- DERIVED VARIABLES ---

def vertical_average(fld, ps, acoef, bcoef):
    """Calculate weighted vertical average using trapezoidal rule. Uses full column."""
    pres = utils.pres_from_hybrid(ps, acoef, bcoef)
    # integral of del_pressure turns out to be just the average of the square of the boundaries:
    # -- assume lev is a coordinate and is nominally in pressure units
    maxlev = pres['lev'].max().item()
    minlev = pres['lev'].min().item()
    dp_integrated = 0.5 * (pres.sel(lev=maxlev)**2 - pres.sel(lev=minlev)**2)
    levaxis = fld.dims.index('lev')  # fld needs to be a dataarray
    assert isinstance(levaxis, int), f'the axis called lev is not an integer: {levaxis}'
    fld_integrated = np.trapezoid(fld * pres, x=pres, axis=levaxis)
    return fld_integrated / dp_integrated

def find_landmask(adf, casename):
    try:
        return _retrieve(adf, 'LANDFRAC', casename)
    except Exception as e:
        print(f"\t WARNING: Could not find LANDFRAC for {casename}: {e}")
        return None

def get_prect(adf, casename, **kwargs):
    if casename == 'Obs':
        return adf.data.load_reference_regrid_da('PRECT')
    else:
        # Try regridded PRECT first
        prect = adf.data.load_regrid_da(casename, 'PRECT')
        if prect is not None:
            return prect
        # Fallback: derive from PRECC + PRECL using regridded versions
        print("\t Need to derive PRECT = PRECC + PRECL (using regridded data)")
        precc = adf.data.load_regrid_da(casename, 'PRECC')
        precl = adf.data.load_regrid_da(casename, 'PRECL')
        if precc is None or precl is None:
            print(f"\t WARNING: Could not derive PRECT for {casename} (missing PRECC or PRECL)")
            return None
        return precc + precl

def get_tropical_land_precip(adf, casename, **kwargs):
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    prect = get_prect(adf, casename)
    if prect is None:
        return None
    # mask to only keep land locations
    prect = xr.DataArray(np.where(landfrac >= .95, prect, np.nan),
                         dims=prect.dims,
                         coords=prect.coords,
                         attrs=prect.attrs)  # threshold could be 1
    return prect.sel(lat=slice(-30,30))


def get_tropical_ocean_precip(adf, casename, **kwargs):
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    prect = get_prect(adf, casename)
    if prect is None:
        return None
    # mask to only keep ocean locations
    prect = xr.DataArray(np.where(landfrac <= 0.05, prect, np.nan),
                         dims=prect.dims,
                         coords=prect.coords,
                         attrs=prect.attrs)
    return prect.sel(lat=slice(-30,30))


def get_surface_pressure(adf, dset, casename):
    if 'PS' in dset.variables:
        #Just use surface pressure in climo file:
        ps = dset['PS']
    else:
        if casename == 'Obs':
            ps = adf.data.load_reference_regrid_da('PS')
        else:
            ps = adf.data.load_regrid_da(casename, 'PS')    
    if ps is None:
        print(f"\t WARNING: Could not load PS for {casename}.")
        return None    
    return ps


def get_var_at_plev(adf, casename, variable, plev):
    if casename == 'Obs':
        dset = adf.data.load_reference_regrid_da(variable)
        if dset is None or 'lev' not in dset.dims:
            print(f"\t WARNING: Obs data for {variable} lacks lev dimension or is unavailable.")
            return None
        return dset.sel(lev=plev, method='nearest') if dset is not None else None
    else:
        dset = adf.data.load_regrid_da(casename, variable)
        if dset is None:
            return None
        ps = get_surface_pressure(adf, dset, casename)
        if ps is None:
            print(f"\t WARNING: Could not load PS for {variable} interpolation in {casename}")
            return None
        # Proceed with gc.interp_hybrid_to_pressure using regridded data
        # (Assumes hyam/hybm are available in dset or can be loaded similarly)
        vplev = gc.interp_hybrid_to_pressure(dset, ps, dset['hyam'], dset['hybm'],
                                             new_levels=np.array([100. * plev]), lev_dim='lev')
        return vplev.squeeze(drop=True).load()


def get_u_at_plev(adf, casename):
    return get_var_at_plev(adf, casename, "U", 300)


def get_vertical_average(adf, casename, varname):
    '''Collect data from case and use `vertical_average` to get result.'''
    if casename == 'Obs':
        ds = adf.data.load_reference_regrid_da(varname)
        if ds is None or 'lev' not in ds.dims:
            print(f"\t WARNING: Obs data for {varname} lacks lev dimension.")
            return None
        return ds.mean(dim='lev')
    else:
        ds = adf.data.load_regrid_da(casename, varname)
        if ds is None:
            return None
        # Try and extract surface pressure:
        ps = get_surface_pressure(adf, ds, casename)
        if ps is None:
            print(f"\t WARNING: Could not load PS for {varname} interpolation in {casename}")
            return None
        # If the climo file is made by ADF, then hyam and hybm will be with VARIABLE:
        return vertical_average(ds[varname], ps, ds['hyam'], ds['hybm'])


def get_virh(adf, casename, **kwargs):
    '''Calculate vertically averaged relative humidity.'''
    return get_vertical_average(adf, casename, "RELHUM")


def get_vit(adf, casename, **kwargs):
    '''Calculate vertically averaged temperature.'''
    return get_vertical_average(adf, casename, "T")

def get_landt2m(adf, casename):
    if casename == 'Obs':
        t = adf.data.load_reference_regrid_da('TREFHT')
    else:
        t = adf.data.load_regrid_da(casename, 'TREFHT')
    if t is None:
        return None
    landfrac = find_landmask(adf, casename)
    if landfrac is None:
        return None
    t = xr.DataArray(np.where(landfrac >= .95, t, np.nan),
                     dims=t.dims, coords=t.coords, attrs=t.attrs)
    return t



def get_eqpactaux(adf, casename):
    """Gets zonal surface wind stress 5S to 5N."""
    if casename == 'Obs':
        taux = adf.data.load_reference_regrid_da('TAUX')
    else:
        taux = adf.data.load_regrid_da(casename, 'TAUX')
    if taux is None:
        print(f"\t WARNING: Could not load TAUX for {casename}")
        return None
    return taux.sel(lat=slice(-5, 5))


def get_derive_func(fld):
    funcs = {'TropicalLandPrecip': get_tropical_land_precip,
    'TropicalOceanPrecip': get_tropical_ocean_precip,
    'U300': get_u_at_plev,
    'ColumnRelativeHumidity': get_virh,
    'ColumnTemperature': get_vit,
    'Land2mTemperature': get_landt2m,
    'EquatorialPacificStress': get_eqpactaux
    }
    if fld not in funcs:
        print(f"We do not have a method for variable: {fld}.")
        return None
    return funcs[fld]


def _retrieve(adfobj, variable, casename, return_dataset=False):
    """Custom function that retrieves a variable using ADF loaders for grid consistency.
    Returns the variable as a DataArray (or Dataset if return_dataset=True).
    """
    v_to_derive = ['TropicalLandPrecip', 'TropicalOceanPrecip', 'EquatorialPacificStress',
                   'U300', 'ColumnRelativeHumidity', 'ColumnTemperature', 'Land2mTemperature']
    
    try:
        if casename == 'Obs':
            if variable not in v_to_derive:
                da = adfobj.data.load_reference_regrid_da(variable)
            else:
                func = get_derive_func(variable)
                if func is None:
                    print(f"\t WARNING: No derivation function for {variable}.")
                    return None
                da = func(adfobj, 'Obs')  # No location needed
        else:  # Model cases
            if variable not in v_to_derive:
                da = adfobj.data.load_regrid_da(casename, variable)
            else:
                func = get_derive_func(variable)
                if func is None:
                    print(f"\t WARNING: No derivation function for {variable}.")
                    return None
                da = func(adfobj, casename)  # No location needed
        
        if da is None:
            print(f"\t WARNING: Could not load {variable} for {casename}.")
            return None
        
        if return_dataset:
            if not isinstance(da, xr.Dataset):
                da = da.to_dataset(name=variable)
        return da
    
    except Exception as e:
        print(f"\t WARNING: Error retrieving {variable} for {casename}: {e}")
        return None
    

def weighted_correlation(x, y, weights):
    # TODO: since we expect masked fields (land/ocean), need to allow for missing values (maybe works already?)
    mean_x = x.weighted(weights).mean()
    mean_y = y.weighted(weights).mean()
    dev_x = x - mean_x
    dev_y = y - mean_y
    cov_xy = (dev_x * dev_y).weighted(weights).mean()
    cov_xx = (dev_x * dev_x).weighted(weights).mean()
    cov_yy = (dev_y * dev_y).weighted(weights).mean()
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def weighted_std(x, weights):
    """Weighted standard deviation.
    x -> xr.DataArray
    weights -> array-like of weights, probably xr.DataArray
    If weights is not the same shape as x, will use `broadcast_like` to
    create weights array.
    Returns the weighted standard deviation of the full x array.
    """
    xshape = x.shape
    wshape = weights.shape
    if xshape != wshape:
        wa = weights.broadcast_like(x)
    else:
        wa = weights
    mean_x = x.weighted(weights).mean()
    dev_x = x - mean_x
    swdev = (weights * dev_x**2).sum()
    total_weights = wa.where(x.notnull()).sum()
    return np.sqrt(swdev / total_weights)



def taylor_stats_single(casedata, refdata, w=True):
    """This replicates the basic functionality of 'taylor_stats' from NCL.
    input:
        casedata : input data, DataArray
        refdata  : reference case data, DataArray
        w        : if true use cos(latitude) as spatial weight, if false assume uniform weight
    returns:
        pattern_correlation, ratio of standard deviation (case/ref), bias
    """
    lat = casedata['lat']
    if w:
        wgt = np.cos(np.radians(lat))
    else:
        wgt = np.ones(len(lat))
    correlation = weighted_correlation(casedata, refdata, wgt).item()
    a_sigma = weighted_std(casedata, wgt)
    b_sigma = weighted_std(refdata, wgt)
    mean_case = casedata.weighted(wgt).mean()
    mean_ref = refdata.weighted(wgt).mean()
    bias = (100*((mean_case - mean_ref)/mean_ref)).item()
    return correlation, a_sigma/b_sigma, bias


def taylor_plot_setup(title,baseline):
    """Constructs Figure and Axes objects for basic Taylor Diagram."""
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection':'polar'})
    corr_labels = np.array([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.])
    corr_locations = np.pi/2 - np.arccos((corr_labels))  # azim. ticks in radians.
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim([0, 1.6])  # Works better than set_rmin / set_rmax
    ax.set_theta_zero_location("N") # zero at top,
    ax.set_theta_direction(-1)  # angle increases clockwise
    thetalines, thetalabels = ax.set_thetagrids(np.degrees(corr_locations), corr_labels)
    ax.grid(axis='x', linewidth=0)  # turn off radial grid
    ax.set_rgrids(np.arange(0, 1.75, .25))
    # ax.set_yticks([0.0, 1.0, 1.25]) # same effect as set_rgrids()
    ax.set_ylabel("Standardized Deviations")
    # Add tick marks along azimuth
    tick = [ax.get_rmax(),ax.get_rmax()*0.97]
    for t in corr_locations:
        ax.plot([t,t], tick, lw=0.72, color="k")
    ax.text(np.radians(50), ax.get_rmax()*1.1, "Correlation", ha='center', rotation=-50, fontsize=15)
    ax.text(np.radians(95), 1.0, "REF", ha='center')
    st = fig.suptitle(title, fontsize=18)
    st.set_y(1.)
    ax.set_title(baseline, fontsize=10,pad=15)
    return fig, ax


def plot_taylor_data(wks, df, **kwargs):
    """Apply data on top of the Taylor Diagram Axes.
        wks -> Axes object, probably from taylor_plot_setup
        df  -> DataFrame holding the Taylor stats.
        kwargs -> optional arguments
          look for 'use_bias'
          look for 'case_color'
    """
    # option is whether to stylize the markers by the bias:
    use_bias = False
    if 'use_bias' in kwargs:
        if kwargs['use_bias']:
            use_bias = True
            df['bias_digi'] = np.digitize(df['bias'].values, [-20, -10, -5, -1, 1, 5, 10, 20])
            marker_list = ["v", "v", "v", "v", "o", "^", "^", "^", "^"]
            marker_size = [24, 16, 8, 4, 4, 4, 8, 16, 24]
    # option: has color been specified as case_color?
    # --> expect the case labeling to be done external to this function
    if 'case_color' in kwargs:
        color = kwargs['case_color']
        if isinstance(color, int):
            # assume we should use this as an index
            color = mpl.cm.tab20(color) # convert to RGBA
            # TODO: allow colormap to be specified.
    annos = []  # list will hold strings for legend
    k = 1
    for ndx, row in df.iterrows():
        # NOTE: ndx will be the DataFrame index, and we expect that to be the variable name
        if np.isnan(row['corr']) or np.isnan(row['ratio']):
            continue  # Skip plotting if data is missing
        theta = np.pi/2 - np.arccos(row['corr'])  # Transform DATA
        if use_bias:
            mk = marker_list[row['bias_digi']]
            mksz = marker_size[row['bias_digi']]
            wks.plot(theta, row['ratio'], marker=mk, markersize=mksz, color=color)
        else:
            wks.plot(theta, row['ratio'], marker='o', markersize=16, color=color)
        annos.append(f"{k} - {ndx.replace('_','')}")
        wks.annotate(str(k), (theta, row['ratio']), ha='center', va='bottom',
                            xytext=(0,5), textcoords='offset points', fontsize='x-large',
                            color=color)
        k += 1  # increment the annotation number (THIS REQUIRES CASES TO HAVE SAME ORDER IN DataFrame)
    return wks


def taylor_plot_finalize(wks, test_nicknames, casecolors, syear_cases, eyear_cases, needs_bias_labels=True):
    """Apply final formatting to a Taylor diagram.
        wks -> Axes object that has passed through taylor_plot_setup and plot_taylor_data
        casenames -> list of case names for the legend
        casecolors -> list of colors for the cases
        needs_bias_labels -> Bool, if T make the legend for the bias-sized markers.
    """
    # CASE LEGEND -- Color-coded
    bottom_of_text = 0.05

    height_of_lines = 0.03
    wks.text(0.052, 0.08, "Cases:",
            color='k', ha='left', va='bottom', transform=wks.transAxes, fontsize=11)
    n = 0
    for case_idx, (s, c) in enumerate(zip(test_nicknames, casecolors)):

            wks.text(0.052, bottom_of_text + n*height_of_lines, f"{s}  yrs: {syear_cases[case_idx]}-{eyear_cases[case_idx]}",
            color=c, ha='left', va='bottom', transform=wks.transAxes, fontsize=10)
            n += 1
    # BIAS LEGEND
    if needs_bias_labels:
        # produce an info-box showing the markers/sizes based on bias
        bias_legend_elements = [(Line2D([0], [0], marker="v", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0), Line2D([0], [0], marker="^", color='k', label="> 20%", markersize=24, fillstyle='none', linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="10-20%", markersize=16, linewidth=0), Line2D([0], [0], marker="^", color='k', label="10-20%", markersize=16, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label="5-10%", markersize=8, linewidth=0), Line2D([0], [0], marker="^", color='k', label="5-10%", markersize=8, linewidth=0)),
                                (Line2D([0], [0], marker="v", color='k', label=">1-5%", markersize=4, linewidth=0), Line2D([0], [0], marker="^", color='k', label=">1-5%", markersize=4, linewidth=0)),
                                Line2D([0], [0], marker="o", color='k', label="< 1%", markersize=4, linewidth=0),
                                ]
        bias_legend_labels = ["> 20%", "10-20%", "5-10%", "1-5%", "< 1%"]
        wks.legend(handles=bias_legend_elements, labels=bias_legend_labels, loc='upper left', handler_map={tuple: HandlerTuple(ndivide=None, pad=2.)}, labelspacing=2, handletextpad=2, frameon=False, title=" - / + Bias",
                    title_fontsize=18)
    return wks