"""
adf_histogram

Create histogram(s) of all specified (2D) variables.

- Constructs global histograms: total, ocean, land.
  + for consistency with other ADF, will default to making annual and seasonal histograms.
- Saves the resulting histogram data as netCDF (into the plots directory by default)
- Plots the histograms.

Options:
  - specify whether to do spatial distribution from climo files
    or combine temporal and spatial using time series files

Possible future enhancements:
  - specify a landmask file
  - specific histogram options in the variable defaults file (looks for hist_bins)
  - ability to specify additional regions (?)
  - specify the output location for netCDF files
  - improved detection of dimensions that (like cosp height)
  - allow for rgridding; right now if LANDFRAC and data are not the same, just skips.

"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import plotting_functions as pf
import warnings  # use to warn user about missing files.
def my_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'
warnings.formatwarning = my_formatwarning

#
# USER-ADJUSTABLE PARAMETERS:
#

# climo or time series:
use_time_series = True
# climo files are much smaller, so calculation will be faster,
# time series are usually more statistically robust.

add_dimension_annotation = True  # will print the dimensions of the input data on the plot

redo_histogram_files = True

if use_time_series:
    plot_name_string = "GlobalHistogramTS"
else:
    plot_name_string = "GlobalHistogramClimo"

#Set seasonal ranges:
seasons = {"ANN": np.arange(1,13,1),
            "DJF": [12, 1, 2],
            "JJA": [6, 7, 8],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11]}

def adf_histogram(adfobj):
    var_list = adfobj.diag_var_list

    if use_time_series:
        load_ref_func = adfobj.data.load_reference_timeseries_da # field [this is a mistake, the args should have been the same for all of these functions]
        load_func = adfobj.data.load_timeseries_da # case, variablename
    else:
        load_ref_func = adfobj.data.load_reference_climo_da # case, variablename
        load_func = adfobj.data.load_climo_da # case, variablename

    def get_load_args(adfobj, case, variablename):
        if use_time_series and (case == adfobj.data.ref_case_label):
            return (variablename, )
        else:
            return (case, variablename)

    # Standard ADF stuff:
    plot_locations = adfobj.plot_location
    syear_cases = adfobj.climo_yrs["syears"]
    eyear_cases = adfobj.climo_yrs["eyears"]
    syear_baseline = adfobj.climo_yrs["syear_baseline"]
    eyear_baseline = adfobj.climo_yrs["eyear_baseline"]
    res = adfobj.variable_defaults
    basic_info_dict = adfobj.read_config_var("diag_basic_info")
    plot_type = basic_info_dict.get("plot_type", "png")
    print(f"\t NOTE: Plot type is set to {plot_type}")
    res = adfobj.variable_defaults

    # check if existing plots need to be redone
    redo_plot = adfobj.get_basic_info("redo_plot")
    print(f"\t NOTE: redo_plot is set to {redo_plot}")
    #
    # SECTION 1: determine which plots need to be made
    #
    for case_idx, case_name in enumerate(adfobj.data.case_names):
        # Set output plot location:
        plot_loc = Path(plot_locations[case_idx])
        print(f"PLOT LOCATION: {plot_loc}")
        # Loop over the variables for each season
        skip_make_plot = []
        for var in var_list:
            for s in seasons:
                plot_name = plot_loc / f"{var}_{s}_{plot_name_string}_Mean.{plot_type}"
                plot_exists = plot_name.is_file()
                print(
                    f"Projected file name: {plot_name.name}. Exists: {plot_exists}"
                )
                if plot_exists and (not redo_plot):
                    skip_make_plot.append(plot_name)
    # make histogram files for each variable
    # output: variable(season, region, bin)
    
    # "reference case" first:
    ref_land = load_ref_func(*get_load_args(adfobj, adfobj.data.ref_case_label, "LANDFRAC"))
    for var in var_list:

        ref_hist_file = plot_loc / f"{adfobj.data.ref_case_label}_{var}_{plot_name_string}.nc"
        ref_histogram_file_exists = ref_hist_file.is_file()

        if var in res:
            vres = res[var]
            adfobj.debug_log(f"adf_histogram: Found variable defaults for {var}")
        else:
            vres = {}

        # probably have to make sure no "lev" dim (but gets confused about other dimensions)
        da = load_ref_func(*get_load_args(adfobj, adfobj.data.ref_case_label, var))
        if da is None:
            print(f"failed to load {var} for {adfobj.data.ref_case_label}... skip")
            continue

        if ("lev" in da.dims) or ("ilev" in da.dims):
            print(f"{var}: Looks like lev/ilev present... skip")
            continue
        
        has_lat_lon = pf.lat_lon_validate_dims(da)
        if not has_lat_lon:
            print(f"INFO: {var} looks like it is on unstructured mesh. Has ncol: {'ncol' in da.dims}. Histogram does not need to regrid.")

        if (not ref_histogram_file_exists) or (redo_histogram_files):
            ref_result = make_histograms(da, ref_land, vres)
            print(f"Writing Reference Case Histogram: {ref_hist_file}")
            ref_result.to_netcdf(ref_hist_file)
        else:
            print("reference histogram file exists, will use that one.")

    for case_idx, case_name in enumerate(adfobj.data.case_names):
        case_land = load_func(*get_load_args(adfobj, case_name, "LANDFRAC"))
        for var in var_list:
            if var in res:
                vres = res[var]
                adfobj.debug_log(f"adf_histogram: Found variable defaults for {var}")
            else:
                vres = {}
            hist_file = plot_loc / f"{case_name}_{var}_{plot_name_string}.nc"
            histogram_file_exists = hist_file.is_file()
            if (not histogram_file_exists) or (redo_histogram_files):
                da = load_func(*get_load_args(adfobj, case_name, var))
                if da is None:
                    print(f"Failed to load {var} for {case_name}... skip")
                    continue
                if ("lev" in da.dims) or ("ilev" in da.dims):
                    print(f"{var}: Looks like lev/ilev present... skip")
                    continue
                has_lat_lon = pf.lat_lon_validate_dims(da)
                if not has_lat_lon:
                    print(f"INFO: {var} looks like it is on unstructured mesh. Has ncol: {'ncol' in da.dims}. Histogram does not need to regrid.")
                
                result = make_histograms(da, case_land, vres)
                print(f"Writing Case Histogram: {hist_file}")
                result.to_netcdf(hist_file)
            else:
                print(f"{case_name} histogram file exists, will use that one.")

    print(f"HISTOGRAM PLOT LOCATION:\n{plot_loc}")
    for var in var_list:
        ref_hist_file = plot_loc / f"{adfobj.data.ref_case_label}_{var}_{plot_name_string}.nc"
        if not ref_hist_file.is_file():
            print(f"ERROR: histogram file not found for {var}, {adfobj.data.ref_case_label}... skip.")
            continue
        ref_h_ds = xr.open_dataset(ref_hist_file)['histogram']
        add_annot = False
        if "input_dims" in ref_h_ds.attrs:
            add_annot = True
            annot = f"input dimensions: {ref_h_ds.attrs['input_dims']}"
        case_hist_files = [plot_loc / f"{case_name}_{var}_{plot_name_string}.nc" for case_name in adfobj.data.case_names]
        if not all([f.is_file() for f in case_hist_files]):
            print(f"ERROR: histogram files not found for {var}, {adfobj.data.case_names}... skip")
            continue
        case_h_ds = {c: xr.open_dataset(case_hist_files[i])['histogram'] for i, c in enumerate(adfobj.data.case_names)}

        for season in seasons:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            regions = ['ALL', 'OCEAN', 'LAND']
            for i, region in enumerate(regions):
                ax = axes[i]
                ref_hist = ref_h_ds.sel(season=season, region=region)
                ax.step(ref_hist['bin'], ref_hist, where='mid', label=adfobj.data.ref_case_label, linestyle='--', color='gray')
                for case_name, case_ds in case_h_ds.items():
                    case_hist = case_ds.sel(season=season, region=region)
                    ax.step(case_hist['bin'], case_hist, where='mid', label=case_name)
                ax.set_title(f"{region} - {season}")
                ax.set_xlabel(var)
                if i == 0:
                    ax.set_ylabel('Frequency')
                ax.set_ylim([0,1])
                ax.legend()
            fig.suptitle(f"{var} Histogram - {season}")
            if add_annot:
                # Add the annotation in the lower right corner
                axes[-1].annotate(annot, 
                    xy=(0.8, -0.02),             # Coordinates of the point to annotate (lower right corner of the figure)
                    xycoords='figure fraction', # Coordinate system is the figure fraction
                    xytext=(-10, 10),         # Offset the text from the point, in points
                    textcoords='offset points',
                    ha='right',             # Horizontal alignment: right
                    va='bottom')            # Vertical alignment: bottom

            plot_name = plot_loc / f"{var}_{season}_{plot_name_string}_Mean.{plot_type}"
            fig.savefig(plot_name, bbox_inches='tight', dpi=72)
            plt.close(fig)
            adfobj.add_website_data(
                plot_name,
                var,
                None,
                season=season,
                multi_case=True,
                plot_type=plot_name_string,
            )
            print(f"\t Saved {var} Histogram for {season}: {plot_name.name}")

def make_histograms(data, land, vres):

    do_region_masks = True
    if land.shape != data.shape:
        print("LAND and DATA are different shapes... will not do region masking.")
        do_region_masks = False

    # determine the bins
    if 'hist_bins' in vres:
        bins = vres['hist_bins']
    elif 'contour_levels_range' in vres:
        bins = np.arange(*vres['contour_levels_range'])
    else:
        print("WARNING: no sensible defaults found -- histogram will use 25 bins (bins may differ across cases)")
        bins = np.linspace(data.min().values, data.max().values, 26)

    # extend bins to catch all values
    hbins = np.insert(bins, 0, np.finfo(float).min)
    hbins = np.append(hbins, np.finfo(float).max)
    # Usually you would want bin center values like so:
    # bin_centers = (hbins[:-1] + hbins[1:]) / 2
    # but if we use massive numbers for the endpoints, then make more sensible "centers":
    bin_centers = bins[0:-1] + 0.5 * np.diff(bins)
    bin_centers = np.insert(bin_centers, 0, bins[0] - 0.5 * (bins[1] - bins[0]))
    bin_centers = np.append(bin_centers, bins[-1] + 0.5 * (bins[-1] - bins[-2]))

    # create masks
    # seasons = ['DJF', 'MAM', 'JJA', 'SON', 'ANN']
    if use_time_series:
        season_masks = {
            'DJF': (data.time.dt.month.isin([12, 1, 2])),
            'MAM': (data.time.dt.month.isin([3, 4, 5])),
            'JJA': (data.time.dt.month.isin([6, 7, 8])),
            'SON': (data.time.dt.month.isin([9, 10, 11])),
            'ANN': (np.ones_like(data.time) == 1)
        }
    else:
        season_masks = {
            'DJF': (data.time.isin([12, 1, 2])),
            'MAM': (data.time.isin([3, 4, 5])),
            'JJA': (data.time.isin([6, 7, 8])),
            'SON': (data.time.isin([9, 10, 11])),
            'ANN': (np.ones_like(data.time) == 1)
        }
    
    region_masks = {
        'ALL': (np.ones_like(land) == 1),
        'OCEAN': (land <= 0),
        'LAND': (land > 0)
    }

    # calculate histograms
    hist_data = np.zeros((len(seasons), len(region_masks), len(bin_centers)))
    for i, season in enumerate(seasons):
        for j, (region, rmask) in enumerate(region_masks.items()):
            if do_region_masks:
                masked_data = xr.where(rmask, data, np.nan)
            else:
                if region == 'ALL':
                    masked_data = data
                else:
                    hist_data[i, j, :] = np.nan
                    continue
            masked_data = masked_data.isel(time=season_masks[season]) # data.where(season_masks[season])
            hist, _ = np.histogram(masked_data, bins=hbins, density=True)
            hist = hist * np.diff(hbins)  # convert to probability mass function
            hist_data[i, j, :] = hist
    # create DataArray
    histogram_da = xr.DataArray(
        hist_data,
        dims=['season', 'region', 'bin'],
        coords={
            'season': list(seasons.keys()),
            'region': list(region_masks.keys()),
            'bin': bin_centers,
        },
        attrs={'input_dims':f"({','.join(list(data.dims))})"},
        name='histogram'
    )
    
    return histogram_da