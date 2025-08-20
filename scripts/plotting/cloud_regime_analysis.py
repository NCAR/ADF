
from math import ceil
import warnings
from pathlib import Path

import numpy as np
import xesmf

try:
    import wasserstein
except:
    print(
        "  Wasserstein package is not installed so wasserstein distance cannot be used. Attempting to use wasserstein distance will raise an error."
    )
    print(
        "  To use wasserstein distance please install the wasserstein package in your environment: https://pypi.org/project/Wasserstein/ "
    )
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

import dask
import xarray as xr
import matplotlib as mpl
from shapely.geometry import Point
from shapely.prepared import prep

try:
    from numba import njit
except ImportError:

    def njit(func=None):
        if func is None:
            return njit
        # Issue a warning that numba is not available
        warnings.warn(
            "NumPy performance optimization using Numba is not available. "
            "Fallback to standard Python execution.",
            UserWarning,
        )
        return func


# --- bpm refactor
# --- set up dataclass to get metadata per variable:
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableNames:
    product_name: str
    data_var: str
    ht_var: str
    tau_var: str
    obs_data_var: str
    obs_ht_var: str
    obs_tau_var: str


# Consolidate data into a single dictionary of dataclass objects
ALL_VARS = {
    "FISCCP1_COSP": VariableNames(
        product_name="ISCCP",
        data_var="FISCCP1_COSP",
        ht_var="cosp_prs",
        tau_var="cosp_tau",
        obs_data_var="n_pctaudist",
        obs_ht_var="levtau",
        obs_tau_var="levpc",
    ),
    "CLD_MISR": VariableNames(
        product_name="MISR",
        data_var="CLD_MISR",
        ht_var="cosp_htmisr",
        tau_var="cosp_tau",
        obs_data_var="clMISR",
        obs_ht_var="tau",
        obs_tau_var="cth",
    ),
    "CLMODIS": VariableNames(
        product_name="MODIS",
        data_var="CLMODIS",
        ht_var="cosp_prs",
        tau_var="cosp_tau_modis",
        obs_data_var="MODIS_CLD_HISTO",
        obs_ht_var="COT",
        obs_tau_var="PRES",
    ),
}
# ---


def cloud_regime_analysis(
    adf,
    wasserstein_or_euclidean="euclidean",
    premade_cloud_regimes=None,
    lat_range=None,
    lon_range=None,
    only_ocean_or_land=False,
):
    """
    This script/function is designed to generate 2-D lat/lon maps of Cloud Regimes (CRs), as well as plots of the CR
    centers themselves. It can fit data into CRs using either Wasserstein (AKA Earth Movers Distance) or the more conventional
    Euclidean distance. To use this script, the user should add the appropriate COSP variables to the diag_var_list in the yaml file.
    The appropriate variables are FISCCP1_COSP for ISCCP, CLD_MISR for MISR, and CLMODIS for MODIS. All three should be added to
    diag_var_list if you wish to perform analysis on all three. The user can also specify to perform analysis for just one or for
    all three of the data products (ISCCP, MODIS, and MISR) that there exists COSP output for. A user can also choose to use only
    a specfic lat and lon range, or to use data only over water or over land. Lastly if a user has CRs that they have custom made,
    these can be passed in and the script will fit data into them rather than the premade CRs that the script already points to.
    There are a total of 6 sets of premade CRs, two for each data product. One set made with euclidean distance and one set made
    with Wasserstein distance for ISCCP, MODIS, and MISR. Therefore when the wasserstein_or_euclidean variables is changed it is
    important to undertand that not only the distance metric used to fit data into CRs is changing, but also the CRs themselves
    unless the user is passing in a set of premade CRs with the premade_cloud_regimes variable.

    Description of kwargs:
    wasserstein_or_euclidean        -> Whether to use wasserstein or euclidean distance to fit CRs, enter "wasserstein" for wasserstein or
                                       "euclidean" for euclidean. This also changes the default CRs that data is fit into from ones created
                                       with kmeans using euclidean distance to ones using kmeans with wasserstein distance.  Default is euclidean distance.
    premade_cloud_regimes           -> If the user wishes to use custom CRs rather than the pre-loaded ones, enter them here as a path to a numpy
                                       array of shape (k, n_tau_bins * n_pressure_bins)
    lat_range                       -> Range of latitudes to use enetered as a list, Ex. [-30,30]. Default is use all available latitudes
    lon_range                       -> Range of longitudes to use enetered as a list, Ex. [-90,90]. Default is use all available longitudes
    only_ocean_or_land              -> Set to "O" to perform analysis with only points over water, "L" for only points over land, or False
                                       to use data over land and water. Default is False
    """
    dask.config.set({"array.slicing.split_large_chunks": False})

    # Plot LatLon plots of the frequency of occrence of the baseline/obs and test case
    def plot_rfo_obs_base_diff(cluster_labels, cluster_labels_d, adf, field=None):
        k = cluster_labels.attrs.get("k")
        COLOR = "black"
        mpl.rcParams["text.color"] = COLOR
        mpl.rcParams["axes.labelcolor"] = COLOR
        mpl.rcParams["xtick.color"] = COLOR
        mpl.rcParams["ytick.color"] = COLOR
        plt.rcParams.update({"font.size": 13})
        plt.rcParams["figure.dpi"] = 500
        fig_height = 7

        # Comparing obs or baseline?
        if adf.compare_obs == True:
            obs_or_base = "Observation"
        else:
            obs_or_base = "Baseline"

        for cluster in range(k):
            fig, ax = plt.subplots(
                ncols=2,
                nrows=2,
                subplot_kw={"projection": ccrs.PlateCarree()},
                figsize=(12, fig_height),
            )
            plt.subplots_adjust(wspace=0.08, hspace=0.002)
            aa = ax.ravel()

            # Calculating and plotting rfo of baseline/obs
            X, Y = np.meshgrid(cluster_labels_d.lon, cluster_labels_d.lat)
            rfo_d = (
                np.sum(cluster_labels_d == cluster, axis=0)
                / np.sum(cluster_labels_d >= 0, axis=0)
                * 100
            )
            aa[0].set_extent([-180, 180, -90, 90])
            aa[0].coastlines()
            mesh = aa[0].pcolormesh(
                X,
                Y,
                rfo_d,
                transform=ccrs.PlateCarree(),
                rasterized=True,
                cmap="GnBu",
                vmin=0,
                vmax=100,
            )
            total_rfo_num = cluster_labels_d == cluster
            total_rfo_num = np.sum(
                total_rfo_num * np.cos(np.deg2rad(cluster_labels_d.lat))
            )
            total_rfo_denom = cluster_labels_d >= 0
            total_rfo_denom = np.sum(
                total_rfo_denom * np.cos(np.deg2rad(cluster_labels_d.lat))
            )
            total_rfo_d = total_rfo_num / total_rfo_denom * 100
            aa[0].set_title(
                f"{obs_or_base}, RFO = {round(float(total_rfo_d),1)}", pad=4
            )

            # Calculating and plotting rfo of test_case
            X, Y = np.meshgrid(cluster_labels.lon, cluster_labels.lat)
            rfo = (
                np.sum(cluster_labels == cluster, axis=0)
                / np.sum(cluster_labels >= 0, axis=0)
                * 100
            )
            aa[1].set_extent([-180, 180, -90, 90])
            aa[1].coastlines()
            mesh = aa[1].pcolormesh(
                X,
                Y,
                rfo,
                transform=ccrs.PlateCarree(),
                rasterized=True,
                cmap="GnBu",
                vmin=0,
                vmax=100,
            )
            total_rfo_num = cluster_labels == cluster
            total_rfo_num = np.sum(
                total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat))
            )
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(
                total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat))
            )
            total_rfo = total_rfo_num / total_rfo_denom * 100
            aa[1].set_title(f"Test Case, RFO = {round(float(total_rfo),1)}", pad=4)

            # Making colorbar
            cax = fig.add_axes(
                [
                    aa[1].get_position().x1 + 0.01,
                    aa[1].get_position().y0,
                    0.02,
                    aa[1].get_position().height,
                ]
            )
            cb = plt.colorbar(mesh, cax=cax)
            cb.set_label(label="RFO (%)")

            # Calculating and plotting difference
            # If observation/baseline is a higher resolution interpolate from obs/baseline to CAM grid
            if len(cluster_labels_d.lat) * len(cluster_labels_d.lon) > len(
                cluster_labels.lat
            ) * len(cluster_labels.lon):
                rfo_d = rfo_d.interp_like(rfo, method="nearest")

            # If CAM is a higher resolution interpolate from CAM to obs/baseline grid
            if len(cluster_labels_d.lat) * len(cluster_labels_d.lon) <= len(
                cluster_labels.lat
            ) * len(cluster_labels.lon):
                rfo = rfo.interp_like(rfo_d, method="nearest")
                X, Y = np.meshgrid(cluster_labels_d.lon, cluster_labels_d.lat)

            rfo_diff = rfo - rfo_d

            aa[2].set_extent([-180, 180, -90, 90])
            aa[2].coastlines()
            mesh = aa[2].pcolormesh(
                X,
                Y,
                rfo_diff,
                transform=ccrs.PlateCarree(),
                rasterized=True,
                cmap="coolwarm",
                vmin=-100,
                vmax=100,
            )
            total_rfo_num = cluster_labels == cluster
            total_rfo_num = np.sum(
                total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat))
            )
            total_rfo_denom = cluster_labels >= 0
            total_rfo_denom = np.sum(
                total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat))
            )
            total_rfo = total_rfo_num / total_rfo_denom * 100
            aa[2].set_title(
                f"Test - {obs_or_base}, ΔRFO = {round(float(total_rfo-total_rfo_d),1)}",
                pad=4,
            )

            # Setting yticks
            aa[0].set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
            aa[2].set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            aa[0].yaxis.set_major_formatter(lat_formatter)
            aa[2].yaxis.set_major_formatter(lat_formatter)

            # making colorbar for diff plot
            cax = fig.add_axes(
                [
                    aa[2].get_position().x1 + 0.01,
                    aa[2].get_position().y0,
                    0.02,
                    aa[2].get_position().height,
                ]
            )
            cb = plt.colorbar(mesh, cax=cax)
            cb.set_label(label="ΔRFO (%)")

            # plotting x labels
            aa[1].set_xticks(
                [
                    -120,
                    -60,
                    0,
                    60,
                    120,
                ],
                crs=ccrs.PlateCarree(),
            )
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            aa[1].xaxis.set_major_formatter(lon_formatter)
            aa[2].set_xticks(
                [
                    -120,
                    -60,
                    0,
                    60,
                    120,
                ],
                crs=ccrs.PlateCarree(),
            )
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            aa[2].xaxis.set_major_formatter(lon_formatter)

            bbox = aa[1].get_position()
            p1 = bbox.p1
            plt.suptitle(
                f"CR{cluster+1} Relative Frequency of Occurence",
                y=p1[1] + (1 / fig_height * 0.5),
            )

            aa[-1].remove()

            save_path = adf.plot_location[0] + f"/{field}_CR{cluster+1}_LatLon_mean"
            plt.savefig(save_path)

            if adf.create_html:
                adf.add_website_data(
                    save_path + ".png", field, case_name=None, multi_case=True
                )

            # Closing the figure
            plt.close()

    ###################################################################
    # MAIN
    ###################################################################

    # Checking if kwargs have been entered correctly
    if wasserstein_or_euclidean not in ["euclidean", "wasserstein"]:
        print(
            '  WARNING: Invalid option for wasserstein_or_euclidean. Please enter "wasserstein" or "euclidean". Proceeding with default of euclidean distance'
        )
        wasserstein_or_euclidean = "euclidean"
    if premade_cloud_regimes != None:
        if type(premade_cloud_regimes) != str:
            print(
                "  WARNING: Invalid option for premade_cloud_regimes. Please enter a path to a numpy array of Cloud Regime centers of shape (n_clusters, n_dimensions_of_data). Proceeding with default clusters"
            )
            premade_cloud_regimes = None
    if lat_range != None:
        if type(lat_range) != list or len(lat_range) != 2:
            print(
                "  WARNING: Invalid option for lat_range. Please enter two values in square brackets sperated by a comma. Example: [-30,30]. Proceeding with entire latitude range"
            )
            lat_range = None
    if lon_range != None:
        if type(lon_range) != list or len(lon_range) != 2:
            print(
                "  WARNING: Invalid option for lon_range. Please enter two values in square brackets sperated by a comma. Example: [0,90]. Proceeding with entire longitude range"
            )
            lon_range = None
    if only_ocean_or_land not in [False, "L", "O"]:
        print(
            '  WARNING: Invalid option for only_ocean_or_land. Please enter "L" for land only, "O" for ocean only. Set to False or leave blank for both land and water. Proceeding with default of False'
        )
        only_ocean_or_land = False

    # NOTE: probably have to move into case loop
    time_range = [
        str(adf.get_cam_info("start_year")[0]),
        str(adf.get_cam_info("end_year")[0]),
    ]

    # ---  BPM refactor ---
    # determine which variables to try
    cr_vars = []
    landfrac_present = "LANDFRAC" in adf.diag_var_list
    print(f"Did we find LANDFRAC in the variable list: {landfrac_present}")
    for field in adf.diag_var_list:
        if field in ["FISCCP1_COSP", "CLD_MISR", "CLMODIS"]:
            cr_vars.append(field)

    # process each each COSP cloud variable
    for field in cr_vars:
        print(f"WORK ON {field}")
        cluster_spec = premade_cloud_regimes if premade_cloud_regimes is not None else wasserstein_or_euclidean

        ht_var_name = ALL_VARS[field].ht_var
        tau_var_name = ALL_VARS[field].tau_var
        if adf.compare_obs:
            ref_ht_var_name = ALL_VARS[field].obs_ht_var
            ref_tau_var_name  = ALL_VARS[field].obs_tau_var
        else:
            ref_ht_var_name = ALL_VARS[field].ht_var
            ref_tau_var_name = ALL_VARS[field].tau_var

        # GET REFERENCE DATA, use for all cases
        ref_data = load_reference_data(adf, field)
        if adf.compare_obs:
            # ref_data should be a dataset in this case
            # reference regime labels or cloud data (to be labeled)
            if premade_cloud_regimes is None:
                if wasserstein_or_euclidean == "wasserstein":
                    ds_o = ref_data.emd_cluster_labels
                else:
                    ds_o = ref_data.euclidean_cluster_labels
            else:
                ds_o = ref_data[adf.variable_defaults[field]['obs_var_name']]
        else:
            ds_o = ref_data # already a dataarray

        for case_name in adf.data.case_names:
            c_ts_da = adf.data.load_timeseries_da(case_name, field)
            if c_ts_da is None:
                print(
                    f"\t WARNING: Variable {field} for case '{case_name}' provides None type. Skipping this variable"
                )
                skip_var = True
                continue
            else:
                print(
                    f"\t Loaded time series for {field} ==> {c_ts_da.shape = }, {c_ts_da.coords = }"
                )
            if "ncol" in c_ts_da.dims:
                # right now we are remapping to fv09 grid because that
                # is the mapping available. 
                # TODO: generalize; would save time to remap to sat data grid
                print("Trigger regrid (ne30-to-fv09 ONLY)")
                regrid_weights_file = Path(
                    "/glade/work/brianpm/mapping_ne30pg3_to_fv09_esmfbilin.nc"
                )
                rg = make_se_regridder(
                    regrid_weights_file, Method="bilinear"
                )  # algorithm needs to match
                ds = regrid_se_data_bilinear(
                    rg, c_ts_da, column_dim_name="ncol"
                )
            else:
                ds = c_ts_da # assumption: already on lat-lon grid

            ##### DATA PRE-PROCESSING
            # Adjusting lon to run from -180 to 180 if it doesnt already
            if np.max(ds.lon) > 180:
                ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
                ds = ds.sortby(ds.lon)

            # Selecting only points over ocean or points over land if only_ocean_or_land has been used
            ds = apply_land_ocean_mask(ds, only_ocean_or_land, landfrac_present)
            if ds is None:
                return  # Error occurred
            # Turning dataset into a dataarray
            if isinstance(ds, xr.Dataset):
                ds = ds[field]
            ds = spatial_subset(ds, lat_range, lon_range)
            ds = temporal_subset(ds, time_range)
            ds = select_valid_tau_height(ds, tau_var_name, ht_var_name)
            #####

            # CLUSTER CENTERS
            cl = load_cluster_centers(adf, cluster_spec, field)
            if cl is None:
                print(f"Skipping cloud regime analysis for {field} due to failed cluster center loading.")
                continue  # Skip to the next variable in cr_vars
    

            # COSP ISCCP data has one extra tau bin than the satellite data, and misr has an extra height bin. 
            # This checks roughly if we are comparing against the
            # satellite data, and if so removes the extra tau or ht bin. 
            # If a user passes home made CRs from CESM data, no data will be removed
            if ALL_VARS[field].product_name == "ISCCP" and cl.shape[1] == 42:
                sel_dict = {tau_var_name: slice(np.min(ds[tau_var_name]) + 1e-11, None)}
                ds = ds.sel(sel_dict)
                print(f"\t Dropping smallest tau bin ({tau_var_name}) to be comparable with observational cloud regimes")
            if ALL_VARS[field].product_name == "MISR" and cl.shape[1] == 105:
                sel_dict = {ht_var_name: slice(np.min(ds[ht_var_name]) + 1e-11, None)}
                ds = ds.sel(sel_dict)
                print(f"\t Dropping lowest height bin ({ht_var_name}) to be comparable with observational cloud regimes")
            
            # CASE CLUSTER LABELING:
            cluster_labels = compute_cluster_labels(ds, tau_var_name, ht_var_name, cl, wasserstein_or_euclidean)
            print(f"{case_name} {field} cluster labels calculated.")

            ref_opts =  {"premade_cloud_regimes":premade_cloud_regimes,
                        "distance": wasserstein_or_euclidean,
                        "landsea": only_ocean_or_land,
                        "landfrac": landfrac_present, # need to deal with this better 
                        "lat_range": lat_range,
                        "lon_range": lon_range,
                        "time_range": time_range,
                        "tau_name": ref_tau_var_name,
                        "ht_name": ref_ht_var_name,
                        "data": ALL_VARS[field].product_name
                        }
            cluster_labels_ref = compute_ref_cluster_labels(adf, ds_o, field, ref_opts)

            # PLOTS
            taucoord = ds[tau_var_name]
            htcoord = ds[ht_var_name]
            # let cluster_labels know the number of clusters:
            cluster_labels.attrs['k'] = cl.shape[0]
            # `plot_rfo_obs_base_diff` expects `cluster_labels_ref` to be latxlon
            if adf.compare_obs:
                plot_hists_obs(
                        field, cl, cluster_labels, cluster_labels_ref, ds, ds_o, ht_var_name, tau_var_name, htcoord, taucoord, adf
                    )
                plot_rfo_obs_base_diff(cluster_labels, cluster_labels_ref, adf, field=field)
            else:
                plot_hists_baseline(
                    field,
                    cl,
                    cluster_labels,
                    cluster_labels_ref,
                    ds,
                    ds_o, # only is ref histograms for simulation, right
                    ht_var_name,
                    tau_var_name,
                    htcoord,
                    taucoord,
                    adf,
                )
            plot_rfo_obs_base_diff(cluster_labels, cluster_labels_ref, adf, field=field)
    # ^^^  BPM refactor ^^^


def compute_ref_cluster_labels(adf, ds_ref, field, opts):
    if adf.compare_obs == True:
        ds_o = ds_ref
        obs_var = adf.variable_defaults[field]['obs_var_name']
        # Adjusting lon to run from -180 to 180 if it doesnt already
        if np.max(ds_o.lon) > 180:
            ds_o.coords["lon"] = (ds_o.coords["lon"] + 180) % 360 - 180
            ds_o = ds_o.sortby(ds_o.lon)

        # this landfrac_present is probably not for ref dataset.
        ds_o = apply_land_ocean_mask(ds_o, opts['landsea'], opts['landfrac'])
        if ds_o is None:
            print("[CRA compute_ref_cluster_labels] reference data is None.")
            return  # Error occurred
        ds_o = spatial_subset(ds_o, opts['lat_range'], opts['lon_range']) # bpm
        if ds_o is None:
            print("[CRA compute_ref_cluster_labels] reference data is None.")
            return  # Error occurred

        if opts['premade_cloud_regimes'] is None:
            print(f"[CRA compute_ref_cluster_labels] {opts['premade_cloud_regimes'] = }")
            cluster_labels_o = ds_o
            cluster_labels_ref = cluster_labels_o.stack(
                spacetime=("time", "lat", "lon")
            ).unstack() ## <- do we want to unstack here?
        else:
            print(f"[CRA compute_ref_cluster_labels] {opts['premade_cloud_regimes'] = }")
            ds_o = select_valid_tau_height(ds_o, opts['tau_name'], opts['ht_name'])
            cluster_labels_ref = finish_cluster_labels(ds_o, opts['tau_name'], opts['ht_name'])
    else:
        # Compare to simulation case.
        ds_b = ds_ref
        time_range_b = [
            str(adf.get_baseline_info("start_year")),
            str(adf.get_baseline_info("end_year")),
        ]
        landfrac_present = opts['landfrac']
        # Adjusting lon to run from -180 to 180 if it doesnt already
        if np.max(ds_b.lon) > 180:
            ds_b.coords["lon"] = (ds_b.coords["lon"] + 180) % 360 - 180
            ds_b = ds_b.sortby(ds_b.lon)

        # this landfrac_present is porbably not for ds_b
        ds_b = apply_land_ocean_mask(ds_b, opts['landsea'], opts['landfrac'])
        if ds_b is None:
            return  # Error occurred
        ds_b = spatial_subset(ds_b, opts['lat_range'], opts['lon_range'])
        ds_b = temporal_subset(ds_b, time_range)

        # Turning dataset into a dataarray
        if isinstance(ds_b, xr.Dataset):
            ds_b = ds_b[field]

        ds_b = select_valid_tau_height(ds_b,  opts['tau_name'], opts['ht_name'])

        # COSP ISCCP data has one extra tau bin than the satellite data, and misr has an extra height bin. This checks roughly if we are comparing against the
        # satellite data, and if so removes the extra tau or ht bin. If a user passes home made CRs from CESM data, no data will be removed
        if data == "ISCCP" and cl.shape[1] == 42:
            # A slightly hacky way to drop the smallest tau bin, but is robust incase tau is flipped in a future version
            ds_b = ds_b.sel(
                cosp_tau=slice(np.min(ds_b.cosp_tau) + 1e-11, np.inf)
            )
            print(
                "\t Dropping smallest tau bin to be comparable with observational cloud regimes"
            )
        if data == "MISR" and cl.shape[1] == 105:
            # A slightly hacky way to drop the lowest height bin, but is robust incase height is flipped in a future version
            ds_b = ds_b.sel(
                cosp_htmisr=slice(np.min(ds_b.cosp_htmisr) + 1e-11, np.inf)
            )
            print(
                "\t Dropping lowest height bin to be comparable with observational cloud regimes"
            )

        cluster_labels_ref = finish_cluster_labels(ds_b, opts['tau_name'], opts['ht_name']) #bpm new func
    return cluster_labels_ref


def precomputed_clusters(mat, cl, wasserstein_or_euclidean, ds):
    """Compute cluster labels from precomputed cluster centers with appropriate distance"""
    if wasserstein_or_euclidean == "euclidean":
        cluster_dists = np.sum((mat[:, :, None] - cl.T[None, :, :]) ** 2, axis=1)
        cluster_labels_temp = np.argmin(cluster_dists, axis=1)
    elif wasserstein_or_euclidean == "wasserstein":
        # A function to convert mat into the form required for the EMD calculation
        @njit()
        def stacking(position_matrix, centroids):
            centroid_list = []

            for i in range(len(centroids)):
                x = np.empty((3, len(mat[0]))).T
                x[:, 0] = centroids[i]
                x[:, 1] = position_matrix[0]
                x[:, 2] = position_matrix[1]
                centroid_list.append(x)

            return centroid_list

        # setting shape
        n1 = len(ds[tau_var_name])
        n2 = len(ds[ht_var_name])

        # Calculating the max distance between two points to be used as hyperparameter in EMD
        # This is not necesarily the only value for this variable that can be used, see Wasserstein documentation
        # on R hyper-parameter for more information
        R = (n1**2 + n2**2) ** 0.5

        # Creating a flattened position matrix to pass wasersstein.PairwiseEMD
        position_matrix = np.zeros((2, n1, n2))
        position_matrix[0] = np.tile(np.arange(n2), (n1, 1))
        position_matrix[1] = np.tile(np.arange(n1), (n2, 1)).T
        position_matrix = position_matrix.reshape(2, -1)

        # Initialising wasserstein.PairwiseEMD
        emds = wasserstein.PairwiseEMD(
            R=R, norm=True, dtype=np.float32, verbose=1, num_threads=162
        )

        # Rearranging mat to be in the format necesary for wasserstein.PairwiseEMD
        events = stacking(position_matrix, mat)
        centroid_list = stacking(position_matrix, cl)
        emds(events, centroid_list)
        print("\t Calculating Wasserstein distances")
        print(
            "\t Warning: This can be slow, but scales very well with additional processors"
        )
        distances = emds.emds()
        labels = np.argmin(distances, axis=1)

        cluster_labels_temp = np.argmin(distances, axis=1)
    else:
        print("[CRA: precomuted_clusters] ERROR -- must specify Wasserstein or Euclidean.")
        return
    return cluster_labels_temp


def load_reference_data(adfobj, varname):
    """Load and reference data.

    Make usual ADF assumption that reference case could be simulation or observation.

    If compare_obs, returns a xr.Dataset,
    otherwise returns time series xr.DataArray.

    """
    base_name = adfobj.data.ref_case_label
    ref_var_nam = adfobj.data.ref_var_nam[varname] # shuld work for obs/sim
    print(f"[CRA: load_reference_data] {base_name = }, {ref_var_nam = }")

    if adfobj.compare_obs:
        ocase = adfobj.data.ref_case_label
        fils = adfobj.data.ref_var_loc.get(varname, None)
        if not isinstance(fils, list):
            fils = [fils]
        ds = adfobj.data.load_dataset(fils)
        if ds is None:
            warnings.warn(f"\t    WARNING: Load failed reference data for {varname}")
            return None
        print(f"[CRA: load_reference_data] return observation dataset")
        return ds
    else:
        print(f"[CRA: load_reference_data] returning simulation dataarray")
        return adfobj.data.load_reference_timeseries_da(varname) 


def load_cluster_centers(adf, cluster_spec: str | Path, variablename: str) -> np.ndarray | None:
    """
    Loads cluster center data from a specified source.

    Args:
        cluster_spec: A string ('wasserstein', 'euclidean', or a file path)
                      or a Path object pointing to a .npy or .nc file.
        variablename: The name of the variable to look up in ALL_VARS to
                      determine the data product name.

    Returns:
        A NumPy array containing the cluster center data, or None if an error occurs.
    """
    if isinstance(cluster_spec, str):
        if cluster_spec in ('wasserstein', 'euclidean'):
            try:
                # Use variablename to find the data product name
                data = ALL_VARS[variablename].product_name
                obs_data_loc = Path(adf.get_basic_info("obs_data_loc"))
                data_key = f"{data}_{cluster_spec}_centers"
                cluster_centers_path = adf.variable_defaults[data_key]["obs_file"]
                file_path = obs_data_loc / cluster_centers_path
            except KeyError as e:
                print(
                    f"[ERROR] Could not find '{variablename}' in ALL_VARS or default file path for '{cluster_spec}'. "
                    f"Original error: {e}"
                )
                return None
        else:
            # Assume it's a direct file path
            file_path = Path(cluster_spec)

    elif isinstance(cluster_spec, Path):
        file_path = cluster_spec
    else:
        print(f"[ERROR] cluster_spec must be a string or a Path object, but got {type(cluster_spec)}")
        return None

    # Check that the path exists before trying to load
    if not file_path.exists():
        print(f"[ERROR] File not found at: {file_path}")
        return None

    # Load the data based on the file extension
    try:
        if file_path.suffix == ".nc":
            with xr.open_dataset(file_path) as ds:
                if 'centers' not in ds:
                    print(f"[ERROR] NetCDF file {file_path.name} does not contain a 'centers' variable.")
                    return None
                cl = ds['centers'].values
        elif file_path.suffix == ".npy":
            cl = np.load(file_path)
        else:
            print(f"[ERROR] Unsupported file type: {file_path.suffix}")
            return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading {file_path.name}: {e}")
        return None

    return cl


def compute_cluster_labels(ds, tau_var_name, ht_var_name, cl, wasserstein_or_euclidean):
    # Selcting only the relevant data and stacking it to shape n_histograms, n_tau * n_pc
    dims = list(ds.dims)
    dims.remove(tau_var_name)
    dims.remove(ht_var_name)
    histograms = ds.stack(spacetime=(dims), tau_ht=(tau_var_name, ht_var_name))
    weights = np.cos(
        np.deg2rad(histograms.lat.values)
    )  # weights array to use with emd-kmeans

    # Turning into a numpy array for clustering
    mat = histograms.values

    # Removing all histograms with 1 or more nans in them
    indices = np.arange(len(mat))
    is_valid = ~np.isnan(mat.mean(axis=1))
    is_valid = is_valid.astype(np.int32)
    valid_inds = indices[is_valid == 1]
    mat = mat[valid_inds]
    weights = weights[valid_inds]

    print(f"\t Fitting data")

    # Compute cluster labels
    cluster_labels_temp = precomputed_clusters(
        mat, cl, wasserstein_or_euclidean, ds
    )

    # taking the flattened cluster_labels_temp array, 
    # and turning it into a datarray the shape of ds.var_name, 
    # and reinserting NaNs in place of missing data
    cluster_labels = np.full(len(indices), np.nan, dtype=np.int32)
    cluster_labels[valid_inds] = cluster_labels_temp
    cluster_labels = xr.DataArray(
        data=cluster_labels,
        coords={"spacetime": histograms.spacetime},
        dims=("spacetime"),
    )
    cluster_labels = cluster_labels.unstack()
    return cluster_labels

def spatial_subset(ds_o, lat_range, lon_range):
    # Selecting lat range
    if lat_range:
        if ds_o.lat[0] > ds_o.lat[-1]:
            ds_o = ds_o.sel(lat=slice(lat_range[1], lat_range[0]))
        else:
            ds_o = ds_o.sel(lat=slice(lat_range[0], lat_range[1]))

    # Selecting Lon range
    if lon_range:
        if ds_o.lon[0] > ds_o.lon[-1]:
            ds_o = ds_o.sel(lon=slice(lon_range[1], lon_range[0]))
        else:
            ds_o = ds_o.sel(lon=slice(lon_range[0], lon_range[1]))
    return ds_o


def temporal_subset(ds, time_range):
    """
    Subset dataset by time range, handling various None/empty cases.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset with time dimension
    time_range : list, tuple, or None
        Time range as [start, end]. Can contain None, "None", or be None/empty
        
    Returns:
    --------
    ds : xarray.Dataset
        Time-subsetted dataset, or original if no valid time range
    """
    def is_valid_time(value):
        """Check if a time value is valid (not None, "None", or empty string)"""
        return value is not None and value != "None" and value != ""
    
    # Handle None, empty, or too short time_range
    if not time_range or len(time_range) < 2:
        return ds
    
    start, end = time_range[0], time_range[1]
    
    # Check if we have any valid time values
    start_valid = is_valid_time(start)
    end_valid = is_valid_time(end)
    
    if not start_valid and not end_valid:
        return ds  # No valid time range, return original
    
    # Set defaults for invalid values
    if not start_valid:
        start = ds.time[0]
    if not end_valid:
        end = ds.time[-1]
    
    return ds.sel(time=slice(start, end))

def select_valid_tau_height(ds, tau_var_name, ht_var_name, max_value=9999999999999):
    """
    Select only valid tau and height/pressure range from dataset.
    
    Excludes failed retrievals (typically -1 values) by selecting from 0 to max_value.
    Handles both pressure (decreasing) and altitude (increasing) coordinate ordering.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing tau and height/pressure variables
    tau_var_name : str
        Name of the tau variable
    ht_var_name : str  
        Name of the height/pressure variable
    max_value : int, optional
        Maximum value for selection range (default: 9999999999999)
        
    Returns:
    --------
    ds : xarray.Dataset
        Dataset with valid tau and height range selected
    """
    # Select valid tau range (exclude negative/failed retrievals)
    tau_selection = {tau_var_name: slice(0, max_value)}
    
    # Handle height/pressure coordinate ordering
    # Pressure: decreasing (high to low) -> slice(max, 0)  
    # Altitude: increasing (low to high) -> slice(0, max)
    if ds[ht_var_name][0] > ds[ht_var_name][-1]:
        # Decreasing coordinate (pressure)
        ht_selection = {ht_var_name: slice(max_value, 0)}
    else:
        # Increasing coordinate (altitude)
        ht_selection = {ht_var_name: slice(0, max_value)}
    
    # Apply selections
    return ds.sel(tau_selection).sel(ht_selection)


def finish_cluster_labels(ds_b, tau_var_name, ht_var_name):
    """
    Compute cluster labels for cloud regime analysis.
    
    Parameters:
    -----------
    ds_b : xarray.Dataset
        Input dataset containing histogram data
    tau_var_name : str
        Name of tau variable
    ht_var_name : str
        Name of height variable
        
    Returns:
    --------
    cluster_labels_b : xarray.DataArray
        Cluster labels with same coordinates as input, NaN for invalid data
    """
    # Selcting only the relevant data and 
    # stacking it to shape n_histograms, n_tau * n_pc
    other_dims = [dim for dim in ds_b.dims if dim not in (tau_var_name, ht_var_name)]
    histograms_b = ds_b.stack(
        spacetime=other_dims, 
        tau_ht=(tau_var_name, ht_var_name)
    )
    # convert to numpy array & compute weights
    # TODO: weights abstraction
    weights_b = np.cos(np.deg2rad(histograms_b.lat.values))
    mat_b = histograms_b.values

    # Find valid histograms (no NaNs) using boolean indexing
    is_valid = ~np.isnan(mat_b).any(axis=1)
    if not is_valid.any():
        print("[cloud_regime_analysis_error] No valid histograms found")
        return None
    # Check for negative values in valid data only
    if (mat_b[is_valid] < 0).any():
        print(f"[cloud_regime_analysis_error] Found negative values in data. "
              f"If these are fill values, convert to NaNs and try again")
        return None
    # Compute clusters only for valid data
    valid_mat = mat_b[is_valid]
    valid_weights = weights_b[is_valid]
    cluster_labels_valid = precomputed_clusters(
        valid_mat, cl, wasserstein_or_euclidean, ds_b
    )
    
    # Create output array with NaNs, then fill valid positions
    cluster_labels_flat = np.full(len(mat_b), np.nan, dtype=np.float32)
    cluster_labels_flat[is_valid] = cluster_labels_valid
    
    # Convert back to DataArray and unstack
    cluster_labels_b = xr.DataArray(
        data=cluster_labels_flat,
        coords={"spacetime": histograms_b.spacetime},
        dims=("spacetime"),
        name="cluster_labels"
    )
    return cluster_labels_b.unstack()


################
# REGRIDDING
################

def make_se_regridder(weight_file, Method="conservative"):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )
    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        # results seem insensitive to this method choice
        # choices are coservative_normed, coservative, and bilinear
        method=Method,
        reuse_weights=True,
        periodic=True,
    )
    return regridder


def regrid_se_data_bilinear(regridder, data_to_regrid, column_dim_name="ncol"):
    if isinstance(data_to_regrid, xr.Dataset):
        vars_with_ncol = [
            name
            for name in data_to_regrid.variables
            if column_dim_name in data_to_regrid[name].dims
        ]
        updated = data_to_regrid.copy().update(
            data_to_regrid[vars_with_ncol]
            .transpose(..., "ncol")
            .expand_dims("dummy", axis=-2)
        )
    elif isinstance(data_to_regrid, xr.DataArray):
        updated = data_to_regrid.transpose(..., column_dim_name).expand_dims(
            "dummy", axis=-2
        )
    else:
        raise ValueError(
            f"Something is wrong because the data to regrid isn't xarray: {type(data_to_regrid)}"
        )
    regridded = regridder(updated)
    return regridded

#
# LAND MASK CODE (probably need to simplify and move out of here)
#
def apply_land_ocean_mask(ds, only_ocean_or_land, landfrac_present=None):
    """
    Apply land or ocean mask to dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset with lat/lon coordinates
    only_ocean_or_land : str or False
        "L" for land only, "O" for ocean only, False for no masking
    landfrac_present : bool, optional
        Whether LANDFRAC variable is available. Auto-detected if None.
        
    Returns:
    --------
    ds : xarray.Dataset
        Masked dataset, or None if invalid option
    """
    # No masking requested
    if only_ocean_or_land is False:
        return ds
    
    # Validate input
    if only_ocean_or_land not in ["L", "O"]:
        print('[cloud_regime_analysis ERROR] Invalid option for only_ocean_or_land: '
              'Please enter "O" for ocean only, "L" for land only, or set to False for both')
        return None
    
    # Auto-detect LANDFRAC if not specified
    if landfrac_present is None:
        landfrac_present = "LANDFRAC" in ds.data_vars or "LANDFRAC" in ds.coords
    
    # Use LANDFRAC if available
    if landfrac_present:
        land_mask_value = 1 if only_ocean_or_land == "L" else 0
        return ds.where(ds.LANDFRAC == land_mask_value)
    
    # Otherwise use cartopy-based land mask
    land_mask = create_land_mask(ds)
    
    # Make land mask broadcastable with dataset
    land_mask = _make_mask_broadcastable(land_mask, ds)
    
    # Apply mask
    mask_value = 1 if only_ocean_or_land == "L" else 0
    return ds.where(land_mask == mask_value)


def _make_mask_broadcastable(mask, ds):
    """
    Make 2D land mask broadcastable with dataset by adding dimensions.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        2D mask array (lat, lon)
    ds : xarray.Dataset
        Target dataset
        
    Returns:
    --------
    mask : numpy.ndarray
        Broadcastable mask array
    """
    # Add dimensions for any dims that aren't lat/lon
    for i, dim in enumerate(ds.dims):
        if dim not in ("lat", "lon"):
            mask = np.expand_dims(mask, axis=i)
    return mask


def create_land_mask(ds):
    """
    Create land mask using cartopy Natural Earth data.
    Improved version with better performance and cleaner code.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with lat/lon coordinates
        
    Returns:
    --------
    land_mask : numpy.ndarray
        2D array (lat, lon) with 1 for land, 0 for ocean
    """
    from cartopy import feature as cfeature
    from shapely.geometry import Point
    from shapely.prepared import prep
    import numpy as np
    from numba import njit
    
    # Get land polygons
    land_110m = cfeature.NaturalEarthFeature("physical", "land", "110m")
    land_polygons = [prep(geom) for geom in land_110m.geometries()]
    # Create coordinate arrays
    lats, lons = ds.lat.values, ds.lon.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Flatten coordinates for easier processing
    lon_flat, lat_flat = lon_grid.flatten(), lat_grid.flatten()
    points = [Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)]
    # Find land points
    land_coords = []
    for polygon in land_polygons:
        land_coords.extend([
            (point.x, point.y) for point in points if polygon.covers(point)
        ])
    # Convert to numpy array for numba processing
    land_array = np.array(land_coords)
    coord_array = np.column_stack([lon_flat, lat_flat])
    # Use numba for fast coordinate matching
    land_mask_flat = _find_land_points(coord_array, land_array)
    # Reshape to original grid
    return land_mask_flat.reshape(len(lats), len(lons))


@njit()
def _find_land_points(coord_array, land_coords):
    """
    Numba-compiled function to quickly identify land points.
    
    Parameters:
    -----------
    coord_array : numpy.ndarray
        Array of (lon, lat) coordinates
    land_coords : numpy.ndarray
        Array of known land coordinates
        
    Returns:
    --------
    mask : numpy.ndarray
        1D mask array with 1 for land, 0 for ocean
    """
    mask = np.zeros(len(coord_array), dtype=np.int32)

    for i in range(len(coord_array)):
        coord = coord_array[i]
        for j in range(len(land_coords)):
            if np.allclose(coord, land_coords[j], atol=1e-10):
                mask[i] = 1
                break
    return mask

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from math import ceil
import xarray as xr

# =============================================================================
# CORE PLOTTING FUNCTIONS
# =============================================================================

def plot_hists_baseline(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord, adf):
    """Plot cloud regime centers for observations, baseline, and test case."""
    plot_data = _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord)
    plot_data['columns'] = ['observation', 'baseline', 'test_case']
    plot_data['figsize'] = (17, plot_data['fig_height'])
    plot_data['save_suffix'] = '_CR_centers'
    
    _plot_cloud_regimes(plot_data, adf, baseline_mode=True)


def plot_hists_obs(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord, adf):
    """Plot cloud regime centers for observations and test case."""
    plot_data = _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord)
    plot_data['columns'] = ['observation', 'test_case']
    plot_data['figsize'] = (12, plot_data['fig_height'])
    plot_data['save_suffix'] = '_CR_centers'
    
    _plot_cloud_regimes(plot_data, adf, baseline_mode=False)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_plot_data(fld, cl, cluster_labels, cluster_labels_o, histograms, histograms_ref, ht_var_name, tau_var_name, htcoord, taucoord):
    """Prepare all data needed for plotting.
    
    fld: str
        name of variable

    cl
        cluster centers
    
    cluster_labels: xr.DataArray
        labels for case ([time], lat, lon)
    cluster_labels_o: array-like
        labels for reference data
    histograms: xr.DataAray (?)
    
    """

    print(f"[_prepare_plot_data] {histograms.coords = }")

    data = ALL_VARS[fld].product_name
    k = len(cl)
    ylabels = htcoord.values
    xlabels = taucoord.values
    
    # Create meshgrid
    X2, Y2 = np.meshgrid(np.arange(len(xlabels) + 1), np.arange(len(ylabels) + 1))
    
    # Calculate figure height
    fig_height = (1 + 10 / 3 * ceil(k / 3)) * 3
    
    # Create weights for RFO calculations
    weights = np.cos(np.deg2rad(cluster_labels.stack(z=("time", "lat", "lon")).lat.values))
    valid_inds = ~np.isnan(cluster_labels.stack(z=("time", "lat", "lon")))
    weights = weights[valid_inds]
    
    return {
        'field': fld,
        'data_product': data,
        'k': k,
        'cl': cl,
        'cluster_labels': cluster_labels,
        'cluster_labels_o': cluster_labels_o,
        'xlabels': xlabels,
        'ylabels': ylabels,
        'X2': X2,
        'Y2': Y2,
        'fig_height': fig_height,
        'weights': weights,
        'ht_var_name': ht_var_name,
        'tau_var_name': tau_var_name,
        'fld': fld,
        'histograms': histograms,
        'histograms_ref': histograms_ref
    }


def _plot_cloud_regimes(plot_data, adf, baseline_mode=True):
    """Main plotting function that handles both baseline and obs-only modes."""
    # Setup
    plt.rcParams.update({"font.size": 14})
    cmap, norm = _create_colormap()
    
    # Create figure
    ncols = 3 if baseline_mode else 2
    fig, ax = plt.subplots(
        figsize=plot_data['figsize'],
        ncols=ncols,
        nrows=plot_data['k'],
        sharex=True, # was "all"
        sharey=True
    )
    fig.subplots_adjust(right=0.88)  # Leave space for colorbar

    # Handle y-axis inversion
    if plot_data['data_product'] != "MISR":
        ax.ravel()[1].invert_yaxis()
    
    # Plot columns
    if baseline_mode:
        _plot_observation_column(ax[:, 0], plot_data, cmap, norm, include_rfo=True)
        _plot_baseline_column(ax[:, 1], plot_data, cmap, norm)
        _plot_test_case_column(ax[:, 2], plot_data, cmap, norm)
    else:
        _plot_observation_column(ax[:, 0], plot_data, cmap, norm, include_rfo=True)
        _plot_test_case_column(ax[:, 1], plot_data, cmap, norm)
    
    # Configure axes and labels
    _configure_axes(ax, plot_data['data_product'])
    _add_figure_labels(fig, ax, plot_data, baseline_mode)
    _add_colorbar(fig, ax.ravel(), cmap, norm)
    
    # Save
    _save_figure(fig, plot_data, adf)
    plt.close()


def _plot_observation_column(ax_col, plot_data, cmap, norm, include_rfo=False):
    """Plot the observation cluster centers."""
    for i in range(plot_data['k']):
        # Plot cluster center
        im = ax_col[i].pcolormesh(
            plot_data['X2'], plot_data['Y2'],
            plot_data['cl'][i].reshape(len(plot_data['xlabels']), len(plot_data['ylabels'])).T,
            norm=norm, cmap=cmap
        )
        
        # Set title with optional RFO
        if include_rfo:
            rfo = _calculate_rfo(plot_data['cluster_labels_o'], i)
            ax_col[i].set_title(f"Observation CR {i+1}, RFO = {np.round(float(rfo), 1)}%")
        else:
            ax_col[i].set_title(f"Observation CR {i+1}")


def _plot_baseline_column(ax_col, plot_data, cmap, norm):
    """Plot the baseline cluster centers with weighted means and RFO."""
    for i in range(plot_data['k']):
        # Calculate RFO
        rfo = _calculate_rfo(cluster_labels_b, i)  # Global variable
        
        # Calculate weighted mean
        wmean = _calculate_weighted_mean_xr(i, plot_data['cluster_labels_o'], plot_data['histograms_ref'])
        # Plot
        im = ax_col[i].pcolormesh(plot_data['X2'], plot_data['Y2'], wmean, norm=norm, cmap=cmap)
        ax_col[i].set_title(f"Baseline Case CR {i+1}, RFO = {np.round(rfo, 1)}%")


def _plot_test_case_column(ax_col, plot_data, cmap, norm):
    """Plot the test case cluster centers with weighted means and RFO."""
    for i in range(plot_data['k']):
        # Calculate RFO
        rfo = _calculate_rfo(plot_data['cluster_labels'], i)
        
        # Calculate weighted mean
        wmean = _calculate_weighted_mean_xr(i, plot_data['cluster_labels'], plot_data['histograms'])
        
        # Plot
        im = ax_col[i].pcolormesh(plot_data['X2'], plot_data['Y2'], wmean, norm=norm, cmap=cmap)
        ax_col[i].set_title(f"Test Case CR {i+1}, RFO = {np.round(rfo, 1)}%")


def _calculate_rfo(cluster_labels, cluster_i):
    """Calculate area-weighted relative frequency of occurrence for a cluster."""
    total_rfo_num = cluster_labels == cluster_i
    total_rfo_num = np.sum(total_rfo_num * np.cos(np.deg2rad(cluster_labels.lat)))
    
    total_rfo_denom = cluster_labels >= 0
    total_rfo_denom = np.sum(total_rfo_denom * np.cos(np.deg2rad(cluster_labels.lat)))
    
    total_rfo = total_rfo_num / total_rfo_denom * 100
    return total_rfo.values

def _calculate_weighted_mean_xr(cluster_i, cluster_labels, hists):
    weights = np.cos(np.radians(hists['lat']))
    cluster_data = xr.where(cluster_labels==cluster_i, hists, np.nan)
    dims = [dim for dim in hists.dims if dim in ["ncol","lat","lon","time"]]
    return cluster_data.weighted(weights).mean(dim=dims)

def _calculate_weighted_mean(cluster_i, cluster_labels, hists, weights, xlabels, ylabels):
    """Calculate area-weighted mean histogram for a cluster.
    
    PARAMETERS
    ----------
    cluster_i : int
        cluster number (label)
    cluster_labels : array-like
        the ([time,] lat, lon) array of cluster labels for the data
    hists: array-like
        the ([time,] ht, tau, lat, lon) array of histograms
    weights: array-like
        area weights
    xlabels: array
        tau values
    ylabels: array
        height/pressure values

    RETURNS
    -------
    wmean: array
        Area-weigthed mean of the histograms in cluster (in percent)
    """
    print(f"[_calculate_weighted_mean] cluster {cluster_i}, {cluster_labels.shape = }, {hists.shape = }, {weights.shape = }")
    pts_i = np.where(cluster_labels == cluster_i) # identify points in cluster
    n = pts_i.sum() # number of points in cluster
    w = [pts_i] # cos(lat)
    if n > 0:
        weighted_hists = hists[indices_i] * weights[indices_i][:, np.newaxis]
        wmean = np.sum(weighted_hists, axis=0) / np.sum(weights[indices_i])
    else:
        wmean = np.zeros(len(xlabels) * len(ylabels))
    
    # Reshape and convert to percentage if needed
    wmean = wmean.reshape(len(xlabels), len(ylabels)).T
    if np.max(wmean) <= 1:
        wmean *= 100
        
    return wmean


def _create_colormap():
    """Create standardized colormap and normalization."""
    p = [0, 0.2, 1, 2, 3, 4, 6, 8, 10, 15, 99]
    colors = [
        "white",
        (0.19215686274509805, 0.25098039215686274, 0.5607843137254902),
        (0.23529411764705882, 0.3333333333333333, 0.6313725490196078),
        (0.32941176470588235, 0.5098039215686274, 0.6980392156862745),
        (0.39215686274509803, 0.6, 0.43137254901960786),
        (0.44313725490196076, 0.6588235294117647, 0.21568627450980393),
        (0.4980392156862745, 0.6784313725490196, 0.1843137254901961),
        (0.5725490196078431, 0.7137254901960784, 0.16862745098039217),
        (0.7529411764705882, 0.8117647058823529, 0.2),
        (0.9568627450980393, 0.8980392156862745, 0.1607843137254902),
    ]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(p, cmap.N, clip=True)
    return cmap, norm


# =============================================================================
# AXIS CONFIGURATION FUNCTIONS  
# =============================================================================

def _configure_axes(ax, data_product):
    """Configure axis ticks and labels based on data product."""
    config_functions = {
        "MODIS": _configure_modis_axes,
        "MISR": _configure_misr_axes,
        "ISCCP": _configure_isccp_axes
    }
    
    if data_product in config_functions:
        config_functions[data_product](ax[0, 0])


def _configure_modis_axes(ax):
    """Configure axes for MODIS data."""
    ylabels = [0, 180, 310, 440, 560, 680, 800, 1000]
    xlabels = [0, 0.3, 1.3, 3.6, 9.4, 23, 60, 150]
    
    ax.set_yticks(np.arange(8))
    ax.set_xticks(np.arange(8))
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)


def _configure_misr_axes(ax):
    """Configure axes for MISR data."""
    xlabels = [0.2, 0.8, 2.4, 6.5, 16.2, 41.5, 100]
    ylabels = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 6, 8, 10, 12, 14, 16, 20]
    
    ax.set_yticks(np.arange(0, 16, 2) + 0.5)
    ax.set_yticklabels(ylabels[0::2])
    ax.set_xticks(np.array([1, 2, 3, 4, 5, 6, 7]) - 0.5)
    ax.set_xticklabels(xlabels, fontsize=16)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    xticks[-1].set_visible(False)


def _configure_isccp_axes(ax):
    """Configure axes for ISCCP data."""
    xlabels = [0, 1.3, 3.6, 9.4, 22.6, 60.4, 450]
    ylabels = [10, 180, 310, 440, 560, 680, 800, 1025]
    
    # Get current ticks
    yticks = ax.get_yticks().tolist()
    xticks = ax.get_xticks().tolist()
    
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(ylabels)
    ax.set_xticklabels(xlabels)
    
    # Hide first and last x-tick labels
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)


def _add_figure_labels(fig, ax, plot_data, baseline_mode):
    """Add figure title and axis labels."""
    data = plot_data['data_product']
    ht_var_name = plot_data['ht_var_name']
    fig_height = plot_data['fig_height']
    # bpm: hacky attempt to get coordinate units:
    if ("prs" in ht_var_name):
        ht_label = "Pressure"
        ht_unit = "hPa"
    else:
        ht_label = "Height"
        ht_unit = "m"

    # Determine height or pressure
    # height_or_pressure = "h" if data == "MISR" else "p"

    # Y-label
    x_pos = 0.07 if baseline_mode else 0.05
    fig.supylabel(f"Cloud-top {ht_label} ({ht_unit})", x=x_pos)

    # Title positioning
    bbox = ax[1, 0].get_position()  # Use first column for positioning
    fig.suptitle(
        f"{data} Cloud Regimes",
        # x=0.5,
        # y=bbox.p1[1] + (1 / fig_height * 0.5) + 0.007,
        fontsize=18,
    )
    
    # X-label positioning
    bbox = ax[-1, -1].get_position()  # Use last subplot for positioning
    fig.supxlabel("Optical Depth", y=bbox.p0[1] - (1 / fig_height * 0.5) - 0.007)


def _add_colorbar(fig, ax, cmap, norm):
    """Add colorbar to the figure."""
    p = [0, 0.2, 1, 2, 3, 4, 6, 8, 10, 15, 99]
    # cbar_ax = fig.add_axes([1.01, 0.25, 0.04, 0.5])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])  # Required for colorbar
    cb = fig.colorbar(sm, 
                      ax=ax, 
                      orientation='vertical', 
                      fraction=0.025, 
                      pad=0.02,
                      aspect=40,
                      ticks=p)
    cb.set_label(label="Cloud Cover (%)", size=16)
    cb.ax.tick_params(labelsize=14)


def _save_figure(fig, plot_data, adf):
    """Save figure and add to website if requested."""
    data = plot_data['data_product']
    save_path = adf.plot_location[0] + f"/{data}{plot_data['save_suffix']}"
    plt.savefig(save_path)
    
    if adf.create_html:
        if hasattr(adf, 'compare_obs') and adf.compare_obs:
            # For obs comparison mode
            adf.add_website_data(save_path + ".png", plot_data['field'], case_name=None, multi_case=True)
        else:
            # For baseline comparison mode  
            adf.add_website_data(save_path + ".png", plot_data['field'], adf.get_baseline_info("cam_case_name"))