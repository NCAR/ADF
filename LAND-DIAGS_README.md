### Here's a quick start that you can use for Land Diagnostics
#### Download the adf repo
On casper:
1. Navigate to the directory where you want the adf (e.g., `cd ~`)
2. Clone the ADF
`git clone https://github.com/NCAR/ADF.git`
3. (Optional) Add your personal fork as a second remote, so you can push your own changes to it
```
cd ADF
git remote add fork https://github.com/<git_user_name>/ADF.git
```
4. Switch to the land/native-grid development branch
`git switch -c ldf-native-grid-merge-main origin/ldf-native-grid-merge-main`

#### Set up your computing environment
5. Create the ADF conda environment, which the land diagnostics use too (it includes uxarray 2026.10.0; plotting on the native grid needs at least 2025.6.0). On NCAR's CISL machines (derecho and casper), run the following on the command line:
```
module load conda
conda env create -f env/conda_environment.yaml
conda activate adf_v1.0.6
```
(The environment's name is set by the `name:` line at the top of `env/conda_environment.yaml`; add `-n <name>` to `conda env create` to choose another.)

Also, along with these python requirements, the `ncrcat` NetCDF Operator (NCO) is also needed.  On the CISL machines this can be loaded by simply running the following on the command line:

```
module load nco
```

## Running ADF diagnostics

Detailed instructions for users and developers are availabe on this repository's [wiki](https://github.com/NCAR/ADF/wiki).

For a quick try of land diagnostics, set `user` near the top of `config_clm_tutorial_example.yaml` to your username, then run

`./run_adf_diag config_clm_tutorial_example.yaml`

This compares two f09 land-only runs from the CLM tutorial data, and should generate a collection of time series files, climatology (climo) files, re-gridded climo files, and example ADF diagnostic figures, all in their respective directories.

Land runs are recognised from `clm2` in the case's `hist_str` (e.g. `hist_str: clm2.h0a`).

**NOTE:** If you get NCO failures at the generate timeseries stage that end up causing LDF to fail, see issue [#365](https://github.com/NCAR/ADF/issues/365)

When additional memory is needed sometimes need to run interactive session on casper:
`execcasper -A <project_code> -l select=1:ncpus=4:mem=64GB`

### Land observations are currently unavailable

The land observation files the variable defaults point to (`/glade/campaign/cesm/community/lmwg/diag/lnd_diag_data/obs_data/`) have been removed from glade, and a new location has not been settled yet. Until it is, the regional climatology plots are made without their observation lines (GPCP, MODIS, CERES, FLUXNET, GRUN), and the run still finishes without an error. Model-vs-model comparisons are not affected.

## Config settings for land runs

These are the settings land users are most likely to need; all of them are documented in `config_cam_baseline_example.yaml`.

- **Time series and climatology years.** `start_year`/`end_year` (also accepted as `ts_start_year`/`ts_end_year`) set the years of the time series. `climo_start_year`/`climo_end_year` optionally set a narrower range for the climatology; without them the climatology uses the time-series years. Climo and regridded files carry their years in their names (e.g. `..._climo_0085-0104.nc`), so changing the years makes new files instead of reusing old ones.
- **`calc_cam_ts` replaces `cam_ts_done`**, with the opposite meaning: `calc_cam_ts: true` makes time series from the history files. The old key still works and prints a deprecation note. Time series already in `cam_ts_loc` are reused per variable.
- **Regional plots.** `regions_file: lib/regions_lnd.yaml` (in `diag_basic_info`) defines the regions; `region_list` and `regional_var_list` choose which regions and variables to plot.
- **Tables.** List `statistics_table` in `analysis_scripts`. It replaces `lmwg_table` (and `amwg_table`); the old names still work and print a note. The output file is still named `lmwg_table_<case>.csv` for land cases.
- **Variable names.** `TOTECOSYSC_1m` is now `TOTECOSYSC` (and `TOTSOMC` is new); both are totals in PgC in the tables. A config listing `TOTECOSYSC_1m` needs the new name.

## TEST for Land Diags:

For this branch there are (3) ways to run the ADF:

1) On native grid with unstructured plotting via Uxarray
2) On native grid but gridded to lat/lon
3) On already lat/lon gridded input files (hist, ts, or climo)

Each test case and the baseline can start from history, time series or climatology files.

For (1), the config yaml file will be essentially the same, but with a couple of additional arguments:
  - in `diag_basic_info` set the `unstructured_plotting` argument to `true`
  - in each native-grid test and baseline section supply the case's ESMF mesh file in the `mesh_file` argument

  ESMF mesh files for the standard CESM grids are on glade in `/glade/campaign/cesm/cesmdata/inputdata/share/meshes/`, e.g.

  `mesh_file: /glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc`

  If one side of the comparison is on a lat/lon grid (a lat/lon baseline, or observations), it is remapped onto the native case's mesh automatically.

  Example yaml file: `config_clm_unstructured_plots.yaml`

For (2), the config yaml file will need some additional arguments:
  - in `diag_basic_info` set (or leave) `unstructured_plotting: false`
  - in each native-grid test and baseline section, supply the following arguments:

    Weights file (required), mapping the native grid onto the lat/lon grid:

    `weights_file: /glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_TO_fv0.9x1.25_aave.240826.nc`

    Weights files for many grid pairs are on glade in `/glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/` (`_aave` files are conservative, `_blin` files bilinear).

    Regridding method (optional):

    `regrid_method: conservative`

    The weights file decides the remapping algorithm; `regrid_method` only sets how xESMF applies the weights, `conservative` (the default) or `bilinear`. (The old misspelling `coservative` is still accepted.)

    Lat/lon land mask file (optional, land only): a file on the target lat/lon grid with a `landmask` variable, used to mask the gridding:

    `latlon_file: /path/to/a/clm2/history/file/on/the/target/grid.nc`

    Conservative weights file (optional): a second weights file used only for the variables marked `regrid_conservative: true` in the variable defaults (fluxes such as precipitation and radiation), so that `weights_file` can be e.g. a bilinear map for everything else:

    `conservative_weights_file: /glade/campaign/cesm/cesmdata/inputdata/cpl/gridmaps/ne30pg3/map_ne30pg3_TO_fv0.9x1.25_aave.240826.nc`

  Example yaml file: `config_clm_native_grid_to_latlon.yaml`

For (3), no extra arguments are needed: leave `unstructured_plotting: false` and do not set `mesh_file` or `weights_file`. When the two cases are on different lat/lon grids, the finer one is regridded onto the coarser one.

  Example yaml file: `config_clm_tutorial_example.yaml`
