### Here's a quick start that you can use for Land Diagnostics
#### Download the adf repo
On casper:
1. Navigate to the directory where you want the adf (e.g., `cd ~`)
2. Clone the ADF
`git clone https://github.com/NCAR/ADF.git`
3. Set your personal repository as the upstream repo
```
cd ADF
git remote add upstream https://github.com/<git_user_name>/ADF.git
```
4. Switch to the clm-diags branch
`git switch -c clm-diags origin/clm-diags`

#### Set up your computing environment
5. Create the ADF conda environment, which the land diagnostics use too (it includes the uxarray version, 2025.03.0, they need). On NCAR's CISL machines (derecho and casper), run the following on the command line:
```
module load conda
conda env create -f env/conda_environment.yaml
conda activate adf_v1.0.3
```
(The environment's name is set by the `name:` line at the top of `env/conda_environment.yaml`; add `-n <name>` to `conda env create` to choose another.)

Also, along with these python requirements, the `ncrcat` NetCDF Operator (NCO) is also needed.  On the CISL machines this can be loaded by simply running the following on the command line:

```
module load nco
```

## Running ADF diagnostics

Detailed instructions for users and developers are availabe on this repository's [wiki](https://github.com/NCAR/ADF/wiki). 

You'll have to add your username to the appropriate config file, but after that, for a quick try of land diagnostics

`./run_adf_diag config_clm_unstructured_plots.yaml`

This should generate a collection of time series files, climatology (climo) files, re-gridded climo files, and example ADF diagnostic figures, all in their respective directories.

**NOTE:** If you get NCO failures at the generate timeseries stage that end up causing LDF to fail, see issue [#365](https://github.com/NCAR/ADF/issues/365) 

When additional memory is needed sometimes need to run interactive session on casper:
`execcasper -A P93300041 -l select=1:ncpus=4:mem=64GB`

## TEST for Land Diags:

For this branch there are (3) ways to run the ADF:

1) On native grid with unstructured plotting via Uxarray
2) On native grid but gridded to lat/lon
3) On already lat/lon gridded input files (hist, ts, or climo)

For (1), the config yaml file will be essentially the same, but with a couple of additional arguments:
  - in `diag_basic_info` set the `unstructured_plotting` argument to `true`
  - in each of the test and baseline section supply a mesh file in the `mesh_file` argument

  Example yaml file: `config_clm_unstructured_plots.yaml`

For (2), the config yaml file will need some additional arguments:
  - in each of the test and baseline sections, supply the following arguments:

    Weights file:
    
    `weights_file: /glade/work/wwieder/map_ne30pg3_to_fv0.9x1.25_scripgrids_conserve_nomask_c250108.nc`
    
    Regridding method:
    
    `regrid_method: 'coservative'`
    (Yes, spelled incorectly for a bug in xESMF)

    Lat/lon file:
    
    `latlon_file: /glade/derecho/scratch/wwieder/ctsm5.3.018_SP_f09_t232_mask/run/ctsm5.3.018_SP_f09_t232_mask.clm2.h0.0001-01.nc`

  NOTE: The regridding method set in `regrid_method` MUST match the method in the weights file
  
  Example yaml file: `config_clm_native_grid_to_latlon.yaml`

  
:
