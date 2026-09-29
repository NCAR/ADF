# ADF diagnostics

[![Framework Unit Tests](https://github.com/NCAR/ADF/actions/workflows/ADF_unit_tests.yaml/badge.svg)](https://github.com/NCAR/ADF/actions/workflows/ADF_unit_tests.yaml) [![pre-commit](https://github.com/NCAR/ADF/actions/workflows/ADF_pre-commit.yaml/badge.svg)](https://github.com/NCAR/ADF/actions/workflows/ADF_pre-commit.yaml) [![CC BY 4.0][cc-by-shield]][cc-by]

This repository contains the Atmosphere Model Working Group (AMWG) Diagnostics Framework (ADF) diagnostics python package, which includes numerous different averaging,
re-gridding, and plotting scripts, most of which are provided by users of CAM itself.

Specifically, this package is currently designed to generate standard climatological comparisons between either two
different CAM simulations, or between a CAM simulation and observational and reanalysis datasets.  Ideally
this will allow for a quick evaluation of a CAM simulation, without requiring the user to generate numerous
different figures on their own.

Currently, this package only uses standard CAM monthly time-slice (h0) outputs or single-variable monthly time series files.  However, if there is user interest then
additional model input options can be added.

Finally, if you are interested in general (but non-supported) tools used by AMP scientists and engineers in their work, then please check out the [AMP Toolbox](https://github.com/NCAR/AMP_toolbox).

## Required software environment

These diagnostics require Python 3.9 or higher (CI runs the test suite on 3.9-3.13). The exact, version-pinned
set of non-standard python libraries/modules (PyYAML, Xarray, Matplotlib, Cartopy, GeoCAT, uxarray, xESMF,
xskillscore, Pint, netCDF4, Scipy, Pandas, ...) is kept in [`env/conda_environment.yaml`](env/conda_environment.yaml)
rather than duplicated here, so it doesn't go stale.

Create and activate the environment from that file with:

```
conda env create -f env/conda_environment.yaml
conda activate adf_v1.0.0
```

### On NCAR HPC (derecho/casper)

Load the NCAR-provided conda module first, then create the environment as above:
```
module load conda
conda env create -f env/conda_environment.yaml
conda activate adf_v1.0.0
```
By default this places the environment under `/glade/work/$USER/conda-envs/`. See the
[NCAR HPC conda documentation](https://ncar-hpc-docs.readthedocs.io/en/v24.08/environment-and-software/user-environment/conda/)
for details (e.g. changing the environment location, using conda in batch jobs).

### If your machine has no conda module (e.g. CGD machines)

Some machines don't provide a `conda` module, so you'll need your own. Install
[Miniforge](https://github.com/conda-forge/miniforge), which defaults to the conda-forge channel that supplies
nearly all of `env/conda_environment.yaml`:
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) is an equivalent alternative. Once
installed, create the environment with the same `conda env create` command above; it lands under your own conda
installation's `envs/` directory. Note that on NCAR HPC systems a personal conda install is shadowed by
`module load conda` whenever that module is loaded.

### Non-python requirements (all machines)

The `ncrcat` NetCDF Operator (NCO) is also needed.  On NCAR HPC (derecho/casper) it can be loaded by simply
running:
```
module load nco
```
on the command line.

On CGD machines the `tool/nco` modules are currently not usable: the unversioned `tool/nco` resolves to a
modulefile for an install that isn't present and adds nothing to `PATH`, and `tool/nco/4.5.2` puts `ncrcat` on
`PATH` but fails at runtime looking for `libexpat.so.0`, which current CGD systems no longer ship. Install NCO
from conda-forge into your ADF environment instead:
```
conda activate adf_v1.0.0
conda install -c conda-forge nco
```

Finally, if you also want to run the [Climate Variability Diagnostics Package](https://www.cesm.ucar.edu/working_groups/CVC/cvdp/) (CVDP) as part of the ADF then you'll also need NCL.  On NCAR HPC (derecho/casper) this can be done using the command:
```
module load ncl
```
or on the CGD machines by using the command:
```
module load tool/ncl/6.6.2
```
on the command line.

## Running ADF diagnostics

Detailed instructions for users and developers are availabe on this repository's [wiki](https://github.com/NCAR/ADF/wiki).


To run an example of the ADF diagnostics, simply download this repo, setup your computing environment as described in the [Required software environment](#required-software-environment) section above, modify the `config_cam_baseline_example.yaml` file (or create one of your own) to point to the relevant directories and run:

`./run_adf_diag config_cam_baseline_example.yaml`

This generates time series files, climatology (climo) files, re-gridded climo files, diagnostic figures, tables, and a website, each in its own directory. See [What the ADF produces](#what-the-adf-produces) below for what to expect.

### What the ADF produces

A run goes through the stages below in order. Each writes to a location set in the `diag_basic_info`
or case block of the config file. Time series can be skipped with `cam_ts_done: true`, the website with
`create_html: false`, and the other stages by removing their scripts from the config.

| Stage | Output | Config key |
| --- | --- | --- |
| Time series | one file per variable | `cam_ts_loc` |
| Climatologies | 12 monthly means per variable | `cam_climo_loc` |
| Regridding | fields on the comparison grid and pressure levels | `cam_regrid_loc` |
| Analysis and plots | figures and tables | `cam_diag_plot_loc` |
| Website | HTML pages that link to the figures and tables | `cam_diag_plot_loc` (`create_html: true`) |

For a model-vs-model run the files look like this (`<VAR>` is a variable name, `<hist_str>` the history
stream, for example `cam.h0a`):

```
<cam_ts_loc>/<case>.<hist_str>.<VAR>.<YYYYMM>-<YYYYMM>.nc
<cam_climo_loc>/<case>_<hist_str>_<VAR>_climo.nc
<cam_regrid_loc>/<baseline>_<test>_<VAR>_regridded.nc       (baseline copy: <baseline>_<VAR>_baseline.nc)
<cam_regrid_loc>/regrid_weights/
<cam_diag_plot_loc>/<test>_<syr>_<eyr>_vs_<baseline>_<syr>_<eyr>/
    <VAR>_<SEASON>_LatLon_Mean.png        maps
    <VAR>_<SEASON>_Zonal_Mean.png         zonal means
    amwg_table_<case>.csv                 global-mean table for each case
    amwg_table_comp.csv                   test vs. baseline table
    website/index.html                    start page of the website
```

A model-vs-observations run is the same, but the directory is named `<case>_<syr>_<eyr>_vs_Obs` and
the regridded observations are used in place of a baseline.

**Figures.** Each plotting script in `plotting_scripts` writes one image per variable and season
(`ANN`, `DJF`, `MAM`, `JJA`, `SON`); the file name ends in the plot type, for example `_LatLon_Mean`,
`_LatLon_Vector_Mean`, `_Zonal_Mean`, `_Meridional_Mean`, or a polar-map name. A map has four panels: the test case,
the baseline or observations, the percent difference, and the difference (with its RMSE). A 3-D variable gets one map per level in
`plot_press_levels` (the file name has the level, for example `T_850hpa_ANN_LatLon_Mean.png`)
and a zonal mean against pressure. Images are PNG unless `plot_type` says otherwise. The Taylor
diagram script writes one figure per season with all the test cases on it.

**Tables.** `amwg_table` writes, for each case, the global mean of every variable with its sample
size, standard deviation, standard error, 95% confidence interval, and linear trend with its
p-value. The comparison table has the columns `variable`, `unit`, `test`, `control` (the baseline or observations), and `diff`.

**Website.** With `create_html: true`, open `website/index.html` (under the plot directory shown
above) in a browser. It links to every figure, grouped by variable category and season, to the
tables, and to a page that records the configuration and software environment of the run. The
directory is self-contained, so it can be copied to a web server. A run with more than one test case
also writes a `main_website/` directory under `cam_diag_plot_loc` that covers all the cases. In that case the
Taylor diagram, the baseline table, and the comparison table are written once, in the first case's plot directory.

The ADF only plots what a run contains. If a variable is missing and cannot be derived from
other variables, the ADF prints a message and moves on to the next one.

### ADF Tutorial/Demo

Jupyter Book detailing the ADF including ADF basics, guided examples, quick runs, and references
  - https://justin-richling.github.io/ADF-Tutorial/README.html

## Developing the ADF

Detailed developer instructions live on the [wiki](https://github.com/NCAR/ADF/wiki); one
repository-level setting is worth doing right after you clone.

### Formatting checks (`pre-commit`)

The framework code under `lib/` is formatted with [black](https://black.readthedocs.io/), and
the [`ADF_pre-commit.yaml`](.github/workflows/ADF_pre-commit.yaml) workflow re-checks it on
every pull request, so an unformatted `lib/` file is a failing CI check. `pre-commit` is part
of `env/conda_environment.yaml`, so with the ADF environment activated you can run the same
check CI runs:
```
pre-commit run -a
```
Better, install it as a git hook so it runs automatically on each commit:
```
pre-commit install
```
The black version is pinned in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) and all of
its settings live in [`pyproject.toml`](pyproject.toml), so local runs and CI always agree.
`scripts/` is intentionally not covered by the hook.

## Troubleshooting

Any problems or issues with this software should be posted on the ADF discussions page located online [here](https://github.com/NCAR/ADF/discussions).

Please note that registration may be required before a message can
be posted.  However, feel free to search the forums for similar issues
(and possible solutions) without needing to register or sign in.

Good luck, and have a great day!

##

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
