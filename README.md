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

This should generate a collection of time series files, climatology (climo) files, re-gridded climo files, and example ADF diagnostic figures, all in their respective directories.

### ADF Tutorial/Demo

Jupyter Book detailing the ADF including ADF basics, guided examples, quick runs, and references
  - https://justin-richling.github.io/ADF-Tutorial/README.html

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
