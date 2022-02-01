# ADF diagnostics

[![Framework Unit Tests](https://github.com/NCAR/ADF/actions/workflows/ADF_unit_tests.yaml/badge.svg)](https://github.com/NCAR/ADF/actions/workflows/ADF_unit_tests.yaml)

This repository contains the Atmosphere Diagnostics Framework (ADF) diagnostics python package, which includes numerous different averaging,
re-gridding, and plotting scripts, most of which are provided by users of CAM itself.

Specifically, this package is currently designed to generate standard climatological comparisons between either two
different CAM simulations, or between a CAM simulation and observational and reanalysis datasets.  Ideally
this will allow for a quick evaluation of a CAM simulation, without requiring the user to generate numerous
different figures on their own.

Currently, this figure only uses standard CAM monthly (h0) outputs.  However, if there is user interest then
additional diagnostic options can be added.

## Required software environment

These diagnostics currently require Python version 3.6 or highter.  They also require the following non-standard python libraries/modules:

- PyYAML
- Numpy
- Xarray
- Matplotlib
- Cartopy
- GeoCAT

If one wants to generate the "AMWG" model variable statistics table as well, then these additional python libraries are also needed:

- Scipy
- Pandas

On NCAR's CISL machines (cheyenne and casper), these can be loaded by running the following on the command line
```
module load python/3.7.12
ncar_pylib
```
If you are using conda on a non-CISL machine, then you can create and activate the appropriate python enviroment using the `env/conda_environment.yaml` file like so:

```
conda env create -f env/conda_environment.yaml
conda activate adf_v0.07
```

Finally, along with these python requirements, the `ncrcat` NetCDF Operator (NCO) is also needed.  On the CISL machines, this can be loaded by simply running `module load nco` on the command line.

## Running ADF diagnostics

Detailed instructions for users and developers are availabe on this repository's [wiki](https://github.com/NCAR/ADF/wiki).


To run an example of the ADF diagnostics, simply download this repo, setup your computing environment as described in the [Required software environment](https://github.com/NCAR/CAM_diagnostics/blob/main/README.md#required-software-environment) section above, modify the `config_cam_baseline_example.yaml` file (or create one of your own) to point to the relevant diretories and run:

`./run_adf_diag config_cam_baseline_example.yaml`

This should generate a collection of time series files, climatology (climo) files, re-gridded climo files, and example ADF diagnostic figures, all in their respective directories.

## Troubleshooting

Any problems or issues with this software should be posted on the ADF discussions page located online [here](https://github.com/NCAR/ADF/discussions).

Please note that registration may be required before a message can
be posted.  However, feel free to search the forums for similar issues
(and possible solutions) without needing to register or sign in.

Good luck, and have a great day!

