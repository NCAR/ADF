# CAM diagnostics

This repository contains the CAM diagnostics python package, which includes numerous different averaging,
re-gridding, and plotting scripts, most of which are provided by users of CAM itself.

Specifically, this package is designed to generate standard climatological comparisons between either two
different CAM simulations, or between a CAM simulation and observational and reanalysis datasets.  Ideally
this will allow for a quick evaluation of a CAM simulation, without requiring the user to generate numerous
different figures on there own.

Currently, this figure only uses standard CAM monthly (h0) outputs.  However, if there is user interest then
additional diagnostic options can be added.

To run an example of the CAM diagnostics on either Cheyenne or Casper, simply download this repo and run:

`./run_diag --namelist example_namelist.yaml`

This should generate a new `plots` directory which contains example CAM diagnostic figures.

Finally, any problems or issues with this script should be posted on the
DiscussCESM CAM forum located online [here](https://xenforo.cgd.ucar.edu/cesm/forums/cam.133/).

Please note that registration may be required before a message can
be posted.  However, feel free to search the forums for similar issues
(and possible solutions) without needing to register or sign in.

Good luck, and have a great day!

