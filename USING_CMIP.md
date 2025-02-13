# ADF diagnostics vs CMIP runs

This branch is meant to serve as a temporary ADF for running diagnostics of model vs CMIP-like datasets.

There are still rigourous tests that need to be completed as well as vetted reviews, but this will allow the user to experiment before we merge into the main branch.

Included in this branch is an example config yaml file `config_model_vs_cmip.yaml`


NOTE: There is one major change to the config yaml file: `cam_ts_done` will be changed to `calc_cam_ts`. The motivation here is to follow suit with the climo arguments, ie `calc_cam_climo`.

So now there are two ways to deal with time series files:

* `calc_cam_ts`: true

This will for creation of time series files from the history files

NOTE: This will require `cam_ts_loc` to be present. `cam_ts_save` and `cam_overwrite_ts` are optional

* `calc_cam_ts`: false

This can have two scenarios

1. if user is supplying premade time series files (ie CMIP or previous ADF generated files) -> `cam_ts_loc` is then required for location of those files
2. time series files are not needed at all -> `cam_ts_loc` <strong>has</strong> to be ignored
  * This is for ADF diags that either don't need time series files (highly unlikely) or if the user is supplying premade climo files
  