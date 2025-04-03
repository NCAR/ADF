# List of ideas for LDF hackathon:
### Simple tasks work
-[] Expand list of default variables in `config_clm_baseline_example.yml`
-[] Improve plotting aesthetics: expand list of variables in `adf/lib/ldf_variable_defaults.yml` 
-[] Identify list of regions and bounding boxes where we want to make timeseries or climo plots

### Integration
-[x] Integrate `regrid_se_to_fv` regridding script into ADF workflow.
-[x] Integrate `plot_unstructured_map_and_save` function into `/scripts/plotting/global_unstructured_latlon_map`
-[x] Develop coherent way to handled structured vs. unstructured input data (maybe adapt all to uxarray)?

### Development
-[] Seperate time bounds for time series and climo generation.
-[] Write python function to make regional timeseries or climo plots
-[] Check applicaiton of adf timeseries plots for land
-[x] Handle h1 files for PFT specific results
-[] Integrate observations!

#
 
