# List of ideas for SEWG hackathon:
### Simple / busy work
- Identify list of default variables in `config_clm_baseline_example.yml`
- Adapt list of variables in `adf/lib/ldf_variable_defaults.yml` (plotting controls for list above)
- Identify list of regions and bounding boxes where we want to make timeseries or climo plots

### Integration
- Integrate `regrid_se_to_fv` regridding script into ADF workflow.
- Integrate `plot_unstructured_map_and_save` function into `/scripts/plotting/global_unstructured_latlon_map`
- Develop coherent way to handled structured vs. unstructured input data (maybe adapt all to uxarray)?

### Development
- Seperate time bounds for time series and climo generation.
- Write python function to make regional timeseries or climo plots
- Adapt adf timeseries plots for land
- Handle h1 files for PFT specific results

#
 
