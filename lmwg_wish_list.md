# List of ideas for LDF Codefest:
### Simple tasks work
-[ ] Expand list of default variables in `config_clm_baseline_example.yml`
-[ ] Improve plotting aesthetics: expand list of variables in `adf/lib/ldf_variable_defaults.yml` 
-[ ] Identify list of regions and bounding boxes where we want to make timeseries or climo plots

### Integration
-[x] Integrate `regrid_se_to_fv` regridding script into ADF workflow.
  - check how this is working in revised 
-[x] Integrate `plot_unstructured_map_and_save` function into `/scripts/plotting/global_unstructured_latlon_map`
-[x] Develop coherent way to handled structured vs. unstructured input data (maybe adapt all to uxarray)?

### Development
-[ ] Separate time bounds for time series and climo generation.
-[ ] *Top need* Write python function to make regional timeseries or climo plots
-[ ] Check application of adf timeseries plots for land, 
  - this was working on wwieder/clm-test branch
-[ ] Handle h1 files for PFT specific results
  - Need to convert notebook for unstructured data in to python fuction that uses upstream ADF workflow & adds plots to website
  - also need unstructured example
-[ ] Integrate observations!
  - Populate datasets in central location (e.g. `/glade/campaign/cgd/amp/amwg/ADF_obs`), maybe this lives in a CUPiD observations home?
  - Regrid to standard resolution(s), start with f09_t232
  

#
 
