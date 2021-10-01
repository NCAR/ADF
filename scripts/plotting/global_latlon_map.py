def global_latlon_map(case_name, model_rgrid_loc, data_name, data_loc,
                 var_list, data_list, plot_location):

    """
    This script/function is designed to generate global
    2-D lat/lon maps of model fields with continental overlays.

    This script is actually the same as 'plot_example', except
    that it uses the shared diagnostics plotting library, as
    opposed to being entirely self-contained.

    Description of function inputs:

    case_name        -> Name of CAM case provided by "cam_case_name".
    model_rgrid_loc  -> Location of re-gridded CAM climo files provided by "cam_regrid_loc".
    data_name        -> Name of data set CAM case is being compared against,
                        which is always either "obs" or the baseline CAM case name,
                        depending on whether "compare_obs" is true or false.
    data_loc         -> Location of comparison data, which is either "obs_climo_loc"
                        or "cam_baseline_climo_loc", depending on whether
                        "compare_obs" is true or false.
    var_list         -> List of CAM output variables provided by "diag_var_list"
    data_list        -> List of data sets CAM will be compared against, which
                        is simply the baseline case name in situations when
                        "compare_obs" is false.
    plot_location    -> Location where plot files will be written to, which is
                        specified by "cam_diag_plot_loc".
    """

    #Import necessary modules:
    #------------------------
    from pathlib import Path  # python standard library

    # data loading / analysis
    import xarray as xr
    import numpy as np

    #CAM diagnostic plotting functions:
    import plotting_functions as pf
    #-------------------------

    # Steps:
    # - load regridded climo files for model and obs
    # - calculate all-time and seasonal fields (from individual months)
    # - Take difference, calculate statistics
    # - make plot

    #Notify user that script has started:
    print("  Generating lat/lon maps...")

    #Set input/output data path variables:
    #------------------------------------
    dclimo_loc    = Path(data_loc)
    mclimo_rg_loc = Path(model_rgrid_loc)
    plot_loc      = Path(plot_location)
    #-----------------------------------

    #Set seasonal ranges:
    seasons = {"ANN": np.arange(1,13,1),
               "DJF": [12, 1, 2],
               "JJA": [6, 7, 8],
               "MAM": [3, 4, 5],
               "SON": [9, 10, 11]}

    #Set plot file type:
    plot_type = 'png'

    #Check if plot output directory exists, and if not, then create it:
    if not plot_loc.is_dir():
        print("    {} not found, making new directory".format(plot_loc))
        plot_loc.mkdir(parents=True)

    # probably want to do this one variable at a time:
    for var in var_list:

        #Notify user of variable being plotted:
        print("\t \u231B lat/lon maps for {}".format(var))

        #loop over different data sets to plot model against:
        for data_src in data_list:

            # load data (observational) commparison files (we should explore intake as an alternative to having this kind of repeated code):
            oclim_fils = sorted(list(dclimo_loc.glob("{}_{}_*.nc".format(data_src, var))))

            if len(oclim_fils) > 1:
                oclim_ds = xr.open_mfdataset(oclim_fils, combine='by_coords')
            elif len(oclim_fils) == 1:
                sfil = str(oclim_fils[0])
                oclim_ds = xr.open_dataset(sfil)
            else:
                print("ERROR: Did not find any oclim_fils. Will try to skip.")
                print(f"INFO: Data Location, dclimo_loc is {dclimo_loc}")
                print(f"INFO: The glob is: {data_src}_{var}_*.nc")
                continue

            # load re-gridded model files:
            mclim_fils = sorted(list(mclimo_rg_loc.glob("{}_{}_{}_*.nc".format(data_src, case_name, var))))

            if len(mclim_fils) > 1:
                mclim_ds = xr.open_mfdataset(mclim_fils, combine='by_coords')
            else:
                mclim_ds = xr.open_dataset(mclim_fils[0])

            #Extract variable of interest
            odata = oclim_ds[var].squeeze()  # squeeze in case of degenerate dimensions
            mdata = mclim_ds[var].squeeze()

            #Determine dimensions of variable:
            has_dims = pf.lat_lon_validate_dims(odata)
            if has_dims:
                #If observations/baseline CAM have the correct
                #dimensions, does the input CAM run have correct
                #dimensions as well?
                has_dims_cam = pf.lat_lon_validate_dims(mdata)

                #If both fields have the required dimensions, then
                #proceed with plotting:
                if has_dims_cam:

                    #
                    # Seasonal Averages
                    # Note: xarray can do seasonal averaging,
                    # but depends on having time accessor,
                    # which these prototype climo files do not have.
                    #

                    #Create new dictionaries:
                    mseasons = dict()
                    oseasons = dict()
                    dseasons = dict() # hold the differences

                    #Loop over season dictionary:
                    for s in seasons:
                        mseasons[s] = mdata.sel(time=seasons[s]).mean(dim='time')
                        oseasons[s] = odata.sel(time=seasons[s]).mean(dim='time')
                        # difference: each entry should be (lat, lon)
                        dseasons[s] = mseasons[s] - oseasons[s]

                        # time to make plot; here we'd probably loop over whatever plots we want for this variable
                        # I'll just call this one "LatLon_Mean"  ... would this work as a pattern [operation]_[AxesDescription] ?
                        plot_name = plot_loc / "{}_{}_LatLon_Mean.{}".format(var, s, plot_type)

                        #Remove old plot, if it already exists:
                        if plot_name.is_file():
                            plot_name.unlink()

                        #Create new plot:
                        pf.plot_map_and_save(plot_name, mseasons[s], oseasons[s], dseasons[s])

                else: #mdata dimensions check
                    print("\t \u231B skipping lat/lon map for {} as it doesn't have only lat/lon dims.".format(var))

            else: #odata dimensions check
                print("\t \u231B skipping lat/lon map for {} as it doesn't have only lat/lon dims.".format(var))

    #Notify user that script has ended:
    print("  ...lat/lon maps have been generated successfully.")

##############
#END OF SCRIPT
