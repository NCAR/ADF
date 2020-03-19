from pathlib import Path
import argparse
import json
import xarray as xr

def cesm_load_data(fils, correct_time=True):
    print(fils)
    if isinstance(fils, list):
        if len(fils) == 1:
            ds = xr.open_dataset(fils[0], decode_times=correct_time)
        else:
            ds = xr.open_mfdataset(fils, decode_times=correct_time, combine='by_coords')
    else:
        ds = xr.open_dataset(fils, decode_times=correct_time)
    if correct_time:
        if 'time_bnds' in ds:
             ds['time'].values = ds['time_bnds'].mean(dim='nbnd')
             ds = xr.decode_cf(ds)
    return ds


def write_out(ds, ofil):
    # deal with getting non-nan fill values.
    enc_dv = {xname: {'_FillValue': None, 'zlib': True, 'complevel': 4} for xname in ds.data_vars}
    enc_c = {xname: {'_FillValue': None} for xname in ds.coords}
    enc = {**enc_c, **enc_dv}
    ds.to_netcdf(ofil, format='NETCDF4', encoding=enc)


def main(fils, output_location):
    ds = cesm_load_data(fils)
    ds_climo = ds.groupby('time.month').mean(dim='time')
    ds_climo = ds_climo.rename({'month':'time'})
    write_out(ds_climo, output_location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate climo file from one or more time series files.')
    parser.add_argument('namelist', type=str, help="Path to json namelist file")
    args = parser.parse_args()
    #
    # read the input file
    #
    with open(args.namelist) as f:
        inputs = json.load(f)
    #
    # now we make sure that necessary input is supplied
    #
    assert 'input_loc' in inputs  # The directory path of input files
    assert 'case_name' in inputs  
    assert 'output_loc' in inputs  # Directory path for output files
    #
    # sort the list of files
    #
    starting_location = Path(inputs['input_loc'])
    out_loc = Path(inputs['output_loc'])
    assert out_loc.is_dir()    
    if 'variables' in inputs:
        for v in inputs['variables']:
            files = sorted(list(starting_location.glob(f'{inputs["case_name"]}*{v}*.nc')))
            output_file = out_loc / f"{inputs['case_name']}_{v}_climo.nc"
            main(files, output_file)
    else:
        files = sorted(list(starting_location.glob(f"{inputs['case_name']}*.nc")))
        output_file = out_loc / f"{inputs['case_name']}_climo.nc"
        main(files, output_file)
