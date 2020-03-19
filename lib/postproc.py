import argparse
import glob
import sys
import subprocess
from pathlib import Path
import json


def main(in_files, var_list, output_dir):
    """Use specified files and variable list to construct ncrcat command and save outut file."""
    first_in = Path(in_files[0])
    case_spec = first_in.stem # this will be the filename without suffix
    for v in var_list:
        out_file = output_dir / f"{case_spec}.ncrcat.{v}.nc"
        print(f"Writing to: {out_file}")
        cmd = ["ncrcat", "-4", "-v", v] + in_files + ["-o", out_file]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract a timeseries file from a list of files.')
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
    assert 'file_desc' in inputs  # Describe the files, for example "*.h0.*.nc"
    assert 'variables' in inputs  # list of variables to process into time series
    assert 'output_loc' in inputs  # Directory path for output files
    #
    # sort the list of files
    #
    starting_location = Path(inputs['input_loc'])
    files = sorted(list(starting_location.glob(inputs['file_desc'])))
    if not files:
        print('File does not exist.' )
    else:
        print(f"Input seems to exist, pass {len(files)} to ncrcat")
        out_loc = Path(inputs['output_loc'])
    assert out_loc.is_dir()
    main(files, inputs['variables'], out_loc)
