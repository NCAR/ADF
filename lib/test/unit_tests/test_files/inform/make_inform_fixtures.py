"""Cut the small INFORM fixtures in this directory out of the full per-flight
cloud regime files on GLADE.

Each fixture keeps the first few rows of the first few blocks of every
block_label, in the original file order, from two flights per campaign.  The
fixtures are real data, just small enough (a few tens of KB) to commit.

The expected outputs (*_combined_RF_fixture.csv) are written by the unit test
itself, with INFORM_UPDATE_REFERENCE=1, so they always come from the code under
test rather than from this script.

Run from anywhere:

    python make_inform_fixtures.py [/path/to/composited data]
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

SRC = Path(sys.argv[1] if len(sys.argv) > 1
           else "/glade/u/home/patnaude/inform/composited data")
OUT = Path(__file__).parent

#Two flights per campaign, chosen so every block_label appears:
FLIGHTS = {
    "CSET": ["CSET_RF02_20260210.nc", "CSET_RF03_20260210.nc"],
    "SOCRATES": ["SOCRATES_RF01_20260213.nc", "SOCRATES_RF02_20260213.nc"],
}
BLOCKS_PER_LABEL = 2
ROWS_PER_BLOCK = 4


def _slice(ds):
    """Index positions of the rows to keep, in file order."""
    labels = ds["block_label"].values
    indices = ds["block_index"].values
    keep = []
    for label in dict.fromkeys(labels):
        in_label = labels == label
        for idx in list(dict.fromkeys(indices[in_label]))[:BLOCKS_PER_LABEL]:
            rows = np.flatnonzero(in_label & (indices == idx))
            keep.extend(rows[:ROWS_PER_BLOCK])
    return sorted(keep)


for campaign, fnames in FLIGHTS.items():
    for fname in fnames:
        with xr.open_dataset(SRC / fname) as ds:
            small = ds.isel(index=_slice(ds)).load()
        rf = fname.split("_")[1]
        out = OUT / f"{campaign}_{rf}_fixture.nc"
        small.to_netcdf(out)
        print(f"{out.name}: {small.sizes['index']} rows from {fname}")
