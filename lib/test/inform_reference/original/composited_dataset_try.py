###### INFORM .py files necessary for processing aircraft data
import inform_utils as inform # INFORM python module inform_utils.py
import process_data_products_utils as pdp # INFORM python module, process_data_products
#######

import xarray as xr
from datetime import time
import warnings
warnings.filterwarnings('ignore')
from tqdm.notebook import tqdm, trange
import dask
import pandas as pd

campaign = 'SOCRATES' # SOCRATES or CSET

if campaign == 'SOCRATES':
    dir = '/glade/work/patnaude/SOCRATES/aircraft_1hz/'
    ccn_dir = '/glade/work/patnaude/SOCRATES/CCN'
elif campaign  == 'CSET':
    dir = '/glade/work/patnaude/CSET/aircraft_1hz/'

ccn_df = inform.load_ccn_for_campaign(campaign)  # loads once

#flight_paths = inform.find_flight_fnames(dir)
#flight_paths = find_flight_fnames(dir)
flight_paths = inform.find_flight_fnames(campaign)
all_fblks = {}

for i in tqdm(range(0,len(flight_paths))):
    
    """
    def load_flight_data(
        campaign: str,
        idx: int = 0,
        add_sizedist: bool = True,
        ccn_df: pd.DataFrame | None = None,
        add_sigma_w: bool = True,
        sigma_min_samples: int = 20,
        asof: bool = False,
        tol: str = "1s",
    )
    """
    df = inform.load_flight_data(campaign, i, ccn_df=ccn_df)

    
    if ('PLWC' in df.columns) and (campaign == 'SOCRATES'): # RF12 is the only fligth with this variable and it messes up the final product
        df = df.drop(columns=['PLWC'])
    blocks = pdp.VAP_process_flight_data(df, i)

    # Select ERA5 data
    ds = pdp.select_ERA5_4flight(df, campaign)
    rf_id = f"RF{i+1:02d}"
    print(rf_id)
          
    # Collocate ERA5 data and calculate environmental controlling factors
    fblks_coll = pdp.collocate_ERA5_dat(ds, blocks)
    #print("fblks_coll",fblks_coll,"\n\n-------------------------------\n\n")
    # Select cloud regime type based on cloud controlling factors
    fblks_cr = pdp.cloud_regime(fblks_coll, campaign=campaign)
    # Write to NetCDF for this flight
    pdp.write_RF_nc(fblks_cr, rf_id, campaign)

    # Save to master dictionary
    # all_fblks[rf_id] = fblks_cr