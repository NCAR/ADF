Ryan has composited data at `/glade/u/home/patnaude/inform/composited data`
    - Has individual flight nc files. Are these from EOL?? If not how do these get created?

Inside Ryan's dir `/glade/u/home/patnaude/inform/` are post processed nc files for the composited sonde data
    - SOCRATES_sonde_data_composite.nc
    - CSET_sonde_data_composite.nc


Yes, the two files are different in several places. The sizes also differ (34 MB for the old one, 29 MB for the new one). Both have the same `index` dimension (98,941 points) and the same global attributes.

**Variables that were added, removed or renamed:**
- `drop_id` (a string ID for each sonde, e.g. `'1_d1'`) is only in the **old** file (`ryan_old_composited_data`).
- `w700` in the old file is called `Omega700` in the new file. The two correlate at exactly **−1.000**, and the new values span a much wider range (old [−1.51, 0.72], new [−6.31, 12.9]). So they look like the same field with the sign flipped and a different scale, most likely vertical velocity *w* in the old file and pressure velocity *ω* in the new one.

**Variables with different values:**

| Variable                                      | Values that differ | Max abs diff | Notes                                                                           |
| --------------------------------------------- | ------------------ | ------------ | ------------------------------------------------------------------------------- |
| `EIS`                                         | 48,085             | 6.47         | new range [0.53, 15] vs old [−5.94, 11.3]; corr 0.87                            |
| `M`                                           | 48,085             | 4.51         | new range [−18.4, −2.17] vs old [−15.6, 0.83]; corr 0.89                        |
| `RH700`                                       | 48,085             | 11.15        | corr 1.000, mean abs diff about 1.1                                             |
| `w700`/`Omega700`                             | 48,085             | n/a          | see above                                                                       |
| `Tadv`                                        | 24,568             | 6.46         | corr 0.98                                                                       |
| `cloud_regime`                                | 26,691             | n/a          | mostly `Unknown` → `Open-Cell` (16,813) and `Unknown` → `Stratocumulus` (4,040) |
| `ERA5_SST`, `Wind_shear`, `Wind_sp`, `deltaT` | 2 each             | ≤ 0.4        | tiny                                                                            |

- **The 700 hPa fields change together.** `EIS`, `M`, `RH700` and `w700`/`Omega700` differ at exactly the same 48,085 points, and those points include all 55 drops. This suggests the 700 hPa input data or how it's handled changed between the two runs.
- **`cloud_regime` follows from that.** Its changes are probably a knock-on effect of the new EIS/M/omega values, with many points that were `Unknown` now classified.
- **The 2-value differences are probably not real.** The mismatch in `ERA5_SST`, `Wind_shear`, `Wind_sp` and `deltaT` is likely a missing value in one file compared against a real value in the other. I tried to find which points they are, but that last check was faulty (it counted matching missing values as differences), so I didn't confirm it.

All other variables are identical, with missing values treated as equal. I used the NPL conda Python (`/glade/u/apps/opt/conda/envs/npl/bin/python`) because the default `python` doesn't have xarray. I can pin down those 2-point differences or dig into the 700 hPa change if you want.


So my version is not calculating the same composited data as Ryan's?


write_RF_nc