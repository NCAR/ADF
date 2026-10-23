# INFORM input data: which code made the saved files

This note records how the INFORM observation files that the ADF reads were
produced, how the current INFORM-COMPASS cookbook code differs from the code
that produced them, and what the ADF now carries to regenerate them.  Findings
are from checks run on Casper on 2026-09-25 and 2026-09-28, in the
`compass-env` conda environment (xarray 2025.3.1, pandas 2.2.3, numpy 1.26.4).

## Summary

| Saved files | Reproduced by the current code? | Code that reproduces them |
|---|---|---|
| Per-flight cloud regime files, `<CAMPAIGN>_RF<nn>_<date>.nc` | **No.** `M` and `EIS` differ on almost every row; 27 % of SOCRATES RF10 rows change regime | **Found**: the current code with four settings changed (below).  Bit-for-bit on SOCRATES RF10 and CSET RF01, apart from two droplet column names |
| Dropsonde composites, `<campaign>_sonde_data_composite.nc` | **No.** `M`, `RH700`, `EIS`, `Tadv` differ; 1 of 5 SOCRATES RF10 sondes changes regime | **Not found yet**.  Older than the flight files: its `RH700` also differs |

The current code also has a probable bug: in `select_ERA5_4flight`,
`ds_t850 = ds_t700.sortby(...)` reads the 700 hPa field, so the "T850" used in
`EIS` is really T700 (see [The T850 line](#the-t850-line)).

## The files

| Files | Where | Dates |
|---|---|---|
| 16 CSET and 15 SOCRATES per-flight files | `/glade/u/home/patnaude/inform/composited data/` | CSET `*_20260210.nc`, SOCRATES `*_20260213.nc` (the date in the name is the day `write_RF_nc` ran) |
| `SOCRATES_sonde_data_composite.nc` | `/glade/u/home/patnaude/inform/` | 2025-12-03 |
| `CSET_sonde_data_composite.nc` | `/glade/u/home/patnaude/inform/` | 2026-03-04 |

Copies elsewhere are byte-identical: the 31 flight files in
`/glade/derecho/scratch/richling/inform/ryan_composited_data/` and
`ryan_campaign_data/<campaign>/`, and both sonde composites in
`ryan_composited_data/`.  The unit tests do not read any of these: they use the
small fixtures in `lib/test/unit_tests/test_files/inform/`, and the full-data
checks read the flight files from patnaude's directory.

## The code versions

`process_data_products_utils.py` in the cookbook
(`https://github.com/NCAR/INFORM-COMPASS-cookbook`) has three commits:

| Commit | Date | Relevant content |
|---|---|---|
| `fd60b0b` | 2025-06-11 | File created |
| `4706811` | 2025-10-15 | Surface θ from SST; T850 = (T2m + T700)/2; RH700 from ERA5's `R` field; no dropsonde functions (`inform_utils` has `read_sonde2df` only) |
| `ab6f8da` | 2026-03-06 | Surface θ from T2m and surface pressure; T850 from ERA5 850 hPa (with the typo); RH700 computed from specific humidity; dropsonde functions (`read_sonde`, `collocate_ERA5_sonde`, `cloud_regime_sonde`) added |

The cookbook's latest commit is `6947666` (2026-05-27, branch
`data-wrapper-updates_030626`), and that commit is what the ADF tests check
against (`cookbook_ref.json`).  Patnaude's working copies
(`/glade/u/home/patnaude/inform/`, `process_data_products_utils.py` dated
2026-08-18 and `inform_utils.py` dated 2026-09-11) match it in every function
used here except `read_vars`.  The other
differences are a `cloud_regime` docstring (`Tadv >= -4` → `-3`; the code is
unchanged) and removed `print` calls.

`read_vars` (checked 2026-10-07): patnaude's copy also excludes aircraft
variables containing `_2`, `1DC` or `HDC` (and `PLW*` with `_2` or `1DC`
instead of `2V`).  Flight files rebuilt with his copy would lose
`CONC1DC_*`, `CONC2DSA_2H/2V`, `PLWC1DC_*`, `PLWC2DSR_2H` and `PLWC2DSA_2H`
(SOCRATES) or `CONC1DC_*` and `PLWC1DC_*` (CSET).  The saved files and the
ADF keep them.  No ADF diagnostic reads these columns, and the change does not
touch `M`, `EIS`, `RH700`, `Tadv` or the regimes.

Both kinds of saved file predate `ab6f8da` but come from code newer than
`4706811`.  That code was never committed.  The flight files could still be
traced because `composite_flight_data_ERA5.ipynb` (patnaude, 2026-02-18, now in
`original/`) contains the intermediate formulas.

## Per-flight files

### How the current code differs

Rebuilding SOCRATES RF10 with the current code (the notebook's per-flight loop,
original functions) and comparing with `SOCRATES_RF10_20260213.nc`:

- Same flight blocking and rows (26,297), and identical `RH700`, SST, winds and
  every aircraft variable.
- `M` differs on 25,380 rows (max 2.85 K) and `EIS` on 25,380 rows (max 3.91 K).
- `cloud_regime` differs on 7,105 rows (27 %):

  | Saved \ current | Open-Cell | Stratocumulus | Undetermined |
  |---|---:|---:|---:|
  | Open-Cell | 3,471 | 0 | 0 |
  | Stratocumulus | 1,070 | 15,721 | 1,616 |
  | Undetermined | 3,520 | 899 | 0 |

- Columns: the saved file has `Nliq_2DC_ge35`, `Nliq_CDP_lt35` and
  `Nliq_stitched_35um`.  The current code writes `Nliq_2DS_ge25`,
  `Nliq_CDP_lt25` and `Nliq_stitched_25um`, plus CCN columns (`CCN_const`,
  `CCN_error_const`, `CCN_scan`, `CCN_error_scan`, `SS_percent_const`,
  `SS_percent_scan`).

### Finding the version that made them

`M` depends only on surface θ, and `EIS` also depends on T850.  Every
combination of the surface-θ and T850 formulas in the git history and the
notebook was tried against SOCRATES RF10, with everything else unchanged:

| Surface θ | T850 | `M` exact | max \|ΔM\| (K) | `EIS` exact | max \|ΔEIS\| (K) |
|---|---|:---:|---:|:---:|---:|
| T2m·(p₀/pₛ)^κ (current) | T700 (current typo) | no | 2.85 | no | 3.91 |
| T2m·(p₀/pₛ)^κ (current) | (T2m + T700)/2 | no | 2.85 | no | 2.85 |
| T2m·(p₀/pₛ)^κ (current) | true T850 | no | 2.85 | no | 2.98 |
| T2m | T700 | no | 2.24 | no | 3.95 |
| T2m | (T2m + T700)/2 | no | 2.24 | no | 2.24 |
| T2m | true T850 | no | 2.24 | no | 2.44 |
| SST | T700 | **yes** | 0 | no | 2.53 |
| **SST** | **(T2m + T700)/2** | **yes** | **0** | **yes** | **0** |
| SST | true T850 | **yes** | 0 | no | 1.05 |

### The version that made them

The current code with these three changes:

| | Saved files (February 2026) | Current code |
|---|---|---|
| Surface θ, in `select_ERA5_4flight` | `theta_sfc = ds_sst.SSTK*(1)**(Rd/Cp)` | `theta_sfc = ds_t2m.VAR_2T*(101325/ds_sp.SP)**(Rd/Cp)` |
| T850, in `select_ERA5_4flight` | `T850 = (ds_t2m.VAR_2T+ds_t700.T)/2` | `T850 = ds_t850.T` (really T700, see below) |
| Liquid droplet size split, `calc_concs_from_sd` | `d_split_um=35.0` | `d_split_um=25.0` |

The flight blocking (`VAP_process_flight_data`, `assign_flight_type`,
`block_flight`), the RH700 calculation from specific humidity,
`collocate_ERA5_dat`, `cloud_regime` and `write_RF_nc` are already identical,
and neither merges CCN (`ccn_df=None`).

Two column differences remain, both from an older `calc_concs_from_sd`:

- The ≥ 35 µm probe column was named `Nliq_2DC_ge35`; the current code names it
  `Nliq_2DS_ge35`.  The values are identical.
- CSET files have no `Nliq_CDP_lt35` column; the current code adds it (that
  branch is marked "NEW" in the code).

With those accounted for, the rebuilt files are **identical to the saved ones
in every variable, value and attribute**: SOCRATES RF10 (67 variables) and
CSET RF01 (57 variables).  Other flights were not rebuilt.

### Reproducing it

The experiment, using the ADF's aircraft loaders (they differ from the
original only in how they read the config) and the original
`process_data_products_utils`:

```python
import ast, functools, types
import inform_utils as iu                               # ADF lib/
from original_loader import load_originals, original_dir  # lib/test/inform_reference/
pdp = load_originals()["process_data_products_utils"]       # the pinned cookbook commit

# select_ERA5_4flight with the February 2026 surface theta and T850:
src = (original_dir() / "process_data_products_utils.py").read_text()
fn = next(n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "select_ERA5_4flight")
code = (ast.get_source_segment(src, fn)
        .replace("theta_sfc = ds_t2m.VAR_2T*(101325/ds_sp.SP)**(Rd/Cp)",
                 "theta_sfc = ds_sst.SSTK*(1)**(Rd/Cp)")
        .replace("T850 = ds_t850.T", "T850 = (ds_t2m.VAR_2T+ds_t700.T)/2"))
ns = dict(pdp.__dict__); exec(code, ns)
select_feb2026 = ns["select_ERA5_4flight"]

iu.calc_concs_from_sd = functools.partial(iu.calc_concs_from_sd, d_split_um=35.0)

campaign, rf = "SOCRATES", 10
P = f"/glade/work/patnaude/{campaign}"
icf = types.SimpleNamespace(campaigns_dict={
    "name": [campaign],
    "air_1hz_dir": [f"{P}/aircraft_1hz"],
    "air_25hz_dir": [f"{P}/aircraft_25hz"]})
df = iu.load_flight_data(campaign, rf - 1, ccn_df=None, icf_obj=icf)   # no CCN
if ('PLWC' in df.columns) and (campaign == 'SOCRATES'):
    df = df.drop(columns=['PLWC'])
blocks = pdp.VAP_process_flight_data(df, rf - 1)
fblks = pdp.cloud_regime(pdp.collocate_ERA5_dat(select_feb2026(df, campaign), blocks),
                         campaign=campaign)
pdp.write_RF_nc(fblks, f"RF{rf:02d}", campaign)       # writes to the working directory
```

## Dropsonde composites

Rebuilding the five SOCRATES RF10 sondes with the current code (the notebook's
sonde cells, original functions) and comparing with the RF10 rows of
`SOCRATES_sonde_data_composite.nc`:

- Same rows (9,388) and identical `ERA5_SST`, `Wind_sp`, `Wind_shear` and
  `deltaT`.
- Differences, over the 4,577 rows with valid values:

  | Variable | Rows differing | max \|Δ\| | Saved mean |
  |---|---:|---:|---:|
  | `M` | 4,577 | 2.66 K | −10.58 |
  | `RH700` | 4,577 | 5.73 % | 43.9 |
  | `EIS` | 4,577 | 2.29 K | 7.44 |
  | `Tadv` | 1,861 | 0.71 K day⁻¹ | 1.61 |

- Regime labels, one per sonde (drops 1–3: Stratocumulus in both; drop 5:
  Open-Cell in both): **drop 4 changes from Stratocumulus to Unknown**.
- Columns: the saved file has `w700` and `drop_id`; the current code writes
  `Omega700` instead and no `drop_id`.

Unlike the flight files, `RH700` differs too, so the sonde composites are older
than the flight files and probably take RH700 from ERA5's `R` field (as
`4706811` did).  The `4706811` code cannot be run as-is: it has no dropsonde
functions.  A search like the one above, with RH700 as a third choice, has
not been done yet.

The cookbook code that builds the composites is extracted verbatim, with source
line references, in `build_sonde_composite_original.py`.

## The T850 line

In the current `select_ERA5_4flight`:

```python
ds_t850 = xr.open_mfdataset(get_matching_files(f"{filepath_pl}{dir_date}/*_t.*.nc", start_dt, end_dt),
                             combine='nested', concat_dim='time')[['T']].sel(level=850).drop_vars('level', errors='ignore')
ds_t850 = ds_t700.sortby('time').sel(latitude=lat_slice, longitude=lon_slice, time=slice(start_dt,end_dt))
```

The second line overwrites the 850 hPa field with the 700 hPa one, so
`T850 = ds_t850.T` is T700, and the moist-adiabatic lapse rate in `EIS` is
evaluated at 700 hPa.  The saved files used neither: they used
(T2m + T700)/2.  For SOCRATES RF10, the maximum `EIS` difference from the saved
files is 2.53 K with this line as written, and 1.05 K with the true 850 hPa
field.  This is presumably meant to be `ds_t850.sortby(...)`; it should be
confirmed with Ryan and fixed in the cookbook and the ADF together.

## What the ADF now carries

In `lib/inform_utils.py`, all verified against the current cookbook code
(`lib/test/unit_tests/test_inform_vs_original.py`; these tests need the raw
data and RDA ERA5 on GLADE and are skipped elsewhere):

| Functions | Ported from | Test |
|---|---|---|
| `select_ERA5_4flight`, `wrap180`, `nearest_time_indices`, `collocate_ERA5_sonde`, `cloud_regime_sonde` | `process_data_products_utils.py`; `select_ERA5_4flight` uses this module's `datetime`/`timedelta` names | `test_sonde_composite_matches_original` (3 raw sondes per campaign) |
| `sonde_composite_dataset`, `build_sonde_composite` | the notebook's sonde cells | same |
| `VAP_process_flight_data`, `assign_flight_type`, `block_flight`, `collocate_ERA5_dat`, `cloud_regime` | `process_data_products_utils.py`, verbatim | `test_flight_file_matches_original` (SOCRATES RF11, CSET RF01) |
| `write_RF_nc` | `process_data_products_utils.py`; this module's `datetime` name and an optional `out_dir` | same |
| `build_flight_cloud_regime_files` | the notebook's per-flight loop | same |

How they are used:

- `inform_model_analysis` builds `<campaign>_sonde_data_composite.nc` into
  `composited_data` when that file is missing or `redo_composited` is true, and
  always reads it back from the file.
- The flight builder is not wired into the ADF run.  Its files carry the date
  they were written, so do not write them into a folder that already holds
  flight files: `inform_model_analysis` reads `*<campaign>*RF*.nc` from
  `campaign_data` and would read every flight twice.
- Both builders use the **current** science, so neither reproduces the saved
  files.

## Open items

1. **Option to reproduce the February 2026 flight files.** Add settings for
   surface θ and T850 to `select_ERA5_4flight`, and for the droplet split, CCN
   and the column names to `build_flight_cloud_regime_files`.  A test could then
   require a bit-for-bit match with patnaude's saved files as well as with the
   current cookbook.
2. **Find the version that made the sonde composites**: the same search, with
   RH700 (ERA5 `R` vs specific humidity) added, on SOCRATES RF10 and a CSET
   flight.
3. **The T850 line**: confirm with Ryan and fix in both places.
4. **Consistency**: the ADF compares CAM against flight files and sonde
   composites built with different code versions from each other, and from
   the current cookbook.  Decide which version the diagnostics should use
   before regenerating either.
