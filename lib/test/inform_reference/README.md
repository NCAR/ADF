# INFORM reference material

The INFORM/COMPASS diagnostics in the ADF were ported from the scripts and
notebooks of the INFORM-COMPASS cookbook.  The unit tests check every change to
the ADF port against that code **as committed on GitHub**, at the commit pinned
in `cookbook_ref.json`.

`DATA_PROVENANCE.md` records which version of this code made the saved INFORM
observation files the ADF reads (the per-flight `*_RF*.nc` files and the
dropsonde composites), and how that differs from the current code.

## The reference

| | |
|---|---|
| Repository | `https://github.com/NCAR/INFORM-COMPASS-cookbook` |
| Branch | `data-wrapper-updates_030626` (`main` is older and does not have this code) |
| Commit | `6947666a0d311ba94fd964825261b74b42b1b20c` ("Mod 2 py code and notebook", 2026-05-27) |

`original_loader.py` extracts the pinned commit's files with `git archive`, so
only committed content is used, never a working copy.  It gets the commit from:

1. `INFORM_COOKBOOK_DIR`, if set: a local clone that has the commit.  Its
   checked-out branch and uncommitted edits do not matter.
2. Otherwise a shallow fetch of that one commit from GitHub, kept as a bare
   repository in the cache.

The files land in `$INFORM_COOKBOOK_CACHE/<commit>/` (default
`~/.cache/adf/inform-cookbook/<commit>/`, a few MB), and later runs reuse
them.  If the commit cannot be obtained (no git, no network and no clone), the
tests that need it are skipped.  `test_cookbook_pin_resolves` warns when the
branch on GitHub has moved past the pin.

### `original/`

The working-copy snapshot the tests used before 2026-10-06 (commit `6947666`
plus uncommitted edits, taken 2026-09-25).  The tests no longer read it.  It is
kept because it holds files the cookbook never committed:

- `composite_flight_data_ERA5.ipynb` (patnaude, 2026-02-18): the formulas that
  traced the February 2026 flight files (see `DATA_PROVENANCE.md`);
- `composited_dataset_try.py`, `Untitled.ipynb`.

`composite_flight_data_ERA5.ipynb` was re-run in place on 2026-10-08 (to make
the `SOCRATES_*_20261008.nc` outputs, which are not committed) and the
2026-09-25 copy could not be recovered, so its `MANIFEST.json` hash is that of
the 2026-10-08 file.

Its other files differ from the commit only by three `print()` calls in
`inform_utils.py` and edits to the notebooks (paths, timing cells, and the
per-flight loop passing `ccn_df=ccn_df`, which the ADF does not follow; see below).  **Do not edit
anything in `original/`**: `MANIFEST.json` holds a SHA-256 for each file, and
`test_reference_snapshot_unchanged` fails if one changes.

## What maps to what

The ADF model analysis (`scripts/plotting/inform_model_analysis.py` and the
`model_analysis_*_scripts/` it calls) follows
`INFORM_cam_assessment_diagnostics.ipynb`, using the cookbook's
`inform_utils.py`, `cam_era5_functions.py` and `cam_diagnostic.py`, which the ADF
carries as `lib/inform_utils.py`, `lib/cam_era5_functions.py` and
`lib/cam_diagnostic.py`.

Cell numbers are those of the notebooks at the pinned commit.

| Notebook step (cell) | ADF |
|---|---|
| `inform.load_nc_cldrgme` (4, 13) | `load_nc_cldrgme` in `inform_model_analysis.py` (and `lib/inform_utils.py`) |
| `cfun.load_era5_files`, `load_cam_files`, `compute_cam_aerosol_micphys_metrics` (6) | `inform_model_analysis.py` |
| `cfun.load_cam_ccfs`, `subset_cam_by_campaign` (8) | `inform_model_analysis.py` |
| Dropsonde collocation and 3-panel profile plot (11) | `model_analysis_macro_scripts/cam_obs_dropsonde_comp.py` |
| `cdog.plot_obs_cam_nd_lwc_pdfs` (13) | `model_analysis_micro_scripts/cam_obs_nd_lwc_pdf.py` |
| `cdog.plot_sensitivity_vs_sigmaw_2x2_obs_cam` (16) | `model_analysis_process_scripts/sensitivity_vs_sigmaw_2x2_obs_cam.py` |
| — (no original) | `model_analysis_process_scripts/alpha_sigmaw_regime_comparison_2x2.py` |

`INFORM_process_system_database.ipynb` (with `process_data_products_utils.py`)
is what builds the per-flight `*_RF*.nc` cloud regime files and the
`*_sonde_data_composite.nc` files the ADF reads.  The ADF does not port it; it
is kept here for reference.

`build_sonde_composite_original.py` pulls out, verbatim, everything that builds
`<campaign>_sonde_data_composite.nc`: the notebook's sonde cells plus every
`inform_utils.py` and `process_data_products_utils.py` function they call, each
labelled with its source lines.  The ADF now carries this code in
`lib/inform_utils.py` (`build_sonde_composite` and the ERA5 collocation/regime
functions), and `inform_model_analysis` builds the composite into
`composited_data` when it is missing or `redo_composited` is true.  The current
code does not reproduce the saved composites below: they came from an earlier,
unrecoverable version (different M, RH700, EIS, Tadv and some regime labels).  As written, the notebook's CSET branch uses `np` without importing numpy,
and the file is written to the current directory.  The composites the ADF uses
are patnaude's originals (`/glade/u/home/patnaude/inform/`, SOCRATES dated
2025-12-03, CSET 2026-03-04).

## The checks

`lib/test/unit_tests/test_inform_vs_original.py`:

1. **Reference** — the pinned commit can be fetched, and `original/` still
   matches `MANIFEST.json`.
2. **Source drift** — every function in the three ADF modules is registered as
   either *same as the original* (its code, ignoring docstrings, comments,
   `print` calls and `timer` lines, must match the original's), *changed and
   verified by a behavioural test*, or *ADF only*.  Edit a function that was the
   same as the original and this fails until you revert it or register it as
   changed, with a behavioural test.
3. **Behaviour** — the original and the ADF code are run on the same small real
   fixtures (`lib/test/unit_tests/test_files/inform/`: one research flight per
   campaign, with its CAM, ERA5, dropsonde and aircraft data) and must give
   identical results: the combined flight table, the CAM datasets and regime
   composites, the Nd/LWC PDFs, the sensitivity tables, the dropsonde profiles,
   and every curve the original plots draw.
4. **ADF regression** — each ADF diagnostic is run end to end on the fixtures and
   the data behind every figure it saves is compared with a stored fingerprint;
   it is then run again from its caches with the CAM and ERA5 inputs replaced by
   objects that fail on any access, which proves a cached run needs no model
   I/O and draws the same figures.

Run them (they need the ADF conda environment, plus pytest, and git):

    pytest lib/test/unit_tests/test_inform_vs_original.py -v

    # or use your own clone of the cookbook instead of fetching from GitHub:
    INFORM_COOKBOOK_DIR=/path/to/INFORM-COMPASS-cookbook pytest lib/test/unit_tests/test_inform_vs_original.py -v

The flight-file and sonde-composite comparisons use one flight and three
sondes per campaign.  To repeat them over every flight (31) and every sonde
(175), which takes hours and should run as a batch job on Casper:

    INFORM_FULL_REGRESSION=1 pytest lib/test/unit_tests/test_inform_vs_original.py -k "every_flight or full_sonde or flight_count"

CCN: the cookbook's per-flight loop (notebook cell 19) passes `ccn_df=None`,
so its flight files have no CCN columns.  The ADF's
`build_flight_cloud_regime_files` does the same, even with a `ccn_dir`
configured, and the flight files must be identical.  (The edited notebook in
`original/` passes `ccn_df=ccn_df` instead; that is not what the tests follow.)

After an intended change to the ADF output, regenerate the stored fingerprints
and commit them, so the diff shows what changed:

    INFORM_UPDATE_REFERENCE=1 pytest lib/test/unit_tests/test_inform_vs_original.py

## Moving the pin

Only when the cookbook itself has moved on and the ADF port should follow it:
put the new commit (and branch, and any new files the tests need) in
`cookbook_ref.json`, update the table above, and run the tests.  Expect the
drift and behaviour checks to show exactly where the port now differs.  The
tests that copy notebook cells cite cell numbers; check those cells against the
new commit's notebooks too.
