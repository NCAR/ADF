# AGENTS.md — reviewing pull requests to the ADF

**Status: draft.** This file tells an AI agent (e.g. Claude) how to review a pull request to
the AMWG Diagnostics Framework (ADF). It is about *review*, not about running the ADF — user
and developer instructions live on the [wiki](https://github.com/NCAR/ADF/wiki).

Upstream repo is `NCAR/ADF`; PRs target `main`.

---

## 1. What the ADF is, in one paragraph

The ADF is a Python framework that produces standard climatological comparisons of CAM runs
(case vs. case, or case vs. obs/reanalysis). A run is driven by a single YAML config file
(`./run_adf_diag config_cam_baseline_example.yaml`) and proceeds in fixed stages: create time
series → create climatologies → regrid/vertically interpolate → analysis → plotting → build a
static website. The framework (`lib/`) owns configuration, data access, and the website; the
science (`scripts/`) is contributed largely by CAM users and is plugged in by name from the
config file.

**Key consequence for review:** almost everything under `scripts/` is dynamically imported by
name and called with a single argument. There is no static wiring to catch a broken interface,
and CI does not lint these files (see §3). Interface and runtime errors here surface only in a
real ADF run, so the review has to do that job.

## 2. Repository layout

| Path | What it is |
| --- | --- |
| `run_adf_diag` | Top-level executable driver; orders the stages. |
| `lib/adf_base.py` | `AdfBase`: debug log, `debug_log()`, `end_diag_fail()`, `AdfError`. |
| `lib/adf_config.py` | `AdfConfig`: YAML read, `${var}` reference expansion, `read_config_var()`. |
| `lib/adf_info.py` | `AdfInfo`: derived run info and the properties/getters scripts read (`diag_var_list`, `plot_location`, `climo_yrs`, `case_nicknames`, `hist_string`, `get_basic_info()`, `get_cam_info()`, `get_baseline_info()`, …). |
| `lib/adf_dataset.py` | `AdfData`: **the** data-access layer — `load_climo_da`, `load_regrid_da`, `load_reference_*`, `get_*_file`, unit converters. |
| `lib/adf_derive.py` | Variable derivation (`check_derive`, `derive_variable`). |
| `lib/adf_obs.py` | Observation-file bookkeeping. |
| `lib/adf_web.py` | `AdfWeb`: `add_website_data()` plus HTML generation from `lib/website_templates/`. |
| `lib/adf_diag.py` | `AdfDiag`: the stage methods and the dynamic script caller. |
| `lib/plotting_functions.py`, `lib/plotting_utils.py`, `lib/adf_utils.py` | Shared plotting/utility helpers used by `scripts/`. |
| `lib/adf_variable_defaults.yaml` | Per-variable plotting/obs/vector/website/derivation defaults. `..._era5-1deg.yaml` is the 1° ERA5 variant. |
| `scripts/averaging/`, `scripts/regridding/`, `scripts/analysis/`, `scripts/plotting/` | Pluggable stage scripts, selected by the config file's `time_averaging_scripts`, `regridding_scripts`, `analysis_scripts`, `plotting_scripts`. |
| `lib/test/unit_tests/` | pytest suite (currently `adf_base`, `adf_config` only). |
| `lib/test/pylintrc` | pylint config used by CI. |
| `.github/workflows/`, `.github/scripts/` | CI (see §3). |
| `config_*.yaml` | Example configs kept in sync with the code. |

## 3. What CI actually checks — and what it does not

Three workflows run on every PR:

1. **`ADF_unit_tests.yaml`** — `pytest lib/test/unit_tests` on Python 3.9–3.13. Only PyYAML and
   pytest are installed, so anything importing xarray/matplotlib cannot be unit-tested as written.
2. **`ADF_linting.yaml`** — `.github/scripts/pr_mod_file_tests.py` runs pylint at
   **threshold 9.5** with `lib/test/pylintrc`, but only on modified files in this hard-coded
   `testable_files` set:
   `lib/adf_base.py`, `lib/adf_config.py`, `lib/adf_info.py`, `lib/adf_obs.py`,
   `lib/adf_web.py`, `lib/adf_diag.py`.
3. **`ADF_pre-commit.yaml`** — `pre-commit run -a`; the only hook configured is `check-yaml`.

**Un-linted, un-tested by CI:** everything in `scripts/`, plus `lib/adf_dataset.py`,
`lib/adf_derive.py`, `lib/adf_utils.py`, `lib/plotting_functions.py`, `lib/plotting_utils.py`.
Weight the review accordingly — a PR touching only `scripts/plotting/` gets a green checkmark
from a suite that never imported it.

Reproduce the checks locally:

```bash
pytest lib/test/unit_tests
pylint --rcfile=lib/test/pylintrc lib/adf_diag.py          # any file in testable_files
pre-commit run -a
python -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" config_cam_baseline_example.yaml
```

A useful extra pass that CI will not do — syntax/import sanity on changed scripts:

```bash
python -m py_compile $(git diff --name-only main...HEAD -- '*.py')
```

## 4. Contracts to verify

These are the invariants that make the plug-in architecture work. Check every one that the
diff touches.

### 4.1 The stage-script interface

For a script `scripts/<stage>/<name>.py` listed in the config:

- The **module basename and the entry-point function name must match** (`global_latlon_map.py`
  must define `global_latlon_map`). `AdfDiag.__function_caller()` does
  `importlib.import_module(name)` then `getattr(module, name)`, and calls
  `end_diag_fail()` if the name is absent.
- The entry point is called as `func(adfobj)` — or `func(adfobj, **kwargs)` when the config
  entry is the dict form `{name: {kwargs: {...}}}`. New required positional arguments break the
  contract; new options belong in `kwargs` or the config file with defaults.
  (Caveat: the dict form's `module:` key only redirects the file-existence check and the
  `sys.path` insert — `__function_caller()` still imports `func_name`, so `module:` does not
  actually let the function live in a differently-named file. Don't ask a PR to rely on it, and
  treat a PR that fixes it as a framework change needing its own review.)
- `scripts/` sub-directories are all appended to `sys.path`, so **module basenames share one
  flat namespace**. A new file must not collide with an existing script or a stdlib/third-party
  module name.
- Adding a script is not enough to run it — flag PRs that add a script but never list it in
  `config_cam_baseline_example.yaml` (commented-out with a note is the established pattern for
  opt-in scripts), and PRs that enable an expensive or site-specific script by default.

### 4.2 Data access

- Read data through `AdfData` (`adfobj.data.load_climo_da`, `load_regrid_da`,
  `load_reference_regrid_da`, `get_*_file`, …), not by hand-rolling `glob` + `xr.open_dataset`
  over the output directories. Hand-rolled paths silently miss the multi-`hist_str`,
  derived-variable, and obs-vs-baseline cases the class already handles.
- Reject **new hard-coded paths** (`/glade/...`, someone's scratch directory). Several existing
  scripts do this (`ENSO_acrossRuns.py`, `MOPITT.py`, `ozone_diagnostics.py`,
  `enso_comparison_plots.py`, `global_mean_timeseries.py`) — that is debt to be contained, not
  precedent. New data locations belong in the config file or `adf_variable_defaults.yaml`
  (`obs_file` + `obs_data_loc`).
- Missing input files are normal (a variable absent from a run, obs not staged). The expected
  behavior is a `warnings.warn`/print plus `continue` to the next variable — never a traceback
  that kills the whole run, and never a silently blank plot.

### 4.3 Config file and defaults

- Any new config key needs: a default or an explicit `required=False` read via
  `read_config_var()`/`get_basic_info()`, an entry with an explanatory comment in
  `config_cam_baseline_example.yaml`, and **backwards compatibility** — existing user configs
  in the wild must keep working. A PR that makes an old config fail needs that called out
  loudly in the PR description.
- Renaming or removing a key, a variable-defaults field, or a `adfobj` property/method is a
  breaking change to user configs and to every contributed script. Grep the whole repo
  (`scripts/` included) before agreeing it is safe.
- New `adf_variable_defaults.yaml` entries: keep the alphabetical/grouped placement of the
  file, document any new *field* in the header comment block, and respect the documented
  mutual exclusions (`contour_levels` vs `contour_levels_range`, `diff_contour_levels` vs
  `diff_contour_range`). Check whether the change also belongs in
  `adf_variable_defaults_era5-1deg.yaml`.
- YAML uses `${var}` self-references expanded by `AdfConfig.expand_references()`; verify new
  references point at keys that exist and are set.

### 4.4 Website output

- Plots/tables reach the website only via
  `adfobj.add_website_data(web_data, web_name, case_name, category=…, season=…,
  non_season=…, plot_type=…, multi_case=…)`. A new plot with no such call produces a file no
  one ever sees.
- `add_website_data` no-ops when `create_html` is false — so website changes must not be the
  only place a code path is exercised.

### 4.5 Errors, logging, parallelism

- Routines under the `scripts` directory should avoid raising an exception (or calling `end_diag_fail`)
  whenever possible, so if the code can likely continue running, please flag the line during review.
- Framework code fails via `self.end_diag_fail(msg)` (raises `AdfError`, no traceback spam),
  not `sys.exit()` or a bare `raise`. Diagnostic scripts should warn-and-skip.
- Debug output goes through `adfobj.debug_log(...)`, not stray `print()` in library code.
- Time-series creation (`lib/adf_diag.py`) and climatology creation
  (`scripts/averaging/create_climo_files.py`) use `multiprocessing.Pool` sized by
  `adfobj.num_procs`. Anything new inside those paths must be picklable and must not share
  mutable state or write to one file from several workers.
- Keep the `try: import x / except ImportError: print(...); sys.exit(1)` idiom in the modules
  that already use it when adding a third-party dependency — and check whether that dependency
  is in `env/conda_environment.yaml`. Unpinned or newly-added dependencies deserve a comment.

## 5. Review priorities

In rough order of how often these actually bite in this codebase:

1. **Interface breakage** — the §4.1 contract, a renamed `adfobj` property, a changed helper
   signature in `lib/plotting_functions.py` used by scripts the author did not open.
2. **Silent wrongness in the science** — unit/scale handling (`scale_factor`, `add_offset`,
   `new_unit`), weighted vs. unweighted spatial means (cos(lat) weighting), seasonal averaging
   without month-length weighting, missing-value/NaN handling, masking (`mask: ocean`),
   vertical-level and time-bounds selection, off-by-one in year ranges (cf. 9adbccb),
   hemisphere/longitude conventions (0–360 vs. −180–180) and central-longitude handling.
   These produce a plausible-looking plot that is wrong, so they matter more than style.
3. **Case/reference asymmetry** — logic that works for model-vs-baseline but not
   model-vs-obs (`adfobj.compare_obs`), for one case but not `num_cases > 1`, or that ignores
   multiple history streams (`hist_string`).
4. **Robustness across runs** — missing variables, missing obs, single-year runs, variables
   with no defaults entry, derived variables (`derivable_from`).
5. **Performance** — needless `.load()`/`.compute()` of full fields, per-variable re-reads
   inside loops, unclosed datasets in long loops. The ADF runs on decades of CAM output.
6. **Style/consistency** — last, and only as far as §6.

Do not require the PR to fix pre-existing problems in the file it touches. Note them as
optional/follow-up (an issue is often the right home) and keep them clearly separated from
blocking findings.

## 6. Style conventions

Two of the rules below — `black` formatting and Sphinx-ready docstrings — are **conventions for
new code, not yet enforced by CI**. Neither `black` nor `sphinx` is in
`env/conda_environment.yaml` or `.pre-commit-config.yaml`, and there is no `docs/` tree or
`conf.py` in the repo. So: hold new code to them, and do not fail a PR because the
existing file around it does not comply.

### 6.1 Formatting: `black` on new code

All code **added or rewritten** by a PR should be `black`-formatted (default settings, 88-column
target). The existing tree is not black-clean, and converting it is out of scope for any PR that
is not explicitly a formatting PR — `black` normalizes `#comment` to `# comment`, so running it
over a file written in the older ADF idiom (`#Comment`, `#+++++` banners, `#End if`) rewrites
essentially every line and buries the real change.

How to check without demanding a whole-file reformat:

```bash
pip install black                                    # not in the conda env
git diff main...HEAD --name-only -- '*.py' | xargs black --check --diff
```

Read the resulting diff and **only report hunks that overlap lines the PR added or changed**;
ignore the rest. To get just those lines, `darker` (black restricted to changed lines) is the
tool that does this properly:

```bash
pip install darker && darker --check --diff --revision main...HEAD .
```

Notes:
- black's 88 columns sits inside pylint's `max-line-length=100` (`lib/test/pylintrc`), so
  black-formatted code will not trip the length check. black does *not* wrap comments,
  docstrings, or long string literals, though — those can still exceed 100 and fail pylint on
  the six linted files (§3).
- Where black and the surrounding file disagree on comment style, black wins **for the new
  lines only**. Mixed style within a file during the transition is expected and acceptable.
- Keep whitespace-only churn out of the diff; it hides the real change from reviewers. If a PR
  genuinely needs to reformat a file, that belongs in a separate commit — ideally a separate PR
  — so the functional diff stays reviewable.

### 6.2 Docstrings: Sphinx/ReadTheDocs-ready

Write every docstring so it renders correctly when Sphinx autodoc + napoleon is turned on. The
established ADF style is **NumPy-style** (used consistently in `lib/plotting_functions.py`,
`lib/adf_dataset.py`, `lib/adf_utils.py`, `lib/adf_derive.py`, `scripts/plotting/zonal_mean.py`,
`scripts/plotting/global_latlon_map.py`, and others) — new docstrings must match it, not
Google style and not free prose.

Require of new or modified public functions, classes, and modules:

- **Present at all.** Module docstring at the top of every file; docstring on every public
  function, class, and method. Names starting with `_` are exempt (`no-docstring-rgx=^_`).
- **A one-line summary** first, ending in a period, then a blank line before any further text.
- **NumPy-style sections with correct underlines** — the underline of dashes must be at least as
  long as the heading, or Sphinx emits a title-underline-too-short warning and drops the
  section:
  ```
  Parameters
  ----------
  adfobj : AdfDiag
      The diagnostics object that contains all the configuration information.

  Returns
  -------
  None
      Produces plots and saves files.
  ```
- **Every parameter documented, with a type**, in `name : type` form (space before the colon)
  and the description indented beneath it. Names must match the signature — a stale or missing
  parameter name is a findable error in review. Use `Raises` for exceptions the caller should
  expect (`AdfError`), `Yields` for generators.
- **Valid reStructuredText.** The common breakages to look for:
  - Literals need double backticks (``` ``None`` ```), not markdown single backticks; code
    blocks need a `::` and an indented block, not triple backticks.
  - A `*args`/`**kwargs` or a bare `*` in prose starts emphasis in reST — escape it or wrap it
    in double backticks.
  - Docstrings containing `\n`, `\t`, LaTeX, or Windows paths must be raw strings (`r"""..."""`).
  - Lists and indented blocks need a blank line before them.
- **Keep the ADF-specific `Notes` convention** for stage scripts: `scripts/` entry points list
  which `adfobj` attributes/methods and which helper functions they use. It is the fastest way
  for a reviewer or maintainer to see a script's dependency surface — see
  `scripts/plotting/global_latlon_map.py`. Ask new plotting/analysis scripts to include it, and
  ask a PR that starts using a new `adfobj` attribute to add it to that list.
- Cross-reference other ADF objects with Sphinx roles (`` :class:`AdfDiag` ``,
  `` :meth:`add_website_data` ``, `` :func:`derive_variable` ``) rather than plain text, so the
  links work once docs are built.

There is no docs build to run against yet, so this is primarily a read-and-judge check. For a
PR with large or heavily formatted docstrings, `numpydoc`'s validator catches the mechanical
problems (missing sections, undocumented or misnamed parameters, bad summary lines):

```bash
pip install numpydoc                                       # not in the conda env
conda activate adf_v1.0.0                                  # imports need xarray, matplotlib, ...
PYTHONPATH=lib:scripts/plotting python -m numpydoc --validate adf_dataset.AdfData.load_climo_da
```

Neither `numpydoc` nor a docs build is set up in this repo, so treat the command as a
convenience, not a gate — and if it will not run, review the docstrings by eye against the
checklist above rather than skipping the check.

### 6.3 Everything else

- `snake_case` for functions/arguments/attributes, `PascalCase` for classes, `UPPER_CASE` for
  constants (enforced by pylint on the six linted files, expected everywhere).
- No bare `except:` in new code, even though existing scripts have them; catch a specific
  exception, or `except Exception` with a comment explaining why.
- Legacy idioms (`#+++++` banner blocks, `#End if` / `#End for` closers) are fine to leave in
  place and fine to keep using inside a file that already uses them. Do not ask a PR to add them
  to new files, and do not ask a PR to strip them from old ones.

## 7. How to run a review

1. Establish the diff: `git diff main...HEAD --stat`, then read the full diff. For a GitHub PR
   use `gh pr view <n>` / `gh pr diff <n>`.
2. Read the PR description and any linked issue; state up front whether the change actually
   addresses it.
3. Classify the PR — framework (`lib/`), new/modified diagnostic (`scripts/`), config/defaults,
   environment/CI, docs — and apply the matching §4 contracts.
4. For each changed file, read enough *surrounding* code to judge the change in context, and
   check the callers of anything whose signature or behavior moved.
5. Run the §3 checks that apply.
6. Verify the artifacts that CI cannot: the entry-point name, the config entry, the
   `add_website_data` call, the `adf_variable_defaults.yaml` entry, `env/conda_environment.yaml`
   if imports changed, and the docstrings or comments if any become mis-matched due to the new changes.

**Report as:**

- A two-or-three sentence summary: what the PR does, and whether it is ready.
- **Blocking** — correctness, contract, or compatibility breakage. Give `file.py:line`, why it
  is wrong, and a concrete failure scenario (which config, which variable, which case setup).
- **Non-blocking** — improvements worth making now.
- **Optional / follow-up** — pre-existing debt and nice-to-haves.
- **Not verified** — anything that needs a real ADF run on HPC data, science judgment from the
  variable's owner, or knowledge of a dataset not in the repo. Say so plainly rather than
  guessing; a plotting change usually cannot be fully validated without looking at the plot.

Be specific and cite lines. Do not pad the review to look thorough: if the PR is a clean
two-line fix, say so and stop.
