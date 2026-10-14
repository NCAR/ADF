"""Check the ADF port of the INFORM diagnostics against the original cookbook code.

See lib/test/inform_reference/README.md for what the originals are and how the
two map onto each other.  Four layers:

A. The reference: the cookbook commit pinned in
   lib/test/inform_reference/cookbook_ref.json, as committed on GitHub (see
   original_loader.py for where it is fetched from), and the files the cookbook
   never committed (lib/test/inform_reference/original/) are unchanged.
B. Source drift: every function in the ported modules is registered as the same
   as its original (checked here, ignoring docstrings, print() calls and timer
   lines), as changed and verified by a named behavioural test, as pinned
   (changed, outside the model analysis, re-review on every edit), or as ADF only.
C. Behaviour: the original and the ADF code, run on the same small real
   fixtures (test_files/inform/), must give identical results.  The dropsonde
   composite builder is checked the same way, on a few raw sondes (GLADE only).
D. ADF regression: each ADF diagnostic, run end to end on the fixtures, must
   draw the same figures and write the same dashboard JSON as the stored
   fingerprints, both from scratch and again from its caches with every CAM
   and ERA5 input replaced by an object that fails on access.
E. inform_model_analysis's lazy loading: each CAM case and ERA5 are loaded at
   most once, and not at all when the diagnostics' caches are complete.
H. Opt in (INFORM_FULL_REGRESSION=1, hours, GLADE only): C's flight-file and
   sonde-composite comparisons over every flight and every sonde.

After an intended change to the ADF output:

    INFORM_UPDATE_REFERENCE=1 pytest lib/test/unit_tests/test_inform_vs_original.py
"""

import ast
import datetime
import functools
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")
dask = pytest.importorskip("dask")
pytest.importorskip("metpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_HERE = Path(__file__).parent
_ADF_LIB_DIR = _HERE.parents[1]
_ADF_PLOT_DIR = _HERE.parents[2] / "scripts" / "plotting"
_REF_DIR = _ADF_LIB_DIR / "test" / "inform_reference"
_FIXTURES = _HERE / "test_files" / "inform"
_MANIFEST = _REF_DIR / "MANIFEST.json"
_GOLDEN = _FIXTURES / "inform_diagnostics_golden.json"

for _p in (_ADF_LIB_DIR, _ADF_PLOT_DIR, _REF_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import original_loader  # noqa: E402
from original_loader import LOCAL_ONLY_DIR, REF, CookbookUnavailable  # noqa: E402

_UPDATE = os.environ.get("INFORM_UPDATE_REFERENCE", "") not in ("", "0")

CAMPAIGNS = ["SOCRATES", "CSET"]
SLICES = {
    "SOCRATES": (slice(-62, -42), slice(135, 165)),
    "CSET": (slice(15, 45), slice(200, 240)),
}
_META = json.loads((_FIXTURES / "inform_model_fixtures.json").read_text())


def _orig_dir():
    """The pinned cookbook commit's files; skips if they cannot be fetched."""
    try:
        return original_loader.original_dir()
    except CookbookUnavailable as err:
        pytest.skip(f"cookbook {REF['commit'][:7]} unavailable: {err}")


def load_originals():
    _orig_dir()
    return original_loader.load_originals()


@pytest.fixture(autouse=True, scope="module")
def _single_threaded_dask():
    """The original load_cam_files opens files with parallel=True, and the
    sensitivity script sets xr.set_options(file_cache_maxsize=1) on import; with
    both the original and the ADF datasets reading the same fixture files, dask's
    threads keep reopening them and HDF5 fails.  The scheduler does not change
    any result, so run everything here on one thread."""
    with dask.config.set(scheduler="synchronous"):
        yield


# ==========================================================================
# A. The reference
# ==========================================================================


def test_cookbook_pin_resolves():
    """The pinned commit's files can be fetched.  Warns, without failing, when
    the pinned branch has moved on: the port then lags the cookbook, and moving
    the pin is a deliberate step (see the README)."""
    src = _orig_dir()
    missing = [f for f in REF["files"] if not (src / f).is_file()]
    assert not missing, f"commit {REF['commit']} has no {missing}"
    head = original_loader.branch_head()
    if head and head != REF["commit"]:
        import warnings

        warnings.warn(
            f"cookbook branch {REF['branch']} is now at {head[:7]}; the ADF "
            f"is checked against {REF['commit'][:7]}"
        )


def _snapshot():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(LOCAL_ONLY_DIR.iterdir())
        if p.is_file()
    }


def test_reference_snapshot_unchanged():
    """original/ (files the cookbook never committed, plus the old working-copy
    snapshot) must stay byte-for-byte what was copied."""
    current = _snapshot()
    if _UPDATE:
        manifest = json.loads(_MANIFEST.read_text()) if _MANIFEST.is_file() else {}
        manifest["files"] = current
        _MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        pytest.skip("MANIFEST.json regenerated")
    expected = json.loads(_MANIFEST.read_text())["files"]
    changed = sorted(n for n in expected if current.get(n) != expected[n])
    added = sorted(set(current) - set(expected))
    assert not changed and not added, (
        f"lib/test/inform_reference/original/ was modified (changed: {changed}, "
        f"added: {added}).  Those files are the reference - restore them, or, if "
        "the cookbook really moved on, follow 'Refreshing the snapshot' in the README."
    )


# ==========================================================================
# B. Source drift
# ==========================================================================

SAME = "same as the original"


class Changed:
    """Differs from the original; `test` proves the results still match."""

    def __init__(self, test, why):
        self.test, self.why = test, why


class Pinned:
    """Differs from the original and is outside the model analysis, so no
    behavioural test runs it.  `sha` pins the reviewed version: any edit fails
    until it is re-reviewed and the pin updated."""

    def __init__(self, sha, why):
        self.sha, self.why = sha, why


class AdfOnly:
    """No original.  `covered_by` names what exercises it."""

    def __init__(self, covered_by):
        self.covered_by = covered_by


class From:
    """Ported from a different original module than the ADF module it now lives
    in; `status` (SAME, Changed or Pinned) is checked against that module."""

    def __init__(self, module, status):
        self.module, self.status = module, status


_NOT_IN_ANALYSIS = (
    "config lookup rewritten for AdfInfo.campaigns_dict; builds flight "
    "tables from raw aircraft data, which the ADF model analysis never does"
)

REGISTRY = {
    "cam_era5_functions": {
        "load_cam_files": Changed(
            "test_load_cam_files_matches_original",
            "open_mfdataset without parallel=True",
        ),
        "load_era5_files": SAME,
        "load_cam_ccfs": SAME,
        "compute_cam_aerosol_micphys_metrics": SAME,
        "subset_cam_by_campaign": SAME,
        "collocate_cam_points_only": SAME,
    },
    "cam_diagnostic": {
        "disk_size": AdfOnly("logging only"),
        "safe_pickle_load": AdfOnly(
            "test_diagnostics_match_golden_and_rerun_from_cache"
        ),
        "atomic_pickle_dump": AdfOnly(
            "test_diagnostics_match_golden_and_rerun_from_cache"
        ),
        "inform_cache_root": AdfOnly("test_inform_cache_root_choices"),
        "_canonical": AdfOnly(
            "test_cached_source_recomputes_only_when_fingerprint_changes"
        ),
        "case_cache_source": AdfOnly("test_cache_shared_across_configs_by_case_name"),
        "obs_cache_source": AdfOnly("test_cache_shared_across_configs_by_case_name"),
        "sonde_cache_source": AdfOnly("test_cache_shared_across_configs_by_case_name"),
        "cam_case_cache_path": AdfOnly("test_load_cam_case_saves_then_reuses_cam_data"),
        "cam_case_fingerprint": AdfOnly(
            "test_load_cam_case_saves_then_reuses_cam_data"
        ),
        "read_cam_case_cache": AdfOnly(
            "test_saved_cam_data_rebuilds_the_regime_composites"
        ),
        "write_cam_case_cache": AdfOnly(
            "test_saved_cam_data_rebuilds_the_regime_composites"
        ),
        "comp_from_masks": AdfOnly(
            "test_saved_cam_data_rebuilds_the_regime_composites"
        ),
        "collocate_profiles_for_regime": AdfOnly(
            "test_macro_matches_original (compute_case_stats is built on it)"
        ),
        "cached_source": AdfOnly(
            "test_cached_source_recomputes_only_when_fingerprint_changes"
        ),
        "_autodetect_pressure_col": SAME,
        "_as_percent": SAME,
        "sonde_profile": SAME,
        "_get_nearest_time_index": SAME,
        "get_model_column_raw": SAME,
        "collocate_model_profiles_for_regime": SAME,
        "mean_std_from_profiles": SAME,
        "plot_three_panels_profiles_with_era": Pinned(
            "c406c487858ac0f6",
            "ERA5 3-panel figure; its call is commented out in the ADF",
        ),
        "dropsonde_cam_obs_2x2": AdfOnly("unused"),
        "plot_three_panels_cam_obs_delta_cam_minus_obs": Changed(
            "test_macro_matches_original",
            "takes the cached per-case profile dict; draws every case and a 4th panel",
        ),
        "find_obs_nd_lwc_columns": AdfOnly("test_micro_matches_original"),
        "compute_pdf_1d": AdfOnly("test_micro_matches_original"),
        "compute_pdf_dask": AdfOnly("test_micro_matches_original"),
        "plot_obs_cam_nd_lwc_pdfs": Changed(
            "test_micro_matches_original",
            "several CAM cases, 2x2 layout, optional precomputed PDFs",
        ),
        "_linregress_from_moments": SAME,
        "weighted_linear_fit": SAME,
        "r2_of_line": SAME,
        "cam_binned_alpha_vs_sigmaw": SAME,
        "slope_logNd_logCCN_vs_sigmaw": SAME,
        "plot_sensitivity_vs_sigmaw_2x2_obs_cam": Changed(
            "test_sensitivity_matches_original",
            "plots precomputed per-source tables (computed by the ADF script)",
        ),
        "plot_alpha_sigmaw_regime_comparison_2x2": AdfOnly(
            "test_diagnostics_match_golden_and_rerun_from_cache"
        ),
    },
    "inform_utils": {
        "_campaign_cfg_for": AdfOnly("not in the model analysis"),
        "find_flight_fnames": Pinned("9bd4a1c10ced9374", _NOT_IN_ANALYSIS),
        "find_nc_fnames": SAME,
        "open_nc": SAME,
        "read_flight_nc_1hz": SAME,
        "read_flight_nc_25hz": SAME,
        "read_flight_nc": SAME,
        "read_vars": SAME,
        "read_sizedist_vars": SAME,
        "_prep_probe": SAME,
        "_sum_range_by_upper_edge": SAME,
        "calc_concs_from_sd": SAME,
        "load_flight_data": Pinned("c0be0765fd53ca87", _NOT_IN_ANALYSIS),
        "find_sondes": SAME,
        "_assign_rf_cset": SAME,
        "_infer_rf_from_name": SAME,
        "_sort_key_default": SAME,
        "_read_cls_multi": SAME,
        "_read_eol_one": SAME,
        "_read_nc_one": SAME,
        "read_sonde": SAME,
        "load_nc_cldrgme": SAME,
        "_parse_first_float": SAME,
        "_read_header": SAME,
        "_extract_date_from_header": SAME,
        "_extract_const_ss_percent": SAME,
        "_extract_rf_from_path": SAME,
        "_find_col": SAME,
        "load_ccn_file": SAME,
        "load_all_ccn": SAME,
        "split_ccn_streams": SAME,
        "merge_two_ccn_streams_into_aircraft": SAME,
        "load_ccn_for_campaign": Pinned("1a69210bf050b385", _NOT_IN_ANALYSIS),
        "select_ERA5_4flight": From(
            "process_data_products_utils",
            Changed(
                "test_sonde_composite_matches_original",
                "uses this module's datetime/timedelta names instead of datetime.datetime",
            ),
        ),
        "wrap180": From("process_data_products_utils", SAME),
        "nearest_time_indices": From("process_data_products_utils", SAME),
        "collocate_ERA5_sonde": From("process_data_products_utils", SAME),
        "cloud_regime_sonde": From("process_data_products_utils", SAME),
        "sonde_composite_dataset": AdfOnly("test_sonde_composite_matches_original"),
        "build_sonde_composite": AdfOnly("test_sonde_composite_matches_original"),
        "VAP_process_flight_data": From("process_data_products_utils", SAME),
        "assign_flight_type": From("process_data_products_utils", SAME),
        "block_flight": From("process_data_products_utils", SAME),
        "collocate_ERA5_dat": From("process_data_products_utils", SAME),
        "cloud_regime": From("process_data_products_utils", SAME),
        "write_RF_nc": From(
            "process_data_products_utils",
            Changed(
                "test_flight_file_matches_original",
                "this module's datetime name; optional out_dir (default: working directory)",
            ),
        ),
        "build_flight_cloud_regime_files": AdfOnly("test_flight_file_matches_original"),
    },
}


def _is_timer(call):
    """time.perf_counter() or <anything>.timer(...) / timer(...)."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr == "timer" or (
            f.attr == "perf_counter" and getattr(f.value, "id", None) == "time"
        )
    return isinstance(f, ast.Name) and f.id == "timer"


class _StripLogging(ast.NodeTransformer):
    """Remove statements that cannot change a result: docstrings and other bare
    strings (commented-out code), print() calls, and timing lines."""

    @staticmethod
    def _inert(stmt):
        if isinstance(stmt, ast.Expr):
            v = stmt.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                return True
            if isinstance(v, ast.Call) and (
                _is_timer(v) or (isinstance(v.func, ast.Name) and v.func.id == "print")
            ):
                return True
        return (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and _is_timer(stmt.value)
        )

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            if isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt):
                setattr(
                    node,
                    field,
                    [s for s in stmts if not self._inert(s)] or [ast.Pass()],
                )
        return node


def _functions(path):
    tree = ast.parse(Path(path).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _normalized(node):
    tree = _StripLogging().visit(ast.parse(ast.unparse(node)))
    # Python 3.12 added an (empty unless generic) "type_params" field to
    # function and class definitions; drop it so the same code gives the same
    # fingerprint on every Python (ast.dump skips missing fields):
    for n in ast.walk(tree):
        if getattr(n, "type_params", None) == []:
            del n.type_params
    return ast.dump(tree)


def _sha(node):
    return hashlib.sha256(_normalized(node).encode()).hexdigest()[:16]


@pytest.mark.parametrize("module", sorted(REGISTRY))
def test_ported_functions_match_registry(module):
    adf = _functions(_ADF_LIB_DIR / f"{module}.py")
    orig = _functions(_orig_dir() / f"{module}.py")
    registry = REGISTRY[module]
    this_module = sys.modules[__name__]
    problems = []

    for name in sorted(set(adf) - set(registry)):
        problems.append(
            f"{name}: new function - register it in REGISTRY "
            f"({'SAME/Changed/Pinned' if name in orig else 'AdfOnly'})."
        )
    for name in sorted(set(registry) - set(adf)):
        problems.append(f"{name}: registered but no longer in lib/{module}.py.")

    for name, status in registry.items():
        if name not in adf:
            continue
        if isinstance(status, AdfOnly):
            if name in orig:
                problems.append(
                    f"{name}: registered as AdfOnly but the original has it."
                )
            continue
        source = orig
        if isinstance(status, From):
            source = _functions(_orig_dir() / f"{status.module}.py")
            status = status.status
        if name not in source:
            problems.append(
                f"{name}: the original has no {name}; register it as AdfOnly."
            )
            continue
        same = _normalized(adf[name]) == _normalized(source[name])
        if status == SAME and not same:
            problems.append(
                f"{name}: now differs from the original.  Revert, or register it as "
                "Changed(<behavioural test>, <why>) with a test in this file that runs "
                "both versions on the fixtures."
            )
        elif isinstance(status, Changed) and not hasattr(this_module, status.test):
            problems.append(
                f"{name}: its behavioural test {status.test} does not exist."
            )
        elif isinstance(status, Pinned) and _sha(adf[name]) != status.sha:
            problems.append(
                f"{name}: changed since it was last reviewed ({status.why}).  Re-review "
                f"against the original, then update its pin to {_sha(adf[name])!r}."
            )
        elif isinstance(status, (Changed, Pinned)) and same:
            problems.append(f"{name}: matches the original again; register it as SAME.")

    assert not problems, f"lib/{module}.py:\n  " + "\n  ".join(problems)


# ==========================================================================
# C. Behaviour: original vs ADF on the same fixtures
# ==========================================================================


def _window(campaign):
    m = _META[campaign]
    return (
        datetime.datetime.fromisoformat(m["tmin"]),
        datetime.datetime.fromisoformat(m["tmax"]),
    )


@functools.lru_cache(maxsize=None)
def _pipeline(campaign, side):
    """INFORM_cam_assessment_diagnostics.ipynb cells 4-8 on the fixtures, with
    the original ("orig") or the ADF ("adf") modules.  Cached: several tests share it.  Nothing is
    .load()ed: as in a real run, the CAM data stay dask-backed, which some of
    the functions under test rely on."""
    if side == "orig":
        o = load_originals()
        cfun, inform = o["cam_era5_functions"], o["inform_utils"]
    else:
        import cam_era5_functions as cfun
        import inform_utils as inform
    tmin, tmax = _window(campaign)
    lat_slice, lon_slice = SLICES[campaign]
    ds_cam = cfun.load_cam_files(
        _FIXTURES / f"{campaign}_cam", "cam.h0", tmin, tmax, campaign=campaign
    )
    ds_out = cfun.compute_cam_aerosol_micphys_metrics(
        ds_cam, lat_slice=lat_slice, lon_slice=lon_slice
    )
    ccfs = cfun.load_cam_ccfs(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice)
    comp = cfun.subset_cam_by_campaign(ds_out, ccfs, campaign=campaign)
    air = inform.load_nc_cldrgme([str(_FIXTURES / _META[campaign]["incloud_fixture"])])
    return dict(ds_cam=ds_cam, ds_out=ds_out, ccfs=ccfs, comp=comp, air=air)


def _sondes():
    return {
        c: xr.open_dataset(_FIXTURES / f"{c}_sonde_fixture.nc").to_dataframe()
        for c in CAMPAIGNS
    }


def _era5(campaign):
    return xr.open_dataset(_FIXTURES / f"{campaign}_era5_fixture.nc").load()


def _assert_arrays_equal(new, ref, what):
    new, ref = np.asarray(new), np.asarray(ref)
    assert new.shape == ref.shape, f"{what}: shape {new.shape} vs original {ref.shape}"
    assert np.array_equal(new, ref, equal_nan=True), (
        f"{what}: differs from the original, max |diff| "
        f"{np.nanmax(np.abs(new.astype(float) - ref.astype(float)))}"
    )


def _curves(fig):
    """Every drawn line, polygon and line collection in `fig`, as exact bytes."""
    out = set()
    for ax in fig.axes:
        for line in ax.get_lines():
            xy = np.column_stack(
                [
                    np.asarray(line.get_xdata(), float),
                    np.asarray(line.get_ydata(), float),
                ]
            )
            out.add(xy.tobytes())
        for coll in ax.collections:
            if hasattr(coll, "get_segments"):
                out.update(np.asarray(s, float).tobytes() for s in coll.get_segments())
            else:
                out.update(
                    np.asarray(p.vertices, float).tobytes() for p in coll.get_paths()
                )
    return out


def _assert_draws_original_curves(adf_fig, orig_fig, what):
    missing = _curves(orig_fig) - _curves(adf_fig)
    assert (
        not missing
    ), f"{what}: {len(missing)} curve(s) the original draws are not in the ADF figure"


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_load_nc_cldrgme_matches_original(campaign):
    """Both ADF copies (lib/inform_utils and inform_model_analysis) vs the original."""
    import inform_utils
    import inform_model_analysis

    files = [str(_FIXTURES / _META[campaign]["incloud_fixture"])]
    files += sorted(
        str(p) for p in _FIXTURES.glob(f"{campaign}_RF[0-9][0-9]_fixture.nc")
    )
    ref = load_originals()["inform_utils"].load_nc_cldrgme(files)
    pd.testing.assert_frame_equal(
        inform_utils.load_nc_cldrgme(files), ref, check_exact=True
    )
    pd.testing.assert_frame_equal(
        inform_model_analysis.load_nc_cldrgme(files), ref, check_exact=True
    )


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_load_cam_files_matches_original(campaign):
    xr.testing.assert_identical(
        _pipeline(campaign, "adf")["ds_cam"], _pipeline(campaign, "orig")["ds_cam"]
    )


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_cam_chain_matches_original(campaign):
    """compute_cam_aerosol_micphys_metrics, load_cam_ccfs, subset_cam_by_campaign."""
    adf, orig = _pipeline(campaign, "adf"), _pipeline(campaign, "orig")
    xr.testing.assert_identical(adf["ds_out"], orig["ds_out"])
    xr.testing.assert_identical(adf["ccfs"], orig["ccfs"])
    for regime in ("strat", "open"):
        assert (
            adf["comp"][regime].sizes.get("time", 0) > 0
        ), f"no CAM {regime} points in the fixture"
        xr.testing.assert_identical(adf["comp"][regime], orig["comp"][regime])
        xr.testing.assert_identical(
            adf["comp"]["masks"][regime], orig["comp"]["masks"][regime]
        )


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_micro_matches_original(campaign):
    """Nd/LWC PDFs: the ADF's cached stats and its plot vs notebook cell 13."""
    from model_analysis_micro_scripts.cam_obs_nd_lwc_pdf import (
        compute_case_pdf_stats,
        compute_obs_pdf_stats,
    )
    import cam_diagnostic as cdog

    adf, orig = _pipeline(campaign, "adf"), _pipeline(campaign, "orig")

    orig_fig, _, ref = load_originals()["cam_diagnostic"].plot_obs_cam_nd_lwc_pdfs(
        All_rf_df=orig["air"], ds_cam_comp=orig["comp"], savepath=None, show=False
    )
    obs = compute_obs_pdf_stats(adf["air"])
    cam = compute_case_pdf_stats(adf["comp"])
    adf_fig, _, plotted = cdog.plot_obs_cam_nd_lwc_pdfs(
        All_rf_df=adf["air"], cam_cases={"CAM6": adf["comp"]}
    )

    for regime, key in (("Open-Cell", "open"), ("Stratocumulus", "strat")):
        for var in ("Nd", "LWC"):
            _assert_arrays_equal(
                obs[f"{var}_{key}"], ref[f"{var}_obs"][regime], f"obs {var} {regime}"
            )
            _assert_arrays_equal(
                cam[f"{var}_{key}"], ref[f"{var}_cam"][regime], f"CAM {var} {regime}"
            )
            _assert_arrays_equal(
                plotted[f"{var}_obs"][regime],
                ref[f"{var}_obs"][regime],
                f"plotted obs {var} {regime}",
            )
            _assert_arrays_equal(
                plotted[f"{var}_cam"]["CAM6"][regime],
                ref[f"{var}_cam"][regime],
                f"plotted CAM {var} {regime}",
            )
    _assert_draws_original_curves(adf_fig, orig_fig, "Nd/LWC PDF figure")
    plt.close("all")


# Notebook cell 16, but with fewer/smaller bins: the fixtures hold one flight,
# not a campaign, and both sides get the same settings.
_SENS_OBS = dict(
    Nd_col="CONCD_RWIO",
    lwc_col="PLWCD_RWIO",
    CCN_col="Nccn_UH_CVIU",
    sigmaw_col="Sigma_w",
    regime_col="cloud_regime",
    Ndriz_col="Ndriz_2DC",
    drizzle_threshold=0.0,
)
_SENS_CAM = dict(
    cam_temp_threshold_K=268.15,
    cam_Nd_var="cam_Nc",
    cam_CCN_var="cam_N_UHSAS",
    cam_sigmaw_var="sigma_w",
    cam_rain_var="cam_Nr",
    cam_rain_threshold=0.0,
    cam_regime_map={"Stratocumulus": "strat", "Open-Cell": "open"},
)
_SENS_BINS = dict(n_bins=4, min_n=10, binning="quantile")


def test_sensitivity_matches_original():
    """alpha vs sigma_w tables (SOCRATES; CSET has no CCN/droplet columns)."""
    from model_analysis_process_scripts.sensitivity_vs_sigmaw_2x2_obs_cam import (
        compute_case_sensitivity_stats,
        compute_obs_sensitivity_stats,
    )

    adf, orig = _pipeline("SOCRATES", "adf"), _pipeline("SOCRATES", "orig")

    _, _, ref = load_originals()[
        "cam_diagnostic"
    ].plot_sensitivity_vs_sigmaw_2x2_obs_cam(
        df_obs=orig["air"],
        ds_cam_comp=orig["comp"],
        **_SENS_OBS,
        **_SENS_CAM,
        **_SENS_BINS,
    )
    obs = compute_obs_sensitivity_stats(adf["air"], **_SENS_OBS, **_SENS_BINS)
    cam = compute_case_sensitivity_stats("CAM6", adf["comp"], **_SENS_CAM, **_SENS_BINS)

    for source, new in (("OBS", obs), ("CAM", cam)):
        ref_rows = (
            ref[ref["source"] == source].drop(columns="source").reset_index(drop=True)
        )
        new_rows = new.drop(columns=["source", "case"]).reset_index(drop=True)
        assert len(ref_rows), f"no {source} rows: the fixture does not exercise this"
        # The original returns OBS and CAM concatenated, and CAM has no "source"
        # column, so the concat turns "drizzle" from bool into object.  Same values:
        assert ref_rows["drizzle"].tolist() == new_rows["drizzle"].tolist()
        ref_rows["drizzle"] = ref_rows["drizzle"].astype(new_rows["drizzle"].dtype)
        pd.testing.assert_frame_equal(new_rows, ref_rows, check_exact=True)
    plt.close("all")


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_macro_matches_original(campaign, tmp_path):
    """Dropsonde collocation and profiles: ADF compute_case_stats vs notebook cell 11."""
    from model_analysis_macro_scripts.cam_obs_dropsonde_comp import compute_case_stats
    import cam_diagnostic as cdog

    o_cd = load_originals()["cam_diagnostic"]
    adf, orig = _pipeline(campaign, "adf"), _pipeline(campaign, "orig")
    ds_era5 = _era5(campaign)
    sonde_var, cam_var, era_var = "rh", "RH", "RH"

    # --- Notebook cell 11, verbatim apart from names ---
    df_sonde = _sondes()[campaign]
    df_sonde["tdry"] = df_sonde["tdry"] + 273.15
    df_sonde["dp"] = df_sonde["dp"] + 273.15
    ds_cam_sub = orig["ds_out"][[cam_var]].chunk({"time": -1, "lev": -1}).persist()
    ds_era = ds_era5[[era_var]].chunk({"time": -1, "level": -1}).persist()
    kw = dict(
        sonde_var=sonde_var,
        cam_var=cam_var,
        era_var=era_var,
        cam_lat_name="lat",
        cam_lon_name="lon",
        cam_lev_name="lev",
        era_lat_name="latitude",
        era_lon_name="longitude",
        era_lev_name="level",
        to_percent_models=True,
    )
    ref = {}
    profs = {}
    for key, regime in (("open", "Open-Cell"), ("strat", "Stratocumulus")):
        df_reg, p_cam_raw, cam_arr, p_era_raw, era_arr = (
            o_cd.collocate_model_profiles_for_regime(
                df_sonde, ds_cam_sub, ds_era, regime_name=regime, **kw
            )
        )
        prof = o_cd.sonde_profile(df_reg, sonde_var, bin_hPa=10, to_percent=True)
        profs[key] = prof
        ref[f"p_sonde_{key}"] = prof["p_hPa"].values
        ref[f"sonde_{key}_mean"] = prof["mean"].values
        ref[f"sonde_{key}_std"] = prof["std"].values
        (ref[f"p_cam_{key}"], ref[f"cam_{key}_mean"], ref[f"cam_{key}_std"]) = (
            o_cd.mean_std_from_profiles(p_cam_raw, cam_arr)
        )
        (ref[f"p_era_{key}"], ref[f"era_{key}_mean"], ref[f"era_{key}_std"]) = (
            o_cd.mean_std_from_profiles(p_era_raw, era_arr)
        )
        ref[f"N_{key}"] = df_reg.groupby(["RF", "drop_num"]).ngroups
    orig_fig = o_cd.plot_three_panels_cam_obs_delta_cam_minus_obs(
        profs["open"],
        profs["strat"],
        ref["p_cam_open"],
        ref["cam_open_mean"],
        ref["cam_open_std"],
        ref["p_cam_strat"],
        ref["cam_strat_mean"],
        ref["cam_strat_std"],
        ref["N_open"],
        ref["N_strat"],
        var_label="Relative Humidity (%)",
        ylim_hPa=(1000, 500),
        xlim_hPa=(-10, 110),
        xlim_delta=(-20, 35),
        cam_label="CAM6",
    )

    # --- ADF ---
    df_sonde = _sondes()[campaign]
    df_sonde["tdry"] = df_sonde["tdry"] + 273.15
    df_sonde["dp"] = df_sonde["dp"] + 273.15
    new = compute_case_stats(
        "CAM6", adf["ds_out"], ds_era5, df_sonde, sonde_var, cam_var, era_var
    )
    assert ref["N_open"] and ref["N_strat"], "the fixture needs sondes in both regimes"
    for key, value in ref.items():
        _assert_arrays_equal(new[key], value, key)

    adf_fig = cdog.plot_three_panels_cam_obs_delta_cam_minus_obs(
        {"CAM6": new},
        var_label="Relative Humidity (%)",
        ylim_hPa=(1000, 500),
        xlim_hPa=(-10, 110),
        xlim_delta=(-20, 35),
    )
    _assert_draws_original_curves(adf_fig, orig_fig, "dropsonde profile figure")
    plt.close("all")


_SONDE_DIRS = {
    "SOCRATES": Path("/glade/work/patnaude/SOCRATES/dropsondes"),
    "CSET": Path("/glade/work/patnaude/CSET/dropsondes"),
}
_ERA5_ROOT = Path("/glade/campaign/collections/rda/data/d633000")


def _notebook_cell_25(all_sondes, campaign, out_file):
    """INFORM_process_system_database.ipynb cell 25, verbatim apart from the
    output path (the notebook writes to its working directory)."""
    if campaign == "CSET":
        # --- copy to avoid side effects ---
        d = all_sondes.copy()

        # --- fix dtypes so netCDF can serialize ---
        d["Time"] = pd.to_datetime(d["Time"], utc=True, errors="coerce").dt.tz_convert(
            None
        )

        # RF / drop_num: real numpy ints (netcdf-safe)
        for c in ["RF", "drop_num"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype(np.int16)

        # strings: MUST be plain python str/object, NOT pandas "string"
        for c in ["cloud_regime", "drop_id", "trajectory"]:
            if c in d.columns:
                d[c] = d[c].fillna("").astype(str)

        # OPTIONAL: drop header junk if it exists
        for c in ["--", "UTC"]:
            if c in d.columns:
                d = d.drop(columns=[c])

        # OPTIONAL: rename to match SOCRATES naming
        rename_map = {
            "Press": "pres",
            "Temp": "tdry",
            "Dewpt": "dp",
            "RH": "rh",
            "Uwind": "u_wind",
            "Vwind": "v_wind",
            "Wspd": "wspd",
            "Dir": "wdir",
            "GPSAlt": "gpsalt",
            "GeoPoAlt": "alt",
        }
        d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})

        # --- make a simple RangeIndex called 'index' like SOCRATES ---
        d = d.reset_index(drop=True)
        d.index.name = "index"

        # --- convert to xarray in the SAME "flat table" layout ---
        ds = xr.Dataset.from_dataframe(d)
        # --- compression (skip strings + datetimes) ---
        encoding = {}
        for v in ds.data_vars:
            if ds[v].dtype.kind in ("U", "S", "O", "M"):  # M = datetime64
                continue
            encoding[v] = {"zlib": True, "complevel": 4}

        ds.to_netcdf(out_file, encoding=encoding)

    elif campaign == "SOCRATES":
        # SOCRATES
        ds = xr.Dataset.from_dataframe(all_sondes)
        ds.to_netcdf(out_file)


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_sonde_composite_matches_original(campaign, tmp_path):
    """<campaign>_sonde_data_composite.nc from three real raw sondes: the ADF's
    build_sonde_composite vs the notebook's sonde cells (23, 25) run with the
    original modules.  Needs the raw sondes and RDA ERA5 on GLADE; skipped
    elsewhere.  (It does not compare against the saved composites the ADF used
    before: those came from an earlier, unrecoverable version of this code.)"""
    import inform_utils

    sonde_dir = _SONDE_DIRS[campaign]
    if not sonde_dir.is_dir() or not _ERA5_ROOT.is_dir():
        pytest.skip("raw dropsondes or RDA ERA5 not available")
    # SOCRATES RF10 and CSET RF12 (2015-08-01), the fixture flights:
    pick = {"SOCRATES": "_RF10_", "CSET": "D20150801_"}[campaign]
    paths = [p for p in inform_utils.find_sondes(sonde_dir) if pick in Path(p).name][:3]
    assert (
        len(paths) == 3
    ), f"expected raw {campaign} sondes matching {pick!r} in {sonde_dir}"
    _compare_sonde_composite(campaign, paths, tmp_path)


def _compare_sonde_composite(campaign, paths, tmp_path):
    """build_sonde_composite on `paths` vs notebook cells 23 and 25 on the same
    sondes with the original modules: identical files."""
    import inform_utils

    # --- Notebook cells 23 and 25, with the original modules ---
    orig = load_originals()
    inform, pdp = orig["inform_utils"], orig["process_data_products_utils"]
    dfs, _ = inform.read_sonde(paths, campaign=campaign)
    sonde_dat = []
    for i in range(0, len(dfs)):
        ds = pdp.select_ERA5_4flight(dfs[i], campaign=campaign, dat_type="dropsonde")
        sonde_coll = pdp.collocate_ERA5_sonde(ds, dfs[i])
        regime_sonde = pdp.cloud_regime_sonde(sonde_coll, campaign=campaign)
        sonde_dat.append(regime_sonde)
    all_sondes = pd.concat(sonde_dat)
    ref_file = tmp_path / "original.nc"
    _notebook_cell_25(all_sondes, campaign, ref_file)

    # --- ADF ---
    new_file = tmp_path / "adf.nc"
    inform_utils.build_sonde_composite(campaign, paths, out_file=new_file)
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"

    with xr.open_dataset(new_file) as new, xr.open_dataset(ref_file) as ref:
        new, ref = new.load(), ref.load()
        xr.testing.assert_identical(new, ref)
        for v in ref.variables:
            assert new[v].encoding.get("zlib", False) == ref[v].encoding.get(
                "zlib", False
            ), v
        assert set(np.unique(ref["cloud_regime"].values)) - {""}, "no regime labels"


_AIRCRAFT = {c: Path(f"/glade/work/patnaude/{c}") for c in CAMPAIGNS}


@pytest.mark.parametrize("campaign, rf_num", [("SOCRATES", 11), ("CSET", 1)])
def test_flight_file_matches_original(campaign, rf_num, tmp_path, monkeypatch):
    """<CAMPAIGN>_RF<nn>_<date>.nc for one research flight (the campaign's
    smallest): the ADF's build_flight_cloud_regime_files vs the notebook's
    per-flight loop run with the original process_data_products_utils.  Both
    sides use the ADF's aircraft loaders (load_flight_data and friends are
    pinned: only their config lookup differs from the original).  Needs the raw
    aircraft data and RDA ERA5 on GLADE; skipped elsewhere.  (The saved flight
    files the ADF reads came from an earlier version of this code and are not
    reproduced: M/EIS and about a quarter of the regime labels differ.)"""
    _compare_flight_file(campaign, rf_num, tmp_path, monkeypatch)


def _compare_flight_file(campaign, rf_num, tmp_path, monkeypatch):
    """build_flight_cloud_regime_files for one flight vs the notebook's
    per-flight loop: identical files.  Like the notebook, the ADF does not
    merge CCN, even with a ccn_dir configured."""
    import types
    import inform_utils

    root = _AIRCRAFT[campaign]
    if not (root / "aircraft_1hz").is_dir() or not _ERA5_ROOT.is_dir():
        pytest.skip("raw aircraft data or RDA ERA5 not available")
    cfg = {
        "name": [campaign],
        "air_1hz_dir": [str(root / "aircraft_1hz")],
        "air_25hz_dir": [str(root / "aircraft_25hz")],
    }
    if (root / "CCN").is_dir():
        cfg["ccn_dir"] = [str(root / "CCN")]
    icf = types.SimpleNamespace(campaigns_dict=cfg)

    # --- Notebook per-flight loop (INFORM_process_system_database.ipynb cell 19),
    # original process_data_products_utils.  As committed, the loop loads the
    # CCN data but passes ccn_df=None to load_flight_data. ---
    pdp = load_originals()["process_data_products_utils"]
    ref_dir = tmp_path / "original"
    ref_dir.mkdir()
    i = rf_num - 1
    df = inform_utils.load_flight_data(campaign, idx=i, ccn_df=None, icf_obj=icf)
    if ("PLWC" in df.columns) and (campaign == "SOCRATES"):
        df = df.drop(columns=["PLWC"])
    blocks = pdp.VAP_process_flight_data(df, i)
    ds = pdp.select_ERA5_4flight(df, campaign)
    rf_id = f"RF{i+1:02d}"
    fblks_coll = pdp.collocate_ERA5_dat(ds, blocks)
    fblks_cr = pdp.cloud_regime(fblks_coll, campaign=campaign)
    monkeypatch.chdir(ref_dir)  # the original writes to the working directory
    pdp.write_RF_nc(fblks_cr, rf_id, campaign)
    monkeypatch.undo()
    (ref_file,) = ref_dir.glob(f"{campaign}_{rf_id}_*.nc")

    # --- ADF ---
    new_dir = tmp_path / "adf"
    new_dir.mkdir()
    (new_file,) = inform_utils.build_flight_cloud_regime_files(
        campaign, icf, new_dir, flights=[rf_num]
    )
    assert Path(new_file).name == ref_file.name

    with xr.open_dataset(new_file) as new, xr.open_dataset(ref_file) as ref:
        new, ref = new.load(), ref.load()
        xr.testing.assert_identical(new, ref)


# ==========================================================================
# D. ADF diagnostics end to end, from scratch and from cache
# ==========================================================================


class _FakeAdf:
    """The parts of AdfDiag the INFORM diagnostics use."""

    def __init__(self, plot_loc):
        self.plot_location = [str(plot_loc)]
        self.config_file_dict = {"name": "fixture_config"}
        self.create_html = True
        self.website = []
        self.inform_pages = {}

    def add_inform_page(self, campaign, page, json_path):
        self.inform_pages[(campaign, page)] = Path(json_path).name

    def add_website_data(self, path, name, case, **kwargs):
        self.website.append(
            [
                Path(path).name,
                name,
                case,
                {k: str(v) for k, v in sorted(kwargs.items()) if k != "plot_loc"},
            ]
        )


class _Unavailable(Mapping):
    """Stands in for model data a cached run must not touch."""

    def __init__(self, what, keys=()):
        self._what, self._keys = what, list(keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        raise AssertionError(f"cached run read {self._what}[{key!r}]")

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AssertionError(f"cached run read {self._what}.{name}")


def _fingerprint(fig):
    texts = sorted(
        t.get_text()
        for ax in fig.axes
        for t in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        if t.get_text()
    )
    legends = sorted(
        t.get_text()
        for ax in fig.axes
        if ax.get_legend()
        for t in ax.get_legend().get_texts()
    )
    h = hashlib.sha256()
    for c in sorted(_curves(fig)):
        h.update(c)
    h.update(json.dumps([texts, legends]).encode())
    return h.hexdigest()


def _run_diagnostics(
    campaign, plot_loc, ds_cams, ds_outs, ds_era5, case_info=None, campaigns=None
):
    """Call every INFORM diagnostic the way inform_model_analysis does.

    `campaigns` stands in for the config's campaigns section (e.g. to set
    inform_cache_dir); without it the caches go under plot_loc.
    """
    from model_analysis_macro_scripts import cam_obs_dropsonde_comp
    from model_analysis_micro_scripts import cam_obs_nd_lwc_pdf
    from model_analysis_process_scripts import (
        alpha_sigmaw_regime_comparison_2x2,
        sensitivity_vs_sigmaw_2x2_obs_cam,
    )

    adf = _FakeAdf(plot_loc)
    if campaigns is not None:
        adf.campaigns_dict = campaigns
    figures = {}
    real_savefig = Figure.savefig

    def record(fig, fname, *args, **kwargs):  # no PNGs: only what was drawn
        figures[Path(fname).name] = _fingerprint(fig)

    all_rf_df = _pipeline(campaign, "adf")["air"]
    df_sonde = _sondes()[campaign]
    Figure.savefig = record
    try:
        cam_obs_dropsonde_comp(adf, campaign, df_sonde, ds_outs, ds_era5, case_info)
        cam_obs_nd_lwc_pdf(adf, all_rf_df, ds_cams, campaign, case_info)
        if campaign == "SOCRATES":
            sensitivity_vs_sigmaw_2x2_obs_cam(
                adf, all_rf_df, ds_cams, campaign, case_info
            )
            alpha_sigmaw_regime_comparison_2x2(
                adf, all_rf_df, ds_cams, campaign, case_info
            )
    finally:
        Figure.savefig = real_savefig
        plt.close("all")

    dashboards = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((Path(plot_loc) / "website").glob("*.json"))
    }
    return {"figures": figures, "json": dashboards, "website": adf.website}


def _compare(new, ref, when):
    problems = []
    for kind in ("figures", "json"):
        for name in sorted(set(ref[kind]) | set(new[kind])):
            if name not in new[kind]:
                problems.append(f"{kind}: {name} no longer written")
            elif name not in ref[kind]:
                problems.append(f"{kind}: {name} is new")
            elif new[kind][name] != ref[kind][name]:
                problems.append(f"{kind}: {name} changed")
    if new["website"] != ref["website"]:
        problems.append("website registrations changed")
    assert not problems, (
        f"{when}:\n  "
        + "\n  ".join(problems)
        + ("\nIf intended, regenerate with INFORM_UPDATE_REFERENCE=1 and commit it.")
    )


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_diagnostics_match_golden_and_rerun_from_cache(campaign, tmp_path):
    """Two CAM cases, so the multi_CAM suites are exercised too."""
    adf = _pipeline(campaign, "adf")
    cases = ["caseA", "caseB"]
    ds_cams = {c: adf["comp"] for c in cases}
    ds_outs = {c: adf["ds_out"] for c in cases}

    cold = _run_diagnostics(campaign, tmp_path, ds_cams, ds_outs, _era5(campaign))
    if _UPDATE:
        golden = json.loads(_GOLDEN.read_text()) if _GOLDEN.is_file() else {}
        golden[campaign] = cold
        _GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    else:
        _compare(cold, json.loads(_GOLDEN.read_text())[campaign], "first run")

    # Same plot directory, so every stat is now cached: nothing may touch CAM/ERA5.
    warm = _run_diagnostics(
        campaign,
        tmp_path,
        _Unavailable("ds_cams", cases),
        _Unavailable("ds_outs", cases),
        _Unavailable("ds_era5"),
    )
    _compare(warm, cold, "run from cache")


# ==========================================================================
# E. inform_model_analysis loads model data only when a diagnostic needs it
# ==========================================================================


def test_lazy_cases_load_once_and_only_on_indexing():
    from inform_model_analysis import LazyCases, LazyDataset

    calls = []
    loaders = {
        c: (lambda c=c: calls.append(c) or {"comp": f"{c}-comp", "out": f"{c}-out"})
        for c in ("caseA", "caseB")
    }
    shared = {}
    comps, outs = LazyCases(loaders, "comp", shared), LazyCases(loaders, "out", shared)

    assert list(comps) == ["caseA", "caseB"] and len(outs) == 2 and "caseA" in comps
    assert "caseC" not in comps and list(comps.keys()) == ["caseA", "caseB"]
    assert not calls, "listing the cases must not load them"
    assert comps["caseA"] == "caseA-comp" and outs["caseA"] == "caseA-out"
    assert calls == ["caseA"], "one load per case, shared by both views"

    era = LazyDataset(
        lambda: calls.append("era") or xr.Dataset({"RH": ("x", [1.0, 2.0])}), "ERA5"
    )
    assert calls == ["caseA"]
    assert float(era[["RH"]]["RH"].sum()) == 3.0 and era.sizes["x"] == 2
    assert calls == ["caseA", "era"]


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_diagnostics_load_each_input_once_then_never(campaign, tmp_path):
    """The diagnostics through inform_model_analysis's lazy stand-ins."""
    from inform_model_analysis import LazyCases, LazyDataset

    adf = _pipeline(campaign, "adf")
    loads = []

    def run():
        loaders = {
            c: (
                lambda c=c: loads.append(c)
                or {"comp": adf["comp"], "out": adf["ds_out"]}
            )
            for c in ("caseA", "caseB")
        }
        shared = {}
        era5 = LazyDataset(lambda: loads.append("ERA5") or _era5(campaign), "ERA5")
        return _run_diagnostics(
            campaign,
            tmp_path,
            LazyCases(loaders, "comp", shared),
            LazyCases(loaders, "out", shared),
            era5,
        )

    first = run()
    assert sorted(loads) == ["ERA5", "caseA", "caseB"], f"first run loaded {loads}"
    _compare(
        first, json.loads(_GOLDEN.read_text())[campaign], "first run, lazily loaded"
    )

    loads.clear()
    run()
    assert loads == [], f"run from cache still loaded {loads}"


# ==========================================================================
# F. The per-source result cache is keyed by case name and fingerprint
# ==========================================================================


def _case_info(nicknames, case_names, tmax="2018-02-24 00:00:00"):
    return {
        nick: {
            "case_name": name,
            "hist_str": "cam.h0i",
            "cam_hist_loc": f"/archive/{name}",
            "tmin": "2018-01-15 00:00:00",
            "tmax": tmax,
            "lat_slice": slice(-62, -42),
            "lon_slice": slice(135, 165),
        }
        for nick, name in zip(nicknames, case_names)
    }


def test_cached_source_recomputes_only_when_fingerprint_changes(tmp_path):
    import cam_diagnostic as cdog

    calls = []

    def get(settings, version=1, key="run.001"):
        return cdog.cached_source(
            tmp_path,
            "diag",
            "SOCRATES",
            key,
            {"case_name": key},
            settings,
            version,
            compute=lambda: calls.append(1) or len(calls),
        )

    assert get({"bins": [1, 2]}) == 1
    assert get({"bins": [1, 2]}) == 1 and len(calls) == 1, "same fingerprint must hit"
    assert get({"bins": [1, 3]}) == 2, "changed settings must recompute"
    assert get({"bins": [1, 3]}, version=2) == 3, "a new CACHE_VERSION must recompute"
    assert (
        get({"bins": [1, 3]}, version=2, key="run.002") == 4
    ), "another case is its own entry"
    assert sorted(p.name for p in (tmp_path / "diag").iterdir()) == [
        "SOCRATES__run.001.pkl",
        "SOCRATES__run.002.pkl",
    ]
    (tmp_path / "diag" / "SOCRATES__run.001.pkl").write_bytes(b"truncated")
    assert get({"bins": [1, 3]}, version=2) == 5, "a corrupt cache file is recomputed"


def test_inform_cache_root_choices(tmp_path):
    import cam_diagnostic as cdog

    class _Adf:
        plot_location = [str(tmp_path / "plots")]
        campaigns_dict = {
            "name": ["SOCRATES", "CSET"],
            "composited_data": [str(tmp_path / "s"), str(tmp_path / "c")],
        }

    adf = _Adf()
    assert cdog.inform_cache_root(adf, "CSET") == tmp_path / "c" / "inform_cache"
    adf.campaigns_dict = {
        **_Adf.campaigns_dict,
        "inform_cache_dir": str(tmp_path / "shared"),
    }
    assert cdog.inform_cache_root(adf, "CSET") == tmp_path / "shared"
    adf.campaigns_dict = {**_Adf.campaigns_dict, "inform_cache_dir": ["/a", "/b"]}
    assert cdog.inform_cache_root(adf, "CSET") == Path("/b")
    assert (
        cdog.inform_cache_root(_FakeAdf(tmp_path), "CSET") == tmp_path / "inform_cache"
    )


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_cache_shared_across_configs_by_case_name(campaign, tmp_path):
    """Two configs with one inform_cache_dir: the second, with different
    nicknames for the same cases, reads no model data and labels its output
    with its own nicknames; changing one case's window recomputes only it."""
    from inform_model_analysis import LazyCases, LazyDataset

    adf = _pipeline(campaign, "adf")
    campaigns = {"name": [campaign], "inform_cache_dir": str(tmp_path / "shared_cache")}
    case_names = ["f.e30.run.001", "f.e30.run.002"]
    loads = []

    def run(plot_dir, nicknames, case_info):
        loaders = {
            c: (
                lambda c=c: loads.append(c)
                or {"comp": adf["comp"], "out": adf["ds_out"]}
            )
            for c in nicknames
        }
        shared = {}
        era5 = LazyDataset(lambda: loads.append("ERA5") or _era5(campaign), "ERA5")
        return _run_diagnostics(
            campaign,
            tmp_path / plot_dir,
            LazyCases(loaders, "comp", shared),
            LazyCases(loaders, "out", shared),
            era5,
            case_info,
            campaigns,
        )

    run("config_6hr", ["caseA", "caseB"], _case_info(["caseA", "caseB"], case_names))
    assert sorted(loads) == ["ERA5", "caseA", "caseB"]
    cache = tmp_path / "shared_cache" / "dropsonde"
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        [f"{campaign}__Obs_ERA5.pkl"] + [f"{campaign}__{n}.pkl" for n in case_names]
    ), "dropsonde caches Obs/ERA5 once, plus one CAM file per case"

    loads.clear()
    renamed = run(
        "config_12hr", ["nickX", "nickY"], _case_info(["nickX", "nickY"], case_names)
    )
    assert (
        loads == []
    ), f"second config recomputed {loads} instead of reusing the shared cache"
    combined = json.loads(
        (
            tmp_path
            / "config_12hr"
            / "website"
            / f"nd_lwc_pdfs_{campaign}_fixture_config.json"
        ).read_text()
    )
    assert set(combined["cases"]) == {
        "Obs",
        "nickX",
        "nickY",
    }, "output must use this config's nicknames"
    assert any("nickX" in name for name in renamed["figures"])

    loads.clear()
    changed = _case_info(["nickX", "nickY"], case_names)
    changed["nickY"]["tmax"] = "2018-02-25 00:00:00"
    run("config_12hr", ["nickX", "nickY"], changed)
    assert loads == [
        "nickY"
    ], f"only the changed case should reload (not ERA5), got {loads}"


# ==========================================================================
# G. Each case's derived CAM data is saved once and reused
# ==========================================================================


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_saved_cam_data_rebuilds_the_regime_composites(campaign, tmp_path):
    """What the diagnostics get from a saved case is exactly what
    compute_cam_aerosol_micphys_metrics + subset_cam_by_campaign give."""
    import cam_diagnostic as cdog

    adf = _pipeline(campaign, "adf")
    fields = list(cdog.CAM_CACHE_VARS)
    fingerprint = cdog.cam_case_fingerprint({"case_name": "f.e30.run.001"})
    path = cdog.cam_case_cache_path(tmp_path, campaign, "f.e30.run.001")

    cdog.write_cam_case_cache(path, adf["ds_out"], adf["comp"]["masks"], fingerprint)
    assert not list(path.parent.glob("*.tmp*")), "temporary file left behind"
    ds_out, masks = cdog.read_cam_case_cache(path, fingerprint)
    comp = cdog.comp_from_masks(ds_out, masks)

    ref_out = adf["ds_out"][fields].load()
    new_out = ds_out[fields].load()
    for v in fields:
        assert new_out[v].identical(ref_out[v]), f"saved ds_out[{v}] differs"
    for regime in ("strat", "open"):
        ref, new = adf["comp"][regime][fields].load(), comp[regime][fields].load()
        assert dict(new.sizes) == dict(
            ref.sizes
        ), f"{regime} composite has different dims"
        for v in fields:
            assert new[v].identical(ref[v]), f"rebuilt {regime} composite {v} differs"
        assert comp["masks"][regime].load().equals(adf["comp"]["masks"][regime].load())

    other = cdog.cam_case_fingerprint({"case_name": "f.e30.run.001", "tmax": "later"})
    assert (
        cdog.read_cam_case_cache(path, other) is None
    ), "a different fingerprint must not match"
    path.write_bytes(b"not a netcdf file")
    assert (
        cdog.read_cam_case_cache(path, fingerprint) is None
    ), "an unreadable file must not match"


@pytest.mark.parametrize("campaign", CAMPAIGNS[:1])
def test_load_cam_case_saves_then_reuses_cam_data(campaign, tmp_path, monkeypatch):
    """The first load reads the history files and saves the case; the next
    reads only the saved file; a changed fingerprint reads the history again."""
    import cam_diagnostic as cdog
    import inform_model_analysis as ima

    adf = _pipeline(campaign, "adf")
    reads = []
    monkeypatch.setattr(
        ima.cfun,
        "load_cam_files",
        lambda *a, **k: reads.append("history") or adf["ds_cam"],
    )
    monkeypatch.setattr(
        ima.cfun, "compute_cam_aerosol_micphys_metrics", lambda *a, **k: adf["ds_out"]
    )
    monkeypatch.setattr(ima.cfun, "load_cam_ccfs", lambda *a, **k: adf["ccfs"])

    tmin, tmax = _window(campaign)
    lat_slice, lon_slice = SLICES[campaign]
    source = {"case_name": "f.e30.run.001", "tmin": tmin, "tmax": tmax}
    path = cdog.cam_case_cache_path(tmp_path, campaign, source["case_name"])

    def load(src):
        return ima.load_cam_case(
            Path("unused"),
            "cam.h0i",
            campaign,
            tmin,
            tmax,
            lat_slice,
            lon_slice,
            cache_path=path,
            cache_fingerprint=cdog.cam_case_fingerprint(src),
        )

    first = load(source)
    assert reads == ["history"] and path.is_file()
    second = load(source)
    assert reads == ["history"], "the saved case must be reused"
    for regime in ("strat", "open"):
        a, b = (
            first["comp"][regime]["cam_Nc"].load(),
            second["comp"][regime]["cam_Nc"].load(),
        )
        assert a.identical(
            b
        ), "fresh and saved loads must give the diagnostics the same data"
    load({**source, "tmax": "a different window"})
    assert reads == [
        "history",
        "history",
    ], "a changed fingerprint must rebuild the saved case"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ==========================================================================
# H. Every flight and every sonde (opt in: INFORM_FULL_REGRESSION=1)
# ==========================================================================
# The tests in C use one flight and three sondes per campaign.  These repeat the
# same comparisons over each campaign's full raw data, which takes hours (each
# flight and sonde reads RDA ERA5), so they run only when asked, e.g. in a PBS
# batch job.  GLADE only.

_FULL = os.environ.get("INFORM_FULL_REGRESSION", "") not in ("", "0")
_N_FLIGHTS = {"SOCRATES": 15, "CSET": 16}
_full_only = pytest.mark.skipif(not _FULL, reason="set INFORM_FULL_REGRESSION=1")


@_full_only
@pytest.mark.parametrize(
    "campaign, rf_num", [(c, n) for c in CAMPAIGNS for n in range(1, _N_FLIGHTS[c] + 1)]
)
def test_every_flight_file_matches_original(campaign, rf_num, tmp_path, monkeypatch):
    """test_flight_file_matches_original for every research flight."""
    _compare_flight_file(campaign, rf_num, tmp_path, monkeypatch)


@_full_only
def test_flight_count():
    """_N_FLIGHTS still lists every flight in the raw aircraft data."""
    import types
    import inform_utils

    for campaign in CAMPAIGNS:
        root = _AIRCRAFT[campaign]
        if not (root / "aircraft_1hz").is_dir():
            pytest.skip("raw aircraft data not available")
        icf = types.SimpleNamespace(
            campaigns_dict={
                "name": [campaign],
                "air_1hz_dir": [str(root / "aircraft_1hz")],
            }
        )
        n = len(inform_utils.find_flight_fnames(campaign, icf_obj=icf))
        assert (
            n == _N_FLIGHTS[campaign]
        ), f"{campaign}: {n} flights, _N_FLIGHTS says {_N_FLIGHTS[campaign]}"


@_full_only
@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_full_sonde_composite_matches_original(campaign, tmp_path):
    """<campaign>_sonde_data_composite.nc from every raw sonde, as the notebook
    builds it: the ADF's build_sonde_composite vs notebook cells 23 and 25."""
    import inform_utils

    sonde_dir = _SONDE_DIRS[campaign]
    if not sonde_dir.is_dir() or not _ERA5_ROOT.is_dir():
        pytest.skip("raw dropsondes or RDA ERA5 not available")
    paths = inform_utils.find_sondes(sonde_dir)
    assert paths, f"no raw {campaign} sondes in {sonde_dir}"
    _compare_sonde_composite(campaign, paths, tmp_path)
