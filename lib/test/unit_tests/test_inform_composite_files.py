"""Bit-for-bit regression tests for the INFORM composite files.

The per-flight cloud regime files and the composites built from them are far
too large for the repository, so the checks come in two layers:

1. Fixtures (run anywhere pandas/xarray are installed):
   test_files/inform/*_RF*_fixture.nc are small real slices of the flight files
   (see make_inform_fixtures.py).  They are run through load_nc_cldrgme and the
   CSV is compared byte-for-byte with test_files/inform/*_combined_RF_fixture.csv.

2. Full data (run on GLADE):
   the full flight files are run through load_nc_cldrgme and the CSV is compared
   with a fingerprint in test_files/inform/inform_fingerprints.json: a SHA-256
   of the whole file (the bit-for-bit check), and, to explain a mismatch, the
   row count, the columns, a SHA-256 per column, a SHA-256 per input file and
   the library versions used.  Skipped wherever the inputs are not visible
   (including CI).  Point at a different copy with:

       INFORM_COMPOSITED_DATA=/path/to/composited_data pytest lib/test/unit_tests

After an intended change to the output, regenerate the expected CSVs and the
fingerprints (then commit them, so the diff shows what changed):

    INFORM_UPDATE_REFERENCE=1 pytest lib/test/unit_tests/test_inform_composite_files.py
"""
import hashlib
import io
import json
import os
import platform
import sys
from glob import glob
from pathlib import Path

import pytest

# CI installs only pyyaml and pytest (see .github/workflows/ADF_unit_tests.yaml),
# so skip rather than fail collection when the science stack is absent.
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")

_REF_DIR = Path(__file__).parent / "test_files" / "inform"
_FINGERPRINTS = _REF_DIR / "inform_fingerprints.json"
_ADF_LIB_DIR = Path(__file__).parents[2]
_ADF_PLOT_DIR = Path(__file__).parents[3] / "scripts" / "plotting"

sys.path.append(str(_ADF_LIB_DIR))
sys.path.append(str(_ADF_PLOT_DIR))

_INPUT_DIR = Path(os.environ.get(
    "INFORM_COMPOSITED_DATA",
    "/glade/u/home/patnaude/inform/composited data",
))
_UPDATE = os.environ.get("INFORM_UPDATE_REFERENCE", "") not in ("", "0")

CAMPAIGNS = ["CSET", "SOCRATES"]
#inform_model_analysis carries its own copy of load_nc_cldrgme, and it is the
#one the ADF actually runs, so both are held to the same references.  The
#references are only ever rewritten from the copy the ADF runs:
SOURCES = ["inform_utils", "inform_model_analysis"]
_WRITER = "inform_model_analysis"


def _load_nc_cldrgme(source):
    """Return the load_nc_cldrgme implementation from `source`."""
    if source == "inform_utils":
        import inform_utils
        return inform_utils.load_nc_cldrgme
    try:
        import inform_model_analysis
    except ImportError as err:
        pytest.skip(f"cannot import inform_model_analysis: {err}")
    return inform_model_analysis.load_nc_cldrgme


def _combined_csv(source, files):
    """Build the combined CSV exactly as inform_model_analysis writes it."""
    if _UPDATE and source != _WRITER:
        pytest.skip(f"references are being regenerated from {_WRITER}")
    df = _load_nc_cldrgme(source)(files)
    return df.to_csv(index=False).encode()


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_as_text(csv_bytes):
    """Every field as the text that was written, so nothing is re-parsed."""
    return pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)


def _environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xarray": xr.__version__,
    }


def _fingerprint(csv_bytes, files):
    table = _read_as_text(csv_bytes)
    return {
        "sha256": _sha256(csv_bytes),
        "rows": len(table),
        "columns": list(table.columns),
        "column_sha256": {c: _sha256("\n".join(table[c]).encode())
                          for c in table.columns},
        "inputs": {Path(f).name: _sha256(Path(f).read_bytes()) for f in files},
        "environment": _environment(),
    }


def _explain_fingerprint(new, ref):
    """Say what moved between two fingerprints."""
    msg = []

    added = sorted(set(new["inputs"]) - set(ref["inputs"]))
    removed = sorted(set(ref["inputs"]) - set(new["inputs"]))
    changed = sorted(f for f in set(new["inputs"]) & set(ref["inputs"])
                     if new["inputs"][f] != ref["inputs"][f])
    if added or removed or changed:
        msg.append("  INPUT FILES CHANGED (the difference may not be the code):")
        msg.extend(f"    added:   {f}" for f in added)
        msg.extend(f"    removed: {f}" for f in removed)
        msg.extend(f"    changed: {f}" for f in changed)

    env = {k: (ref["environment"].get(k), v)
           for k, v in new["environment"].items()
           if ref["environment"].get(k) != v}
    if env:
        msg.append("  Library versions differ from the ones that made the "
                   "fingerprint (float formatting can change with them):")
        msg.extend(f"    {k}: {old} -> {cur}" for k, (old, cur) in env.items())

    if new["columns"] != ref["columns"]:
        msg.append(f"  Columns added:   {[c for c in new['columns'] if c not in ref['columns']]}")
        msg.append(f"  Columns removed: {[c for c in ref['columns'] if c not in new['columns']]}")
        if set(new["columns"]) == set(ref["columns"]):
            msg.append("  (same columns, different order)")
    if new["rows"] != ref["rows"]:
        msg.append(f"  Rows: {new['rows']} regenerated vs {ref['rows']} in reference")

    cols = [c for c in ref["columns"] if c in new["column_sha256"]
            and new["column_sha256"][c] != ref["column_sha256"][c]]
    if cols:
        msg.append(f"  Columns whose values changed: {', '.join(cols)}")
    return msg


def _describe_csv_difference(new_bytes, ref_bytes):
    """Explain, cell by cell, where a regenerated CSV departs from the expected one."""
    new = _read_as_text(new_bytes)
    ref = _read_as_text(ref_bytes)
    msg = []

    if list(new.columns) != list(ref.columns):
        msg.append(f"  Columns added:   {[c for c in new.columns if c not in ref.columns]}")
        msg.append(f"  Columns removed: {[c for c in ref.columns if c not in new.columns]}")
        if set(new.columns) == set(ref.columns):
            msg.append("  (same columns, different order)")
    if len(new) != len(ref):
        msg.append(f"  Rows: {len(new)} regenerated vs {len(ref)} expected")

    common = [c for c in ref.columns if c in new.columns]
    if len(new) == len(ref) and common:
        diff = new[common].ne(ref[common])
        changed = diff.sum()
        changed = changed[changed > 0]
        if not changed.empty:
            msg.append("  Columns that changed (rows affected):")
            msg.extend(f"    {c}: {n}" for c, n in changed.items())
            row = int(diff.any(axis=1).to_numpy().argmax())
            msg.append(f"  First changed row (data row {row}):")
            msg.extend(f"    {c}: {ref.at[row, c]!r} -> {new.at[row, c]!r}"
                       for c in changed.index if diff.at[row, c])
    return msg


def _fail(name, details):
    pytest.fail("\n".join(
        [f"{name}: regenerated output is not bit-for-bit identical."]
        + details
        + ["If this change is intended, regenerate the references with "
           "INFORM_UPDATE_REFERENCE=1 and commit them."]), pytrace=False)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_fixture_combined_csv_is_bit_for_bit(campaign, source):
    """Small committed slices of the flight files -> expected CSV, byte for byte."""
    files = sorted(glob(f"{_REF_DIR}/{campaign}_RF[0-9][0-9]_fixture.nc"))
    assert files, f"no {campaign} fixtures in {_REF_DIR}"
    expected = _REF_DIR / f"{campaign}_combined_RF_fixture.csv"

    new_bytes = _combined_csv(source, files)

    if _UPDATE and source == _WRITER:
        expected.write_bytes(new_bytes)
        pytest.skip(f"expected output {expected.name} regenerated")
    assert expected.is_file(), f"missing expected output {expected}"

    ref_bytes = expected.read_bytes()
    if new_bytes != ref_bytes:
        _fail(expected.name, _describe_csv_difference(new_bytes, ref_bytes))


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_full_combined_csv_matches_fingerprint(campaign, source):
    """<campaign>_combined_RF_data.csv from the full flight files on GLADE."""
    if not _INPUT_DIR.is_dir():
        pytest.skip(f"INFORM input directory not found: {_INPUT_DIR}")
    files = sorted(glob(f"{_INPUT_DIR}/*{campaign}*RF*.nc"))
    if not files:
        pytest.skip(f"no {campaign} *RF*.nc files in {_INPUT_DIR}")

    new_bytes = _combined_csv(source, files)
    name = f"{campaign}_combined_RF_data.csv"

    if _UPDATE and source == _WRITER:
        prints = json.loads(_FINGERPRINTS.read_text()) if _FINGERPRINTS.is_file() else {}
        prints[name] = _fingerprint(new_bytes, files)
        _FINGERPRINTS.write_text(json.dumps(prints, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"fingerprint for {name} regenerated")

    assert _FINGERPRINTS.is_file(), f"missing fingerprints {_FINGERPRINTS}"
    ref = json.loads(_FINGERPRINTS.read_text()).get(name)
    assert ref is not None, f"no fingerprint for {name} in {_FINGERPRINTS.name}"

    #The whole-file hash is the test; the rest is only worked out to explain a miss:
    if _sha256(new_bytes) != ref["sha256"]:
        _fail(name, _explain_fingerprint(_fingerprint(new_bytes, files), ref))


@pytest.mark.parametrize("campaign", CAMPAIGNS)
def test_sonde_composite_is_bit_for_bit(campaign):
    """<campaign>_sonde_data_composite.nc

    inform_utils.build_sonde_composite now builds this file, and
    test_inform_vs_original.py::test_sonde_composite_matches_original checks it
    against the cookbook code.  It cannot be pinned to the saved composites the
    ADF used before: those came from an earlier version of that code, which is
    not recoverable, and differ in M/RH700/EIS/Tadv and some regime labels.
    Once a composite built by the ADF is adopted, fingerprint it here the same
    way as the combined CSV.
    """
    pytest.skip("no adopted reference for the regenerated sonde composite yet")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
