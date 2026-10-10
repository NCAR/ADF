"""
Check that plot-level settings in adf_variable_defaults.yaml load as numbers.

YAML 1.1 (what PyYAML implements) does not accept a bare `1e7` as a float -- it
needs `1.0e+7` -- so an entry written as `contour_levels: [-5e7,0,5e7]` loads as
a list of *strings* with one int in the middle.  That parses fine and validates
fine, and then blows up much later inside the plotting code with

    TypeError: '<' not supported between instances of 'int' and 'str'

Only PyYAML is needed here, so this runs in CI rather than skipping.
"""

from pathlib import Path

import pytest
import yaml

# Keys whose values feed straight into numeric plotting calls.
NUMERIC_KEYS = (
    "contour_levels",
    "contour_levels_range",
    "diff_contour_range",
    "pct_diff_contour_levels",
    "pct_diff_contour_range",
    "contour_adjust",
    "scale_factor",
    "add_offset",
    "obs_scale_factor",
    "obs_add_offset",
)

DEFAULTS_FILE = Path(__file__).parents[3] / "lib" / "adf_variable_defaults.yaml"


def _numeric_offenders(defaults):
    """Return [(variable, key, value), ...] for settings that are not numbers."""
    bad = []
    for var, settings in defaults.items():
        if not isinstance(settings, dict):
            continue
        for key in NUMERIC_KEYS:
            if key not in settings:
                continue
            value = settings[key]
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, (int, float)) for v in values):
                bad.append((var, key, value))
    return bad


def test_defaults_file_exists():
    assert DEFAULTS_FILE.is_file(), f"missing {DEFAULTS_FILE}"


def test_numeric_settings_parse_as_numbers():
    defaults = yaml.safe_load(DEFAULTS_FILE.read_text())
    offenders = _numeric_offenders(defaults)
    assert not offenders, (
        "These settings did not parse as numbers -- most likely written as "
        "`1e7` instead of `1.0e+7` or a plain decimal:\n"
        + "\n".join(f"  {var}: {key}: {val!r}" for var, key, val in offenders)
    )


@pytest.mark.parametrize(
    "text,expected_numeric",
    [
        ("V:\n  contour_levels: [-5e7,0,5e7]\n", False),  # the bug
        ("V:\n  contour_levels: [-5.0e+7,0,5.0e+7]\n", True),
        ("V:\n  contour_levels: [-50000000,0,50000000]\n", True),
    ],
)
def test_detector_catches_the_yaml_float_trap(text, expected_numeric):
    """The check above is only useful if it actually flags the bare-exponent form."""
    assert (not _numeric_offenders(yaml.safe_load(text))) is expected_numeric


def test_pmid_is_declared_a_support_variable():
    """PMID is added to 'diag_var_list' by the ADF itself (adf_derive adds it for
    the aerosol calculations, and the vertical interpolation needs it), so a user
    who never asked for it would otherwise get a set of pressure plots.  The
    plotting scripts skip it via AdfObs.plot_var_list, which reads this flag.

    Only the declaration is checked here: importing AdfObs pulls in xarray, which
    the lint/test CI environment does not install, so a test that imported it
    would silently skip instead of running.
    """
    defaults = yaml.safe_load(DEFAULTS_FILE.read_text())
    assert "PMID" in defaults, "PMID lost its variable_defaults entry"
    assert defaults["PMID"].get("plot_diagnostics") is False, (
        "PMID must declare 'plot_diagnostics: False' or the ADF will plot the "
        "pressure field it adds to diag_var_list on the user's behalf"
    )


def test_plot_diagnostics_is_documented():
    """The key is only discoverable if the header block explains it."""
    header = DEFAULTS_FILE.read_text().split("PMID:")[0]
    assert (
        "plot_diagnostics" in header
    ), "document 'plot_diagnostics' in the header of adf_variable_defaults.yaml"
