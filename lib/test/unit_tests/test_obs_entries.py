"""
Collection of python unit tests for the observation entries in
"adf_variable_defaults.yaml".

These check that the configuration is internally consistent.  They cannot check
that the observation files themselves exist, since those live on GLADE and CI
has no access to them, but the mistakes they do catch are the ones that are
easy to make and quiet to miss: an obs_file with no obs_var_name, a scale
factor left behind on a variable that no longer has observations, or two
entries claiming different source names for the same file.
"""

# +++++++++++++++++++++++
# Import required modules
# +++++++++++++++++++++++

import os
import os.path
import unittest

import yaml

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)
_DEFAULTS = os.path.join(_ADF_LIB_DIR, "adf_variable_defaults.yaml")

# Keys that only mean something alongside an "obs_file":
_OBS_ONLY_KEYS = (
    "obs_var_name",
    "obs_name",
    "obs_scale_factor",
    "obs_add_offset",
    "obs_lev",
)


def _is_comparison(block):
    """Whether an entry is a variable-against-observations comparison.

    The cloud regime entries (ISCCP_emd_centers and friends) also use
    'obs_file', but they point at .npy arrays of cluster centres that
    cloud_regime_analysis.py reads directly rather than at a data set the ADF
    compares a variable against, so they carry no obs_var_name and are not
    subject to the checks below.
    """
    obs_file = block.get("obs_file")
    return bool(obs_file) and not str(obs_file).endswith(".npy")


class _NoDuplicatesLoader(yaml.SafeLoader):
    """A loader that refuses duplicate keys instead of silently keeping one.

    PyYAML's default behaviour is to take the last of a repeated key, so a
    block that sets obs_scale_factor twice parses cleanly and reads correctly
    while the file itself is wrong.  That happened on this file: an entry kept
    its old scale factor and gained a second copy of it, and nothing noticed
    until the repository's check-yaml hook ran.
    """


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AssertionError(f"duplicate key '{key}' at {key_node.start_mark}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicatesLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def _variable_blocks():
    """Every mapping entry in the defaults file, keyed by variable name."""
    with open(_DEFAULTS, encoding="utf-8") as fil:
        defaults = yaml.safe_load(fil)
    return {k: v for k, v in defaults.items() if isinstance(v, dict)}


class AdfObsEntriesTestRoutine(unittest.TestCase):
    """Consistency checks on the observation entries."""

    def test_no_duplicate_keys(self):
        """A repeated key parses fine and hides a half-finished edit."""
        with open(_DEFAULTS, encoding="utf-8") as fil:
            try:
                yaml.load(fil, Loader=_NoDuplicatesLoader)
            except AssertionError as err:
                self.fail(f"adf_variable_defaults.yaml has a {err}")

    def test_obs_file_has_variable_and_name(self):
        """A variable compared against observations needs to say which
        variable in which data set, otherwise the comparison cannot be set up."""
        for var, block in _variable_blocks().items():
            if not _is_comparison(block):
                continue
            for key in ("obs_var_name", "obs_name"):
                self.assertIn(
                    key,
                    block,
                    msg=f"'{var}' sets obs_file but not {key}",
                )

    def test_no_orphan_obs_keys(self):
        """obs_scale_factor and friends do nothing without an obs_file, so
        finding one alone means an entry was half-edited."""
        for var, block in _variable_blocks().items():
            if "obs_file" in block:
                continue
            for key in _OBS_ONLY_KEYS:
                self.assertNotIn(
                    key,
                    block,
                    msg=f"'{var}' sets {key} but has no obs_file",
                )

    def test_one_source_name_per_file(self):
        """Two variables reading the same file should agree about what that
        file is; disagreeing names show up in plot titles and file names."""
        names_by_file = {}
        for var, block in _variable_blocks().items():
            if not _is_comparison(block):
                continue
            names_by_file.setdefault(block["obs_file"], {}).setdefault(
                block.get("obs_name"), []
            ).append(var)
        for obs_file, names in names_by_file.items():
            self.assertEqual(
                len(names),
                1,
                msg=f"'{obs_file}' is given more than one obs_name: "
                f"{ {n: v for n, v in names.items()} }",
            )

    def test_obs_name_is_path_safe(self):
        """obs_name is pasted into the regridded file name, so a '/' in it
        sends the write into a directory that does not exist and the run dies
        far from the cause."""
        for var, block in _variable_blocks().items():
            name = block.get("obs_name")
            if name is None:
                continue
            for bad in ("/", os.sep):
                self.assertNotIn(
                    bad,
                    str(name),
                    msg=f"'{var}' has an obs_name containing '{bad}': {name}",
                )

    def test_scale_factors_are_numbers(self):
        """A quoted scale factor multiplies nothing and raises later."""
        for var, block in _variable_blocks().items():
            for key in (
                "scale_factor",
                "add_offset",
                "obs_scale_factor",
                "obs_add_offset",
            ):
                if key in block:
                    self.assertIsInstance(
                        block[key],
                        (int, float),
                        msg=f"'{var}' has a non-numeric {key}: {block[key]!r}",
                    )


# +++++++++++++++++++++++
if __name__ == "__main__":
    unittest.main()
