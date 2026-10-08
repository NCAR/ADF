"""
Collection of python unit tests
for "adf_derive.derive_variable".

Covers how a derived variable is built from its constituents' time series
files: which files are selected (case, history stream, chunking), what the
resulting file is called, and which situations are refused rather than turned
into a plausible-looking but wrong field.

NOTE: adf_derive imports xarray and (through adf_utils) the rest of the
scientific stack, so these are skipped in CI, which installs only PyYAML and
pytest.  They run in a full ADF environment.
"""

# +++++++++++++++++++++++
# Import required modules
# +++++++++++++++++++++++

import unittest
import sys
import os
import os.path
import tempfile
from pathlib import Path

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory to python path:
sys.path.append(_ADF_LIB_DIR)

try:
    import numpy as np
    import xarray as xr
    from adf_derive import constit_read_chunks, derive_variable

    _HAS_ADF_DERIVE = True
except ImportError:
    _HAS_ADF_DERIVE = False


class _StubData:
    """Stand-in for AdfData, whose load_dataset is all derive_variable uses."""

    def load_dataset(self, fils, chunks=None):
        """Mirror AdfData.load_dataset: a list in, a Dataset (or None) out."""
        if len(fils) == 0:
            return None
        if len(fils) > 1:
            return xr.open_mfdataset(fils, combine="by_coords", chunks=chunks)
        return xr.open_dataset(str(fils[0]), chunks=chunks)


class _StubAdf:
    """Minimal stand-in for the AdfDiag object derive_variable is handed."""

    def __init__(self):
        self.data = _StubData()
        self.messages = []

    def debug_log(self, msg):
        """Collect rather than write, so tests can assert on the log."""
        self.messages.append(msg)


def _write_ts(path, var, start_year, nyears):
    """Write a small time series file holding `var` over `nyears` years."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    time = xr.date_range(
        f"{start_year:04d}-01-01",
        periods=nyears * 12,
        freq="MS",
        calendar="noleap",
        use_cftime=True,
    )
    ds = xr.Dataset(
        {var: (("time",), np.ones(len(time), dtype="f4"))}, coords={"time": time}
    )
    ds[var].attrs = {"units": "W/m2", "long_name": f"{var} long name"}
    ds.to_netcdf(path)


# Variables that get multiplied by dry air density; SO4 is one of them:
_AEROSOL_RES = {"aerosol_zonal_list": ["SO4"], "Rgas": 287.04}


@unittest.skipUnless(_HAS_ADF_DERIVE, "adf_derive dependencies not available")
class AdfDeriveTestRoutine(unittest.TestCase):
    """
    Unit tests for derived time series file creation.
    """

    def test_single_file_constituents(self):
        """
        The ordinary case, and the one every existing configuration uses: one
        file per constituent produces one derived file named for that span.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("FSNT", "FLNT"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-002012.nc", var, 1, 20
                )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).glob("*RESTOM*.nc"))
            self.assertEqual(
                [f.name for f in out], ["case.cam.h0a.RESTOM.000101-002012.nc"]
            )
            with xr.open_dataset(out[0]) as ds:
                self.assertEqual(len(ds.time), 240)

    def test_ressurf_formula_and_stream_name(self):
        """
        RESSURF is built from eight constituents, whatever the history stream is
        called, and the result matches the formula.
        """

        # Constant values, so the expected answer is a single number:
        vals = dict(
            FSNS=200.0,
            FLNS=60.0,
            SHFLX=20.0,
            QFLX=2.0e-5,
            PRECC=1.0e-8,
            PRECL=2.0e-8,
            PRECSC=0.3e-8,
            PRECSL=0.7e-8,
        )
        lat_vap, lat_fus = 2.501e6, 3.337e5
        expected = (
            200.0
            - 60.0
            - 20.0
            - (lat_vap + lat_fus) * 2.0e-5
            + lat_fus * 1.0e3 * (1.0e-8 + 2.0e-8 - 0.3e-8 - 0.7e-8)
        )

        for stream in ("cam.h0", "cam.h0a"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for var, val in vals.items():
                    fil = Path(tmpdir) / f"case.{stream}.{var}.000101-000112.nc"
                    _write_ts(fil, var, 1, 1)
                    with xr.open_dataset(fil) as ds:
                        ds = ds.load()
                    ds[var][:] = val
                    ds.to_netcdf(fil)

                derive_variable(
                    _StubAdf(),
                    "case",
                    "RESSURF",
                    res={},
                    ts_dir=tmpdir,
                    constit_list=list(vals),
                    hist_str=stream,
                )

                out = sorted(Path(tmpdir).glob("*RESSURF*.nc"))
                self.assertEqual(
                    [f.name for f in out], [f"case.{stream}.RESSURF.000101-000112.nc"]
                )
                with xr.open_dataset(out[0]) as ds:
                    self.assertEqual(ds["RESSURF"].attrs["units"], "W/m2")
                    np.testing.assert_allclose(
                        ds["RESSURF"].values, expected, rtol=1e-4
                    )

    def test_chunked_constituents_span_all_chunks(self):
        """
        A constituent split into consecutive chunks must produce one derived
        file covering the whole period, named for that whole period.  Taking
        only the first chunk gave a half-length variable that the AMWG table
        then reported next to full-length ones.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("FSNT", "FLNT"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-001012.nc", var, 1, 10
                )
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.001101-002012.nc", var, 11, 10
                )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).glob("*RESTOM*.nc"))
            self.assertEqual(
                [f.name for f in out], ["case.cam.h0a.RESTOM.000101-002012.nc"]
            )
            with xr.open_dataset(out[0]) as ds:
                self.assertEqual(len(ds.time), 240)

    def test_multiple_history_streams(self):
        """
        Derivation runs once per configured history stream.  Without being
        told which one, the search matches both streams' copies of a
        constituent, whose dates are identical, and the variable is dropped.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for hstr in ("cam.h0a", "cam.h0b"):
                for var in ("FSNT", "FLNT"):
                    _write_ts(
                        Path(tmpdir) / f"case.{hstr}.{var}.000101-002012.nc", var, 1, 20
                    )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).glob("*RESTOM*.nc"))
            self.assertEqual(
                [f.name for f in out], ["case.cam.h0a.RESTOM.000101-002012.nc"]
            )

    def test_aerosol_multiple_history_streams(self):
        """
        The same for the aerosol path: PMID and T present under two streams
        must not be combined, which xarray cannot do and reports as an
        uncaught ValueError that ends the run.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("so4_a1", "so4_a2"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-002012.nc", var, 1, 20
                )
            for hstr in ("cam.h0a", "cam.h0b"):
                for var in ("PMID", "T"):
                    _write_ts(
                        Path(tmpdir) / f"case.{hstr}.{var}.000101-002012.nc", var, 1, 20
                    )

            derive_variable(
                _StubAdf(),
                "case",
                "SO4",
                res=_AEROSOL_RES,
                ts_dir=tmpdir,
                constit_list=["so4_a1", "so4_a2"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).glob("*.SO4.*.nc"))
            self.assertEqual(
                [f.name for f in out], ["case.cam.h0a.SO4.000101-002012.nc"]
            )
            with xr.open_dataset(out[0]) as ds:
                self.assertFalse(np.isnan(ds["SO4"].values).any())
                self.assertEqual(ds["SO4"].attrs["units"], "kg/m3")

    def test_aerosol_refuses_mismatched_pmid(self):
        """
        PMID and T covering a different period than the constituents must be
        refused.  xarray aligns on time with an outer join, so this does not
        raise -- it silently writes an all-NaN field that looks like a real
        one.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("so4_a1", "so4_a2"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-002012.nc", var, 1, 20
                )
            # Only another case's PMID/T are present, over a disjoint period:
            for var in ("PMID", "T"):
                _write_ts(
                    Path(tmpdir) / f"other.cam.h0a.{var}.010001-011912.nc", var, 100, 20
                )

            derive_variable(
                _StubAdf(),
                "case",
                "SO4",
                res=_AEROSOL_RES,
                ts_dir=tmpdir,
                constit_list=["so4_a1", "so4_a2"],
                hist_str="cam.h0a",
            )

            self.assertEqual(sorted(Path(tmpdir).glob("*.SO4.*.nc")), [])

    def test_overlapping_constituent_files_refused(self):
        """
        Two sets of files for one constituent (a run extended and re-processed
        over a longer period) cannot be combined: the time axis would contain
        duplicates.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("FSNT", "FLNT"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-002012.nc", var, 1, 20
                )
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-004012.nc", var, 1, 40
                )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            self.assertEqual(sorted(Path(tmpdir).glob("*RESTOM*.nc")), [])

    def test_overlapping_constituents_resolved_by_years(self):
        """
        The same two sets, but with the years being processed given: the set
        covering them is used and the derived variable is produced, which is
        the point of passing the years down at all.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("FSNT", "FLNT"):
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-002012.nc", var, 1, 20
                )
                _write_ts(
                    Path(tmpdir) / f"case.cam.h0a.{var}.000101-004012.nc", var, 1, 40
                )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
                syr=1,
                eyr=40,
            )

            out = sorted(Path(tmpdir).glob("*RESTOM*.nc"))
            self.assertEqual(
                [f.name for f in out], ["case.cam.h0a.RESTOM.000101-004012.nc"]
            )
            with xr.open_dataset(out[0]) as ds:
                self.assertEqual(len(ds.time), 480)

    def test_other_case_in_same_tree_ignored(self):
        """
        Several cases can share one time series tree, and the search recurses.
        A derivation must use its own case's files, not whichever sorts first.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            mine = Path(tmpdir) / "mycase" / "proc" / "tseries"
            other = Path(tmpdir) / "acase" / "proc" / "tseries"
            for var in ("FSNT", "FLNT"):
                _write_ts(mine / f"mycase.cam.h0a.{var}.000101-002012.nc", var, 1, 20)
                _write_ts(other / f"acase.cam.h0a.{var}.002101-004012.nc", var, 21, 20)

            derive_variable(
                _StubAdf(),
                "mycase",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).rglob("*RESTOM*.nc"))
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].name, "mycase.cam.h0a.RESTOM.000101-002012.nc")
            self.assertEqual(out[0].parent, mine)
            with xr.open_dataset(out[0]) as ds:
                self.assertEqual(len(ds.time), 240)

    def test_case_name_containing_constituent_name(self):
        """
        The derived file name is built by substituting whole dot-separated
        tokens.  A plain string replace rewrites a case name that happens to
        contain the constituent's name too, producing a file that
        AdfData.get_timeseries_file (which searches on the case name) can
        never find.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            for var in ("FSNT", "FLNT"):
                _write_ts(
                    Path(tmpdir) / f"bFSNTtest.cam.h0a.{var}.000101-002012.nc",
                    var,
                    1,
                    20,
                )

            derive_variable(
                _StubAdf(),
                "bFSNTtest",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            out = sorted(Path(tmpdir).glob("*RESTOM*.nc"))
            self.assertEqual(
                [f.name for f in out], ["bFSNTtest.cam.h0a.RESTOM.000101-002012.nc"]
            )

    def test_missing_constituent_writes_nothing(self):
        """
        A constituent absent from the run means the variable cannot be built;
        say so and move on rather than writing a partial field.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_ts(
                Path(tmpdir) / "case.cam.h0a.FSNT.000101-002012.nc", "FSNT", 1, 20
            )

            derive_variable(
                _StubAdf(),
                "case",
                "RESTOM",
                res={},
                ts_dir=tmpdir,
                constit_list=["FSNT", "FLNT"],
                hist_str="cam.h0a",
            )

            self.assertEqual(sorted(Path(tmpdir).glob("*RESTOM*.nc")), [])

    def test_read_chunks_whole_fields(self):
        """
        Constituents are read as whole fields, as many time steps at a time
        as fit the target size, in whole multiples of the on-disk time chunk.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "case.cam.h3a.PRECC.000101-000112.nc"
            ds = xr.Dataset(
                {"PRECC": (("time", "lat", "lon"), np.ones((100, 4, 6), "f4"))},
                coords={"time": np.arange(100.0)},
            )
            ds.to_netcdf(
                path,
                unlimited_dims="time",
                encoding={"PRECC": {"chunksizes": [7, 2, 3]}},
            )

            # One step is 4 x 6 x 4 bytes = 96; 30 steps fit, 28 = 4 x 7.
            chunks = constit_read_chunks([str(path)], target_bytes=96 * 30)
            self.assertEqual(chunks, {"time": 28, "lat": -1, "lon": -1})
            # Less than one disk chunk fits: no rounding, at least one step.
            chunks = constit_read_chunks([str(path)], target_bytes=10)
            self.assertEqual(chunks["time"], 1)


# ++++++++++++++++++

# Run unit tests if this script is called directly:
if __name__ == "__main__":
    unittest.main()

#############
# End of file
