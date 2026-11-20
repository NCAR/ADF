"""
Collection of python unit tests for how gridded native-grid time series are
written: "write_netcdf_synchronous" in adf_diag.py, "ts_time_chunk" in
adf_utils.py, and gridding a dataset in dask time chunks.

Threaded dask writes hung intermittently on large files, so these tests check
that every chunk is computed in the calling thread, that the file only
appears under its final name once the write succeeds, that a failed write
leaves no file behind, and that gridding in time chunks gives the same file as
gridding all at once.  Each test writes small synthetic files to a temporary
directory.

NOTE: these tests import adf_diag, adf_utils and xesmf, so they are skipped
in CI, which installs only PyYAML and pytest.  They run in a full ADF
environment.
"""

import os
import os.path
import sys
import tempfile
import threading
import unittest

# Set relevant path variables:
_CURRDIR = os.path.abspath(os.path.dirname(__file__))
_ADF_LIB_DIR = os.path.join(_CURRDIR, os.pardir, os.pardir)

# Add ADF "lib" directory and this directory to python path:
sys.path.append(_ADF_LIB_DIR)
sys.path.append(_CURRDIR)

try:
    import numpy as np
    import xarray as xr
    import dask.array as da
    import xesmf  # noqa: F401  (grid_to_latlon needs it)
    from adf_utils import chunk_native_in_time, grid_to_latlon, ts_time_chunk
    from adf_diag import save_to_nc, write_netcdf_synchronous
    from test_grid_to_latlon_time import _identity_weights, _native_time_series

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _make_dataset(record=None):
    """Small dask-backed dataset with 4 time chunks.

    If `record` is a list, the id of the thread that computes each chunk is
    appended to it.
    """

    values = np.arange(4 * 3 * 5, dtype="f4").reshape(4, 3, 5)
    data = da.from_array(values, chunks=(1, 3, 5))
    if record is not None:

        def _note_thread(block):
            record.append(threading.get_ident())
            return block

        data = data.map_blocks(_note_thread, dtype=data.dtype)
    # End if
    ds = xr.Dataset(
        {"PRECT": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(4.0),
            "lat": np.arange(3.0),
            "lon": np.arange(5.0),
        },
    )
    return ds, values


@unittest.skipUnless(_HAS_DEPS, "adf_diag/xesmf dependencies not available")
class WriteNetcdfSynchronousTestRoutine(unittest.TestCase):
    """
    Unit tests for write_netcdf_synchronous.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.outname = os.path.join(self._tmp.name, "test.PRECT.nc")

    def tearDown(self):
        self._tmp.cleanup()

    def test_chunks_computed_in_calling_thread(self):
        """Every chunk is computed in the calling thread (no dask thread pool)."""

        threads = []
        ds, _ = _make_dataset(record=threads)
        write_netcdf_synchronous(ds.to_netcdf, self.outname)
        self.assertGreaterEqual(len(threads), 4)
        self.assertEqual(set(threads), {threading.get_ident()})

    def test_output_matches_input(self):
        """The file is written under its final name, unchanged, with no .tmp left."""

        ds, values = _make_dataset()
        write_netcdf_synchronous(lambda path: save_to_nc(ds, path), self.outname)
        self.assertTrue(os.path.exists(self.outname))
        self.assertFalse(os.path.exists(self.outname + ".tmp"))
        with xr.open_dataset(self.outname) as out:
            np.testing.assert_array_equal(out["PRECT"].values, values)

    def test_failed_write_leaves_no_file(self):
        """A failed write creates no final file and removes its .tmp."""

        def _fail(path):
            with open(path, "w", encoding="utf-8") as fil:
                fil.write("partial")
            raise RuntimeError("simulated write failure")

        with self.assertRaises(RuntimeError):
            write_netcdf_synchronous(_fail, self.outname)
        self.assertFalse(os.path.exists(self.outname))
        self.assertFalse(os.path.exists(self.outname + ".tmp"))

    def test_failed_write_keeps_existing_file(self):
        """A failed write leaves an existing file at the final name untouched."""

        with open(self.outname, "w", encoding="utf-8") as fil:
            fil.write("original")

        def _fail(path):
            raise RuntimeError(f"simulated write failure for {path}")

        with self.assertRaises(RuntimeError):
            write_netcdf_synchronous(_fail, self.outname)
        with open(self.outname, encoding="utf-8") as fil:
            self.assertEqual(fil.read(), "original")


@unittest.skipUnless(_HAS_DEPS, "adf_utils dependencies not available")
class TsTimeChunkTestRoutine(unittest.TestCase):
    """
    Unit tests for ts_time_chunk.
    """

    def test_chunk_size(self):
        """A chunk holds as many whole time steps as fit in the target size."""

        ds = xr.Dataset({"T": (("time", "ncol"), np.zeros((100, 250), dtype="f4"))})
        # one step is 250 * 4 = 1000 bytes:
        self.assertEqual(ts_time_chunk(ds, "T", target_bytes=10_500), 10)
        # at least one step, even if a step is bigger than the target:
        self.assertEqual(ts_time_chunk(ds, "T", target_bytes=10), 1)

    def test_no_time(self):
        """A variable without a time dimension gets no time chunk."""

        ds = xr.Dataset({"area": (("ncol",), np.ones(4))})
        self.assertIsNone(ts_time_chunk(ds, "area"))


@unittest.skipUnless(_HAS_DEPS, "adf_utils/xesmf dependencies not available")
class GridInTimeChunksTestRoutine(unittest.TestCase):
    """
    Gridding a native-grid dataset in dask time chunks, as
    AdfDiag.grid_native_time_series does, gives the same file as gridding it
    all at once.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.wgt = os.path.join(self._tmp.name, "wgts.nc")
        _identity_weights(self.wgt)
        rng = np.random.default_rng(0)
        self.native = xr.Dataset(
            {"Q": (("time", "ncol"), rng.random((24, 4)).astype("f4"))},
            coords={"time": ("time", np.arange(24.0) + 0.5)},
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _grid_and_write(self, ds, name):
        out = os.path.join(self._tmp.name, name)
        gridded = grid_to_latlon(ds, "Q", wgt_file=self.wgt, method="bilinear")
        write_netcdf_synchronous(lambda path: save_to_nc(gridded, path), out)
        with xr.open_dataset(out) as result:
            return result.load()

    def test_chunked_equals_unchunked(self):
        """Same values, in the same places, chunked or not."""

        whole = self._grid_and_write(self.native, "whole.nc")
        chunked_ds = self.native.chunk({"time": 5})
        chunked = self._grid_and_write(chunked_ds, "chunked.nc")
        self.assertEqual(whole["Q"].dims, chunked["Q"].dims)
        np.testing.assert_array_equal(whole["Q"].values, chunked["Q"].values)
        np.testing.assert_array_equal(whole["time"].values, chunked["time"].values)

    def test_precision_default_and_kept(self):
        """Gridded fields are 64-bit by default, native (32-bit) with keep_dtype."""

        default = grid_to_latlon(self.native, "Q", wgt_file=self.wgt, method="bilinear")
        kept = grid_to_latlon(
            self.native, "Q", wgt_file=self.wgt, method="bilinear", keep_dtype=True
        )
        self.assertEqual(default["Q"].dtype, np.float64)
        self.assertEqual(kept["Q"].dtype, np.float32)
        np.testing.assert_array_equal(
            kept["Q"].values, default["Q"].values.astype("f4")
        )

    def test_time_series_file_with_bounds(self):
        """A time series file with decoded time bounds, chunked as
        grid_native_time_series chunks it, grids and writes like the unchunked
        file (time bounds stay unchunked: xarray cannot write them chunked)."""

        native = os.path.join(self._tmp.name, "native.nc")
        _native_time_series(native)
        with xr.open_dataset(native) as ts_ds:
            chunked_ds = chunk_native_in_time(ts_ds, "Q", target_bytes=7 * 4 * 4)
            self.assertEqual(chunked_ds["Q"].chunks[0][0], 7)
            self.assertIsNone(chunked_ds["time_bounds"].chunks)
            chunked = self._grid_and_write(chunked_ds, "chunked_bounds.nc")
            whole = self._grid_and_write(ts_ds, "whole_bounds.nc")
        np.testing.assert_array_equal(whole["Q"].values, chunked["Q"].values)
        np.testing.assert_array_equal(
            whole["time_bounds"].values, chunked["time_bounds"].values
        )


if __name__ == "__main__":
    unittest.main()
