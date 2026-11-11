"""The debug log names the grid a plot's two sides are compared on, and how.

The same pair of cases gives slightly different statistics when compared on
the native mesh (UXarray mode) and on a lat/lon grid (gridded mode), so the
debug log says which grid each plot type used.  The regridding stage writes into each
file how that side got onto the comparison grid -- the case's weights file and
method, a lat/lon regrid, or a remap onto the mesh -- and AdfData.log_regrid_note
puts the two sides together in one line of the debug log per plot type.
"""

# The tests drive private helpers directly:
# pylint: disable=protected-access

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest  # pylint: disable=import-error

# Skip rather than fail collection when the science stack is absent, as in a
# local run outside the ADF environment.  CI runs these in the full ADF
# environment (.github/workflows/ADF_unit_tests.yaml), where a skip is a failure.
np = pytest.importorskip("numpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("xesmf")

sys.path.append(str(Path(__file__).parents[2]))
sys.path.append(str(Path(__file__).parents[3] / "scripts" / "regridding"))

# pylint: disable-next=import-error,wrong-import-position,wrong-import-order
import regrid_and_vert_interp as rvi  # noqa: E402

# pylint: disable-next=import-error,wrong-import-position
import adf_utils as utils  # noqa: E402

# pylint: disable-next=import-error,wrong-import-position
from adf_dataset import AdfData  # noqa: E402


def _latlon(nlat, nlon):
    lat = -90.0 + 180.0 / nlat * (np.arange(nlat) + 0.5)
    lon = 360.0 / nlon * (np.arange(nlon) + 0.5)
    return xr.DataArray(
        np.zeros((nlat, nlon)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="T",
    )


def test_grid_label():
    """lat/lon grids by size, native grids by cell count."""
    assert utils.grid_label(_latlon(192, 288)) == "192x288 lat/lon grid"
    native = xr.Dataset({"T": ("ncol", np.zeros(21600))})
    assert utils.grid_label(native) == "21,600-cell native grid"
    assert utils.grid_label(xr.DataArray(np.zeros(3), dims="x")) == "unknown grid"


def test_method_names():
    """Methods as a reader would name them, whatever the code stores."""
    assert rvi._method_name("coservative") == "conservative"
    assert rvi._method_name(None) == "conservative"
    assert rvi._method_name("ESMF weights_latlon_192_288_x_to_ne30_conserve.nc") == (
        "ESMF conservative"
    )
    assert rvi._method_name("uxarray bilinear") == "bilinear, uxarray"
    assert rvi._method_name("uxarray inverse_distance_weighted") == (
        "inverse-distance, uxarray"
    )


def _adf(cons=None, flagged=False):
    """An AdfDiag stand-in with one native test case and a native baseline."""
    return SimpleNamespace(
        latlon_wgt_files={
            "test_wgts_file": ["/maps/ne30pg3_TO_f09-bilin.nc"],
            "test_cons_wgts_file": [cons],
            "baseline_wgts_file": "/maps/ne30pg3_TO_f09-bilin.nc",
            "baseline_cons_wgts_file": None,
        },
        latlon_regrid_method={
            "test_regrid_method": ["bilinear"],
            "baseline_regrid_method": "coservative",
        },
        variable_defaults={"PRECT": {"regrid_conservative": flagged}},
    )


def test_gridding_note_names_the_configured_weights():
    """The note names the weights file and method the config gives."""
    assert rvi._gridding_note(_adf(), "PRECT", 0) == (
        "gridded from its native grid with ne30pg3_TO_f09-bilin.nc (bilinear)"
    )
    assert rvi._gridding_note(_adf(), "PRECT") == (
        "gridded from its native grid with ne30pg3_TO_f09-bilin.nc (conservative)"
    )


def test_gridding_note_with_a_conservative_map():
    """A flagged flux gridded from time series names the conservative map.

    The conservative map is only used for time series (PR #510), not when the
    regridding stage grids native climo.
    """
    adf = _adf(cons="/maps/map_aave.nc", flagged=True)
    assert rvi._gridding_note(adf, "PRECT", 0, from_climo=False) == (
        "gridded from its native grid with map_aave.nc (conservative)"
    )
    assert "bilin" in rvi._gridding_note(adf, "PRECT", 0, from_climo=True)
    assert "bilin" in rvi._gridding_note(
        _adf(cons="/maps/map_aave.nc"), "PRECT", 0, from_climo=False
    )


def test_horizontal_note():
    """Nothing on the same grid; the grids and method otherwise."""
    f09, one_deg = _latlon(192, 288), _latlon(180, 360)
    assert rvi._horizontal_note(f09, f09.to_dataset(), f09) == ""
    assert rvi._horizontal_note(f09, one_deg.to_dataset(), one_deg) == (
        "regridded from 192x288 lat/lon grid to 180x360 lat/lon grid (conservative)"
    )
    mesh = xr.Dataset({"T": ("ncol", np.zeros(21600))})
    on_mesh = xr.DataArray(
        np.zeros(21600), dims="ncol", attrs={"remap_method": "ESMF w_conserve.nc"}
    )
    assert rvi._horizontal_note(f09, mesh, on_mesh) == (
        "remapped from 192x288 lat/lon grid onto the reference's mesh (ESMF conservative)"
    )
    assert rvi._horizontal_note(on_mesh, mesh, on_mesh) == ""
    assert rvi._join_notes("a", "", "b") == "a; then b"


def _note_data(test_file, ref_file):
    """An AdfData stand-in whose regridded files are the two given."""
    obj = SimpleNamespace(adf=SimpleNamespace(logged=[]))
    obj.adf.debug_log = obj.adf.logged.append
    obj._regrid_files = lambda case, field, kwargs: [test_file] if test_file else []
    obj._reference_files = lambda field, kwargs: [ref_file] if ref_file else []
    obj._make_regrid_note = lambda case, field, unstructured: AdfData._make_regrid_note(
        obj, case, field, unstructured
    )
    return obj


def test_regrid_note_reads_both_files(tmp_path):
    """The note names the comparison grid and each side's regrid steps."""
    test_file, ref_file = tmp_path / "test.nc", tmp_path / "ref.nc"
    _latlon(192, 288).to_dataset().assign_attrs(
        adf_regrid="gridded from its native grid with map.nc (bilinear)"
    ).to_netcdf(test_file)
    _latlon(192, 288).to_dataset().to_netcdf(ref_file)
    obj = _note_data(test_file, ref_file)
    note = (
        "Compared on the 192x288 lat/lon grid.  Test: gridded from its native grid "
        "with map.nc (bilinear).  Baseline: not regridded."
    )
    assert AdfData.log_regrid_note(obj, "zonal_mean", "case", "T") == note
    assert obj.adf.logged == [f"regrid: zonal_mean, case (T): {note}"]
    # Asked again, the note comes from the cache, not the (now gone) files:
    test_file.unlink()
    assert AdfData.log_regrid_note(obj, "zonal_mean", "case", "T") == note


def test_one_log_line_per_plot_type():
    """Each plot type logs a note once, and again only for a different note."""
    obj = _note_data(None, None)
    # A flux variable gridded with the conservative map has its own note:
    obj._make_regrid_note = lambda case, field, unstructured: (
        "conservative" if field == "PRECT" else "bilinear"
    )
    for field in ("T", "T", "U", "PRECT", "Q", "PRECT"):  # seasons, variables
        AdfData.log_regrid_note(obj, "polar_map", "case", field)
    AdfData.log_regrid_note(obj, "zonal_mean", "case", "T")
    assert obj.adf.logged == [
        "regrid: polar_map, case (T): bilinear",
        "regrid: polar_map, case (PRECT): conservative",
        "regrid: zonal_mean, case (T): bilinear",
    ]


def test_no_note_without_a_test_file():
    """No file for the case: no note, and nothing logged."""
    obj = _note_data(None, None)
    assert AdfData.log_regrid_note(obj, "polar_map", "case", "T") == ""
    assert not obj.adf.logged
