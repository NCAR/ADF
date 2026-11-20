"""Prototype: compare a native-grid case with lat/lon data on the native mesh.

The ADF today grids native output to lat/lon for comparisons with lat/lon
cases and observations, and skips the comparison when plotting on the mesh
('unstructured_plotting', e.g. N09a, N09b, L14).  This puts the lat/lon side
onto the test case's mesh instead and draws the ADF's usual four-panel map
(test, reference, difference, % difference) with plot_map_and_save.

Two ways onto the mesh, side by side:
  ux     uxarray remap.bilinear (adf_utils.latlon_to_mesh): no weights file,
         any lat/lon grid
  esmf   ESMF conservative weights made on the fly (write_latlon_scrip,
         make_mesh_weights, apply_mesh_weights): keeps area means

Pairs (atmosphere, ne30pg3, years 1-5, from the LDF-merge test runs):
  same   native .092 vs the f09 remap of .092 (should be ~0: checks the remap)
  base   native .092 vs f09 .091 (what N09b would plot)
  obs    native .092 vs observations (what N09a would plot)

usage: python latlon_on_mesh_demo.py OUTDIR [VAR ...]
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import uxarray as ux

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
import adf_utils as utils  # noqa: E402  pylint: disable=wrong-import-position
import plotting_functions as pf  # noqa: E402  pylint: disable=wrong-import-position

RUNS = Path("/glade/derecho/scratch/richling/adf-output/tests/LDF-merge")
NATIVE = RUNS / "N04_test-native-hist_base-native-hist_uxarray/climo"
F09 = RUNS / "S01_test-hist_base-hist-again/climo"
MESH = "/glade/campaign/cesm/cesmdata/inputdata/share/meshes/ne30pg3_ESMFmesh_cdf5_c20211018.nc"
OBS = Path("/glade/campaign/cgd/amp/amwg/ADF_obs")
TEST, BASE = (
    "b.e23_alpha17f.BLT1850.ne30_t232.092",
    "b.e23_alpha17f.BLT1850.ne30_t232.091",
)

# Variable: (scale factor for model output, obs file, obs variable, units, plot options)
VARS = {
    "PRECT": (
        86400000.0,
        "PRECT_GPCP_v2.3_monthly_climo_198301-202512.nc",
        "PRECT",
        "mm/day",
        {
            "colormap": "Blues",
            "contour_levels_range": [0, 16, 1],
            "diff_colormap": "BrBG",
            "diff_contour_range": [-5, 5, 0.5],
        },
    ),
    "SWCF": (
        1.0,
        "CERES_EBAF_Ed4.1_2001-2020.nc",
        "toa_cre_sw_mon",
        "W/m2",
        {
            "colormap": "Blues_r",
            "contour_levels_range": [-150, 0, 10],
            "diff_colormap": "BrBG",
            "diff_contour_range": [-40, 40, 5],
        },
    ),
    "LWCF": (
        1.0,
        "CERES_EBAF_Ed4.1_2001-2020.nc",
        "toa_cre_lw_mon",
        "W/m2",
        {
            "colormap": "Oranges",
            "contour_levels_range": [0, 80, 5],
            "diff_colormap": "BrBG",
            "diff_contour_range": [-30, 30, 5],
        },
    ),
}


def climo(folder, case, var):
    return next((folder / case / "yrs_1_5").glob(f"{case}_cam.h0a_{var}_climo_*.nc"))


def ann(da):
    """Annual mean of a 12-month climatology."""
    return da.mean("time") if "time" in da.dims else da


def latlon_mean(da):
    wgt = np.cos(np.deg2rad(da["lat"])) * xr.ones_like(da)
    wgt = wgt.where(np.isfinite(da))
    return float((da * wgt).sum() / wgt.sum())


def mesh_mean(values, area):
    values = np.asarray(values)
    ok = np.isfinite(values)
    return float((values[ok] * area[ok]).sum() / area[ok].sum())


def rmesh_method(da):
    return da.attrs.get("remap_method", "remap").replace(".nc", "")


def main(outdir, variables):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = ux.open_grid(MESH)
    area = grid.face_areas.values
    rows = []
    for var in variables:
        scale, obs_file, obs_var, units, opts = VARS[var]
        test = ann(ux.open_dataset(MESH, climo(NATIVE, TEST, var))[var]) * scale
        refs = {
            "same": (
                ann(xr.open_dataset(climo(F09, TEST, var))[var]) * scale,
                "f09 .092",
                "f09",
            ),
            "base": (
                ann(xr.open_dataset(climo(F09, BASE, var))[var]) * scale,
                "f09 .091",
                "f09",
            ),
        }
        obs = xr.open_dataset(OBS / obs_file)[obs_var]
        refs["obs"] = (
            ann(obs),
            obs_file.split("_")[0] if var == "PRECT" else "CERES-EBAF",
            "obs",
        )
        for pair, (ref, label, gridname) in refs.items():
            ref = ref.astype("float64")
            ref.attrs["units"] = units
            weights = (
                outdir
                / f"weights_{gridname}{ref.sizes['lat']}x{ref.sizes['lon']}_to_ne30pg3_conserve.nc"
            )
            if not weights.exists():
                scrip = utils.write_latlon_scrip(
                    ref["lat"], ref["lon"], outdir / "src_scrip.nc"
                )
                utils.make_mesh_weights(scrip, MESH, weights)
            try:
                ux_ref = utils.latlon_to_mesh(ref, grid, "bilinear")
            except (TypeError, ValueError) as err:
                # uxarray 2025.6.0's bilinear remap fails on some grids (GPCP 2.5 deg):
                print(
                    f"{var} {pair}: uxarray bilinear failed ({err}); using inverse_distance_weighted"
                )
                ux_ref = utils.latlon_to_mesh(ref, grid, "inverse_distance_weighted")
            on_mesh = {
                "ux": ux_ref,
                "esmf": utils.apply_mesh_weights(ref, weights, grid),
            }
            for how, rmesh in on_mesh.items():
                rmesh = ux.UxDataArray(
                    rmesh.values, dims=["n_face"], uxgrid=grid, attrs={"units": units}
                )
                tmesh = ux.UxDataArray(
                    test.values, dims=["n_face"], uxgrid=grid, attrs={"units": units}
                )
                diff = tmesh - rmesh
                pct = ux.UxDataArray(
                    (diff / np.abs(rmesh) * 100).where(np.isfinite(diff / rmesh)),
                    uxgrid=grid,
                )
                kw = dict(
                    opts,
                    unstructured_plotting=True,
                    mesh_file=MESH,
                    wgt=xr.DataArray(area / area.sum(), dims=["n_face"]),
                    units=units,
                    var_name=var,
                    season="ANN",
                    obs_file=obs_file,
                    obs_var_name=obs_var,
                    tiString=f"{var} ANN ({rmesh_method(on_mesh[how])} of {label})",
                )
                wks = outdir / f"{var}_{pair}_{how}_ANN_LatLon_Mean.png"
                pf.plot_map_and_save(
                    None,
                    wks,
                    "native .092",
                    label,
                    [1, 5],
                    [1, 5] if pair != "obs" else [],
                    tmesh,
                    rmesh,
                    diff,
                    pct,
                    obs=(pair == "obs"),
                    **kw,
                )
                rows.append(
                    (
                        var,
                        pair,
                        how,
                        latlon_mean(ref),
                        mesh_mean(rmesh.values, area),
                        mesh_mean(test.values, area),
                        float(np.nanmax(np.abs(diff.values))),
                        int(np.isnan(rmesh.values).sum()),
                    )
                )
                print(f"{var:6s} {pair:5s} {how:4s} wrote {wks.name}", flush=True)
    with open(outdir / "summary.tsv", "w") as fh:
        fh.write(
            "var\tpair\tremap\tref mean (lat/lon)\tref mean (mesh)\ttest mean\tmax |diff|\tmissing faces\n"
        )
        for r in rows:
            fh.write(
                "\t".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in r)
                + "\n"
            )
    print((outdir / "summary.tsv").read_text())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(VARS))
