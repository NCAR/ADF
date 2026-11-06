# Prototype: lat/lon data on the native mesh

On `ldf-native-grid-merge-main` (first written on branch `ldf-native-grid-main-merge-update`, `d978565`).

## Why

With `unstructured_plotting: true` the ADF plots a native case on its mesh. When the
reference is on a lat/lon grid (an f09 case, or any observations), the map, polar and vector
scripts skip the comparison ("mixed grids"; LDF-merge runs N09a, N09b, L14). This prototype
puts the lat/lon side onto the test case's mesh so the usual four-panel maps can be drawn there.

## What is here

`lib/adf_utils.py`:

| Function | What it does |
| --- | --- |
| `latlon_to_mesh(data, mesh, method="bilinear")` | uxarray remap (`bilinear`, `inverse_distance_weighted`, `nearest_neighbor`) from any lat/lon grid onto the mesh. No weights file. Not conservative. |
| `write_latlon_scrip(lat, lon, path)` | SCRIP file for a regular lat/lon grid, for ESMF. |
| `make_mesh_weights(src_scrip, mesh_file, weights_file, method="conserve")` | Runs `ESMF_RegridWeightGen` (in the `adf_v1.0.5` env) for lat/lon → mesh weights. |
| `apply_mesh_weights(data, weights_file, mesh)` | Applies the weights as a sparse matrix; missing source cells are left out and the rest renormalized. |

`scripts/prototypes/latlon_on_mesh_demo.py` draws the ADF's own `plot_map_and_save` four-panel
maps on the mesh for PRECT, SWCF and LWCF (ne30pg3, years 1–5, inputs from the LDF-merge runs
N04 and S01), each with both remaps:

| Pair | Test | Reference |
| --- | --- | --- |
| `same` | native .092 | f09 remap of .092 (should be close to zero) |
| `base` | native .092 | f09 .091 (what N09b would plot) |
| `obs` | native .092 | GPCP v2.3 / CERES-EBAF Ed4.1 (what N09a would plot) |

Output (2026-10-26): `/glade/derecho/scratch/richling/adf-output/prototypes/latlon_on_mesh/`,
18 PNGs and `summary.tsv`. Unit tests: `lib/test/unit_tests/test_latlon_to_mesh.py`.

## Results

Global means of the reference before (cos-lat weighted on its grid) and after (face-area
weighted on the mesh):

| Var | Reference | lat/lon | uxarray | ESMF conserve |
| --- | --- | --- | --- | --- |
| PRECT (mm/day) | f09 .091 | 3.10282 | 3.10275 | 3.10294 |
| PRECT | GPCP 2.5° | 2.68886 | 2.68835 (IDW) | 2.68895 |
| SWCF (W/m²) | f09 .091 | −43.7256 | −43.7295 | −43.7269 |
| SWCF | CERES 1° | −45.2536 | −45.2506 | −45.2536 |
| LWCF (W/m²) | CERES 1° | 25.7741 | 25.7728 | 25.7742 |

- Both remaps keep global means to 1e-4 or better. ESMF conservative is closest, as expected.
- Native vs f09 .091 SWCF: RMSE 3.19 W/m². That's what N09b would show instead of skipping.
- The `same` pair is not zero. The f09 file is a bilinear remap of the native field (ncremap),
  so going back cannot restore grid-scale detail. Differences are largest at sharp gradients
  such as coasts and the ITCZ edges, so a mesh comparison against a lat/lon case is limited by
  the lat/lon side's resolution.
- **uxarray 2025.6.0 bug:** `remap.bilinear` fails on the GPCP 2.5° grid (`_barycentric_weights`,
  "only length-1 arrays can be converted"). The demo falls back to
  `inverse_distance_weighted` there. Report upstream, or prefer ESMF for obs.

## Next steps (for the follow-up PR)

1. **Regrid step.** For `unstructured_plotting` with a native test case and a lat/lon
   reference, write the reference onto the test mesh as an `ncol` regridded file
   (`…_regridded_YYYY-YYYY.nc`), where the step skips today. Plot scripts then see two mesh
   fields and need no change, apart from removing the mixed-grid skips (map, vector, polar).
2. **Choice of remap.** A config key, e.g. `mesh_regrid_method: conserve | bilinear`, defaulting
   to ESMF conservative (keeps area means; needs weights per reference grid, cached under
   `cam_regrid_loc`). Keep uxarray as the no-ESMF fallback.
3. **Land.** Remap with a land mask (landfrac on both sides) so coastal cells don't mix in ocean,
   as in the land branch's note 7b.
4. **Zonal and meridional means.** These already work on the mesh, single-case today; with the
   reference on the mesh they become real differences.
5. **Tests.** N09a, N09b and L14 move from "skips" to "plots". Add a check that the remapped
   reference's global mean matches the lat/lon one, and a native-vs-f09 same-case run as a
   sanity bound.
6. **Science question for AMWG/LMWG:** compare on the mesh (native resolution) or on the
   lat/lon grid (today's gridded runs)? The numbers in plot headers differ between the two.
