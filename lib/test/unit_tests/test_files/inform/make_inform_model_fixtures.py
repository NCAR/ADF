"""Cut the model-side INFORM fixtures in this directory out of the full data on
GLADE: CAM history files, ERA5, the dropsonde composite and one flight's
in-cloud aircraft data, all for one research flight per campaign.

    <campaign>_cam/<campaign>_fixture.cam.h0i.<date>-<sec>.nc
        The CAM history files load_cam_files picks for the flight, cut to the
        variables the INFORM pipeline reads, a small box around the flight's
        dropsondes and the times around the flight.  File names keep the
        date-seconds stamp that load_cam_files selects on.
    <campaign>_era5_fixture.nc
        load_era5_files output (T, RH) for the same window, around the sondes.
    <campaign>_sonde_fixture.nc
        That flight's dropsondes from the sonde composite.
    <campaign>_RF<nn>_incloud_fixture.nc
        That flight's in-cloud Open-Cell/Stratocumulus rows (a per-flight
        cloud regime file, as load_nc_cldrgme reads).
    inform_model_fixtures.json
        The flight window and box, read by the tests.

Run on GLADE (reads CAM from /glade/derecho/scratch/islas and ERA5 from RDA):

    python make_inform_model_fixtures.py
"""
import bisect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

OUT = Path(__file__).parent
sys.path.append(str(Path(__file__).parents[4]))  # ADF lib/
import cam_era5_functions as cfun  # noqa: E402

FLIGHT_DATA = Path("/glade/u/home/patnaude/inform/composited data")
SONDE_COMPOSITES = Path("/glade/derecho/scratch/richling/inform/ryan_composited_data")
ARCHIVE = Path("/glade/derecho/scratch/islas/archive")

FLIGHTS = {
    "SOCRATES": dict(
        rf=10,
        flight_file="SOCRATES_RF10_20260213.nc",
        cam_case="f.e30_cam6_4_120.FHIST_BGC.f09_f09_mg17.SOCRATES_nudgeUVTfull_withCOSP_tau6h.002",
        #Covers the sondes and the Open-Cell CAM clouds west of them:
        box=dict(lat=(-61.0, -46.0), lon=(134.5, 148.0)),
        #ERA5 is only collocated with the sondes:
        era5_box=dict(lat=(-61.0, -46.0), lon=(143.5, 148.0)),
    ),
    "CSET": dict(
        rf=12,
        flight_file="CSET_RF12_20260210.nc",
        cam_case="f.e30_cam6_4_120.FHIST_BGC.f09_f09_mg17.CSET_nudgeUVTfull_withCOSP_tau6h.001",
        #Covers the sondes and the Stratocumulus CAM clouds east of them:
        box=dict(lat=(22.0, 30.5), lon=(206.0, 234.5)),
        era5_box=dict(lat=(22.0, 30.5), lon=(206.0, 216.5)),
    ),
}

#Campaign-wide slices, exactly as inform_model_analysis uses them:
SLICES = {
    "SOCRATES": (slice(-62, -42), slice(135, 165)),
    "CSET": (slice(15, 45), slice(200, 240)),
}

#The raw CAM variables the pipeline reads (traced from load_cam_files ->
#compute_cam_aerosol_micphys_metrics / load_cam_ccfs / subset_cam_by_campaign):
CAM_VARS = [
    "ANRAIN", "AQRAIN", "AWNC", "CCN4", "CCN5", "CLDLIQ", "FREQL", "FREQR",
    "OMEGA", "PS", "Q", "T", "TS", "U", "V", "WP2_ZT_CLUBB", "bc_a1",
    "dgnd_a01", "dgnd_a02", "dgnd_a03", "dgnd_a04", "dst_a1", "dst_a3",
    "hyam", "hybm", "ncl_a1", "ncl_a2", "ncl_a3", "num_a1", "num_a2",
    "num_a3", "num_a4", "pom_a1", "pom_a4", "so4_a1", "so4_a2", "so4_a3",
    "soa_a1", "soa_a2",
]
SUFFIX = {"SOCRATES": "_130e_to_170e_35s_to_65s", "CSET": "_200e_to_245e_15n_to_45n"}
DIMS = {
    "SOCRATES": ("lat_35s_to_65s", "lon_130e_to_170e"),
    "CSET": ("lat_15n_to_45n", "lon_200e_to_245e"),
}
REGIMES = ["Open-Cell", "Stratocumulus"]
IN_CLOUD = ["In-Cloud Level FT", "In-Cloud Profiles"]


def _zlib(ds):
    return {v: {"zlib": True, "complevel": 4} for v in ds.data_vars
            if ds[v].dtype.kind not in "USOM"}


def _cam_files(cam_dir, tmin, tmax):
    """The files load_cam_files selects, by the same rule."""
    pat = re.compile(r".*(\d{4})-(\d{2})-(\d{2})-(\d{5})\.nc$")
    stamped = []
    for p in sorted(p for p in cam_dir.glob("*.nc") if "cam.h0" in p.name):
        m = pat.match(p.name)
        if m:
            y, mo, d, sod = map(int, m.groups())
            stamped.append((pd.Timestamp(y, mo, d) + pd.Timedelta(seconds=sod), p))
    times = [t for t, _ in stamped]
    pad = pd.Timedelta(minutes=60)
    i0 = bisect.bisect_left(times, tmin - pad)
    i1 = bisect.bisect_right(times, tmax + pad) - 1
    return [p for _, p in stamped[max(i0 - 1, 0):min(i1 + 1, len(stamped) - 1) + 1]]


def main():
    meta = {}
    for campaign, cfg in FLIGHTS.items():
        rf = cfg["rf"]
        lat0, lat1 = cfg["box"]["lat"]
        lon0, lon1 = cfg["box"]["lon"]

        #Dropsondes for this flight:
        sondes = xr.open_dataset(SONDE_COMPOSITES / f"{campaign}_sonde_data_composite.nc")
        keep = (sondes["RF"] == rf).values
        sondes = sondes.isel(index=np.flatnonzero(keep)).load()
        sondes.to_netcdf(OUT / f"{campaign}_sonde_fixture.nc", encoding=_zlib(sondes))
        tmin = pd.Timestamp(sondes["Time"].min().values).floor("s").to_pydatetime()
        tmax = pd.Timestamp(sondes["Time"].max().values).floor("s").to_pydatetime()
        print(f"{campaign} RF{rf:02d}: {int(keep.sum())} sonde rows, {tmin} -> {tmax}")

        #In-cloud aircraft rows of the same flight:
        with xr.open_dataset(FLIGHT_DATA / cfg["flight_file"]) as ds:
            rows = np.flatnonzero(ds["block_label"].isin(IN_CLOUD).values
                                  & ds["cloud_regime"].isin(REGIMES).values)
            air = ds.isel(index=rows).load()
        air_name = f"{campaign}_RF{rf:02d}_incloud_fixture.nc"
        air.to_netcdf(OUT / air_name, encoding=_zlib(air))
        print(f"  {air_name}: {len(rows)} rows")

        #CAM history files, cut down:
        cam_out = OUT / f"{campaign}_cam"
        cam_out.mkdir(exist_ok=True)
        for old in cam_out.glob("*.nc"):
            old.unlink()
        latd, lond = DIMS[campaign]
        names = [v if v in ("hyam", "hybm") else v + SUFFIX[campaign] for v in CAM_VARS]
        for path in _cam_files(ARCHIVE / cfg["cam_case"] / "atm" / "hist", tmin, tmax):
            with xr.open_dataset(path) as ds:
                small = ds[names].sel({latd: slice(lat0, lat1), lond: slice(lon0, lon1)})
                t = small.indexes["time"]
                lo = type(t[0])(*(tmin - pd.Timedelta(hours=1)).timetuple()[:6])
                hi = type(t[0])(*(tmax + pd.Timedelta(hours=1)).timetuple()[:6])
                inside = (t >= lo) & (t <= hi)
                #Keep at least one step so every selected file still opens:
                small = small.isel(time=np.flatnonzero(inside) if inside.any() else [0]).load()
            stamp = path.name.split(".cam.")[1]
            out = cam_out / f"{campaign}_fixture.cam.{stamp}"
            small.to_netcdf(out, encoding=_zlib(small))
            print(f"  {out.name}: {small.sizes}")

        #ERA5 for the window, then the box:
        lat_slice, lon_slice = SLICES[campaign]
        era = cfun.load_era5_files(tmin, tmax, lat_slice, lon_slice)
        #(load_era5_files sorts latitude ascending)
        (elat0, elat1), (elon0, elon1) = cfg["era5_box"]["lat"], cfg["era5_box"]["lon"]
        era = era[["T", "RH"]].sel(latitude=slice(elat0, elat1),
                                   longitude=slice(elon0, elon1)).load()
        era.to_netcdf(OUT / f"{campaign}_era5_fixture.nc", encoding=_zlib(era))
        print(f"  ERA5: {era.sizes}")

        meta[campaign] = dict(
            rf=rf, tmin=tmin.isoformat(), tmax=tmax.isoformat(),
            box=cfg["box"], era5_box=cfg["era5_box"], cam_case=cfg["cam_case"],
            flight_file=cfg["flight_file"], incloud_fixture=air_name,
        )

    (OUT / "inform_model_fixtures.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
