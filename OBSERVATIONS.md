# ADF observations manifest

Which observational data set each variable is compared against when
`compare_obs: true`, where those files come from, and what to watch out for
when reading the comparison.

The files live in the shared observations directory on GLADE,

    /glade/campaign/cgd/amp/amwg/ADF_obs

which is what `obs_data_loc` points at in the example configuration files.  The
mapping itself is in `lib/adf_variable_defaults.yaml`; this document describes
what is on the other end of it.  Raw data, the scripts that built the
climatologies, and per-source notes on how to update them are kept in
`/glade/campaign/cgd/cas/brianpm/observations/`, and each build script is also
copied next to the data it wrote in `ADF_obs`.

Everything in the table below was read from the files themselves.  Periods
marked † are not recorded in the file metadata and are taken from the file
name.

## Reading the comparison

**Reanalysis is not observation.**  Of the variables with a reference, about a
third use ERA5, ERA-Interim, MERRA-2 or CAMS.  For the upper-air state
(`T`, `U`, `Q`, `RELHUM`, `OMEGA`) that is the appropriate choice: reanalysis
assimilates the observations and is the best available estimate.  For derived
quantities such as `PBLH` and `CAPE`, and for anything cloud-related, a
reanalysis is a model with its own parameterizations, and the comparison is
model against model.  The source column says which is which.

**Satellite fields are not model fields.**  An instrument sees cloud from
above: it misses thin cloud, and cloud below other cloud.  The model field
counts everything in the column.  Scored against a CAM run with COSP enabled
(2003-2005), raw `CLDTOT` had the smallest mean bias of the three available
model fields (+3.6% against MODIS) and the worst pattern correlation (0.64),
while the simulator field `CLTMODIS` showed a genuine -16% bias at correlation
0.86.  The agreement in the mean was two errors cancelling.  **Where a run has
COSP output, compare the COSP variables.**

**The COSP mean quantities carry a cloud-fraction weighting.**  CAM writes
them multiplied by the relevant cloud fraction in percent, which only the
`long_name` records:

    LWPMODIS        MODIS Cloud Liquid Water Path*CLWMODIS
    PCTMODIS        MODIS Cloud Top Pressure*CLTMODIS
    MEANPTOP_ISCCP  Mean cloud top pressure*CLDTOT_ISCCP

Only the cloud fractions themselves are unweighted.  The observation files are
built with the same weighting, so the comparison is like for like; recover a
physical value as mean(X*f)/mean(f) on either side.  Comparing an unweighted
reference against these is wrong by a factor of the cloud fraction, which for
total optical thickness is about 70.

**Ocean-only references show land as missing.**  `TGCLDLWP` (MAC-LWP), `SST`
and `ICEFRAC` (HadISST), `LHFLX`, `SHFLX`, `QFLX` (OAFlux) and `U10` (CCMP)
have no land values at all; the coverage column gives the fraction of grid
points with data.

**Ice water path is uncertain at the factor-of-two level.**  Observational IWP
products differ by up to a factor of five (Duncan and Eriksson 2018,
doi:10.5194/acp-18-11205-2018), and the radar records that are the recommended
best estimate measure *total* frozen water including snow, which is not the
cloud-ice-only quantity CAM reports.  `TGCLDIWP` uses MODIS because an optical
retrieval excludes precipitating snow and so matches in definition.  Treat a
bias smaller than a factor of two as not established.

**Sign and unit conventions that are applied in the configuration.**  CERES
reports surface net longwave positive downward while CAM's `FLNS` is positive
upward, so `FLNS` and `FLNSC` carry `obs_scale_factor: -1`.  Cloud fractions
are a fraction in CAM and a percentage in the satellite products, hence
`scale_factor: 100` on the model side of `CLDTOT` and friends.

## Alternatives

Several variables have more than one reference in `ADF_obs`; the alternatives
are named in comments on the relevant block of `adf_variable_defaults.yaml`.
The main ones:

| variable | default | alternatives |
|---|---|---|
| `PRECT` | GPCP v2.3 | GPCP v3.3 (14% wetter in the global mean), GPCP v2.1 |
| `TGCLDLWP` | MAC-LWP (ocean, microwave) | MODIS and CLARA-A3 (global, optical), ERA5 |
| `TMQ` | WV_cci/COMBI | NVAP-M, RSS TPW v7, ERA-Interim |
| `U10` | CCMP v3.1 | OAFlux |
| `LHFLX` | OAFlux | ERA-Interim |

**A level out of a three-dimensional reanalysis is taken as it is read.**  CAM
writes `U200` as a two-dimensional field, already on the 200 hPa surface, and the
ADF compares fields of the same shape.  Rather than stage a file per level, the
variable defaults name the level to take: `obs_lev: 200` on the `U200` entry
reads the 200 hPa surface out of `U_ERA5_monthly_climo_197901-202112.nc`, and
`U850`, `V200` and `V850` read theirs out of the same two files.  It is the same
numbers as comparing the 3-D `U` at 200 hPa through `plot_press_levels`, not a
second data set.  Any other surface is one line in the variable defaults away.

`U200_ERA5_monthly_climo_197901-202112.nc` and its 1-degree twin, written by
`make_u200_era5_climo.py` before `obs_lev` existed, hold exactly the same values
and are kept for configurations that point at them.

## The files

| file | source | grid | period | coverage | variables served |
|---|---|---|---|---|---|
| `BURDENBC_MERRA2_monthly_climo_1degree_200001-202506.nc` | MERRA2 | 180x360 (1 deg) | 200001–202506 † | 100% of grid points valid | `BURDENBC` |
| `BURDENDUST_CAMS_monthly_climo_1degree_200301-202412.nc` | CAMS | 180x360 (1 deg) | 200301–202412 † | 100% of grid points valid | `BURDENDUST` |
| `BURDENSEASALT_CAMS_monthly_climo_1degree_200301-202412.nc` | CAMS | 180x360 (1 deg) | 200301–202412 † | 100% of grid points valid | `BURDENSEASALT` |
| `BURDENSO4_CAMS_monthly_climo_1degree_200301-202412.nc` | CAMS | 180x360 (1 deg) | 200301–202412 † | 100% of grid points valid | `BURDENSO4` |
| `CALIPSO_GOCCP_3.1.2_climo_200606-202012.nc` | CALIPSO | 192x288 (0.942408 deg) | 200606–202012 † | 92% of grid points valid | `CLDHGH_CAL`, `CLDLOW_CAL`, `CLDMED_CAL`, `CLDTOT_CAL`, `CLD_CAL` |
| `CAPE_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `CAPE` |
| `CERES_EBAF_Ed4.1_2001-2020.nc` | CERES_EBAF_Ed4.1 | 180x360 (1 deg) | 2001–2020 † | 100% of grid points valid | `FLDS`, `FLNS`, `FLNSC`, `FLNT`, `FLNTC`, `FLUT`, `FLUTC`, `FSDS`, `FSDSC`, `FSNS`, `FSNSC`, `FSNT`, `FSNTOA`, `FSUTOA`, `LWCF`, `SOLIN`, `SWCF` |
| `CLDICE_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `CLDICE` |
| `CLDLIQ_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `CLDLIQ` |
| `COSP_MODIS_MCD06COSP_monthly_climo_200301-202512.nc` | MODIS-COSP | 180x360 (1 deg) | 2003-01 to 2025-12 | global land and ocean, daytime retrievals | `CLDHGH`, `CLDLOW`, `CLDMED`, `CLDTOT`, `CLHMODIS`, `CLIMODIS`, `CLLMODIS`, `CLMMODIS`, `CLTMODIS`, `CLWMODIS`, `IWPMODIS`, `LWPMODIS`, `PCTMODIS`, `REFFCLIMODIS`, `REFFCLWMODIS`, `TAUIMODIS`, `TAUTMODIS`, `TAUWMODIS` |
| `CS_qualitative_clusters.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `ISCCP_euclidean_centers` |
| `ERA5_LSM_1deg_conservativeregrid.nc` | ERA5 | 180x360 (1 deg) | — | 100% of grid points valid | `LANDFRAC` |
| `ERAI_all_climo.nc` | ERAI | 121x240 (1.5 deg) | — | 100% of grid points valid | `RELHUM`, `TS` |
| `HadISST_PD_monthly_climo_199901-200812.nc` | HadISST_OIv2_PD | 180x360 (1 deg) | 1999-01 to 2008-12 | 48% of grid points valid | `ICEFRAC`, `SST` |
| `ISCCP-H_monthly_climo_198307-201706.nc` | ISCCP-H | 180x360 (1 deg) | 1983-07 to 2017-06 | global | `CLDTOT_ISCCP`, `MEANPTOP_ISCCP`, `MEANTAU_ISCCP` |
| `ISCCP_emd-means_n_init5_centers_1.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `ISCCP_emd_centers` |
| `ISCCP_obs_data.nc` | ISCCP | 180x360 (1 deg) | — | 86% of grid points valid | `FISCCP1_COSP` |
| `MISR_6C_weather_state_centers.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `MISR_euclidean_centers` |
| `MISR_emd-means_n_init5_centers_1.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `MISR_emd_centers` |
| `MISR_obs_data.nc` | MISR | 180x360 (1 deg) | — | 49% of grid points valid | `CLD_MISR` |
| `MOD08_M3_192x288_AOD_2001-2020_climo.nc` | MODIS | 192x288 (0.942408 deg) | 2001–2020 † | 73% of grid points valid | `AODVIS`, `AODVISdn` |
| `MODIS_6C_weather_state_centers.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `MODIS_euclidean_centers` |
| `MODIS_emd-means_n_init5_centers_1.npy` | — | — | — | cluster centres read by `cloud_regime_analysis.py` | `MODIS_emd_centers` |
| `MODIS_obs_data.nc` | MODIS | 180x360 (1 deg) | — | 89% of grid points valid | `CLMODIS` |
| `OAFlux_monthly_climo_200101-201812.nc` | OAFlux | 180x360 (1 deg) | 2001-01 to 2018-12 | ice-free global ocean only; land and sea ice are missing | `LHFLX`, `QFLX`, `SHFLX` |
| `OMEGA_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `OMEGA` |
| `PBLH_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `PBLH` |
| `PRECT_GPCP_v2.3_monthly_climo_198301-202512.nc` | GPCP_v2.3 | 72x144 (2.5 deg) | 1983-01 to 2025-12 | 100% of grid points valid | `PRECT` |
| `PS_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `PS` |
| `PSL_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `PSL` |
| `Q_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `Q` |
| `T_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `T` |
| `TAUX_ERA5_monthly_climo_197901-202212.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2022-12 | 100% of grid points valid | `TAUX` |
| `TAUY_ERA5_monthly_climo_197901-202212.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2022-12 | 100% of grid points valid | `TAUY` |
| `TEM_ERA5.nc` | ERA5 | — | — | 100% of grid points valid | `EPFY`, `EPFZ`, `PSITEM`, `UTENDEPFD`, `UTENDVTEM`, `UTENDWTEM`, `UZM`, `VTEM`, `VZM`, `WTEM` |
| `TGCLDIWP_MODIS-COSP_gridmean_monthly_climo_200301-202512.nc` | MODIS-COSP | 180x360 (1 deg) | 2003-01 to 2025-12 | global land and ocean, daytime optical retrievals | `TGCLDIWP` |
| `TGCLDLWP_MAC-LWP_monthly_climo_198801-201612.nc` | MAC-LWP | 180x360 (1 deg) | 1988-01 to 2016-12 | global ocean only; land is missing | `TGCLDLWP` |
| `TMQ_WV_cci_COMBI_monthly_climo_200301-201712.nc` | WV_cci_COMBI | 360x720 (0.5 deg) | 2003-01 to 2017-12 | global land, coast, sea ice (MERIS/MODIS near-infrared) an | `TMQ` |
| `TREFHT_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `TREFHT` |
| `U10_CCMP_v3.1_monthly_climo_199301-202512.nc` | CCMP_v3.1 | 720x1440 (0.25 deg) | 1993-01 to 2025-12 | ice-free global ocean only; land is missing | `U10` |
| `U_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `U`, `U200`, `U850` |
| `U200_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `U200` |
| `V_ERA5_monthly_climo_197901-202112.nc` | ERA5 | 721x1440 (0.25 deg) | 1979-01 to 2021-12 | 100% of grid points valid | `V`, `V200`, `V850` |

## Keeping this accurate

`lib/test/unit_tests/test_obs_entries.py` checks that the configuration is
internally consistent: that every entry with an `obs_file` also says which
variable and which data set, that no leftover `obs_scale_factor` sits on a
variable with no observations, that a file is not given two different source
names, and that an `obs_name` contains no path separator (it is pasted into the
regridded file name).  It cannot check the files themselves, since CI has no
access to GLADE.

When adding a data set: put the raw data and the build script somewhere durable
with a README recording where it came from and how to update it, write the
climatology with `time` = 1..12 and provenance in the global attributes, copy
it into `ADF_obs`, then add the entry here and in
`lib/adf_variable_defaults.yaml`.
