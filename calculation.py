#!/usr/bin/env python3
"""
calculations.py – Compare DDS_BoatTracker, Smartphone and Vakaros GPS logs
and create:

    • 8 PNG figures (3 × scatter, 3 × error-vs-time, 2 × all-pairs)
    • results/gps_tracks_darkmap.html – dark interactive Folium map

Dependencies
    pip install pandas numpy matplotlib folium
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import math, re
from pathlib import Path

# ───────────────────────────────────────────────────────────────────
# 1)  FILE PATHS
# ───────────────────────────────────────────────────────────────────
PATH_DDS   = "data/boat1_20250528-2123.csv"
PATH_PHONE = "data/Smartphone-2025-05-28_15-03-25_wtsutc.csv"
PATH_VAK   = "data/MyVakaros 5-28-2025.csv"

# ───────────────────────────────────────────────────────────────────
# 2)  COLUMN DEFINITIONS
# ───────────────────────────────────────────────────────────────────
# DDS logger
COL_DDS_TIME = "ISODateTimeUTC"
COL_DDS_LAT  = "Lat"
COL_DDS_LON  = "Lon"

# Smartphone log
COL_PHONE_TIME    = "ISODateTimeUTC"
COL_PHONE_LAT     = "latitude"
COL_PHONE_LON     = "longitude"
COL_PHONE_ELAPSED = "seconds_elapsed"     # comment out if not present

# Vakaros logger
COL_VAK_TIME = "timestamp"
COL_VAK_LAT  = "latitude"
COL_VAK_LON  = "longitude"

# ───────────────────────────────────────────────────────────────────
# 3)  TIME WINDOW (UTC) TO ANALYSE
# ───────────────────────────────────────────────────────────────────
START = "2025-05-28 15:20:00"
END   = "2025-05-28 15:37:00"

# ───────────────────────────────────────────────────────────────────
# 4)  HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────
def parse_time_from_name(fname: str) -> pd.Timestamp | None:
    """Extract start time from filename (two possible patterns)."""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})', fname)
    if m:
        return pd.Timestamp("-".join(m.groups()[:3]) + " " +
                            ":".join(m.groups()[3:]), tz="UTC")
    m = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
    if m:
        y, mo, d, H, M = m.groups()
        return pd.Timestamp(f"{y}-{mo}-{d} {H}:{M}:00", tz="UTC")
    return None


def load_latlon(path: str,
                tcol: str,
                latcol: str,
                loncol: str,
                relsec: str | None = None) -> pd.DataFrame:
    """
    Read a CSV and return a tidy DataFrame with:
        • time_utc  (datetime64[ns, UTC])
        • lat, lon  (float)
    If *relsec* is given, absolute time = file-name timestamp + relsec seconds.
    """
    df = pd.read_csv(path)

    # build absolute time
    if relsec and relsec in df.columns:
        start_ts = parse_time_from_name(Path(path).name)
        if start_ts is None:
            raise ValueError(f"Start time not found in filename: {path}")
        df["time_utc"] = start_ts + pd.to_timedelta(df[relsec], unit="s")
    else:
        df["time_utc"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")

    # numeric columns
    df["lat"] = pd.to_numeric(df[latcol], errors="coerce")
    df["lon"] = pd.to_numeric(df[loncol], errors="coerce")

    return (df[["time_utc", "lat", "lon"]]
            .dropna()
            .sort_values("time_utc"))


def nearest_join(base: pd.DataFrame,
                 ref: pd.DataFrame,
                 tol: pd.Timedelta = pd.Timedelta("150ms"),
                 suffix: str = "_ref") -> pd.DataFrame:
    """As-of join with ±tol tolerance and keep only rows with a match."""
    out = pd.merge_asof(base, ref,
                        on="time_utc",
                        direction="nearest",
                        tolerance=tol,
                        suffixes=("", suffix))
    return out.dropna(subset=[f"lat{suffix}", f"lon{suffix}"])


def flat_residuals(df: pd.DataFrame, suffix: str):
    """
    Convert lat/lon differences to East/North (metres) using a simple
    flat-Earth approximation (good for <~100 km spans).
    """
    dlat = df["lat"] - df[f"lat{suffix}"]
    dlon = df["lon"] - df[f"lon{suffix}"]
    lat0 = df["lat"].mean()
    mdeg = 111_320                        # metres per ° latitude (≈ constant)
    dE = dlon * mdeg * math.cos(math.radians(lat0))
    dN = dlat * mdeg
    return dE.to_numpy(), dN.to_numpy()


def metrics(dE: np.ndarray, dN: np.ndarray):
    """Return bias_E, bias_N, |bias| and RMSE (all metres)."""
    biasE, biasN = dE.mean(), dN.mean()
    return (biasE,
            biasN,
            math.hypot(biasE, biasN),
            math.sqrt(((dE - biasE) ** 2 + (dN - biasN) ** 2).mean()))

# ───────────────────────────────────────────────────────────────────
# 5)  LOAD CSV FILES & CUT TO COMMON TIME WINDOW
# ───────────────────────────────────────────────────────────────────
dds   = load_latlon(PATH_DDS,   COL_DDS_TIME,   COL_DDS_LAT,   COL_DDS_LON)
phone = load_latlon(PATH_PHONE, COL_PHONE_TIME, COL_PHONE_LAT, COL_PHONE_LON,
                    relsec=COL_PHONE_ELAPSED)
vak   = load_latlon(PATH_VAK,   COL_VAK_TIME,   COL_VAK_LAT,   COL_VAK_LON)

START_ts = pd.Timestamp(START, tz="UTC") if START else max(df.time_utc.min() for df in (dds, phone, vak))
END_ts   = pd.Timestamp(END,   tz="UTC") if END   else min(df.time_utc.max() for df in (dds, phone, vak))

dds, phone, vak = (df.query("@START_ts <= time_utc <= @END_ts")
                   for df in (dds, phone, vak))

# ───────────────────────────────────────────────────────────────────
# 6)  MATCH DATA PAIRS & CALCULATE METRICS
# ───────────────────────────────────────────────────────────────────
dds_phone = nearest_join(dds,   phone, suffix="_phone")
dds_vak   = nearest_join(dds,   vak,   suffix="_vak")
phone_vak = nearest_join(phone, vak,   suffix="_vak")

# ── convert to residuals in metres ────────────────────────────────
dE_dp, dN_dp = flat_residuals(dds_phone, "_phone")
dE_dv, dN_dv = flat_residuals(dds_vak,   "_vak")
dE_pv, dN_pv = flat_residuals(phone_vak, "_vak")

# time-aligned error magnitude series
err_dp, err_dv, err_pv = map(np.hypot,
                             (dE_dp, dE_dv, dE_pv),
                             (dN_dp, dN_dv, dN_pv))
time_dp, time_dv, time_pv = (dds_phone.time_utc,
                             dds_vak.time_utc,
                             phone_vak.time_utc)

# human-readable summary table
summary = pd.DataFrame(
    [("DDS_BoatTracker vs Smartphone",) + metrics(dE_dp, dN_dp),
     ("DDS_BoatTracker vs Vakaros",)    + metrics(dE_dv, dN_dv),
     ("Smartphone vs Vakaros",)         + metrics(dE_pv, dN_pv)],
    columns=["Pair",
             "Bias_E_m", "Bias_N_m",
             "Bias_mag_m", "RMSE_m"]
)

print("Analysis window:", START_ts, "→", END_ts)
pd.options.display.float_format = "{:,.3f}".format
print("\nResidual statistics (metres):")
print(summary.to_string(index=False))

# ───────────────────────────────────────────────────────────────────
# 7)  STATIC PLOT CREATION
# ───────────────────────────────────────────────────────────────────
OUT = Path("results"); OUT.mkdir(exist_ok=True)
DPI = 300
C   = {"dp": "#E2007A", "dv": "#0077B6", "pv": "#FFA701"}   # colour palette

def save(fig: plt.Figure, fname: str):
    """Tight-layout & save then close."""
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=DPI)
    plt.close(fig)


def scatter(dE, dN, title, fname, color):
    """Residual cloud for one pair."""
    fig = plt.figure()
    plt.scatter(dE, dN, s=3, color=color, alpha=0.6)
    plt.axhline(0, lw=0.5, color="#888")
    plt.axvline(0, lw=0.5, color="#888")
    plt.title(title)
    plt.xlabel("East error [m]")
    plt.ylabel("North error [m]")
    plt.gca().set_aspect('equal')
    plt.grid(ls="--", lw=0.3)
    save(fig, fname)


def line(t, e, title, fname, color):
    """Error-vs-time line for one pair."""
    fig = plt.figure()
    plt.plot(t, e, color=color, lw=0.8)
    plt.title(title)
    plt.xlabel("UTC time")
    plt.ylabel("Horizontal error [m]")
    plt.grid(ls="--", lw=0.3)
    fig.autofmt_xdate()
    save(fig, fname)


# individual plots
scatter(dE_dp, dN_dp, "DDS_BoatTracker vs Smartphone residuals",
        "scatter_DDS_vs_Smartphone.png", C["dp"])
scatter(dE_dv, dN_dv, "DDS_BoatTracker vs Vakaros residuals",
        "scatter_DDS_vs_Vakaros.png",    C["dv"])
scatter(dE_pv, dN_pv, "Smartphone vs Vakaros residuals",
        "scatter_Smartphone_vs_Vakaros.png", C["pv"])

line(time_dp, err_dp,
     "Error vs time: DDS_BoatTracker → Smartphone",
     "error_time_DDS_vs_Smartphone.png", C["dp"])
line(time_dv, err_dv,
     "Error vs time: DDS_BoatTracker → Vakaros",
     "error_time_DDS_vs_Vakaros.png",    C["dv"])
line(time_pv, err_pv,
     "Error vs time: Smartphone → Vakaros",
     "error_time_Smartphone_vs_Vakaros.png", C["pv"])

# all-pairs error-vs-time
fig = plt.figure()
plt.plot(time_dp, err_dp, label="DDS → Smartphone",      color=C["dp"], lw=0.9)
plt.plot(time_dv, err_dv, label="DDS → Vakaros",         color=C["dv"], lw=0.9)
plt.plot(time_pv, err_pv, label="Smartphone → Vakaros",  color=C["pv"], lw=0.9)
plt.legend()
plt.title("Error vs time – all pairs")
plt.xlabel("UTC time")
plt.ylabel("Horizontal error [m]")
plt.grid(ls="--", lw=0.3)
fig.autofmt_xdate()
save(fig, "error_time_ALL_pairs.png")

# all-pairs residual cloud
fig = plt.figure()
plt.scatter(dE_dp, dN_dp, s=3, alpha=0.55, color=C["dp"],
            label="DDS → Smartphone")
plt.scatter(dE_dv, dN_dv, s=3, alpha=0.55, color=C["dv"],
            label="DDS → Vakaros")
plt.scatter(dE_pv, dN_pv, s=3, alpha=0.55, color=C["pv"],
            label="Smartphone → Vakaros")
plt.axhline(0, lw=0.5, color="#888")
plt.axvline(0, lw=0.5, color="#888")
plt.title("Residuals (East/North) – all pairs")
plt.xlabel("East error [m]")
plt.ylabel("North error [m]")
plt.legend(markerscale=3)
plt.gca().set_aspect('equal')
plt.grid(ls="--", lw=0.3)
save(fig, "scatter_residuals_ALL_pairs.png")

# ───────────────────────────────────────────────────────────────────
# 8)  INTERACTIVE DARK MAP
# ───────────────────────────────────────────────────────────────────
center_lat = pd.concat([dds.lat, phone.lat, vak.lat]).mean()
center_lon = pd.concat([dds.lon, phone.lon, vak.lon]).mean()

fmap = folium.Map(location=[center_lat, center_lon],
                  tiles="CartoDB dark_matter",
                  zoom_start=13,
                  control_scale=True)

folium.PolyLine(dds[["lat", "lon"]].values,
                color=C["dv"], weight=3, tooltip="DDS_BoatTracker").add_to(fmap)
folium.PolyLine(phone[["lat", "lon"]].values,
                color=C["dp"], weight=3, tooltip="Smartphone").add_to(fmap)
folium.PolyLine(vak[["lat", "lon"]].values,
                color=C["pv"], weight=3, tooltip="Vakaros").add_to(fmap)

html_path = OUT / "gps_tracks_darkmap.html"
fmap.save(html_path)

print("\nPNG figures and dark map saved into :", OUT.resolve())
