#!/usr/bin/env python3
"""
Data Validation dashboard for the **Data-Driven Sailing** system

• Loads three GPS logs (DDS BoatTracker, Smartphone, Vakaros)  
• Lets you pick a 5-minute-step time window & outlier cutoff  
• Shows:
    – Auto-zoom dark map  
    – Residual statistics table  
    – Three residual clouds  
    – Three error-vs-time plots (uniform Y-axis)  
    – Two combined plots (equal height)

Run:
    streamlit run app.py

Requires:
    pip install streamlit pandas numpy matplotlib folium streamlit-folium
"""
# ───────────────────────────────────────────────────────────────────
# 1) IMPORTS & STREAMLIT CONFIG
# ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd, numpy as np, math, re, datetime as dt
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Data Validation of the DataDrivenSailing System",
    layout="wide",
    page_icon=":world_map:"
)

# ───────────────────────────────────────────────────────────────────
# 2) DATA FILES, COLOURS & SCHEMA
# ───────────────────────────────────────────────────────────────────
COLORS = {"dp": "#E2007A",  # DDS → Smartphone
          "dv": "#0077B6",  # DDS → Vakaros
          "pv": "#FFA701"}  # Smartphone → Vakaros

FILES = dict(
    dds   = "data/boat1_20250528-2123.csv",
    phone = "data/Smartphone-2025-05-28_15-03-25_wtsutc.csv",
    vak   = "data/MyVakaros 5-28-2025.csv",
)

SCHEMA = dict(    # column names for each logger
    dds   = dict(time="ISODateTimeUTC", lat="Lat",       lon="Lon",        rel=None),
    phone = dict(time="ISODateTimeUTC", lat="latitude",  lon="longitude",  rel="seconds_elapsed"),
    vak   = dict(time="timestamp",      lat="latitude",  lon="longitude",  rel=None),
)

# ───────────────────────────────────────────────────────────────────
# 3) HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────────
def parse_start_from_name(fname: str) -> pd.Timestamp | None:
    """Try two filename patterns to get an absolute start‐timestamp."""
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})[_-](\d{2})-(\d{2})-(\d{2})', fname)
    if m:
        return pd.Timestamp("-".join(m.groups()[:3]) + " "
                            + ":".join(m.groups()[3:]), tz="UTC")
    m = re.search(r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})', fname)
    if m:
        y, mo, d, H, M = m.groups()
        return pd.Timestamp(f"{y}-{mo}-{d} {H}:{M}:00", tz="UTC")
    return None


@st.cache_data
def load_track(path: str, spec: dict) -> pd.DataFrame:
    """
    Load one CSV → tidy DataFrame with UTC time & numeric lat/lon.
    If a *relative seconds* column is present, reconstruct absolute time
    from the start time encoded in the filename.
    """
    df = pd.read_csv(path)

    # absolute timestamp
    if spec["rel"] and spec["rel"] in df.columns:
        anchor = parse_start_from_name(path)
        if anchor is None:
            raise ValueError(f"Cannot parse start time from {path}")
        df["time_utc"] = anchor + pd.to_timedelta(df[spec["rel"]], unit="s")
    else:
        df["time_utc"] = pd.to_datetime(df[spec["time"]], utc=True, errors="coerce")

    # numeric columns
    df["lat"] = pd.to_numeric(df[spec["lat"]], errors="coerce")
    df["lon"] = pd.to_numeric(df[spec["lon"]], errors="coerce")

    df = df.dropna(subset=["time_utc", "lat", "lon"]).sort_values("time_utc")
    # naive helper for slider filtering
    df["t_naive"] = df.time_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def nearest_join(a: pd.DataFrame, b: pd.DataFrame, suf: str):
    """Time-nearest join (±150 ms) and drop unmatched rows."""
    return (pd.merge_asof(a, b,
                          on="time_utc",
                          direction="nearest",
                          tolerance=pd.Timedelta("150ms"),
                          suffixes=("", suf))
            .dropna(subset=[f"lat{suf}", f"lon{suf}"]))


def flat_residuals(df: pd.DataFrame, suf: str):
    """Convert lat/lon differences to East / North metres (flat-Earth)."""
    dlat = df["lat"] - df[f"lat{suf}"]
    dlon = df["lon"] - df[f"lon{suf}"]
    lat0 = df["lat"].mean()
    mdeg = 111_320
    dE = dlon * mdeg * math.cos(math.radians(lat0))
    dN = dlat * mdeg
    return dE, dN


def residual_metrics(dE: np.ndarray, dN: np.ndarray) -> pd.Series:
    """Return bias_E, bias_N, |bias| and RMSE in one Series."""
    bE, bN = dE.mean(), dN.mean()
    return pd.Series(dict(
        Bias_E=bE,
        Bias_N=bN,
        Bias_mag=math.hypot(bE, bN),
        RMSE=math.sqrt(((dE - bE) ** 2 + (dN - bN) ** 2).mean())
    ))

# ───────────────────────────────────────────────────────────────────
# 4) LOAD ALL THREE DATASETS
# ───────────────────────────────────────────────────────────────────
dds, phone, vak = [load_track(FILES[k], SCHEMA[k]) for k in ("dds", "phone", "vak")]

# ───────────────────────────────────────────────────────────────────
# 5) SIDEBAR – CONTROLS (logo, window slider, outlier slider)
# ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("dds.png", width=260)                          # logo
    st.header("Window (5-min steps)")

    # global min / max time for slider
    tmin = min(df.t_naive.min() for df in (dds, phone, vak)).to_pydatetime()
    tmax = max(df.t_naive.max() for df in (dds, phone, vak)).to_pydatetime()

    # time-window slider
    start_naive, end_naive = st.slider(
        "Start / End",
        min_value=tmin, max_value=tmax,
        value=(dt.datetime(2025, 5, 28, 15, 20),
               dt.datetime(2025, 5, 28, 15, 37)),
        step=dt.timedelta(minutes=5),
        format="YYYY-MM-DD  HH:mm"
    )

    st.markdown("---")                                       # separator

    # outlier threshold slider
    outlier = st.slider(
        "Outlier cutoff (horizontal error [m])",
        0.5, 20.0, 20.0, 0.5
    )

# cut tracks to user-selected window
window_filter = lambda df: df[(df.t_naive >= start_naive) & (df.t_naive <= end_naive)]
dds, phone, vak = map(window_filter, (dds, phone, vak))

# ───────────────────────────────────────────────────────────────────
# 6) BUILD PAIRS & CALCULATE RESIDUALS
# ───────────────────────────────────────────────────────────────────
pairs = dict(
    dp=dict(df=nearest_join(dds,   phone, "_p"),
            suf="_p", lbl="DDS → Smartphone",           col=COLORS["dp"]),
    dv=dict(df=nearest_join(dds,   vak,   "_v"),
            suf="_v", lbl="DDS → Vakaros",              col=COLORS["dv"]),
    pv=dict(df=nearest_join(phone, vak,   "_v"),
            suf="_v", lbl="Smartphone → Vakaros",       col=COLORS["pv"]),
)

for p in pairs.values():
    # convert to metres
    dE, dN = flat_residuals(p["df"], p["suf"])
    err = np.hypot(dE, dN)

    # apply outlier mask
    keep = err <= outlier
    p["dE"], p["dN"], p["err"] = dE[keep], dN[keep], err[keep]
    p["time"] = p["df"].time_utc.to_numpy()[keep]
    p["stats"] = residual_metrics(p["dE"], p["dN"])

# common plot limits
lim = max(max(abs(p["dE"]).max() if len(p["dE"]) else 0,
              abs(p["dN"]).max() if len(p["dN"]) else 0)
          for p in pairs.values()) * 1.05 or 1
err_lim = max((p["err"].max() if len(p["err"]) else 0)
              for p in pairs.values()) * 1.05 or 1

# ───────────────────────────────────────────────────────────────────
# 7) PAGE LAYOUT – TITLE & MAP
# ───────────────────────────────────────────────────────────────────
st.title("Data Validation")
st.caption(f"Window: **{start_naive} → {end_naive}**")

# --- auto-zoom dark map ---
m = folium.Map(tiles="CartoDB dark_matter", control_scale=True)

folium.PolyLine(dds[["lat", "lon"]].values,
                color=COLORS["dv"], weight=3,
                tooltip="DDS_BoatTracker").add_to(m)
folium.PolyLine(phone[["lat", "lon"]].values,
                color=COLORS["dp"], weight=3,
                tooltip="Smartphone").add_to(m)
folium.PolyLine(vak[["lat", "lon"]].values,
                color=COLORS["pv"], weight=3,
                tooltip="Vakaros").add_to(m)

# zoom to bounds of all visible points
all_lat = pd.concat([dds.lat, phone.lat, vak.lat])
all_lon = pd.concat([dds.lon, phone.lon, vak.lon])
m.fit_bounds([[all_lat.min(), all_lon.min()],
              [all_lat.max(), all_lon.max()]])

st_folium(m, width=1100, height=550)

# ───────────────────────────────────────────────────────────────────
# 8) TABLE & INDIVIDUAL PLOTS
# ───────────────────────────────────────────────────────────────────
# statistics table
st.subheader("Residual statistics")
st.dataframe(
    pd.concat({p["lbl"]: p["stats"] for p in pairs.values()}, axis=1).T
      .style.format("{:.3f}"),
    use_container_width=True
)

# residual clouds
st.subheader("Residual clouds")
for col, p in zip(st.columns(3), pairs.values()):
    with col:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(p["dE"], p["dN"], s=4, color=p["col"], alpha=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(p["lbl"])
        ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
        ax.set_aspect("equal"); ax.grid(ls="--", lw=0.3)
        st.pyplot(fig)

# error-vs-time plots
st.subheader("Error vs time")
for col, p in zip(st.columns(3), pairs.values()):
    with col:
        fig, ax = plt.subplots()
        ax.plot(p["time"], p["err"], color=p["col"], lw=0.8)
        ax.set_ylim(0, err_lim)
        ax.set_title(p["lbl"])
        ax.set_xlabel("UTC time"); ax.set_ylabel("Horiz. error [m]")
        ax.grid(ls="--", lw=0.3); fig.autofmt_xdate()
        st.pyplot(fig)

# ───────────────────────────────────────────────────────────────────
# 9) ALL-PAIRS COMBINED PLOTS (EQUAL HEIGHT)
# ───────────────────────────────────────────────────────────────────
st.subheader("All pairs combined")
left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in pairs.values():
        ax.plot(p["time"], p["err"],
                label=p["lbl"], color=p["col"], lw=1)
    ax.set_ylim(0, err_lim)
    ax.legend()
    ax.set_xlabel("UTC time"); ax.set_ylabel("Horiz. error [m]")
    ax.grid(ls="--", lw=0.3); fig.autofmt_xdate()
    st.pyplot(fig, use_container_width=True)

with right:
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in pairs.values():
        ax.scatter(p["dE"], p["dN"],
                   s=3, alpha=0.55,
                   color=p["col"], label=p["lbl"])
    ax.axhline(0, lw=0.5, color="#888"); ax.axvline(0, lw=0.5, color="#888")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.legend(markerscale=3)
    ax.set_aspect("auto")           # same height as left plot
    ax.grid(ls="--", lw=0.3)
    st.pyplot(fig, use_container_width=True)
