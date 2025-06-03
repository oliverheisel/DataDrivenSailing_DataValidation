#!/usr/bin/env python3
# ───────────────────────────────────────────────────────────────────
# 0) ABOUT THIS APP
# ───────────────────────────────────────────────────────────────────
"""
Data Validation dashboard for the **Data-Driven Sailing** system

• Loads three GPS logs (DDS BoatTracker, Smartphone, Vakaros)  
• Lets you pick a 5-minute-step time window & an outlier cutoff  
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
    """Extract start timestamp from two possible filename patterns."""
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
    Load CSV → DataFrame with UTC time & numeric lat/lon.
    If *rel* column is present, reconstruct absolute time from filename.
    """
    df = pd.read_csv(path)

    if spec["rel"] and spec["rel"] in df.columns:
        anchor = parse_start_from_name(path)
        if anchor is None:
            raise ValueError(f"Cannot parse start time from {path}")
        df["time_utc"] = anchor + pd.to_timedelta(df[spec["rel"]], unit="s")
    else:
        df["time_utc"] = pd.to_datetime(df[spec["time"]], utc=True, errors="coerce")

    df["lat"] = pd.to_numeric(df[spec["lat"]], errors="coerce")
    df["lon"] = pd.to_numeric(df[spec["lon"]], errors="coerce")
    df = df.dropna(subset=["time_utc", "lat", "lon"]).sort_values("time_utc")
    df["t_naive"] = df.time_utc.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def nearest_join(a: pd.DataFrame, b: pd.DataFrame, suf: str):
    """Time-nearest merge with ±150 ms tolerance."""
    return (pd.merge_asof(a, b, on="time_utc",
                          direction="nearest",
                          tolerance=pd.Timedelta("150ms"),
                          suffixes=("", suf))
            .dropna(subset=[f"lat{suf}", f"lon{suf}"]))


def flat_residuals(df: pd.DataFrame, suf: str):
    """Return East & North residuals in metres (flat-Earth)."""
    dlat = df["lat"] - df[f"lat{suf}"]
    dlon = df["lon"] - df[f"lon{suf}"]
    lat0 = df["lat"].mean()
    mdeg = 111_320
    dE = dlon * mdeg * math.cos(math.radians(lat0))
    dN = dlat * mdeg
    return dE, dN


def residual_metrics(dE: np.ndarray, dN: np.ndarray) -> pd.Series:
    """Bias_E, Bias_N, |bias|, RMSE (metres)."""
    bE, bN = dE.mean(), dN.mean()
    return pd.Series(dict(
        Bias_E=bE,
        Bias_N=bN, 
        Bias_mag=math.hypot(bE, bN),
        RMSE=math.sqrt(((dE - bE) ** 2 + (dN - bN) ** 2).mean())
    ))

# ───────────────────────────────────────────────────────────────────
# 4) LOAD DATA
# ───────────────────────────────────────────────────────────────────
dds, phone, vak = [load_track(FILES[k], SCHEMA[k]) for k in ("dds", "phone", "vak")]

# ───────────────────────────────────────────────────────────────────
# 5) SIDEBAR – LOGO & CONTROLS
# ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("dds.png", width=260)
    st.write("**Adjust the window and outlier threshold** to explore the sensor agreement.")

    st.header("Window (5-min steps)")

    tmin = min(df.t_naive.min() for df in (dds, phone, vak)).to_pydatetime()
    tmax = max(df.t_naive.max() for df in (dds, phone, vak)).to_pydatetime()

    start_naive, end_naive = st.slider(
        "Start / End",
        min_value=tmin, max_value=tmax,
        value=(dt.datetime(2025, 5, 28, 15, 20),
               dt.datetime(2025, 5, 28, 15, 37)),
        step=dt.timedelta(minutes=5),
        format="YYYY-MM-DD  HH:mm"
    )

    st.markdown("---")

    outlier = st.slider(
        "Outlier cutoff (horizontal error [m])",
        0.5, 20.0, 20.0, 0.5
    )

# filter to selected window
window_filter = lambda df: df[(df.t_naive >= start_naive) & (df.t_naive <= end_naive)]
dds, phone, vak = map(window_filter, (dds, phone, vak))

# ───────────────────────────────────────────────────────────────────
# 6) RESIDUAL COMPUTATION
# ───────────────────────────────────────────────────────────────────
pairs = dict(
    dp=dict(df=nearest_join(dds,   phone, "_p"),
            suf="_p", lbl="DDS → Smartphone", col=COLORS["dp"]),
    dv=dict(df=nearest_join(dds,   vak,   "_v"),
            suf="_v", lbl="DDS → Vakaros",    col=COLORS["dv"]),
    pv=dict(df=nearest_join(phone, vak,   "_v"),
            suf="_v", lbl="Smartphone → Vakaros", col=COLORS["pv"]),
)

for p in pairs.values():
    dE, dN = flat_residuals(p["df"], p["suf"])
    err = np.hypot(dE, dN)
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
# 7) MAIN PAGE INTRO –  TEXT 2/3  |  IMAGE 1/3  +  FORMULA PANEL
# ───────────────────────────────────────────────────────────────────

# Left-align every st.latex block
st.markdown(
    "<style>.katex-display{text-align:left !important;}</style>",
    unsafe_allow_html=True,
)

st.title("DataDrivenSailing — GPS Validation")

left, right = st.columns([2, 1], gap="large")

# ── left column ────────────────────────────────────────────────────
with left:
    st.markdown(
        """
        ### What this dashboard contains

        <span style='color:#E2007A;font-weight:700'>Dark map</span> – raw tracks in the selected window<br>
        <span style='color:#E2007A;font-weight:700'>Residual statistics</span> – mean East/North bias & RMSE<br>
        <span style='color:#E2007A;font-weight:700'>Residual clouds</span> – East/North error at every timestamp<br>
        <span style='color:#E2007A;font-weight:700'>Error-vs-time</span> – horizontal error history<br>
        <span style='color:#E2007A;font-weight:700'>All pairs combined</span> – overlay of curves and residual clouds
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        **Legend** (device-pair colours)<br>
        <span style='color:{COLORS["dp"]};font-weight:700'>■ DDS → Smartphone</span><br>
        <span style='color:{COLORS["dv"]};font-weight:700'>■ DDS → Vakaros</span><br>
        <span style='color:{COLORS["pv"]};font-weight:700'>■ Smartphone → Vakaros</span>
        """,
        unsafe_allow_html=True,
    )

# ── right column: setup photo ─────────────────────────────────────
with right:
    st.image(
        "DataValidationSetup.png",
        caption="Logging setup on the boat",
        use_container_width=True,
    )

# ───────────────────────────────────────────────────────────────────
# Formula panel  (all formulas now truly inside the grey area)
# ───────────────────────────────────────────────────────────────────
# everything added to this container is a child of the grey div
with st.container():
    left_formula, right_formula = st.columns([1, 1], gap="large")

    with left_formula:
        st.markdown("#### How the numbers are computed")

        st.latex(r"\Delta E_i = (\lambda_i^A-\lambda_i^B)\,\cos\varphi_0\,M_{\text{deg}}")
        st.caption("East error – longitude difference converted to metres at mean latitude.", unsafe_allow_html=True)

        st.latex(r"\Delta N_i = (\varphi_i^A-\varphi_i^B)\,M_{\text{deg}}")
        st.caption("North error – latitude difference converted to metres.", unsafe_allow_html=True)

        st.latex(r"\text{Bias}_E = \frac{1}{N}\sum_{i=1}^{N}\Delta E_i")
        st.caption("Average eastward offset (systematic error).", unsafe_allow_html=True)

    with right_formula:
        st.markdown(
            r"*$(\lambda,\varphi)$ = longitude & latitude (°) • "
            r"$\varphi_0$ = mean latitude • "
            r"$M_{\text{deg}} = 111{,}320\,$m per degree*"
        )
                
        st.latex(r"\text{Bias}_N = \frac{1}{N}\sum_{i=1}^{N}\Delta N_i")
        st.caption("Average northward offset (systematic error).", unsafe_allow_html=True)

        st.latex(r"\lvert\text{Bias}\rvert = \sqrt{\text{Bias}_E^{2}+\text{Bias}_N^{2}}")
        st.caption("Magnitude of that systematic offset vector.", unsafe_allow_html=True)

        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\bigl[(\Delta E_i-\text{Bias}_E)^2 + (\Delta N_i-\text{Bias}_N)^2\bigr]}")
        st.caption("Random scatter after bias removal – lower means tighter agreement.", unsafe_allow_html=True)

# ---- auto-zoom dark map (full-width) ----
m = folium.Map(tiles="CartoDB dark_matter", control_scale=True)

folium.PolyLine(dds[["lat", "lon"]].values,
                color=COLORS["dv"], weight=3, tooltip="DDS_BoatTracker").add_to(m)
folium.PolyLine(phone[["lat", "lon"]].values,
                color=COLORS["dp"], weight=3, tooltip="Smartphone").add_to(m)
folium.PolyLine(vak[["lat", "lon"]].values,
                color=COLORS["pv"], weight=3, tooltip="Vakaros").add_to(m)

all_lat = pd.concat([dds.lat, phone.lat, vak.lat])
all_lon = pd.concat([dds.lon, phone.lon, vak.lon])
m.fit_bounds([[all_lat.min(), all_lon.min()],
              [all_lat.max(), all_lon.max()]])

#  FULL-WIDTH  ▸ keep height = 550 px
st_folium(m, height=550, use_container_width=True)

# ───────────────────────────────────────────────────────────────────
# 8) TABLE & INDIVIDUAL PLOTS
# ───────────────────────────────────────────────────────────────────
st.subheader("Residual statistics")
st.dataframe(
    pd.concat({p["lbl"]: p["stats"] for p in pairs.values()}, axis=1).T
      .style.format("{:.3f}"),
    use_container_width=True
)

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
# 9) COMBINED PLOTS
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
    ax.set_aspect("auto")
    ax.grid(ls="--", lw=0.3)
    st.pyplot(fig, use_container_width=True)
