import io

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import logic as L
from demo_data import demo

st.set_page_config(page_title="Road Traffic Assessment - EWDFC", layout="wide")
st.title("Road Traffic Assessment - EWDFC")
st.caption("Road OD pairs (pincodes) -> nearest DFCC / IR freight station -> shift to the "
           "Eastern or Western DFC under the lead, first/last-mile and junction rules")


def read_table(f, sheet=None):
    """CSV, or Excel (named sheet if present, else the first sheet)."""
    if f is None:
        return None
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f)
    xl = pd.ExcelFile(f)
    return xl.parse(sheet if sheet in xl.sheet_names else xl.sheet_names[0])


def guess_corridor(filename: str) -> str:
    n = filename.lower()
    if "edf" in n or "east" in n:
        return "EDFC"
    if "wdf" in n or "west" in n:
        return "WDFC"
    return filename.rsplit(".", 1)[0].upper()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    use_demo = st.checkbox("Use built-in demo data (synthetic, not real geography)", False)
    st.header("Rules")
    min_dfcc = st.number_input("Minimum km on DFCC", 0.0, 5000.0, 200.0, 10.0)
    min_jn = st.number_input("Junction count required", 0, 20, 2, 1)
    default_x = st.number_input("Default catchment X (km), if blank in station table",
                                1.0, 500.0, 50.0, 5.0)
    st.header("Distance assumptions")
    road_f = st.number_input("Road factor (straight line -> road)", 1.0, 2.0, 1.30, 0.05)
    rail_f = st.number_input("Rail factor (only where no RBS km)", 1.0, 2.0, 1.25, 0.05)
    ir_max = st.number_input("Max road km, pincode -> IR freight station", 10.0, 500.0, 150.0, 10.0)
    freight_only = st.checkbox("Use only freight-handling IR stations", True)
    st.header("Optional baseline check")
    use_base = st.checkbox("Also require DFCC chain <= direct road km", False)
    tol = st.number_input("Tolerance (%)", 0.0, 200.0, 0.0, 5.0) / 100

# ------------------------------------------------------------------- inputs
lines = {}
if use_demo:
    d = demo()
    od, pins_raw, ir_raw, dfcc_raw, lines = d["od"], d["pins"], d["ir"], d["dfcc"], d["lines"]
    rbs_raw = None
    st.info("Demo data loaded: two synthetic corridors (EDFC, WDFC). Click **Run assessment**.")
else:
    c1, c2 = st.columns(2)
    with c1:
        od_f = st.file_uploader("1. OD pairs: origin_pincode, dest_pincode, tonnes, "
                                "commodity (optional), road_km (optional)", ["csv", "xlsx"])
        pin_f = st.file_uploader("2. Pincode directory: pincode, latitude, longitude", ["csv", "xlsx"])
        ir_f = st.file_uploader("3. IR stations: code, name, lat, lon, freight Y/N", ["csv", "xlsx"])
    with c2:
        dfcc_f = st.file_uploader("4. DFCC stations: corridor, code, name, lat, lon, chainage_km, "
                                  "junction Y/N, catchment_km, ir_station_code", ["csv", "xlsx"])
        aln_fs = st.file_uploader("4b. Alignment files (.kml / .kmz / zipped shapefile), one per corridor",
                                  ["kml", "kmz", "zip"], accept_multiple_files=True)
        rbs_f = st.file_uploader("5. (Optional) RBS km: from_station, to_station, km", ["csv", "xlsx"])

    od = read_table(od_f, "OD")
    pins_raw = read_table(pin_f, "Pincodes")
    ir_raw = read_table(ir_f, "IR_Stations")
    dfcc_raw = read_table(dfcc_f, "DFCC_Stations")
    rbs_raw = read_table(rbs_f, "RBS")

    for f in aln_fs or []:
        label = st.text_input(f"Corridor name for {f.name}", guess_corridor(f.name), key=f"lbl_{f.name}")
        try:
            ln, _pts = (L.read_shapefile_zip(f.getvalue()) if f.name.lower().endswith(".zip")
                        else L.read_kml(f.getvalue()))
            lines[label.strip()] = ln
            km = L.hav(*(np.array(ln)[:-1].T), *(np.array(ln)[1:].T)).sum() if len(ln) > 1 else 0
            st.caption(f"{label}: {len(ln)} vertices, about {km:,.0f} km")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read {f.name}: {e}")

    # fill chainage from the alignment when the station table has none
    if dfcc_raw is not None and lines and L._col(dfcc_raw, "chainage_km", "chainage", "km") is None:
        la_c, lo_c = L._col(dfcc_raw, "lat", "latitude"), L._col(dfcc_raw, "lon", "long", "longitude")
        cc = L._col(dfcc_raw, "corridor")
        if cc is None and len(lines) == 1:
            dfcc_raw["corridor"] = next(iter(lines))
            cc = "corridor"
        if cc is None:
            st.error("Station table needs a 'corridor' column when there is more than one alignment.")
        else:
            dfcc_raw["chainage_km"] = np.nan
            dfcc_raw["offset_from_line_km"] = np.nan
            for name, idx in dfcc_raw.groupby(cc).groups.items():
                ln = lines.get(str(name).strip())
                if ln is None:
                    st.warning(f"No alignment file named '{name}'; its stations have no chainage.")
                    continue
                g = dfcc_raw.loc[idx]
                ch, off = L.chainage_along_line(ln, g[la_c].astype(float), g[lo_c].astype(float))
                dfcc_raw.loc[idx, "chainage_km"] = ch.round(2)
                dfcc_raw.loc[idx, "offset_from_line_km"] = off.round(2)
            st.info("Chainage computed from the alignment (km along the line from its first point). "
                    "Compare with official chainages before using the results.")
            far = dfcc_raw["offset_from_line_km"] > 2
            if far.any():
                st.warning("Stations more than 2 km from their alignment: "
                           + ", ".join(dfcc_raw.loc[far, L._col(dfcc_raw, "code", "station_code")].astype(str)))

if dfcc_raw is not None:
    with st.expander("DFCC station table in use"):
        st.dataframe(dfcc_raw)

corr_options = (sorted(L.prep_dfcc(dfcc_raw, default_x).corridor.unique())
                if dfcc_raw is not None and L._col(dfcc_raw, "chainage_km", "chainage", "km") is not None
                else [])
corridors = st.multiselect("Corridors to assess", corr_options, default=corr_options)

# ---------------------------------------------------------------------- run
if st.button("Run assessment", type="primary"):
    try:
        if any(x is None for x in (od, pins_raw, ir_raw, dfcc_raw)):
            st.error("Upload files 1-4 first (or tick the demo-data box).")
            st.stop()
        if not corridors:
            st.error("Select at least one corridor.")
            st.stop()
        pins = L.clean_pincodes(pins_raw)
        dfcc = L.prep_dfcc(dfcc_raw, default_x)
        dfcc = dfcc[dfcc.corridor.isin(corridors)].reset_index(drop=True)
        ir = L.prep_ir(ir_raw, freight_only)
        rbs = L.rbs_to_dict(rbs_raw) if rbs_raw is not None else {}
        p = L.Params(min_dfcc_km=min_dfcc, min_jn_count=int(min_jn), road_factor=road_f,
                     rail_factor=rail_f, ir_max_km=ir_max, use_baseline=use_base,
                     baseline_tolerance=tol)
        with st.spinner("Assessing OD pairs..."):
            res = L.run_multi(od, pins, dfcc, ir, rbs, p)
        st.session_state["res"] = res
        st.session_state["dfcc_used"] = dfcc
        st.session_state["lines_used"] = {k: v for k, v in lines.items() if k in corridors}
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        st.stop()

res = st.session_state.get("res")
if res is None:
    st.stop()

dfcc_used = st.session_state["dfcc_used"]
lines_used = st.session_state["lines_used"]
has_t = "tonnes" in res
res = res.copy()
res["_w"] = res["tonnes"].astype(float) if has_t else 1.0
shifted = res[res.status == "Shifted"]
unit = "tonnes" if has_t else "pairs"

# ----------------------------------------------------------------- headline
m = st.columns(5)
m[0].metric("OD pairs", f"{len(res):,}")
m[1].metric("Pairs shifted", f"{len(shifted):,}", f"{100 * len(shifted) / max(len(res), 1):.1f}%")
m[2].metric(f"{unit.capitalize()} shifted", f"{shifted._w.sum():,.0f}")
m[3].metric(f"Share of {unit}", f"{100 * shifted._w.sum() / max(res._w.sum(), 1):.1f}%")
if has_t:
    tkm = (shifted._w * shifted.dfcc_km).sum() / 1e6
    m[4].metric("Tonne-km on DFCC (million)", f"{tkm:,.1f}")

tabs = st.tabs(["Summary", "Station-wise", "Commodity", "Map", "OD pairs", "Why not shifted"])

# ------------------------------------------------------------------ summary
with tabs[0]:
    by_c = (shifted.groupby("corridor")
            .agg(pairs=("_w", "size"), volume=("_w", "sum"), avg_dfcc_km=("dfcc_km", "mean"),
                 avg_first_last_mile_km=("first_mile_km", lambda s: (s + shifted.loc[s.index, "last_mile_km"]).mean()))
            .round(1).reset_index())
    by_c = by_c.rename(columns={"volume": unit})
    st.subheader("Shift by corridor")
    st.dataframe(by_c)
    st.subheader("Shift by Near / Away combination")
    combo = (res.groupby(["origin_type", "dest_type", "status"])._w.agg(["size", "sum"])
             .rename(columns={"size": "pairs", "sum": unit}).round(0).reset_index())
    st.dataframe(combo)

# ------------------------------------------------------------- station-wise
with tabs[1]:
    load = shifted.groupby(["corridor", "entry"])._w.sum().rename("loading")
    unl = shifted.groupby(["corridor", "exit"])._w.sum().rename("unloading")
    load.index.names, unl.index.names = ["corridor", "station"], ["corridor", "station"]
    stn = pd.concat([load, unl], axis=1).fillna(0).reset_index()
    stn["total"] = stn.loading + stn.unloading
    stn = stn.sort_values(["corridor", "total"], ascending=[True, False])
    st.subheader(f"Station-wise loading / unloading ({unit})")
    st.dataframe(stn.round(0))
    if len(stn):
        st.bar_chart(stn.set_index("station")[["loading", "unloading"]])

# ---------------------------------------------------------------- commodity
with tabs[2]:
    if "commodity" not in res:
        st.info("Add a 'commodity' column to the OD file to see this breakdown.")
    else:
        cm = res.groupby("commodity")._w.sum().rename("total")
        sm = shifted.groupby("commodity")._w.sum().rename("shifted")
        t = pd.concat([cm, sm], axis=1).fillna(0)
        t["share_%"] = (100 * t.shifted / t.total).round(1)
        st.dataframe(t.sort_values("total", ascending=False).round(0))
        st.bar_chart(t[["total", "shifted"]])

# ---------------------------------------------------------------------- map
with tabs[3]:
    top_n = st.slider("Show top N shifted flows (by volume)", 10, 1000, 200, 10)
    arcs = (shifted.dropna(subset=["o_lat", "o_lon", "d_lat", "d_lon"])
            .sort_values("_w", ascending=False).head(top_n).copy())
    if len(arcs):
        arcs["width"] = 1 + 5 * arcs._w / arcs._w.max()
    layers = []
    paths = [{"corridor": k, "path": [[lo, la] for la, lo in v]} for k, v in lines_used.items()]
    if paths:
        layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_width=3000,
                                get_color=[30, 100, 200], width_min_pixels=2))
    stn_pts = dfcc_used.assign(color=dfcc_used.junction.map(lambda j: [220, 60, 40] if j else [90, 90, 90]))
    layers.append(pdk.Layer("ScatterplotLayer", stn_pts, get_position=["lon", "lat"],
                            get_fill_color="color", get_radius=9000, radius_min_pixels=3,
                            pickable=True))
    if len(arcs):
        layers.append(pdk.Layer("ArcLayer", arcs, get_source_position=["o_lon", "o_lat"],
                                get_target_position=["d_lon", "d_lat"], get_width="width",
                                get_source_color=[0, 150, 80, 140], get_target_color=[240, 140, 0, 140]))
    view = pdk.ViewState(latitude=float(dfcc_used.lat.mean()), longitude=float(dfcc_used.lon.mean()), zoom=4.3)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{name}"}))
    st.caption("Red dots = junction stations, grey = other DFCC stations, blue = alignment, "
               "arcs = shifted OD flows (origin green, destination orange). Arcs show demand, not the route taken.")

# ---------------------------------------------------------------- pair list
export_cols = [c for c in res.columns if c not in ("_w", "o_lat", "o_lon", "d_lat", "d_lon")]
with tabs[4]:
    st.dataframe(res[export_cols])
with tabs[5]:
    why = res[res.status != "Shifted"].groupby("reason").size().sort_values(ascending=False).rename("pairs")
    st.dataframe(why)
    if len(why):
        st.bar_chart(why)

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    by_c.to_excel(w, sheet_name="By corridor", index=False)
    stn.to_excel(w, sheet_name="Station-wise", index=False)
    if "commodity" in res:
        t.reset_index().to_excel(w, sheet_name="Commodity", index=False)
    shifted[export_cols].to_excel(w, sheet_name="Shifted", index=False)
    res[export_cols].to_excel(w, sheet_name="All pairs", index=False)
st.download_button("Download results (Excel)", buf.getvalue(), "ewdfc_road_assessment.xlsx")
