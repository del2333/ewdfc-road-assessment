"""Core logic: pincode -> station assignment and DFCC shift test.

Rules implemented (as specified by the user)
-------------------------------------------
* Each OD end is classified NEAR / AWAY relative to the DFCC.
  - NEAR : at least one DFCC station whose (road-adjusted) distance to the
           pincode is <= that station's own catchment_km.  Any of those
           stations may be used as entry/exit (junction or not).
  - AWAY : otherwise.  Pincode -> nearest freight IR station (road), then
           IR station -> DFCC *junction* station (RBS km if supplied, else
           an estimate), and the DFCC is entered/exited at a junction only.
* A pair shifts to the DFCC if, for some entry/exit combination:
  1. km travelled on the DFCC (chainage difference) >= min_dfcc_km (200)
  2. first-mile + last-mile km < km travelled on the DFCC
  3. junction rule: count >= min_jn_count  (see count_jn() below)
  4. (optional) door-to-door DFCC chain <= baseline * (1 + tolerance)
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

R_EARTH = 6371.0088


# ----------------------------------------------------------------- geometry
def hav(lat1, lon1, lat2, lon2):
    """Haversine km, numpy-broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def _col(df, *names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def clean_pincodes(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per pincode (mean of its post offices) with valid India coords."""
    pc = _col(df, "pincode", "pin", "pin_code")
    la = _col(df, "latitude", "lat")
    lo = _col(df, "longitude", "lon", "long", "lng")
    if not (pc and la and lo):
        raise ValueError("Pincode file needs columns: pincode, latitude, longitude")
    d = pd.DataFrame({
        "pincode": pd.to_numeric(df[pc], errors="coerce"),
        "lat": pd.to_numeric(df[la], errors="coerce"),
        "lon": pd.to_numeric(df[lo], errors="coerce"),
    }).dropna()
    d = d[d.lat.between(6, 38) & d.lon.between(67, 98)]
    d["pincode"] = d["pincode"].astype(int)
    return d.groupby("pincode", as_index=False)[["lat", "lon"]].mean()


# ---------------------------------------------------------- alignment files
def _parse_coords(text):
    pts = []
    for tok in re.split(r"\s+", text.strip()):
        parts = tok.split(",")
        if len(parts) >= 2:
            try:
                pts.append((float(parts[1]), float(parts[0])))  # lat, lon
            except ValueError:
                pass
    return pts


def read_kml(data: bytes):
    """Return (line_pts, points_df).  Accepts .kml or .kmz bytes.

    line_pts : list of (lat, lon) - all LineStrings joined in file order
    points_df: name, lat, lon for every Point placemark (station candidates)
    """
    if data[:2] == b"PK":  # kmz
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
            data = z.read(name)
    root = ET.fromstring(data)
    strip = lambda t: t.split("}")[-1]
    line, pts = [], []
    for pm in root.iter():
        if strip(pm.tag) != "Placemark":
            continue
        name = ""
        for ch in pm:
            if strip(ch.tag) == "name":
                name = (ch.text or "").strip()
        for el in pm.iter():
            t = strip(el.tag)
            if t == "LineString":
                for c in el.iter():
                    if strip(c.tag) == "coordinates" and c.text:
                        line.extend(_parse_coords(c.text))
            elif t == "Point":
                for c in el.iter():
                    if strip(c.tag) == "coordinates" and c.text:
                        p = _parse_coords(c.text)
                        if p:
                            pts.append((name, p[0][0], p[0][1]))
    return line, pd.DataFrame(pts, columns=["name", "lat", "lon"])


def read_shapefile_zip(data: bytes):
    """Read a zipped shapefile (.shp/.dbf/.shx inside).  Needs `pyshp`.
    Coordinates must be lon/lat (EPSG:4326); re-project in QGIS first if not.
    """
    import shapefile  # pyshp

    z = zipfile.ZipFile(io.BytesIO(data))
    get = lambda ext: io.BytesIO(z.read(next(n for n in z.namelist()
                                             if n.lower().endswith(ext))))
    sf = shapefile.Reader(shp=get(".shp"), dbf=get(".dbf"),
                          shx=get(".shx"))
    line, pts = [], []
    fields = [f[0] for f in sf.fields[1:]]
    for sr in sf.shapeRecords():
        rec = dict(zip(fields, sr.record))
        nm = str(next((rec[k] for k in rec
                       if k.lower() in ("name", "station", "stn_name", "code")), ""))
        t = sr.shape.shapeType
        if t in (3, 13, 23):  # polyline
            parts = list(sr.shape.parts) + [len(sr.shape.points)]
            for a, b in zip(parts[:-1], parts[1:]):
                line.extend((y, x) for x, y in sr.shape.points[a:b])
        elif t in (1, 11, 21) and sr.shape.points:
            x, y = sr.shape.points[0]
            pts.append((nm, y, x))
    return line, pd.DataFrame(pts, columns=["name", "lat", "lon"])


def chainage_along_line(line, lat, lon):
    """Chainage (km from line start) of each point after projecting it on the
    polyline, plus offset km from the line.  Local flat-earth projection per
    segment (fine at segment scale)."""
    L = np.asarray(line, float)
    seg = hav(L[:-1, 0], L[:-1, 1], L[1:, 0], L[1:, 1])
    cum = np.concatenate([[0], np.cumsum(seg)])
    out_ch, out_off = [], []
    for la, lo in zip(np.atleast_1d(lat), np.atleast_1d(lon)):
        k = np.cos(np.radians(la))
        ax, ay = L[:-1, 1] * k, L[:-1, 0]
        bx, by = L[1:, 1] * k, L[1:, 0]
        px, py = lo * k, la
        dx, dy = bx - ax, by - ay
        den = dx * dx + dy * dy
        t = np.clip(np.where(den > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(den > 0, den, 1), 0), 0, 1)
        qy, qx = ay + t * dy, (ax + t * dx) / k
        d = hav(la, lo, qy, qx)
        i = int(np.argmin(d))
        out_ch.append(cum[i] + t[i] * seg[i])
        out_off.append(d[i])
    return np.array(out_ch), np.array(out_off)


# ------------------------------------------------------------- table prep
def prep_dfcc(df: pd.DataFrame, default_catchment: float = 50.0) -> pd.DataFrame:
    c = {k: _col(df, *v) for k, v in {
        "code": ("code", "station_code"), "name": ("name", "station", "station_name"),
        "lat": ("lat", "latitude"), "lon": ("lon", "long", "longitude"),
        "chainage_km": ("chainage_km", "chainage", "km"),
        "junction": ("junction", "is_junction", "interchange"),
        "catchment_km": ("catchment_km", "x_km", "threshold_km"),
        "ir_station_code": ("ir_station_code", "ir_code", "interchange_code"),
    }.items()}
    for need in ("code", "lat", "lon", "chainage_km"):
        if c[need] is None:
            raise ValueError(f"DFCC stations table is missing column '{need}'")
    out = pd.DataFrame({
        "code": df[c["code"]].astype(str).str.strip(),
        "name": df[c["name"]].astype(str) if c["name"] else df[c["code"]].astype(str),
        "lat": pd.to_numeric(df[c["lat"]], errors="coerce"),
        "lon": pd.to_numeric(df[c["lon"]], errors="coerce"),
        "chainage_km": pd.to_numeric(df[c["chainage_km"]], errors="coerce"),
    })
    jn = df[c["junction"]].astype(str).str.strip().str.upper() if c["junction"] else "N"
    out["junction"] = pd.Series(jn, index=out.index).isin(["Y", "YES", "TRUE", "1", "J"])
    out["catchment_km"] = (pd.to_numeric(df[c["catchment_km"]], errors="coerce")
                           if c["catchment_km"] else np.nan)
    out["catchment_km"] = out["catchment_km"].fillna(default_catchment)
    out["ir_station_code"] = (df[c["ir_station_code"]].astype(str).str.strip().str.upper()
                              if c["ir_station_code"] else "")
    cc = _col(df, "corridor")
    out["corridor"] = df[cc].astype(str).str.strip() if cc else "DFCC"
    out = out.dropna(subset=["lat", "lon", "chainage_km"]).sort_values(["corridor", "chainage_km"])
    return out.reset_index(drop=True)


def prep_ir(df: pd.DataFrame, freight_only: bool = True) -> pd.DataFrame:
    code = _col(df, "code", "station_code")
    la = _col(df, "lat", "latitude")
    lo = _col(df, "lon", "long", "longitude")
    nm = _col(df, "name", "station_name")
    fr = _col(df, "freight", "goods", "goods_shed")
    if not (code and la and lo):
        raise ValueError("IR stations table needs: code, lat, lon (name, freight optional)")
    out = pd.DataFrame({
        "code": df[code].astype(str).str.strip().str.upper(),
        "name": df[nm].astype(str) if nm else df[code].astype(str),
        "lat": pd.to_numeric(df[la], errors="coerce"),
        "lon": pd.to_numeric(df[lo], errors="coerce"),
    })
    if fr:
        out["freight"] = df[fr].astype(str).str.strip().str.upper().isin(["Y", "YES", "TRUE", "1"])
    else:
        out["freight"] = True
    out = out.dropna(subset=["lat", "lon"])
    out = out[out.lat.between(6, 38) & out.lon.between(67, 98)]
    if freight_only and out["freight"].any():
        out = out[out["freight"]]
    return out.drop_duplicates("code").reset_index(drop=True)


# ------------------------------------------------------------ end assignment
class Params:
    def __init__(self, **kw):
        self.min_dfcc_km = 200.0
        self.min_jn_count = 2
        self.road_factor = 1.30     # straight line -> road km
        self.rail_factor = 1.25     # straight line -> rail km (when no RBS km)
        self.ir_max_km = 150.0      # max pincode -> IR freight station (road)
        self.use_baseline = False
        self.baseline_tolerance = 0.0
        self.__dict__.update(kw)


def assign_end(lat, lon, dfcc, ir, rbs, p: Params):
    """Return dict(kind, cands) for one OD end.
    cands: arrays idx (DFCC station index), access (km), near (bool), via (text)."""
    d_road = hav(lat, lon, dfcc.lat.values, dfcc.lon.values) * p.road_factor
    within = d_road <= dfcc.catchment_km.values
    if within.any():
        idx = np.where(within)[0]
        return dict(kind="Near", idx=idx, access=d_road[idx],
                    near=np.ones(len(idx), bool),
                    via=[f"road to {dfcc.code[i]}" for i in idx], ir_stn=None)

    if len(ir) == 0:
        return dict(kind="Away", idx=np.array([], int), access=np.array([]),
                    near=np.array([], bool), via=[], ir_stn=None)
    d_ir = hav(lat, lon, ir.lat.values, ir.lon.values) * p.road_factor
    j = int(np.argmin(d_ir))
    if d_ir[j] > p.ir_max_km:
        return dict(kind="Away", idx=np.array([], int), access=np.array([]),
                    near=np.array([], bool), via=[], ir_stn=None)
    st = ir.iloc[j]
    jn_idx = np.where(dfcc.junction.values)[0]
    acc, via = [], []
    for i in jn_idx:
        icode = dfcc.ir_station_code[i]
        km = rbs.get((st.code, icode)) if icode else None
        est = ""
        if km is None:
            km = hav(st.lat, st.lon, dfcc.lat[i], dfcc.lon[i]) * p.rail_factor
            est = " (est.)"
        acc.append(d_ir[j] + km)
        via.append(f"road to {st.code} + IR to {dfcc.code[i]}{est}")
    return dict(kind="Away", idx=jn_idx, access=np.array(acc),
                near=np.zeros(len(jn_idx), bool), via=via, ir_stn=st.code)


def count_jn(dfcc, i_in, i_out, near_in, near_out):
    """Junction-rule tally.

    = junction stations strictly between entry and exit (by chainage)
    + 1 for entry if it is a directly assigned (NEAR) station, any type
    + 1 for exit  if it is a directly assigned (NEAR) station, any type
    """
    lo, hi = sorted((dfcc.chainage_km[i_in], dfcc.chainage_km[i_out]))
    between = dfcc.junction.values & (dfcc.chainage_km.values > lo) & (dfcc.chainage_km.values < hi)
    return int(between.sum()) + int(near_in) + int(near_out)


def evaluate_pair(o, d, dfcc, p: Params, baseline=None):
    if len(o["idx"]) == 0:
        return dict(status="Not shifted", reason="Origin has no DFCC/IR access within limits")
    if len(d["idx"]) == 0:
        return dict(status="Not shifted", reason="Destination has no DFCC/IR access within limits")
    ch = dfcc.chainage_km.values
    best, why = None, "DFCC run below minimum"
    ever_len = ever_mile = ever_jn = False
    for a, io_ in enumerate(o["idx"]):
        for b, id_ in enumerate(d["idx"]):
            if io_ == id_:
                continue
            leg = abs(ch[id_] - ch[io_])
            if leg < p.min_dfcc_km:
                continue
            ever_len = True
            mile = o["access"][a] + d["access"][b]
            if not mile < leg:
                continue
            ever_mile = True
            jn = count_jn(dfcc, io_, id_, o["near"][a], d["near"][b])
            if jn < p.min_jn_count:
                continue
            ever_jn = True
            total = mile + leg
            if p.use_baseline and baseline and total > baseline * (1 + p.baseline_tolerance):
                why = "DFCC chain longer than baseline"
                continue
            if best is None or total < best["total"]:
                best = dict(total=total, a=a, b=b, io=io_, id=id_, leg=leg, jn=jn)
    if best is None:
        if not ever_len:
            why = f"DFCC run < {p.min_dfcc_km:.0f} km"
        elif not ever_mile:
            why = "First+last mile >= DFCC run"
        elif not ever_jn:
            why = f"Junction rule not met (< {p.min_jn_count})"
        return dict(status="Not shifted", reason=why)
    a, b = best["a"], best["b"]
    return dict(
        status="Shifted", reason="",
        entry=f"{dfcc.code[best['io']]} {dfcc.name[best['io']]}",
        exit=f"{dfcc.code[best['id']]} {dfcc.name[best['id']]}",
        origin_access=o["via"][a], dest_access=d["via"][b],
        first_mile_km=round(o["access"][a], 1), last_mile_km=round(d["access"][b], 1),
        dfcc_km=round(best["leg"], 1), junctions_counted=best["jn"],
        dfcc_chain_km=round(best["total"], 1))


# ---------------------------------------------------------------- the driver
def run(od: pd.DataFrame, pins: pd.DataFrame, dfcc: pd.DataFrame,
        ir: pd.DataFrame, rbs: dict, p: Params) -> pd.DataFrame:
    oc = _col(od, "origin_pincode", "origin", "from_pincode")
    dc = _col(od, "dest_pincode", "destination_pincode", "destination", "to_pincode")
    tc = _col(od, "tonnes", "tonnage", "tons", "quantity")
    bc = _col(od, "road_km", "distance_km", "road_distance_km")
    cm = _col(od, "commodity", "commodity_name", "comm")
    if not (oc and dc):
        raise ValueError("OD file needs columns: origin_pincode, dest_pincode")
    pin_xy = pins.set_index("pincode")[["lat", "lon"]]

    cache = {}

    def end(pin):
        if pin not in cache:
            try:
                r = pin_xy.loc[int(pin)]
                cache[pin] = assign_end(r.lat, r.lon, dfcc, ir, rbs, p)
            except (KeyError, ValueError):
                cache[pin] = None
        return cache[pin]

    rows = []
    for _, r in od.iterrows():
        row = dict(origin_pincode=r[oc], dest_pincode=r[dc])
        if tc:
            row["tonnes"] = r[tc]
        if cm:
            row["commodity"] = r[cm]
        o, d = end(r[oc]), end(r[dc])
        if o is None or d is None:
            miss = "Origin" if o is None else "Destination"
            row.update(origin_type="", dest_type="", status="Not shifted",
                       reason=f"{miss} pincode has no coordinates")
            rows.append(row)
            continue
        base = None
        if bc and pd.notna(r[bc]):
            base = float(r[bc])
        else:
            try:
                a, b = pin_xy.loc[int(r[oc])], pin_xy.loc[int(r[dc])]
                base = float(hav(a.lat, a.lon, b.lat, b.lon) * p.road_factor)
            except (KeyError, ValueError):
                pass
        a_xy, b_xy = pin_xy.loc[int(r[oc])], pin_xy.loc[int(r[dc])]
        row.update(origin_type=o["kind"], dest_type=d["kind"], baseline_km=
                   None if base is None else round(base, 1),
                   o_lat=a_xy.lat, o_lon=a_xy.lon, d_lat=b_xy.lat, d_lon=b_xy.lon)
        row.update(evaluate_pair(o, d, dfcc, p, base))
        rows.append(row)
    res = pd.DataFrame(rows)
    first = ["origin_pincode", "dest_pincode", "commodity", "tonnes", "origin_type",
             "dest_type", "status", "reason", "entry", "exit", "origin_access",
             "dest_access", "first_mile_km", "dfcc_km", "last_mile_km",
             "junctions_counted", "dfcc_chain_km", "baseline_km",
             "o_lat", "o_lon", "d_lat", "d_lon"]
    return res[[c for c in first if c in res.columns]]


def run_multi(od, pins, dfcc_all, ir, rbs, p: Params) -> pd.DataFrame:
    """Evaluate every corridor in dfcc_all['corridor'] (e.g. EDFC, WDFC)
    separately, then keep for each OD pair the shifted option with the shorter
    door-to-door chain.  Pairs that shift nowhere get both corridors' reasons."""
    outs = {}
    for name, g in dfcc_all.groupby("corridor", sort=False):
        outs[name] = run(od, pins, g.sort_values("chainage_km").reset_index(drop=True),
                         ir, rbs, p)
    allr = pd.concat([o.assign(corridor=n, _row=np.arange(len(o)))
                      for n, o in outs.items()], ignore_index=True)
    sh = (allr[allr.status == "Shifted"].sort_values("dfcc_chain_km")
          .drop_duplicates("_row"))
    ns = allr[(allr.status != "Shifted") & ~allr._row.isin(sh._row)]
    joined = (ns.assign(_r=ns.corridor + ": " + ns.reason.fillna(""))
              .groupby("_row")["_r"].agg(" | ".join)
              if len(outs) > 1 else ns.set_index("_row").reason)
    ns = ns.drop_duplicates("_row").copy()
    ns["reason"] = ns._row.map(joined)
    ns["corridor"] = ""
    res = pd.concat([sh, ns]).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    cols = list(res.columns)
    cols.insert(cols.index("status") + 1, cols.pop(cols.index("corridor")))
    return res[cols]


def rbs_to_dict(df: pd.DataFrame) -> dict:
    f = _col(df, "from_station", "from", "origin")
    t = _col(df, "to_station", "to", "destination")
    k = _col(df, "km", "distance_km", "rbs_km")
    if not (f and t and k):
        raise ValueError("RBS file needs: from_station, to_station, km")
    return {(str(a).strip().upper(), str(b).strip().upper()): float(c)
            for a, b, c in zip(df[f], df[t], df[k]) if pd.notna(c)}
