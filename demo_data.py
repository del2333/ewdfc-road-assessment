"""Synthetic demo data - NOT real geography.  Two parallel straight corridors
(EDFC and WDFC) so the app can be tried end to end without any uploads."""
import numpy as np
import pandas as pd

import logic as L


def demo(seed: int = 0, n_od: int = 400):
    rng = np.random.default_rng(seed)
    corridors = {"EDFC": 82.0, "WDFC": 73.0}          # corridor -> longitude
    lat0, lat1, n_stn = 20.0, 30.0, 21

    rows, lines = [], {}
    for name, lon in corridors.items():
        lats = np.linspace(lat0, lat1, n_stn)
        lines[name] = [(la, lon) for la in np.linspace(lat0, lat1, 201)]
        for i, la in enumerate(lats):
            rows.append(dict(corridor=name, code=f"{name[0]}{i:02d}", name=f"{name} Stn {i}",
                             lat=la, lon=lon, junction="Y" if i % 3 == 0 else "N",
                             catchment_km=int(rng.choice([30, 40, 60])),
                             ir_station_code=f"IR{name[0]}{i:02d}" if i % 3 == 0 else ""))
    dfcc = pd.DataFrame(rows)

    # IR freight stations: a grid across both corridors, incl. interchange stations
    ir_rows = [dict(code=r.ir_station_code, name=f"IR {r.name}", lat=r.lat + 0.05, lon=r.lon + 0.05,
                    freight="Y") for r in dfcc.itertuples() if r.ir_station_code]
    for k, (la, lo) in enumerate(zip(rng.uniform(19, 31, 60), rng.uniform(70, 85, 60))):
        ir_rows.append(dict(code=f"G{k:03d}", name=f"Goods shed {k}", lat=la, lon=lo, freight="Y"))
    ir = pd.DataFrame(ir_rows)

    # pincodes: half near the corridors, half scattered
    near = dfcc.sample(60, replace=True, random_state=seed)
    pin_lat = np.concatenate([near.lat + rng.normal(0, 0.15, 60), rng.uniform(19, 31, 80)])
    pin_lon = np.concatenate([near.lon + rng.normal(0, 0.15, 60), rng.uniform(70, 85, 80)])
    pins = pd.DataFrame({"pincode": 100000 + np.arange(140), "latitude": pin_lat, "longitude": pin_lon})

    comm = ["Foodgrains", "Cement", "Fertiliser", "Steel", "Container", "Coal"]
    o = rng.choice(pins.pincode, n_od)
    d = rng.choice(pins.pincode, n_od)
    keep = o != d
    od = pd.DataFrame({"origin_pincode": o[keep], "dest_pincode": d[keep],
                       "commodity": rng.choice(comm, keep.sum()),
                       "tonnes": rng.integers(50, 5000, keep.sum())})

    # chainage from the alignment, per corridor
    out = []
    for name, g in dfcc.groupby("corridor"):
        g = g.copy()
        ch, _ = L.chainage_along_line(lines[name], g.lat.values, g.lon.values)
        g["chainage_km"] = ch.round(2)
        out.append(g)
    dfcc = pd.concat(out, ignore_index=True)
    return dict(od=od, pins=pins, ir=ir, dfcc=dfcc, lines=lines)
