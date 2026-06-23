"""
build_dataset.py  --  turn a folder of Citi Bike trip CSVs into one tidy file.

Works whether the CSVs sit directly in the folder OR are nested in one
sub-folder per month (e.g. Raw_Data/202310-citibike-tripdata/...csv).  It
searches recursively, so just point it at the top Raw_Data folder.

What it does
------------
1. finds every *.csv under the folder, recursively (ignores .DS_Store / junk),
2. auto-detects the column schema -- the modern layout (started_at,
   start_station_id, start_lat, ...) and the old 2013 layout (starttime,
   start station id, start station latitude, ...),
3. groups files by month (from the YYYYMM in the name); the 2..4 split parts
   of a month are disjoint, so it aggregates each file and sums -- this keeps
   memory low (one ~200 MB part in RAM at a time, never a whole month),
4. ONLY if a month has both a full file and split parts (possible double
   download) does it fall back to trip-level de-duplication (by ride_id),
5. aggregates to station x day departures and arrivals,
6. writes  station_day_counts.csv  and  stations.csv.

Usage (Linux)
-------------
    python build_dataset.py /home/luca/Dokumente/Vertiefungsseminar/Project/Raw_Data
    python build_dataset.py <Raw_Data>  --out .     # write next to this script
"""
import argparse
import glob
import os
import re
import sys
import pandas as pd

# canonical name -> accepted raw names (after strip/lower/space->underscore)
ACCEPTED = {
    "start_time":       {"starttime", "started_at", "start_time"},
    "end_time":         {"stoptime", "ended_at", "end_time", "stop_time"},
    "start_station_id": {"start_station_id"},
    "end_station_id":   {"end_station_id"},
    "start_lat":        {"start_station_latitude", "start_lat"},
    "start_lng":        {"start_station_longitude", "start_lng"},
    "end_lat":          {"end_station_latitude", "end_lat"},
    "end_lng":          {"end_station_longitude", "end_lng"},
    "bike_id":          {"bikeid", "bike_id"},
    "ride_id":          {"ride_id"},
}
_ACCEPT_FLAT = {raw: canon for canon, raws in ACCEPTED.items() for raw in raws}


def _norm(col):
    return col.strip().strip('"').lower().replace(" ", "_")


def _read_one(path):
    keep = lambda c: _norm(c) in _ACCEPT_FLAT
    # dtype=str is essential: station IDs look like "6948.10" and must NOT be
    # parsed as floats (that would drop the trailing zero -> "6948.1" and split
    # one station into two).  lat/lng are converted back to numbers later.
    df = pd.read_csv(path, usecols=keep, dtype=str, low_memory=False)
    return df.rename(columns={c: _ACCEPT_FLAT[_norm(c)] for c in df.columns})


def _to_datetime(s):
    """Robust across Citi Bike's historical timestamp formats (incl. fractional sec)."""
    dt = pd.to_datetime(s, format="ISO8601", errors="coerce")
    if dt.isna().mean() > 0.3:
        dt = pd.to_datetime(s, format="mixed", errors="coerce")
    return dt


def _canon_station(s):
    """Canonical station id.  Citi Bike stores numeric ids inconsistently
    ('6948.10' in some files, the same station as '6948.1' in others, because
    it was saved as a float).  Normalising numeric ids to 2 decimals makes
    '6948.1' and '6948.10' the SAME key.  Alphanumeric ids (e.g. 'HB101')
    are left untouched."""
    try:
        return f"{float(s):.2f}"
    except (ValueError, TypeError):
        return str(s).strip()


def _month_of(path):
    m = re.search(r"(\d{6})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def _is_part(path):
    """True if the filename ends in _<number> (a split part)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return bool(re.search(r"_\d+$", stem))


def _aggregate(trips):
    """One trips DataFrame -> (departures, arrivals, coords, n_used)."""
    trips = trips.copy()
    trips["start_time"] = _to_datetime(trips["start_time"])
    trips["end_time"]   = _to_datetime(trips["end_time"])
    trips = trips.dropna(subset=["start_time", "end_time",
                                 "start_station_id", "end_station_id"])
    trips["start_station_id"] = trips["start_station_id"].astype(str).str.strip()
    trips["end_station_id"]   = trips["end_station_id"].astype(str).str.strip()

    dep = (trips.assign(day=trips["start_time"].dt.normalize())
                .groupby(["start_station_id", "day"]).size()
                .rename("departures").reset_index()
                .rename(columns={"start_station_id": "station"}))
    arr = (trips.assign(day=trips["end_time"].dt.normalize())
                .groupby(["end_station_id", "day"]).size()
                .rename("arrivals").reset_index()
                .rename(columns={"end_station_id": "station"}))
    coords = None
    if "start_lat" in trips.columns:
        coords = (trips[["start_station_id", "start_lat", "start_lng"]]
                  .rename(columns={"start_station_id": "station",
                                   "start_lat": "lat", "start_lng": "lon"}))
        coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
        coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
    return dep, arr, coords, len(trips)


def build(folder, out_dir="."):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith(".")]
    if not files:
        sys.exit(f"No .csv files found under {folder!r}")

    by_month = {}
    for f in files:
        by_month.setdefault(_month_of(f), []).append(f)

    dep_parts, arr_parts, coord_parts = [], [], []
    print(f"Found {len(files)} CSV file(s) in {len(by_month)} month(s) "
          f"under {folder}\n")

    for month, mfiles in sorted(by_month.items()):
        has_full  = any(not _is_part(f) for f in mfiles)
        has_parts = any(_is_part(f) for f in mfiles)

        if has_full and has_parts:
            # ambiguous: a full file AND split parts -> de-duplicate trips
            frames = []
            for f in mfiles:
                try:
                    frames.append(_read_one(f))
                except Exception as e:
                    print(f"  ! skipped {os.path.basename(f)}: {e}")
            trips = pd.concat(frames, ignore_index=True)
            n_raw = len(trips)
            if "ride_id" in trips.columns:
                trips = trips.drop_duplicates("ride_id")
            else:
                key = [c for c in ["start_time", "end_time", "start_station_id",
                                   "end_station_id", "bike_id"] if c in trips.columns]
                trips = trips.drop_duplicates(subset=key)
            dep, arr, coords, n = _aggregate(trips)
            dep_parts.append(dep); arr_parts.append(arr)
            if coords is not None:
                coord_parts.append(coords)
            print(f"  {month}: {len(mfiles)} file(s), de-duplicated "
                  f"{n_raw:,} -> {n:,} trips")
        else:
            # normal case: disjoint parts -> aggregate each file and sum
            n_total = 0
            for f in mfiles:
                try:
                    dep, arr, coords, n = _aggregate(_read_one(f))
                except Exception as e:
                    print(f"  ! skipped {os.path.basename(f)}: {e}")
                    continue
                dep_parts.append(dep); arr_parts.append(arr)
                if coords is not None:
                    coord_parts.append(coords)
                n_total += n
            print(f"  {month}: {len(mfiles)} file(s), {n_total:,} trips")

    # ---- combine all months (canonicalise ids first so 6948.1 == 6948.10) ----
    dep = pd.concat(dep_parts); arr = pd.concat(arr_parts)
    dep["station"] = dep["station"].map(_canon_station)
    arr["station"] = arr["station"].map(_canon_station)
    dep = dep.groupby(["station", "day"], as_index=False).sum()
    arr = arr.groupby(["station", "day"], as_index=False).sum()
    panel = dep.merge(arr, on=["station", "day"], how="outer").fillna(0)
    panel[["departures", "arrivals"]] = panel[["departures", "arrivals"]].astype(int)
    panel["day"] = pd.to_datetime(panel["day"])
    panel["is_weekend"] = (panel["day"].dt.weekday >= 5).astype(int)

    if coord_parts:
        coords = pd.concat(coord_parts).dropna()
        coords["station"] = coords["station"].map(_canon_station)
        coords = coords.groupby("station")[["lat", "lon"]].median().reset_index()
        panel = panel.merge(coords, on="station", how="left")

    panel = panel.sort_values(["station", "day"]).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    panel_path = os.path.join(out_dir, "station_day_counts.csv")
    panel.to_csv(panel_path, index=False)
    if coord_parts:
        coords.to_csv(os.path.join(out_dir, "stations.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"stations          : {panel['station'].nunique():,}")
    print(f"station-day rows  : {len(panel):,}")
    print(f"date range        : {panel['day'].min().date()} .. {panel['day'].max().date()}")
    print(f"total departures  : {panel['departures'].sum():,}")
    print(f"total arrivals    : {panel['arrivals'].sum():,}")
    print(f"\nwrote: {panel_path}")
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aggregate Citi Bike CSVs to station-day counts.")
    ap.add_argument("folder", help="top folder containing the trip CSVs (searched recursively)")
    ap.add_argument("--out", default=".", help="output directory (default: current)")
    args = ap.parse_args()
    build(args.folder, args.out)