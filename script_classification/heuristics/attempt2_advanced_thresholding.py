#!/usr/bin/env python3
# rule_grid_search.py
# --------------------------------------------------------------------
# Search for the best point‑based heuristic to identify behavioural‑
# biometric JavaScript using static features pulled from Postgres.
# --------------------------------------------------------------------

import psycopg2
import json, itertools
import pandas as pd

# ────────────────────────────────────────────────────────────────────
# 1.  Database fetch  (LIMIT 50 for quick tuning)
# ────────────────────────────────────────────────────────────────────
SQL = """
SELECT
    script_id,
    script_url,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fp_api_agg_count,
    max_aggregated_apis,
    max_behavioral_api_aggregation_score,
    aggregated_behavioral_apis,
    max_fingerprinting_api_aggregation_score,
    aggregated_fingerprinting_apis,
    attached_listeners,
    fingerprinting_source_apis,
    behavioral_source_apis,
    behavioral_source_api_count,
    fingerprinting_source_api_count,
    behavioral_apis_access_count,
    fingerprinting_api_access_count,
    graph_construction_failure,
    dataflow_to_sink,
    apis_going_to_sink,
    submission_url,
    label
FROM multicore_static_info_known_companies
WHERE label IN (0,1)
"""

conn = psycopg2.connect(
    host="localhost",
    port=5434,
    dbname="vv8_backend",
    user="vv8",
    password="vv8"
)
df = pd.read_sql(SQL, conn)
conn.close()

# ────────────────────────────────────────────────────────────────────
# 2.  Parse JSON‑like text columns → dict / list
# ────────────────────────────────────────────────────────────────────
json_cols = [
    "max_aggregated_apis",
    "aggregated_behavioral_apis",
    "aggregated_fingerprinting_apis",
    "attached_listeners",
    "fingerprinting_source_apis",
    "behavioral_source_apis",
    "behavioral_apis_access_count",
    "fingerprinting_api_access_count",
    "apis_going_to_sink"
]

def parse_json(x):
    if x is None:
        return {}
    if isinstance(x, (dict, list)):
        return x
    x = x.strip()
    if not x:
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        # fallback: Postgres might return “{…}” instead of JSON – ignore for now
        return {}

for col in json_cols:
    df[col] = df[col].apply(parse_json)

# ────────────────────────────────────────────────────────────────────
# 3.  Feature engineering helpers
# ────────────────────────────────────────────────────────────────────
def build_extra_features(row):
    b_cnt = row["behavioral_source_api_count"] or 0
    f_cnt = row["fingerprinting_source_api_count"] or 0
    behav_fp_ratio = b_cnt / (f_cnt + 1)           # +1 avoids div‑by‑zero

    behav_access = row.get("behavioral_apis_access_count", {}) or {}
    total_calls = sum(behav_access.values())
    max_burst_norm = (max(behav_access.values()) / (total_calls + 1)) if behav_access else 0

    sinks = row.get("apis_going_to_sink", {}) or {}
    sink_roots = {k.split('.')[0] for k in sinks}
    writes_storage = any(root in {"Window", "HTMLInputElement"} for root in sink_roots)
    uses_beacon    = any("sendBeacon" in k for k in sinks)
    script_inject  = any("HTMLScriptElement" in root for root in sink_roots)

    behav_apis = set(row.get("behavioral_source_apis", []) or [])
    flags = {
        "has_devicemotion"   : any("DeviceMotion"      in api for api in behav_apis),
        "has_orientation"    : any("DeviceOrientation" in api for api in behav_apis),
        "has_touch_event"    : any("TouchEvent"        in api or "Touch." in api for api in behav_apis),
        "has_wheel_event"    : any("WheelEvent"        in api for api in behav_apis),
        "has_keyboard_event" : any("KeyboardEvent"     in api for api in behav_apis)
    }

    return pd.Series({
        "behav_fp_ratio": behav_fp_ratio,
        "max_burst_norm": max_burst_norm,
        "writes_storage": writes_storage,
        "uses_beacon"   : uses_beacon,
        "script_inject" : script_inject,
        **flags
    })

df = df.join(df.apply(build_extra_features, axis=1))

# ────────────────────────────────────────────────────────────────────
# 4.  Scoring function with tunable weights + threshold
# ────────────────────────────────────────────────────────────────────
def point_score(row, w, thresh):
    """
    w = dict of integer weights, keys:
        bc, fc, ratio, agg, burst, sink, flag
    thresh = score threshold to predict positive
    """
    score = 0
    # counts
    if row.behavioral_source_api_count >= 4:
        score += w["bc"]
    if row.fingerprinting_source_api_count >= 2:
        score += w["fc"]
    # ratio
    if row.behav_fp_ratio >= 0.6:
        score += w["ratio"]
    # aggregation
    if (row.max_behavioral_api_aggregation_score or 0) >= 5:
        score += w["agg"]
    # burst / normalised access count
    if row.max_burst_norm >= 0.02:
        score += w["burst"]
    # sinks
    if row.writes_storage or row.uses_beacon or row.script_inject:
        score += w["sink"]
    # high‑signal event types
    if any([row.has_devicemotion, row.has_orientation,
            row.has_touch_event, row.has_keyboard_event]):
        score += w["flag"]

    return int(score >= thresh)

# ────────────────────────────────────────────────────────────────────
# 5.  Light grid‑search on integer weights + threshold
# ────────────────────────────────────────────────────────────────────
weight_options = {
    "bc"   : [1, 2],
    "fc"   : [1],        # fingerprint weight stays low
    "ratio": [1, 2],
    "agg"  : [1, 2],
    "burst": [1, 2],
    "sink" : [2, 3],
    "flag" : [1, 2],
}
thresholds = [6, 7, 8]

best = None
best_f1 = -1

for bc in weight_options["bc"]:
    for fc in weight_options["fc"]:
        for rt in weight_options["ratio"]:
            for ag in weight_options["agg"]:
                for bu in weight_options["burst"]:
                    for sk in weight_options["sink"]:
                        for fl in weight_options["flag"]:
                            for th in thresholds:
                                weights = dict(bc=bc, fc=fc, ratio=rt, agg=ag,
                                               burst=bu, sink=sk, flag=fl)
                                preds = df.apply(lambda r: point_score(r, weights, th), axis=1)

                                tp = ((preds == 1) & (df.label == 1)).sum()
                                fp = ((preds == 1) & (df.label == 0)).sum()
                                fn = ((preds == 0) & (df.label == 1)).sum()
                                if tp + fp == 0 or tp + fn == 0:
                                    continue
                                prec = tp / (tp + fp)
                                rec  = tp / (tp + fn)
                                f1   = 2 * prec * rec / (prec + rec)

                                if f1 > best_f1:
                                    best_f1 = f1
                                    best = dict(weights=weights, threshold=th,
                                                TP=tp, FP=fp, FN=fn, precision=prec,
                                                recall=rec, f1=f1)

# ────────────────────────────────────────────────────────────────────
# 6.  Report
# ────────────────────────────────────────────────────────────────────
print("Best weight configuration on 50‑row slice")
print("=========================================")
print(f"Weights   : {best['weights']}")
print(f"Threshold : {best['threshold']}")
print("\nConfusion Matrix")
print("----------------")
print(f"TP = {best['TP']}")
print(f"FP = {best['FP']}")
print(f"FN = {best['FN']}")
tn = len(df) - best['TP'] - best['FP'] - best['FN']
print(f"TN = {tn}")

print("\nMetrics")
print(f"Precision : {best['precision']:.3f}")
print(f"Recall    : {best['recall']:.3f}")
print(f"F1‑score  : {best['f1']:.3f}")
