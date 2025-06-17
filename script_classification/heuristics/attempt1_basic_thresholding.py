import psycopg2, json, pandas as pd
from collections import Counter

# ── 1.  DB connection ─────────────────────────────────────────────────────────
conn = psycopg2.connect(
        host="localhost",
        port=5434,
        dbname="vv8_backend",
        user="vv8",
        password="vv8")

qry = """
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

df = pd.read_sql(qry, conn)
conn.close()

# Many fields come back as text → convert JSON‑like columns to dict/list
json_cols = [
    "max_aggregated_apis", "aggregated_behavioral_apis",
    "aggregated_fingerprinting_apis", "attached_listeners",
    "fingerprinting_source_apis", "behavioral_source_apis",
    "behavioral_apis_access_count", "fingerprinting_api_access_count",
    "apis_going_to_sink"
]
for c in json_cols:
    df[c] = df[c].apply(lambda x: json.loads(x) if isinstance(x,str) and x.strip() else {} if x is None else x)

# ── 2.  Rule‑based heuristic (same thresholds we discussed) ──────────────────
def is_behavioral_biometric(script):
    behavioral_cnt   = script.get("behavioral_source_api_count"     , 0) or 0
    fingerprint_cnt  = script.get("fingerprinting_source_api_count" , 0) or 0
    behav_agg_score  = script.get("max_behavioral_api_aggregation_score", 0) or 0
    total_agg_score  = script.get("max_api_aggregation_score", 0) or 0
    graph_failed     = script.get("graph_construction_failure", True)
    to_sink          = script.get("dataflow_to_sink", False)
    sinks            = script.get("apis_going_to_sink", {}) or {}
    acc_counts       = script.get("behavioral_apis_access_count", {}) or {}

    # R1: must use *both* sets of APIs
    if behavioral_cnt < 3 or fingerprint_cnt < 2:
        return 0

    # R2: enough aggregation
    if behav_agg_score < 5 and total_agg_score < 5:
        return 0

    # R3: repeated behavioural accesses
    if not any(v >= 5 for v in acc_counts.values()):
        return 0

    # R4: flow to sink if graph succeeded
    if not graph_failed:
        return int(to_sink or bool(sinks))

    # fallback if graph failed but very strong behaviour
    return int(behav_agg_score >= 10 and behavioral_cnt >= 5)

df["predicted"] = df.apply(is_behavioral_biometric, axis=1)

# ── 3.  Confusion matrix + simple metrics ────────────────────────────────────
tp = ((df.predicted==1) & (df.label==1)).sum()
tn = ((df.predicted==0) & (df.label==0)).sum()
fp = ((df.predicted==1) & (df.label==0)).sum()
fn = ((df.predicted==0) & (df.label==1)).sum()

conf = pd.DataFrame(
        [["TP",tp],["FP",fp],["FN",fn],["TN",tn]],
        columns=["Cell","Count"]).set_index("Cell")
print(conf,"\n")

total = len(df)
acc  = (tp+tn)/total
prec = tp/(tp+fp) if (tp+fp) else 0
rec  = tp/(tp+fn) if (tp+fn) else 0
print(f"Accuracy : {acc:0.3f}")
print(f"Precision : {prec:0.3f}")
print(f"Recall    : {rec:0.3f}")
