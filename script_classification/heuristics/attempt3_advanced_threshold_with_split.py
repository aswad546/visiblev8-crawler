#!/usr/bin/env python3
# vendor_aware_rule_grid_search.py
# --------------------------------------------------------------------
# Search for the best point‑based heuristic to identify behavioural‑
# biometric JavaScript using static features pulled from Postgres.
# Uses vendor-aware splitting to prevent data leakage.
# --------------------------------------------------------------------

import psycopg2
import json, itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# ────────────────────────────────────────────────────────────────────
# 1.  Database fetch
# ────────────────────────────────────────────────────────────────────
SQL = """
SELECT
    script_id,
    script_url,
    vendor,
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
WHERE label IN (0,1,-1)
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
df.loc[df['label'] == -1, 'label'] = 0  # Convert -1 to 0 for normal scripts

print(f"Loaded {len(df)} samples: {(df.label == 1).sum()} positives, {(df.label == 0).sum()} negatives")

# ────────────────────────────────────────────────────────────────────
# 2.  Analyze vendor column (only populated for label=1)
# ────────────────────────────────────────────────────────────────────
# Vendor column is only populated for behavioral biometric scripts (label=1)

positive_samples = df[df['label'] == 1]
negative_samples = df[df['label'] == 0]

print(f"Positive samples with vendor info: {positive_samples['vendor'].notna().sum()}/{len(positive_samples)}")
print(f"Negative samples with vendor info: {negative_samples['vendor'].notna().sum()}/{len(negative_samples)}")

# Show vendor distribution for positives
positive_vendor_counts = positive_samples['vendor'].value_counts()
print(f"\nVendor distribution for behavioral biometric scripts (label=1):")
print(f"Found {positive_vendor_counts.nunique()} unique vendors")
print(f"Top 10 vendors by sample count:")
print(positive_vendor_counts.head(10))

# ────────────────────────────────────────────────────────────────────
# 3.  Parse JSON‑like text columns → dict / list
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
        return {}

for col in json_cols:
    df[col] = df[col].apply(parse_json)

# ────────────────────────────────────────────────────────────────────
# 4.  Feature engineering helpers
# ────────────────────────────────────────────────────────────────────
def build_extra_features(row):
    b_cnt = row["behavioral_source_api_count"] or 0
    f_cnt = row["fingerprinting_source_api_count"] or 0
    behav_fp_ratio = b_cnt / (f_cnt + 1)

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
# 5.  Vendor-Aware Split Function
# ────────────────────────────────────────────────────────────────────
def create_vendor_aware_split(features_df, test_size=0.3, random_state=42):
    """
    Create train/test split where:
    - Negatives (label=0) are split randomly - vendor column is null for these
    - Positives (label=1) are split with vendor awareness to prevent leakage
    """
    np.random.seed(random_state)
    
    # Separate positives and negatives
    positives = features_df[features_df['label'] == 1].copy()
    negatives = features_df[features_df['label'] == 0].copy()
    
    print(f"\nVendor-aware splitting:")
    print(f"Behavioral biometric scripts (label=1): {len(positives)} samples")
    print(f"Normal scripts (label=0): {len(negatives)} samples")
    
    # Analyze positive vendor distribution (only these have vendor info)
    vendor_counts = positives['vendor'].value_counts()
    high_volume_vendors = vendor_counts[vendor_counts > 20].index.tolist()
    medium_volume_vendors = vendor_counts[(vendor_counts >= 5) & (vendor_counts <= 20)].index.tolist()
    low_volume_vendors = vendor_counts[vendor_counts < 5].index.tolist()
    
    print(f"\nVendor categories for behavioral scripts:")
    print(f"High-volume vendors (>20 scripts): {len(high_volume_vendors)}")
    print(f"Medium-volume vendors (5-20 scripts): {len(medium_volume_vendors)}")
    print(f"Low-volume vendors (<5 scripts): {len(low_volume_vendors)}")
    
    train_pos_indices = []
    test_pos_indices = []
    
    # High volume vendors: Split scripts within vendor (70-30)
    for vendor in high_volume_vendors:
        vendor_scripts = positives[positives['vendor'] == vendor].index.tolist()
        np.random.shuffle(vendor_scripts)
        
        n_test = max(1, int(len(vendor_scripts) * test_size))
        test_pos_indices.extend(vendor_scripts[:n_test])
        train_pos_indices.extend(vendor_scripts[n_test:])
        
        print(f"  {vendor}: {len(vendor_scripts)} scripts → {len(vendor_scripts)-n_test} train, {n_test} test")
    
    # Medium volume vendors: 60% vendors to train, 40% vendors to test
    if medium_volume_vendors:
        np.random.shuffle(medium_volume_vendors)
        n_train_vendors = max(1, int(len(medium_volume_vendors) * 0.6))
        
        train_medium_vendors = medium_volume_vendors[:n_train_vendors]
        test_medium_vendors = medium_volume_vendors[n_train_vendors:]
        
        for vendor in train_medium_vendors:
            vendor_scripts = positives[positives['vendor'] == vendor].index.tolist()
            train_pos_indices.extend(vendor_scripts)
            print(f"  {vendor}: {len(vendor_scripts)} scripts → all to train")
        
        for vendor in test_medium_vendors:
            vendor_scripts = positives[positives['vendor'] == vendor].index.tolist()
            test_pos_indices.extend(vendor_scripts)
            print(f"  {vendor}: {len(vendor_scripts)} scripts → all to test")
    
    # Low volume vendors: 50% to train, 50% to test (by vendor)
    if low_volume_vendors:
        np.random.shuffle(low_volume_vendors)
        n_test_low_vendors = len(low_volume_vendors) // 2
        
        train_low_vendors = low_volume_vendors[n_test_low_vendors:]
        test_low_vendors = low_volume_vendors[:n_test_low_vendors]
        
        for vendor in train_low_vendors:
            vendor_scripts = positives[positives['vendor'] == vendor].index.tolist()
            train_pos_indices.extend(vendor_scripts)
            print(f"  {vendor}: {len(vendor_scripts)} scripts → all to train")
        
        for vendor in test_low_vendors:
            vendor_scripts = positives[positives['vendor'] == vendor].index.tolist()
            test_pos_indices.extend(vendor_scripts)
            print(f"  {vendor}: {len(vendor_scripts)} scripts → all to test")
    
    # Split negatives randomly (they don't have vendor info, so simple random split)
    neg_indices = negatives.index.tolist()
    np.random.shuffle(neg_indices)
    n_test_neg = int(len(neg_indices) * test_size)
    
    train_neg_indices = neg_indices[n_test_neg:]
    test_neg_indices = neg_indices[:n_test_neg]
    
    print(f"\nNormal scripts (label=0): {len(train_neg_indices)} train, {len(test_neg_indices)} test (random split)")
    
    # Combine indices
    train_indices = train_pos_indices + train_neg_indices
    test_indices = test_pos_indices + test_neg_indices
    
    print(f"\nFinal dataset split:")
    print(f"Train: {len(train_pos_indices)} behavioral + {len(train_neg_indices)} normal = {len(train_indices)} total")
    print(f"Test:  {len(test_pos_indices)} behavioral + {len(test_neg_indices)} normal = {len(test_indices)} total")
    
    return train_indices, test_indices

# ────────────────────────────────────────────────────────────────────
# 6.  Scoring function with tunable weights + threshold
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

def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'precision': precision, 'recall': recall, 
        'f1': f1, 'accuracy': accuracy
    }

# ────────────────────────────────────────────────────────────────────
# 7.  Create vendor-aware train/test split
# ────────────────────────────────────────────────────────────────────
train_indices, test_indices = create_vendor_aware_split(df, test_size=0.3, random_state=42)

train_df = df.loc[train_indices].copy()
test_df = df.loc[test_indices].copy()

print(f"\nTraining set vendor distribution (behavioral scripts only):")
train_pos_vendors = train_df[train_df['label'] == 1]['vendor'].value_counts()
print(train_pos_vendors.head(10))

print(f"\nTest set vendor distribution (behavioral scripts only):")
test_pos_vendors = test_df[test_df['label'] == 1]['vendor'].value_counts()
print(test_pos_vendors.head(10))

# Check for vendor overlap in behavioral scripts
train_vendors = set(train_df[train_df['label'] == 1]['vendor'].unique())
test_vendors = set(test_df[test_df['label'] == 1]['vendor'].unique())
overlap = train_vendors.intersection(test_vendors)
print(f"\nVendor overlap in behavioral scripts: {len(overlap)} vendors")
if overlap:
    print("Overlapping vendors (expected for high-volume vendors):", sorted(list(overlap))[:10])

# ────────────────────────────────────────────────────────────────────
# 8.  Grid‑search on training set only
# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# 8.  Parallel grid‑search on training set
# ────────────────────────────────────────────────────────────────────

def evaluate_single_combination(args):
    """
    Evaluate a single weight/threshold combination
    This function will be called in parallel
    """
    weights, threshold, train_data = args
    
    # Calculate predictions for this combination
    scores = []
    for _, row in train_data.iterrows():
        score = 0
        # counts
        if row.behavioral_source_api_count >= 4:
            score += weights["bc"]
        if row.fingerprinting_source_api_count >= 2:
            score += weights["fc"]
        # ratio
        if row.behav_fp_ratio >= 0.6:
            score += weights["ratio"]
        # aggregation
        if (row.max_behavioral_api_aggregation_score or 0) >= 5:
            score += weights["agg"]
        # burst / normalised access count
        if row.max_burst_norm >= 0.02:
            score += weights["burst"]
        # sinks
        if row.writes_storage or row.uses_beacon or row.script_inject:
            score += weights["sink"]
        # high‑signal event types
        if any([row.has_devicemotion, row.has_orientation,
                row.has_touch_event, row.has_keyboard_event]):
            score += weights["flag"]
        
        scores.append(int(score >= threshold))
    
    preds = np.array(scores)
    y_true = train_data['label'].values
    
    # Calculate metrics
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return {
        'weights': weights,
        'threshold': threshold,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'precision': precision, 'recall': recall, 
        'f1': f1, 'accuracy': accuracy
    }

def generate_all_combinations():
    """Generate all weight/threshold combinations"""
    weight_options = {
        "bc"   : [1, 2, 3],
        "fc"   : [1, 2],        
        "ratio": [1, 2, 3],
        "agg"  : [1, 2, 3],
        "burst": [1, 2, 3],
        "sink" : [2, 3, 4],
        "flag" : [1, 2, 3],
    }
    thresholds = [5, 6, 7, 8, 9, 10]
    
    combinations = []
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
                                    combinations.append((weights, th))
    return combinations

# Generate all combinations
all_combinations = generate_all_combinations()
total_combinations = len(all_combinations)

print(f"\nStarting parallel grid search on training set ({len(train_df)} samples)...")
print(f"Total combinations to test: {total_combinations}")
print(f"Available CPU cores: {cpu_count()}")
print(f"Using {min(cpu_count(), 160)} cores for parallel processing")

# Prepare data for parallel processing
# Convert train_df to a format that can be pickled and shared across processes
train_data_for_parallel = train_df.copy()

# Create argument tuples for parallel processing
args_list = [(weights, threshold, train_data_for_parallel) 
             for weights, threshold in all_combinations]

# Run parallel grid search
start_time = time.time()
n_cores = min(cpu_count(), 160)  # Use all available cores up to 160

print(f"Starting parallel evaluation with {n_cores} processes...")

with Pool(processes=n_cores) as pool:
    # Use chunksize to balance load across cores
    chunksize = max(1, total_combinations // (n_cores * 4))
    results = pool.map(evaluate_single_combination, args_list, chunksize=chunksize)

end_time = time.time()
print(f"Grid search completed in {end_time - start_time:.2f} seconds")

# Find best result
best = max(results, key=lambda x: x['f1'])
best_f1 = best['f1']

print(f"Evaluated {len(results)} combinations")
print(f"Best F1 score: {best_f1:.3f}")

# ────────────────────────────────────────────────────────────────────
# 9.  Evaluate best configuration on test set
# ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING SET RESULTS (used for hyperparameter tuning)")
print("="*60)
print(f"Best weights   : {best['weights']}")
print(f"Best threshold : {best['threshold']}")
print(f"Training F1    : {best['f1']:.3f}")
print(f"Training Precision: {best['precision']:.3f}")
print(f"Training Recall   : {best['recall']:.3f}")
print(f"Training Accuracy : {best['accuracy']:.3f}")

# Test on held-out test set using vectorized operations for speed
def vectorized_point_score(df, weights, threshold):
    """Vectorized version of point_score for faster evaluation"""
    scores = np.zeros(len(df))
    
    # Vectorized conditions
    bc_mask = df['behavioral_source_api_count'] >= 4
    fc_mask = df['fingerprinting_source_api_count'] >= 2
    ratio_mask = df['behav_fp_ratio'] >= 0.6
    agg_mask = df['max_behavioral_api_aggregation_score'].fillna(0) >= 5
    burst_mask = df['max_burst_norm'] >= 0.02
    sink_mask = df['writes_storage'] | df['uses_beacon'] | df['script_inject']
    flag_mask = df['has_devicemotion'] | df['has_orientation'] | df['has_touch_event'] | df['has_keyboard_event']
    
    # Apply weights
    scores += bc_mask * weights['bc']
    scores += fc_mask * weights['fc']
    scores += ratio_mask * weights['ratio']
    scores += agg_mask * weights['agg']
    scores += burst_mask * weights['burst']
    scores += sink_mask * weights['sink']
    scores += flag_mask * weights['flag']
    
    return (scores >= threshold).astype(int)

test_preds = vectorized_point_score(test_df, best['weights'], best['threshold'])
test_metrics = evaluate_predictions(test_df['label'], test_preds)

print("\n" + "="*60)
print("TEST SET RESULTS (unbiased evaluation)")
print("="*60)
print(f"Test F1        : {test_metrics['f1']:.3f}")
print(f"Test Precision : {test_metrics['precision']:.3f}")
print(f"Test Recall    : {test_metrics['recall']:.3f}")
print(f"Test Accuracy  : {test_metrics['accuracy']:.3f}")

print(f"\nConfusion Matrix (Test Set):")
print(f"TP = {test_metrics['TP']}")
print(f"FP = {test_metrics['FP']}")
print(f"FN = {test_metrics['FN']}")
print(f"TN = {test_metrics['TN']}")

print(f"\nPerformance Drop (Train → Test):")
print(f"F1 drop        : {best['f1'] - test_metrics['f1']:.3f}")
print(f"Precision drop : {best['precision'] - test_metrics['precision']:.3f}")
print(f"Recall drop    : {best['recall'] - test_metrics['recall']:.3f}")
print(f"Accuracy drop  : {best['accuracy'] - test_metrics['accuracy']:.3f}")

# ────────────────────────────────────────────────────────────────────
# 10. Analyze performance by vendor in test set
# ────────────────────────────────────────────────────────────────────
print(f"\n" + "="*60)
print("TEST SET VENDOR ANALYSIS")
print("="*60)

test_pos_df = test_df[test_df['label'] == 1].copy()
test_pos_df['prediction'] = vectorized_point_score(test_pos_df, best['weights'], best['threshold'])

vendor_performance = []
for vendor in test_pos_df['vendor'].unique():
    vendor_data = test_pos_df[test_pos_df['vendor'] == vendor]
    n_scripts = len(vendor_data)
    n_correct = (vendor_data['prediction'] == 1).sum()
    recall = n_correct / n_scripts
    vendor_performance.append({
        'vendor': vendor,
        'n_scripts': n_scripts,
        'n_correct': n_correct,
        'recall': recall
    })

vendor_df = pd.DataFrame(vendor_performance).sort_values('n_scripts', ascending=False)
print(f"Per-vendor recall on behavioral test samples:")
print(vendor_df.head(15).to_string(index=False, float_format='%.3f'))

avg_recall = vendor_df['recall'].mean()
print(f"\nAverage per-vendor recall: {avg_recall:.3f}")
print(f"Vendors with 100% recall: {(vendor_df['recall'] == 1.0).sum()}/{len(vendor_df)}")
print(f"Vendors with 0% recall: {(vendor_df['recall'] == 0.0).sum()}/{len(vendor_df)}")