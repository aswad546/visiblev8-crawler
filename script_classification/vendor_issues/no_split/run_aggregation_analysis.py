#!/usr/bin/env python3
"""
Execute aggregation features analysis directly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import psycopg2
import json
import warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

def load_aggregation_data():
    """Load aggregation features from database"""
    conn = psycopg2.connect(
        host="localhost",
        port=5434,
        database="vv8_backend",
        user="vv8",
        password="vv8"
    )
    
    query = """
    SELECT 
        script_id,
        max_api_aggregation_score,
        behavioral_api_agg_count,
        fp_api_agg_count,
        max_aggregated_apis,
        max_behavioral_api_aggregation_score,
        aggregated_behavioral_apis,
        max_fingerprinting_api_aggregation_score,
        aggregated_fingerprinting_apis,
        attached_listeners,
        dataflow_to_sink,
        apis_going_to_sink,
        graph_construction_failure,
        label,
        vendor,
        fingerprinting_source_apis,
        behavioral_source_apis,
        behavioral_apis_access_count,
        fingerprinting_api_access_count
    FROM multicore_static_info_known_companies
    LIMIT 1000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} scripts with aggregation features")
    return df

def create_aggregation_features(df):
    """Create vendor-agnostic aggregation features"""
    features_list = []
    
    for idx, row in df.iterrows():
        try:
            features = {}
            
            # Basic aggregation scores
            max_agg = row['max_api_aggregation_score'] if row['max_api_aggregation_score'] != -1 else 0
            behavioral_agg = row['behavioral_api_agg_count'] if row['behavioral_api_agg_count'] != -1 else 0
            fp_agg = row['fp_api_agg_count'] if row['fp_api_agg_count'] != -1 else 0
            
            features['max_api_aggregation_score'] = max_agg
            features['behavioral_api_agg_count'] = behavioral_agg
            features['fp_api_agg_count'] = fp_agg
            features['total_aggregation_count'] = behavioral_agg + fp_agg
            features['has_aggregation'] = int(max_agg > 0)
            
            # Store metadata
            features['script_id'] = int(row['script_id'])
            features['label'] = int(row['label'])
            features['vendor'] = row['vendor'] if pd.notna(row['vendor']) else 'negative'
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing script {row.get('script_id', 'unknown')}: {e}")
            continue
    
    return pd.DataFrame(features_list)

def main():
    """Main execution function"""
    print("=== AGGREGATION FEATURES ANALYSIS ===")
    
    # Load data
    print("\n1. Loading data...")
    df = load_aggregation_data()
    
    # Create features
    print("\n2. Creating aggregation features...")
    agg_features_df = create_aggregation_features(df)
    
    # Filter to binary classification
    binary_df = agg_features_df[agg_features_df['label'].isin([0, 1])].copy()
    print(f"Binary classification samples: {len(binary_df)}")
    
    if len(binary_df) < 10:
        print("Not enough samples for analysis")
        return
    
    # Feature analysis
    print("\n3. Analyzing aggregation features...")
    feature_cols = [col for col in binary_df.columns if col not in ['script_id', 'label', 'vendor']]
    
    # Basic statistics
    positive_samples = binary_df[binary_df['label'] == 1]
    negative_samples = binary_df[binary_df['label'] == 0]
    
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    # Feature importance
    print("\n4. Feature importance analysis...")
    for feature in feature_cols:
        pos_mean = positive_samples[feature].mean()
        neg_mean = negative_samples[feature].mean()
        difference = abs(pos_mean - neg_mean)
        print(f"{feature}: pos={pos_mean:.3f}, neg={neg_mean:.3f}, diff={difference:.3f}")
    
    # Simple ML test
    print("\n5. Testing aggregation features with ML...")
    if len(feature_cols) > 0:
        X = binary_df[feature_cols]
        y = binary_df['label']
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(rf, X, y, cv=3, scoring='roc_auc')
        
        print(f"Cross-validation ROC AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
        
        # Feature importance
        rf.fit(X, y)
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop features by importance:")
        print(importances.head(10))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()