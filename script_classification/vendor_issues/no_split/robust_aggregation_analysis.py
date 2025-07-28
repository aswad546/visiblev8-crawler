#!/usr/bin/env python3
"""
Robust aggregation features analysis - simplified version
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import psycopg2
import json
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load data with core aggregation features only"""
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
        label,
        vendor
    FROM multicore_static_info_known_companies
    WHERE label IN (0, 1)
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} scripts")
    return df

def create_aggregation_features(df):
    """Create robust aggregation features"""
    features_list = []
    
    for _, row in df.iterrows():
        features = {}
        
        # Handle aggregation scores (convert -1 to 0)
        max_agg = row['max_api_aggregation_score']
        max_agg = 0 if (pd.isna(max_agg) or max_agg == -1) else max_agg
        
        behavioral_agg = row['behavioral_api_agg_count'] 
        behavioral_agg = 0 if (pd.isna(behavioral_agg) or behavioral_agg == -1) else behavioral_agg
        
        fp_agg = row['fp_api_agg_count']
        fp_agg = 0 if (pd.isna(fp_agg) or fp_agg == -1) else fp_agg
        
        # Core aggregation features
        features['max_api_aggregation_score'] = max_agg
        features['behavioral_api_agg_count'] = behavioral_agg
        features['fp_api_agg_count'] = fp_agg
        features['total_aggregation_count'] = behavioral_agg + fp_agg
        features['has_aggregation'] = int(max_agg > 0)
        
        # Derived features
        total_agg = behavioral_agg + fp_agg
        if total_agg > 0:
            features['behavioral_agg_ratio'] = behavioral_agg / total_agg
            features['fp_agg_ratio'] = fp_agg / total_agg
        else:
            features['behavioral_agg_ratio'] = 0
            features['fp_agg_ratio'] = 0
        
        # Aggregation intensity categories
        features['has_behavioral_aggregation'] = int(behavioral_agg > 0)
        features['has_fp_aggregation'] = int(fp_agg > 0)
        features['has_both_aggregation_types'] = int(behavioral_agg > 0 and fp_agg > 0)
        
        # Store metadata
        features['script_id'] = int(row['script_id'])
        features['label'] = int(row['label'])
        features['vendor'] = row['vendor'] if pd.notna(row['vendor']) else 'negative'
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def analyze_features(df):
    """Analyze aggregation features"""
    print("\n=== AGGREGATION FEATURES ANALYSIS ===")
    
    # Feature columns
    feature_cols = [col for col in df.columns if col not in ['script_id', 'label', 'vendor']]
    
    # Split by label
    positive = df[df['label'] == 1]
    negative = df[df['label'] == 0]
    
    print(f"\nDataset: {len(df)} samples")
    print(f"Positive: {len(positive)} samples")
    print(f"Negative: {len(negative)} samples")
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"{'Feature':<30} {'Pos Mean':<10} {'Neg Mean':<10} {'Difference':<10}")
    print("-" * 65)
    
    feature_scores = []
    
    for feature in feature_cols:
        pos_mean = positive[feature].mean()
        neg_mean = negative[feature].mean()
        difference = abs(pos_mean - neg_mean)
        
        feature_scores.append({
            'feature': feature,
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'difference': difference
        })
        
        print(f"{feature:<30} {pos_mean:<10.3f} {neg_mean:<10.3f} {difference:<10.3f}")
    
    # Sort by difference
    feature_scores = sorted(feature_scores, key=lambda x: x['difference'], reverse=True)
    
    print(f"\n🎯 TOP DISCRIMINATIVE FEATURES:")
    for i, score in enumerate(feature_scores[:5], 1):
        print(f"  {i}. {score['feature']:<30} (diff: {score['difference']:.3f})")
    
    return feature_cols, feature_scores

def test_model_performance(df, feature_cols):
    """Test ML performance"""
    print(f"\n🤖 MACHINE LEARNING PERFORMANCE:")
    
    X = df[feature_cols]
    y = df['label']
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    scores = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')
    
    print(f"5-fold Cross-Validation ROC AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    # Feature importance
    rf.fit(X, y)
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔥 FEATURE IMPORTANCE:")
    for _, row in importances.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    return scores.mean(), importances

def main():
    """Main execution"""
    print("=== ROBUST AGGREGATION FEATURES ANALYSIS ===")
    
    # Load and process data
    df = load_data()
    agg_df = create_aggregation_features(df)
    
    # Analyze features
    feature_cols, feature_scores = analyze_features(agg_df)
    
    # Test performance
    auc_score, importances = test_model_performance(agg_df, feature_cols)
    
    # Summary
    print(f"\n✅ SUMMARY:")
    print(f"  - Aggregation features created: {len(feature_cols)}")
    print(f"  - Cross-validation ROC AUC: {auc_score:.4f}")
    
    if auc_score > 0.90:
        print(f"  - ✅ EXCELLENT performance - Aggregation features are highly valuable!")
    elif auc_score > 0.80:
        print(f"  - ✅ GOOD performance - Aggregation features are valuable")
    elif auc_score > 0.70:
        print(f"  - ⚠️  MODERATE performance - Aggregation features may be useful")
    else:
        print(f"  - ❌ POOR performance - Aggregation features need improvement")
    
    print(f"\n🎯 RECOMMENDED FEATURES FOR INTEGRATION:")
    top_features = importances.head(5)
    for i, row in top_features.iterrows():
        print(f"  - {row['feature']:<30} (importance: {row['importance']:.4f})")

if __name__ == "__main__":
    main()