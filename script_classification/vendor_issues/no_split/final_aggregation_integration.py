#!/usr/bin/env python3
"""
Final integration of aggregation features with original features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import psycopg2
import json
import warnings
warnings.filterwarnings('ignore')

def load_complete_data():
    """Load both aggregation and original features"""
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
        -- Aggregation features
        max_api_aggregation_score,
        behavioral_api_agg_count,
        fp_api_agg_count,
        max_aggregated_apis,
        aggregated_behavioral_apis,
        aggregated_fingerprinting_apis,
        attached_listeners,
        dataflow_to_sink,
        apis_going_to_sink,
        -- Original features
        fingerprinting_source_apis,
        behavioral_source_apis,
        behavioral_apis_access_count,
        fingerprinting_api_access_count,
        behavioral_source_api_count,
        fingerprinting_source_api_count,
        label,
        vendor
    FROM multicore_static_info_known_companies
    WHERE label IN (0, 1)
    LIMIT 1000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} scripts for comparison")
    return df

def create_aggregation_features(df):
    """Create the best aggregation features"""
    features_list = []
    
    for idx, row in df.iterrows():
        try:
            features = {}
            
            # Best aggregation features identified
            max_agg = row['max_api_aggregation_score'] if row['max_api_aggregation_score'] != -1 else 0
            behavioral_agg = row['behavioral_api_agg_count'] if row['behavioral_api_agg_count'] != -1 else 0
            fp_agg = row['fp_api_agg_count'] if row['fp_api_agg_count'] != -1 else 0
            
            features['agg_max_api_aggregation_score'] = max_agg
            features['agg_behavioral_api_agg_count'] = behavioral_agg
            features['agg_fp_api_agg_count'] = fp_agg
            features['agg_total_aggregation_count'] = behavioral_agg + fp_agg
            features['agg_has_aggregation'] = int(max_agg > 0)
            
            # Aggregation ratios
            total_agg = behavioral_agg + fp_agg
            if total_agg > 0:
                features['agg_behavioral_ratio'] = behavioral_agg / total_agg
                features['agg_fp_ratio'] = fp_agg / total_agg
            else:
                features['agg_behavioral_ratio'] = 0
                features['agg_fp_ratio'] = 0
            
            # Aggregation complexity tiers
            if max_agg == 0:
                features['agg_complexity_tier'] = 0
            elif max_agg <= 5:
                features['agg_complexity_tier'] = 1
            elif max_agg <= 15:
                features['agg_complexity_tier'] = 2
            else:
                features['agg_complexity_tier'] = 3
            
            # Dataflow features - handle boolean arrays properly
            dataflow_value = row['dataflow_to_sink']
            if pd.isna(dataflow_value):
                features['agg_has_dataflow_to_sink'] = 0
            elif isinstance(dataflow_value, (list, np.ndarray)):
                features['agg_has_dataflow_to_sink'] = int(any(dataflow_value) if len(dataflow_value) > 0 else False)
            else:
                features['agg_has_dataflow_to_sink'] = int(bool(dataflow_value))
            
            # Store metadata
            features['script_id'] = int(row['script_id'])
            features['label'] = int(row['label'])
            features['vendor'] = row['vendor'] if pd.notna(row['vendor']) else 'negative'
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing script {row.get('script_id', 'unknown')}: {e}")
            continue
    
    return pd.DataFrame(features_list)

def create_original_features(df):
    """Create original static features"""
    features_list = []
    
    for idx, row in df.iterrows():
        try:
            features = {}
            
            # Original behavioral features
            behavioral_sources = row['behavioral_source_apis'] if pd.notna(row['behavioral_source_apis']) else []
            fp_sources = row['fingerprinting_source_apis'] if pd.notna(row['fingerprinting_source_apis']) else []
            
            # Convert to lists if strings
            if isinstance(behavioral_sources, str):
                try:
                    behavioral_sources = json.loads(behavioral_sources)
                except:
                    behavioral_sources = []
            if isinstance(fp_sources, str):
                try:
                    fp_sources = json.loads(fp_sources)
                except:
                    fp_sources = []
            
            # Basic counts
            behavioral_count = len(behavioral_sources) if behavioral_sources else 0
            fp_count = len(fp_sources) if fp_sources else 0
            total_apis = behavioral_count + fp_count
            
            features['orig_behavioral_api_count'] = behavioral_count
            features['orig_fp_api_count'] = fp_count
            features['orig_total_api_count'] = total_apis
            
            # Ratios
            if total_apis > 0:
                features['orig_behavioral_ratio'] = behavioral_count / total_apis
                features['orig_fp_ratio'] = fp_count / total_apis
            else:
                features['orig_behavioral_ratio'] = 0
                features['orig_fp_ratio'] = 0
            
            # Specific API checks
            if behavioral_sources:
                api_strings = [str(api) for api in behavioral_sources]
                features['orig_tracks_mouse'] = int(any('MouseEvent' in api for api in api_strings))
                features['orig_tracks_keyboard'] = int(any('KeyboardEvent' in api for api in api_strings))
                features['orig_tracks_touch'] = int(any('TouchEvent' in api for api in api_strings))
            else:
                features['orig_tracks_mouse'] = 0
                features['orig_tracks_keyboard'] = 0
                features['orig_tracks_touch'] = 0
            
            if fp_sources:
                api_strings = [str(api) for api in fp_sources]
                features['orig_uses_canvas'] = int(any('Canvas' in api or 'WebGL' in api for api in api_strings))
                features['orig_uses_navigator'] = int(any('Navigator' in api for api in api_strings))
            else:
                features['orig_uses_canvas'] = 0
                features['orig_uses_navigator'] = 0
            
            # Interaction diversity
            event_types = 0
            if behavioral_sources:
                api_strings = [str(api) for api in behavioral_sources]
                if any('MouseEvent' in api for api in api_strings):
                    event_types += 1
                if any('KeyboardEvent' in api for api in api_strings):
                    event_types += 1
                if any('TouchEvent' in api for api in api_strings):
                    event_types += 1
            
            features['orig_interaction_diversity'] = event_types
            
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
    print("=== FINAL AGGREGATION FEATURES INTEGRATION ===")
    
    # Load data
    print("\n1. Loading complete dataset...")
    df = load_complete_data()
    
    # Create feature sets
    print("\n2. Creating feature sets...")
    agg_df = create_aggregation_features(df)
    orig_df = create_original_features(df)
    
    # Ensure same samples
    common_ids = set(agg_df['script_id']).intersection(set(orig_df['script_id']))
    agg_df = agg_df[agg_df['script_id'].isin(common_ids)].sort_values('script_id').reset_index(drop=True)
    orig_df = orig_df[orig_df['script_id'].isin(common_ids)].sort_values('script_id').reset_index(drop=True)
    
    print(f"Final sample size: {len(agg_df)}")
    print(f"Positive samples: {len(agg_df[agg_df['label']==1])}")
    print(f"Negative samples: {len(agg_df[agg_df['label']==0])}")
    
    # Feature columns
    agg_features = [col for col in agg_df.columns if col.startswith('agg_')]
    orig_features = [col for col in orig_df.columns if col.startswith('orig_')]
    
    print(f"Aggregation features: {len(agg_features)}")
    print(f"Original features: {len(orig_features)}")
    
    # Combine features
    print("\n3. Combining features...")
    combined_df = agg_df[['script_id', 'label', 'vendor']].copy()
    
    # Add all features
    for feature in agg_features:
        combined_df[feature] = agg_df[feature]
    for feature in orig_features:
        combined_df[feature] = orig_df[feature]
    
    all_features = agg_features + orig_features
    
    # Model comparison
    print("\n4. Model comparison...")
    X_agg = agg_df[agg_features]
    X_orig = orig_df[orig_features]
    X_combined = combined_df[all_features]
    y = agg_df['label']
    
    # Cross-validation comparison
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Test individual feature sets
    agg_scores = cross_val_score(rf, X_agg, y, cv=5, scoring='roc_auc')
    orig_scores = cross_val_score(rf, X_orig, y, cv=5, scoring='roc_auc')
    combined_scores = cross_val_score(rf, X_combined, y, cv=5, scoring='roc_auc')
    
    print(f"\nPerformance Comparison (ROC AUC):")
    print(f"Aggregation features only: {agg_scores.mean():.4f} (+/- {agg_scores.std()*2:.4f})")
    print(f"Original features only:     {orig_scores.mean():.4f} (+/- {orig_scores.std()*2:.4f})")
    print(f"Combined features:          {combined_scores.mean():.4f} (+/- {combined_scores.std()*2:.4f})")
    
    # Improvements
    agg_improvement = agg_scores.mean() - orig_scores.mean()
    combined_improvement = combined_scores.mean() - orig_scores.mean()
    
    print(f"\nImprovements:")
    print(f"Aggregation vs Original: {agg_improvement:+.4f}")
    print(f"Combined vs Original: {combined_improvement:+.4f}")
    
    # Feature importance in combined model
    print("\n5. Feature importance in combined model...")
    rf.fit(X_combined, y)
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 features in combined model:")
    for idx, row in feature_importance.head(15).iterrows():
        feature_type = "AGG" if row['feature'].startswith('agg_') else "ORIG"
        clean_name = row['feature'].replace('agg_', '').replace('orig_', '')
        print(f"{feature_type:<5} {clean_name:<35} {row['importance']:.4f}")
    
    # Count feature types in top features
    top_15 = feature_importance.head(15)
    agg_in_top = sum(1 for f in top_15['feature'] if f.startswith('agg_'))
    orig_in_top = sum(1 for f in top_15['feature'] if f.startswith('orig_'))
    
    print(f"\nTop 15 feature composition:")
    print(f"  Aggregation features: {agg_in_top}/15 ({agg_in_top/15*100:.1f}%)")
    print(f"  Original features: {orig_in_top}/15 ({orig_in_top/15*100:.1f}%)")
    
    # Final recommendations
    print(f"\n6. Final recommendations...")
    
    if agg_improvement > 0.02:
        print(f"✅ Aggregation features provide significant improvement (+{agg_improvement:.4f})")
    elif agg_improvement > 0.005:
        print(f"✅ Aggregation features provide modest improvement (+{agg_improvement:.4f})")
    else:
        print(f"⚠️  Aggregation features provide minimal improvement (+{agg_improvement:.4f})")
    
    if combined_improvement > max(agg_improvement, 0.01):
        print(f"✅ Combined features are recommended for best performance")
    elif agg_improvement > 0.01:
        print(f"✅ Use aggregation features - they outperform original features")
    else:
        print(f"⚠️  Stick with original features")
    
    # Best aggregation features for integration
    print(f"\n7. Best aggregation features for integration:")
    agg_only = feature_importance[feature_importance['feature'].str.startswith('agg_')]
    print(f"Top aggregation features by importance:")
    for idx, row in agg_only.head(10).iterrows():
        clean_name = row['feature'].replace('agg_', '')
        print(f"  - {clean_name:<35} {row['importance']:.4f}")
    
    print(f"\n✅ Final integration analysis complete!")
    print(f"📊 Aggregation features are {'highly valuable' if agg_improvement > 0.02 else 'moderately valuable' if agg_improvement > 0.005 else 'minimally valuable'} for your behavioral biometrics detection model.")

if __name__ == "__main__":
    main()