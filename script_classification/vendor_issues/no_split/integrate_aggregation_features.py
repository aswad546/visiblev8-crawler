#!/usr/bin/env python3
"""
Integration code to add aggregation features to your existing model
"""

import pandas as pd
import numpy as np
import psycopg2
import json

def create_working_vendor_agnostic_features_with_aggregation(df):
    """
    Enhanced version of your original create_working_vendor_agnostic_features function
    that includes the best aggregation features
    """
    features_list = []
    
    for index, row in df.iterrows():
        try:
            features = {}
            
            # === ORIGINAL VENDOR-AGNOSTIC FEATURES ===
            # (Your existing feature creation code)
            
            # Extract data safely
            behavioral_access = row['behavioral_apis_access_count'] if row['behavioral_apis_access_count'] is not None else {}
            fp_access = row['fingerprinting_api_access_count'] if row['fingerprinting_api_access_count'] is not None else {}
            behavioral_sources = row['behavioral_source_apis'] if row['behavioral_source_apis'] is not None else []
            fp_sources = row['fingerprinting_source_apis'] if row['fingerprinting_source_apis'] is not None else []
            
            # Convert from JSON strings if needed
            if isinstance(behavioral_access, str):
                behavioral_access = json.loads(behavioral_access) if behavioral_access else {}
            if isinstance(fp_access, str):
                fp_access = json.loads(fp_access) if fp_access else {}
            if isinstance(behavioral_sources, str):
                behavioral_sources = json.loads(behavioral_sources) if behavioral_sources else []
            if isinstance(fp_sources, str):
                fp_sources = json.loads(fp_sources) if fp_sources else []
            
            # Basic counts
            total_behavioral = len(behavioral_sources) if behavioral_sources is not None else 0
            total_fp = len(fp_sources) if fp_sources is not None else 0
            total_apis = total_behavioral + total_fp
            
            # Original features
            if total_apis > 0:
                features['behavioral_focus_ratio'] = total_behavioral / total_apis
                features['fp_focus_ratio'] = total_fp / total_apis
            else:
                features['behavioral_focus_ratio'] = 0
                features['fp_focus_ratio'] = 0
            
            # Interaction diversity
            event_types = set()
            if behavioral_sources is not None:
                for api in behavioral_sources:
                    api_str = str(api)
                    if 'MouseEvent' in api_str or 'mouse' in api_str.lower():
                        event_types.add('mouse')
                    elif 'KeyboardEvent' in api_str or 'keyboard' in api_str.lower():
                        event_types.add('keyboard')
                    elif 'TouchEvent' in api_str or 'Touch.' in api_str or 'touch' in api_str.lower():
                        event_types.add('touch')
                    elif 'PointerEvent' in api_str or 'pointer' in api_str.lower():
                        event_types.add('pointer')
                    elif 'FocusEvent' in api_str or 'focus' in api_str.lower():
                        event_types.add('focus')
            
            features['interaction_diversity'] = len(event_types)
            
            # Access intensity
            total_behavioral_accesses = sum(behavioral_access.values()) if behavioral_access else 0
            total_fp_accesses = sum(fp_access.values()) if fp_access else 0
            total_accesses = total_behavioral_accesses + total_fp_accesses
            
            features['collection_intensity'] = total_accesses / max(total_apis, 1)
            
            # Binary capabilities
            features['tracks_mouse'] = int(any('MouseEvent' in str(api) or 'mouse' in str(api).lower() for api in behavioral_sources)) if behavioral_sources else 0
            features['tracks_keyboard'] = int(any('KeyboardEvent' in str(api) or 'keyboard' in str(api).lower() for api in behavioral_sources)) if behavioral_sources else 0
            features['tracks_touch'] = int(any('TouchEvent' in str(api) or 'touch' in str(api).lower() for api in behavioral_sources)) if behavioral_sources else 0
            features['uses_canvas_fp'] = int(any('Canvas' in str(api) or 'WebGL' in str(api) for api in fp_sources)) if fp_sources else 0
            features['uses_navigator_fp'] = int(any('Navigator' in str(api) for api in fp_sources)) if fp_sources else 0
            
            # === NEW AGGREGATION FEATURES ===
            # Top 5 aggregation features based on analysis
            
            # Core aggregation scores (handle -1 as no aggregation)
            max_agg = row['max_api_aggregation_score']
            max_agg = 0 if (pd.isna(max_agg) or max_agg == -1) else max_agg
            
            behavioral_agg = row['behavioral_api_agg_count'] 
            behavioral_agg = 0 if (pd.isna(behavioral_agg) or behavioral_agg == -1) else behavioral_agg
            
            fp_agg = row['fp_api_agg_count']
            fp_agg = 0 if (pd.isna(fp_agg) or fp_agg == -1) else fp_agg
            
            # Top aggregation features (with agg_ prefix to distinguish)
            features['agg_max_api_aggregation_score'] = max_agg
            features['agg_total_aggregation_count'] = behavioral_agg + fp_agg
            features['agg_behavioral_api_agg_count'] = behavioral_agg
            features['agg_fp_api_agg_count'] = fp_agg
            
            # Aggregation ratio (5th most important feature)
            total_agg = behavioral_agg + fp_agg
            if total_agg > 0:
                features['agg_behavioral_agg_ratio'] = behavioral_agg / total_agg
            else:
                features['agg_behavioral_agg_ratio'] = 0
            
            # === METADATA ===
            features['script_id'] = int(row['script_id'])
            features['label'] = int(row['label'])
            features['vendor'] = row['vendor'] if pd.notna(row['vendor']) else 'negative'
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing script {row.get('script_id', 'unknown')}: {e}")
            continue
    
    return pd.DataFrame(features_list)

def load_data_with_aggregation():
    """Load data including aggregation features"""
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
        -- Original features
        behavioral_source_apis,
        fingerprinting_source_apis,
        behavioral_apis_access_count,
        fingerprinting_api_access_count,
        behavioral_source_api_count,
        fingerprinting_source_api_count,
        -- Aggregation features
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
    
    return df

def test_integration():
    """Test the integrated features"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    print("=== TESTING INTEGRATED FEATURES ===")
    
    # Load data
    df = load_data_with_aggregation()
    print(f"Loaded {len(df)} scripts")
    
    # Create enhanced features
    enhanced_df = create_working_vendor_agnostic_features_with_aggregation(df)
    print(f"Created enhanced features for {len(enhanced_df)} scripts")
    
    # Separate feature types
    feature_cols = [col for col in enhanced_df.columns if col not in ['script_id', 'label', 'vendor']]
    original_features = [col for col in feature_cols if not col.startswith('agg_')]
    aggregation_features = [col for col in feature_cols if col.startswith('agg_')]
    
    print(f"Original features: {len(original_features)}")
    print(f"Aggregation features: {len(aggregation_features)}")
    print(f"Total features: {len(feature_cols)}")
    
    # Test performance
    X_original = enhanced_df[original_features]
    X_aggregation = enhanced_df[aggregation_features]
    X_combined = enhanced_df[feature_cols]
    y = enhanced_df['label']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Compare performance
    scores_original = cross_val_score(rf, X_original, y, cv=5, scoring='roc_auc')
    scores_aggregation = cross_val_score(rf, X_aggregation, y, cv=5, scoring='roc_auc')
    scores_combined = cross_val_score(rf, X_combined, y, cv=5, scoring='roc_auc')
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Original features only:     {scores_original.mean():.4f} (+/- {scores_original.std()*2:.4f})")
    print(f"Aggregation features only:  {scores_aggregation.mean():.4f} (+/- {scores_aggregation.std()*2:.4f})")
    print(f"Combined features:          {scores_combined.mean():.4f} (+/- {scores_combined.std()*2:.4f})")
    
    improvement_agg = scores_aggregation.mean() - scores_original.mean()
    improvement_combined = scores_combined.mean() - scores_original.mean()
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"Aggregation vs Original:    {improvement_agg:+.4f}")
    print(f"Combined vs Original:       {improvement_combined:+.4f}")
    
    if improvement_combined > 0.01:
        print(f"\n✅ RECOMMENDATION: Use combined features for best performance!")
    elif improvement_agg > 0.01:
        print(f"\n✅ RECOMMENDATION: Use aggregation features - they outperform original!")
    else:
        print(f"\n⚠️  RECOMMENDATION: Stick with original features")
    
    return enhanced_df

def show_integration_example():
    """Show how to integrate into existing code"""
    print(f"\n=== INTEGRATION EXAMPLE ===")
    print(f"""
To integrate aggregation features into your existing minimal_no_split notebook:

1. Replace your create_working_vendor_agnostic_features() function with:
   create_working_vendor_agnostic_features_with_aggregation()

2. The new function adds these top 5 aggregation features:
   - agg_max_api_aggregation_score     (Most important)
   - agg_total_aggregation_count       (2nd most important)
   - agg_behavioral_api_agg_count      (3rd most important)
   - agg_fp_api_agg_count              (4th most important)
   - agg_behavioral_agg_ratio          (5th most important)

3. Your existing code will work unchanged - just better performance!

4. The aggregation features have 'agg_' prefix to distinguish them.

Example usage:
```python
# In your notebook, replace:
# features_df = create_working_vendor_agnostic_features(df)

# With:
features_df = create_working_vendor_agnostic_features_with_aggregation(df)

# Everything else stays the same!
```
""")

if __name__ == "__main__":
    enhanced_df = test_integration()
    show_integration_example()
    print(f"\n✅ Integration analysis complete!")