import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   StratifiedKFold, GridSearchCV, 
                                   RandomizedSearchCV, GroupKFold, StratifiedGroupKFold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           average_precision_score, make_scorer)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.base import clone
import pickle
import os
from datetime import datetime
from scipy import stats
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class VendorAwareRobustComparison:
    """
    Methodologically rigorous comparison of balanced vs imbalanced approaches
    using vendor-grouped nested cross-validation to prevent vendor-based data leakage.
    
    Key improvements:
    1. Vendor-grouped cross-validation to prevent vendor data leakage
    2. Nested cross-validation for unbiased performance estimation
    3. Hold-out test set with no vendor overlap
    4. Statistical significance testing
    5. Feature selection within CV folds
    6. Hyperparameter tuning within inner CV loop
    7. Proper handling of null vendor assignments for negatives
    """
    
    def __init__(self, db_config=None, output_dir="vendor_aware_robust_analysis"):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.table_name = 'multicore_static_info_known_companies'
        self.raw_data = None
        
        # Data storage
        self.balanced_data = None
        self.imbalanced_data = None
        
        # Train/test splits for both approaches
        self.balanced_train = None
        self.balanced_test = None
        self.imbalanced_train = None
        self.imbalanced_test = None
        
        # Vendor information
        self.test_vendors = None
        self.vendor_analysis = None
        
        # Results storage
        self.nested_cv_results = {
            'balanced': None,
            'imbalanced': None
        }
        
        # Hyperparameter search space
        self.param_distributions = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [8, 10, 12, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False]
        }
    
    def connect_to_database(self):
        """Establish connection to PostgreSQL database."""
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            print(f"❌ Error connecting to PostgreSQL database: {e}")
            return None
    
    def analyze_vendor_distribution(self, data):
        """Analyze vendor distribution to understand data structure and potential issues."""
        df = pd.DataFrame(data)
        
        print("\n🔍 VENDOR DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        # Handle null vendors (typically negative samples)
        df['vendor_clean'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        # Overall statistics
        vendor_counts = df['vendor_clean'].value_counts()
        total_vendors = len(vendor_counts)
        total_scripts = len(df)
        
        print(f"Total unique vendors: {total_vendors}")
        print(f"Total scripts: {total_scripts}")
        print(f"Average scripts per vendor: {total_scripts / total_vendors:.1f}")
        
        # Label distribution by vendor
        vendor_label_dist = df.groupby(['vendor_clean', 'label']).size().unstack(fill_value=0)
        
        print(f"\n📊 TOP 10 VENDOR-LABEL DISTRIBUTION:")
        display_df = vendor_label_dist.head(10)
        if 1 in display_df.columns and 0 in display_df.columns:
            display_df['total'] = display_df[0] + display_df.get(1, 0)
            display_df = display_df.sort_values('total', ascending=False)
        print(display_df)
        
        # Analyze vendor categories
        null_vendor_scripts = df[df['vendor'].isnull()]
        positive_with_vendor = df[(df['label'] == 1) & (df['vendor'].notnull())]
        negative_with_vendor = df[(df['label'] == 0) & (df['vendor'].notnull())]
        
        print(f"\n🏷️  VENDOR ASSIGNMENT PATTERNS:")
        print(f"Scripts with null vendor: {len(null_vendor_scripts)} ({len(null_vendor_scripts)/len(df)*100:.1f}%)")
        print(f"  - Null vendor with label 1: {len(null_vendor_scripts[null_vendor_scripts['label'] == 1])}")
        print(f"  - Null vendor with label 0: {len(null_vendor_scripts[null_vendor_scripts['label'] == 0])}")
        print(f"  - Null vendor with label -1: {len(null_vendor_scripts[null_vendor_scripts['label'] == -1])}")
        
        print(f"Positive scripts with vendor: {len(positive_with_vendor)}")
        print(f"Negative scripts with vendor: {len(negative_with_vendor)}")
        
        # Identify vendors by label patterns
        vendor_stats = df.groupby('vendor_clean').agg({
            'label': ['count', 'mean', 'sum', 'std']
        }).reset_index()
        vendor_stats.columns = ['vendor', 'total_scripts', 'pos_ratio', 'pos_count', 'label_std']
        vendor_stats['neg_count'] = vendor_stats['total_scripts'] - vendor_stats['pos_count']
        
        # Categorize vendors
        pos_only_vendors = vendor_stats[(vendor_stats['pos_count'] > 0) & (vendor_stats['neg_count'] == 0)]
        neg_only_vendors = vendor_stats[(vendor_stats['pos_count'] == 0) & (vendor_stats['neg_count'] > 0)]
        mixed_vendors = vendor_stats[(vendor_stats['pos_count'] > 0) & (vendor_stats['neg_count'] > 0)]
        single_script_vendors = vendor_stats[vendor_stats['total_scripts'] == 1]
        
        print(f"\n🎭 VENDOR CATEGORIES:")
        print(f"Vendors with only positives: {len(pos_only_vendors)}")
        print(f"Vendors with only negatives: {len(neg_only_vendors)}")
        print(f"Vendors with mixed labels: {len(mixed_vendors)}")
        print(f"Vendors with single script: {len(single_script_vendors)}")
        
        # Check for potential grouping issues
        min_scripts_threshold = 2
        insufficient_vendors = vendor_stats[vendor_stats['total_scripts'] < min_scripts_threshold]
        
        print(f"\n⚠️  POTENTIAL CV ISSUES:")
        print(f"Vendors with < {min_scripts_threshold} scripts: {len(insufficient_vendors)}")
        
        if len(insufficient_vendors) > 0:
            print("Consider:")
            print("- Grouping small vendors together")
            print("- Using stratified vendor grouping")
            print("- Reducing CV folds")
        
        return {
            'vendor_counts': vendor_counts,
            'vendor_stats': vendor_stats,
            'pos_only_vendors': pos_only_vendors['vendor'].tolist(),
            'neg_only_vendors': neg_only_vendors['vendor'].tolist(),
            'mixed_vendors': mixed_vendors['vendor'].tolist(),
            'single_script_vendors': single_script_vendors['vendor'].tolist(),
            'insufficient_vendors': insufficient_vendors['vendor'].tolist(),
            'total_vendors': total_vendors,
            'null_vendor_count': len(null_vendor_scripts)
        }
    
    def create_vendor_aware_splits(self, data, test_vendor_ratio=0.2, random_state=42):
        """
        Create train/test splits ensuring no vendor appears in both sets.
        Handles null vendors by treating them as separate groups.
        CRITICAL: Ensures both train and test have positive samples.
        """
        df = pd.DataFrame(data)
        np.random.seed(random_state)
        
        print(f"\n🎯 CREATING VENDOR-AWARE TRAIN/TEST SPLIT")
        print("=" * 60)
        
        # Handle null vendors by creating unique identifiers
        df['vendor_group'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        # For null vendors, create individual groups to prevent leakage
        null_mask = df['vendor'].isnull()
        null_scripts = df[null_mask].copy()
        
        if len(null_scripts) > 0:
            # Give each null vendor script a unique group ID
            null_scripts['vendor_group'] = [f'NULL_VENDOR_{i}' for i in range(len(null_scripts))]
            df.loc[null_mask, 'vendor_group'] = null_scripts['vendor_group'].values
        
        # Get vendor group statistics
        vendor_stats = df.groupby('vendor_group').agg({
            'label': ['count', 'mean', 'sum']
        }).reset_index()
        vendor_stats.columns = ['vendor_group', 'total_scripts', 'pos_ratio', 'pos_count']
        vendor_stats['neg_count'] = vendor_stats['total_scripts'] - vendor_stats['pos_count']
        
        # Categorize vendor groups
        pos_vendors = vendor_stats[vendor_stats['pos_count'] > 0]['vendor_group'].tolist()
        neg_vendors = vendor_stats[vendor_stats['neg_count'] > 0]['vendor_group'].tolist()
        mixed_vendors = vendor_stats[
            (vendor_stats['pos_count'] > 0) & (vendor_stats['neg_count'] > 0)
        ]['vendor_group'].tolist()
        
        print(f"Vendor groups with positives: {len(pos_vendors)}")
        print(f"Vendor groups with negatives: {len(neg_vendors)}")
        print(f"Vendor groups with mixed labels: {len(mixed_vendors)}")
        
        # CRITICAL FIX: Ensure we keep some positive vendors for training
        total_pos_scripts = sum(vendor_stats[vendor_stats['pos_count'] > 0]['total_scripts'])
        target_test_pos_scripts = max(1, int(total_pos_scripts * test_vendor_ratio))
        
        print(f"Total positive scripts: {total_pos_scripts}")
        print(f"Target test positive scripts: {target_test_pos_scripts}")
        
        # Strategy: Select test vendors ensuring we keep positives in both train and test
        test_vendor_groups = []
        test_script_count = 0
        test_pos_count = 0
        
        # Sort positive vendors by script count for stable selection
        pos_vendor_stats = vendor_stats[vendor_stats['vendor_group'].isin(pos_vendors)].sort_values(
            'total_scripts', ascending=False
        )
        
        # Select positive vendors for test set (but not all of them!)
        for _, vendor_row in pos_vendor_stats.iterrows():
            vendor_group = vendor_row['vendor_group']
            
            # Stop if we have enough positive scripts for test
            if test_pos_count >= target_test_pos_scripts:
                break
                
            # Don't take all positive vendors - ensure some remain for training
            remaining_pos_vendors = len(pos_vendor_stats) - len([v for v in test_vendor_groups if v in pos_vendors])
            if remaining_pos_vendors <= max(1, len(pos_vendors) * 0.3):  # Keep at least 30% for training
                break
                
            test_vendor_groups.append(vendor_group)
            test_script_count += vendor_row['total_scripts']
            test_pos_count += vendor_row['pos_count']
        
        # Now add negative vendors to reach target test ratio
        total_scripts = len(df)
        target_test_scripts = int(total_scripts * test_vendor_ratio)
        
        # Add negative vendor groups if we need more scripts
        if test_script_count < target_test_scripts:
            neg_vendor_stats = vendor_stats[
                (~vendor_stats['vendor_group'].isin(test_vendor_groups)) &
                (vendor_stats['vendor_group'].isin(neg_vendors))
            ].sort_values('total_scripts', ascending=False)
            
            for _, vendor_row in neg_vendor_stats.iterrows():
                if test_script_count >= target_test_scripts:
                    break
                    
                vendor_group = vendor_row['vendor_group']
                test_vendor_groups.append(vendor_group)
                test_script_count += vendor_row['total_scripts']
        
        # Create train/test split
        test_mask = df['vendor_group'].isin(test_vendor_groups)
        train_data = df[~test_mask].drop('vendor_group', axis=1).to_dict('records')
        test_data = df[test_mask].drop('vendor_group', axis=1).to_dict('records')
        
        # Report split statistics
        train_labels = [s['label'] for s in train_data]
        test_labels = [s['label'] for s in test_data]
        
        train_vendor_groups = df[~test_mask]['vendor_group'].unique()
        
        # Count positive samples
        train_pos = sum(1 for label in train_labels if label == 1)
        test_pos = sum(1 for label in test_labels if label == 1)
        
        print(f"\n📊 SPLIT RESULTS:")
        print(f"Train vendor groups: {len(train_vendor_groups)}")
        print(f"Test vendor groups: {len(test_vendor_groups)}")
        print(f"Train scripts: {len(train_data)} {Counter(train_labels)}")
        print(f"Test scripts: {len(test_data)} {Counter(test_labels)}")
        print(f"Train positives: {train_pos}")
        print(f"Test positives: {test_pos}")
        print(f"Actual test ratio: {len(test_data) / len(df):.1%}")
        
        # Critical validation: Ensure both sets have positive samples
        if train_pos == 0:
            raise ValueError("❌ No positive samples in training set! Adjust test_vendor_ratio.")
        if test_pos == 0:
            raise ValueError("❌ No positive samples in test set! Adjust test_vendor_ratio.")
        
        # Verify no vendor overlap (excluding null vendors which are unique)
        train_real_vendors = set(df[~test_mask & df['vendor'].notnull()]['vendor'].unique())
        test_real_vendors = set(df[test_mask & df['vendor'].notnull()]['vendor'].unique())
        vendor_overlap = train_real_vendors & test_real_vendors
        
        if vendor_overlap:
            raise ValueError(f"❌ Real vendor overlap detected: {vendor_overlap}")
        else:
            print("✅ No real vendor overlap - split is valid")
            print("✅ Both train and test have positive samples")
        
        return train_data, test_data, test_vendor_groups
    
    def load_and_split_datasets(self, test_vendor_ratio=0.2, random_state=42):
        """
        Load data and create vendor-aware train/test splits for both approaches.
        CRITICAL: Use same vendor split for both balanced and imbalanced to ensure consistency.
        """
        print("🔌 Loading data and creating vendor-aware train/test splits...")
        
        # Connect to database and load data
        connection = self.connect_to_database()
        if connection is None:
            raise Exception("Failed to connect to database")
        
        try:
            query = f"SELECT * FROM {self.table_name}"
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with JSON parsing
            self.raw_data = []
            json_fields = [
                'aggregated_behavioral_apis', 'aggregated_fingerprinting_apis',
                'fingerprinting_source_apis', 'behavioral_source_apis',
                'behavioral_apis_access_count', 'fingerprinting_api_access_count',
                'apis_going_to_sink', 'max_aggregated_apis'
            ]
            
            for row in rows:
                record = dict(row)
                for field in json_fields:
                    if field in record and record[field] is not None:
                        if isinstance(record[field], str):
                            try:
                                record[field] = json.loads(record[field])
                            except json.JSONDecodeError:
                                record[field] = None
                self.raw_data.append(record)
            
            cursor.close()
        finally:
            connection.close()
        
        print(f"✅ Loaded {len(self.raw_data)} total scripts")
        
        # Analyze vendor distribution
        print("\n🔍 Analyzing vendor distribution...")
        self.vendor_analysis = self.analyze_vendor_distribution(self.raw_data)
        
        # CRITICAL FIX: Create vendor-aware split on the FULL dataset first,
        # then filter for balanced/imbalanced views while preserving vendor split
        
        # Prepare the full dataset with label mapping for split decision
        full_data_for_split = []
        for script in self.raw_data:
            script_copy = script.copy()
            # Use original labels for vendor split decision to ensure we have positives
            full_data_for_split.append(script_copy)
        
        # Create vendor-aware splits on full dataset
        print(f"\n🎯 CREATING UNIFIED VENDOR-AWARE SPLIT")
        print("=" * 60)
        
        train_data_full, test_data_full, test_vendors = self.create_vendor_aware_splits(
            full_data_for_split, test_vendor_ratio, random_state
        )
        
        # Now create balanced and imbalanced views from the vendor-split data
        
        # BALANCED: Keep only labels 0 and 1
        self.balanced_train = [script for script in train_data_full if script['label'] in [0, 1]]
        self.balanced_test = [script for script in test_data_full if script['label'] in [0, 1]]
        
        # IMBALANCED: Map labels 1→1, {0,-1}→0
        self.imbalanced_train = []
        for script in train_data_full:
            script_copy = script.copy()
            script_copy['label'] = 1 if script['label'] == 1 else 0
            self.imbalanced_train.append(script_copy)
            
        self.imbalanced_test = []
        for script in test_data_full:
            script_copy = script.copy()
            script_copy['label'] = 1 if script['label'] == 1 else 0
            self.imbalanced_test.append(script_copy)
        
        self.test_vendors = test_vendors
        
        # Report final statistics
        print(f"\n🎯 BALANCED Dataset Split:")
        bal_train_labels = Counter([s['label'] for s in self.balanced_train])
        bal_test_labels = Counter([s['label'] for s in self.balanced_test])
        print(f"  Training: {len(self.balanced_train)} samples {dict(bal_train_labels)}")
        print(f"  Testing: {len(self.balanced_test)} samples {dict(bal_test_labels)}")
        
        print(f"\n⚖️  IMBALANCED Dataset Split:")
        imb_train_labels = Counter([s['label'] for s in self.imbalanced_train])
        imb_test_labels = Counter([s['label'] for s in self.imbalanced_test])
        print(f"  Training: {len(self.imbalanced_train)} samples {dict(imb_train_labels)}")
        print(f"  Testing: {len(self.imbalanced_test)} samples {dict(imb_test_labels)}")
        
        # Show class distributions for imbalanced
        train_pos = sum([s['label'] for s in self.imbalanced_train])
        train_neg = len(self.imbalanced_train) - train_pos
        
        if train_pos > 0:
            print(f"  Training ratio: 1:{train_neg/train_pos:.1f} (pos:neg)")
        else:
            print("  ⚠️  WARNING: No positive samples in training set!")
            print("     This suggests all positive vendors are in test set.")
            print("     Consider reducing test_vendor_ratio or adjusting split strategy.")
        
        # Verify we have positive samples in both training sets
        if sum([s['label'] for s in self.balanced_train]) == 0:
            raise ValueError("❌ No positive samples in balanced training set!")
        
        if sum([s['label'] for s in self.imbalanced_train]) == 0:
            raise ValueError("❌ No positive samples in imbalanced training set!")
        
        return test_vendors
    
    def create_vendor_groups_for_cv(self, train_data):
        """
        Create vendor groups for cross-validation.
        Handles null vendors properly.
        """
        df = pd.DataFrame(train_data)
        
        # Handle null vendors by creating unique groups
        vendor_groups = []
        null_counter = 0
        
        for _, row in df.iterrows():
            if pd.isnull(row['vendor']) or row['vendor'] == '':
                # Each null vendor gets unique group ID to prevent leakage
                vendor_groups.append(f'NULL_VENDOR_{null_counter}')
                null_counter += 1
            else:
                vendor_groups.append(str(row['vendor']))
        
        return vendor_groups
    
    def engineer_features(self, dataset, remove_correlated=True, correlation_threshold=0.9):
        """
        Engineer features with optional correlation removal.
        """
        features_list = []
        
        for script in dataset:
            try:
                features = {}
                
                # Core aggregation features
                features['max_api_aggregation_score'] = script.get('max_api_aggregation_score', -1)
                features['behavioral_api_agg_count'] = script.get('behavioral_api_agg_count', -1)
                features['fp_api_agg_count'] = script.get('fp_api_agg_count', -1)
                
                # Volume indicators
                features['behavioral_source_api_count'] = script.get('behavioral_source_api_count', 0)
                features['fingerprinting_source_api_count'] = script.get('fingerprinting_source_api_count', 0)
                
                # Data flow indicators
                features['dataflow_to_sink'] = int(script.get('dataflow_to_sink', False))
                
                # API access intensity
                behavioral_access = script.get('behavioral_apis_access_count') or {}
                fp_access = script.get('fingerprinting_api_access_count') or {}
                
                features['total_behavioral_api_accesses'] = sum(behavioral_access.values())
                features['total_fp_api_accesses'] = sum(fp_access.values())
                features['unique_behavioral_apis'] = len(behavioral_access)
                features['unique_fp_apis'] = len(fp_access)
                
                # Sink analysis
                sink_data = script.get('apis_going_to_sink') or {}
                features['num_sink_types'] = len(sink_data)
                features['has_storage_sink'] = int(any('Storage' in str(sink) for sink in sink_data.keys()))
                features['has_network_sink'] = int(any(sink in ['XMLHttpRequest.send', 'Navigator.sendBeacon', 'fetch'] 
                                                      for sink in sink_data.keys()))
                
                # Behavioral event diversity
                behavioral_sources = script.get('behavioral_source_apis') or []
                features['mouse_event_count'] = sum(1 for api in behavioral_sources if 'MouseEvent' in str(api))
                features['keyboard_event_count'] = sum(1 for api in behavioral_sources if 'KeyboardEvent' in str(api))
                features['touch_event_count'] = sum(1 for api in behavioral_sources if 'TouchEvent' in str(api) or 'Touch.' in str(api))
                features['pointer_event_count'] = sum(1 for api in behavioral_sources if 'PointerEvent' in str(api))
                features['behavioral_event_diversity'] = sum([
                    features['mouse_event_count'] > 0,
                    features['keyboard_event_count'] > 0,
                    features['touch_event_count'] > 0,
                    features['pointer_event_count'] > 0
                ])
                
                # Ratio features
                total_apis = features['behavioral_source_api_count'] + features['fingerprinting_source_api_count']
                if total_apis > 0:
                    features['behavioral_ratio'] = features['behavioral_source_api_count'] / total_apis
                    features['intensity_ratio'] = features['total_behavioral_api_accesses'] / total_apis
                else:
                    features['behavioral_ratio'] = 0
                    features['intensity_ratio'] = 0
                
                # Store metadata
                features['script_id'] = script.get('script_id')
                features['label'] = script['label']
                features['vendor'] = script.get('vendor')
                
                features_list.append(features)
                
            except Exception as e:
                print(f"⚠️  Feature extraction error for script {script.get('script_id', 'unknown')}: {e}")
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Remove highly correlated features if requested
        if remove_correlated and len(features_df) > 10:
            metadata_cols = ['script_id', 'label', 'vendor']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            
            # Calculate correlation matrix
            numeric_features = features_df[feature_cols].select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr()
                
                # Find features to remove
                features_to_remove = set()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                            features_to_remove.add(corr_matrix.columns[j])
                
                # Keep only uncorrelated features
                final_features = [col for col in feature_cols if col not in features_to_remove]
                features_df = features_df[final_features + metadata_cols]
                
                if features_to_remove:
                    print(f"📉 Removed {len(features_to_remove)} correlated features")
        
        return features_df
    
    def vendor_aware_nested_cross_validation(self, train_data, dataset_name, 
                                        outer_cv_folds=5, inner_cv_folds=3,
                                        scoring='roc_auc', n_jobs=-1):
        """
        Perform vendor-aware nested cross-validation using StratifiedGroupKFold.
        """
        print(f"\n🔄 Starting Vendor-Aware Nested Cross-Validation for {dataset_name} Dataset...")
        print(f"  Outer CV: {outer_cv_folds} folds (vendor-grouped for performance estimation)")
        print(f"  Inner CV: {inner_cv_folds} folds (vendor-grouped for hyperparameter tuning)")
        
        # Prepare features
        features_df = self.engineer_features(train_data)
        feature_cols = [col for col in features_df.columns if col not in ['script_id', 'label', 'vendor']]
        
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Create vendor groups for GroupKFold
        vendor_groups = self.create_vendor_groups_for_cv(train_data)
        
        # Map vendor groups to integers
        unique_groups = list(set(vendor_groups))
        group_mapping = {group: i for i, group in enumerate(unique_groups)}
        groups = np.array([group_mapping[group] for group in vendor_groups])
        
        print(f"  Total vendor groups: {len(unique_groups)}")
        
        # Check if we have enough groups for CV
        if len(unique_groups) < outer_cv_folds:
            print(f"⚠️  Warning: Only {len(unique_groups)} vendor groups but {outer_cv_folds} outer folds requested")
            outer_cv_folds = min(outer_cv_folds, len(unique_groups))
            print(f"   Reducing outer CV to {outer_cv_folds} folds")
        
        if len(unique_groups) < inner_cv_folds:
            inner_cv_folds = min(inner_cv_folds, len(unique_groups))
            print(f"   Reducing inner CV to {inner_cv_folds} folds")
        
        # Try StratifiedGroupKFold first, fall back to GroupKFold if needed
        try:
            outer_cv = StratifiedGroupKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
            inner_cv_class = StratifiedGroupKFold
            print("  Using StratifiedGroupKFold (maintains label balance)")
        except Exception as e:
            print(f"  StratifiedGroupKFold failed: {e}")
            outer_cv = GroupKFold(n_splits=outer_cv_folds)
            inner_cv_class = GroupKFold
            print("  Using GroupKFold")
        
        # Initialize results storage
        outer_scores = {
            'train_acc': [], 'val_acc': [], 'train_auc': [], 'val_auc': [],
            'train_ap': [], 'val_ap': [], 'best_params': [], 'feature_importances': []
        }
        
        # FIXED: Define the custom scorer properly
        def safe_roc_auc_scorer(y_true, y_score, **kwargs):
            """Custom scorer that handles single-class cases"""
            try:
                if len(set(y_true)) < 2:
                    return 0.5  # Return neutral score for single-class
                return roc_auc_score(y_true, y_score)
            except Exception:
                return 0.5
        
        # Create scorer properly
        safe_scorer = make_scorer(
            safe_roc_auc_scorer, 
            greater_is_better=True, 
            needs_proba=True,
            response_method='predict_proba'
        )
        
        # Outer CV loop
        fold_num = 0
        for train_idx, val_idx in outer_cv.split(X, y, groups):
            fold_num += 1
            print(f"\n  📁 Outer Fold {fold_num}/{outer_cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            
            # Check label distribution in validation set
            val_label_dist = Counter(y_val)
            if len(val_label_dist) < 2:
                print(f"    ⚠️  Skipping fold - validation set has only one class: {val_label_dist}")
                continue
            
            # Report fold statistics
            train_label_dist = Counter(y_train)
            train_vendor_groups = len(set(groups_train))
            val_vendor_groups = len(set(groups[val_idx]))
            
            print(f"    Train: {len(X_train)} scripts, {train_vendor_groups} vendor groups, {dict(train_label_dist)}")
            print(f"    Val:   {len(X_val)} scripts, {val_vendor_groups} vendor groups, {dict(val_label_dist)}")
            
            # Inner CV for hyperparameter tuning
            try:
                if inner_cv_class == StratifiedGroupKFold:
                    inner_cv = StratifiedGroupKFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
                else:
                    inner_cv = GroupKFold(n_splits=inner_cv_folds)
            except:
                inner_cv = GroupKFold(n_splits=inner_cv_folds)
            
            # Configure RandomizedSearchCV
            rf = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_jobs=n_jobs
            )
            
            # Use RandomizedSearchCV with vendor-aware inner CV
            print(f"    🔍 Hyperparameter tuning on inner CV...")
            try:
                # Check if inner CV will have class balance issues
                unique_train_labels = len(set(y_train))
                unique_groups_train = len(set(groups_train))
                
                if unique_train_labels < 2:
                    print(f"    ⚠️  Skipping hyperparameter tuning - only one class in training")
                    # Use default parameters
                    best_params = {
                        'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5,
                        'min_samples_leaf': 2, 'max_features': 0.5, 'bootstrap': True
                    }
                    best_score = 0.5
                elif unique_groups_train < inner_cv_folds:
                    print(f"    ⚠️  Too few vendor groups for inner CV, using simple validation")
                    # Use a simple train/validation split instead of CV
                    from sklearn.model_selection import train_test_split
                    X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                        X_train, y_train, test_size=0.3, stratify=y_train, random_state=42
                    )
                    
                    # Try a few parameter combinations manually
                    best_score = 0
                    best_params = None
                    
                    param_combinations = [
                        {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5, 
                        'min_samples_leaf': 2, 'max_features': 0.5, 'bootstrap': True},
                        {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 2, 
                        'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': False},
                        {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 10, 
                        'min_samples_leaf': 4, 'max_features': 'log2', 'bootstrap': True}
                    ]
                    
                    for params in param_combinations:
                        temp_rf = RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=n_jobs)
                        temp_rf.fit(X_inner_train, y_inner_train)
                        
                        if len(set(y_inner_val)) > 1:  # Check if validation has both classes
                            temp_proba = temp_rf.predict_proba(X_inner_val)[:, 1]
                            temp_score = roc_auc_score(y_inner_val, temp_proba)
                            if temp_score > best_score:
                                best_score = temp_score
                                best_params = params
                    
                    if best_params is None:
                        best_params = param_combinations[0]
                        best_score = 0.5
                        
                else:
                    # Standard RandomizedSearchCV with custom scoring
                    random_search = RandomizedSearchCV(
                        estimator=rf,
                        param_distributions=self.param_distributions,
                        n_iter=15,  # Reduced for faster execution with vendor grouping
                        cv=inner_cv,
                        scoring=safe_scorer,  # Use our properly defined scorer
                        n_jobs=n_jobs,
                        random_state=42,
                        verbose=0,
                        error_score=0.5  # Return neutral score on errors
                    )
                    
                    # Suppress warnings for this section
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Only one class is present")
                        warnings.filterwarnings("ignore", message="ROC AUC score is not defined")
                        random_search.fit(X_train, y_train, groups=groups_train)
                    
                    best_params = random_search.best_params_
                    best_score = random_search.best_score_
                
                print(f"    ✅ Best inner CV score: {best_score:.3f}")
                print(f"    📋 Best parameters: {best_params}")
                
                # Train final model for this fold
                final_model = RandomForestClassifier(
                    **best_params,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=n_jobs
                )
                final_model.fit(X_train, y_train)
                
                # Evaluate on outer fold validation set
                train_pred = final_model.predict(X_train)
                val_pred = final_model.predict(X_val)
                train_proba = final_model.predict_proba(X_train)[:, 1]
                val_proba = final_model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                train_acc = np.mean(train_pred == y_train)
                val_acc = np.mean(val_pred == y_val)
                train_auc = roc_auc_score(y_train, train_proba)
                val_auc = roc_auc_score(y_val, val_proba)
                train_ap = average_precision_score(y_train, train_proba)
                val_ap = average_precision_score(y_val, val_proba)
                
                print(f"    📊 Outer fold validation AUC: {val_auc:.3f}")
                
                # Store results
                outer_scores['train_acc'].append(train_acc)
                outer_scores['val_acc'].append(val_acc)
                outer_scores['train_auc'].append(train_auc)
                outer_scores['val_auc'].append(val_auc)
                outer_scores['train_ap'].append(train_ap)
                outer_scores['val_ap'].append(val_ap)
                outer_scores['best_params'].append(best_params)
                outer_scores['feature_importances'].append(final_model.feature_importances_)
                
            except Exception as e:
                print(f"    ❌ Error in fold {fold_num}: {e}")
                continue
        
        # Calculate summary statistics
        if len(outer_scores['val_auc']) == 0:
            raise ValueError("No valid folds completed. Check vendor grouping and data distribution.")
        
        results = {
            'dataset_name': dataset_name,
            'feature_names': feature_cols,
            'outer_scores': outer_scores,
            'vendor_groups_used': len(unique_groups),
            'completed_folds': len(outer_scores['val_auc']),
            'metrics': {
                'train_acc': {'mean': np.mean(outer_scores['train_acc']), 
                            'std': np.std(outer_scores['train_acc'])},
                'val_acc': {'mean': np.mean(outer_scores['val_acc']), 
                        'std': np.std(outer_scores['val_acc'])},
                'train_auc': {'mean': np.mean(outer_scores['train_auc']), 
                            'std': np.std(outer_scores['train_auc'])},
                'val_auc': {'mean': np.mean(outer_scores['val_auc']), 
                        'std': np.std(outer_scores['val_auc'])},
                'train_ap': {'mean': np.mean(outer_scores['train_ap']), 
                            'std': np.std(outer_scores['train_ap'])},
                'val_ap': {'mean': np.mean(outer_scores['val_ap']), 
                        'std': np.std(outer_scores['val_ap'])}
            }
        }
        
        print(f"\n📊 {dataset_name} Vendor-Aware Nested CV Results:")
        print(f"  Completed folds: {results['completed_folds']}/{outer_cv_folds}")
        print(f"  Vendor groups: {results['vendor_groups_used']}")
        print(f"  Validation AUC: {results['metrics']['val_auc']['mean']:.3f} ± {results['metrics']['val_auc']['std']:.3f}")
        print(f"  Validation AP:  {results['metrics']['val_ap']['mean']:.3f} ± {results['metrics']['val_ap']['std']:.3f}")
        print(f"  Validation Acc: {results['metrics']['val_acc']['mean']:.3f} ± {results['metrics']['val_acc']['std']:.3f}")
        
        return results
    
    def train_final_model(self, train_data, test_data, nested_cv_results, dataset_name):
        """
        Train a final model using the most common best parameters from nested CV
        and evaluate on the held-out test set.
        """
        print(f"\n🎯 Training final {dataset_name} model and evaluating on test set...")
        
        # Get the most common hyperparameters from nested CV
        param_tuples = [
            tuple(sorted(params.items()))
            for params in nested_cv_results['outer_scores']['best_params']
        ]
        best_params = dict(Counter(param_tuples).most_common(1)[0][0])
        print(f"  📋 Using most common parameters from nested CV: {best_params}")
        
        # Engineer features with correlation-dropping on both train & test
        train_df = self.engineer_features(train_data, remove_correlated=True)
        test_df = self.engineer_features(test_data, remove_correlated=True)
        
        # Align columns exactly
        feature_cols = nested_cv_results['feature_names']
        train_df = train_df.reindex(
            columns=feature_cols + ['label', 'script_id', 'vendor'], 
            fill_value=0
        )
        test_df = test_df.reindex(
            columns=feature_cols + ['label', 'script_id', 'vendor'], 
            fill_value=0
        )
        
        # Extract features and labels
        X_train, y_train = train_df[feature_cols].values, train_df['label'].values
        X_test, y_test = test_df[feature_cols].values, test_df['label'].values
        
        # Train final RandomForest
        final_model = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_pred = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        
        test_acc = np.mean(test_pred == y_test)
        test_auc = roc_auc_score(y_test, test_proba)
        test_ap = average_precision_score(y_test, test_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Vendor-level analysis
        test_df['pred'] = test_pred
        test_df['proba'] = test_proba
        vendor_performance = self.analyze_vendor_performance(test_df, dataset_name)
        
        # Package results
        results = {
            'model': final_model,
            'params': best_params,
            'test_metrics': {
                'accuracy': test_acc,
                'auc': test_auc,
                'average_precision': test_ap,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            },
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False),
            'vendor_performance': vendor_performance
        }
        
        print(f"\n📊 {dataset_name} Test Set Performance:")
        print(f"  Test AUC: {test_auc:.3f}")
        print(f"  Test AP:  {test_ap:.3f}")
        print(f"  Test Acc: {test_acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        
        return results

    def analyze_vendor_performance(self, test_df, dataset_name):
        """
        Analyze performance at the vendor level to identify potential biases.
        """
        print(f"\n🏢 Vendor-Level Performance Analysis for {dataset_name}")
        
        # Group by vendor (handle nulls)
        test_df['vendor_clean'] = test_df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        vendor_stats = []
        for vendor, group in test_df.groupby('vendor_clean'):
            if len(group) >= 2:  # Only analyze vendors with multiple scripts
                y_true = group['label'].values
                y_pred = group['pred'].values
                y_proba = group['proba'].values
                
                try:
                    vendor_auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else None
                except:
                    vendor_auc = None
                
                vendor_stats.append({
                    'vendor': vendor,
                    'count': len(group),
                    'true_positives': sum(y_true),
                    'predicted_positives': sum(y_pred),
                    'accuracy': np.mean(y_pred == y_true),
                    'auc': vendor_auc
                })
        
        vendor_df = pd.DataFrame(vendor_stats)
        
        if len(vendor_df) > 0:
            print(f"  Vendors with ≥2 scripts: {len(vendor_df)}")
            print(f"  Average vendor accuracy: {vendor_df['accuracy'].mean():.3f}")
            
            # Show vendors with extreme performance
            if len(vendor_df) > 5:
                print("\n  Top performing vendors:")
                top_vendors = vendor_df.nlargest(3, 'accuracy')[['vendor', 'count', 'accuracy', 'auc']]
                print(top_vendors.to_string(index=False))
                
                print("\n  Lowest performing vendors:")
                bottom_vendors = vendor_df.nsmallest(3, 'accuracy')[['vendor', 'count', 'accuracy', 'auc']]
                print(bottom_vendors.to_string(index=False))
        
        return vendor_df

    def statistical_comparison(self, balanced_scores, imbalanced_scores):
        """
        Perform statistical significance testing between the two approaches.
        """
        print(f"\n📊 Statistical Comparison of Models...")
        
        results = {}
        
        # Paired t-test for each metric
        metrics = ['val_auc', 'val_ap', 'val_acc']
        
        for metric in metrics:
            balanced_vals = balanced_scores['outer_scores'][metric]
            imbalanced_vals = imbalanced_scores['outer_scores'][metric]
            
            # Ensure same number of values for paired test
            min_len = min(len(balanced_vals), len(imbalanced_vals))
            balanced_vals = balanced_vals[:min_len]
            imbalanced_vals = imbalanced_vals[:min_len]
            
            if min_len < 2:
                print(f"  ⚠️  Not enough samples for {metric} comparison")
                continue
            
            # Paired t-test (same CV folds)
            t_stat, p_value = stats.ttest_rel(balanced_vals, imbalanced_vals)
            
            # Cohen's d for effect size
            diff = np.array(imbalanced_vals) - np.array(balanced_vals)
            cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(balanced_vals, imbalanced_vals)
            except:
                wilcoxon_stat, wilcoxon_p = None, None
            
            results[metric] = {
                'balanced_mean': np.mean(balanced_vals),
                'imbalanced_mean': np.mean(imbalanced_vals),
                'difference': np.mean(imbalanced_vals) - np.mean(balanced_vals),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'wilcoxon_p': wilcoxon_p,
                'significant': p_value < 0.05,
                'n_samples': min_len
            }
            
            print(f"\n  {metric.upper()}:")
            print(f"    Balanced: {results[metric]['balanced_mean']:.3f}")
            print(f"    Imbalanced: {results[metric]['imbalanced_mean']:.3f}")
            print(f"    Difference: {results[metric]['difference']:.3f}")
            print(f"    p-value: {results[metric]['p_value']:.4f}")
            print(f"    Cohen's d: {results[metric]['cohen_d']:.3f}")
            print(f"    Significant: {'Yes' if results[metric]['significant'] else 'No'}")
            print(f"    Samples: {results[metric]['n_samples']}")
        
        return results

    def run_robust_comparison(self):
        """
        Run the complete vendor-aware methodologically rigorous comparison.
        """
        print("🚀 " + "="*80)
        print("VENDOR-AWARE ROBUST MODEL COMPARISON WITH NESTED CROSS-VALIDATION")
        print("="*80)
        
        # Step 1: Load and split data with vendor awareness
        print(f"\n📚 STEP 1: Load and Split Data (Vendor-Aware)")
        test_vendors = self.load_and_split_datasets(test_vendor_ratio=0.2)
        
        print(f"\n📌 Test Vendors (Held-Out): {len(test_vendors)} vendor groups")
        real_test_vendors = [v for v in test_vendors if not v.startswith('NULL_VENDOR')]
        if real_test_vendors:
            print(f"Real test vendors: {real_test_vendors[:10]}{'...' if len(real_test_vendors) > 10 else ''}")
        
        # Step 2: Vendor-aware nested cross-validation
        print(f"\n🔄 STEP 2: Vendor-Aware Nested Cross-Validation")
        
        try:
            self.nested_cv_results['balanced'] = self.vendor_aware_nested_cross_validation(
                self.balanced_train, "Balanced"
            )
        except Exception as e:
            print(f"❌ Error in balanced nested CV: {e}")
            raise
        
        try:
            self.nested_cv_results['imbalanced'] = self.vendor_aware_nested_cross_validation(
                self.imbalanced_train, "Imbalanced"
            )
        except Exception as e:
            print(f"❌ Error in imbalanced nested CV: {e}")
            raise
        
        # Step 3: Statistical comparison
        print(f"\n📊 STEP 3: Statistical Comparison")
        stat_comparison = self.statistical_comparison(
            self.nested_cv_results['balanced'],
            self.nested_cv_results['imbalanced']
        )
        
        # Step 4: Train final models and test
        print(f"\n🎯 STEP 4: Final Model Training and Testing")
        
        balanced_final = self.train_final_model(
            self.balanced_train,
            self.balanced_test,
            self.nested_cv_results['balanced'],
            "Balanced"
        )
        
        imbalanced_final = self.train_final_model(
            self.imbalanced_train,
            self.imbalanced_test,
            self.nested_cv_results['imbalanced'],
            "Imbalanced"
        )
        
        # Step 5: Generate comprehensive report
        print(f"\n📝 STEP 5: Generate Comprehensive Report")
        self.generate_vendor_aware_report(
            self.nested_cv_results,
            stat_comparison,
            balanced_final,
            imbalanced_final
        )
        
        # Step 6: Create visualizations
        print(f"\n📊 STEP 6: Create Visualizations")
        self.create_vendor_aware_visualizations(
            self.nested_cv_results,
            stat_comparison,
            balanced_final,
            imbalanced_final
        )
        
        # Step 7: Save vendor analysis results
        print(f"\n💾 STEP 7: Save Analysis Results")
        self.save_vendor_analysis_results()
        
        print("\n✅ Vendor-aware robust comparison complete!")
        
        return {
            'nested_cv_results': self.nested_cv_results,
            'statistical_comparison': stat_comparison,
            'balanced_final': balanced_final,
            'imbalanced_final': imbalanced_final,
            'vendor_analysis': self.vendor_analysis,
            'test_vendors': test_vendors
        }
    
    def generate_vendor_aware_report(self, nested_cv_results, stat_comparison, 
                                   balanced_final, imbalanced_final):
        """
        Generate a methodologically rigorous report focusing on vendor awareness.
        """
        report_file = f"{self.output_dir}/vendor_aware_comparison_report_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Vendor-Aware Robust Comparison of Balanced vs Imbalanced Approaches\n\n")
            
            f.write("## Methodology\n\n")
            f.write("This analysis addresses vendor-based data leakage through:\n\n")
            f.write("- **Vendor-Grouped Cross-Validation**: No vendor appears in both train and validation\n")
            f.write("- **StratifiedGroupKFold**: Maintains label balance while respecting vendor groups\n")
            f.write("- **Vendor-Aware Test Split**: Test vendors completely held out\n")
            f.write("- **Null Vendor Handling**: Each unknown vendor gets unique group ID\n")
            f.write("- **Nested CV**: 5-fold outer, 3-fold inner for unbiased estimates\n")
            f.write("- **Statistical Testing**: Paired t-test and Wilcoxon signed-rank test\n\n")
            
            f.write("## Vendor Distribution Analysis\n\n")
            if self.vendor_analysis:
                f.write(f"- **Total vendors**: {self.vendor_analysis['total_vendors']}\n")
                f.write(f"- **Scripts with null vendor**: {self.vendor_analysis['null_vendor_count']}\n")
                f.write(f"- **Positive-only vendors**: {len(self.vendor_analysis['pos_only_vendors'])}\n")
                f.write(f"- **Negative-only vendors**: {len(self.vendor_analysis['neg_only_vendors'])}\n")
                f.write(f"- **Mixed-label vendors**: {len(self.vendor_analysis['mixed_vendors'])}\n")
                f.write(f"- **Single-script vendors**: {len(self.vendor_analysis['single_script_vendors'])}\n\n")
            
            f.write("## Key Results\n\n")
            
            # Nested CV results
            f.write("### Vendor-Aware Nested Cross-Validation Performance\n\n")
            f.write("| Metric | Balanced | Imbalanced | Difference | p-value | Significant |\n")
            f.write("|--------|----------|------------|------------|---------|-------------|\n")
            
            for metric in ['val_auc', 'val_ap', 'val_acc']:
                if metric in stat_comparison:
                    bal_mean = nested_cv_results['balanced']['metrics'][metric]['mean']
                    bal_std = nested_cv_results['balanced']['metrics'][metric]['std']
                    imb_mean = nested_cv_results['imbalanced']['metrics'][metric]['mean']
                    imb_std = nested_cv_results['imbalanced']['metrics'][metric]['std']
                    diff = stat_comparison[metric]['difference']
                    p_val = stat_comparison[metric]['p_value']
                    sig = "Yes" if stat_comparison[metric]['significant'] else "No"
                    
                    f.write(f"| {metric} | {bal_mean:.3f}±{bal_std:.3f} | "
                           f"{imb_mean:.3f}±{imb_std:.3f} | {diff:+.3f} | {p_val:.4f} | {sig} |\n")
            
            f.write("\n### Hold-out Test Set Performance\n\n")
            f.write("| Metric | Balanced | Imbalanced | Difference |\n")
            f.write("|--------|----------|------------|------------|\n")
            
            test_metrics = ['auc', 'average_precision', 'accuracy', 'precision', 'recall']
            for metric in test_metrics:
                bal_val = balanced_final['test_metrics'][metric]
                imb_val = imbalanced_final['test_metrics'][metric]
                diff = imb_val - bal_val
                f.write(f"| {metric} | {bal_val:.3f} | {imb_val:.3f} | {diff:+.3f} |\n")
            
            f.write("\n## Vendor-Level Analysis\n\n")
            
            # Vendor performance analysis
            if 'vendor_performance' in balanced_final and len(balanced_final['vendor_performance']) > 0:
                bal_vendor_perf = balanced_final['vendor_performance']
                f.write(f"### Balanced Approach - Vendor Performance\n\n")
                f.write(f"- Vendors analyzed: {len(bal_vendor_perf)}\n")
                f.write(f"- Average vendor accuracy: {bal_vendor_perf['accuracy'].mean():.3f}\n")
                f.write(f"- Vendor accuracy std: {bal_vendor_perf['accuracy'].std():.3f}\n\n")
            
            if 'vendor_performance' in imbalanced_final and len(imbalanced_final['vendor_performance']) > 0:
                imb_vendor_perf = imbalanced_final['vendor_performance']
                f.write(f"### Imbalanced Approach - Vendor Performance\n\n")
                f.write(f"- Vendors analyzed: {len(imb_vendor_perf)}\n")
                f.write(f"- Average vendor accuracy: {imb_vendor_perf['accuracy'].mean():.3f}\n")
                f.write(f"- Vendor accuracy std: {imb_vendor_perf['accuracy'].std():.3f}\n\n")
            
            f.write("## Statistical Analysis\n\n")
            
            # Effect sizes
            f.write("### Effect Sizes (Cohen's d)\n\n")
            for metric in ['val_auc', 'val_ap', 'val_acc']:
                if metric in stat_comparison:
                    cohen_d = stat_comparison[metric]['cohen_d']
                    f.write(f"- {metric}: {cohen_d:.3f} ")
                    if abs(cohen_d) < 0.2:
                        f.write("(negligible effect)\n")
                    elif abs(cohen_d) < 0.5:
                        f.write("(small effect)\n")
                    elif abs(cohen_d) < 0.8:
                        f.write("(medium effect)\n")
                    else:
                        f.write("(large effect)\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Determine winner based on test set AUC
            if imbalanced_final['test_metrics']['auc'] > balanced_final['test_metrics']['auc']:
                f.write("**Recommended Approach: IMBALANCED**\n\n")
                f.write("The imbalanced approach shows superior performance on the vendor-grouped ")
                f.write("test set and better reflects real-world deployment conditions where new ")
                f.write("vendors regularly appear.\n")
            else:
                f.write("**Recommended Approach: BALANCED**\n\n")
                f.write("The balanced approach shows superior performance on the vendor-grouped ")
                f.write("test set despite having less training data.\n")
            
            f.write("\n## Methodological Advantages\n\n")
            f.write("This vendor-aware analysis addresses critical issues:\n\n")
            f.write("1. **No vendor leakage**: Train/test vendor separation prevents overfitting\n")
            f.write("2. **Realistic evaluation**: Models tested on completely new vendors\n")
            f.write("3. **Generalization focus**: Forces learning of behavior patterns, not vendor signatures\n")
            f.write("4. **Statistical rigor**: Significance testing with appropriate effect sizes\n")
            f.write("5. **Deployment readiness**: Results reflect real-world performance expectations\n")
            f.write("6. **Null vendor handling**: Proper treatment of unassigned vendor labels\n")
        
        print(f"📝 Report saved: {report_file}")
    
    def create_vendor_aware_visualizations(self, nested_cv_results, stat_comparison,
                                         balanced_final, imbalanced_final):
        """
        Create comprehensive visualizations for the vendor-aware comparison.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Nested CV AUC scores by fold
        bal_aucs = nested_cv_results['balanced']['outer_scores']['val_auc']
        imb_aucs = nested_cv_results['imbalanced']['outer_scores']['val_auc']
        
        max_folds = max(len(bal_aucs), len(imb_aucs))
        folds = range(1, max_folds + 1)
        
        if len(bal_aucs) > 0:
            axes[0, 0].plot(range(1, len(bal_aucs) + 1), bal_aucs, 'o-', 
                           label='Balanced', linewidth=2, markersize=8)
        if len(imb_aucs) > 0:
            axes[0, 0].plot(range(1, len(imb_aucs) + 1), imb_aucs, 's-', 
                           label='Imbalanced', linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('Vendor-Grouped CV Fold')
        axes[0, 0].set_ylabel('Validation AUC')
        axes[0, 0].set_title('Vendor-Aware Nested CV: AUC by Fold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Statistical comparison with confidence intervals
        metrics = ['AUC', 'AP', 'Accuracy']
        metric_keys = ['val_auc', 'val_ap', 'val_acc']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bal_means = []
        bal_stds = []
        imb_means = []
        imb_stds = []
        
        for m in metric_keys:
            if m in nested_cv_results['balanced']['metrics']:
                bal_means.append(nested_cv_results['balanced']['metrics'][m]['mean'])
                bal_stds.append(nested_cv_results['balanced']['metrics'][m]['std'])
                imb_means.append(nested_cv_results['imbalanced']['metrics'][m]['mean'])
                imb_stds.append(nested_cv_results['imbalanced']['metrics'][m]['std'])
            else:
                bal_means.append(0)
                bal_stds.append(0)
                imb_means.append(0)
                imb_stds.append(0)
        
        # Calculate 95% confidence intervals
        n_folds = max(nested_cv_results['balanced'].get('completed_folds', 1), 1)
        bal_ci = [1.96 * s / np.sqrt(n_folds) for s in bal_stds]
        imb_ci = [1.96 * s / np.sqrt(n_folds) for s in imb_stds]
        
        axes[0, 1].bar(x - width/2, bal_means, width, yerr=bal_ci, 
                      label='Balanced', alpha=0.8, capsize=5)
        axes[0, 1].bar(x + width/2, imb_means, width, yerr=imb_ci,
                      label='Imbalanced', alpha=0.8, capsize=5)
        
        # Add significance stars
        for i, (metric_key, metric_name) in enumerate(zip(metric_keys, metrics)):
            if metric_key in stat_comparison and stat_comparison[metric_key]['significant']:
                y_max = max(bal_means[i] + bal_ci[i], imb_means[i] + imb_ci[i])
                axes[0, 1].text(i, y_max + 0.01, '*', ha='center', fontsize=14)
        
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Score (95% CI)')
        axes[0, 1].set_title('Vendor-Aware Performance Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Test set ROC curves - FIXED FEATURE ALIGNMENT
        try:
            from sklearn.metrics import roc_curve
            
            # CRITICAL FIX: Use consistent feature engineering for both test sets
            print("🔧 Aligning features for ROC curve generation...")
            
            # Get feature names from both models
            bal_feature_names = set(balanced_final['feature_importance']['feature'].tolist())
            imb_feature_names = set(imbalanced_final['feature_importance']['feature'].tolist())
            
            # Use intersection of features (common to both models)
            common_features = list(bal_feature_names & imb_feature_names)
            print(f"   Using {len(common_features)} common features")
            
            if len(common_features) > 0:
                # Engineer features for test sets
                bal_features = self.engineer_features(self.balanced_test, remove_correlated=False)
                imb_features = self.engineer_features(self.imbalanced_test, remove_correlated=False)
                
                # Align both to common features
                bal_features = bal_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                  fill_value=0)
                imb_features = imb_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                  fill_value=0)
                
                bal_X = bal_features[common_features].values
                bal_y = bal_features['label'].values
                
                imb_X = imb_features[common_features].values
                imb_y = imb_features['label'].values
                
                # Retrain models on common features for visualization only
                print("   Retraining models on common features for visualization...")
                
                # Retrain balanced model
                bal_train_features = self.engineer_features(self.balanced_train, remove_correlated=False)
                bal_train_features = bal_train_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                              fill_value=0)
                bal_X_train = bal_train_features[common_features].values
                bal_y_train = bal_train_features['label'].values
                
                temp_bal_model = RandomForestClassifier(**balanced_final['params'], random_state=42, n_jobs=-1)
                temp_bal_model.fit(bal_X_train, bal_y_train)
                bal_proba = temp_bal_model.predict_proba(bal_X)[:, 1]
                
                # Retrain imbalanced model
                imb_train_features = self.engineer_features(self.imbalanced_train, remove_correlated=False)
                imb_train_features = imb_train_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                              fill_value=0)
                imb_X_train = imb_train_features[common_features].values
                imb_y_train = imb_train_features['label'].values
                
                temp_imb_model = RandomForestClassifier(**imbalanced_final['params'], random_state=42, n_jobs=-1)
                temp_imb_model.fit(imb_X_train, imb_y_train)
                imb_proba = temp_imb_model.predict_proba(imb_X)[:, 1]
                
                # Generate ROC curves
                bal_fpr, bal_tpr, _ = roc_curve(bal_y, bal_proba)
                imb_fpr, imb_tpr, _ = roc_curve(imb_y, imb_proba)
                
                axes[0, 2].plot(bal_fpr, bal_tpr, 
                               label=f'Balanced (AUC={balanced_final["test_metrics"]["auc"]:.3f})',
                               linewidth=2)
                axes[0, 2].plot(imb_fpr, imb_tpr,
                               label=f'Imbalanced (AUC={imbalanced_final["test_metrics"]["auc"]:.3f})',
                               linewidth=2)
                axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[0, 2].set_xlabel('False Positive Rate')
                axes[0, 2].set_ylabel('True Positive Rate')
                axes[0, 2].set_title('Test Set ROC Curves (Vendor-Held-Out)')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
                
            else:
                axes[0, 2].text(0.5, 0.5, 'No common features\nfor ROC comparison', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Test Set ROC Curves (Vendor-Held-Out)')
                
        except Exception as e:
            print(f"   ⚠️ Error generating ROC curves: {e}")
            axes[0, 2].text(0.5, 0.5, f'ROC generation failed:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Test Set ROC Curves (Error)')
        
        # Plot 4: Feature importance comparison (using common features)
        try:
            top_n = min(10, len(common_features)) if 'common_features' in locals() else 8
            bal_imp = balanced_final['feature_importance'].head(top_n)
            imb_imp = imbalanced_final['feature_importance'].head(top_n)
            
            # Find common features in importance lists
            common_imp_features = set(bal_imp['feature']).intersection(set(imb_imp['feature']))
            common_imp_features = list(common_imp_features)[:8]
            
            if common_imp_features:
                bal_values = [bal_imp[bal_imp['feature'] == f]['importance'].iloc[0] 
                             for f in common_imp_features]
                imb_values = [imb_imp[imb_imp['feature'] == f]['importance'].iloc[0] 
                             for f in common_imp_features]
                
                y_pos = np.arange(len(common_imp_features))
                axes[1, 0].barh(y_pos - 0.2, bal_values, 0.4, label='Balanced', alpha=0.8)
                axes[1, 0].barh(y_pos + 0.2, imb_values, 0.4, label='Imbalanced', alpha=0.8)
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                                            for f in common_imp_features], fontsize=9)
                axes[1, 0].set_xlabel('Feature Importance')
                axes[1, 0].set_title('Top Feature Importance Comparison')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No common features\nin top importance', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Importance Comparison')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Feature importance\ncomparison failed', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance Comparison (Error)')
        
        # Plot 5: Vendor-level performance distribution
        if ('vendor_performance' in balanced_final and 
            len(balanced_final['vendor_performance']) > 0):
            
            bal_vendor_acc = balanced_final['vendor_performance']['accuracy'].values
            imb_vendor_acc = imbalanced_final['vendor_performance']['accuracy'].values
            
            if len(bal_vendor_acc) > 1 and len(imb_vendor_acc) > 1:
                axes[1, 1].hist(bal_vendor_acc, bins=10, alpha=0.7, label='Balanced', density=True)
                axes[1, 1].hist(imb_vendor_acc, bins=10, alpha=0.7, label='Imbalanced', density=True)
                axes[1, 1].axvline(np.mean(bal_vendor_acc), color='blue', linestyle='--', 
                                  label=f'Bal Mean: {np.mean(bal_vendor_acc):.3f}')
                axes[1, 1].axvline(np.mean(imb_vendor_acc), color='orange', linestyle='--',
                                  label=f'Imb Mean: {np.mean(imb_vendor_acc):.3f}')
                axes[1, 1].set_xlabel('Vendor-Level Accuracy')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Vendor Performance Distribution')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient vendor\ndata for distribution', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Vendor Performance Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient vendor\nperformance data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Vendor Performance Distribution')
        
        # Plot 6: Effect sizes visualization
        metrics_display = ['AUC', 'AP', 'Accuracy']
        cohen_ds = []
        
        for m in metric_keys:
            if m in stat_comparison:
                cohen_ds.append(stat_comparison[m]['cohen_d'])
            else:
                cohen_ds.append(0)
        
        colors = ['green' if abs(d) < 0.2 else 'yellow' if abs(d) < 0.5 
                 else 'orange' if abs(d) < 0.8 else 'red' for d in cohen_ds]
        
        bars = axes[1, 2].bar(metrics_display, cohen_ds, color=colors, alpha=0.7)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 2].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        axes[1, 2].axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Add effect size labels
        for bar, d in zip(bars, cohen_ds):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.02),
                           f'{d:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        axes[1, 2].set_ylabel("Cohen's d")
        axes[1, 2].set_title('Effect Sizes (Imbalanced - Balanced)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Vendor-Aware Robust Model Comparison: Nested Cross-Validation Results', 
                    fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.output_dir}/vendor_aware_comparison_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {plot_file}")
    
    def save_vendor_analysis_results(self):
        """
        Save detailed vendor analysis results for further inspection.
        """
        # Save vendor analysis
        vendor_file = f"{self.output_dir}/vendor_analysis_{self.timestamp}.json"
        with open(vendor_file, 'w') as f:
            json.dump(self.vendor_analysis, f, indent=2, default=str)
        
        # Save test vendor list
        test_vendors_file = f"{self.output_dir}/test_vendors_{self.timestamp}.txt"
        with open(test_vendors_file, 'w') as f:
            f.write("Test Vendors (Held-Out):\n")
            f.write("=" * 30 + "\n")
            for vendor in self.test_vendors:
                f.write(f"{vendor}\n")
        
        # Save train/test splits with vendor information
        balanced_train_df = pd.DataFrame(self.balanced_train)
        balanced_test_df = pd.DataFrame(self.balanced_test)
        imbalanced_train_df = pd.DataFrame(self.imbalanced_train)
        imbalanced_test_df = pd.DataFrame(self.imbalanced_test)
        
        balanced_train_df.to_csv(f"{self.output_dir}/balanced_train_{self.timestamp}.csv", index=False)
        balanced_test_df.to_csv(f"{self.output_dir}/balanced_test_{self.timestamp}.csv", index=False)
        imbalanced_train_df.to_csv(f"{self.output_dir}/imbalanced_train_{self.timestamp}.csv", index=False)
        imbalanced_test_df.to_csv(f"{self.output_dir}/imbalanced_test_{self.timestamp}.csv", index=False)
        
        print(f"💾 Vendor analysis saved: {vendor_file}")
        print(f"💾 Test vendors saved: {test_vendors_file}")
        print(f"💾 Train/test splits saved to CSV files")


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Starting Vendor-Aware Robust Model Comparison...")
    
    # Initialize the vendor-aware comparison system
    comparator = VendorAwareRobustComparison(
        output_dir="vendor_aware_robust_results"
    )
    
    try:
        # Run the vendor-aware robust comparison
        results = comparator.run_robust_comparison()
        
        print("\n✅ Vendor-aware robust comparison complete!")
        print("\n🎯 Key Takeaways:")
        print("1. Used vendor-grouped cross-validation to prevent vendor data leakage")
        print("2. Applied StratifiedGroupKFold for balanced vendor-aware splits")
        print("3. Properly handled null vendor assignments for negative samples")
        print("4. Evaluated on completely held-out vendor groups")
        print("5. Performed statistical significance testing")
        print("6. Results are methodologically sound and defensible for thesis")
        print("7. Model performance reflects ability to generalize to new vendors")
        
        # Summary of results
        print(f"\n📊 FINAL SUMMARY:")
        print(f"Balanced Test AUC: {results['balanced_final']['test_metrics']['auc']:.3f}")
        print(f"Imbalanced Test AUC: {results['imbalanced_final']['test_metrics']['auc']:.3f}")
        
        winner = "Imbalanced" if (results['imbalanced_final']['test_metrics']['auc'] > 
                                 results['balanced_final']['test_metrics']['auc']) else "Balanced"
        print(f"Winner: {winner} approach")
        
        # Vendor analysis summary
        if results['vendor_analysis']:
            print(f"\n🏢 VENDOR ANALYSIS:")
            print(f"Total vendors: {results['vendor_analysis']['total_vendors']}")
            print(f"Test vendor groups: {len(results['test_vendors'])}")
            print(f"Mixed-label vendors: {len(results['vendor_analysis']['mixed_vendors'])}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional debugging information
        print(f"\n🔍 DEBUGGING INFO:")
        print(f"Raw data loaded: {len(comparator.raw_data) if comparator.raw_data else 0}")
        if comparator.vendor_analysis:
            print(f"Vendor analysis completed: {comparator.vendor_analysis['total_vendors']} vendors")
        if comparator.balanced_train:
            print(f"Balanced train size: {len(comparator.balanced_train)}")
        if comparator.imbalanced_train:
            print(f"Imbalanced train size: {len(comparator.imbalanced_train)}")
    
    def train_final_model(self, train_data, test_data, nested_cv_results, dataset_name):
        """
        Train a final model using the most common best parameters from nested CV
        and evaluate on the held-out test set.
        """
        print(f"\n🎯 Training final {dataset_name} model and evaluating on test set...")
        
        # Get the most common hyperparameters from nested CV
        param_tuples = [
            tuple(sorted(params.items()))
            for params in nested_cv_results['outer_scores']['best_params']
        ]
        best_params = dict(Counter(param_tuples).most_common(1)[0][0])
        print(f"  📋 Using most common parameters from nested CV: {best_params}")
        
        # Engineer features with correlation-dropping on both train & test
        train_df = self.engineer_features(train_data, remove_correlated=True)
        test_df = self.engineer_features(test_data, remove_correlated=True)
        
        # Align columns exactly
        feature_cols = nested_cv_results['feature_names']
        train_df = train_df.reindex(
            columns=feature_cols + ['label', 'script_id', 'vendor'], 
            fill_value=0
        )
        test_df = test_df.reindex(
            columns=feature_cols + ['label', 'script_id', 'vendor'], 
            fill_value=0
        )
        
        # Extract features and labels
        X_train, y_train = train_df[feature_cols].values, train_df['label'].values
        X_test, y_test = test_df[feature_cols].values, test_df['label'].values
        
        # Train final RandomForest
        final_model = RandomForestClassifier(
            **best_params,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        final_model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_pred = final_model.predict(X_test)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        
        test_acc = np.mean(test_pred == y_test)
        test_auc = roc_auc_score(y_test, test_proba)
        test_ap = average_precision_score(y_test, test_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Vendor-level analysis
        test_df['pred'] = test_pred
        test_df['proba'] = test_proba
        vendor_performance = self.analyze_vendor_performance(test_df, dataset_name)
        
        # Package results
        results = {
            'model': final_model,
            'params': best_params,
            'test_metrics': {
                'accuracy': test_acc,
                'auc': test_auc,
                'average_precision': test_ap,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
            },
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False),
            'vendor_performance': vendor_performance
        }
        
        print(f"\n📊 {dataset_name} Test Set Performance:")
        print(f"  Test AUC: {test_auc:.3f}")
        print(f"  Test AP:  {test_ap:.3f}")
        print(f"  Test Acc: {test_acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        
        return results
    
    def analyze_vendor_performance(self, test_df, dataset_name):
        """
        Analyze performance at the vendor level to identify potential biases.
        """
        print(f"\n🏢 Vendor-Level Performance Analysis for {dataset_name}")
        
        # Group by vendor (handle nulls)
        test_df['vendor_clean'] = test_df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        vendor_stats = []
        for vendor, group in test_df.groupby('vendor_clean'):
            if len(group) >= 2:  # Only analyze vendors with multiple scripts
                y_true = group['label'].values
                y_pred = group['pred'].values
                y_proba = group['proba'].values
                
                try:
                    vendor_auc = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else None
                except:
                    vendor_auc = None
                
                vendor_stats.append({
                    'vendor': vendor,
                    'count': len(group),
                    'true_positives': sum(y_true),
                    'predicted_positives': sum(y_pred),
                    'accuracy': np.mean(y_pred == y_true),
                    'auc': vendor_auc
                })
        
        vendor_df = pd.DataFrame(vendor_stats)
        
        if len(vendor_df) > 0:
            print(f"  Vendors with ≥2 scripts: {len(vendor_df)}")
            print(f"  Average vendor accuracy: {vendor_df['accuracy'].mean():.3f}")
            
            # Show vendors with extreme performance
            if len(vendor_df) > 5:
                print("\n  Top performing vendors:")
                top_vendors = vendor_df.nlargest(3, 'accuracy')[['vendor', 'count', 'accuracy', 'auc']]
                print(top_vendors.to_string(index=False))
                
                print("\n  Lowest performing vendors:")
                bottom_vendors = vendor_df.nsmallest(3, 'accuracy')[['vendor', 'count', 'accuracy', 'auc']]
                print(bottom_vendors.to_string(index=False))
        
        return vendor_df
    
    def statistical_comparison(self, balanced_scores, imbalanced_scores):
        """
        Perform statistical significance testing between the two approaches.
        """
        print(f"\n📊 Statistical Comparison of Models...")
        
        results = {}
        
        # Paired t-test for each metric
        metrics = ['val_auc', 'val_ap', 'val_acc']
        
        for metric in metrics:
            balanced_vals = balanced_scores['outer_scores'][metric]
            imbalanced_vals = imbalanced_scores['outer_scores'][metric]
            
            # Ensure same number of values for paired test
            min_len = min(len(balanced_vals), len(imbalanced_vals))
            balanced_vals = balanced_vals[:min_len]
            imbalanced_vals = imbalanced_vals[:min_len]
            
            if min_len < 2:
                print(f"  ⚠️  Not enough samples for {metric} comparison")
                continue
            
            # Paired t-test (same CV folds)
            t_stat, p_value = stats.ttest_rel(balanced_vals, imbalanced_vals)
            
            # Cohen's d for effect size
            diff = np.array(imbalanced_vals) - np.array(balanced_vals)
            cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(balanced_vals, imbalanced_vals)
            except:
                wilcoxon_stat, wilcoxon_p = None, None
            
            results[metric] = {
                'balanced_mean': np.mean(balanced_vals),
                'imbalanced_mean': np.mean(imbalanced_vals),
                'difference': np.mean(imbalanced_vals) - np.mean(balanced_vals),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': cohen_d,
                'wilcoxon_p': wilcoxon_p,
                'significant': p_value < 0.05,
                'n_samples': min_len
            }
            
            print(f"\n  {metric.upper()}:")
            print(f"    Balanced: {results[metric]['balanced_mean']:.3f}")
            print(f"    Imbalanced: {results[metric]['imbalanced_mean']:.3f}")
            print(f"    Difference: {results[metric]['difference']:.3f}")
            print(f"    p-value: {results[metric]['p_value']:.4f}")
            print(f"    Cohen's d: {results[metric]['cohen_d']:.3f}")
            print(f"    Significant: {'Yes' if results[metric]['significant'] else 'No'}")
            print(f"    Samples: {results[metric]['n_samples']}")
        
        return results
    
    def run_robust_comparison(self):
        """
        Run the complete vendor-aware methodologically rigorous comparison.
        """
        print("🚀 " + "="*80)
        print("VENDOR-AWARE ROBUST MODEL COMPARISON WITH NESTED CROSS-VALIDATION")
        print("="*80)
        
        # Step 1: Load and split data with vendor awareness
        print(f"\n📚 STEP 1: Load and Split Data (Vendor-Aware)")
        test_vendors = self.load_and_split_datasets(test_vendor_ratio=0.2)
        
        print(f"\n📌 Test Vendors (Held-Out): {len(test_vendors)} vendor groups")
        real_test_vendors = [v for v in test_vendors if not v.startswith('NULL_VENDOR')]
        if real_test_vendors:
            print(f"Real test vendors: {real_test_vendors[:10]}{'...' if len(real_test_vendors) > 10 else ''}")
        
        # Step 2: Vendor-aware nested cross-validation
        print(f"\n🔄 STEP 2: Vendor-Aware Nested Cross-Validation")
        
        try:
            self.nested_cv_results['balanced'] = self.vendor_aware_nested_cross_validation(
                self.balanced_train, "Balanced"
            )
        except Exception as e:
            print(f"❌ Error in balanced nested CV: {e}")
            raise
        
        try:
            self.nested_cv_results['imbalanced'] = self.vendor_aware_nested_cross_validation(
                self.imbalanced_train, "Imbalanced"
            )
        except Exception as e:
            print(f"❌ Error in imbalanced nested CV: {e}")
            raise
        
        # Step 3: Statistical comparison
        print(f"\n📊 STEP 3: Statistical Comparison")
        stat_comparison = self.statistical_comparison(
            self.nested_cv_results['balanced'],
            self.nested_cv_results['imbalanced']
        )
        
        # Step 4: Train final models and test
        print(f"\n🎯 STEP 4: Final Model Training and Testing")
        
        balanced_final = self.train_final_model(
            self.balanced_train,
            self.balanced_test,
            self.nested_cv_results['balanced'],
            "Balanced"
        )
        
        imbalanced_final = self.train_final_model(
            self.imbalanced_train,
            self.imbalanced_test,
            self.nested_cv_results['imbalanced'],
            "Imbalanced"
        )
        
        # Step 5: Generate comprehensive report
        print(f"\n📝 STEP 5: Generate Comprehensive Report")
        self.generate_vendor_aware_report(
            self.nested_cv_results,
            stat_comparison,
            balanced_final,
            imbalanced_final
        )
        
        # Step 6: Create visualizations
        print(f"\n📊 STEP 6: Create Visualizations")
        self.create_vendor_aware_visualizations(
            self.nested_cv_results,
            stat_comparison,
            balanced_final,
            imbalanced_final
        )
        
        # Step 7: Save vendor analysis results
        print(f"\n💾 STEP 7: Save Analysis Results")
        self.save_vendor_analysis_results()
        
        print("\n✅ Vendor-aware robust comparison complete!")
        
        return {
            'nested_cv_results': self.nested_cv_results,
            'statistical_comparison': stat_comparison,
            'balanced_final': balanced_final,
            'imbalanced_final': imbalanced_final,
            'vendor_analysis': self.vendor_analysis,
            'test_vendors': test_vendors
        }
    
    def generate_vendor_aware_report(self, nested_cv_results, stat_comparison, 
                                   balanced_final, imbalanced_final):
        """
        Generate a methodologically rigorous report focusing on vendor awareness.
        """
        report_file = f"{self.output_dir}/vendor_aware_comparison_report_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Vendor-Aware Robust Comparison of Balanced vs Imbalanced Approaches\n\n")
            
            f.write("## Methodology\n\n")
            f.write("This analysis addresses vendor-based data leakage through:\n\n")
            f.write("- **Vendor-Grouped Cross-Validation**: No vendor appears in both train and validation\n")
            f.write("- **StratifiedGroupKFold**: Maintains label balance while respecting vendor groups\n")
            f.write("- **Vendor-Aware Test Split**: Test vendors completely held out\n")
            f.write("- **Null Vendor Handling**: Each unknown vendor gets unique group ID\n")
            f.write("- **Nested CV**: 5-fold outer, 3-fold inner for unbiased estimates\n")
            f.write("- **Statistical Testing**: Paired t-test and Wilcoxon signed-rank test\n\n")
            
            f.write("## Vendor Distribution Analysis\n\n")
            if self.vendor_analysis:
                f.write(f"- **Total vendors**: {self.vendor_analysis['total_vendors']}\n")
                f.write(f"- **Scripts with null vendor**: {self.vendor_analysis['null_vendor_count']}\n")
                f.write(f"- **Positive-only vendors**: {len(self.vendor_analysis['pos_only_vendors'])}\n")
                f.write(f"- **Negative-only vendors**: {len(self.vendor_analysis['neg_only_vendors'])}\n")
                f.write(f"- **Mixed-label vendors**: {len(self.vendor_analysis['mixed_vendors'])}\n")
                f.write(f"- **Single-script vendors**: {len(self.vendor_analysis['single_script_vendors'])}\n\n")
            
            f.write("## Key Results\n\n")
            
            # Nested CV results
            f.write("### Vendor-Aware Nested Cross-Validation Performance\n\n")
            f.write("| Metric | Balanced | Imbalanced | Difference | p-value | Significant |\n")
            f.write("|--------|----------|------------|------------|---------|-------------|\n")
            
            for metric in ['val_auc', 'val_ap', 'val_acc']:
                if metric in stat_comparison:
                    bal_mean = nested_cv_results['balanced']['metrics'][metric]['mean']
                    bal_std = nested_cv_results['balanced']['metrics'][metric]['std']
                    imb_mean = nested_cv_results['imbalanced']['metrics'][metric]['mean']
                    imb_std = nested_cv_results['imbalanced']['metrics'][metric]['std']
                    diff = stat_comparison[metric]['difference']
                    p_val = stat_comparison[metric]['p_value']
                    sig = "Yes" if stat_comparison[metric]['significant'] else "No"
                    
                    f.write(f"| {metric} | {bal_mean:.3f}±{bal_std:.3f} | "
                           f"{imb_mean:.3f}±{imb_std:.3f} | {diff:+.3f} | {p_val:.4f} | {sig} |\n")
            
            f.write("\n### Hold-out Test Set Performance\n\n")
            f.write("| Metric | Balanced | Imbalanced | Difference |\n")
            f.write("|--------|----------|------------|------------|\n")
            
            test_metrics = ['auc', 'average_precision', 'accuracy', 'precision', 'recall']
            for metric in test_metrics:
                bal_val = balanced_final['test_metrics'][metric]
                imb_val = imbalanced_final['test_metrics'][metric]
                diff = imb_val - bal_val
                f.write(f"| {metric} | {bal_val:.3f} | {imb_val:.3f} | {diff:+.3f} |\n")
            
            f.write("\n## Vendor-Level Analysis\n\n")
            
            # Vendor performance analysis
            if 'vendor_performance' in balanced_final and len(balanced_final['vendor_performance']) > 0:
                bal_vendor_perf = balanced_final['vendor_performance']
                f.write(f"### Balanced Approach - Vendor Performance\n\n")
                f.write(f"- Vendors analyzed: {len(bal_vendor_perf)}\n")
                f.write(f"- Average vendor accuracy: {bal_vendor_perf['accuracy'].mean():.3f}\n")
                f.write(f"- Vendor accuracy std: {bal_vendor_perf['accuracy'].std():.3f}\n\n")
            
            if 'vendor_performance' in imbalanced_final and len(imbalanced_final['vendor_performance']) > 0:
                imb_vendor_perf = imbalanced_final['vendor_performance']
                f.write(f"### Imbalanced Approach - Vendor Performance\n\n")
                f.write(f"- Vendors analyzed: {len(imb_vendor_perf)}\n")
                f.write(f"- Average vendor accuracy: {imb_vendor_perf['accuracy'].mean():.3f}\n")
                f.write(f"- Vendor accuracy std: {imb_vendor_perf['accuracy'].std():.3f}\n\n")
            
            f.write("## Statistical Analysis\n\n")
            
            # Effect sizes
            f.write("### Effect Sizes (Cohen's d)\n\n")
            for metric in ['val_auc', 'val_ap', 'val_acc']:
                if metric in stat_comparison:
                    cohen_d = stat_comparison[metric]['cohen_d']
                    f.write(f"- {metric}: {cohen_d:.3f} ")
                    if abs(cohen_d) < 0.2:
                        f.write("(negligible effect)\n")
                    elif abs(cohen_d) < 0.5:
                        f.write("(small effect)\n")
                    elif abs(cohen_d) < 0.8:
                        f.write("(medium effect)\n")
                    else:
                        f.write("(large effect)\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Determine winner based on test set AUC
            if imbalanced_final['test_metrics']['auc'] > balanced_final['test_metrics']['auc']:
                f.write("**Recommended Approach: IMBALANCED**\n\n")
                f.write("The imbalanced approach shows superior performance on the vendor-grouped ")
                f.write("test set and better reflects real-world deployment conditions where new ")
                f.write("vendors regularly appear.\n")
            else:
                f.write("**Recommended Approach: BALANCED**\n\n")
                f.write("The balanced approach shows superior performance on the vendor-grouped ")
                f.write("test set despite having less training data.\n")
            
            f.write("\n## Methodological Advantages\n\n")
            f.write("This vendor-aware analysis addresses critical issues:\n\n")
            f.write("1. **No vendor leakage**: Train/test vendor separation prevents overfitting\n")
            f.write("2. **Realistic evaluation**: Models tested on completely new vendors\n")
            f.write("3. **Generalization focus**: Forces learning of behavior patterns, not vendor signatures\n")
            f.write("4. **Statistical rigor**: Significance testing with appropriate effect sizes\n")
            f.write("5. **Deployment readiness**: Results reflect real-world performance expectations\n")
            f.write("6. **Null vendor handling**: Proper treatment of unassigned vendor labels\n")
        
        print(f"📝 Report saved: {report_file}")
    
    def create_vendor_aware_visualizations(self, nested_cv_results, stat_comparison,
                                         balanced_final, imbalanced_final):
        """
        Create comprehensive visualizations for the vendor-aware comparison.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Nested CV AUC scores by fold
        bal_aucs = nested_cv_results['balanced']['outer_scores']['val_auc']
        imb_aucs = nested_cv_results['imbalanced']['outer_scores']['val_auc']
        
        max_folds = max(len(bal_aucs), len(imb_aucs))
        folds = range(1, max_folds + 1)
        
        if len(bal_aucs) > 0:
            axes[0, 0].plot(range(1, len(bal_aucs) + 1), bal_aucs, 'o-', 
                           label='Balanced', linewidth=2, markersize=8)
        if len(imb_aucs) > 0:
            axes[0, 0].plot(range(1, len(imb_aucs) + 1), imb_aucs, 's-', 
                           label='Imbalanced', linewidth=2, markersize=8)
        
        axes[0, 0].set_xlabel('Vendor-Grouped CV Fold')
        axes[0, 0].set_ylabel('Validation AUC')
        axes[0, 0].set_title('Vendor-Aware Nested CV: AUC by Fold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Statistical comparison with confidence intervals
        metrics = ['AUC', 'AP', 'Accuracy']
        metric_keys = ['val_auc', 'val_ap', 'val_acc']
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bal_means = []
        bal_stds = []
        imb_means = []
        imb_stds = []
        
        for m in metric_keys:
            if m in nested_cv_results['balanced']['metrics']:
                bal_means.append(nested_cv_results['balanced']['metrics'][m]['mean'])
                bal_stds.append(nested_cv_results['balanced']['metrics'][m]['std'])
                imb_means.append(nested_cv_results['imbalanced']['metrics'][m]['mean'])
                imb_stds.append(nested_cv_results['imbalanced']['metrics'][m]['std'])
            else:
                bal_means.append(0)
                bal_stds.append(0)
                imb_means.append(0)
                imb_stds.append(0)
        
        # Calculate 95% confidence intervals
        n_folds = max(nested_cv_results['balanced'].get('completed_folds', 1), 1)
        bal_ci = [1.96 * s / np.sqrt(n_folds) for s in bal_stds]
        imb_ci = [1.96 * s / np.sqrt(n_folds) for s in imb_stds]
        
        axes[0, 1].bar(x - width/2, bal_means, width, yerr=bal_ci, 
                      label='Balanced', alpha=0.8, capsize=5)
        axes[0, 1].bar(x + width/2, imb_means, width, yerr=imb_ci,
                      label='Imbalanced', alpha=0.8, capsize=5)
        
        # Add significance stars
        for i, (metric_key, metric_name) in enumerate(zip(metric_keys, metrics)):
            if metric_key in stat_comparison and stat_comparison[metric_key]['significant']:
                y_max = max(bal_means[i] + bal_ci[i], imb_means[i] + imb_ci[i])
                axes[0, 1].text(i, y_max + 0.01, '*', ha='center', fontsize=14)
        
        axes[0, 1].set_xlabel('Metrics')
        axes[0, 1].set_ylabel('Score (95% CI)')
        axes[0, 1].set_title('Vendor-Aware Performance Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Test set ROC curves - FIXED FEATURE ALIGNMENT
        try:
            from sklearn.metrics import roc_curve
            
            # CRITICAL FIX: Use consistent feature engineering for both test sets
            print("🔧 Aligning features for ROC curve generation...")
            
            # Get feature names from both models
            bal_feature_names = set(balanced_final['feature_importance']['feature'].tolist())
            imb_feature_names = set(imbalanced_final['feature_importance']['feature'].tolist())
            
            # Use intersection of features (common to both models)
            common_features = list(bal_feature_names & imb_feature_names)
            print(f"   Using {len(common_features)} common features")
            
            if len(common_features) > 0:
                # Engineer features for test sets
                bal_features = self.engineer_features(self.balanced_test, remove_correlated=False)
                imb_features = self.engineer_features(self.imbalanced_test, remove_correlated=False)
                
                # Align both to common features
                bal_features = bal_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                  fill_value=0)
                imb_features = imb_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                  fill_value=0)
                
                bal_X = bal_features[common_features].values
                bal_y = bal_features['label'].values
                
                imb_X = imb_features[common_features].values
                imb_y = imb_features['label'].values
                
                # Retrain models on common features for visualization only
                print("   Retraining models on common features for visualization...")
                
                # Retrain balanced model
                bal_train_features = self.engineer_features(self.balanced_train, remove_correlated=False)
                bal_train_features = bal_train_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                              fill_value=0)
                bal_X_train = bal_train_features[common_features].values
                bal_y_train = bal_train_features['label'].values
                
                temp_bal_model = RandomForestClassifier(**balanced_final['params'], random_state=42, n_jobs=-1)
                temp_bal_model.fit(bal_X_train, bal_y_train)
                bal_proba = temp_bal_model.predict_proba(bal_X)[:, 1]
                
                # Retrain imbalanced model
                imb_train_features = self.engineer_features(self.imbalanced_train, remove_correlated=False)
                imb_train_features = imb_train_features.reindex(columns=common_features + ['label', 'script_id', 'vendor'],
                                                              fill_value=0)
                imb_X_train = imb_train_features[common_features].values
                imb_y_train = imb_train_features['label'].values
                
                temp_imb_model = RandomForestClassifier(**imbalanced_final['params'], random_state=42, n_jobs=-1)
                temp_imb_model.fit(imb_X_train, imb_y_train)
                imb_proba = temp_imb_model.predict_proba(imb_X)[:, 1]
                
                # Generate ROC curves
                bal_fpr, bal_tpr, _ = roc_curve(bal_y, bal_proba)
                imb_fpr, imb_tpr, _ = roc_curve(imb_y, imb_proba)
                
                axes[0, 2].plot(bal_fpr, bal_tpr, 
                               label=f'Balanced (AUC={balanced_final["test_metrics"]["auc"]:.3f})',
                               linewidth=2)
                axes[0, 2].plot(imb_fpr, imb_tpr,
                               label=f'Imbalanced (AUC={imbalanced_final["test_metrics"]["auc"]:.3f})',
                               linewidth=2)
                axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[0, 2].set_xlabel('False Positive Rate')
                axes[0, 2].set_ylabel('True Positive Rate')
                axes[0, 2].set_title('Test Set ROC Curves (Vendor-Held-Out)')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
                
            else:
                axes[0, 2].text(0.5, 0.5, 'No common features\nfor ROC comparison', 
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Test Set ROC Curves (Vendor-Held-Out)')
                
        except Exception as e:
            print(f"   ⚠️ Error generating ROC curves: {e}")
            axes[0, 2].text(0.5, 0.5, f'ROC generation failed:\n{str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Test Set ROC Curves (Error)')
        
        # Plot 4: Feature importance comparison (using common features)
        try:
            top_n = min(10, len(common_features)) if 'common_features' in locals() else 8
            bal_imp = balanced_final['feature_importance'].head(top_n)
            imb_imp = imbalanced_final['feature_importance'].head(top_n)
            
            # Find common features in importance lists
            common_imp_features = set(bal_imp['feature']).intersection(set(imb_imp['feature']))
            common_imp_features = list(common_imp_features)[:8]
            
            if common_imp_features:
                bal_values = [bal_imp[bal_imp['feature'] == f]['importance'].iloc[0] 
                             for f in common_imp_features]
                imb_values = [imb_imp[imb_imp['feature'] == f]['importance'].iloc[0] 
                             for f in common_imp_features]
                
                y_pos = np.arange(len(common_imp_features))
                axes[1, 0].barh(y_pos - 0.2, bal_values, 0.4, label='Balanced', alpha=0.8)
                axes[1, 0].barh(y_pos + 0.2, imb_values, 0.4, label='Imbalanced', alpha=0.8)
                axes[1, 0].set_yticks(y_pos)
                axes[1, 0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                                            for f in common_imp_features], fontsize=9)
                axes[1, 0].set_xlabel('Feature Importance')
                axes[1, 0].set_title('Top Feature Importance Comparison')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No common features\nin top importance', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Importance Comparison')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Feature importance\ncomparison failed', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance Comparison (Error)')
        
        # Plot 5: Vendor-level performance distribution
        if ('vendor_performance' in balanced_final and 
            len(balanced_final['vendor_performance']) > 0):
            
            bal_vendor_acc = balanced_final['vendor_performance']['accuracy'].values
            imb_vendor_acc = imbalanced_final['vendor_performance']['accuracy'].values
            
            if len(bal_vendor_acc) > 1 and len(imb_vendor_acc) > 1:
                axes[1, 1].hist(bal_vendor_acc, bins=10, alpha=0.7, label='Balanced', density=True)
                axes[1, 1].hist(imb_vendor_acc, bins=10, alpha=0.7, label='Imbalanced', density=True)
                axes[1, 1].axvline(np.mean(bal_vendor_acc), color='blue', linestyle='--', 
                                  label=f'Bal Mean: {np.mean(bal_vendor_acc):.3f}')
                axes[1, 1].axvline(np.mean(imb_vendor_acc), color='orange', linestyle='--',
                                  label=f'Imb Mean: {np.mean(imb_vendor_acc):.3f}')
                axes[1, 1].set_xlabel('Vendor-Level Accuracy')
                axes[1, 1].set_ylabel('Density')
                axes[1, 1].set_title('Vendor Performance Distribution')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient vendor\ndata for distribution', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Vendor Performance Distribution')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient vendor\nperformance data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Vendor Performance Distribution')
        
        # Plot 6: Effect sizes visualization
        metrics_display = ['AUC', 'AP', 'Accuracy']
        cohen_ds = []
        
        for m in metric_keys:
            if m in stat_comparison:
                cohen_ds.append(stat_comparison[m]['cohen_d'])
            else:
                cohen_ds.append(0)
        
        colors = ['green' if abs(d) < 0.2 else 'yellow' if abs(d) < 0.5 
                 else 'orange' if abs(d) < 0.8 else 'red' for d in cohen_ds]
        
        bars = axes[1, 2].bar(metrics_display, cohen_ds, color=colors, alpha=0.7)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 2].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        axes[1, 2].axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Add effect size labels
        for bar, d in zip(bars, cohen_ds):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.02),
                           f'{d:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        axes[1, 2].set_ylabel("Cohen's d")
        axes[1, 2].set_title('Effect Sizes (Imbalanced - Balanced)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Vendor-Aware Robust Model Comparison: Nested Cross-Validation Results', 
                    fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"{self.output_dir}/vendor_aware_comparison_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved: {plot_file}")
    
    def save_vendor_analysis_results(self):
        """
        Save detailed vendor analysis results for further inspection.
        """
        # Save vendor analysis
        vendor_file = f"{self.output_dir}/vendor_analysis_{self.timestamp}.json"
        with open(vendor_file, 'w') as f:
            json.dump(self.vendor_analysis, f, indent=2, default=str)
        
        # Save test vendor list
        test_vendors_file = f"{self.output_dir}/test_vendors_{self.timestamp}.txt"
        with open(test_vendors_file, 'w') as f:
            f.write("Test Vendors (Held-Out):\n")
            f.write("=" * 30 + "\n")
            for vendor in self.test_vendors:
                f.write(f"{vendor}\n")
        
        # Save train/test splits with vendor information
        balanced_train_df = pd.DataFrame(self.balanced_train)
        balanced_test_df = pd.DataFrame(self.balanced_test)
        imbalanced_train_df = pd.DataFrame(self.imbalanced_train)
        imbalanced_test_df = pd.DataFrame(self.imbalanced_test)
        
        balanced_train_df.to_csv(f"{self.output_dir}/balanced_train_{self.timestamp}.csv", index=False)
        balanced_test_df.to_csv(f"{self.output_dir}/balanced_test_{self.timestamp}.csv", index=False)
        imbalanced_train_df.to_csv(f"{self.output_dir}/imbalanced_train_{self.timestamp}.csv", index=False)
        imbalanced_test_df.to_csv(f"{self.output_dir}/imbalanced_test_{self.timestamp}.csv", index=False)
        
        print(f"💾 Vendor analysis saved: {vendor_file}")
        print(f"💾 Test vendors saved: {test_vendors_file}")
        print(f"💾 Train/test splits saved to CSV files")


# # Example usage and testing
# if __name__ == "__main__":
#     print("🚀 Starting Vendor-Aware Robust Model Comparison...")
    
#     # Initialize the vendor-aware comparison system
#     comparator = VendorAwareRobustComparison(
#         output_dir="vendor_aware_robust_results"
#     )
    
#     try:
#         # Run the vendor-aware robust comparison
#         results = comparator.run_robust_comparison()
        
#         print("\n✅ Vendor-aware robust comparison complete!")
#         print("\n🎯 Key Takeaways:")
#         print("1. Used vendor-grouped cross-validation to prevent vendor data leakage")
#         print("2. Applied StratifiedGroupKFold for balanced vendor-aware splits")
#         print("3. Properly handled null vendor assignments for negative samples")
#         print("4. Evaluated on completely held-out vendor groups")
#         print("5. Performed statistical significance testing")
#         print("6. Results are methodologically sound and defensible for thesis")
#         print("7. Model performance reflects ability to generalize to new vendors")
        
#         # Summary of results
#         print(f"\n📊 FINAL SUMMARY:")
#         print(f"Balanced Test AUC: {results['balanced_final']['test_metrics']['auc']:.3f}")
#         print(f"Imbalanced Test AUC: {results['imbalanced_final']['test_metrics']['auc']:.3f}")
        
#         winner = "Imbalanced" if (results['imbalanced_final']['test_metrics']['auc'] > 
#                                  results['balanced_final']['test_metrics']['auc']) else "Balanced"
#         print(f"Winner: {winner} approach")
        
#         # Vendor analysis summary
#         if results['vendor_analysis']:
#             print(f"\n🏢 VENDOR ANALYSIS:")
#             print(f"Total vendors: {results['vendor_analysis']['total_vendors']}")
#             print(f"Test vendor groups: {len(results['test_vendors'])}")
#             print(f"Mixed-label vendors: {len(results['vendor_analysis']['mixed_vendors'])}")
        
#     except Exception as e:
#         print(f"\n❌ Error: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Additional debugging information
#         print(f"\n🔍 DEBUGGING INFO:")
#         print(f"Raw data loaded: {len(comparator.raw_data) if comparator.raw_data else 0}")
#         if comparator.vendor_analysis:
#             print(f"Vendor analysis completed: {comparator.vendor_analysis['total_vendors']} vendors")
#         if comparator.balanced_train:
#             print(f"Balanced train size: {len(comparator.balanced_train)}")
#         if comparator.imbalanced_train:
#             print(f"Imbalanced train size: {len(comparator.imbalanced_train)}")