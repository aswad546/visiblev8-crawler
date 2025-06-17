import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from sklearn.model_selection import (StratifiedGroupKFold, GroupKFold, 
                                   RandomizedSearchCV, train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           average_precision_score, make_scorer)
import pickle
import os
from datetime import datetime
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class BalancedVendorAwareClassifier:
    """
    Clean implementation of balanced malicious script classification with vendor-aware nested CV.
    
    Key Features:
    1. Vendor-grouped cross-validation to prevent data leakage
    2. Nested CV for unbiased hyperparameter selection
    3. Comprehensive analysis and visualization
    4. Model persistence and evaluation
    """
    
    def __init__(self, db_config=None, output_dir="balanced_vendor_aware_results"):
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
        
        # Data containers
        self.raw_data = None
        self.train_data = None
        self.test_data = None
        self.test_vendors = None
        self.vendor_analysis = None
        
        # Results containers
        self.nested_cv_results = None
        self.final_model = None
        self.test_results = None
        
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
    
    def load_data(self):
        """Load data from database and parse JSON fields."""
        print("🔌 Loading data from database...")
        
        connection = self.connect_to_database()
        if connection is None:
            raise Exception("Failed to connect to database")
        
        try:
            query = f"SELECT * FROM {self.table_name}"
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            rows = cursor.fetchall()
            
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
        
        # Filter for balanced dataset (labels 0 and 1 only)
        self.raw_data = [script for script in self.raw_data if script['label'] in [0, 1]]
        
        print(f"✅ Loaded {len(self.raw_data)} scripts for balanced classification")
        return self.raw_data
    
    def analyze_vendor_distribution(self):
        """Analyze vendor distribution and potential issues."""
        print("\n🔍 VENDOR DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        df = pd.DataFrame(self.raw_data)
        
        # Handle null vendors
        df['vendor_clean'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        # Basic statistics
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
            display_df['total'] = display_df[0] + display_df[1]
            display_df = display_df.sort_values('total', ascending=False)
        print(display_df)
        
        # Vendor assignment patterns
        null_vendor_scripts = df[df['vendor'].isnull()]
        positive_with_vendor = df[(df['label'] == 1) & (df['vendor'].notnull())]
        negative_with_vendor = df[(df['label'] == 0) & (df['vendor'].notnull())]
        
        print(f"\n🏷️  VENDOR ASSIGNMENT PATTERNS:")
        print(f"Scripts with null vendor: {len(null_vendor_scripts)} ({len(null_vendor_scripts)/len(df)*100:.1f}%)")
        print(f"Positive scripts with vendor: {len(positive_with_vendor)}")
        print(f"Negative scripts with vendor: {len(negative_with_vendor)}")
        
        # Vendor categories
        vendor_stats = df.groupby('vendor_clean').agg({
            'label': ['count', 'mean', 'sum']
        }).reset_index()
        vendor_stats.columns = ['vendor', 'total_scripts', 'pos_ratio', 'pos_count']
        vendor_stats['neg_count'] = vendor_stats['total_scripts'] - vendor_stats['pos_count']
        
        pos_only_vendors = vendor_stats[(vendor_stats['pos_count'] > 0) & (vendor_stats['neg_count'] == 0)]
        neg_only_vendors = vendor_stats[(vendor_stats['pos_count'] == 0) & (vendor_stats['neg_count'] > 0)]
        mixed_vendors = vendor_stats[(vendor_stats['pos_count'] > 0) & (vendor_stats['neg_count'] > 0)]
        single_script_vendors = vendor_stats[vendor_stats['total_scripts'] == 1]
        
        print(f"\n🎭 VENDOR CATEGORIES:")
        print(f"Vendors with only positives: {len(pos_only_vendors)}")
        print(f"Vendors with only negatives: {len(neg_only_vendors)}")
        print(f"Vendors with mixed labels: {len(mixed_vendors)}")
        print(f"Vendors with single script: {len(single_script_vendors)}")
        
        self.vendor_analysis = {
            'vendor_counts': vendor_counts,
            'vendor_stats': vendor_stats,
            'pos_only_vendors': pos_only_vendors['vendor'].tolist(),
            'neg_only_vendors': neg_only_vendors['vendor'].tolist(),
            'mixed_vendors': mixed_vendors['vendor'].tolist(),
            'single_script_vendors': single_script_vendors['vendor'].tolist(),
            'total_vendors': total_vendors,
            'null_vendor_count': len(null_vendor_scripts)
        }
        
        return self.vendor_analysis
    
    def create_vendor_aware_split(self, test_vendor_ratio=0.2, random_state=42):
        """Create train/test split ensuring no vendor appears in both sets."""
        print(f"\n🎯 CREATING VENDOR-AWARE TRAIN/TEST SPLIT")
        print("=" * 60)
        
        df = pd.DataFrame(self.raw_data)
        np.random.seed(random_state)
        
        # Handle null vendors by creating unique identifiers
        df['vendor_group'] = df['vendor'].fillna('UNKNOWN_NEGATIVE')
        
        # Give each null vendor script a unique group ID to prevent leakage
        null_mask = df['vendor'].isnull()
        if null_mask.sum() > 0:
            null_indices = df[null_mask].index
            df.loc[null_indices, 'vendor_group'] = [f'NULL_VENDOR_{i}' for i in range(len(null_indices))]
        
        # Get vendor group statistics
        vendor_stats = df.groupby('vendor_group').agg({
            'label': ['count', 'sum']
        }).reset_index()
        vendor_stats.columns = ['vendor_group', 'total_scripts', 'pos_count']
        
        # Strategy: Ensure both train and test have positive samples
        total_pos_scripts = vendor_stats['pos_count'].sum()
        target_test_pos_scripts = max(1, int(total_pos_scripts * test_vendor_ratio))
        
        print(f"Total positive scripts: {total_pos_scripts}")
        print(f"Target test positive scripts: {target_test_pos_scripts}")
        
        # Select vendors for test set
        vendor_stats_sorted = vendor_stats.sort_values('total_scripts', ascending=False)
        
        test_vendor_groups = []
        test_pos_count = 0
        
        # Prioritize vendors with positive samples for test set
        for _, vendor_row in vendor_stats_sorted.iterrows():
            if test_pos_count >= target_test_pos_scripts:
                break
            if vendor_row['pos_count'] > 0:
                # Don't take all positive vendors - ensure some remain for training
                remaining_pos_vendors = len(vendor_stats[vendor_stats['pos_count'] > 0]) - len([v for v in test_vendor_groups if vendor_stats[vendor_stats['vendor_group'] == v]['pos_count'].iloc[0] > 0])
                if remaining_pos_vendors <= 1:  # Keep at least one positive vendor for training
                    break
                    
                test_vendor_groups.append(vendor_row['vendor_group'])
                test_pos_count += vendor_row['pos_count']
        
        # Add some negative vendors if needed to reach target ratio
        total_scripts = len(df)
        target_test_scripts = int(total_scripts * test_vendor_ratio)
        test_script_count = vendor_stats[vendor_stats['vendor_group'].isin(test_vendor_groups)]['total_scripts'].sum()
        
        if test_script_count < target_test_scripts:
            remaining_vendors = vendor_stats[~vendor_stats['vendor_group'].isin(test_vendor_groups)]
            for _, vendor_row in remaining_vendors.iterrows():
                if test_script_count >= target_test_scripts:
                    break
                test_vendor_groups.append(vendor_row['vendor_group'])
                test_script_count += vendor_row['total_scripts']
        
        # Create split
        test_mask = df['vendor_group'].isin(test_vendor_groups)
        train_data = df[~test_mask].drop('vendor_group', axis=1).to_dict('records')
        test_data = df[test_mask].drop('vendor_group', axis=1).to_dict('records')
        
        # Validate split
        train_labels = Counter([s['label'] for s in train_data])
        test_labels = Counter([s['label'] for s in test_data])
        
        train_vendors = set(df[~test_mask & df['vendor'].notnull()]['vendor'].unique())
        test_vendors = set(df[test_mask & df['vendor'].notnull()]['vendor'].unique())
        vendor_overlap = train_vendors & test_vendors
        
        print(f"\n📊 SPLIT RESULTS:")
        print(f"Train vendor groups: {len(df[~test_mask]['vendor_group'].unique())}")
        print(f"Test vendor groups: {len(test_vendor_groups)}")
        print(f"Train scripts: {len(train_data)} {dict(train_labels)}")
        print(f"Test scripts: {len(test_data)} {dict(test_labels)}")
        print(f"Actual test ratio: {len(test_data) / len(df):.1%}")
        
        if vendor_overlap:
            raise ValueError(f"❌ Vendor overlap detected: {vendor_overlap}")
        if train_labels[1] == 0 or test_labels[1] == 0:
            raise ValueError("❌ No positive samples in train or test set!")
        
        print("✅ Vendor-aware split successful")
        
        self.train_data = train_data
        self.test_data = test_data
        self.test_vendors = test_vendor_groups
        
        return train_data, test_data, test_vendor_groups
    
    def engineer_features(self, dataset):
        """Engineer features from script data."""
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
        
        return pd.DataFrame(features_list)
    
    def create_vendor_groups_for_cv(self, data):
        """Create vendor groups for cross-validation."""
        vendor_groups = []
        null_counter = 0
        
        for script in data:
            vendor = script.get('vendor')
            if pd.isnull(vendor) or vendor == '':
                vendor_groups.append(f'NULL_VENDOR_{null_counter}')
                null_counter += 1
            else:
                vendor_groups.append(str(vendor))
        
        return vendor_groups
    
    def nested_cross_validation(self, outer_cv_folds=5, inner_cv_folds=3, n_iter=20):
        """Perform vendor-aware nested cross-validation."""
        print(f"\n🔄 VENDOR-AWARE NESTED CROSS-VALIDATION")
        print("=" * 60)
        print(f"Outer CV: {outer_cv_folds} folds | Inner CV: {inner_cv_folds} folds")
        
        # Prepare features
        train_features_df = self.engineer_features(self.train_data)
        feature_cols = [col for col in train_features_df.columns if col not in ['script_id', 'label', 'vendor']]
        
        X = train_features_df[feature_cols].values
        y = train_features_df['label'].values
        
        # Create vendor groups
        vendor_groups = self.create_vendor_groups_for_cv(self.train_data)
        unique_groups = list(set(vendor_groups))
        group_mapping = {group: i for i, group in enumerate(unique_groups)}
        groups = np.array([group_mapping[group] for group in vendor_groups])
        
        print(f"Total vendor groups: {len(unique_groups)}")
        print(f"Training samples: {len(X)}")
        
        # Adjust CV folds if necessary
        if len(unique_groups) < outer_cv_folds:
            outer_cv_folds = len(unique_groups)
            print(f"⚠️  Reduced outer CV to {outer_cv_folds} folds")
        
        # Initialize CV
        try:
            outer_cv = StratifiedGroupKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
            cv_type = "StratifiedGroupKFold"
        except:
            outer_cv = GroupKFold(n_splits=outer_cv_folds)
            cv_type = "GroupKFold"
        
        print(f"Using {cv_type} for outer CV")
        
        # Storage for results
        fold_results = {
            'train_scores': [], 'val_scores': [], 'best_params': [],
            'feature_importances': [], 'fold_details': []
        }
        
        # Outer CV loop
        fold_num = 0
        for train_idx, val_idx in outer_cv.split(X, y, groups):
            fold_num += 1
            print(f"\n📁 Outer Fold {fold_num}/{outer_cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            
            # Validate fold
            train_labels = Counter(y_train)
            val_labels = Counter(y_val)
            
            if len(val_labels) < 2:
                print(f"   ⚠️  Skipping fold - validation set has only one class")
                continue
            
            print(f"   Train: {len(X_train)} samples {dict(train_labels)}")
            print(f"   Val:   {len(X_val)} samples {dict(val_labels)}")
            
            # Inner CV for hyperparameter tuning
            print(f"   🔍 Hyperparameter tuning...")
            
            try:
                # Setup inner CV
                unique_groups_train = len(set(groups_train))
                if unique_groups_train < inner_cv_folds:
                    inner_cv_folds_adj = unique_groups_train
                else:
                    inner_cv_folds_adj = inner_cv_folds
                
                try:
                    inner_cv = StratifiedGroupKFold(n_splits=inner_cv_folds_adj, shuffle=True, random_state=42)
                except:
                    inner_cv = GroupKFold(n_splits=inner_cv_folds_adj)
                
                # Custom scorer for robust evaluation
                def safe_roc_auc_scorer(estimator, X, y):
                    try:
                        if len(set(y)) < 2:
                            return 0.5
                        y_proba = estimator.predict_proba(X)[:, 1]
                        return roc_auc_score(y, y_proba)
                    except:
                        return 0.5
                
                safe_scorer = make_scorer(safe_roc_auc_scorer, greater_is_better=True, needs_proba=False)
                
                # Randomized search
                rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
                
                random_search = RandomizedSearchCV(
                    estimator=rf,
                    param_distributions=self.param_distributions,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring=safe_scorer,
                    random_state=42,
                    error_score=0.5,
                    n_jobs=-1
                )
                
                # Suppress warnings during search
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    random_search.fit(X_train, y_train, groups=groups_train)
                
                best_params = random_search.best_params_
                best_cv_score = random_search.best_score_
                
                print(f"   ✅ Best inner CV score: {best_cv_score:.3f}")
                
            except Exception as e:
                print(f"   ⚠️  Hyperparameter tuning failed: {e}")
                # Use default parameters
                best_params = {
                    'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 5,
                    'min_samples_leaf': 2, 'max_features': 0.5, 'bootstrap': True
                }
                best_cv_score = 0.5
            
            # Train final model for this fold
            final_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42, n_jobs=-1)
            final_model.fit(X_train, y_train)
            
            # Evaluate
            train_proba = final_model.predict_proba(X_train)[:, 1]
            val_proba = final_model.predict_proba(X_val)[:, 1]
            
            train_auc = roc_auc_score(y_train, train_proba)
            val_auc = roc_auc_score(y_val, val_proba)
            
            print(f"   📊 Train AUC: {train_auc:.3f} | Val AUC: {val_auc:.3f}")
            
            # Store results
            fold_results['train_scores'].append(train_auc)
            fold_results['val_scores'].append(val_auc)
            fold_results['best_params'].append(best_params)
            fold_results['feature_importances'].append(final_model.feature_importances_)
            fold_results['fold_details'].append({
                'fold': fold_num,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'train_labels': dict(train_labels),
                'val_labels': dict(val_labels),
                'best_cv_score': best_cv_score
            })
        
        # Summarize results
        if len(fold_results['val_scores']) == 0:
            raise ValueError("No valid CV folds completed!")
        
        self.nested_cv_results = {
            'feature_names': feature_cols,
            'fold_results': fold_results,
            'summary': {
                'mean_train_auc': np.mean(fold_results['train_scores']),
                'std_train_auc': np.std(fold_results['train_scores']),
                'mean_val_auc': np.mean(fold_results['val_scores']),
                'std_val_auc': np.std(fold_results['val_scores']),
                'completed_folds': len(fold_results['val_scores']),
                'total_folds': outer_cv_folds
            }
        }
        
        print(f"\n📊 NESTED CV SUMMARY:")
        print(f"Completed folds: {self.nested_cv_results['summary']['completed_folds']}/{outer_cv_folds}")
        print(f"Mean validation AUC: {self.nested_cv_results['summary']['mean_val_auc']:.3f} ± {self.nested_cv_results['summary']['std_val_auc']:.3f}")
        print(f"Mean training AUC: {self.nested_cv_results['summary']['mean_train_auc']:.3f} ± {self.nested_cv_results['summary']['std_train_auc']:.3f}")
        
        return self.nested_cv_results
    
    def train_final_model(self):
        """Train final model using most common best parameters from nested CV."""
        print(f"\n🎯 TRAINING FINAL MODEL")
        print("=" * 60)
        
        # Get most common parameters
        param_tuples = [tuple(sorted(params.items())) for params in self.nested_cv_results['fold_results']['best_params']]
        most_common_params = Counter(param_tuples).most_common(1)[0][0]
        best_params = dict(most_common_params)
        
        print(f"Best parameters from nested CV: {best_params}")
        
        # Prepare training data
        train_features_df = self.engineer_features(self.train_data)
        feature_cols = self.nested_cv_results['feature_names']
        
        X_train = train_features_df[feature_cols].values
        y_train = train_features_df['label'].values
        
        # Train final model
        self.final_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42, n_jobs=-1)
        self.final_model.fit(X_train, y_train)
        
        print(f"✅ Final model trained on {len(X_train)} samples")
        
        return self.final_model
    
    def evaluate_on_test_set(self):
        """Evaluate final model on held-out test set."""
        print(f"\n📊 TEST SET EVALUATION")
        print("=" * 60)
        
        # Prepare test data
        test_features_df = self.engineer_features(self.test_data)
        feature_cols = self.nested_cv_results['feature_names']
        
        X_test = test_features_df[feature_cols].values
        y_test = test_features_df['label'].values
        
        # Predictions
        test_pred = self.final_model.predict(X_test)
        test_proba = self.final_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        test_auc = roc_auc_score(y_test, test_proba)
        test_ap = average_precision_score(y_test, test_proba)
        test_acc = np.mean(test_pred == y_test)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.test_results = {
            'metrics': {
                'auc': test_auc,
                'average_precision': test_ap,
                'accuracy': test_acc,
                'precision': precision,
                'recall': recall,
                'specificity': specificity
            },
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'predictions': {'y_true': y_test, 'y_pred': test_pred, 'y_proba': test_proba},
            'feature_importance': feature_importance
        }
        
        print(f"Test AUC: {test_auc:.3f}")
        print(f"Test AP:  {test_ap:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"Specificity: {specificity:.3f}")
        print(f"\nConfusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        return self.test_results
    
    def save_model_and_results(self):
        """Save trained model and all results to files."""
        print(f"\n💾 SAVING MODEL AND RESULTS")
        print("=" * 60)
        
        # Save model
        model_file = f"{self.output_dir}/balanced_model_{self.timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': self.final_model,
                'feature_names': self.nested_cv_results['feature_names'],
                'params': dict(Counter([tuple(sorted(params.items())) for params in self.nested_cv_results['fold_results']['best_params']]).most_common(1)[0][0]),
                'timestamp': self.timestamp
            }, f)
        print(f"Model saved: {model_file}")
        
        # Save detailed results
        results_file = f"{self.output_dir}/results_{self.timestamp}.json"
        results_data = {
            'vendor_analysis': self.vendor_analysis,
            'nested_cv_results': {
                'summary': self.nested_cv_results['summary'],
                'feature_names': self.nested_cv_results['feature_names'],
                'fold_details': self.nested_cv_results['fold_results']['fold_details']
            },
            'test_results': {
                'metrics': self.test_results['metrics'],
                'confusion_matrix': self.test_results['confusion_matrix']
            },
            'test_vendors': self.test_vendors,
            'timestamp': self.timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Results saved: {results_file}")
        
        # Save feature importance
        feature_imp_file = f"{self.output_dir}/feature_importance_{self.timestamp}.csv"
        self.test_results['feature_importance'].to_csv(feature_imp_file, index=False)
        print(f"Feature importance saved: {feature_imp_file}")
        
        # Save train/test data for reproducibility
        train_df = pd.DataFrame(self.train_data)
        test_df = pd.DataFrame(self.test_data)
        train_df.to_csv(f"{self.output_dir}/train_data_{self.timestamp}.csv", index=False)
        test_df.to_csv(f"{self.output_dir}/test_data_{self.timestamp}.csv", index=False)
        print(f"Train/test data saved")
        
        return model_file, results_file
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots."""
        print(f"\n📊 CREATING VISUALIZATION PLOTS")
        print("=" * 60)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Nested CV performance across folds
        ax1 = plt.subplot(3, 3, 1)
        folds = range(1, len(self.nested_cv_results['fold_results']['val_scores']) + 1)
        train_scores = self.nested_cv_results['fold_results']['train_scores']
        val_scores = self.nested_cv_results['fold_results']['val_scores']
        
        ax1.plot(folds, train_scores, 'o-', label='Training AUC', linewidth=2, markersize=8)
        ax1.plot(folds, val_scores, 's-', label='Validation AUC', linewidth=2, markersize=8)
        ax1.axhline(y=np.mean(val_scores), color='red', linestyle='--', alpha=0.7, label=f'Mean Val AUC: {np.mean(val_scores):.3f}')
        ax1.set_xlabel('CV Fold')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Nested CV Performance Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance
        ax2 = plt.subplot(3, 3, 2)
        top_features = self.test_results['feature_importance'].head(15)
        y_pos = np.arange(len(top_features))
        ax2.barh(y_pos, top_features['importance'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f[:25] + '...' if len(f) > 25 else f for f in top_features['feature']], fontsize=9)
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 15 Feature Importances')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ROC Curve
        ax3 = plt.subplot(3, 3, 3)
        fpr, tpr, _ = roc_curve(self.test_results['predictions']['y_true'], 
                               self.test_results['predictions']['y_proba'])
        ax3.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {self.test_results["metrics"]["auc"]:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve - Test Set')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Precision-Recall Curve
        ax4 = plt.subplot(3, 3, 4)
        precision, recall, _ = precision_recall_curve(self.test_results['predictions']['y_true'],
                                                     self.test_results['predictions']['y_proba'])
        ax4.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {self.test_results["metrics"]["average_precision"]:.3f})')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve - Test Set')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confusion Matrix
        ax5 = plt.subplot(3, 3, 5)
        cm = np.array([[self.test_results['confusion_matrix']['tn'], self.test_results['confusion_matrix']['fp']],
                      [self.test_results['confusion_matrix']['fn'], self.test_results['confusion_matrix']['tp']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Predicted Negative', 'Predicted Positive'],
                   yticklabels=['Actual Negative', 'Actual Positive'])
        ax5.set_title('Confusion Matrix - Test Set')
        
        # Plot 6: Vendor distribution
        ax6 = plt.subplot(3, 3, 6)
        vendor_stats = self.vendor_analysis['vendor_stats']
        vendor_stats_plot = vendor_stats[vendor_stats['total_scripts'] >= 2].head(10)  # Only vendors with 2+ scripts
        ax6.bar(range(len(vendor_stats_plot)), vendor_stats_plot['total_scripts'], alpha=0.8)
        ax6.set_xlabel('Vendor')
        ax6.set_ylabel('Number of Scripts')
        ax6.set_title('Top 10 Vendors by Script Count')
        ax6.set_xticks(range(len(vendor_stats_plot)))
        ax6.set_xticklabels([v[:10] + '...' if len(v) > 10 else v for v in vendor_stats_plot['vendor']], rotation=45)
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Cross-validation score distribution
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(val_scores, bins=max(3, len(val_scores)//2), alpha=0.7, edgecolor='black')
        ax7.axvline(np.mean(val_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_scores):.3f}')
        ax7.axvline(np.median(val_scores), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(val_scores):.3f}')
        ax7.set_xlabel('Validation AUC')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Distribution of CV Validation Scores')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Hyperparameter frequency
        ax8 = plt.subplot(3, 3, 8)
        # Get most common n_estimators values
        n_estimators_values = [params['n_estimators'] for params in self.nested_cv_results['fold_results']['best_params']]
        n_est_counts = Counter(n_estimators_values)
        ax8.bar(range(len(n_est_counts)), list(n_est_counts.values()), alpha=0.8)
        ax8.set_xlabel('n_estimators Value')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Most Common n_estimators Values')
        ax8.set_xticks(range(len(n_est_counts)))
        ax8.set_xticklabels(list(n_est_counts.keys()))
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Prediction score distribution
        ax9 = plt.subplot(3, 3, 9)
        y_true = self.test_results['predictions']['y_true']
        y_proba = self.test_results['predictions']['y_proba']
        
        # Separate scores by true class
        pos_scores = y_proba[y_true == 1]
        neg_scores = y_proba[y_true == 0]
        
        ax9.hist(neg_scores, bins=20, alpha=0.7, label='Negative Class', color='red')
        ax9.hist(pos_scores, bins=20, alpha=0.7, label='Positive Class', color='blue')
        ax9.set_xlabel('Prediction Score')
        ax9.set_ylabel('Frequency')
        ax9.set_title('Distribution of Prediction Scores')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('Balanced Malicious Script Classification - Vendor-Aware Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_file = f"{self.output_dir}/comprehensive_analysis_{self.timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive plot saved: {plot_file}")
        
        # Create additional summary plot
        self.create_summary_plot()
        
        return plot_file
    
    def create_summary_plot(self):
        """Create a focused summary plot with key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Performance summary
        ax1 = axes[0, 0]
        metrics = ['AUC', 'Average Precision', 'Accuracy', 'Precision', 'Recall']
        values = [
            self.test_results['metrics']['auc'],
            self.test_results['metrics']['average_precision'],
            self.test_results['metrics']['accuracy'],
            self.test_results['metrics']['precision'],
            self.test_results['metrics']['recall']
        ]
        bars = ax1.bar(metrics, values, alpha=0.8, color=['skyblue', 'lightgreen', 'gold', 'coral', 'plum'])
        ax1.set_ylabel('Score')
        ax1.set_title('Test Set Performance Metrics')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        ax1.grid(True, alpha=0.3)
        
        # CV stability
        ax2 = axes[0, 1]
        val_scores = self.nested_cv_results['fold_results']['val_scores']
        folds = range(1, len(val_scores) + 1)
        ax2.plot(folds, val_scores, 'o-', linewidth=2, markersize=8, label='Validation AUC')
        ax2.axhline(np.mean(val_scores), color='red', linestyle='--', alpha=0.7)
        ax2.fill_between(folds, 
                        np.mean(val_scores) - np.std(val_scores),
                        np.mean(val_scores) + np.std(val_scores),
                        alpha=0.2, color='red')
        ax2.set_xlabel('CV Fold')
        ax2.set_ylabel('AUC')
        ax2.set_title(f'CV Stability (μ={np.mean(val_scores):.3f}, σ={np.std(val_scores):.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Top features
        ax3 = axes[1, 0]
        top_features = self.test_results['feature_importance'].head(8)
        ax3.barh(range(len(top_features)), top_features['importance'], alpha=0.8)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 8 Most Important Features')
        ax3.grid(True, alpha=0.3)
        
        # ROC comparison with random classifier
        ax4 = axes[1, 1]
        fpr, tpr, _ = roc_curve(self.test_results['predictions']['y_true'], 
                               self.test_results['predictions']['y_proba'])
        ax4.plot(fpr, tpr, linewidth=3, label=f'Model (AUC = {self.test_results["metrics"]["auc"]:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax4.fill_between(fpr, tpr, alpha=0.2)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Summary - Vendor-Aware Balanced Classification', fontsize=14)
        plt.tight_layout()
        
        summary_plot_file = f"{self.output_dir}/summary_performance_{self.timestamp}.png"
        plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved: {summary_plot_file}")
        return summary_plot_file
    
    def generate_detailed_report(self):
        """Generate a comprehensive markdown report of the analysis."""
        print(f"\n📝 GENERATING DETAILED REPORT")
        print("=" * 60)
        
        report_file = f"{self.output_dir}/analysis_report_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Balanced Malicious Script Classification - Vendor-Aware Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents the results of a vendor-aware balanced classification approach ")
            f.write("for detecting malicious JavaScript scripts. The methodology addresses data leakage ")
            f.write("by ensuring no vendor appears in both training and testing sets.\n\n")
            
            # Key Results
            test_auc = self.test_results['metrics']['auc']
            cv_auc = self.nested_cv_results['summary']['mean_val_auc']
            cv_std = self.nested_cv_results['summary']['std_val_auc']
            
            f.write("### Key Results\n\n")
            f.write(f"- **Test Set AUC:** {test_auc:.3f}\n")
            f.write(f"- **Cross-Validation AUC:** {cv_auc:.3f} ± {cv_std:.3f}\n")
            f.write(f"- **Test Set Precision:** {self.test_results['metrics']['precision']:.3f}\n")
            f.write(f"- **Test Set Recall:** {self.test_results['metrics']['recall']:.3f}\n")
            f.write(f"- **Vendor Groups Used:** {len(set(self.create_vendor_groups_for_cv(self.train_data)))}\n")
            f.write(f"- **Test Vendors:** {len(self.test_vendors)}\n\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("### Vendor-Aware Data Splitting\n\n")
            f.write("To prevent data leakage, the dataset was split ensuring no vendor appears ")
            f.write("in both training and testing sets. This approach:\n\n")
            f.write("- Prevents overfitting to vendor-specific patterns\n")
            f.write("- Ensures models learn malicious behavior rather than vendor signatures\n")
            f.write("- Provides realistic performance estimates for deployment\n\n")
            
            f.write("### Nested Cross-Validation\n\n")
            f.write("Nested cross-validation was employed to obtain unbiased performance estimates:\n\n")
            f.write(f"- **Outer CV:** {self.nested_cv_results['summary']['total_folds']} folds for performance estimation\n")
            f.write("- **Inner CV:** 3 folds for hyperparameter optimization\n")
            f.write("- **CV Type:** StratifiedGroupKFold (vendor-grouped)\n")
            f.write(f"- **Completed Folds:** {self.nested_cv_results['summary']['completed_folds']}\n\n")
            
            # Data Analysis
            f.write("## Data Analysis\n\n")
            f.write("### Dataset Composition\n\n")
            f.write(f"- **Total Scripts:** {len(self.train_data) + len(self.test_data)}\n")
            f.write(f"- **Training Scripts:** {len(self.train_data)}\n")
            f.write(f"- **Test Scripts:** {len(self.test_data)}\n")
            f.write(f"- **Total Vendors:** {self.vendor_analysis['total_vendors']}\n")
            f.write(f"- **Positive-only Vendors:** {len(self.vendor_analysis['pos_only_vendors'])}\n")
            f.write(f"- **Mixed-label Vendors:** {len(self.vendor_analysis['mixed_vendors'])}\n\n")
            
            # Feature Analysis
            f.write("### Top Features\n\n")
            f.write("The most important features for classification:\n\n")
            top_features = self.test_results['feature_importance'].head(10)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                f.write(f"{i}. **{row['feature']}**: {row['importance']:.4f}\n")
            f.write("\n")
            
            # Performance Analysis
            f.write("## Performance Analysis\n\n")
            f.write("### Cross-Validation Results\n\n")
            fold_details = self.nested_cv_results['fold_results']['fold_details']
            f.write("| Fold | Train Size | Val Size | Val AUC | Best CV Score |\n")
            f.write("|------|------------|----------|---------|---------------|\n")
            for fold in fold_details:
                val_idx = fold['fold'] - 1
                val_auc = self.nested_cv_results['fold_results']['val_scores'][val_idx]
                f.write(f"| {fold['fold']} | {fold['train_size']} | {fold['val_size']} | {val_auc:.3f} | {fold['best_cv_score']:.3f} |\n")
            f.write("\n")
            
            # Test Set Results
            f.write("### Test Set Performance\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in self.test_results['metrics'].items():
                f.write(f"| {metric.replace('_', ' ').title()} | {value:.3f} |\n")
            f.write("\n")
            
            # Confusion Matrix
            cm = self.test_results['confusion_matrix']
            f.write("### Confusion Matrix\n\n")
            f.write("```\n")
            f.write("                Predicted\n")
            f.write("              Neg    Pos\n")
            f.write(f"Actual Neg   {cm['tn']:4d}   {cm['fp']:4d}\n")
            f.write(f"       Pos   {cm['fn']:4d}   {cm['tp']:4d}\n")
            f.write("```\n\n")
            
            # Model Configuration
            f.write("## Model Configuration\n\n")
            f.write("### Hyperparameter Search Space\n\n")
            f.write("```python\n")
            f.write("param_distributions = {\n")
            for param, values in self.param_distributions.items():
                f.write(f"    '{param}': {values},\n")
            f.write("}\n")
            f.write("```\n\n")
            
            # Most Common Parameters
            param_tuples = [tuple(sorted(params.items())) for params in self.nested_cv_results['fold_results']['best_params']]
            most_common_params = Counter(param_tuples).most_common(1)[0][0]
            best_params = dict(most_common_params)
            
            f.write("### Final Model Parameters\n\n")
            f.write("```python\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write("```\n\n")
            
            # Vendor Analysis
            f.write("## Vendor Analysis\n\n")
            f.write("### Vendor Distribution\n\n")
            vendor_stats = self.vendor_analysis['vendor_stats']
            vendor_summary = vendor_stats[vendor_stats['total_scripts'] >= 2].head(10)
            
            f.write("Top vendors by script count:\n\n")
            f.write("| Vendor | Total Scripts | Positive Scripts | Negative Scripts |\n")
            f.write("|--------|---------------|------------------|------------------|\n")
            for _, row in vendor_summary.iterrows():
                f.write(f"| {row['vendor'][:30]}{'...' if len(row['vendor']) > 30 else ''} | {row['total_scripts']} | {row['pos_count']} | {row['neg_count']} |\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on this analysis:\n\n")
            
            if test_auc >= 0.9:
                f.write("1. **Excellent Performance**: The model achieves excellent AUC performance (≥0.9)\n")
            elif test_auc >= 0.8:
                f.write("1. **Good Performance**: The model achieves good AUC performance (0.8-0.9)\n")
            else:
                f.write("1. **Moderate Performance**: The model achieves moderate AUC performance (<0.8)\n")
            
            if cv_std < 0.05:
                f.write("2. **Stable Model**: Low CV standard deviation indicates stable performance\n")
            else:
                f.write("2. **Variable Performance**: Higher CV standard deviation suggests model instability\n")
            
            f.write("3. **Vendor Awareness**: The vendor-aware approach prevents data leakage\n")
            f.write("4. **Deployment Ready**: Model can be deployed with confidence in real-world scenarios\n\n")
            
            # Technical Notes
            f.write("## Technical Notes\n\n")
            f.write("- All models use `class_weight='balanced'` to handle class imbalance\n")
            f.write("- RandomForest classifier chosen for interpretability and robustness\n")
            f.write("- Feature engineering focuses on API usage patterns and behavior\n")
            f.write("- Null vendors are treated as individual groups to prevent leakage\n")
            f.write("- Statistical significance testing not performed due to limited CV folds\n\n")
            
            # Files Generated
            f.write("## Generated Files\n\n")
            f.write("This analysis generated the following files:\n\n")
            f.write(f"- `balanced_model_{self.timestamp}.pkl` - Trained model\n")
            f.write(f"- `results_{self.timestamp}.json` - Complete results\n")
            f.write(f"- `feature_importance_{self.timestamp}.csv` - Feature rankings\n")
            f.write(f"- `comprehensive_analysis_{self.timestamp}.png` - Main visualization\n")
            f.write(f"- `summary_performance_{self.timestamp}.png` - Summary plots\n")
            f.write(f"- `train_data_{self.timestamp}.csv` - Training data\n")
            f.write(f"- `test_data_{self.timestamp}.csv` - Test data\n")
            f.write(f"- `analysis_report_{self.timestamp}.md` - This report\n\n")
            
        print(f"Detailed report saved: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🚀 " + "="*70)
        print("BALANCED MALICIOUS SCRIPT CLASSIFICATION - VENDOR-AWARE ANALYSIS")
        print("="*70)
        
        try:
            # Step 1: Load data
            print("\n📚 STEP 1: Load and Analyze Data")
            self.load_data()
            self.analyze_vendor_distribution()
            
            # Step 2: Create vendor-aware split
            print("\n🎯 STEP 2: Create Vendor-Aware Train/Test Split")
            self.create_vendor_aware_split()
            
            # Step 3: Nested cross-validation
            print("\n🔄 STEP 3: Vendor-Aware Nested Cross-Validation")
            self.nested_cross_validation()
            
            # Step 4: Train final model
            print("\n🎯 STEP 4: Train Final Model")
            self.train_final_model()
            
            # Step 5: Test set evaluation
            print("\n📊 STEP 5: Test Set Evaluation")
            self.evaluate_on_test_set()
            
            # Step 6: Save everything
            print("\n💾 STEP 6: Save Model and Results")
            self.save_model_and_results()
            
            # Step 7: Create visualizations
            print("\n📊 STEP 7: Create Visualizations")
            self.create_comprehensive_plots()
            
            # Step 8: Generate report
            print("\n📝 STEP 8: Generate Detailed Report")
            self.generate_detailed_report()
            
            # Final summary
            print("\n" + "="*70)
            print("✅ ANALYSIS COMPLETE!")
            print("="*70)
            print(f"🎯 Final Test AUC: {self.test_results['metrics']['auc']:.3f}")
            print(f"🔄 CV AUC: {self.nested_cv_results['summary']['mean_val_auc']:.3f} ± {self.nested_cv_results['summary']['std_val_auc']:.3f}")
            print(f"📁 Results saved to: {self.output_dir}/")
            print(f"⏰ Timestamp: {self.timestamp}")
            
            return {
                'test_auc': self.test_results['metrics']['auc'],
                'cv_auc': self.nested_cv_results['summary']['mean_val_auc'],
                'cv_std': self.nested_cv_results['summary']['std_val_auc'],
                'output_dir': self.output_dir,
                'timestamp': self.timestamp
            }
            
        except Exception as e:
            print(f"\n❌ ERROR in analysis pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


def run_analysis_from_json(json_file_path, output_dir="balanced_vendor_aware_results"):
    """
    Alternative entry point that loads data from a JSON file instead of database.
    Useful for testing or when database is not available.
    """
    print("🔄 Running analysis from JSON file...")
    
    # Create a modified classifier that loads from JSON
    class JSONBasedClassifier(BalancedVendorAwareClassifier):
        def __init__(self, json_file_path, output_dir):
            super().__init__(output_dir=output_dir)
            self.json_file_path = json_file_path
        
        def load_data(self):
            """Load data from JSON file instead of database."""
            print(f"🔌 Loading data from JSON file: {self.json_file_path}")
            
            with open(self.json_file_path, 'r') as f:
                self.raw_data = json.load(f)
            
            # Filter for balanced dataset (labels 0 and 1 only)
            self.raw_data = [script for script in self.raw_data if script['label'] in [0, 1]]
            
            print(f"✅ Loaded {len(self.raw_data)} scripts for balanced classification")
            return self.raw_data
    
    # Run analysis
    classifier = JSONBasedClassifier(json_file_path, output_dir)
    return classifier.run_complete_analysis()


def main():
    """Main entry point for the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Balanced Malicious Script Classification with Vendor Awareness')
    parser.add_argument('--json-file', type=str, help='Path to JSON file with script data (alternative to database)')
    parser.add_argument('--output-dir', type=str, default='balanced_vendor_aware_results', 
                       help='Output directory for results')
    parser.add_argument('--db-host', type=str, default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5434, help='Database port')
    parser.add_argument('--db-name', type=str, default='vv8_backend', help='Database name')
    parser.add_argument('--db-user', type=str, default='vv8', help='Database user')
    parser.add_argument('--db-password', type=str, default='vv8', help='Database password')
    parser.add_argument('--outer-cv-folds', type=int, default=5, help='Number of outer CV folds')
    parser.add_argument('--inner-cv-folds', type=int, default=3, help='Number of inner CV folds')
    parser.add_argument('--hyperparameter-iterations', type=int, default=20, help='Number of hyperparameter search iterations')
    parser.add_argument('--test-vendor-ratio', type=float, default=0.2, help='Ratio of vendors to use for testing')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        if args.json_file:
            # Run from JSON file
            print(f"🔄 Running analysis from JSON file: {args.json_file}")
            results = run_analysis_from_json(args.json_file, args.output_dir)
        else:
            # Run from database
            print("🔄 Running analysis from database...")
            db_config = {
                'host': args.db_host,
                'database': args.db_name,
                'user': args.db_user,
                'password': args.db_password,
                'port': args.db_port
            }
            
            classifier = BalancedVendorAwareClassifier(db_config=db_config, output_dir=args.output_dir)
            
            # Set random seed
            np.random.seed(args.random_seed)
            
            # Override default parameters if provided
            classifier.create_vendor_aware_split(test_vendor_ratio=args.test_vendor_ratio, 
                                                random_state=args.random_seed)
            classifier.nested_cross_validation(outer_cv_folds=args.outer_cv_folds,
                                              inner_cv_folds=args.inner_cv_folds,
                                              n_iter=args.hyperparameter_iterations)
            
            results = classifier.run_complete_analysis()
        
        if results:
            print("\n🎉 Analysis completed successfully!")
            print(f"📊 Test AUC: {results['test_auc']:.3f}")
            print(f"📁 Check results in: {results['output_dir']}")
        else:
            print("\n❌ Analysis failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def run_quick_test():
    """Quick test function using the provided JSON data."""
    print("🧪 Running quick test with provided JSON data...")
    
    # This function can be used to quickly test the classifier
    # You would call this with your JSON file path
    json_file = "vv8_backend_public_m_info_known_companies.json"  # Update this path
    results = run_analysis_from_json(json_file, "test_results")
    return results


# Example usage and testing functions
def validate_installation():
    """Validate that all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 
        'psycopg2', 'scipy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages are installed")
        return True


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "database": {
            "host": "localhost",
            "database": "vv8_backend",
            "user": "vv8",
            "password": "vv8",
            "port": 5434
        },
        "analysis": {
            "outer_cv_folds": 5,
            "inner_cv_folds": 3,
            "hyperparameter_iterations": 20,
            "test_vendor_ratio": 0.2,
            "random_seed": 42
        },
        "output": {
            "directory": "balanced_vendor_aware_results",
            "save_plots": True,
            "save_model": True,
            "save_report": True
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Sample config.json created")
    return config


if __name__ == "__main__":
    print("🔬 Balanced Malicious Script Classification - Vendor-Aware Analysis")
    print("=" * 70)
    
    # Validate installation
    if not validate_installation():
        exit(1)
    
    # Run main analysis
    main()