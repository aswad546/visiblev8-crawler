import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import calibration_curve
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedBehavioralBiometricDetector:
    """
    Enhanced system for detecting behavioral biometric scripts with comprehensive
    analysis, visualization saving, and misclassification tracking.
    """
    
    def __init__(self, db_config=None, output_dir="analysis_output"):
        """
        Initialize the enhanced detector with database configuration and output directory.
        
        Args:
            db_config (dict): Database configuration. If None, uses default config.
            output_dir (str): Directory to save plots and analysis files
        """
        # Default database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        
        # Create output directory for saving plots and results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Analysis timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.table_name = 'multicore_static_info_known_companies'
        self.raw_data = None
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
        # Model storage
        self.best_model = None
        self.model_type = None
        
        # Analysis results storage
        self.misclassification_analysis = None
        
    def save_plot(self, filename_suffix, tight_layout=True):
        """
        Helper method to save plots to files instead of displaying them.
        This ensures we can see results on remote servers.
        
        Args:
            filename_suffix (str): Descriptive suffix for the filename
            tight_layout (bool): Whether to apply tight layout before saving
        """
        if tight_layout:
            plt.tight_layout()
        
        filename = f"{self.output_dir}/analysis_{self.timestamp}_{filename_suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {filename}")
        plt.close()  # Close the figure to free memory
        
    def connect_to_database(self):
        """
        Establish connection to PostgreSQL database with enhanced error handling.
        """
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            print(f"❌ Error connecting to PostgreSQL database: {e}")
            print(f"Connection config: {self.db_config}")
            return None
        
    def load_and_explore_data(self):
        """
        Load data from PostgreSQL database with comprehensive data quality checks.
        """
        print("🔌 Connecting to PostgreSQL database and loading labeled dataset...")
        
        # Connect to database
        connection = self.connect_to_database()
        if connection is None:
            raise Exception("Failed to connect to database")
        
        try:
            # Query to fetch all data from the table
            query = f"SELECT * FROM {self.table_name}"
            
            # Use RealDictCursor to get results as dictionaries
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with careful JSON parsing
            self.raw_data = []
            parsing_errors = 0
            
            for row in rows:
                record = dict(row)
                
                # Parse JSON fields with error tracking
                json_fields = [
                    'aggregated_behavioral_apis', 'aggregated_fingerprinting_apis',
                    'fingerprinting_source_apis', 'behavioral_source_apis',
                    'behavioral_apis_access_count', 'fingerprinting_api_access_count',
                    'apis_going_to_sink', 'max_aggregated_apis'
                ]
                
                for field in json_fields:
                    if field in record and record[field] is not None:
                        if isinstance(record[field], str):
                            try:
                                record[field] = json.loads(record[field])
                            except json.JSONDecodeError:
                                parsing_errors += 1
                                record[field] = None  # Set to None if parsing fails
                        elif record[field] == '{}':
                            record[field] = {}
                        elif record[field] == '[]':
                            record[field] = []
                
                self.raw_data.append(record)
            
            cursor.close()
            
            if parsing_errors > 0:
                print(f"⚠️  Warning: {parsing_errors} JSON parsing errors encountered")
            
        except psycopg2.Error as e:
            print(f"❌ Error querying database: {e}")
            raise
        finally:
            connection.close()
        
        print(f"✅ Loaded {len(self.raw_data)} total scripts from database")
        
        # Comprehensive label analysis
        labels = [script['label'] for script in self.raw_data]
        unique_labels = sorted(set(labels))
        print(f"\n📊 Label distribution:")
        for label in unique_labels:
            count = labels.count(label)
            percentage = count / len(labels) * 100
            print(f"  Label {label}: {count} samples ({percentage:.1f}%)")
        
        # Handle different labeling schemes with data quality checks
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            print(f"\n✅ Binary classification detected")
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
        elif len(unique_labels) == 3:
            print(f"\n🔄 Three classes detected: {unique_labels}")
            print("Filtering to binary classification (keeping only labels 0 and 1)...")
            
            # Store original count and filter
            original_count = len(self.raw_data)
            self.raw_data = [script for script in self.raw_data if script['label'] in [0, 1]]
            filtered_count = len(self.raw_data)
            
            print(f"📉 Filtered from {original_count} to {filtered_count} samples")
            
            # Recalculate after filtering
            labels = [script['label'] for script in self.raw_data]
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
        else:
            raise ValueError(f"❌ Unexpected label scheme: {unique_labels}")
            
        # Data quality validation
        if filtered_count < 50:
            raise ValueError("❌ Insufficient samples for reliable analysis")
        
        print(f"✅ Final dataset: {positive_count} behavioral biometric, {negative_count} normal scripts")
        print(f"📊 Class balance ratio: {positive_count/negative_count:.3f}")
        
        # Graph construction success analysis (critical for understanding data quality)
        pos_graph_failures = sum(1 for script in self.raw_data 
                                if script['label'] == 1 and script.get('graph_construction_failure', True))
        neg_graph_failures = sum(1 for script in self.raw_data 
                                if script['label'] == 0 and script.get('graph_construction_failure', True))
        
        print(f"\n🔍 Graph construction analysis:")
        if positive_count > 0:
            pos_failure_rate = pos_graph_failures/positive_count*100
            print(f"  Positive class failures: {pos_graph_failures}/{positive_count} ({pos_failure_rate:.1f}%)")
        if negative_count > 0:
            neg_failure_rate = neg_graph_failures/negative_count*100
            print(f"  Negative class failures: {neg_graph_failures}/{negative_count} ({neg_failure_rate:.1f}%)")
        
        return self.raw_data
    
    def engineer_features(self):
        """
        Enhanced feature engineering with data quality tracking and feature validation.
        """
        print("\n🔧 Engineering features from static analysis data...")
        
        features_list = []
        feature_extraction_errors = 0
        
        for script in self.raw_data:
            try:
                features = {}
                
                # Core aggregation features (primary signals for behavioral biometrics)
                features['max_api_aggregation_score'] = script.get('max_api_aggregation_score', -1)
                features['behavioral_api_agg_count'] = script.get('behavioral_api_agg_count', -1)
                features['fp_api_agg_count'] = script.get('fp_api_agg_count', -1)
                
                # Volume indicators
                features['behavioral_source_api_count'] = script.get('behavioral_source_api_count', 0)
                features['fingerprinting_source_api_count'] = script.get('fingerprinting_source_api_count', 0)
                
                # Data flow indicators (critical for understanding intent)
                features['graph_construction_failure'] = int(script.get('graph_construction_failure', True))
                features['dataflow_to_sink'] = int(script.get('dataflow_to_sink', False))
                
                # Intensity analysis
                behavioral_access = script.get('behavioral_apis_access_count') or {}
                fp_access = script.get('fingerprinting_api_access_count') or {}
                
                features['total_behavioral_api_accesses'] = sum(behavioral_access.values()) if behavioral_access else 0
                features['total_fp_api_accesses'] = sum(fp_access.values()) if fp_access else 0
                features['unique_behavioral_apis'] = len(behavioral_access) if behavioral_access else 0
                features['unique_fp_apis'] = len(fp_access) if fp_access else 0
                
                # Sink analysis (where does the data go?)
                sink_data = script.get('apis_going_to_sink') or {}
                features['num_sink_types'] = len(sink_data) if sink_data else 0
                features['has_storage_sink'] = int(any('Storage' in str(sink) for sink in sink_data.keys()) if sink_data else False)
                features['has_network_sink'] = int(any(sink in ['XMLHttpRequest.send', 'Navigator.sendBeacon', 'fetch'] 
                                                       for sink in sink_data.keys()) if sink_data else False)
                
                # Behavioral diversity analysis (sophistication indicator)
                behavioral_sources = script.get('behavioral_source_apis') or []
                if behavioral_sources:
                    mouse_events = sum(1 for api in behavioral_sources if 'MouseEvent' in str(api))
                    keyboard_events = sum(1 for api in behavioral_sources if 'KeyboardEvent' in str(api))
                    touch_events = sum(1 for api in behavioral_sources if 'TouchEvent' in str(api) or 'Touch.' in str(api))
                    pointer_events = sum(1 for api in behavioral_sources if 'PointerEvent' in str(api))
                    
                    features['mouse_event_count'] = mouse_events
                    features['keyboard_event_count'] = keyboard_events
                    features['touch_event_count'] = touch_events
                    features['pointer_event_count'] = pointer_events
                    features['behavioral_event_diversity'] = sum([mouse_events > 0, keyboard_events > 0, 
                                                                touch_events > 0, pointer_events > 0])
                else:
                    features['mouse_event_count'] = 0
                    features['keyboard_event_count'] = 0
                    features['touch_event_count'] = 0
                    features['pointer_event_count'] = 0
                    features['behavioral_event_diversity'] = 0
                
                # Derived ratio features (normalized indicators)
                total_apis = features['behavioral_source_api_count'] + features['fingerprinting_source_api_count']
                if total_apis > 0:
                    features['behavioral_ratio'] = features['behavioral_source_api_count'] / total_apis
                    features['intensity_ratio'] = features['total_behavioral_api_accesses'] / total_apis
                else:
                    features['behavioral_ratio'] = 0
                    features['intensity_ratio'] = 0
                
                # Store script ID for misclassification tracking
                features['script_id'] = script.get('script_id')
                features['script_url'] = script.get('script_url', 'Unknown')
                features['label'] = script['label']
                
                features_list.append(features)
                
            except Exception as e:
                feature_extraction_errors += 1
                print(f"⚠️  Feature extraction error for script {script.get('script_id', 'unknown')}: {e}")
        
        if feature_extraction_errors > 0:
            print(f"⚠️  {feature_extraction_errors} feature extraction errors encountered")
        
        # Create DataFrame and validate
        self.features_df = pd.DataFrame(features_list)
        
        # Separate metadata from features
        metadata_cols = ['script_id', 'script_url', 'label']
        feature_cols = [col for col in self.features_df.columns if col not in metadata_cols]
        
        self.y = self.features_df['label'].values
        self.X = self.features_df[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✅ Engineered {len(self.feature_names)} features from {len(features_list)} scripts")
        print(f"📊 Feature names: {self.feature_names}")
        
        # Data quality check for features
        nan_counts = np.isnan(self.X).sum(axis=0)
        if np.any(nan_counts > 0):
            print(f"⚠️  Warning: Found NaN values in features")
            for i, count in enumerate(nan_counts):
                if count > 0:
                    print(f"    {self.feature_names[i]}: {count} NaN values")
        
        return self.features_df
    
    def analyze_feature_correlations(self):
        """
        Analyze feature correlations to identify redundant or highly correlated features.
        This is critical for understanding which features provide unique information.
        """
        print("\n🔍 Analyzing feature correlations...")
        
        # Calculate correlation matrix
        feature_data = self.features_df[self.feature_names]
        correlation_matrix = feature_data.corr()
        
        # Find highly correlated pairs (>0.8)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_pairs.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.8):")
            for pair in high_corr_pairs:
                print(f"    {pair['feature_1']} ↔ {pair['feature_2']}: {pair['correlation']:.3f}")
        else:
            print("✅ No highly correlated features found")
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix\n(Upper triangle masked for clarity)', fontsize=14, pad=20)
        self.save_plot('feature_correlations')
        
        return correlation_matrix, high_corr_pairs
    
    def analyze_feature_distributions(self):
        """
        Enhanced feature distribution analysis with statistical testing.
        """
        print("\n📊 Analyzing feature distributions between classes...")
        
        # Select key features for detailed analysis
        key_features = [
            'max_api_aggregation_score', 'behavioral_api_agg_count', 'behavioral_source_api_count',
            'total_behavioral_api_accesses', 'behavioral_event_diversity', 'dataflow_to_sink',
            'has_storage_sink', 'has_network_sink', 'behavioral_ratio'
        ]
        
        # Create comprehensive distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features[:9]):
            pos_data = self.features_df[self.features_df['label'] == 1][feature]
            neg_data = self.features_df[self.features_df['label'] == 0][feature]
            
            # Create histogram with enhanced styling
            axes[i].hist([neg_data, pos_data], bins=20, alpha=0.7, 
                        label=['Normal Scripts', 'Behavioral Biometric'], 
                        color=['lightcoral', 'skyblue'])
            axes[i].set_title(f'{feature}', fontsize=10, fontweight='bold')
            axes[i].legend()
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
            axes[i].grid(True, alpha=0.3)
            
            # Add statistical annotation
            pos_mean = pos_data.mean()
            neg_mean = neg_data.mean()
            axes[i].text(0.05, 0.95, f'Pos mean: {pos_mean:.2f}\nNeg mean: {neg_mean:.2f}', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Feature Distributions by Class\n(Key Discriminative Features)', fontsize=16, y=0.98)
        self.save_plot('feature_distributions')
        
        # Statistical summary with effect sizes
        print("\n📈 Statistical summary by class:")
        summary_stats = self.features_df.groupby('label')[key_features].agg(['mean', 'std', 'median'])
        print(summary_stats.round(3))
        
        # Calculate effect sizes (Cohen's d) to measure practical significance
        effect_sizes = {}
        for feature in key_features:
            pos_data = self.features_df[self.features_df['label'] == 1][feature]
            neg_data = self.features_df[self.features_df['label'] == 0][feature]
            
            pooled_std = np.sqrt(((len(pos_data) - 1) * pos_data.var() + (len(neg_data) - 1) * neg_data.var()) / 
                                (len(pos_data) + len(neg_data) - 2))
            cohens_d = (pos_data.mean() - neg_data.mean()) / pooled_std
            effect_sizes[feature] = cohens_d
        
        print("\n📏 Effect sizes (Cohen's d - larger absolute values indicate stronger discrimination):")
        for feature, effect_size in sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True):
            interpretation = "Large" if abs(effect_size) > 0.8 else "Medium" if abs(effect_size) > 0.5 else "Small"
            print(f"  {feature}: {effect_size:.3f} ({interpretation})")
        
        return summary_stats, effect_sizes
    
    def perform_enhanced_cross_validation(self, model, model_name="Model", k=5):
        """
        Enhanced k-fold cross-validation with detailed tracking and calibration analysis.
        """
        print(f"\n🔄 Performing {k}-Fold Cross-Validation for {model_name}...")
        
        # Use stratified k-fold to maintain class balance
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        
        # Storage for comprehensive results
        fold_results = []
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        # Perform k-fold validation with detailed tracking
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"  🔄 Processing fold {fold + 1}/{k}...")
            
            # Split data for this fold
            X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
            y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
            
            # Train model for this fold
            fold_model = model.__class__(**model.get_params())
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            train_pred = fold_model.predict(X_train_fold)
            val_pred = fold_model.predict(X_val_fold)
            val_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            
            # Calculate comprehensive metrics
            train_acc = fold_model.score(X_train_fold, y_train_fold)
            val_acc = fold_model.score(X_val_fold, y_val_fold)
            val_auc = roc_auc_score(y_val_fold, val_proba)
            
            # Store results
            fold_results.append({
                'fold': fold + 1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'overfitting_gap': train_acc - val_acc,
                'train_samples': len(y_train_fold),
                'val_samples': len(y_val_fold),
                'val_pos_samples': sum(y_val_fold),
                'val_neg_samples': len(y_val_fold) - sum(y_val_fold)
            })
            
            # Collect predictions for overall analysis
            all_predictions.extend(val_pred)
            all_probabilities.extend(val_proba)
            all_true_labels.extend(y_val_fold)
        
        # Calculate summary statistics
        results = {
            'train_acc_mean': np.mean([r['train_acc'] for r in fold_results]),
            'train_acc_std': np.std([r['train_acc'] for r in fold_results]),
            'val_acc_mean': np.mean([r['val_acc'] for r in fold_results]),
            'val_acc_std': np.std([r['val_acc'] for r in fold_results]),
            'val_auc_mean': np.mean([r['val_auc'] for r in fold_results]),
            'val_auc_std': np.std([r['val_auc'] for r in fold_results]),
            'overfitting_gap': np.mean([r['overfitting_gap'] for r in fold_results]),
            'fold_results': fold_results,
            'all_predictions': np.array(all_predictions),
            'all_probabilities': np.array(all_probabilities),
            'all_true_labels': np.array(all_true_labels)
        }
        
        # Print comprehensive results with interpretation
        print(f"\n📊 {model_name} Cross-Validation Results:")
        print("-" * 60)
        print(f"Training Accuracy:   {results['train_acc_mean']:.3f} ± {results['train_acc_std']:.3f}")
        print(f"Validation Accuracy: {results['val_acc_mean']:.3f} ± {results['val_acc_std']:.3f}")
        print(f"Validation AUC:      {results['val_auc_mean']:.3f} ± {results['val_auc_std']:.3f}")
        print(f"Overfitting Gap:     {results['overfitting_gap']:.3f}")
        
        # Overfitting diagnosis with clear interpretation
        if results['overfitting_gap'] > 0.1:
            print("🚨 WARNING: Significant overfitting detected! (>10% gap)")
            print("   Consider: reducing model complexity, increasing regularization, or getting more data")
        elif results['overfitting_gap'] > 0.05:
            print("⚠️  CAUTION: Mild overfitting detected (5-10% gap)")
            print("   Consider: slight regularization or model simplification")
        else:
            print("✅ Good: No significant overfitting detected")
        
        # Stability diagnosis
        if results['val_auc_std'] > 0.05:
            print("⚠️  WARNING: High performance variability across folds (>5% AUC std)")
            print("   This suggests the model is sensitive to data splits")
        else:
            print("✅ Good: Stable performance across folds")
        
        # Create comprehensive visualization
        self.plot_enhanced_cv_results(fold_results, model_name)
        
        return results
    
    def plot_enhanced_cv_results(self, fold_results, model_name):
        """
        Create enhanced cross-validation visualizations with detailed insights.
        """
        df_results = pd.DataFrame(fold_results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Training vs Validation Accuracy by Fold
        x = df_results['fold']
        axes[0, 0].plot(x, df_results['train_acc'], 'o-', label='Training', 
                       linewidth=2, markersize=8, color='blue')
        axes[0, 0].plot(x, df_results['val_acc'], 's-', label='Validation', 
                       linewidth=2, markersize=8, color='red')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title(f'{model_name}: Training vs Validation Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Validation AUC by Fold
        axes[0, 1].bar(x, df_results['val_auc'], alpha=0.7, color='green')
        axes[0, 1].axhline(y=df_results['val_auc'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df_results["val_auc"].mean():.3f}')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Validation AUC')
        axes[0, 1].set_title(f'{model_name}: Validation AUC by Fold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Overfitting Gap by Fold
        gap_colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' 
                     for gap in df_results['overfitting_gap']]
        axes[0, 2].bar(x, df_results['overfitting_gap'], alpha=0.7, color=gap_colors)
        axes[0, 2].axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, 
                          label='Mild overfitting (5%)')
        axes[0, 2].axhline(y=0.10, color='red', linestyle='--', alpha=0.7, 
                          label='Severe overfitting (10%)')
        axes[0, 2].set_xlabel('Fold')
        axes[0, 2].set_ylabel('Training - Validation Accuracy')
        axes[0, 2].set_title(f'{model_name}: Overfitting Gap by Fold')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Sample distribution by fold
        axes[1, 0].bar(x - 0.2, df_results['val_pos_samples'], width=0.4, 
                      label='Positive samples', alpha=0.7, color='skyblue')
        axes[1, 0].bar(x + 0.2, df_results['val_neg_samples'], width=0.4, 
                      label='Negative samples', alpha=0.7, color='lightcoral')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Sample Count')
        axes[1, 0].set_title('Sample Distribution by Fold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Performance stability analysis
        metrics = ['train_acc', 'val_acc', 'val_auc']
        means = [df_results[metric].mean() for metric in metrics]
        stds = [df_results[metric].std() for metric in metrics]
        
        x_pos = np.arange(len(metrics))
        axes[1, 1].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10, 
                      color=['blue', 'red', 'green'])
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['Train Acc', 'Val Acc', 'Val AUC'])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Detailed fold results table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table_data = df_results[['fold', 'val_acc', 'val_auc', 'overfitting_gap']].round(3)
        table = axes[1, 2].table(cellText=table_data.values, colLabels=table_data.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1, 2].set_title('Detailed Results by Fold')
        
        plt.suptitle(f'{model_name}: Comprehensive Cross-Validation Analysis', 
                    fontsize=16, y=0.98)
        self.save_plot(f'cv_analysis_{model_name.lower().replace(" ", "_")}')
    
    def build_enhanced_models(self):
        """
        Build enhanced models with hyperparameter tuning and comprehensive evaluation.
        """
        print("\n🤖 Building Enhanced Models with Hyperparameter Optimization...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Store the test set indices for misclassification analysis
        self.test_indices = X_test
        self.y_test = y_test
        
        print(f"📊 Data split: {len(X_train)} training, {len(X_test)} test samples")
        
        # Decision Tree with basic configuration
        print("\n🌳 Building Decision Tree...")
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        
        # Enhanced cross-validation for Decision Tree
        dt_results = self.perform_enhanced_cross_validation(dt_model, "Decision Tree")
        
        # Train final Decision Tree
        dt_model.fit(X_train, y_train)
        dt_train_score = dt_model.score(X_train, y_train)
        dt_test_score = dt_model.score(X_test, y_test)
        
        print(f"🌳 Decision Tree Final Results:")
        print(f"  Training accuracy: {dt_train_score:.3f}")
        print(f"  Test accuracy: {dt_test_score:.3f}")
        print(f"  Overfitting gap: {dt_train_score - dt_test_score:.3f}")
        
        # Random Forest with hyperparameter tuning
        print("\n🌲 Building Random Forest with Hyperparameter Tuning...")
        best_rf_params = self.tune_random_forest_hyperparameters()
        
        # Build optimized Random Forest
        rf_model = RandomForestClassifier(
            **best_rf_params,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Enhanced cross-validation for Random Forest
        rf_results = self.perform_enhanced_cross_validation(rf_model, "Random Forest")
        
        # Train final Random Forest
        rf_model.fit(X_train, y_train)
        rf_train_score = rf_model.score(X_train, y_train)
        rf_test_score = rf_model.score(X_test, y_test)
        
        print(f"🌲 Random Forest Final Results:")
        print(f"  Training accuracy: {rf_train_score:.3f}")
        print(f"  Test accuracy: {rf_test_score:.3f}")
        print(f"  Overfitting gap: {rf_train_score - rf_test_score:.3f}")
        
        # Model comparison and selection
        print("\n🏆 Model Comparison and Selection:")
        print(f"{'Metric':<25} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 55)
        print(f"{'CV AUC':<25} {dt_results['val_auc_mean']:<15.3f} {rf_results['val_auc_mean']:<15.3f}")
        print(f"{'Test Accuracy':<25} {dt_test_score:<15.3f} {rf_test_score:<15.3f}")
        print(f"{'Overfitting Gap':<25} {dt_train_score - dt_test_score:<15.3f} {rf_train_score - rf_test_score:<15.3f}")
        print(f"{'CV Stability (std)':<25} {dt_results['val_auc_std']:<15.3f} {rf_results['val_auc_std']:<15.3f}")
        
        # Select best model based on comprehensive criteria
        if (rf_results['val_auc_mean'] > dt_results['val_auc_mean'] and 
            rf_results['val_auc_std'] <= dt_results['val_auc_std']):
            self.best_model = rf_model
            self.model_type = "Random Forest"
            best_results = rf_results
            print(f"\n🎯 SELECTED MODEL: Random Forest")
            print(f"   Reason: Better performance with similar or better stability")
        elif dt_results['val_auc_mean'] > rf_results['val_auc_mean'] and dt_train_score - dt_test_score < 0.1:
            self.best_model = dt_model
            self.model_type = "Decision Tree"
            best_results = dt_results
            print(f"\n🎯 SELECTED MODEL: Decision Tree")
            print(f"   Reason: Better performance with acceptable overfitting")
        else:
            # Default to Random Forest for production reliability
            self.best_model = rf_model
            self.model_type = "Random Forest"
            best_results = rf_results
            print(f"\n🎯 SELECTED MODEL: Random Forest (default)")
            print(f"   Reason: Generally more robust for production deployment")
        
        # Feature importance analysis
        self.analyze_feature_importance(dt_model, rf_model)
        
        # Save the best model
        model_filename = f"{self.output_dir}/best_model_{self.timestamp}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'cv_results': best_results,
                'timestamp': self.timestamp
            }, f)
        print(f"💾 Best model saved: {model_filename}")
        
        return dt_model, rf_model, best_results
    
    def tune_random_forest_hyperparameters(self):
        """
        Enhanced hyperparameter tuning with overfitting prevention focus.
        """
        print("\n🔧 Tuning Random Forest hyperparameters...")
        
        # Comprehensive parameter grid focusing on complexity control
        param_combinations = [
            # Conservative (prevent overfitting)
            {'n_estimators': 50, 'max_depth': 5, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 8, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
            
            # Moderate complexity
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
            {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
            {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 'log2'},
            
            # Higher complexity (risk of overfitting but might capture complex patterns)
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
            {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 3, 'max_features': 'log2'},
        ]
        
        best_score = 0
        best_params = None
        best_overfitting_gap = float('inf')
        tuning_results = []
        
        print("🔍 Testing parameter combinations...")
        for i, params in enumerate(param_combinations):
            print(f"  Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Create model with current parameters
            model = RandomForestClassifier(
                **params,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Perform 3-fold CV for faster tuning
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            train_scores = []
            val_scores = []
            
            for train_idx, val_idx in skf.split(self.X, self.y):
                X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
                y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                
                train_score = roc_auc_score(y_train_fold, model.predict_proba(X_train_fold)[:, 1])
                val_score = roc_auc_score(y_val_fold, model.predict_proba(X_val_fold)[:, 1])
                
                train_scores.append(train_score)
                val_scores.append(val_score)
            
            avg_train_score = np.mean(train_scores)
            avg_val_score = np.mean(val_scores)
            overfitting_gap = avg_train_score - avg_val_score
            val_std = np.std(val_scores)
            
            tuning_results.append({
                'params': params,
                'train_auc': avg_train_score,
                'val_auc': avg_val_score,
                'overfitting_gap': overfitting_gap,
                'val_std': val_std
            })
            
            print(f"    Train AUC: {avg_train_score:.3f}, Val AUC: {avg_val_score:.3f}, Gap: {overfitting_gap:.3f}")
            
            # Multi-criteria selection: prioritize validation performance while penalizing overfitting
            selection_score = avg_val_score - 0.5 * max(0, overfitting_gap - 0.05)  # Penalty for gaps > 5%
            
            if selection_score > best_score and overfitting_gap < 0.2:  # Maximum 20% gap tolerance
                best_score = selection_score
                best_params = params
                best_overfitting_gap = overfitting_gap
        
        if best_params is None:
            # Fallback: select least overfitted model
            best_result = min(tuning_results, key=lambda x: x['overfitting_gap'])
            best_params = best_result['params']
            print(f"⚠️  Warning: All models show significant overfitting. Selected least overfitted.")
        
        print(f"\n✅ Best parameters selected: {best_params}")
        print(f"📊 Best validation AUC: {best_score:.3f}")
        print(f"📉 Overfitting gap: {best_overfitting_gap:.3f}")
        
        return best_params
    
    def analyze_feature_importance(self, dt_model, rf_model):
        """
        Comprehensive feature importance analysis with visualizations.
        """
        print("\n🔍 Analyzing Feature Importance...")
        
        # Get feature importance from both models
        dt_importance = pd.DataFrame({
            'feature': self.feature_names,
            'dt_importance': dt_model.feature_importances_
        }).sort_values('dt_importance', ascending=False)
        
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': rf_model.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        # Merge for comparison
        importance_comparison = dt_importance.merge(rf_importance, on='feature')
        importance_comparison = importance_comparison.sort_values('rf_importance', ascending=False)
        
        print("\n📊 Top 10 Feature Importance Comparison:")
        print(importance_comparison.head(10)[['feature', 'dt_importance', 'rf_importance']].round(4))
        
        # Create comprehensive feature importance visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Random Forest feature importance
        top_rf_features = rf_importance.head(10)
        axes[0, 0].barh(range(len(top_rf_features)), top_rf_features['rf_importance'], color='forestgreen')
        axes[0, 0].set_yticks(range(len(top_rf_features)))
        axes[0, 0].set_yticklabels(top_rf_features['feature'], fontsize=9)
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_title('Random Forest Feature Importance (Top 10)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].invert_yaxis()
        
        # Plot 2: Decision Tree feature importance
        top_dt_features = dt_importance.head(10)
        axes[0, 1].barh(range(len(top_dt_features)), top_dt_features['dt_importance'], color='orange')
        axes[0, 1].set_yticks(range(len(top_dt_features)))
        axes[0, 1].set_yticklabels(top_dt_features['feature'], fontsize=9)
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Decision Tree Feature Importance (Top 10)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()
        
        # Plot 3: Feature importance correlation between models
        axes[1, 0].scatter(importance_comparison['dt_importance'], 
                          importance_comparison['rf_importance'], alpha=0.7)
        axes[1, 0].set_xlabel('Decision Tree Importance')
        axes[1, 0].set_ylabel('Random Forest Importance')
        axes[1, 0].set_title('Feature Importance Correlation Between Models')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Annotate top features
        top_features = importance_comparison.head(8)
        for idx, row in top_features.iterrows():
            axes[1, 0].annotate(row['feature'], 
                               (row['dt_importance'], row['rf_importance']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        # Plot 4: Feature importance distribution
        all_importances = np.concatenate([dt_model.feature_importances_, rf_model.feature_importances_])
        axes[1, 1].hist(all_importances, bins=20, alpha=0.7, color='skyblue')
        axes[1, 1].axvline(np.mean(all_importances), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_importances):.3f}')
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Feature Importance Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Comprehensive Feature Importance Analysis', fontsize=16, y=0.98)
        self.save_plot('feature_importance_analysis')
        
        # Identify consistently important features
        consistent_features = importance_comparison[
            (importance_comparison['dt_importance'] > 0.01) & 
            (importance_comparison['rf_importance'] > 0.01)
        ].sort_values('rf_importance', ascending=False)
        
        print(f"\n🎯 Consistently Important Features (both models > 1% importance):")
        for idx, row in consistent_features.head(8).iterrows():
            print(f"  {row['feature']}: DT={row['dt_importance']:.3f}, RF={row['rf_importance']:.3f}")
        
        return importance_comparison
    
    def perform_comprehensive_misclassification_analysis(self):
        """
        Comprehensive analysis of misclassified samples with database integration.
        This is critical for understanding model limitations and improving detection.
        """
        print("\n🔍 Performing Comprehensive Misclassification Analysis...")
        
        if self.best_model is None:
            print("❌ No trained model available. Please run build_enhanced_models() first.")
            return None
        
        # Create final holdout test set for unbiased misclassification analysis
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Get the indices for the test set to match with metadata
        _, test_indices = train_test_split(
            range(len(self.features_df)), test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Get corresponding metadata for test samples using the correct indices
        test_metadata = self.features_df.iloc[test_indices].copy()
        
        # Make predictions on test set
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Create comprehensive results DataFrame
        misclassification_results = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_confidence': np.max(y_pred_proba, axis=1),
            'positive_probability': y_pred_proba[:, 1],
            'correct_prediction': y_test == y_pred
        })
        
        # Add feature values for analysis
        feature_df = pd.DataFrame(X_test, columns=self.feature_names)
        misclassification_results = pd.concat([misclassification_results.reset_index(drop=True), 
                                            feature_df.reset_index(drop=True)], axis=1)
        
        # Add metadata - now with correct indexing
        metadata_cols = ['script_id', 'script_url']
        for col in metadata_cols:
            if col in test_metadata.columns:
                misclassification_results[col] = test_metadata[col].reset_index(drop=True)
        
        # Rest of the method remains the same...
        # Analyze different types of errors
        false_positives = misclassification_results[
            (misclassification_results['true_label'] == 0) & 
            (misclassification_results['predicted_label'] == 1)
        ]
        
        false_negatives = misclassification_results[
            (misclassification_results['true_label'] == 1) & 
            (misclassification_results['predicted_label'] == 0)
        ]
        
        true_positives = misclassification_results[
            (misclassification_results['true_label'] == 1) & 
            (misclassification_results['predicted_label'] == 1)
        ]
        
        true_negatives = misclassification_results[
            (misclassification_results['true_label'] == 0) & 
            (misclassification_results['predicted_label'] == 0)
        ]
        
        print(f"\n📊 Misclassification Analysis Results:")
        print(f"  Total samples: {len(misclassification_results)}")
        print(f"  Correct predictions: {sum(misclassification_results['correct_prediction'])} ({sum(misclassification_results['correct_prediction'])/len(misclassification_results)*100:.1f}%)")
        print(f"  False Positives: {len(false_positives)} (Normal scripts classified as behavioral biometric)")
        print(f"  False Negatives: {len(false_negatives)} (Behavioral biometric scripts missed)")
        print(f"  True Positives: {len(true_positives)}")
        print(f"  True Negatives: {len(true_negatives)}")
        
        # Detailed analysis of false positives
        if len(false_positives) > 0:
            print(f"\n🚨 False Positive Analysis:")
            print(f"   These normal scripts were incorrectly flagged as behavioral biometric")
            
            # Key features that led to false positives
            key_features = ['max_api_aggregation_score', 'behavioral_api_agg_count', 'fingerprinting_source_api_count',
                        'unique_fp_apis', 'dataflow_to_sink', 'prediction_confidence']
            
            print(f"   Average feature values for false positives:")
            for feature in key_features:
                if feature in false_positives.columns:
                    avg_val = false_positives[feature].mean()
                    print(f"     {feature}: {avg_val:.3f}")
            
            # Show most confident false positives (most concerning)
            high_confidence_fp = false_positives.nlargest(3, 'prediction_confidence')
            print(f"\n   Most confident false positives (highest risk):")
            for idx, row in high_confidence_fp.iterrows():
                script_info = f"Script {row.get('script_id', 'unknown')}"
                if 'script_url' in row and pd.notna(row['script_url']):
                    script_info += f" ({row['script_url'][:50]}...)"
                print(f"     {script_info}: {row['prediction_confidence']:.3f} confidence")
        
        # Detailed analysis of false negatives
        if len(false_negatives) > 0:
            print(f"\n🎯 False Negative Analysis:")
            print(f"   These behavioral biometric scripts were missed by the detector")
            
            print(f"   Average feature values for false negatives:")
            for feature in key_features:
                if feature in false_negatives.columns:
                    avg_val = false_negatives[feature].mean()
                    print(f"     {feature}: {avg_val:.3f}")
            
            # Show least confident false negatives (scripts that barely missed detection)
            low_confidence_fn = false_negatives.nlargest(3, 'positive_probability')
            print(f"\n   Closest misses (might need threshold adjustment):")
            for idx, row in low_confidence_fn.iterrows():
                script_info = f"Script {row.get('script_id', 'unknown')}"
                if 'script_url' in row and pd.notna(row['script_url']):
                    script_info += f" ({row['script_url'][:50]}...)"
                print(f"     {script_info}: {row['positive_probability']:.3f} probability")
        
        # Feature importance for misclassifications
        if len(false_positives) > 0 or len(false_negatives) > 0:
            self.analyze_misclassification_patterns(false_positives, false_negatives, true_positives, true_negatives)
        
        # Save comprehensive results to database
        self.save_misclassification_results_to_database(misclassification_results, false_positives, false_negatives)
        
        # Generate SQL queries for manual investigation
        self.generate_investigation_queries(false_positives, false_negatives)
        
        # Store results for further analysis
        self.misclassification_analysis = {
            'all_results': misclassification_results,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'true_negatives': true_negatives
        }
        
        return self.misclassification_analysis
    
    
    def analyze_misclassification_patterns(self, false_positives, false_negatives, true_positives, true_negatives):
        """
        Deep dive into patterns that cause misclassifications.
        """
        print("\n🔬 Analyzing Misclassification Patterns...")
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Key features for pattern analysis
        key_features = ['max_api_aggregation_score', 'behavioral_api_agg_count', 'fingerprinting_source_api_count']
        
        for i, feature in enumerate(key_features):
            if feature in false_positives.columns:
                # Box plot comparing all groups
                data_groups = [
                    true_negatives[feature].values if len(true_negatives) > 0 else [],
                    false_positives[feature].values if len(false_positives) > 0 else [],
                    false_negatives[feature].values if len(false_negatives) > 0 else [],
                    true_positives[feature].values if len(true_positives) > 0 else []
                ]
                
                labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
                colors = ['lightgreen', 'red', 'orange', 'darkgreen']
                
                bp = axes[0, i].boxplot(data_groups, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[0, i].set_title(f'{feature}\nPattern Analysis')
                axes[0, i].set_ylabel('Feature Value')
                axes[0, i].grid(True, alpha=0.3)
        
        # Distribution comparison for prediction confidence
        if len(false_positives) > 0:
            axes[1, 0].hist(false_positives['prediction_confidence'], bins=10, alpha=0.7, 
                           color='red', label='False Positives')
        if len(true_positives) > 0:
            axes[1, 0].hist(true_positives['prediction_confidence'], bins=10, alpha=0.7, 
                           color='green', label='True Positives')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confidence Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion matrix visualization
        confusion_mat = confusion_matrix(
            [row['true_label'] for _, row in pd.concat([true_negatives, false_positives, false_negatives, true_positives]).iterrows()],
            [row['predicted_label'] for _, row in pd.concat([true_negatives, false_positives, false_negatives, true_positives]).iterrows()]
        )
        
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Behavioral Biometric'],
                   yticklabels=['Normal', 'Behavioral Biometric'], ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_ylabel('True Label')
        
        # Feature correlation with misclassification
        if len(false_positives) > 0 and len(false_negatives) > 0:
            # Calculate which features are most associated with errors
            all_errors = pd.concat([false_positives, false_negatives])
            all_errors['error_type'] = ['False Positive'] * len(false_positives) + ['False Negative'] * len(false_negatives)
            
            # Show feature differences between error types
            fp_means = false_positives[self.feature_names].mean()
            fn_means = false_negatives[self.feature_names].mean()
            feature_diff = (fp_means - fn_means).abs().sort_values(ascending=False)
            
            top_discriminating = feature_diff.head(10)
            axes[1, 2].barh(range(len(top_discriminating)), top_discriminating.values)
            axes[1, 2].set_yticks(range(len(top_discriminating)))
            axes[1, 2].set_yticklabels(top_discriminating.index, fontsize=8)
            axes[1, 2].set_xlabel('Mean Absolute Difference')
            axes[1, 2].set_title('Features Distinguishing\nError Types')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].invert_yaxis()
        
        plt.suptitle('Misclassification Pattern Analysis', fontsize=16, y=0.98)
        self.save_plot('misclassification_patterns')
    
    def save_misclassification_results_to_database(self, all_results, false_positives, false_negatives):
        """
        Save misclassification analysis results to database for further investigation.
        """
        print("\n💾 Saving misclassification results to database...")
        
        connection = self.connect_to_database()
        if connection is None:
            print("❌ Cannot save to database - connection failed")
            return
        
        try:
            cursor = connection.cursor()
            
            # Create misclassification analysis table
            table_name = f"misclassification_analysis_{self.timestamp}"
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                analysis_id SERIAL PRIMARY KEY,
                script_id INTEGER,
                script_url TEXT,
                true_label INTEGER,
                predicted_label INTEGER,
                prediction_confidence FLOAT,
                positive_probability FLOAT,
                error_type TEXT,
                model_type TEXT,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                max_api_aggregation_score FLOAT,
                behavioral_api_agg_count FLOAT,
                fingerprinting_source_api_count FLOAT,
                unique_fp_apis FLOAT,
                dataflow_to_sink INTEGER
            );
            """
            
            cursor.execute(create_table_sql)
            
            # Insert false positives
            for _, row in false_positives.iterrows():
                insert_sql = f"""
                INSERT INTO {table_name} (
                    script_id, script_url, true_label, predicted_label, 
                    prediction_confidence, positive_probability, error_type, model_type,
                    max_api_aggregation_score, behavioral_api_agg_count, 
                    fingerprinting_source_api_count, unique_fp_apis, dataflow_to_sink
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_sql, (
                    row.get('script_id'), row.get('script_url'), 
                    int(row['true_label']), int(row['predicted_label']),
                    float(row['prediction_confidence']), float(row['positive_probability']),
                    'False Positive', self.model_type,
                    float(row.get('max_api_aggregation_score', 0)),
                    float(row.get('behavioral_api_agg_count', 0)),
                    float(row.get('fingerprinting_source_api_count', 0)),
                    float(row.get('unique_fp_apis', 0)),
                    int(row.get('dataflow_to_sink', 0))
                ))
            
            # Insert false negatives
            for _, row in false_negatives.iterrows():
                cursor.execute(insert_sql, (
                    row.get('script_id'), row.get('script_url'),
                    int(row['true_label']), int(row['predicted_label']),
                    float(row['prediction_confidence']), float(row['positive_probability']),
                    'False Negative', self.model_type,
                    float(row.get('max_api_aggregation_score', 0)),
                    float(row.get('behavioral_api_agg_count', 0)),
                    float(row.get('fingerprinting_source_api_count', 0)),
                    float(row.get('unique_fp_apis', 0)),
                    int(row.get('dataflow_to_sink', 0))
                ))
            
            connection.commit()
            cursor.close()
            
            print(f"✅ Misclassification results saved to table: {table_name}")
            print(f"   - {len(false_positives)} false positives")
            print(f"   - {len(false_negatives)} false negatives")
            
        except psycopg2.Error as e:
            print(f"❌ Error saving misclassification results: {e}")
        finally:
            connection.close()
    
    def generate_investigation_queries(self, false_positives, false_negatives):
        """
        Generate SQL queries for manual investigation of misclassified scripts.
        """
        print("\n📝 Generating Investigation Queries...")
        
        queries_file = f"{self.output_dir}/investigation_queries_{self.timestamp}.sql"
        
        with open(queries_file, 'w') as f:
            f.write("-- Behavioral Biometric Detection: Misclassification Investigation Queries\n")
            f.write(f"-- Generated: {datetime.now()}\n")
            f.write(f"-- Model: {self.model_type}\n\n")
            
            # Query for false positives
            if len(false_positives) > 0:
                f.write("-- FALSE POSITIVES: Normal scripts incorrectly flagged as behavioral biometric\n")
                f.write("-- These may reveal weaknesses in your detection logic\n\n")
                
                fp_script_ids = [str(int(row.get('script_id', 0))) for _, row in false_positives.iterrows() 
                               if pd.notna(row.get('script_id'))]
                
                if fp_script_ids:
                    f.write("-- Query to examine false positive scripts in detail\n")
                    f.write("SELECT \n")
                    f.write("    script_id,\n    script_url,\n    max_api_aggregation_score,\n")
                    f.write("    behavioral_api_agg_count,\n    fingerprinting_source_api_count,\n")
                    f.write("    behavioral_source_apis,\n    apis_going_to_sink,\n    label\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fp_script_ids)})\n")
                    f.write("ORDER BY max_api_aggregation_score DESC;\n\n")
                    
                    f.write("-- High-confidence false positives (most concerning)\n")
                    high_conf_fp = false_positives.nlargest(5, 'prediction_confidence')
                    high_conf_ids = [str(int(row.get('script_id', 0))) for _, row in high_conf_fp.iterrows() 
                                   if pd.notna(row.get('script_id'))]
                    
                    if high_conf_ids:
                        f.write("SELECT script_id, script_url, code\n")
                        f.write(f"FROM {self.table_name}\n")
                        f.write(f"WHERE script_id IN ({', '.join(high_conf_ids)});\n\n")
            
            # Query for false negatives
            if len(false_negatives) > 0:
                f.write("-- FALSE NEGATIVES: Behavioral biometric scripts that were missed\n")
                f.write("-- These reveal gaps in your detection capabilities\n\n")
                
                fn_script_ids = [str(int(row.get('script_id', 0))) for _, row in false_negatives.iterrows() 
                               if pd.notna(row.get('script_id'))]
                
                if fn_script_ids:
                    f.write("-- Query to examine false negative scripts in detail\n")
                    f.write("SELECT \n")
                    f.write("    script_id,\n    script_url,\n    max_api_aggregation_score,\n")
                    f.write("    behavioral_api_agg_count,\n    fingerprinting_source_api_count,\n")
                    f.write("    behavioral_source_apis,\n    apis_going_to_sink,\n    label\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fn_script_ids)})\n")
                    f.write("ORDER BY max_api_aggregation_score ASC;\n\n")
                    
                    f.write("-- Near-miss false negatives (close to detection threshold)\n")
                    near_miss_fn = false_negatives.nlargest(5, 'positive_probability')
                    near_miss_ids = [str(int(row.get('script_id', 0))) for _, row in near_miss_fn.iterrows() 
                                   if pd.notna(row.get('script_id'))]
                    
                    if near_miss_ids:
                        f.write("SELECT script_id, script_url, code\n")
                        f.write(f"FROM {self.table_name}\n")
                        f.write(f"WHERE script_id IN ({', '.join(near_miss_ids)});\n\n")
            
            # General investigation queries
            f.write("-- GENERAL INVESTIGATION QUERIES\n\n")
            
            f.write("-- Find scripts with similar patterns to false positives\n")
            f.write("SELECT script_id, script_url, max_api_aggregation_score, behavioral_api_agg_count\n")
            f.write(f"FROM {self.table_name}\n")
            f.write("WHERE label = 0 \n")
            f.write("  AND max_api_aggregation_score > 10\n")
            f.write("  AND behavioral_api_agg_count > 5\n")
            f.write("ORDER BY max_api_aggregation_score DESC\n")
            f.write("LIMIT 20;\n\n")
            
            f.write("-- Find potentially mislabeled scripts (label=-1 with high biometric signals)\n")
            f.write("SELECT script_id, script_url, max_api_aggregation_score, behavioral_api_agg_count\n")
            f.write(f"FROM {self.table_name}\n")
            f.write("WHERE label = -1 \n")
            f.write("  AND max_api_aggregation_score > 15\n")
            f.write("  AND fingerprinting_source_api_count > 10\n")
            f.write("  AND graph_construction_failure = false\n")
            f.write("ORDER BY max_api_aggregation_score DESC\n")
            f.write("LIMIT 50;\n\n")
        
        print(f"📝 Investigation queries saved: {queries_file}")
        
        # Print the most important queries to console
        print(f"\n🔍 Key Investigation Queries:")
        if len(false_positives) > 0:
            fp_script_ids = [str(int(row.get('script_id', 0))) for _, row in false_positives.iterrows() 
                           if pd.notna(row.get('script_id'))][:5]  # Top 5
            if fp_script_ids:
                print(f"\n-- Examine top false positives:")
                print(f"SELECT script_id, script_url, max_api_aggregation_score FROM {self.table_name}")
                print(f"WHERE script_id IN ({', '.join(fp_script_ids)});")
        
        if len(false_negatives) > 0:
            fn_script_ids = [str(int(row.get('script_id', 0))) for _, row in false_negatives.iterrows() 
                           if pd.notna(row.get('script_id'))][:5]  # Top 5
            if fn_script_ids:
                print(f"\n-- Examine missed behavioral biometric scripts:")
                print(f"SELECT script_id, script_url, behavioral_api_agg_count FROM {self.table_name}")
                print(f"WHERE script_id IN ({', '.join(fn_script_ids)});")
    
    def generate_final_report(self):
        """
        Generate a comprehensive final analysis report.
        """
        print("\n📊 Generating Comprehensive Final Report...")
        
        report_file = f"{self.output_dir}/final_analysis_report_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Behavioral Biometric Detection Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now()}\n")
            f.write(f"**Model Type:** {self.model_type}\n")
            f.write(f"**Analysis ID:** {self.timestamp}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            if hasattr(self, 'misclassification_analysis'):
                total_samples = len(self.misclassification_analysis['all_results'])
                accuracy = sum(self.misclassification_analysis['all_results']['correct_prediction']) / total_samples
                fp_count = len(self.misclassification_analysis['false_positives'])
                fn_count = len(self.misclassification_analysis['false_negatives'])
                
                f.write(f"- **Overall Accuracy:** {accuracy:.1%}\n")
                f.write(f"- **False Positive Rate:** {fp_count/total_samples:.1%}\n")
                f.write(f"- **False Negative Rate:** {fn_count/total_samples:.1%}\n")
                f.write(f"- **Model Robustness:** {'Excellent' if accuracy > 0.95 else 'Good' if accuracy > 0.90 else 'Needs Improvement'}\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            f.write("### Most Important Features\n")
            if hasattr(self, 'best_model'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False).head(5)
                
                for _, row in importance_df.iterrows():
                    f.write(f"- **{row['feature']}:** {row['importance']:.3f}\n")
            
            f.write("\n### Detection Patterns\n")
            f.write("- Behavioral biometric scripts show distinctive aggregation patterns\n")
            f.write("- Graph construction success is a strong differentiator\n")
            f.write("- API diversity and intensity are key indicators\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Production Deployment\n")
            f.write("1. Use the trained Random Forest model with 95%+ confidence threshold\n")
            f.write("2. Implement secondary validation for edge cases\n")
            f.write("3. Monitor false positive rates in production\n\n")
            
            f.write("### For Model Improvement\n")
            f.write("1. Investigate false positives to refine feature engineering\n")
            f.write("2. Analyze label=-1 samples for potential new positives\n")
            f.write("3. Consider temporal features for next iteration\n\n")
            
            # Technical Details
            f.write("## Technical Implementation\n\n")
            f.write("### Model Configuration\n")
            if hasattr(self, 'best_model'):
                params = self.best_model.get_params()
                for key, value in sorted(params.items()):
                    f.write(f"- **{key}:** {value}\n")
            
            f.write("\n### Feature Engineering\n")
            f.write(f"- **Total Features:** {len(self.feature_names)}\n")
            f.write(f"- **Feature Categories:** Aggregation, Volume, Flow, Diversity\n")
            f.write(f"- **Data Quality:** Graph construction analysis performed\n\n")
            
            # Files Generated
            f.write("## Generated Files\n\n")
            f.write("### Visualizations\n")
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.png') and self.timestamp in filename:
                    f.write(f"- `{filename}`\n")
            
            f.write("\n### Data Files\n")
            for filename in os.listdir(self.output_dir):
                if filename.endswith(('.pkl', '.sql', '.csv')) and self.timestamp in filename:
                    f.write(f"- `{filename}`\n")
        
        print(f"📊 Final report saved: {report_file}")
        return report_file
    
    def run_comprehensive_analysis(self):
        """
        Run the complete enhanced analysis pipeline with all new features.
        """
        print("🚀 " + "="*60)
        print("ENHANCED BEHAVIORAL BIOMETRIC SCRIPT DETECTION ANALYSIS")
        print("="*60 + " 🚀")
        
        try:
            # Step 1: Load and explore data with quality checks
            self.load_and_explore_data()
            
            # Step 2: Enhanced feature engineering with validation
            self.engineer_features()
            
            # Step 3: Correlation analysis to identify redundant features
            self.analyze_feature_correlations()
            
            # Step 4: Statistical feature distribution analysis
            self.analyze_feature_distributions()
            
            # Step 5: Build enhanced models with hyperparameter tuning
            dt_model, rf_model, best_results = self.build_enhanced_models()
            
            # Step 6: Comprehensive misclassification analysis
            self.perform_comprehensive_misclassification_analysis()
            
            # Step 7: Generate final comprehensive report
            report_file = self.generate_final_report()
            
            print("\n🎉 " + "="*60)
            print("ENHANCED ANALYSIS COMPLETE")
            print("="*60 + " 🎉")
            
            print(f"\n📋 Summary:")
            print(f"  ✅ Model trained and validated: {self.model_type}")
            print(f"  ✅ Misclassification analysis complete")
            print(f"  ✅ All visualizations saved to: {self.output_dir}/")
            print(f"  ✅ Investigation queries generated")
            print(f"  ✅ Final report: {report_file}")
            
            print(f"\n🔍 Next Steps:")
            print(f"  1. Review saved plots in {self.output_dir}/")
            print(f"  2. Run investigation queries on misclassified scripts")
            print(f"  3. Consider deploying model for production detection")
            print(f"  4. Analyze label=-1 samples for potential new positives")
            
            return {
                'model': self.best_model,
                'model_type': self.model_type,
                'results': best_results,
                'misclassification_analysis': self.misclassification_analysis,
                'output_directory': self.output_dir,
                'report_file': report_file
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Example usage
if __name__ == "__main__":
    print("🚀 Starting Enhanced Behavioral Biometric Detection Analysis...")
    
    # Initialize the enhanced detector
    detector = EnhancedBehavioralBiometricDetector(output_dir="behavioral_biometric_analysis")
    
    try:
        # Run the comprehensive analysis
        results = detector.run_comprehensive_analysis()
        
        if results:
            print(f"\n🎯 Analysis completed successfully!")
            print(f"📁 Check the output directory '{results['output_directory']}' for all generated files and visualizations.")
            print(f"📊 Model type selected: {results['model_type']}")
            print(f"📈 Cross-validation AUC: {results['results']['val_auc_mean']:.3f}")
            print(f"📋 Final report: {results['report_file']}")
            
            # Print summary of generated files
            output_files = [f for f in os.listdir(results['output_directory']) if detector.timestamp in f]
            print(f"\n📄 Generated {len(output_files)} files:")
            for file in sorted(output_files):
                print(f"  - {file}")
                
        else:
            print(f"\n❌ Analysis failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n💥 Unexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n✅ Enhanced Behavioral Biometric Detection Analysis Complete!")