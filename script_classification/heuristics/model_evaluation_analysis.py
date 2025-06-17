import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import psycopg2.extras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           precision_recall_curve, roc_curve, average_precision_score,
                           precision_score, recall_score, f1_score)
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluationAnalysis:
    """
    Comprehensive evaluation and failure analysis for the recommended behavioral biometric detection model.
    
    This class provides detailed analysis of:
    - Precision/Recall performance
    - False Positive/False Negative analysis
    - Script-level failure investigation
    - Feature importance for misclassified cases
    - Threshold optimization
    """
    
    def __init__(self, model_path, db_config=None, output_dir="model_evaluation_analysis"):
        """
        Initialize the evaluation system.
        
        Args:
            model_path (str): Path to the trained model pickle file
            db_config (dict): Database configuration
            output_dir (str): Directory to save analysis results
        """
        # Database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        
        self.table_name = 'multicore_static_info_known_companies'
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load the model
        self.load_model(model_path)
        
        # Data storage
        self.raw_data = None
        self.features_df = None
        self.X = None
        self.y = None
        self.y_pred = None
        self.y_pred_proba = None
        
        # Analysis results
        self.evaluation_results = {}
        self.failure_analysis = {}
        
    def save_plot(self, filename_suffix, tight_layout=True):
        """Helper method to save plots."""
        if tight_layout:
            plt.tight_layout()
        filename = f"{self.output_dir}/evaluation_{self.timestamp}_{filename_suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved: {filename}")
        plt.close()
    
    def load_model(self, model_path):
        """Load the trained model and extract metadata."""
        print(f"📦 Loading model from: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data.get('model_type', 'Unknown')
            
            print(f"✅ Loaded: {self.model_type}")
            print(f"📊 Features: {len(self.feature_names)}")
            print(f"🎯 Feature names: {self.feature_names}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def connect_to_database(self):
        """Establish database connection."""
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            print(f"❌ Error connecting to database: {e}")
            return None
    
    def load_evaluation_dataset(self):
        """
        Load the dataset that the model was trained on.
        For imbalanced model: label 1 vs (0 + -1)
        """
        print("🔌 Loading evaluation dataset...")
        
        connection = self.connect_to_database()
        if connection is None:
            raise Exception("Failed to connect to database")
        
        try:
            query = f"SELECT * FROM {self.table_name}"
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert to list with JSON parsing
            self.raw_data = []
            for row in rows:
                record = dict(row)
                
                # Parse JSON fields
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
                                record[field] = None
                        elif record[field] == '{}':
                            record[field] = {}
                        elif record[field] == '[]':
                            record[field] = []
                
                # Convert to imbalanced dataset format: 1 vs (0 + -1)
                if record['label'] == 1:
                    record['label'] = 1  # Behavioral biometric
                else:
                    record['label'] = 0  # Normal (includes original 0 and -1)
                
                self.raw_data.append(record)
            
            cursor.close()
            
        except psycopg2.Error as e:
            print(f"❌ Error querying database: {e}")
            raise
        finally:
            connection.close()
        
        # Print dataset statistics
        labels = [script['label'] for script in self.raw_data]
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        
        print(f"✅ Loaded {len(self.raw_data)} scripts")
        print(f"📊 Distribution: {pos_count} behavioral biometric, {neg_count} normal")
        print(f"📊 Imbalance ratio: 1:{neg_count/pos_count:.1f}")
        
        return self.raw_data
    
    def engineer_features(self):
        """Engineer features using the same pipeline as training."""
        print("🔧 Engineering features for evaluation...")
        
        features_list = []
        
        for script in self.raw_data:
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
                features['graph_construction_failure'] = int(script.get('graph_construction_failure', True))
                features['dataflow_to_sink'] = int(script.get('dataflow_to_sink', False))
                
                # Intensity analysis
                behavioral_access = script.get('behavioral_apis_access_count') or {}
                fp_access = script.get('fingerprinting_api_access_count') or {}
                
                features['total_behavioral_api_accesses'] = sum(behavioral_access.values()) if behavioral_access else 0
                features['total_fp_api_accesses'] = sum(fp_access.values()) if fp_access else 0
                features['unique_behavioral_apis'] = len(behavioral_access) if behavioral_access else 0
                features['unique_fp_apis'] = len(fp_access) if fp_access else 0
                
                # Sink analysis
                sink_data = script.get('apis_going_to_sink') or {}
                features['num_sink_types'] = len(sink_data) if sink_data else 0
                features['has_storage_sink'] = int(any('Storage' in str(sink) for sink in sink_data.keys()) if sink_data else False)
                features['has_network_sink'] = int(any(sink in ['XMLHttpRequest.send', 'Navigator.sendBeacon', 'fetch'] 
                                                       for sink in sink_data.keys()) if sink_data else False)
                
                # Behavioral diversity analysis
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
                
                # Derived ratio features
                total_apis = features['behavioral_source_api_count'] + features['fingerprinting_source_api_count']
                if total_apis > 0:
                    features['behavioral_ratio'] = features['behavioral_source_api_count'] / total_apis
                    features['intensity_ratio'] = features['total_behavioral_api_accesses'] / total_apis
                else:
                    features['behavioral_ratio'] = 0
                    features['intensity_ratio'] = 0
                
                # Store metadata
                features['script_id'] = script.get('script_id')
                features['script_url'] = script.get('script_url', 'Unknown')
                features['label'] = script['label']
                
                features_list.append(features)
                
            except Exception as e:
                print(f"⚠️ Feature extraction error for script {script.get('script_id', 'unknown')}: {e}")
        
        # Create DataFrame
        self.features_df = pd.DataFrame(features_list)
        
        # Extract features that match the model
        self.X = self.features_df[self.feature_names].values
        self.y = self.features_df['label'].values
        
        print(f"✅ Engineered features for {len(self.features_df)} scripts")
        print(f"📊 Feature matrix shape: {self.X.shape}")
        
        return self.features_df
    
    def evaluate_model_performance(self):
        """Comprehensive model performance evaluation."""
        print("\n🎯 Evaluating model performance...")
        
        # Make predictions
        self.y_pred = self.model.predict(self.X)
        self.y_pred_proba = self.model.predict_proba(self.X)[:, 1]
        
        # Calculate comprehensive metrics
        accuracy = np.mean(self.y_pred == self.y)
        precision = precision_score(self.y, self.y_pred)
        recall = recall_score(self.y, self.y_pred)
        f1 = f1_score(self.y, self.y_pred)
        auc = roc_auc_score(self.y, self.y_pred_proba)
        ap = average_precision_score(self.y, self.y_pred_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(self.y, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Store results
        self.evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'average_precision': ap,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'confusion_matrix': cm
        }
        
        # Print comprehensive results
        print("\n📊 COMPREHENSIVE PERFORMANCE METRICS:")
        print("="*50)
        print(f"🎯 Overall Accuracy:     {accuracy:.3f}")
        print(f"🎯 Precision:           {precision:.3f}")
        print(f"🎯 Recall:              {recall:.3f}")
        print(f"🎯 F1-Score:            {f1:.3f}")
        print(f"🎯 AUC:                 {auc:.3f}")
        print(f"🎯 Average Precision:   {ap:.3f}")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  True Positives:       {tp}")
        print(f"  False Positives:      {fp}")
        print(f"  True Negatives:       {tn}")
        print(f"  False Negatives:      {fn}")
        
        print(f"\n⚠️ ERROR RATES:")
        print(f"  False Positive Rate:  {fpr:.3f} ({fp}/{fp+tn})")
        print(f"  False Negative Rate:  {fnr:.3f} ({fn}/{fn+tp})")
        
        # Classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y, self.y_pred, 
                                  target_names=['Normal Scripts', 'Behavioral Biometric']))
        
        return self.evaluation_results
    
    def analyze_prediction_failures(self):
        """Detailed analysis of false positives and false negatives."""
        print("\n🔍 Analyzing prediction failures...")
        
        # Create detailed results DataFrame
        results_df = self.features_df.copy()
        results_df['predicted_label'] = self.y_pred
        results_df['predicted_probability'] = self.y_pred_proba
        results_df['prediction_confidence'] = np.max(self.model.predict_proba(self.X), axis=1)
        results_df['correct_prediction'] = self.y == self.y_pred
        
        # Identify different prediction types
        false_positives = results_df[
            (results_df['label'] == 0) & (results_df['predicted_label'] == 1)
        ].copy()
        
        false_negatives = results_df[
            (results_df['label'] == 1) & (results_df['predicted_label'] == 0)
        ].copy()
        
        true_positives = results_df[
            (results_df['label'] == 1) & (results_df['predicted_label'] == 1)
        ].copy()
        
        true_negatives = results_df[
            (results_df['label'] == 0) & (results_df['predicted_label'] == 0)
        ].copy()
        
        print(f"\n📊 PREDICTION BREAKDOWN:")
        print(f"  ✅ True Positives:   {len(true_positives)}")
        print(f"  ✅ True Negatives:   {len(true_negatives)}")
        print(f"  ❌ False Positives:  {len(false_positives)}")
        print(f"  ❌ False Negatives:  {len(false_negatives)}")
        
        # Analyze false positives
        if len(false_positives) > 0:
            print(f"\n🚨 FALSE POSITIVE ANALYSIS:")
            print(f"   Count: {len(false_positives)}")
            print(f"   Average Confidence: {false_positives['prediction_confidence'].mean():.3f}")
            print(f"   Average Probability: {false_positives['predicted_probability'].mean():.3f}")
            
            # Top false positives by confidence
            top_fp = false_positives.nlargest(10, 'prediction_confidence')
            print(f"\n   🔥 TOP 10 MOST CONFIDENT FALSE POSITIVES:")
            for idx, row in top_fp.iterrows():
                script_info = f"Script {row['script_id']}"
                if pd.notna(row['script_url']) and row['script_url'] != 'Unknown':
                    url_short = row['script_url'][:60] + "..." if len(row['script_url']) > 60 else row['script_url']
                    script_info += f" ({url_short})"
                
                print(f"     {script_info}")
                print(f"       Confidence: {row['prediction_confidence']:.3f}")
                print(f"       Key features: fp_apis={row.get('total_fp_api_accesses', 0):.0f}, "
                      f"agg_score={row.get('max_api_aggregation_score', 0):.0f}")
        
        # Analyze false negatives
        if len(false_negatives) > 0:
            print(f"\n🎯 FALSE NEGATIVE ANALYSIS:")
            print(f"   Count: {len(false_negatives)}")
            print(f"   Average Confidence: {false_negatives['prediction_confidence'].mean():.3f}")
            print(f"   Average Probability: {false_negatives['predicted_probability'].mean():.3f}")
            
            # Closest misses (highest probability among false negatives)
            close_misses = false_negatives.nlargest(10, 'predicted_probability')
            print(f"\n   📍 TOP 10 CLOSEST MISSES (Highest Probability FNs):")
            for idx, row in close_misses.iterrows():
                script_info = f"Script {row['script_id']}"
                if pd.notna(row['script_url']) and row['script_url'] != 'Unknown':
                    url_short = row['script_url'][:60] + "..." if len(row['script_url']) > 60 else row['script_url']
                    script_info += f" ({url_short})"
                
                print(f"     {script_info}")
                print(f"       Probability: {row['predicted_probability']:.3f}")
                print(f"       Key features: behavioral_apis={row.get('behavioral_api_agg_count', 0):.0f}, "
                      f"agg_score={row.get('max_api_aggregation_score', 0):.0f}")
        
        # Store failure analysis
        self.failure_analysis = {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'results_df': results_df
        }
        
        return self.failure_analysis
    
    def analyze_feature_patterns_in_failures(self):
        """Analyze feature patterns in misclassified samples."""
        print("\n🔬 Analyzing feature patterns in failures...")
        
        fp_df = self.failure_analysis['false_positives']
        fn_df = self.failure_analysis['false_negatives']
        tp_df = self.failure_analysis['true_positives']
        tn_df = self.failure_analysis['true_negatives']
        
        if len(fp_df) == 0 and len(fn_df) == 0:
            print("✅ No failures to analyze - perfect predictions!")
            return
        
        # Feature analysis for false positives
        if len(fp_df) > 0:
            print(f"\n🚨 FALSE POSITIVE FEATURE ANALYSIS:")
            
            # Compare FP features with TN features
            fp_features = fp_df[self.feature_names].mean()
            tn_features = tn_df[self.feature_names].mean()
            
            print(f"   📊 Average feature values (FP vs TN):")
            feature_diffs = []
            for feature in self.feature_names:
                fp_val = fp_features[feature]
                tn_val = tn_features[feature]
                diff = fp_val - tn_val
                feature_diffs.append((feature, fp_val, tn_val, diff))
                if abs(diff) > 0.1:  # Only show significant differences
                    print(f"     {feature}: FP={fp_val:.2f}, TN={tn_val:.2f} (diff: {diff:+.2f})")
            
            # Find features that most distinguish FPs from TNs
            feature_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"\n   🔍 Top features distinguishing FPs from TNs:")
            for feature, fp_val, tn_val, diff in feature_diffs[:5]:
                print(f"     {feature}: {diff:+.3f} difference")
        
        # Feature analysis for false negatives
        if len(fn_df) > 0:
            print(f"\n🎯 FALSE NEGATIVE FEATURE ANALYSIS:")
            
            # Compare FN features with TP features
            fn_features = fn_df[self.feature_names].mean()
            tp_features = tp_df[self.feature_names].mean()
            
            print(f"   📊 Average feature values (FN vs TP):")
            feature_diffs = []
            for feature in self.feature_names:
                fn_val = fn_features[feature]
                tp_val = tp_features[feature]
                diff = fn_val - tp_val
                feature_diffs.append((feature, fn_val, tp_val, diff))
                if abs(diff) > 0.1:  # Only show significant differences
                    print(f"     {feature}: FN={fn_val:.2f}, TP={tp_val:.2f} (diff: {diff:+.2f})")
            
            # Find features that most distinguish FNs from TPs
            feature_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            print(f"\n   🔍 Top features distinguishing FNs from TPs:")
            for feature, fn_val, tp_val, diff in feature_diffs[:5]:
                print(f"     {feature}: {diff:+.3f} difference")
    
    def optimize_decision_threshold(self):
        """Find optimal decision threshold for precision/recall tradeoff."""
        print("\n⚖️ Optimizing decision threshold...")
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.y_pred_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores[np.isnan(f1_scores)] = 0
        
        # Find optimal thresholds for different objectives
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        
        # Find threshold for high precision (>0.95)
        high_precision_indices = np.where(precisions >= 0.95)[0]
        if len(high_precision_indices) > 0:
            high_precision_threshold = thresholds[high_precision_indices[0]]
            high_precision_recall = recalls[high_precision_indices[0]]
        else:
            high_precision_threshold = None
            high_precision_recall = None
        
        # Find threshold for high recall (>0.95)
        high_recall_indices = np.where(recalls >= 0.95)[0]
        if len(high_recall_indices) > 0:
            high_recall_threshold = thresholds[high_recall_indices[-1]]
            high_recall_precision = precisions[high_recall_indices[-1]]
        else:
            high_recall_threshold = None
            high_recall_precision = None
        
        # Current threshold (0.5) performance
        current_pred = (self.y_pred_proba >= 0.5).astype(int)
        current_precision = precision_score(self.y, current_pred)
        current_recall = recall_score(self.y, current_pred)
        current_f1 = f1_score(self.y, current_pred)
        
        print(f"\n📊 THRESHOLD OPTIMIZATION RESULTS:")
        print(f"="*50)
        print(f"Current Threshold (0.5):")
        print(f"  Precision: {current_precision:.3f}")
        print(f"  Recall:    {current_recall:.3f}")
        print(f"  F1-Score:  {current_f1:.3f}")
        
        print(f"\nOptimal F1 Threshold ({optimal_f1_threshold:.3f}):")
        print(f"  Precision: {precisions[optimal_f1_idx]:.3f}")
        print(f"  Recall:    {recalls[optimal_f1_idx]:.3f}")
        print(f"  F1-Score:  {f1_scores[optimal_f1_idx]:.3f}")
        
        if high_precision_threshold is not None:
            print(f"\nHigh Precision Threshold ({high_precision_threshold:.3f}):")
            print(f"  Precision: ≥0.95")
            print(f"  Recall:    {high_precision_recall:.3f}")
        
        if high_recall_threshold is not None:
            print(f"\nHigh Recall Threshold ({high_recall_threshold:.3f}):")
            print(f"  Precision: {high_recall_precision:.3f}")
            print(f"  Recall:    ≥0.95")
        
        return {
            'optimal_f1_threshold': optimal_f1_threshold,
            'high_precision_threshold': high_precision_threshold,
            'high_recall_threshold': high_recall_threshold,
            'thresholds': thresholds,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores
        }
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for model evaluation."""
        print("\n📊 Creating comprehensive evaluation visualizations...")
        
        # Create a large figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        
        # Plot 1: Confusion Matrix
        cm = self.evaluation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Behavioral Biometric'],
                   yticklabels=['Normal', 'Behavioral Biometric'], ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_ylabel('True Label')
        
        # Plot 2: ROC Curve
        fpr, tpr, _ = roc_curve(self.y, self.y_pred_proba)
        auc = self.evaluation_results['auc']
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y, self.y_pred_proba)
        ap = self.evaluation_results['average_precision']
        axes[0, 2].plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Prediction Probability Distribution
        pos_probs = self.y_pred_proba[self.y == 1]
        neg_probs = self.y_pred_proba[self.y == 0]
        
        axes[1, 0].hist(neg_probs, bins=50, alpha=0.7, label='Normal Scripts', color='blue', density=True)
        axes[1, 0].hist(pos_probs, bins=50, alpha=0.7, label='Behavioral Biometric', color='red', density=True)
        axes[1, 0].axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 1].set_yticks(range(len(feature_importance)))
        axes[1, 1].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in feature_importance['feature']], fontsize=8)
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 10 Feature Importance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Threshold Analysis
        if hasattr(self, 'threshold_results'):
            thresholds = self.threshold_results['thresholds']
            precisions = self.threshold_results['precisions']
            recalls = self.threshold_results['recalls']
            f1_scores = self.threshold_results['f1_scores']
            
            axes[1, 2].plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
            axes[1, 2].plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
            axes[1, 2].plot(thresholds, f1_scores[:-1], label='F1-Score', linewidth=2)
            axes[1, 2].axvline(0.5, color='black', linestyle='--', alpha=0.7, label='Current Threshold')
            axes[1, 2].set_xlabel('Decision Threshold')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_title('Threshold vs Performance Metrics')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 7: Error Analysis - if we have failures
        if len(self.failure_analysis['false_positives']) > 0 or len(self.failure_analysis['false_negatives']) > 0:
            fp_df = self.failure_analysis['false_positives']
            fn_df = self.failure_analysis['false_negatives']
            tp_df = self.failure_analysis['true_positives']
            tn_df = self.failure_analysis['true_negatives']
            
            # Box plot of key features for different prediction types
            key_feature = 'total_fp_api_accesses'  # Most important feature
            if key_feature in self.feature_names:
                data_groups = []
                labels = []
                
                if len(tn_df) > 0:
                    data_groups.append(tn_df[key_feature].values)
                    labels.append(f'True Neg (n={len(tn_df)})')
                
                if len(fp_df) > 0:
                    data_groups.append(fp_df[key_feature].values)
                    labels.append(f'False Pos (n={len(fp_df)})')
                
                if len(fn_df) > 0:
                    data_groups.append(fn_df[key_feature].values)
                    labels.append(f'False Neg (n={len(fn_df)})')
                
                if len(tp_df) > 0:
                    data_groups.append(tp_df[key_feature].values)
                    labels.append(f'True Pos (n={len(tp_df)})')
                
                if data_groups:
                    bp = axes[2, 0].boxplot(data_groups, labels=labels, patch_artist=True)
                    colors = ['lightgreen', 'red', 'orange', 'darkgreen']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    axes[2, 0].set_title(f'{key_feature} Distribution by Prediction Type')
                    axes[2, 0].set_ylabel('Feature Value')
                    axes[2, 0].tick_params(axis='x', rotation=45)
                    axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 8: Confidence vs Accuracy
        results_df = self.failure_analysis['results_df']
        confidence_bins = pd.cut(results_df['prediction_confidence'], bins=10)
        confidence_accuracy = results_df.groupby(confidence_bins)['correct_prediction'].mean()
        bin_centers = [interval.mid for interval in confidence_accuracy.index]
        
        axes[2, 1].plot(bin_centers, confidence_accuracy.values, 'o-', linewidth=2, markersize=8)
        axes[2, 1].set_xlabel('Prediction Confidence')
        axes[2, 1].set_ylabel('Accuracy')
        axes[2, 1].set_title('Confidence vs Accuracy (Calibration)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim([0, 1])
        
        # Plot 9: Summary Metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            self.evaluation_results['accuracy'],
            self.evaluation_results['precision'],
            self.evaluation_results['recall'],
            self.evaluation_results['f1_score'],
            self.evaluation_results['auc']
        ]
        
        bars = axes[2, 2].bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[2, 2].set_ylabel('Score')
        axes[2, 2].set_title('Performance Summary')
        axes[2, 2].set_ylim([0, 1])
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[2, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Comprehensive Model Evaluation Analysis', fontsize=16, y=0.98)
        self.save_plot('comprehensive_evaluation')
    
    def generate_failure_investigation_queries(self):
        """Generate SQL queries to investigate specific failed predictions."""
        print("\n📝 Generating failure investigation queries...")
        
        fp_df = self.failure_analysis['false_positives']
        fn_df = self.failure_analysis['false_negatives']
        
        queries_file = f"{self.output_dir}/failure_investigation_queries_{self.timestamp}.sql"
        
        with open(queries_file, 'w') as f:
            f.write("-- MODEL EVALUATION: FAILURE INVESTIGATION QUERIES\n")
            f.write(f"-- Generated: {datetime.now()}\n")
            f.write(f"-- Model: {self.model_type}\n\n")
            
            # False Positive Investigation
            if len(fp_df) > 0:
                f.write("-- FALSE POSITIVE INVESTIGATION\n")
                f.write("-- Normal scripts incorrectly flagged as behavioral biometric\n\n")
                
                fp_script_ids = [str(int(row['script_id'])) for _, row in fp_df.iterrows() 
                               if pd.notna(row['script_id'])][:20]  # Top 20
                
                if fp_script_ids:
                    f.write("-- High-confidence false positives (detailed analysis)\n")
                    f.write("SELECT \n")
                    f.write("    script_id,\n    script_url,\n    max_api_aggregation_score,\n")
                    f.write("    behavioral_api_agg_count,\n    fingerprinting_source_api_count,\n")
                    f.write("    total_fp_api_accesses,\n    behavioral_source_apis,\n")
                    f.write("    fingerprinting_source_apis,\n    apis_going_to_sink,\n    label\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fp_script_ids)})\n")
                    f.write("ORDER BY fingerprinting_source_api_count DESC;\n\n")
                    
                    # Domain analysis for false positives
                    f.write("-- Domain analysis for false positives\n")
                    f.write("SELECT \n")
                    f.write("    REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\\1') as domain,\n")
                    f.write("    COUNT(*) as fp_count,\n")
                    f.write("    AVG(fingerprinting_source_api_count) as avg_fp_apis\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fp_script_ids)})\n")
                    f.write("    AND script_url IS NOT NULL\n")
                    f.write("GROUP BY REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\\1')\n")
                    f.write("ORDER BY fp_count DESC;\n\n")
            
            # False Negative Investigation
            if len(fn_df) > 0:
                f.write("-- FALSE NEGATIVE INVESTIGATION\n")
                f.write("-- Behavioral biometric scripts that were missed\n\n")
                
                fn_script_ids = [str(int(row['script_id'])) for _, row in fn_df.iterrows() 
                               if pd.notna(row['script_id'])][:20]  # Top 20
                
                if fn_script_ids:
                    f.write("-- Missed behavioral biometric scripts (detailed analysis)\n")
                    f.write("SELECT \n")
                    f.write("    script_id,\n    script_url,\n    max_api_aggregation_score,\n")
                    f.write("    behavioral_api_agg_count,\n    fingerprinting_source_api_count,\n")
                    f.write("    behavioral_source_apis,\n    apis_going_to_sink,\n    label\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fn_script_ids)})\n")
                    f.write("ORDER BY behavioral_api_agg_count ASC;\n\n")
                    
                    f.write("-- Near-miss analysis (scripts with some biometric signals but missed)\n")
                    f.write("SELECT \n")
                    f.write("    script_id,\n    script_url,\n    behavioral_api_agg_count,\n")
                    f.write("    fingerprinting_source_api_count,\n    behavioral_source_apis\n")
                    f.write(f"FROM {self.table_name}\n")
                    f.write(f"WHERE script_id IN ({', '.join(fn_script_ids)})\n")
                    f.write("    AND (behavioral_api_agg_count > 0 OR fingerprinting_source_api_count > 5)\n")
                    f.write("ORDER BY behavioral_api_agg_count DESC;\n\n")
            
            # General investigation queries
            f.write("-- GENERAL INVESTIGATION QUERIES\n\n")
            
            f.write("-- Scripts with similar patterns to false positives\n")
            if len(fp_df) > 0:
                avg_fp_apis = fp_df['total_fp_api_accesses'].mean()
                avg_agg_score = fp_df['max_api_aggregation_score'].mean()
                
                f.write("SELECT script_id, script_url, total_fp_api_accesses, max_api_aggregation_score\n")
                f.write(f"FROM {self.table_name}\n")
                f.write("WHERE label = 0\n")
                f.write(f"  AND total_fp_api_accesses > {avg_fp_apis:.0f}\n")
                f.write(f"  AND max_api_aggregation_score > {avg_agg_score:.0f}\n")
                f.write("ORDER BY total_fp_api_accesses DESC\n")
                f.write("LIMIT 20;\n\n")
            
            f.write("-- Scripts with weak biometric signals (potential mislabels)\n")
            f.write("SELECT script_id, script_url, behavioral_api_agg_count, max_api_aggregation_score\n")
            f.write(f"FROM {self.table_name}\n")
            f.write("WHERE label = 1\n")
            f.write("  AND behavioral_api_agg_count < 5\n")
            f.write("  AND max_api_aggregation_score < 10\n")
            f.write("ORDER BY behavioral_api_agg_count ASC\n")
            f.write("LIMIT 20;\n\n")
        
        print(f"📝 Investigation queries saved: {queries_file}")
        return queries_file
    
    def save_detailed_results(self):
        """Save comprehensive evaluation results."""
        print(f"\n💾 Saving detailed evaluation results...")
        
        # Save evaluation summary
        summary_file = f"{self.output_dir}/evaluation_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            summary = {}
            for key, value in self.evaluation_results.items():
                if isinstance(value, np.ndarray):
                    summary[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32)):
                    summary[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    summary[key] = float(value)
                else:
                    summary[key] = value
            
            json.dump(summary, f, indent=2)
        
        # Save failure analysis results
        failures_file = f"{self.output_dir}/failure_analysis_{self.timestamp}.csv"
        results_df = self.failure_analysis['results_df']
        
        # Add failure type column
        results_df['failure_type'] = 'Correct'
        results_df.loc[(results_df['label'] == 0) & (results_df['predicted_label'] == 1), 'failure_type'] = 'False Positive'
        results_df.loc[(results_df['label'] == 1) & (results_df['predicted_label'] == 0), 'failure_type'] = 'False Negative'
        
        # Save only key columns
        key_columns = ['script_id', 'script_url', 'label', 'predicted_label', 'predicted_probability', 
                      'prediction_confidence', 'failure_type'] + self.feature_names
        results_df[key_columns].to_csv(failures_file, index=False)
        
        print(f"✅ Results saved:")
        print(f"  📊 Summary: {summary_file}")
        print(f"  📋 Detailed results: {failures_file}")
        
        return {
            'summary_file': summary_file,
            'failures_file': failures_file
        }
    
    def run_comprehensive_evaluation(self):
        """Run the complete evaluation pipeline."""
        print("🚀 " + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION ANALYSIS")
        print("Behavioral Biometric Detection Model Performance")
        print("="*60 + " 🚀")
        
        try:
            # Step 1: Load evaluation dataset
            print(f"\n📚 STEP 1: Data Loading")
            self.load_evaluation_dataset()
            
            # Step 2: Engineer features
            print(f"\n🔧 STEP 2: Feature Engineering")
            self.engineer_features()
            
            # Step 3: Evaluate model performance
            print(f"\n🎯 STEP 3: Model Performance Evaluation")
            self.evaluate_model_performance()
            
            # Step 4: Analyze prediction failures
            print(f"\n🔍 STEP 4: Failure Analysis")
            self.analyze_prediction_failures()
            
            # Step 5: Analyze feature patterns in failures
            print(f"\n🔬 STEP 5: Feature Pattern Analysis")
            self.analyze_feature_patterns_in_failures()
            
            # Step 6: Optimize decision threshold
            print(f"\n⚖️ STEP 6: Threshold Optimization")
            self.threshold_results = self.optimize_decision_threshold()
            
            # Step 7: Create visualizations
            print(f"\n📊 STEP 7: Visualization Generation")
            self.create_comprehensive_visualizations()
            
            # Step 8: Generate investigation queries
            print(f"\n📝 STEP 8: Investigation Query Generation")
            queries_file = self.generate_failure_investigation_queries()
            
            # Step 9: Save detailed results
            print(f"\n💾 STEP 9: Save Results")
            saved_files = self.save_detailed_results()
            
            print("\n🎉 " + "="*60)
            print("COMPREHENSIVE EVALUATION COMPLETE")
            print("="*60 + " 🎉")
            
            # Print final summary
            print(f"\n📋 EVALUATION SUMMARY:")
            print(f"  🎯 Overall Accuracy: {self.evaluation_results['accuracy']:.3f}")
            print(f"  🎯 Precision: {self.evaluation_results['precision']:.3f}")
            print(f"  🎯 Recall: {self.evaluation_results['recall']:.3f}")
            print(f"  🎯 F1-Score: {self.evaluation_results['f1_score']:.3f}")
            print(f"  🎯 AUC: {self.evaluation_results['auc']:.3f}")
            
            print(f"\n❌ FAILURE ANALYSIS:")
            print(f"  🚨 False Positives: {self.evaluation_results['false_positives']}")
            print(f"  🎯 False Negatives: {self.evaluation_results['false_negatives']}")
            print(f"  📊 FP Rate: {self.evaluation_results['false_positive_rate']:.3f}")
            print(f"  📊 FN Rate: {self.evaluation_results['false_negative_rate']:.3f}")
            
            print(f"\n📁 Generated Files:")
            print(f"  📊 Visualizations: {self.output_dir}/evaluation_*.png")
            print(f"  📝 Investigation queries: {queries_file}")
            print(f"  📋 Summary: {saved_files['summary_file']}")
            print(f"  📊 Detailed results: {saved_files['failures_file']}")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            fp_count = self.evaluation_results['false_positives']
            fn_count = self.evaluation_results['false_negatives']
            
            if fp_count > fn_count:
                print(f"  🎯 Focus: Reduce false positives ({fp_count} vs {fn_count} FNs)")
                print(f"  📈 Consider: Raising decision threshold or feature refinement")
            elif fn_count > fp_count:
                print(f"  🎯 Focus: Reduce false negatives ({fn_count} vs {fp_count} FPs)")
                print(f"  📈 Consider: Lowering decision threshold or additional features")
            else:
                print(f"  ✅ Balanced: Equal FPs and FNs - model is well-calibrated")
            
            if self.evaluation_results['auc'] > 0.99:
                print(f"  🏆 Excellent: AUC > 0.99 indicates outstanding discrimination")
            
            return {
                'evaluation_results': self.evaluation_results,
                'failure_analysis': self.failure_analysis,
                'threshold_results': self.threshold_results,
                'saved_files': saved_files,
                'queries_file': queries_file,
                'output_directory': self.output_dir
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Usage example
if __name__ == "__main__":
    print("🚀 Starting Comprehensive Model Evaluation...")
    
    # Path to the recommended model from your previous analysis
    model_path = "balanced_vs_imbalanced_analysis/recommended_model_20250526_044211.pkl"
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluationAnalysis(model_path)
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        if results:
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📁 Check the output directory '{results['output_directory']}' for detailed analysis")
            
            # Print key insights
            evaluation = results['evaluation_results']
            print(f"\n🔍 KEY INSIGHTS:")
            print(f"  • Model achieves {evaluation['precision']:.1%} precision")
            print(f"  • Model achieves {evaluation['recall']:.1%} recall") 
            print(f"  • {evaluation['false_positives']} normal scripts misclassified")
            print(f"  • {evaluation['false_negatives']} behavioral biometric scripts missed")
            
        else:
            print(f"❌ Evaluation failed")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n✅ Model Evaluation Analysis Complete!")