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
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

class BehavioralBiometricDetector:
    """
    A comprehensive system for detecting behavioral biometric scripts through
    static analysis features and data flow analysis.
    """
    
    def __init__(self, db_config=None):
        """
        Initialize the detector with database configuration.
        
        Args:
            db_config (dict): Database configuration. If None, uses default config.
        """
        # Default database configuration
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        
        self.table_name = 'multicore_static_info_known_companies'
        self.raw_data = None
        self.features_df = None
        self.X = None
        self.y = None
        self.feature_names = None
        
    def connect_to_database(self):
        """
        Establish connection to PostgreSQL database.
        """
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL database: {e}")
            print(f"Connection config: {self.db_config}")
            return None
        
    def load_and_explore_data(self):
        """
        Load data from PostgreSQL database and perform initial exploration to understand
        the characteristics of behavioral biometric vs normal scripts.
        """
        print("Connecting to PostgreSQL database and loading labeled dataset...")
        
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
            
            # Convert to list of dictionaries (similar to JSON structure)
            self.raw_data = []
            for row in rows:
                # Convert row to dictionary and handle any JSON fields
                record = dict(row)
                
                # Parse JSON fields if they exist as strings
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
                                # If it's not valid JSON, keep as is
                                pass
                        # Handle empty dictionaries/lists stored as strings
                        elif record[field] == '{}':
                            record[field] = {}
                        elif record[field] == '[]':
                            record[field] = []
                
                self.raw_data.append(record)
            
            cursor.close()
            
        except psycopg2.Error as e:
            print(f"Error querying database: {e}")
            raise
        finally:
            connection.close()
        
        print(f"Loaded {len(self.raw_data)} total scripts from database")
        
        # Analyze label distribution first to understand what labels we have
        labels = [script['label'] for script in self.raw_data]
        unique_labels = sorted(set(labels))
        print(f"\nLabel distribution:")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  Label {label}: {count} samples")
        
        # Handle different labeling schemes
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            # Standard binary classification (0=negative, 1=positive)
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
            print(f"\nBinary classification detected:")
            print(f"Positive samples (behavioral biometric): {positive_count}")
            print(f"Negative samples (normal scripts): {negative_count}")
        elif len(unique_labels) == 3:
            # Three classes - need to filter or remap
            print(f"\nThree classes detected: {unique_labels}")
            print("Filtering to binary classification (keeping only labels 0 and 1)...")
            
            # Filter to only keep samples with labels 0 and 1
            original_count = len(self.raw_data)
            self.raw_data = [script for script in self.raw_data if script['label'] in [0, 1]]
            filtered_count = len(self.raw_data)
            
            print(f"Filtered from {original_count} to {filtered_count} samples")
            
            # Recalculate label distribution
            labels = [script['label'] for script in self.raw_data]
            positive_count = sum(labels)
            negative_count = len(labels) - positive_count
            print(f"Positive samples (behavioral biometric): {positive_count}")
            print(f"Negative samples (normal scripts): {negative_count}")
        else:
            print(f"\nUnexpected label scheme: {unique_labels}")
            print("Please check your data labeling.")
            
        if len([script for script in self.raw_data if script['label'] in [0, 1]]) < 10:
            raise ValueError("Not enough samples for analysis after filtering")
        
        # Calculate class balance
        labels = [script['label'] for script in self.raw_data]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        if negative_count > 0:
            print(f"Class balance ratio: {positive_count/negative_count:.2f}")
        
        # Analyze graph construction success rates by class
        pos_graph_failures = sum(1 for script in self.raw_data 
                                if script['label'] == 1 and script['graph_construction_failure'])
        neg_graph_failures = sum(1 for script in self.raw_data 
                                if script['label'] == 0 and script['graph_construction_failure'])
        
        print(f"\nGraph construction failure rates:")
        if positive_count > 0:
            print(f"Positive class: {pos_graph_failures}/{positive_count} ({pos_graph_failures/positive_count*100:.1f}%)")
        if negative_count > 0:
            print(f"Negative class: {neg_graph_failures}/{negative_count} ({neg_graph_failures/negative_count*100:.1f}%)")
        
        return self.raw_data
    
    def preview_database_schema(self):
        """
        Preview the database schema to understand the structure of your data.
        This is helpful for debugging and understanding the data format.
        """
        print("Previewing database schema and sample data...")
        
        connection = self.connect_to_database()
        if connection is None:
            return
        
        try:
            cursor = connection.cursor()
            
            # First, check label distribution
            cursor.execute(f"SELECT label, COUNT(*) as count FROM {self.table_name} GROUP BY label ORDER BY label")
            label_distribution = cursor.fetchall()
            print(f"\nLabel distribution in database:")
            for label, count in label_distribution:
                print(f"  Label {label}: {count} samples")
            
            # Get column information
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = '{self.table_name}'
                ORDER BY ordinal_position
            """)
            
            schema_info = cursor.fetchall()
            print(f"\nDatabase Schema for table '{self.table_name}':")
            print("-" * 60)
            for col_name, data_type, is_nullable in schema_info:
                print(f"{col_name:<30} {data_type:<20} {'NULL' if is_nullable == 'YES' else 'NOT NULL'}")
            
            cursor.close()
            
        except psycopg2.Error as e:
            print(f"Error querying database schema: {e}")
        finally:
            connection.close()
    
    def engineer_features(self):
        """
        Extract and engineer features from the raw JSON data.
        We'll focus on features that capture both the presence of behavioral
        data collection and the intent to use that data.
        """
        print("\nEngineering features from static analysis data...")
        
        features_list = []
        
        for script in self.raw_data:
            features = {}
            
            # Basic aggregation features (core indicators of data collection intent)
            features['max_api_aggregation_score'] = script.get('max_api_aggregation_score', -1)
            features['behavioral_api_agg_count'] = script.get('behavioral_api_agg_count', -1)
            features['fp_api_agg_count'] = script.get('fp_api_agg_count', -1)
            
            # Source API counts (volume of data collection)
            features['behavioral_source_api_count'] = script.get('behavioral_source_api_count', 0)
            features['fingerprinting_source_api_count'] = script.get('fingerprinting_source_api_count', 0)
            
            # Data flow indicators (intent to use collected data)
            features['graph_construction_failure'] = int(script.get('graph_construction_failure', True))
            features['dataflow_to_sink'] = int(script.get('dataflow_to_sink', False))
            
            # API access intensity features
            behavioral_access = script.get('behavioral_apis_access_count', {})
            fp_access = script.get('fingerprinting_api_access_count', {})
            
            features['total_behavioral_api_accesses'] = sum(behavioral_access.values()) if behavioral_access else 0
            features['total_fp_api_accesses'] = sum(fp_access.values()) if fp_access else 0
            features['unique_behavioral_apis'] = len(behavioral_access) if behavioral_access else 0
            features['unique_fp_apis'] = len(fp_access) if fp_access else 0
            
            # Sink analysis features (where the data goes)
            sink_data = script.get('apis_going_to_sink', {})
            features['num_sink_types'] = len(sink_data) if sink_data else 0
            features['has_storage_sink'] = int(any('Storage' in sink for sink in sink_data.keys()) if sink_data else False)
            features['has_network_sink'] = int(any(sink in ['XMLHttpRequest.send', 'Navigator.sendBeacon', 'fetch'] 
                                                   for sink in sink_data.keys()) if sink_data else False)
            
            # Behavioral API diversity features
            behavioral_sources = script.get('behavioral_source_apis', [])
            if behavioral_sources:
                # Count different types of behavioral events
                mouse_events = sum(1 for api in behavioral_sources if 'MouseEvent' in api)
                keyboard_events = sum(1 for api in behavioral_sources if 'KeyboardEvent' in api)
                touch_events = sum(1 for api in behavioral_sources if 'TouchEvent' in api or 'Touch.' in api)
                pointer_events = sum(1 for api in behavioral_sources if 'PointerEvent' in api)
                
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
            
            # Ratios and derived features
            total_apis = features['behavioral_source_api_count'] + features['fingerprinting_source_api_count']
            if total_apis > 0:
                features['behavioral_ratio'] = features['behavioral_source_api_count'] / total_apis
                features['intensity_ratio'] = features['total_behavioral_api_accesses'] / total_apis
            else:
                features['behavioral_ratio'] = 0
                features['intensity_ratio'] = 0
            
            # Add the label
            features['label'] = script['label']
            
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        
        # Separate features and labels
        self.y = self.features_df['label'].values
        self.X = self.features_df.drop('label', axis=1).values
        self.feature_names = list(self.features_df.drop('label', axis=1).columns)
        
        print(f"Engineered {len(self.feature_names)} features")
        print("Feature names:", self.feature_names)
        
        return self.features_df
    
    def analyze_feature_distributions(self):
        """
        Analyze how features differ between positive and negative classes.
        This helps us understand what makes behavioral biometric scripts distinctive.
        """
        print("\nAnalyzing feature distributions between classes...")
        
        # Create comparison plots for key features
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        key_features = [
            'max_api_aggregation_score', 'behavioral_api_agg_count', 'behavioral_source_api_count',
            'total_behavioral_api_accesses', 'behavioral_event_diversity', 'dataflow_to_sink',
            'has_storage_sink', 'has_network_sink', 'behavioral_ratio'
        ]
        
        for i, feature in enumerate(key_features[:9]):
            pos_data = self.features_df[self.features_df['label'] == 1][feature]
            neg_data = self.features_df[self.features_df['label'] == 0][feature]
            
            axes[i].hist([neg_data, pos_data], bins=20, alpha=0.7, 
                        label=['Negative', 'Positive'], color=['lightcoral', 'skyblue'])
            axes[i].set_title(f'{feature}')
            axes[i].legend()
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nSummary statistics by class:")
        summary_stats = self.features_df.groupby('label')[key_features].agg(['mean', 'std', 'median'])
        print(summary_stats.round(3))
    
    def build_simple_models(self):
        """
        Build both Decision Tree and Random Forest models with basic evaluation.
        """
        print("\n" + "="*60)
        print("BUILDING AND COMPARING MODELS")
        print("="*60)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Build Decision Tree
        print("\nBuilding Decision Tree...")
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        dt_model.fit(X_train, y_train)
        
        # Evaluate Decision Tree with k-fold CV
        dt_cv_scores = cross_val_score(dt_model, self.X, self.y, cv=5, scoring='roc_auc')
        dt_train_score = dt_model.score(X_train, y_train)
        dt_test_score = dt_model.score(X_test, y_test)
        
        print(f"Decision Tree Results:")
        print(f"  Training accuracy: {dt_train_score:.3f}")
        print(f"  Test accuracy: {dt_test_score:.3f}")
        print(f"  Cross-validation AUC: {dt_cv_scores.mean():.3f} (+/- {dt_cv_scores.std() * 2:.3f})")
        print(f"  Overfitting gap: {dt_train_score - dt_test_score:.3f}")
        
        if dt_train_score - dt_test_score > 0.1:
            print("  ⚠️  WARNING: Significant overfitting detected!")
        elif dt_train_score - dt_test_score > 0.05:
            print("  ⚠️  CAUTION: Mild overfitting detected")
        else:
            print("  ✅ Good: No significant overfitting")
        
        # Build Random Forest
        print("\nBuilding Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest with k-fold CV
        rf_cv_scores = cross_val_score(rf_model, self.X, self.y, cv=5, scoring='roc_auc')
        rf_train_score = rf_model.score(X_train, y_train)
        rf_test_score = rf_model.score(X_test, y_test)
        
        print(f"\nRandom Forest Results:")
        print(f"  Training accuracy: {rf_train_score:.3f}")
        print(f"  Test accuracy: {rf_test_score:.3f}")
        print(f"  Cross-validation AUC: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
        print(f"  Overfitting gap: {rf_train_score - rf_test_score:.3f}")
        
        if rf_train_score - rf_test_score > 0.1:
            print("  ⚠️  WARNING: Significant overfitting detected!")
        elif rf_train_score - rf_test_score > 0.05:
            print("  ⚠️  CAUTION: Mild overfitting detected")
        else:
            print("  ✅ Good: No significant overfitting")
        
        # Feature importance comparison
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Decision Tree feature importance
        dt_importance = pd.DataFrame({
            'feature': self.feature_names,
            'dt_importance': dt_model.feature_importances_
        }).sort_values('dt_importance', ascending=False)
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': rf_model.feature_importances_
        }).sort_values('rf_importance', ascending=False)
        
        print("\nTop 10 Features - Decision Tree:")
        print(dt_importance.head(10))
        
        print("\nTop 10 Features - Random Forest:")
        print(rf_importance.head(10))
        
        # Detailed classification reports
        dt_pred = dt_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORTS")
        print("="*60)
        
        # Check the actual classes to avoid mismatch
        unique_classes = sorted(list(set(y_test) | set(dt_pred)))
        if len(unique_classes) == 2 and set(unique_classes) == {0, 1}:
            target_names = ['Normal Script', 'Behavioral Biometric']
        else:
            target_names = [f'Class {i}' for i in unique_classes]
        
        print("\nDecision Tree Classification Report:")
        print(classification_report(y_test, dt_pred, target_names=target_names))
        
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, rf_pred, target_names=target_names))
        
        # Model comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        print(f"{'Metric':<25} {'Decision Tree':<15} {'Random Forest':<15}")
        print("-" * 55)
        print(f"{'CV AUC':<25} {dt_cv_scores.mean():<15.3f} {rf_cv_scores.mean():<15.3f}")
        print(f"{'Test Accuracy':<25} {dt_test_score:<15.3f} {rf_test_score:<15.3f}")
        print(f"{'Overfitting Gap':<25} {dt_train_score - dt_test_score:<15.3f} {rf_train_score - rf_test_score:<15.3f}")
        
        # Recommendation
        if rf_cv_scores.mean() > dt_cv_scores.mean() and (rf_train_score - rf_test_score) <= (dt_train_score - dt_test_score):
            print(f"\n🎯 RECOMMENDATION: Use Random Forest")
            print(f"   Better performance with similar or less overfitting")
        elif dt_cv_scores.mean() > rf_cv_scores.mean() and (dt_train_score - dt_test_score) < 0.1:
            print(f"\n🎯 RECOMMENDATION: Use Decision Tree")
            print(f"   Better performance with acceptable overfitting")
        else:
            print(f"\n🎯 RECOMMENDATION: Both models perform similarly")
            print(f"   Consider interpretability needs vs performance requirements")
        
        # Store models for later use
        self.dt_model = dt_model
        self.rf_model = rf_model
        
        return dt_model, rf_model
    
    def run_full_analysis(self):
        """
        Run the complete analysis pipeline from data loading to model comparison.
        """
        print("="*60)
        print("BEHAVIORAL BIOMETRIC SCRIPT DETECTION ANALYSIS")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Engineer features
        self.engineer_features()
        
        # Step 3: Analyze feature distributions
        self.analyze_feature_distributions()
        
        # Step 4: Build and compare models
        self.build_simple_models()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nKey Insights:")
        print("- Feature importance shows which signals matter most for detection")
        print("- Cross-validation provides robust performance estimates")
        print("- Overfitting gaps indicate model generalization capability")
        print("- Class distribution analysis reveals data balance issues")

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = BehavioralBiometricDetector()
    
    # Optional: Preview database schema first
    print("Previewing database structure...")
    detector.preview_database_schema()
    
    # Run the analysis
    detector.run_full_analysis()