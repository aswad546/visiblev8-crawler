#!/usr/bin/env python3
"""
Malware Classification Pipeline for SQL Database
Loads trained model, processes database records, and saves classification results
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import pickle
import joblib
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'malware_classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MalwareClassificationPipeline:
    def __init__(self, model_path: str, db_config: dict):
        """
        Initialize the classification pipeline
        
        Args:
            model_path: Path to the trained model file
            db_config: Database configuration dictionary
        """
        self.model_path = model_path
        self.db_config = db_config
        self.model = None
        self.feature_columns = None
        self.model_metadata = None
        
        # Expected feature columns based on your binary model
        self.expected_features = [
            'complexity_tier',
            'uses_screen_fp',
            'fp_approach_diversity',
            'total_aggregation_count',
            'collection_intensity',
            'interaction_diversity',
            'tracks_coordinates',
            'max_api_aggregation_score',
            'sophistication_score',
            'tracks_device_motion',
            'behavioral_api_agg_count',
            'uses_canvas_fp'
        ]
        
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Try to load as full package first
            try:
                with open(self.model_path, 'rb') as f:
                    package = pickle.load(f)
                    
                if isinstance(package, dict) and 'model' in package:
                    self.model = package['model']
                    self.model_metadata = package.get('metadata', {})
                    self.feature_columns = package.get('feature_columns', self.expected_features)
                    logger.info("Loaded full model package")
                else:
                    self.model = package
                    self.feature_columns = self.expected_features
                    logger.info("Loaded model only")
                    
            except:
                # Try joblib format
                package = joblib.load(self.model_path)
                if isinstance(package, dict) and 'model' in package:
                    self.model = package['model']
                    self.model_metadata = package.get('metadata', {})
                    self.feature_columns = package.get('feature_columns', self.expected_features)
                else:
                    self.model = package
                    self.feature_columns = self.expected_features
                    
            logger.info(f"Model loaded successfully. Expected features: {len(self.feature_columns)}")
            logger.info(f"Feature columns: {self.feature_columns}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def connect_database(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def is_null_or_empty(self, value):
        """Properly check if a value is null/empty, handling pandas arrays"""
        if value is None:
            return True
        try:
            # For pandas arrays, check if all elements are null
            if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                if len(value) == 0:
                    return True
                # Check if it's a pandas null check result array
                if hasattr(value, 'dtype') and 'bool' in str(value.dtype):
                    return False  # It's a boolean array from pd.isna(), so original value exists
                return False
            return pd.isna(value)
        except:
            return value is None

    def create_vendor_agnostic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create vendor-agnostic features from raw database records
        Updated to match the binary model's feature set
        """
        logger.info("Creating vendor-agnostic features...")
        features_list = []
        
        for idx, row in df.iterrows():
            try:
                features = {}
                
                # Safe extraction
                behavioral_access = row['behavioral_apis_access_count'] if row['behavioral_apis_access_count'] is not None else {}
                fp_access = row['fingerprinting_api_access_count'] if row['fingerprinting_api_access_count'] is not None else {}
                behavioral_sources = row['behavioral_source_apis'] if row['behavioral_source_apis'] is not None else []
                fp_sources = row['fingerprinting_source_apis'] if row['fingerprinting_source_apis'] is not None else []
                sink_data = row['apis_going_to_sink'] if row['apis_going_to_sink'] is not None else {}

                # === AGGREGATION FEATURES ===
                max_agg = row.get('max_api_aggregation_score', 0)
                behavioral_agg = row.get('behavioral_api_agg_count', 0)
                fp_agg = row.get('fp_api_agg_count', 0)

                max_agg = 0 if (pd.isna(max_agg) or max_agg == -1) else max_agg
                behavioral_agg = 0 if (pd.isna(behavioral_agg) or behavioral_agg == -1) else behavioral_agg
                fp_agg = 0 if (pd.isna(fp_agg) or fp_agg == -1) else fp_agg

                features['max_api_aggregation_score'] = max_agg
                features['behavioral_api_agg_count'] = behavioral_agg
                features['fp_api_agg_count'] = fp_agg
                features['total_aggregation_count'] = behavioral_agg + fp_agg
                features['has_aggregation'] = int(max_agg > 0)

                total_agg = behavioral_agg + fp_agg
                if total_agg > 0:
                    features['behavioral_agg_ratio'] = behavioral_agg / total_agg
                    features['fp_agg_ratio'] = fp_agg / total_agg
                else:
                    features['behavioral_agg_ratio'] = 0
                    features['fp_agg_ratio'] = 0

                features['has_behavioral_aggregation'] = int(behavioral_agg > 0)
                features['has_fp_aggregation'] = int(fp_agg > 0)
                features['has_both_aggregation_types'] = int(behavioral_agg > 0 and fp_agg > 0)

                # === BEHAVIORAL FOCUS RATIOS ===
                total_behavioral = len(behavioral_sources) if behavioral_sources is not None else 0
                total_fp = len(fp_sources) if fp_sources is not None else 0
                total_apis = total_behavioral + total_fp

                if total_apis > 0:
                    features['behavioral_focus_ratio'] = total_behavioral / total_apis
                    features['fp_focus_ratio'] = total_fp / total_apis
                else:
                    features['behavioral_focus_ratio'] = 0
                    features['fp_focus_ratio'] = 0

                # === INTERACTION PATTERN DIVERSITY ===
                event_types = set()
                for api in behavioral_sources:
                    api_str = str(api)
                    if 'MouseEvent' in api_str:
                        event_types.add('mouse')
                    elif 'KeyboardEvent' in api_str:
                        event_types.add('keyboard')
                    elif 'TouchEvent' in api_str or 'Touch.' in api_str:
                        event_types.add('touch')
                    elif 'PointerEvent' in api_str:
                        event_types.add('pointer')
                    elif 'DeviceMotion' in api_str or 'DeviceOrientation' in api_str:
                        event_types.add('device')
                    elif 'WheelEvent' in api_str:
                        event_types.add('wheel')
                    elif 'FocusEvent' in api_str:
                        event_types.add('focus')

                features['interaction_diversity'] = len(event_types)
                features['has_multi_input_types'] = int(len(event_types) >= 3)

                # === SOPHISTICATION PATTERNS ===
                coordinate_apis = 0
                timing_apis = 0
                device_apis = 0

                for api in behavioral_sources:
                    api_str = str(api)
                    if any(coord in api_str for coord in ['clientX', 'clientY', 'screenX', 'screenY', 'pageX', 'pageY']):
                        coordinate_apis += 1
                    if any(timing in api_str for timing in ['timeStamp', 'interval']):
                        timing_apis += 1
                    if 'DeviceMotion' in api_str or 'DeviceOrientation' in api_str:
                        device_apis += 1

                features['tracks_coordinates'] = int(coordinate_apis > 0)
                features['tracks_timing'] = int(timing_apis > 0)
                features['tracks_device_motion'] = int(device_apis > 0)
                features['sophistication_score'] = features['tracks_coordinates'] + features['tracks_timing'] + features['tracks_device_motion']

                # === FINGERPRINTING CATEGORIES ===
                navigator_apis = 0
                screen_apis = 0
                canvas_apis = 0
                audio_apis = 0

                for api in fp_sources:
                    api_str = str(api)
                    if 'Navigator.' in api_str:
                        navigator_apis += 1
                    if 'Screen.' in api_str:
                        screen_apis += 1
                    if 'Canvas' in api_str or 'WebGL' in api_str:
                        canvas_apis += 1
                    if 'Audio' in api_str:
                        audio_apis += 1

                features['uses_navigator_fp'] = int(navigator_apis > 0)
                features['uses_screen_fp'] = int(screen_apis > 0)
                features['uses_canvas_fp'] = int(canvas_apis > 0)
                features['uses_audio_fp'] = int(audio_apis > 0)
                features['fp_approach_diversity'] = (
                    features['uses_navigator_fp'] + features['uses_screen_fp'] +
                    features['uses_canvas_fp'] + features['uses_audio_fp']
                )

                # === ACCESS INTENSITY ===
                total_behavioral_accesses = sum(behavioral_access.values()) if behavioral_access else 0
                total_fp_accesses = sum(fp_access.values()) if fp_access else 0
                total_accesses = total_behavioral_accesses + total_fp_accesses

                features['collection_intensity'] = total_accesses / max(total_apis, 1)
                features['behavioral_access_ratio'] = total_behavioral_accesses / max(total_accesses, 1) if total_accesses > 0 else 0

                # === DATA FLOW PATTERNS ===
                features['has_data_collection'] = int(len(sink_data) > 0) if sink_data else 0
                features['collection_method_diversity'] = len(sink_data) if sink_data else 0

                # === BINARY TRACKING CAPABILITIES ===
                features['tracks_mouse'] = int(any('MouseEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                features['tracks_keyboard'] = int(any('KeyboardEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                features['tracks_touch'] = int(any('TouchEvent' in str(api) or 'Touch.' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                features['tracks_pointer'] = int(any('PointerEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0

                # === COMPLEXITY CLASSIFICATION ===
                if total_apis == 0:
                    features['complexity_tier'] = 0
                elif total_apis <= 5:
                    features['complexity_tier'] = 1
                elif total_apis <= 15:
                    features['complexity_tier'] = 2
                else:
                    features['complexity_tier'] = 3

                # === BALANCE METRICS ===
                features['is_behavioral_heavy'] = int(total_behavioral > total_fp and total_behavioral > 5)
                features['is_fp_heavy'] = int(total_fp > total_behavioral and total_fp > 5)
                features['is_balanced_tracker'] = int(abs(total_behavioral - total_fp) <= 3 and total_apis > 5)

                # === METADATA ===
                features['script_id'] = int(row['script_id']) if 'script_id' in row and pd.notna(row['script_id']) else idx
                
                features_list.append(features)
                
            except Exception as e:
                logger.warning(f"Error processing script {row.get('script_id', idx)}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Created features for {len(features_df)} scripts")
        
        # Log which features are present vs expected
        missing_features = set(self.feature_columns) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}")
        
        extra_features = set(features_df.columns) - set(self.feature_columns) - {'script_id'}
        if extra_features:
            logger.info(f"Extra features created: {extra_features}")
        
        return features_df

    def load_data_from_database(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load data from specified table"""
        conn = self.connect_database()
        try:
            limit_clause = f" LIMIT {limit}" if limit else ""
            query = f"SELECT * FROM {table_name}{limit_clause}"
            
            logger.info(f"Loading data from {table_name}...")
            df = pd.read_sql(query, conn)
            logger.info(f"Loaded {len(df)} records from {table_name}")
            return df
            
        finally:
            conn.close()

    def classify_scripts(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Classify scripts using the trained model"""
        logger.info("Running classification...")
        
        # Ensure we have all required features
        X = features_df[self.feature_columns].copy()
        
        # Handle missing features by filling with zeros
        for col in self.feature_columns:
            if col not in X.columns:
                logger.warning(f"Missing feature {col}, filling with zeros")
                X[col] = 0
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create results dataframe
        results_df = features_df[['script_id']].copy()
        results_df['prediction'] = predictions
        results_df['behavioral_biometric_probability'] = probabilities[:, 1]  # Probability of behavioral biometric class
        results_df['benign_probability'] = probabilities[:, 0]   # Probability of benign class
        results_df['classification_timestamp'] = datetime.now()
        
        # Add confidence level
        max_probs = np.maximum(probabilities[:, 0], probabilities[:, 1])
        results_df['confidence'] = max_probs
        results_df['confidence_level'] = pd.cut(
            max_probs, 
            bins=[0, 0.6, 0.8, 1.0], 
            labels=['low', 'medium', 'high']
        )
        
        logger.info(f"Classification complete. {predictions.sum()} scripts classified as behavioral biometric")
        return results_df

    def create_results_table(self, table_name: str):
        """Create table to store classification results"""
        conn = self.connect_database()
        try:
            cursor = conn.cursor()
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                script_id INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                behavioral_biometric_probability FLOAT NOT NULL,
                benign_probability FLOAT NOT NULL,
                confidence FLOAT NOT NULL,
                confidence_level VARCHAR(10),
                classification_timestamp TIMESTAMP NOT NULL,
                model_version VARCHAR(100),
                UNIQUE(script_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_{table_name}_script_id ON {table_name}(script_id);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_prediction ON {table_name}(prediction);
            CREATE INDEX IF NOT EXISTS idx_{table_name}_probability ON {table_name}(behavioral_biometric_probability);
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            logger.info(f"Results table {table_name} created successfully")
            
        finally:
            cursor.close()
            conn.close()

    def save_results(self, results_df: pd.DataFrame, table_name: str, batch_size: int = 1000):
        """Save classification results to database"""
        conn = self.connect_database()
        try:
            cursor = conn.cursor()
            
            # Add model version
            model_version = getattr(self.model_metadata, 'training_timestamp', 'unknown') if self.model_metadata else 'unknown'
            results_df['model_version'] = model_version
            
            # Prepare data for insertion
            columns = ['script_id', 'prediction', 'behavioral_biometric_probability', 'benign_probability', 
                      'confidence', 'confidence_level', 'classification_timestamp', 'model_version']
            
            # Insert in batches to handle large datasets
            total_rows = len(results_df)
            for i in range(0, total_rows, batch_size):
                batch_df = results_df.iloc[i:i+batch_size]
                values = [tuple(row[col] for col in columns) for _, row in batch_df.iterrows()]
                
                insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (script_id) DO UPDATE SET
                    prediction = EXCLUDED.prediction,
                    behavioral_biometric_probability = EXCLUDED.behavioral_biometric_probability,
                    benign_probability = EXCLUDED.benign_probability,
                    confidence = EXCLUDED.confidence,
                    confidence_level = EXCLUDED.confidence_level,
                    classification_timestamp = EXCLUDED.classification_timestamp,
                    model_version = EXCLUDED.model_version
                """
                
                execute_values(cursor, insert_sql, values)
                conn.commit()
                
                logger.info(f"Saved batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1}")
            
            logger.info(f"All results saved to {table_name}")
            
        finally:
            cursor.close()
            conn.close()

    def create_classification_view(self, source_table: str, results_table: str, view_name: str):
        """Create a view that joins original data with classification results"""
        conn = self.connect_database()
        try:
            cursor = conn.cursor()
            
            view_sql = f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT 
                s.*,
                r.prediction,
                r.behavioral_biometric_probability,
                r.benign_probability,
                r.confidence,
                r.confidence_level,
                r.classification_timestamp,
                r.model_version,
                CASE 
                    WHEN r.prediction = 1 THEN 'behavioral_biometric'
                    WHEN r.prediction = 0 THEN 'benign'
                    ELSE 'unclassified'
                END as classification_label
            FROM {source_table} s
            LEFT JOIN {results_table} r ON s.script_id = r.script_id;
            """
            
            cursor.execute(view_sql)
            conn.commit()
            logger.info(f"Classification view {view_name} created successfully")
            
        finally:
            cursor.close()
            conn.close()

    def run_pipeline(self, source_table: str, results_table: str, view_name: str, limit: Optional[int] = None):
        """Run the complete classification pipeline"""
        logger.info("🚀 Starting Behavioral Biometric Classification Pipeline")
        logger.info("=" * 60)
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model. Aborting.")
            return False
        
        # Load data
        df = self.load_data_from_database(source_table, limit)
        if df.empty:
            logger.error("No data loaded. Aborting.")
            return False
        
        # Create features
        features_df = self.create_vendor_agnostic_features(df)
        if features_df.empty:
            logger.error("Feature creation failed. Aborting.")
            return False
        
        # Classify
        results_df = self.classify_scripts(features_df)
        
        # Create results table
        self.create_results_table(results_table)
        
        # Save results
        self.save_results(results_df, results_table)
        
        # Create view
        self.create_classification_view(source_table, results_table, view_name)
        
        # Summary statistics
        total_scripts = len(results_df)
        behavioral_biometric_count = int((results_df['prediction'] == 1).sum())
        benign_count = int((results_df['prediction'] == 0).sum())
        avg_confidence = results_df['confidence'].mean()
        
        logger.info("\n📊 CLASSIFICATION SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Total scripts processed: {total_scripts:,}")
        logger.info(f"Classified as behavioral biometric: {behavioral_biometric_count:,} ({behavioral_biometric_count/total_scripts*100:.1f}%)")
        logger.info(f"Classified as benign: {benign_count:,} ({benign_count/total_scripts*100:.1f}%)")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info(f"High confidence predictions: {(results_df['confidence_level'] == 'high').sum():,}")
        logger.info(f"Results saved to table: {results_table}")
        logger.info(f"Classification view created: {view_name}")
        
        # Additional breakdown by confidence level
        confidence_breakdown = results_df['confidence_level'].value_counts()
        logger.info(f"\nConfidence Level Breakdown:")
        for level, count in confidence_breakdown.items():
            logger.info(f"  {level}: {count:,} ({count/total_scripts*100:.1f}%)")
        
        # Behavioral biometric breakdown by confidence
        bb_by_confidence = results_df[results_df['prediction'] == 1]['confidence_level'].value_counts()
        logger.info(f"\nBehavioral Biometric by Confidence Level:")
        for level, count in bb_by_confidence.items():
            pct_of_bb = count/behavioral_biometric_count*100 if behavioral_biometric_count > 0 else 0
            logger.info(f"  {level}: {count:,} ({pct_of_bb:.1f}% of behavioral biometric)")
        
        return True


# Main execution
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        "host": "localhost",
        "port": 5434,
        "database": "vv8_backend", 
        "user": "vv8",
        "password": "vv8"
    }
    
    # File paths - UPDATE THESE TO YOUR ACTUAL PATHS
    MODEL_PATH = "final_models/behavioral_biometric_classifier_binary_20250711_161434.pkl"  # Update with your actual model file
    SOURCE_TABLE = "multicore_static_info_100k_login"
    RESULTS_TABLE = "rf_binary_100k_login_crux"
    VIEW_NAME = "rf_binary_view_100k_login_crux"
    
    # Optional: limit number of records for testing (remove for full dataset)
    LIMIT = None  # Set to a number like 1000 for testing, None for all records
    
    try:
        # Initialize pipeline
        pipeline = MalwareClassificationPipeline(MODEL_PATH, DB_CONFIG)
        
        # Run classification
        success = pipeline.run_pipeline(
            source_table=SOURCE_TABLE,
            results_table=RESULTS_TABLE, 
            view_name=VIEW_NAME,
            limit=LIMIT
        )
        
        if success:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline failed!")
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise