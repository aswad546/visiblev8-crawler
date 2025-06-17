import pandas as pd
import numpy as np
import json
import pickle
import psycopg2
import psycopg2.extras
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ProductionScriptClassifier:
    """
    Production classifier to analyze label=-1 scripts using the trained Random Forest model.
    """
    
    def __init__(self, model_path, db_config=None):
        """
        Initialize the production classifier.
        
        Args:
            model_path (str): Path to the trained model pickle file
            db_config (dict): Database configuration
        """
        # Database configuration (from your original script)
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'vv8_backend',
            'user': 'vv8',
            'password': 'vv8',
            'port': 5434
        }
        
        self.table_name = 'multicore_static_info_known_companies'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load the trained model
        self.load_model(model_path)
        
        # Storage for results
        self.unlabeled_scripts = None
        self.predictions = None
        self.results_df = None
        
    def load_model(self, model_path):
        """
        Load the trained model and feature information.
        """
        print(f"📦 Loading trained model from: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            
            print(f"✅ Successfully loaded {self.model_type} model")
            print(f"📊 Model expects {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def connect_to_database(self):
        """
        Establish connection to PostgreSQL database.
        """
        try:
            connection = psycopg2.connect(**self.db_config)
            return connection
        except psycopg2.Error as e:
            print(f"❌ Error connecting to PostgreSQL database: {e}")
            return None
    
    def load_unlabeled_scripts(self):
        """
        Load all scripts with label = -1 from the database.
        """
        print("🔍 Loading unlabeled scripts (label = -1) from database...")
        
        connection = self.connect_to_database()
        if connection is None:
            raise Exception("Failed to connect to database")
        
        try:
            # Query to fetch scripts with label = -1
            query = f"SELECT * FROM {self.table_name} WHERE label = -1"
            
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(query)
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries with JSON parsing
            self.unlabeled_scripts = []
            parsing_errors = 0
            
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
                                parsing_errors += 1
                                record[field] = None
                        elif record[field] == '{}':
                            record[field] = {}
                        elif record[field] == '[]':
                            record[field] = []
                
                self.unlabeled_scripts.append(record)
            
            cursor.close()
            
            if parsing_errors > 0:
                print(f"⚠️  Warning: {parsing_errors} JSON parsing errors encountered")
            
        except psycopg2.Error as e:
            print(f"❌ Error querying database: {e}")
            raise
        finally:
            connection.close()
        
        print(f"✅ Loaded {len(self.unlabeled_scripts)} unlabeled scripts")
        return self.unlabeled_scripts
    
    def engineer_features_for_scripts(self, scripts):
        """
        Engineer features for the given scripts using the same logic as training.
        """
        print(f"🔧 Engineering features for {len(scripts)} scripts...")
        
        features_list = []
        feature_extraction_errors = 0
        
        for script in scripts:
            try:
                features = {}
                
                # Core aggregation features (same as training)
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
                
                features_list.append(features)
                
            except Exception as e:
                feature_extraction_errors += 1
                print(f"⚠️  Feature extraction error for script {script.get('script_id', 'unknown')}: {e}")
        
        if feature_extraction_errors > 0:
            print(f"⚠️  {feature_extraction_errors} feature extraction errors encountered")
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        print(f"✅ Engineered features for {len(features_df)} scripts")
        return features_df
    
    def classify_scripts(self, confidence_threshold=0.8):
        """
        Classify the unlabeled scripts using the trained model.
        """
        print(f"🤖 Classifying unlabeled scripts with {self.model_type} model...")
        print(f"📊 Using confidence threshold: {confidence_threshold}")
        
        if self.unlabeled_scripts is None:
            raise ValueError("No unlabeled scripts loaded. Call load_unlabeled_scripts() first.")
        
        # Engineer features
        features_df = self.engineer_features_for_scripts(self.unlabeled_scripts)
        
        # Extract feature matrix (same order as training)
        feature_columns = [col for col in features_df.columns if col not in ['script_id', 'script_url']]
        X = features_df[self.feature_names].values
        
        # Make predictions
        print("🔮 Making predictions...")
        probabilities = self.model.predict_proba(X)
        predictions_binary = self.model.predict(X)
        behavioral_probability = probabilities[:, 1]
        
        # Apply confidence threshold
        high_confidence_predictions = (behavioral_probability >= confidence_threshold).astype(int)
        
        # Create results DataFrame
        self.results_df = pd.DataFrame({
            'script_id': features_df['script_id'],
            'script_url': features_df['script_url'],
            'behavioral_probability': behavioral_probability,
            'predicted_label_default': predictions_binary,
            'predicted_label_high_confidence': high_confidence_predictions,
            'confidence_level': np.where(
                behavioral_probability >= confidence_threshold, 'High Confidence Behavioral',
                np.where(behavioral_probability <= (1 - confidence_threshold), 'High Confidence Normal',
                        'Uncertain')
            ),
            'max_api_aggregation_score': features_df['max_api_aggregation_score'],
            'behavioral_api_agg_count': features_df['behavioral_api_agg_count'],
            'fingerprinting_source_api_count': features_df['fingerprinting_source_api_count'],
            'unique_fp_apis': features_df['unique_fp_apis'],
            'dataflow_to_sink': features_df['dataflow_to_sink'],
            'behavioral_event_diversity': features_df['behavioral_event_diversity']
        })
        
        # Sort by behavioral probability (highest first)
        self.results_df = self.results_df.sort_values('behavioral_probability', ascending=False)
        
        # Print summary
        total_scripts = len(self.results_df)
        high_conf_behavioral = sum(self.results_df['confidence_level'] == 'High Confidence Behavioral')
        high_conf_normal = sum(self.results_df['confidence_level'] == 'High Confidence Normal')
        uncertain = sum(self.results_df['confidence_level'] == 'Uncertain')
        
        print(f"\n📊 Classification Results Summary:")
        print(f"  Total scripts analyzed: {total_scripts}")
        print(f"  High Confidence Behavioral Biometric: {high_conf_behavioral} ({high_conf_behavioral/total_scripts*100:.1f}%)")
        print(f"  High Confidence Normal: {high_conf_normal} ({high_conf_normal/total_scripts*100:.1f}%)")
        print(f"  Uncertain (need manual review): {uncertain} ({uncertain/total_scripts*100:.1f}%)")
        
        # Show top behavioral biometric candidates
        top_behavioral = self.results_df[self.results_df['confidence_level'] == 'High Confidence Behavioral'].head(10)
        if len(top_behavioral) > 0:
            print(f"\n🎯 Top {len(top_behavioral)} Behavioral Biometric Candidates:")
            for idx, row in top_behavioral.iterrows():
                print(f"  Script {row['script_id']}: {row['behavioral_probability']:.3f} probability")
                if pd.notna(row['script_url']) and len(str(row['script_url'])) > 5:
                    print(f"    URL: {str(row['script_url'])[:80]}...")
        
        return self.results_df
    
    def save_results_to_database(self):
        """
        Save classification results to a new database table.
        """
        print("\n💾 Saving classification results to database...")
        
        if self.results_df is None:
            raise ValueError("No results to save. Call classify_scripts() first.")
        
        connection = self.connect_to_database()
        if connection is None:
            print("❌ Cannot save to database - connection failed")
            return None
        
        try:
            cursor = connection.cursor()
            
            # Create results table
            table_name = f"production_classification_results_{self.timestamp}"
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                result_id SERIAL PRIMARY KEY,
                script_id INTEGER,
                script_url TEXT,
                behavioral_probability FLOAT,
                predicted_label_default INTEGER,
                predicted_label_high_confidence INTEGER,
                confidence_level TEXT,
                max_api_aggregation_score FLOAT,
                behavioral_api_agg_count FLOAT,
                fingerprinting_source_api_count FLOAT,
                unique_fp_apis FLOAT,
                dataflow_to_sink INTEGER,
                behavioral_event_diversity FLOAT,
                model_type TEXT,
                classification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            cursor.execute(create_table_sql)
            
            # Create indexes separately for better query performance
            index_queries = [
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_script_id ON {table_name}(script_id);",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_behavioral_prob ON {table_name}(behavioral_probability);",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_confidence_level ON {table_name}(confidence_level);",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_predicted_label ON {table_name}(predicted_label_high_confidence);"
            ]
            
            for index_query in index_queries:
                try:
                    cursor.execute(index_query)
                except psycopg2.Error as idx_error:
                    print(f"⚠️  Warning: Could not create index: {idx_error}")
                    # Continue even if index creation fails
            
            # Insert results
            insert_sql = f"""
            INSERT INTO {table_name} (
                script_id, script_url, behavioral_probability, 
                predicted_label_default, predicted_label_high_confidence, confidence_level,
                max_api_aggregation_score, behavioral_api_agg_count, 
                fingerprinting_source_api_count, unique_fp_apis, 
                dataflow_to_sink, behavioral_event_diversity, model_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Insert all results
            for _, row in self.results_df.iterrows():
                cursor.execute(insert_sql, (
                    int(row['script_id']) if pd.notna(row['script_id']) else None,
                    str(row['script_url']) if pd.notna(row['script_url']) else None,
                    float(row['behavioral_probability']),
                    int(row['predicted_label_default']),
                    int(row['predicted_label_high_confidence']),
                    str(row['confidence_level']),
                    float(row['max_api_aggregation_score']),
                    float(row['behavioral_api_agg_count']),
                    float(row['fingerprinting_source_api_count']),
                    float(row['unique_fp_apis']),
                    int(row['dataflow_to_sink']),
                    float(row['behavioral_event_diversity']),
                    str(self.model_type)
                ))
            
            connection.commit()
            cursor.close()
            
            print(f"✅ Results saved to table: {table_name}")
            print(f"📊 Saved {len(self.results_df)} classification results")
            
            return table_name
            
        except psycopg2.Error as e:
            print(f"❌ Error saving results: {e}")
            return None
        finally:
            connection.close()
    
    def generate_summary_queries(self, table_name):
        """
        Generate useful SQL queries for analyzing the results.
        """
        print(f"\n📝 Generating summary queries for table: {table_name}")
        
        queries = f"""
-- PRODUCTION CLASSIFICATION RESULTS ANALYSIS QUERIES
-- Generated: {datetime.now()}
-- Table: {table_name}

-- 1. HIGH CONFIDENCE BEHAVIORAL BIOMETRIC SCRIPTS (most likely to be malicious)
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    max_api_aggregation_score,
    behavioral_api_agg_count,
    fingerprinting_source_api_count
FROM {table_name}
WHERE confidence_level = 'High Confidence Behavioral'
ORDER BY behavioral_probability DESC
LIMIT 20;

-- 2. SUMMARY BY CONFIDENCE LEVEL
SELECT 
    confidence_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {table_name}), 2) as percentage,
    ROUND(AVG(behavioral_probability), 3) as avg_probability,
    ROUND(AVG(max_api_aggregation_score), 1) as avg_aggregation_score
FROM {table_name}
GROUP BY confidence_level
ORDER BY count DESC;

-- 3. TOP DOMAINS BY BEHAVIORAL BIOMETRIC PROBABILITY
SELECT 
    REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\\1') as domain,
    COUNT(*) as script_count,
    ROUND(AVG(behavioral_probability), 3) as avg_probability,
    MAX(behavioral_probability) as max_probability,
    SUM(CASE WHEN confidence_level = 'High Confidence Behavioral' THEN 1 ELSE 0 END) as high_conf_behavioral_count
FROM {table_name}
WHERE script_url IS NOT NULL AND script_url != 'Unknown'
GROUP BY REGEXP_REPLACE(script_url, '^https?://([^/]+).*', '\\1')
HAVING COUNT(*) >= 2
ORDER BY avg_probability DESC
LIMIT 25;

-- 4. UNCERTAIN CASES THAT NEED MANUAL REVIEW
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    max_api_aggregation_score,
    behavioral_api_agg_count
FROM {table_name}
WHERE confidence_level = 'Uncertain'
    AND behavioral_probability BETWEEN 0.3 AND 0.7
ORDER BY behavioral_probability DESC
LIMIT 30;

-- 5. POTENTIAL FALSE NEGATIVES (low scores but some behavioral signals)
SELECT 
    script_id, 
    script_url, 
    behavioral_probability,
    behavioral_api_agg_count,
    fingerprinting_source_api_count,
    behavioral_event_diversity
FROM {table_name}
WHERE confidence_level = 'High Confidence Normal'
    AND (behavioral_api_agg_count > 5 OR fingerprinting_source_api_count > 15)
ORDER BY behavioral_api_agg_count DESC
LIMIT 20;

-- 6. GET ORIGINAL SCRIPT DETAILS FOR HIGH CONFIDENCE BEHAVIORAL
SELECT 
    r.script_id,
    r.script_url,
    r.behavioral_probability,
    o.behavioral_source_apis,
    o.fingerprinting_source_apis,
    o.apis_going_to_sink
FROM {table_name} r
JOIN {self.table_name} o ON r.script_id = o.script_id
WHERE r.confidence_level = 'High Confidence Behavioral'
ORDER BY r.behavioral_probability DESC
LIMIT 10;
"""
        
        # Save queries to file
        queries_file = f"production_classification_queries_{self.timestamp}.sql"
        with open(queries_file, 'w') as f:
            f.write(queries)
        
        print(f"📝 Queries saved to: {queries_file}")
        
        # Print key queries to console
        print(f"\n🔍 Key Query - Top Behavioral Biometric Scripts:")
        print(f"SELECT script_id, script_url, behavioral_probability FROM {table_name}")
        print(f"WHERE confidence_level = 'High Confidence Behavioral' ORDER BY behavioral_probability DESC LIMIT 10;")
        
        return queries_file
    
    def run_production_classification(self, confidence_threshold=0.8):
        """
        Run the complete production classification pipeline.
        """
        print("🚀 " + "="*60)
        print("PRODUCTION SCRIPT CLASSIFICATION")
        print("="*60 + " 🚀")
        
        try:
            # Step 1: Load unlabeled scripts
            self.load_unlabeled_scripts()
            
            if len(self.unlabeled_scripts) == 0:
                print("❌ No unlabeled scripts found in database")
                return None
            
            # Step 2: Classify scripts
            self.classify_scripts(confidence_threshold)
            
            # Step 3: Save results to database
            table_name = self.save_results_to_database()
            
            if table_name:
                # Step 4: Generate analysis queries
                queries_file = self.generate_summary_queries(table_name)
                
                print("\n🎉 " + "="*60)
                print("CLASSIFICATION COMPLETE")
                print("="*60 + " 🎉")
                
                print(f"\n📋 Results Summary:")
                print(f"  ✅ Classified {len(self.results_df)} scripts")
                print(f"  ✅ Results saved to table: {table_name}")
                print(f"  ✅ Analysis queries: {queries_file}")
                
                # Show actionable results
                high_conf_behavioral = self.results_df[self.results_df['confidence_level'] == 'High Confidence Behavioral']
                if len(high_conf_behavioral) > 0:
                    print(f"\n🎯 ACTION REQUIRED: {len(high_conf_behavioral)} High Confidence Behavioral Biometric Scripts Found!")
                    print("   These scripts should be investigated immediately.")
                
                uncertain = self.results_df[self.results_df['confidence_level'] == 'Uncertain']
                if len(uncertain) > 0:
                    print(f"\n⚠️  MANUAL REVIEW: {len(uncertain)} Uncertain Scripts Need Review")
                
                return {
                    'results_df': self.results_df,
                    'table_name': table_name,
                    'queries_file': queries_file,
                    'high_confidence_behavioral_count': len(high_conf_behavioral),
                    'uncertain_count': len(uncertain)
                }
            else:
                print("❌ Failed to save results to database")
                return None
                
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# Usage example
if __name__ == "__main__":
    print("🚀 Starting Production Script Classification...")
    
    # Path to your trained model (update this path)
    model_path = "behavioral_biometric_analysis/best_model_20250526_023608.pkl"
    
    try:
        # Initialize classifier
        classifier = ProductionScriptClassifier(model_path)
        
        # Run classification with 80% confidence threshold
        results = classifier.run_production_classification(confidence_threshold=0.8)
        
        if results:
            print(f"\n✅ Classification completed successfully!")
            print(f"📊 Check your database table: {results['table_name']}")
            print(f"🔍 Use the queries in: {results['queries_file']}")
            
            # Quick stats
            total = len(results['results_df'])
            behavioral = results['high_confidence_behavioral_count']
            uncertain = results['uncertain_count']
            normal = total - behavioral - uncertain
            
            print(f"\n📈 Final Statistics:")
            print(f"  🔴 High Confidence Behavioral: {behavioral} ({behavioral/total*100:.1f}%)")
            print(f"  🟡 Uncertain (Manual Review): {uncertain} ({uncertain/total*100:.1f}%)")
            print(f"  🟢 High Confidence Normal: {normal} ({normal/total*100:.1f}%)")
            
        else:
            print("❌ Classification failed")
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()