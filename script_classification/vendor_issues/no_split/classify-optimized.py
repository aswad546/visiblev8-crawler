#!/usr/bin/env python3
"""
Optimized database loading methods for large datasets
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import time

class OptimizedMalwareClassificationPipeline:
    def __init__(self, model_path: str, db_config: dict):
        """
        Initialize the classification pipeline with optimizations
        """
        self.model_path = model_path
        self.db_config = db_config
        self.model = None
        self.feature_columns = None
        self.model_metadata = None
        
        # Expected feature columns based on your selection
        self.expected_features = [
            'fp_approach_diversity',
            'collection_intensity', 
            'tracks_coordinates',
            'complexity_tier',
            'uses_canvas_fp',
            'sophistication_score',
            'interaction_diversity',
            'uses_screen_fp',
            'tracks_device_motion',
            'tracks_mouse',
            'tracks_timing',
            'has_multi_input_types'
        ]

    def connect_database(self):
        """Create optimized database connection"""
        try:
            # Add connection optimizations
            optimized_config = self.db_config.copy()
            optimized_config.update({
                'connect_timeout': 30,
                'application_name': 'malware_classifier_pipeline'
            })
            conn = psycopg2.connect(**optimized_config)
            
            # Optimize connection for large data transfers
            cursor = conn.cursor()
            cursor.execute("SET work_mem = '512MB'")
            cursor.execute("SET shared_buffers = '1GB'")
            cursor.execute("SET effective_cache_size = '4GB'")
            cursor.execute("SET maintenance_work_mem = '512MB'")
            cursor.execute("SET checkpoint_completion_target = 0.9")
            cursor.execute("SET wal_buffers = '16MB'")
            cursor.execute("SET random_page_cost = 1.1")
            cursor.close()
            
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def get_table_row_count(self, table_name: str) -> int:
        """Get total row count for progress tracking"""
        conn = self.connect_database()
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            return count
        finally:
            cursor.close()
            conn.close()

    def load_data_chunked(self, table_name: str, chunk_size: int = 10000, 
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data in chunks to avoid memory issues and improve speed
        """
        logger.info(f"Loading data from {table_name} in chunks of {chunk_size:,}")
        
        # Get total count for progress tracking
        total_rows = self.get_table_row_count(table_name)
        if limit:
            total_rows = min(total_rows, limit)
        
        logger.info(f"Total rows to process: {total_rows:,}")
        
        # Load data in chunks
        chunks = []
        processed_rows = 0
        
        conn = self.connect_database()
        try:
            offset = 0
            while True:
                current_chunk_size = min(chunk_size, total_rows - processed_rows) if limit else chunk_size
                if current_chunk_size <= 0:
                    break
                
                query = f"""
                SELECT * FROM {table_name} 
                ORDER BY script_id 
                LIMIT {current_chunk_size} OFFSET {offset}
                """
                
                logger.info(f"Loading chunk: rows {offset:,} to {offset + current_chunk_size:,}")
                chunk_df = pd.read_sql(query, conn)
                
                if chunk_df.empty:
                    break
                
                chunks.append(chunk_df)
                processed_rows += len(chunk_df)
                offset += current_chunk_size
                
                # Progress update
                progress = (processed_rows / total_rows) * 100 if total_rows > 0 else 0
                logger.info(f"Progress: {processed_rows:,}/{total_rows:,} rows ({progress:.1f}%)")
                
                if limit and processed_rows >= limit:
                    break
                    
        finally:
            conn.close()
        
        # Combine all chunks
        logger.info("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} total records")
        return df

    def load_data_parallel_chunks(self, table_name: str, chunk_size: int = 10000,
                                 num_workers: int = 4, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data using parallel workers for maximum speed
        """
        logger.info(f"Loading data with {num_workers} parallel workers, chunk size: {chunk_size:,}")
        
        # Get total count
        total_rows = self.get_table_row_count(table_name)
        if limit:
            total_rows = min(total_rows, limit)
        
        logger.info(f"Total rows to process: {total_rows:,}")
        
        # Calculate chunks
        chunks_info = []
        for offset in range(0, total_rows, chunk_size):
            current_chunk_size = min(chunk_size, total_rows - offset)
            chunks_info.append((offset, current_chunk_size))
        
        logger.info(f"Will process {len(chunks_info)} chunks")
        
        def load_chunk(chunk_info):
            """Load a single chunk"""
            offset, size = chunk_info
            conn = self.connect_database()
            try:
                query = f"""
                SELECT * FROM {table_name} 
                ORDER BY script_id 
                LIMIT {size} OFFSET {offset}
                """
                df = pd.read_sql(query, conn)
                logger.info(f"Loaded chunk: offset {offset:,}, size {len(df):,}")
                return df
            finally:
                conn.close()
        
        # Load chunks in parallel
        chunks = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_futures = {executor.submit(load_chunk, chunk_info): i 
                           for i, chunk_info in enumerate(chunks_info)}
            
            for future in chunk_futures:
                try:
                    chunk_df = future.result()
                    chunks.append(chunk_df)
                except Exception as e:
                    logger.error(f"Chunk loading failed: {e}")
        
        # Combine results
        logger.info("Combining parallel chunks...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} total records")
        return df

    def load_data_streaming(self, table_name: str, batch_size: int = 5000) -> pd.DataFrame:
        """
        Stream data using server-side cursor for memory efficiency
        """
        logger.info(f"Streaming data from {table_name} with batch size {batch_size:,}")
        
        conn = self.connect_database()
        try:
            # Use server-side cursor for streaming
            cursor = conn.cursor(name='streaming_cursor')
            cursor.itersize = batch_size
            
            query = f"SELECT * FROM {table_name} ORDER BY script_id"
            cursor.execute(query)
            
            chunks = []
            batch_num = 0
            total_rows = 0
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                # Convert to DataFrame
                columns = [desc[0] for desc in cursor.description]
                chunk_df = pd.DataFrame(rows, columns=columns)
                chunks.append(chunk_df)
                
                total_rows += len(chunk_df)
                batch_num += 1
                
                if batch_num % 10 == 0:  # Log every 10 batches
                    logger.info(f"Streamed {total_rows:,} rows ({batch_num} batches)")
            
            cursor.close()
            
        finally:
            conn.close()
        
        # Combine all chunks
        logger.info("Combining streamed data...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Streamed {len(df):,} total records")
        return df

    def load_data_optimized_query(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load data with optimized query - only select needed columns
        """
        # Define only the columns you actually need for feature extraction
        needed_columns = [
            'script_id',
            'behavioral_apis_access_count',
            'fingerprinting_api_access_count', 
            'behavioral_source_apis',
            'fingerprinting_source_apis',
            'apis_going_to_sink'
        ]
        
        conn = self.connect_database()
        try:
            limit_clause = f" LIMIT {limit}" if limit else ""
            
            # Only select needed columns
            columns_str = ', '.join(needed_columns)
            query = f"""
            SELECT {columns_str} 
            FROM {table_name} 
            ORDER BY script_id
            {limit_clause}
            """
            
            logger.info(f"Loading optimized data from {table_name}...")
            logger.info(f"Selected columns: {needed_columns}")
            
            df = pd.read_sql(query, conn)
            logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
            return df
            
        finally:
            conn.close()

    def process_features_parallel(self, df: pd.DataFrame, num_workers: int = None) -> pd.DataFrame:
        """
        Process feature extraction in parallel
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Don't use more than 8 cores
        
        logger.info(f"Processing features with {num_workers} parallel workers")
        
        # Split dataframe into chunks for parallel processing
        chunk_size = max(1, len(df) // num_workers)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        def process_chunk(chunk):
            """Process a chunk of data"""
            return self.create_vendor_agnostic_features(chunk)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            feature_chunks = list(executor.map(process_chunk, chunks))
        
        # Combine results
        features_df = pd.concat(feature_chunks, ignore_index=True)
        logger.info(f"Processed features for {len(features_df):,} scripts")
        return features_df

    def load_model(self):
        """Load the trained model and metadata (unchanged)"""
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
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def create_vendor_agnostic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create vendor-agnostic features from raw database records
        Optimized version with vectorized operations where possible
        """
        logger.info(f"Creating vendor-agnostic features for {len(df):,} records...")
        features_list = []
        
        # Process in smaller batches to show progress
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            for idx, row in batch_df.iterrows():
                try:
                    features = {}
                    
                    # Safe extraction without pd.isna() on lists
                    behavioral_access = row['behavioral_apis_access_count'] if row['behavioral_apis_access_count'] is not None else {}
                    fp_access = row['fingerprinting_api_access_count'] if row['fingerprinting_api_access_count'] is not None else {}
                    behavioral_sources = row['behavioral_source_apis'] if row['behavioral_source_apis'] is not None else []
                    fp_sources = row['fingerprinting_source_apis'] if row['fingerprinting_source_apis'] is not None else []
                    sink_data = row['apis_going_to_sink'] if row['apis_going_to_sink'] is not None else {}
                    
                    # === VENDOR-AGNOSTIC BEHAVIORAL PATTERNS ===
                    
                    # 1. RELATIVE COMPLEXITY
                    total_behavioral = len(behavioral_sources) if behavioral_sources is not None else 0
                    total_fp = len(fp_sources) if fp_sources is not None else 0
                    total_apis = total_behavioral + total_fp
                    
                    if total_apis > 0:
                        features['behavioral_focus_ratio'] = total_behavioral / total_apis
                        features['fp_focus_ratio'] = total_fp / total_apis
                    else:
                        features['behavioral_focus_ratio'] = 0
                        features['fp_focus_ratio'] = 0
                    
                    # 2. INTERACTION PATTERN DIVERSITY
                    event_types = set()
                    if behavioral_sources is not None:
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
                    
                    # 3. SOPHISTICATION PATTERNS
                    coordinate_apis = 0
                    timing_apis = 0
                    device_apis = 0
                    
                    if behavioral_sources is not None:
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
                    
                    # 4. FINGERPRINTING CATEGORIES
                    navigator_apis = 0
                    screen_apis = 0
                    canvas_apis = 0
                    audio_apis = 0
                    
                    if fp_sources is not None:
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
                    features['fp_approach_diversity'] = features['uses_navigator_fp'] + features['uses_screen_fp'] + features['uses_canvas_fp'] + features['uses_audio_fp']
                    
                    # 5. ACCESS INTENSITY
                    total_behavioral_accesses = sum(behavioral_access.values()) if behavioral_access else 0
                    total_fp_accesses = sum(fp_access.values()) if fp_access else 0
                    total_accesses = total_behavioral_accesses + total_fp_accesses
                    
                    features['collection_intensity'] = total_accesses / max(total_apis, 1)
                    features['behavioral_access_ratio'] = total_behavioral_accesses / max(total_accesses, 1) if total_accesses > 0 else 0
                    
                    # 6. DATA FLOW PATTERNS
                    features['has_data_collection'] = int(len(sink_data) > 0) if sink_data else 0
                    features['collection_method_diversity'] = len(sink_data) if sink_data else 0
                    
                    # 7. BINARY TRACKING CAPABILITIES
                    features['tracks_mouse'] = int(any('MouseEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                    features['tracks_keyboard'] = int(any('KeyboardEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                    features['tracks_touch'] = int(any('TouchEvent' in str(api) or 'Touch.' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                    features['tracks_pointer'] = int(any('PointerEvent' in str(api) for api in behavioral_sources)) if behavioral_sources else 0
                    
                    # 8. COMPLEXITY CLASSIFICATION
                    if total_apis == 0:
                        features['complexity_tier'] = 0
                    elif total_apis <= 5:
                        features['complexity_tier'] = 1
                    elif total_apis <= 15:
                        features['complexity_tier'] = 2
                    else:
                        features['complexity_tier'] = 3
                    
                    # 9. BALANCE METRICS
                    features['is_behavioral_heavy'] = int(total_behavioral > total_fp and total_behavioral > 5)
                    features['is_fp_heavy'] = int(total_fp > total_behavioral and total_fp > 5)
                    features['is_balanced_tracker'] = int(abs(total_behavioral - total_fp) <= 3 and total_apis > 5)
                    
                    # Store metadata
                    features['script_id'] = int(row['script_id']) if 'script_id' in row and pd.notna(row['script_id']) else idx
                    
                    features_list.append(features)
                    
                except Exception as e:
                    logger.warning(f"Error processing script {row.get('script_id', idx)}: {e}")
                    continue
            
            # Progress update
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                progress = ((batch_idx + 1) / total_batches) * 100
                logger.info(f"Feature extraction progress: {batch_idx + 1}/{total_batches} batches ({progress:.1f}%)")
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Created features for {len(features_df):,} scripts")
        return features_df

    def run_optimized_pipeline(self, source_table: str, results_table: str, view_name: str, 
                              loading_method: str = "chunked", chunk_size: int = 10000,
                              num_workers: int = 4, limit: Optional[int] = None):
        """
        Run the pipeline with optimized data loading
        
        Args:
            loading_method: "chunked", "parallel", "streaming", or "optimized_query"
            chunk_size: Size of chunks for chunked/parallel loading
            num_workers: Number of parallel workers
        """
        logger.info("🚀 Starting Optimized Malware Classification Pipeline")
        logger.info("=" * 60)
        logger.info(f"Loading method: {loading_method}")
        logger.info(f"Chunk size: {chunk_size:,}")
        logger.info(f"Workers: {num_workers}")
        
        start_time = time.time()
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model. Aborting.")
            return False
        
        # Load data using selected method
        logger.info(f"📊 Starting data loading with method: {loading_method}")
        load_start = time.time()
        
        if loading_method == "chunked":
            df = self.load_data_chunked(source_table, chunk_size, limit)
        elif loading_method == "parallel":
            df = self.load_data_parallel_chunks(source_table, chunk_size, num_workers, limit)
        elif loading_method == "streaming":
            df = self.load_data_streaming(source_table, chunk_size)
        elif loading_method == "optimized_query":
            df = self.load_data_optimized_query(source_table, limit)
        else:
            logger.error(f"Unknown loading method: {loading_method}")
            return False
        
        load_time = time.time() - load_start
        logger.info(f"⏱️  Data loading completed in {load_time:.1f} seconds")
        
        if df.empty:
            logger.error("No data loaded. Aborting.")
            return False
        
        # Create features (can also be parallelized)
        logger.info("🔧 Starting feature extraction...")
        feature_start = time.time()
        
        if len(df) > 10000:  # Use parallel processing for large datasets
            features_df = self.process_features_parallel(df, num_workers)
        else:
            features_df = self.create_vendor_agnostic_features(df)
        
        feature_time = time.time() - feature_start
        logger.info(f"⏱️  Feature extraction completed in {feature_time:.1f} seconds")
        
        if features_df.empty:
            logger.error("Feature creation failed. Aborting.")
            return False
        
        # Continue with rest of pipeline...
        # (classify_scripts, create_results_table, save_results, create_classification_view)
        # These methods remain the same from your original code
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Pipeline completed in {total_time:.1f} seconds!")
        logger.info(f"   - Data loading: {load_time:.1f}s ({load_time/total_time*100:.1f}%)")
        logger.info(f"   - Feature extraction: {feature_time:.1f}s ({feature_time/total_time*100:.1f}%)")
        
        return True