#!/usr/bin/env python3
import json
import logging
import psycopg2
import requests
import time
import concurrent.futures
from psycopg2.extras import execute_values

# Configuration
DB_HOST = "localhost"
DB_PORT = 5434
DB_NAME = "vv8_backend"
DB_USER = "vv8"
DB_PASSWORD = "vv8"
API_ENDPOINT = "http://localhost:8100/analyze"
BATCH_SIZE = 10000  # Increased batch size
MAX_WORKERS = 10   # Number of concurrent workers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def send_script_ids(ids):
    """
    Send the script IDs to the API endpoint.
    
    Args:
        ids (list): List of script IDs to send
        
    Returns:
        tuple: (status_code, error_detail, error)
    """
    try:
        payload = json.dumps(ids)
        logger.info(f"Sending batch of {len(ids)} IDs")
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(API_ENDPOINT, data=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            error_detail = f"POST request returned status: {response.status_code}, body: {response.text[:100]}"
            return response.status_code, error_detail, Exception(f"POST request returned status: {response.status_code}")
        
        return response.status_code, "", None
        
    except requests.exceptions.RequestException as e:
        error_detail = f"Error executing POST request: {str(e)}"
        return 0, error_detail, e
    except Exception as e:
        error_detail = f"Unexpected error: {str(e)}"
        return 0, error_detail, e

def get_batches(conn, last_id=0, batch_size=BATCH_SIZE):
    """Get batches of IDs to process that don't exist in multicore_static_info"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sf.id FROM script_flow sf
        WHERE sf.id > %s
        AND NOT EXISTS (
            SELECT 1 FROM multicore_static_info msi
            WHERE msi.script_id = sf.id
        )
        ORDER BY sf.id
        LIMIT %s
    """, (last_id, batch_size))
    
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    if ids:
        return ids, max(ids)
    return [], last_id

def update_batch_status(conn, ids, status, error_detail):
    """Update the status of a batch of IDs"""
    cursor = conn.cursor()
    if error_detail:
        cursor.execute("""
            UPDATE script_flow 
            SET api_status = %s, api_error = %s 
            WHERE id = ANY(%s)
        """, (status, error_detail, ids))
    else:
        cursor.execute("""
            UPDATE script_flow 
            SET api_status = %s, api_error = NULL 
            WHERE id = ANY(%s)
        """, (status, ids))
    
    conn.commit()
    cursor.close()
    return len(ids)

def process_batch(ids):
    """Process a single batch of IDs"""
    # Connect to the database for this worker
    worker_conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    
    try:
        # Send the IDs to the API
        status, error_detail, error = send_script_ids(ids)
        
        # Update the database
        update_batch_status(worker_conn, ids, status, error_detail)
        
        # Return success count - if status is 200, all IDs were successful
        success = len(ids) if status == 200 else 0
        return (len(ids), success)
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return (0, 0)
    finally:
        worker_conn.close()

def check_indices(conn):
    """Check if the necessary indices exist and log warnings if they don't"""
    cursor = conn.cursor()
    
    # Check for index on script_id in multicore_static_info
    cursor.execute("""
        SELECT indexname FROM pg_indexes 
        WHERE tablename = 'multicore_static_info' 
        AND indexdef LIKE '%script_id%'
    """)
    
    if not cursor.fetchone():
        logger.warning("No index found on multicore_static_info.script_id column. "
                      "Query performance may be slow. Consider creating an index manually.")
    
    cursor.close()

def main():
    logger.info("Script ID Processor - Sending IDs missing from multicore_static_info")
    
    # Connect to the database
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = False
        logger.info("Successfully connected to the database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    try:
        # Check if indices exist but don't create them
        check_indices(conn)
        
        # Get a rough estimate of total count
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM script_flow sf
                WHERE NOT EXISTS (
                    SELECT 1 FROM multicore_static_info msi
                    WHERE msi.script_id = sf.id
                )
                LIMIT 1000
            ) AS subquery
        """)
        estimate = cursor.fetchone()[0]
        if estimate == 1000:
            logger.info("Over 1000 records to process, using fast batch processing")
        else:
            logger.info(f"Found approximately {estimate} records to process")
        cursor.close()
        
        # Process records in parallel batches
        processed_count = 0
        success_count = 0
        last_id = 0
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while True:
                # Get next batch of IDs
                ids, new_last_id = get_batches(conn, last_id, BATCH_SIZE)
                
                if not ids:
                    logger.info("No more records to process")
                    break
                
                if new_last_id == last_id:
                    logger.info("No progress made, stopping")
                    break
                    
                last_id = new_last_id
                
                # Process batch in thread pool
                future = executor.submit(process_batch, ids)
                
                # Wait for this batch to complete before proceeding to the next one
                # This ensures we don't overload the database with many simultaneous connections
                try:
                    count, success_count_batch = future.result(timeout=60)  # 60 second timeout
                    processed_count += count
                    success_count += success_count_batch
                        
                    # Calculate processing rate
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    
                    logger.info(f"Processed {processed_count} records. Success: {success_count}. "
                               f"Rate: {rate:.2f} records/sec. Last ID: {last_id}")
                    
                except concurrent.futures.TimeoutError:
                    logger.error("Timeout processing batch")
        
        elapsed = time.time() - start_time
        logger.info(f"Script completed in {elapsed:.2f} seconds. "
                   f"Processed {processed_count} records. "
                   f"Success: {success_count}")
    
    except Exception as e:
        logger.error(f"Error in main processing loop: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()