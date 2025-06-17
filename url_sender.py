#!/usr/bin/env python3
"""
URL Sender Script with batching and progress tracking
- Sends URLs in batches of 10,000
- Waits 4 hours between batches
- Tracks progress to allow resuming if interrupted
"""

import requests
import json
import logging
import time
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any
import signal
import sys

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info("Received termination signal. Saving current progress before exiting...")
    # Note: Progress is already saved after each URL, so we just need to exit gracefully
    logger.info("Progress has been saved. Exiting.")
    sys.exit(0)

# Register the signal handler for common termination signals
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("url_sender.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
API_URL = "http://127.0.0.1:4050/api/login_candidates"
FILE_NAME = "urls_100k.txt"  # Default filename for URLs
DELAY = 1         # Delay in seconds between individual requests
TASK_ID = "101"   # Task ID as in the bash script
BATCH_SIZE = 10000  # Number of URLs to send before waiting
WAIT_HOURS = 2    # Hours to wait between batches
PROGRESS_FILE = "url_progress.pkl"  # File to store progress


def read_urls_from_file(file_path: str) -> List[str]:
    """
    Read URLs from a text file, one URL per line.
    
    Args:
        file_path: Path to the file containing URLs
        
    Returns:
        A list of URLs as strings
    """
    urls = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return urls
        
    try:
        with open(file_path, 'r') as file:
            # Read all lines and strip whitespace
            urls = [line.strip() for line in file.readlines()]
            # Filter out empty lines
            urls = [url for url in urls if url]
            logger.info(f"Successfully read {len(urls)} URLs from {file_path}")
            return urls
    except Exception as e:
        logger.error(f"Error reading URLs from file {file_path}: {e}")
        return []


def send_url_to_api(url: str, id_num: int) -> bool:
    """
    Send a single URL to the API endpoint using the format from the bash script.
    
    Args:
        url: The URL without http/https prefix
        id_num: ID number for this URL
        
    Returns:
        Success status (True/False)
    """
    # Check if URL already has a protocol prefix
    if url.startswith("http://") or url.startswith("https://"):
        full_url = url
        # Extract domain properly from a URL with protocol
        scan_domain = url.split('//')[1].split('/')[0]
    else:
        # Original behavior for URLs without protocol
        scan_domain = url.split('/')[0]
        full_url = f"https://{url}"
    
    # Prepare the payload exactly as in the bash script
    payload = {
        "task_id": TASK_ID,
        "candidates": [
            {
                "id": id_num,
                "url": full_url,
                "actions": None,
                "scan_domain": scan_domain
            }
        ]
    }
    
    logger.info(f"[{id_num}] Sending to {full_url}")
    
    try:
        response = requests.post(
            API_URL,
            json=payload,  # Using json parameter instead of data for proper JSON serialization
            headers={'Content-Type': 'application/json'},
        )
        
        if response.status_code != 200:
            logger.warning(f"API responded with status code {response.status_code} for URL {id_num}: {full_url}")
            logger.warning(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error for URL {id_num}: {full_url}. Error: {e}")
        return False
        
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out for URL {id_num}: {full_url}. Error: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error when sending URL {id_num}: {full_url}. Error: {e}")
        return False

    logger.info(f"Successfully sent URL {id_num}: {full_url}")
    return True


def save_progress(last_processed_index: int, successful_sends: int, failed_sends: int):
    """
    Save progress information to a file
    
    Args:
        last_processed_index: Index of the last processed URL
        successful_sends: Count of successfully sent URLs
        failed_sends: Count of failed URL sends
    """
    progress_data = {
        'last_processed_index': last_processed_index,
        'successful_sends': successful_sends,
        'failed_sends': failed_sends,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(PROGRESS_FILE, 'wb') as f:
            pickle.dump(progress_data, f)
        logger.info(f"Progress saved: processed {last_processed_index} URLs, success: {successful_sends}, failed: {failed_sends}")
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


def load_progress():
    """
    Load progress information from file
    
    Returns:
        Tuple of (last_processed_index, successful_sends, failed_sends)
        If no progress file exists, returns (0, 0, 0)
    """
    if not os.path.exists(PROGRESS_FILE):
        logger.info("No progress file found, starting from the beginning")
        return 0, 0, 0
        
    try:
        with open(PROGRESS_FILE, 'rb') as f:
            progress_data = pickle.load(f)
            
        last_idx = progress_data.get('last_processed_index', 0)
        success = progress_data.get('successful_sends', 0)
        failed = progress_data.get('failed_sends', 0)
        timestamp = progress_data.get('timestamp', 'unknown')
        
        logger.info(f"Loaded progress from {timestamp}")
        logger.info(f"Resuming from index {last_idx+1}, already processed {last_idx} URLs")
        logger.info(f"Current stats - successful: {success}, failed: {failed}")
        
        return last_idx, success, failed
        
    except Exception as e:
        logger.error(f"Error loading progress file: {e}")
        logger.info("Starting from the beginning")
        return 0, 0, 0


def main():
    """Main function to run the script"""
    logger.info(f"Starting URL Sender Script with Batching")
    logger.info(f"API endpoint: {API_URL}")
    logger.info(f"Reading URLs from: {FILE_NAME}")
    logger.info(f"Using task_id: {TASK_ID}")
    logger.info(f"Batch size: {BATCH_SIZE}, wait time between batches: {WAIT_HOURS} hours")
    logger.info(f"Progress is saved after each URL is processed for maximum reliability")
    
    # Read URLs from file
    urls = read_urls_from_file(FILE_NAME)
    
    if not urls:
        logger.error("No URLs found in the file. Exiting.")
        return
    
    logger.info(f"Found {len(urls)} URLs to process")
    
    # Load progress if exists
    start_index, successful_sends, failed_sends = load_progress()
    
    # Process URLs in batches
    total_urls = len(urls)
    
    current_index = start_index
    while current_index < total_urls:
        batch_end = min(current_index + BATCH_SIZE, total_urls)
        
        logger.info(f"Processing batch from index {current_index} to {batch_end-1}")
        batch_start_time = datetime.now()
        
        # Process this batch
        for i in range(current_index, batch_end):
            url = urls[i]
            url_id = i + 1  # Use 1-based indexing for IDs as in original script
            
            # Send the URL
            success = send_url_to_api(url, url_id)
            
            if not success:
                logger.warning(f"Failed to send URL {url_id}. Will retry once.")
                # Retry once after a short delay
                time.sleep(1)
                success = send_url_to_api(url, url_id)
                
                if not success:
                    logger.error(f"Failed to send URL {url_id} after retry.")
                    failed_sends += 1
                else:
                    successful_sends += 1
            else:
                successful_sends += 1
            
            # Save progress after every URL to ensure we can resume from exactly where we left off
            save_progress(i, successful_sends, failed_sends)
                
            # Add a delay between requests
            if i < batch_end - 1:  # If not the last URL in batch
                time.sleep(DELAY)
        
        # Update current index and save progress
        current_index = batch_end
        save_progress(current_index - 1, successful_sends, failed_sends)
        
        # If not finished with all URLs, wait before next batch
        if current_index < total_urls:
            batch_end_time = datetime.now()
            elapsed = (batch_end_time - batch_start_time).total_seconds() / 60
            
            next_batch_time = batch_end_time + timedelta(hours=WAIT_HOURS)
            
            logger.info(f"Completed batch in {elapsed:.2f} minutes")
            logger.info(f"Waiting {WAIT_HOURS} hours before next batch")
            logger.info(f"Next batch will start at {next_batch_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create a status file with current information
            with open("status.txt", "w") as status_file:
                status_file.write(f"Current progress: {current_index}/{total_urls} URLs processed\n")
                status_file.write(f"Successful sends: {successful_sends}\n")
                status_file.write(f"Failed sends: {failed_sends}\n")
                status_file.write(f"Last batch completed: {batch_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                status_file.write(f"Next batch starts: {next_batch_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Sleep for the wait period
            time.sleep(WAIT_HOURS * 3600)  # Convert hours to seconds
    
    logger.info(f"Completed sending all URLs")
    logger.info(f"Successful: {successful_sends}/{total_urls}")
    logger.info(f"Failed: {failed_sends}/{total_urls}")
    
    # Clean up progress file after successful completion
    if os.path.exists(PROGRESS_FILE):
        os.rename(PROGRESS_FILE, f"{PROGRESS_FILE}.completed")
        logger.info(f"Renamed progress file to {PROGRESS_FILE}.completed")
    
    # Update status file
    with open("status.txt", "w") as status_file:
        status_file.write(f"COMPLETED: All {total_urls} URLs processed\n")
        status_file.write(f"Successful sends: {successful_sends}\n")
        status_file.write(f"Failed sends: {failed_sends}\n")
        status_file.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # This should be caught by our signal handler, but just in case
        logger.info("Script interrupted by user. Progress has been saved and can be resumed.")
    except Exception as e:
        logger.error(f"Script encountered an error: {e}")
        logger.error("Progress until error has been saved and can be resumed.")