#!/usr/bin/env python3
import json
import requests
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for easy modification
JSON_FILE_PATH = "LoginGPT_banks_8500_result.json"  # Path to your exported JSON file
API_URL = "http://localhost:4050/api/login_candidates"
BATCH_SIZE = 100

def preprocess_candidates(input_json):
    """
    Given an input JSON object containing landscape_analysis_result with login_page_candidates,
    this function deduplicates candidates by URL. If there are duplicates and one candidate has 
    login_page_strategy 'CRAWLING', that candidate is kept.
    
    The output is a list of dicts with keys: id, url, actions, and scan_domain.
    """
    # Extract the list of candidates from the input JSON.
    candidates = input_json.get("landscape_analysis_result", {}).get("login_page_candidates", [])
    
    # First try to extract the scan domain from scan_config, then fall back to the top-level domain.
    scan_domain = input_json.get("scan_config", {}).get("domain")
    if not scan_domain:
        scan_domain = input_json.get("domain", "")
        
    # Group candidates by their URL.
    grouped = {}
    for candidate in candidates:
        url = candidate.get("login_page_candidate", "").strip()
        if not url:
            continue  # Skip if no URL is provided.
        grouped.setdefault(url, []).append(candidate)
    
    # Process each group and choose one candidate per URL.
    output = []
    id_counter = 1
    for url, group in grouped.items():
        # Try to find a candidate with login_page_strategy == 'CRAWLING' (case-insensitive)
        chosen = None
        for candidate in group:
            if candidate.get("login_page_strategy", "").upper() == "CRAWLING":
                chosen = candidate
                break
        # If no candidate is marked as CRAWLING, select the first candidate in the group.
        if not chosen:
            chosen = group[0]
        
        # Extract the 'login_page_actions' if it exists. Otherwise, set to None.
        actions = chosen.get("login_page_actions", None)
        
        # Build the output dictionary including the scan domain.
        output.append({
            "id": id_counter,
            "url": url,
            "actions": actions,  # This will be None if not present.
            "scan_domain": scan_domain
        })
        id_counter += 1

    return output

def send_candidates_to_api(candidates, task_id):
    """
    Send the preprocessed login candidates to a remote API endpoint.
    'candidates' should be a Python list/dict that can be serialized to JSON.
    
    Returns a tuple: (success: bool, status_code: int, error_detail: str or None)
    """
    payload = json.dumps({
        "candidates": candidates, 
        "task_id": task_id
    })
    
    try:
        response = requests.post(
            API_URL,
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        if response.status_code != 200:
            error_detail = (f"API responded with status code {response.status_code}. "
                            f"Response: {response.text}")
            logger.warning("Failed to send candidates to API. %s", error_detail)
            return False, response.status_code, error_detail
    except requests.exceptions.ConnectionError as e:
        error_detail = f"Connection error: {str(e)}"
        logger.error("Connection error: API is down or unreachable. Error: %s", error_detail, exc_info=True)
        return False, 0, error_detail
    except requests.exceptions.Timeout as e:
        error_detail = f"Timeout error: {str(e)}"
        logger.error("Request timed out. API might be slow or down. Error: %s", error_detail, exc_info=True)
        return False, 0, error_detail
    except Exception as e:
        error_detail = f"Unexpected error: {str(e)}"
        logger.error("Unexpected error when sending candidates to API: %s", error_detail, exc_info=True)
        return False, 0, error_detail

    logger.info("Successfully sent login candidates to API at %s", API_URL)
    return True, response.status_code, None

def save_results_to_file(results, output_file_path):
    """Save the processing results to a JSON file."""
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved processing results to {output_file_path}")

def process_document(doc, results):
    """Process a single document and update results."""
    doc_id = doc.get("_id", {}).get("$oid", str(hash(str(doc))))  # Use MongoDB ObjectId or generate a hash
    
    task_id = doc.get("task_config", {}).get("task_id")
    if not task_id:
        logger.warning(f"No task_id found for document {doc_id}, skipping")
        results[doc_id] = {"api_status": None, "api_error": "No task_id found"}
        return False, results
        
    # Preprocess candidates
    candidates = preprocess_candidates(doc)
    logger.info(f"Preprocessed {len(candidates)} candidates for task_id: {task_id}")
    
    # Skip if no candidates found
    if not candidates:
        logger.warning(f"No candidates found for document {doc_id}, skipping")
        results[doc_id] = {"api_status": None, "api_error": "No candidates found"}
        return False, results
        
    # Send candidates to API
    success, status_code, error_detail = send_candidates_to_api(candidates, task_id)
    
    # Store the result
    results[doc_id] = {"api_status": status_code, "api_error": error_detail}
    
    return success, results

def main():
    # Check if JSON file exists
    if not os.path.isfile(JSON_FILE_PATH):
        logger.error(f"JSON file not found: {JSON_FILE_PATH}")
        return
    
    # Load the JSON file line by line (JSONL format)
    logger.info(f"Loading JSON file: {JSON_FILE_PATH}")
    documents = []
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            for line in f:
                try:
                    # Parse each line as a separate JSON object
                    doc = json.loads(line.strip())
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
            
        total_count = len(documents)
        logger.info(f"Loaded {total_count} documents from JSON file")
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return
    
    # Process documents in batches
    success_count = 0
    failure_count = 0
    results = {}  # Dictionary to store processing results for each document
    
    # Process in batches
    for i in range(0, total_count, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_success_count = 0
        
        try:
            # Process this batch
            for doc in batch:
                success, results = process_document(doc, results)
                if success:
                    batch_success_count += 1
                    success_count += 1
                else:
                    failure_count += 1
            
            logger.info(f"Processed {min(i + BATCH_SIZE, total_count)}/{total_count} documents")
            
            # Save results after each batch in case of interruption
            save_results_to_file(results, "processing_results.json")
            
            # Add a small delay to avoid overloading the API
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing batch starting at index {i}: {e}")
            # Wait a bit before retrying
            time.sleep(10)
    
    # Save final results
    save_results_to_file(results, "processing_results.json")
    
    logger.info(f"Completed processing. Success: {success_count}, Failures: {failure_count}")

if __name__ == "__main__":
    main()