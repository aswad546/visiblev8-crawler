#!/usr/bin/env python3
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_login_candidates(json_file_path):
    """
    Count the total number of login candidates in the JSONL file
    and provide some basic statistics.
    """
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    
    try:
        # Stats to track
        total_documents = 0
        documents_with_candidates = 0
        total_candidates = 0
        domain_candidate_counts = {}
        strategy_counts = {}
        
        # Read the file line by line
        with open(json_file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse each line as a separate JSON object
                    doc = json.loads(line.strip())
                    total_documents += 1
                    
                    # Get domain
                    domain = doc.get("domain", "unknown")
                    
                    # Extract candidates
                    candidates = doc.get("landscape_analysis_result", {}).get("login_page_candidates", [])
                    num_candidates = len(candidates)
                    
                    # Update stats
                    if num_candidates > 0:
                        documents_with_candidates += 1
                        total_candidates += num_candidates
                        
                        # Update domain stats
                        domain_candidate_counts[domain] = num_candidates
                        
                        # Update strategy stats
                        for candidate in candidates:
                            strategy = candidate.get("login_page_strategy", "unknown")
                            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
                
        # Print results
        logger.info(f"Total documents processed: {total_documents}")
        logger.info(f"Documents with login candidates: {documents_with_candidates} ({documents_with_candidates/total_documents*100:.2f}%)")
        logger.info(f"Total login candidates: {total_candidates}")
        logger.info(f"Average candidates per document with candidates: {total_candidates/documents_with_candidates if documents_with_candidates else 0:.2f}")
        
        # Display strategy distribution
        logger.info("\nStrategy distribution:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {strategy}: {count} ({count/total_candidates*100:.2f}%)")
        
        # Display top 10 domains with most candidates
        if domain_candidate_counts:
            logger.info("\nTop 10 domains with most login candidates:")
            for domain, count in sorted(domain_candidate_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {domain}: {count}")
        
        return total_candidates, documents_with_candidates, total_documents
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return 0, 0, 0

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "LoginGPT_banks_8500_result.json"
    
    logger.info(f"Counting login candidates in: {file_path}")
    
    candidates, docs_with_candidates, total_docs = count_login_candidates(file_path)
    
    # One-line summary for quick reference
    print(f"\nSummary: {candidates} total login candidates found across {docs_with_candidates}/{total_docs} documents.")