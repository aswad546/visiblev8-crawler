import subprocess as sp
import os
import os.path
import glob
import shutil
import time
import multiprocessing as m
from typing import List, Optional, TypedDict
from bson import ObjectId
import json
import logging


from vv8_worker.app import celery_app
from vv8_worker.config.mongo_config import GridFSTask

dirname = os.path.dirname(__file__)

# Configure logging to include timestamps and log levels
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class CrawlerConfig(TypedDict):
    disable_screenshot: bool
    disable_har: bool
    disable_artifact_collection: bool
    crawler_args: List[str]
    mongo_id: Optional[str]
    delete_log_after_parsing: bool


def remove_entry(filepath):
    if os.path.isdir(filepath):
        shutil.rmtree(filepath)
    else:
        os.remove(filepath)


@celery_app.task(base=GridFSTask, bind=True, name='vv8_worker.process_url')
def process_url(self, url: str, submission_id: str, config: CrawlerConfig, actions=None, scan_domain=None):
    """
    Process a URL using the crawler and handle the results.
    
    Args:
        url: The URL to crawl
        submission_id: Unique ID for this submission
        config: Configuration settings for the crawler
        actions: Optional actions to perform during crawling
        scan_domain: Optional domain to scan (used with actions)
    """
    logger = logging.getLogger(f"vv8_worker.{submission_id}")
    
    # Create a custom logging function that includes URL with each log
    def log_with_url(level, message, exc_info=None):
        log_msg = f"[URL:{url}] [ID:{submission_id}] {message}"
        if level == "INFO":
            logger.info(log_msg, exc_info=exc_info)
        elif level == "ERROR":
            logger.error(log_msg, exc_info=exc_info)
        elif level == "WARNING":
            logger.warning(log_msg, exc_info=exc_info)
        elif level == "DEBUG":
            logger.debug(log_msg, exc_info=exc_info)
        else:
            logger.info(log_msg, exc_info=exc_info)
    
    log_with_url("INFO", f"Starting process_url task")
    start = time.perf_counter()
    
    # Log configuration settings
    log_with_url("INFO", f"Configuration: {json.dumps({k: v for k, v in config.items() if k != 'mongo_id'})}")
    if actions:
        log_with_url("INFO", f"Actions provided: {json.dumps(actions)}")
    
    crawler_path = os.path.join('/app', 'node/crawler.js')
    if not os.path.isfile(crawler_path):
        error_msg = f'Crawler script cannot be found or does not exist. Expected path: {crawler_path}'
        log_with_url("ERROR", error_msg)
        raise Exception(error_msg)
    
    base_wd_path = os.path.join(dirname, 'raw_logs')
    if not os.path.isdir(base_wd_path):
        log_with_url("INFO", f"Creating base working directory: {base_wd_path}")
        os.mkdir(base_wd_path)
    
    # Create working directory for this task
    wd_path = os.path.join(base_wd_path, submission_id)
    if os.path.exists(wd_path):
        log_with_url("INFO", f"Cleaning existing working directory: {wd_path}")
        # Remove all files from working directory
        for entry in glob.glob(os.path.join(wd_path, '*')):
            remove_entry(entry)
    else:
        log_with_url("INFO", f"Creating working directory: {wd_path}")
        os.mkdir(wd_path)
    
    with os.scandir(wd_path) as dir_it:
        for entry in dir_it:
            error_msg = 'Working directory should be empty'
            log_with_url("ERROR", error_msg)
            raise Exception(error_msg)
    
    # Run crawler
    self.update_state(state='PROGRESS', meta={'status': 'Running crawler'})
    if config['disable_screenshot']:
        config['crawler_args'].append('--disable-screenshot')
    
    log_with_url("INFO", f"Crawler arguments: {config['crawler_args']}")
    
    # Determine the crawl URL and parameters
    crawl_url = url if not actions else f"http://{scan_domain}"
    log_with_url("INFO", f"Using crawl URL: {crawl_url}" + (" (from scan_domain)" if actions else ""))
    
    # Prepare the command
    cmd = [
        'node',
        crawler_path,
        'visit',
        url,
        crawl_url,
        str(submission_id)
    ] + config['crawler_args'] + (['--actions', json.dumps(actions)] if actions else [])
    
    log_with_url("INFO", f"Executing command: {' '.join(cmd)}")
    
    ret_code = -1
    timeoutCheck = False
    try:
        crawler_proc = sp.Popen(cmd, cwd=wd_path)
        log_with_url("INFO", f"Crawler process started with PID: {crawler_proc.pid}")
        
        log_with_url("INFO", f"Waiting for crawler to complete with timeout: {config.get('hard_timeout', 'unknown')}s")
        ret_code = crawler_proc.wait(timeout=config.get('hard_timeout', 600))  # Default 10min timeout
        log_with_url("INFO", f"Crawler process completed with return code: {ret_code}")
    except sp.TimeoutExpired:
        log_with_url("WARNING", f"Browser process forcibly killed due to timeout being exceeded")
        sp.run(['pkill', '-P', f'{crawler_proc.pid}'])
        crawler_proc.kill()
        timeoutCheck = True
    except Exception as e:
        log_with_url("ERROR", f"Exception running crawler process: {str(e)}", exc_info=True)
        if crawler_proc and crawler_proc.poll() is None:
            crawler_proc.kill()
    
    # Retry logic for failed actions-based crawl
    if (timeoutCheck or ret_code != 0) and actions:
        log_with_url("INFO", f"Initial crawl with actions failed. Retrying with direct URL: {url}")
        
        # Reset actions to None for direct URL crawl
        retry_cmd = [
            'node',
            crawler_path,
            'visit',
            url,
            url,  # Direct URL this time
            str(submission_id)
        ] + config['crawler_args']
        
        log_with_url("INFO", f"Executing retry command: {' '.join(retry_cmd)}")
        
        timeoutCheck = False
        try:
            crawler_proc = sp.Popen(retry_cmd, cwd=wd_path)
            log_with_url("INFO", f"Retry crawler process started with PID: {crawler_proc.pid}")
            
            ret_code = crawler_proc.wait(timeout=config.get('hard_timeout', 600))
            log_with_url("INFO", f"Retry crawler process completed with return code: {ret_code}")
        except sp.TimeoutExpired:
            log_with_url("WARNING", f"Retry browser process forcibly killed due to timeout being exceeded")
            sp.run(['pkill', '-P', f'{crawler_proc.pid}'])
            crawler_proc.kill()
            timeoutCheck = True
        except Exception as e:
            log_with_url("ERROR", f"Exception running retry crawler process: {str(e)}", exc_info=True)
            if crawler_proc and crawler_proc.poll() is None:
                crawler_proc.kill()
    
    # Processing artifacts
    self.update_state(state='PROGRESS', meta={'status': 'Uploading artifacts to mongodb'})
    
    # Process screenshots
    screenshot_ids = []
    screenshot_files = glob.glob(f'{wd_path}/*.png')
    log_with_url("INFO", f"Found {len(screenshot_files)} screenshot files")
    
    for screenshot in screenshot_files:
        if not config['disable_screenshot']:
            screenshot_name = screenshot.split('/')[-1]
            dest_path = f"/app/screenshots/{screenshot_name}"
            log_with_url("INFO", f"Copying screenshot to {dest_path}")
            
            shutil.copy(screenshot, dest_path)
            
            if not config['disable_artifact_collection']:
                try:
                    file_id = self.gridfs.upload_from_stream(
                        screenshot,
                        open(screenshot, 'rb'),
                        chunk_size_bytes=1024 * 1024,
                        metadata={"contentType": "image/png"}
                    )
                    screenshot_ids.append(file_id)
                    log_with_url("INFO", f"Uploaded screenshot to GridFS with ID: {file_id}")
                except Exception as e:
                    log_with_url("ERROR", f"Failed to upload screenshot {screenshot_name} to GridFS: {str(e)}", exc_info=True)
        
        try:
            os.remove(screenshot)
            log_with_url("INFO", f"Removed temporary screenshot file: {screenshot}")
        except Exception as e:
            log_with_url("WARNING", f"Failed to remove temporary screenshot file {screenshot}: {str(e)}")
    
    # Process HAR files
    har_ids = []
    har_files = glob.glob(f"{wd_path}/*.har")
    log_with_url("INFO", f"Found {len(har_files)} HAR files")
    
    for har in har_files:
        if not config['disable_har']:
            har_name = har.split('/')[-1]
            dest_path = f"/app/har/{har_name}"
            log_with_url("INFO", f"Copying HAR file to {dest_path}")
            
            shutil.copy(har, dest_path)
            
            if not config['disable_artifact_collection']:
                try:
                    file_id = self.gridfs.upload_from_stream(
                        har,
                        open(har, 'rb'),
                        chunk_size_bytes=1024 * 1024,
                        metadata={"contentType": "text/plain"}
                    )
                    har_ids.append(file_id)
                    log_with_url("INFO", f"Uploaded HAR file to GridFS with ID: {file_id}")
                except Exception as e:
                    log_with_url("ERROR", f"Failed to upload HAR file {har_name} to GridFS: {str(e)}", exc_info=True)
            
            try:
                os.remove(har)
                log_with_url("INFO", f"Removed temporary HAR file: {har}")
            except Exception as e:
                log_with_url("WARNING", f"Failed to remove temporary HAR file {har}: {str(e)}")
    
    # Process log files
    log_ids = []
    log_files = glob.glob(os.path.join(wd_path, 'vv8*.log'))
    log_with_url("INFO", f"Found {len(log_files)} VV8 log files")
    
    for entry in log_files:
        if not config['disable_artifact_collection']:
            try:
                file_id = self.gridfs.upload_from_stream(
                    entry,
                    open(entry, 'rb'),
                    chunk_size_bytes=1024 * 1024,
                    metadata={"contentType": "text/plain"}
                )
                log_ids.append(file_id)
                log_with_url("INFO", f"Uploaded log file to GridFS with ID: {file_id}")
            except Exception as e:
                log_with_url("ERROR", f"Failed to upload log file {entry} to GridFS: {str(e)}", exc_info=True)
    
    # Update MongoDB with artifact IDs
    if not config['disable_artifact_collection'] and config.get('mongo_id'):
        try:
            update_result = self.mongo['vv8_logs'].update_one(
                {'_id': ObjectId(config['mongo_id'])},
                {'$set': {
                    'screenshot_ids': screenshot_ids,
                    'har_ids': har_ids,
                    'log_ids': log_ids
                }}
            )
            log_with_url("INFO", f"Updated MongoDB document with artifact IDs. Matched: {update_result.matched_count}, Modified: {update_result.modified_count}")
        except Exception as e:
            log_with_url("ERROR", f"Failed to update MongoDB with artifact IDs: {str(e)}", exc_info=True)
    
    # Handle failure
    if ret_code != 0:
        if config['delete_log_after_parsing']:
            log_with_url("INFO", f"Cleaning up working directory due to crawler failure: {wd_path}")
            shutil.rmtree(wd_path)
        
        error_msg = f'Crawler failed with return code {ret_code}' + (" after timeout" if timeoutCheck else "")
        log_with_url("ERROR", error_msg)
        raise Exception(error_msg)
    
    # Task completed successfully
    end = time.perf_counter()
    duration = end - start
    log_with_url("INFO", f"Crawling completed successfully in {duration:.2f} seconds")
    
    self.update_state(state='SUCCESS', meta={
        'status': 'Crawling done',
        'time': duration,
        'end_time': time.time()
    })
    
    return {
        'status': 'success',
        'url': url,
        'submission_id': submission_id,
        'duration': duration,
        'screenshot_count': len(screenshot_ids),
        'har_count': len(har_ids),
        'log_count': len(log_ids)
    }