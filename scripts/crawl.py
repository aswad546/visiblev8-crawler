import argparse
from typing import List
import requests
import local_data_store
import docker
import csv
from datetime import datetime
import time
import pika
import json
import logging
from flask import Flask, request, jsonify


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class Crawler:
    def __init__(
            self,
            output_format: str,
            post_processors: str,
            delete_log_after_parsing: bool,
            disable_artifact_collection: bool,
            disable_screenshots: bool,
            disable_har: bool,
            crawler_args: List[str],
            server_load_check: bool,
            hard_timeout: int):
        self.output_format = output_format
        self.post_processors = post_processors
        self.delete_log_after_parsing = delete_log_after_parsing
        self.disable_artifact_collection = disable_artifact_collection
        self.disable_screenshots = disable_screenshots
        self.disable_har = disable_har
        self.crawler_args = crawler_args
        self.hard_timeout = hard_timeout
        self.server_load_check = server_load_check
        self.prefetch_count = 128
        self.data_store = local_data_store.init()
        self.url_actions = None
        if self.data_store.server_type == 'local':
            docker.wakeup(self.data_store.data_directory)

        if self.data_store.server_type == 'local' and self.server_load_check:
            requests.get(f'http://{self.data_store.hostname}:5555/api/workers?refresh=1')
            print('Refreshing workers')
            req = requests.get(f'http://{self.data_store.hostname}:5555/api/workers')
            workers = req.json()
            for ke in workers:
                self.prefetch_count = workers[ke]["stats"]["prefetch_count"]
            print('Setting up prefetch counter')

    @app.route('/api/login_candidates', methods=['POST'])
    def login_candidates():
        crawler_inst = app.config["crawler_inst"] 
        data_store = local_data_store.init()
        try:
            # Parse the JSON payload from the request
            data = request.get_json()
            if data is None:
                raise ValueError("No JSON payload provided")
            
            app.logger.info("Received login candidates: %s", data)

            candidates = data.get("candidates")
            if not candidates:
                raise ValueError("No candidates provided in the payload")

            crawler_inst.crawl_with_actions(candidates, data_store)
            
            # TODO: Process or store the data as needed.
            # For example, you might save it to a database, or queue it for further processing.

            # Return a success response
            return jsonify({"status": "success", "message": "Candidates received"}), 200

        except Exception as e:
            app.logger.error("Error processing candidates: %s", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 400

    def crawl(self, urls: List[str])-> str:
        r = None
        submission_identifiers = []
        for url in urls:
            url = url.rstrip('\n')
            r = None
            while self.server_load_check:
                if self.data_store.server_type == 'local' and self.server_load_check:
                    # Use the celery api to check if we have too many reserved tasks ?
                    req = requests.get(f'http://{self.data_store.hostname}:5555/api/tasks?state=RECEIVED')
                    if req.status_code != 200:
                        raise Exception(f'Failed to get workers from celery api. Status code: {req.status_code}')
                    else:
                        tasks = req.json()
                        no_of_tasks = 0
                        for _ke in tasks:
                            no_of_tasks += 1
                        if no_of_tasks >= self.prefetch_count:
                            print('Server is overloaded, sleep for some time')
                            time.sleep(5)
                        else:
                            break

            if self.post_processors:
                print({
                    'url': url,
                    'rerun': True,
                    'crawler_args': self.crawler_args,
                    'disable_artifact_collection': self.disable_artifact_collection,
                    'disable_screenshots': self.disable_screenshots,
                    'disable_har': self.disable_har,
                    'hard_timeout': self.hard_timeout,
                    'parser_config': {
                        'parser': self.post_processors,
                        'delete_log_after_parsing': self.delete_log_after_parsing,
                        'output_format': self.output_format,
                        }
                    })
                r = requests.post(  f'http://{self.data_store.hostname}:4000/api/v1/urlsubmit', json={
                    'url': url,
                    'rerun': True,
                    'crawler_args': self.crawler_args,
                    'disable_artifact_collection': self.disable_artifact_collection,
                    'disable_screenshots': self.disable_screenshots,
                    'disable_har': self.disable_har,
                    'hard_timeout': self.hard_timeout,
                    'parser_config': {
                        'parser': self.post_processors,
                        'delete_log_after_parsing': self.delete_log_after_parsing,
                        'output_format': self.output_format,
                        },
                    })
            else:
                r = requests.post(f'http://{self.data_store.hostname}:4000/api/v1/urlsubmit', json={
                    'url': url,
                    'rerun': True,
                    'disable_screenshots': self.disable_screenshots,
                    'disable_har': self.disable_har,
                })
            submission_id = r.json()['submission_id']
            submission_identifiers.append((submission_id, url, datetime.now()))
        self.data_store.db.executemany('INSERT INTO submissions VALUES ( ?, ?, ? )', submission_identifiers)
        self.data_store.commit()

    def crawl_with_actions(self, login_candidates, data_store)-> str:
        r = None
        submission_identifiers = []
        for candidate in login_candidates:
            url = candidate['url'].rstrip('\n')
            actions = candidate['actions']
            scan_domain = candidate['scan_domain']
            r = None
            while self.server_load_check:
                if data_store.server_type == 'local' and self.server_load_check:
                    # Use the celery api to check if we have too many reserved tasks ?
                    req = requests.get(f'http://{data_store.hostname}:5555/api/tasks?state=RECEIVED')
                    if req.status_code != 200:
                        raise Exception(f'Failed to get workers from celery api. Status code: {req.status_code}')
                    else:
                        tasks = req.json()
                        no_of_tasks = 0
                        for _ke in tasks:
                            no_of_tasks += 1
                        if no_of_tasks >= self.prefetch_count:
                            print('Server is overloaded, sleep for some time')
                            time.sleep(5)
                        else:
                            break
            print({
                'url': url,
                'actions': actions,
                'scan_domain': scan_domain,
                'rerun': True,
                'crawler_args': self.crawler_args,
                'disable_artifact_collection': self.disable_artifact_collection,
                'disable_screenshots': self.disable_screenshots,
                'disable_har': self.disable_har,
                'hard_timeout': self.hard_timeout,
                'parser_config': {
                    'parser': 'flow',
                    'delete_log_after_parsing': self.delete_log_after_parsing,
                    'output_format': self.output_format,
                    }
                })
            r = requests.post(f'http://{data_store.hostname}:4000/api/v1/urlsubmit-actions', json={
                'url': url,
                'actions': actions,
                'scan_domain': scan_domain,
                'rerun': True,
                'crawler_args': self.crawler_args,
                'disable_artifact_collection': self.disable_artifact_collection,
                'disable_screenshots': self.disable_screenshots,
                'disable_har': self.disable_har,
                'hard_timeout': self.hard_timeout,
                'parser_config': {
                    'parser': 'flow', #Post processor will always be flow for my study
                    'delete_log_after_parsing': self.delete_log_after_parsing,
                    'output_format': self.output_format,
                    },
                })
            submission_id = r.json()['submission_id']
            submission_identifiers.append((submission_id, url, datetime.now(), json.dumps(actions) if actions is not None else None, scan_domain))
        data_store.db.executemany('INSERT INTO submissions (submission_id, url, start_time, actions, scan_domain) VALUES (?, ?, ?, ?, ?)', submission_identifiers)
        data_store.commit()
        data_store.conn.close()


    def process_message(body):
        """
        Process a JSON message that includes two lists:
        - "urls": list of URLs (strings)
        - "actions": list of corresponding actions (could be None or list)
        
        Returns a list of dictionaries, each with keys "url" and "actions".
        """
        try:
            data = json.loads(body)
        except Exception as e:
            logger.error("Failed to decode JSON: %s", e)
            return None

        urls = data.get("urls", [])
        actions = data.get("actions", [])
        
        if not urls:
            logger.error("No URLs found in message.")
            return None

        if len(urls) != len(actions):
            logger.warning("Number of URLs and actions do not match; using zip to combine them (some may be dropped).")
        
        # Create a list of dicts pairing each URL with its corresponding action.
        pairs = [{"url": url, "actions": action} for url, action in zip(urls, actions)]
        return pairs
    
# def on_message_callback(channel, method, properties, body):
#     logger.info("Received message from queue.")
#     print("Received message from queue")
#     # Process the message to extract URL-action pairs.
#     pairs = process_message(body)
#     print(pairs)
#     if pairs is None:
#         # Negative acknowledge the message if processing failed.
#         channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#         return

#     # Create an instance of the Crawler. Configure these parameters as needed.
#     crawler_inst = Crawler(
#         output_format="postgresql",         # Adjust based on your needs
#         post_processors="",                 # Adjust if you use any post processors
#         delete_log_after_parsing=False,
#         disable_artifact_collection=False,
#         disable_screenshots=False,
#         disable_har=False,
#         crawler_args=[],                    # Additional arguments if needed
#         server_load_check=True,
#         hard_timeout=20*60                  # e.g., 20 minutes
#     )

#     # Process each URL and its corresponding actions.
#     for pair in pairs:
#         url = pair["url"]
#         actions = pair["actions"]
#         logger.info("Processing URL: %s with actions: %s", url, actions)
#         try:
#             # If your crawler.crawl method only accepts a list of URLs,
#             # you may need to modify it to accept an action as well.
#             # For this example, we're simply passing the URL.
#             crawler_inst.crawl([url])
#             # If actions need to be handled differently, add that logic here.
#         except Exception as e:
#             logger.exception("Crawler encountered an error processing %s: %s", url, e)
#             # Optionally, nack the message and stop processing further.
#             channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
#             return

#     # After processing all pairs, acknowledge the message.
#     channel.basic_ack(delivery_tag=method.delivery_tag)
#     logger.info("Message processed and acknowledged.")

def crawler( args: argparse.Namespace, unknown_args: list[str]):
    output_format = args.output_format
    parsers = args.post_processors
    delete_log_after_parsing = args.delete_log_after_parsing
    disable_artifact_collection = args.disable_artifact_collection
    disable_screenshots = args.disable_screenshots
    disable_har = args.disable_har
    hard_timeout = int(args.timeout)
    server_load_check = args.server_load_check
    crawler_args = unknown_args
    # Set up connection parameters (adjust host/port as needed)
    # connection_params = pika.ConnectionParameters('localhost')
    # connection = pika.BlockingConnection(connection_params)
    # channel = connection.channel()

    # # Declare the queue (ensure the name matches the one used for publishing)
    # queue_name = "login_candidates"
    # channel.queue_declare(queue=queue_name, durable=True)
    
    # # Optionally, limit unacknowledged messages
    # channel.basic_qos(prefetch_count=1)
    
    # # Set up the consumer
    # channel.basic_consume(queue=queue_name, on_message_callback=on_message_callback)
    
    # logger.info("Waiting for messages on queue '%s'. To exit press CTRL+C", queue_name)
    # try:
    #     channel.start_consuming()
    # except KeyboardInterrupt:
    #     logger.info("Interrupt received, stopping consumption.")
    #     channel.stop_consuming()
    # finally:
    #     connection.close()

    crawler_inst = Crawler(
        output_format,
        parsers,
        delete_log_after_parsing,
        disable_artifact_collection,
        disable_screenshots,
        disable_har,
        crawler_args,
        server_load_check,
        hard_timeout)
    

    if args.url:
        crawler_inst.crawl([ args.url ])
    elif args.file:
        with open(args.file, 'r') as f:
            urls = f.readlines()
            crawler_inst.crawl(urls)
    elif args.csv:
        with open(args.csv, 'r') as f:
            raw_file_urls = list(csv.reader(f, delimiter=","))
            urls = []
            for data in raw_file_urls:
                urls.append(f'http://{data[1]}')
            crawler_inst.crawl(urls)
    elif args.sso_monitor: # New sso_monitor mode
        # Store it in the Flask app configuration
        app.config["crawler_inst"] = crawler_inst
        app.run(host='0.0.0.0', port=4050)
    else:
        print(args)
        raise Exception('No url or file specified') # This should never happen, cause arg parser should show an error if neithier url or file is specified

def crawler_parse_args(crawler_arg_parser: argparse.ArgumentParser):
    urls = crawler_arg_parser.add_mutually_exclusive_group(required=True)
    urls.add_argument('-u', '--url', help='url to crawl')
    urls.add_argument('-f', '--file', help='file containing list of urls to crawl seperated by newlines')
    urls.add_argument('-c', '--csv', help='file containing a csv in the tranco list format corresponding to the list of urls to traverse')
    urls.add_argument('-sso', '--sso_monitor', help='Consume messages from the queue (login_candidates) indefinitely', action='store_true')

    crawler_arg_parser.add_argument('-pp', '--post-processors', help='Post processors to run on the crawled url')
    crawler_arg_parser.add_argument('-o', '--output-format', help='Output format to use for the parsed data', default='postgresql')
    crawler_arg_parser.add_argument('-d', '-dr', '--delete-log-after-parsing', help='Parser to use for the crawled url', action='store_true')
    crawler_arg_parser.add_argument('-ds', '--disable-screenshots', help='Prevents screenshots from being generated', action='store_true')
    crawler_arg_parser.add_argument('-dh', '--disable-har', help='Prevents har files from being generated', action='store_true')
    crawler_arg_parser.add_argument('-dac', '--disable-artifact-collection', help='Prevents artifacts from being uploaded to mongoDB', action='store_true')
    crawler_arg_parser.add_argument('-t', '--timeout', help='A timeout value that kills the browser after a certain amount of time has elapsed', default=str(20 * 60)) # 20 minutes
    crawler_arg_parser.add_argument('-slc', '--server-load-check', help='Check if the server is overloaded before submitting a new task', action='store_true')