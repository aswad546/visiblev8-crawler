import log_parser_worker.config.celery_config as cfg

from celery import Celery
from kombu import Queue


celery_app = Celery(
    cfg.celery_id,
    broker=cfg.celery_broker_uri,
    backend=cfg.celery_backend_uri ,
    include=[
        'log_parser_worker.tasks'
    ]
)

celery_app.conf.update(
    result_extended = True,
    result_expires=2592000,  # Keep task results for 30 days
    task_acks_late=True,  # Acknowledge task ONLY after it completes
    task_reject_on_worker_lost=True,  # Requeue task if worker crashes mid-job
    acks_on_failure_or_timeout=True,
    worker_send_task_events=True,  # Enable this for the exporter
    task_send_sent_event=True,  # Enable this for the exporter
    broker_transport_options={
        'visibility_timeout': 3600  # Tasks reappear after 60 minutes if not acknowledged
    }

)


celery_app.conf.task_default_queue = 'default'
celery_app.conf.task_routes = (
    Queue('log_parser', routing_key='log_parser')
)
celery_app.conf.task_default_exchange = 'default'
celery_app.conf.task_default_exchange_type = 'direct'
celery_app.conf.task_default_routing_key = 'default'
celery_app.conf.task_routes = {
    'log_parser_worker.parse_log': {
        'queue': 'log_parser'
    }
}


if __name__ == '__main__':
    celery_app.start()
