from celery import Celery

from app.config import settings

# broker = where task messages are queued (RabbitMQ)
celery_app = Celery("docupilot", broker=settings.rabbitmq_url, include=["app.workers.tasks"])

celery_app.conf.update(
    task_acks_late=True,             # ack AFTER the task finishes, so a crashed worker's task is redelivered
    worker_prefetch_multiplier=1,    # a worker takes one task at a time (tasks are long)
)
