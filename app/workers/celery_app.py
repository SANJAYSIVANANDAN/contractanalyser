from celery import Celery

from app.config import settings

# broker = where task messages are queued. RabbitMQ is gone; Redis is both the broker
# and the result backend, so there is only one messaging service to run.
celery_app = Celery(
    "contractanalyser",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,             # ack AFTER the task finishes, so a crashed worker's task is redelivered
    worker_prefetch_multiplier=1,    # a worker takes one task at a time (tasks are long)
)
