import os
import ssl

import certifi
from celery import Celery
from dotenv import load_dotenv

load_dotenv(override=True)

redis_url = os.getenv("REDIS_URL")

celery_app = Celery(
    "rag_pipeline",
    broker=redis_url,
    backend=redis_url,
    include=["pipeline"],
)

celery_app.conf.broker_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_REQUIRED,
    "ssl_ca_certs": certifi.where(),
}
celery_app.conf.redis_backend_use_ssl = {
    "ssl_cert_reqs": ssl.CERT_REQUIRED,
    "ssl_ca_certs": certifi.where(),
}
