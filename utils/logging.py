import logging
from logging_loki import LokiQueueHandler
from queue import Queue

def setup_logging(app_name: str = "chat-ebook-ai"):
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        loki_handler = LokiQueueHandler(
            Queue(-1),
            url="http://loki:3100/loki/api/v1/push",
            tags={"application": app_name, "env": "dev"},
            version="1",
        )
        loki_handler.setLevel(logging.INFO)
        loki_handler.setFormatter(console_formatter)
        logger.addHandler(loki_handler)

    return logger