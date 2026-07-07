import os
import logging
from pathlib import Path
from configparser import ConfigParser

def createLogger():

    config = ConfigParser()
    config.read("config/config.ini")

    logs_dir = config["LOGS"]["directory"]
    logs_file = config["LOGS"]["file"]

    os.makedirs(logs_dir, exist_ok = True)

    logger = logging.getLogger("Sentiment_Analyser")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )

    log_file_path = Path(f"{logs_dir}/{logs_file}")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = createLogger()