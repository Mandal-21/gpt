import logging
from logging.handlers import TimedRotatingFileHandler

def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    filehandler = TimedRotatingFileHandler(f'logs/{log_path}.log', when='midnight', backupCount=5)
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger
