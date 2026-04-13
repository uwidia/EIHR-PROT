import logging

def setup_logging():
    logger = logging.getLogger()  # root logger
    
    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a")
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        handlers=handlers
    )
    
    return logger

logger = logging.getLogger(__name__)
logger.info("Logger is ready for console and file!")