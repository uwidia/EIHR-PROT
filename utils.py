import logging

def setup_logging():
    handlers = [logging.StreamHandler()]            
    handlers.append(logging.FileHandler("pipeline.log", mode="a")) 

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers
    )

logger = logging.getLogger(__name__)
logger.info("Logger is ready for console and file!")