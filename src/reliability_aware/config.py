from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT/ "data"

def setup_logging():
    logger = logging.getLogger()  
    
    
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