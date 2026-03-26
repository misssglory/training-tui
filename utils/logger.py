import sys
from loguru import logger
import time
from pathlib import Path

def setup_logger(config):
    logs_dir = Path(config.get('paths.logs_dir'))
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'training_{timestamp}.log'
    
    logger.remove()
    logger.add(sys.stdout, format="{time} | {level} | {message}", level="INFO")
    logger.add(log_file, format="{time} | {level} | {message}", level="DEBUG", rotation="10 MB")
    
    return logger
