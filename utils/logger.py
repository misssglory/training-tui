import sys
from loguru import logger
import time
from pathlib import Path
import traceback


def setup_logger(config):
    """Setup logger with file and console output."""
    logs_dir = Path(config.get('paths.logs_dir'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'app_{timestamp}.log'
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (will be captured by UI)
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )
    
    # Add error-only file
    error_file = logs_dir / f'errors_{timestamp}.log'
    logger.add(
        error_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="10 MB"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def log_exception(func):
    """Decorator to log exceptions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
    return wrapper
