import sys
import logging
import requests
from pathlib import Path
from tqdm import tqdm

def create_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger with the specified name.

    Args:
        name (str): a string as the logger name.
        level (int): an integer to set the logger level.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(ch)
    return logger

def data_downloader(url: str, save_path: str) -> None:
    """
    A data download helper with progress bar.

    Args:
        url (str): the url of the dataset
        save_path (str): the path to save the dataset

    Returns:
        None
    """
    if Path(save_path).exists():
        print('Found local copy.', file=sys.stderr)
    else:
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True
        )
        with open(save_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

