import re
import sys
import lzma
import json
import pickle
import logging
from typing import Optional
from pathlib import Path

import requests
from tqdm import tqdm

import numpy as np
#import torch

LZMA_EXTS = ('.xz', '.lzma')

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

def data_downloader(url: str, save_path: Path | str) -> None:
    """
    A data download helper with progress bar.
    Code from https://github.com/snap-stanford/GEARS/blob/master/gears/utils.py.

    Args:
        url (str): the url of the dataset
        save_path (Path | str): the path to save the dataset

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

def get_type(x) -> str:
    """
    Get the type of the variable as a string.
    Code from https://github.com/dzyim/learn.py/blob/dev/Lib/structure.py.

    Args:
        x (Any): the input variable

    Returns:
        str
    """
    _type = str(type(x))
    return re.match("<class '(.+)'>", _type).groups()[0]

def to_numpy(x) -> np.ndarray:
    """
    Detach a PyTorch tensor with gradient to a numpy array.

    Args:
        x (torch.Tensor): the input PyTorch tensor.

    Returns:
        numpy.ndarray
    """
    return x.detach().cpu().numpy()

class MyJsonEncoder(json.JSONEncoder):
    """
    MyJsonEncoder is a custom JSON encoder that handles NumPy objects.
    """
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def read_json(path: Path | str):
    """
    Read json from a file path.

    Args:
        path (Path | str): the file path

    Returns:
        Any
    """
    if Path(path).suffix in LZMA_EXTS:
        with lzma.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)

def read_pickle(path: Path | str):
    """
    Read the pickled representation from a file path and return the object.

    Args:
        path (Path | str): the file path

    Returns:
        Any
    """
    if Path(path).suffix in LZMA_EXTS:
        with lzma.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)

def write_json(x, path: Path | str, compress: Optional[bool] = None) -> Path:
    """
    Write json to a file path.

    Args:
        x (Any): an object to be serialized to JSON
        path (Path | str): the file path to save data
        compress (bool | None): compress the data or not (default: None)

    Returns:
        Path: the final path to save data
    """
    if compress is None:
        if Path(path).suffix in LZMA_EXTS:
            compress = True

    if compress:
        if Path(path).suffix not in LZMA_EXTS:
            path = str(path) + '.xz'
        with lzma.open(path, 'wt', encoding='utf-8') as f:
            json.dump(x, f, cls=MyJsonEncoder)
    else:
        with open(path, 'w') as f:
            json.dump(x, f, cls=MyJsonEncoder)

    return Path(path)

def write_pickle(x, path: Path | str, compress: Optional[bool] = None) -> Path:
    """
    Write the pickled representation of an object to a file path.

    Args:
        x (Any): an object to be serialized
        path (Path | str): the file path to save data
        compress (bool): compress the data or not (default: None)

    Returns:
        Path: the final path to save data
    """
    if compress is None:
        if Path(path).suffix in LZMA_EXTS:
            compress = True

    if compress:
        if Path(path).suffix not in LZMA_EXTS:
            path = str(path) + '.xz'
        with lzma.open(path, 'wb') as f:
            pickle.dump(x, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(x, f)

    return Path(path)

