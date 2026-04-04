import os
from pathlib import Path
def data_path():
    """
    Returns the path to the data directory.
    
    This function constructs the path to the data directory based on the current file's location.
    
    Returns:
        str: The absolute path to the data directory.
    """
    return Path('data')
