import logging
import pathlib


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), "INFO"),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def infer_fs_seeds(fs_dir: str, start: int = None, end: int = None):
    """
    Infer filesystem seeds from the cache directory.
    
    Args:
        fs_dir: Path to filesystem directory
        start: Start seed (inclusive)
        end: End seed (inclusive)
        
    Returns:
        Sorted list of filesystem seeds
    """
    all_seeds = sorted([
        int(f.stem) for f in pathlib.Path(fs_dir).glob('*.json')
        if f.stem.isdigit()
    ])
    
    if start is not None and end is not None:
        seeds = [s for s in all_seeds if start <= s <= end]
    elif start is not None:
        seeds = [s for s in all_seeds if s >= start]
    elif end is not None:
        seeds = [s for s in all_seeds if s <= end]
    else:
        seeds = all_seeds
    
    return seeds
