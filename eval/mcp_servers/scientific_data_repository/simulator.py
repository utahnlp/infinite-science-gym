import os
import pathlib
import sys

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulator import CacheOnlySimulator
from simulator.config import FileConfig
from simulator.status import SuccessStatus
from data.utils import load_config


################# Init Simulator #################

cfg = load_config(os.environ["CONFIG_PATH"], os.environ["DEFAULT_CONFIG_PATH"])['simulator']
simulator = CacheOnlySimulator(cfg=cfg, file_config=FileConfig())

# run quick tests to ensure filesystems and files are available
fs, status = simulator.get_filesystem(seed=1)
assert status == SuccessStatus.OK, status
_, status = simulator.read_file(seed=1, path=fs.tree.get_paths()[0])
assert status == SuccessStatus.OK, status
