from functools import lru_cache
import hashlib
import json
import logging
import os
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .simulator import Simulator
from .file import FileGenerator
from .status import SuccessStatus
from .utils import set_global_seed
from .config import FileConfig
from .types import File, FileSystem


class CacheOnlySimulator(Simulator):

    logger = logging.getLogger("CacheOnlySimulator")

    def __init__(self, cfg: Dict[str, Any], file_config: FileConfig):
        
        self.cfg = cfg
        self.file_generator = FileGenerator(file_config=file_config)


    # @lru_cache(maxsize=128)
    def get_filesystem(self, seed: int) -> Tuple[Optional[FileSystem], SuccessStatus]:
        
        cache_path = os.path.join(self.cfg['cache_dir'], f'{seed}.json')

        # using local cache
        if self.cfg['source'] == 'local':
            try:
                with open(cache_path) as f:
                    fs = FileSystem.from_json(json.load(f))
            except FileNotFoundError as e:
                self.logger.info(f"filesystem doesn't exist: {cache_path}")
                return None, SuccessStatus.NOT_CACHED
            except Exception as e:
                self.logger.exception(f"error while loading filesystem: {cache_path}")
                self.logger.exception(traceback.format_exc())
                return None, SuccessStatus.ERROR

        # using AWS s3 cache
        elif self.cfg['source'] == 's3':
            raise NotImplementedError

        return fs, SuccessStatus.OK


    def read_file(self, seed: int, path: str, fs: FileSystem = None) -> Tuple[Optional[File], SuccessStatus]:

        if fs is None:
            fs, status = self.get_filesystem(seed)
            if fs is None:
                self.logger.info(f'failed to get filesystem: {seed=} {path=}')
                return None, status
        
        # get tree
        tree = fs.tree

        if path[0] == '/':
            path = path[1:]

        # check if path is actually in tree
        if not tree.contains_path(path, must_be_file=True):
            self.logger.info(f'invalid path: {seed=} {path=}')
            return None, SuccessStatus.INVALID_PATH
        
        if path == fs.tree.readme_path:
            return File(data=fs.tree.readme_str, extension='.md'), SuccessStatus.OK

        # path is in tree, hash the path as seed for generating file contents
        hex_hash = hashlib.sha256(bytes(path, encoding="utf-8")).hexdigest()
        file_seed = int(hex_hash, 16) % (2 ** 32) # to be represented by 32-bit integer
        set_global_seed(file_seed)
        rng = np.random.default_rng(file_seed)

        # populate file using information in filesystem
        try:
            file = self.file_generator.populate_file(fs=fs, path=path, rng=rng)
        except Exception as e:
            cache_path = os.path.join(self.cfg['cache_dir'], f'{seed}.json')
            self.logger.exception("error while generating file")
            self.logger.exception(f"cached filesystem: {cache_path}")
            self.logger.exception(f"file path: {path}")
            self.logger.exception(traceback.format_exc())
            return None, SuccessStatus.ERROR
        
        return file, SuccessStatus.OK


    def get_fs_extension(self, seed: int) -> Tuple[Optional[str], SuccessStatus]:
        fs, status = self.get_filesystem(seed)
        if fs is None:
            self.logger.info(f'failed to get filesystem: {seed=}')
            return None, status
