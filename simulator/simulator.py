import hashlib
import logging

import numpy as np

from .file import FileGenerator
from .filesystem import FileSystemGenerator
from .llm import LLM
from .story import StoryGenerator
from .utils import set_global_seed
from .config import FileConfig, FileSystemConfig, StoryConfig
from .types import File, FileSystem, Story


class Simulator:
    """
    Manages the process of building the file system for a simulated experiment repository.
    Coordinates config generation, directory creation, and file population.
    """
    logger = logging.getLogger("Simulator")

    def __init__(
            self, 
            story_config: StoryConfig,
            fs_config: FileSystemConfig,
            file_config: FileConfig,
            device_map: str = "cpu", 
            model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
        
        self.llm = LLM(model_name=model_name, device_map=device_map, sampling_parameters=fs_config.sampling_parameters)
        self.story_generator = StoryGenerator(story_config=story_config, llm=self.llm)
        self.fs_generator = FileSystemGenerator(fs_config=fs_config, file_config=file_config, llm=self.llm)
        self.file_generator = FileGenerator(file_config=file_config)


    def get_filesystem(self, seed: int, story: Story = None) -> FileSystem:
        """
        Build the file system according to the generated config and story.
        """
        # Create background story for research project
        if not story:
            self.logger.info("Generating story...")
            story = self.story_generator.generate(seed=seed)
            self.logger.info(f"Generated story:\n{story.to_str(exclude_description=True)}")

        # Generate file system config based on the story
        self.logger.info("Generating file system config...")
        fs = self.fs_generator.generate_config(story=story, seed=seed)
        self.logger.info(f"Generated file system config has {fs.tree.num_files:,} files")

        return fs        


    def read_file(self, seed: int, path: str) -> File:
        fs = self.get_filesystem(seed)
        return self.read_file(fs, path)
        

    def read_file(self, fs: FileSystem, path: str) -> File:
        # get tree
        tree = fs.tree
        
        if path[0] == '/':
            path = path[1:]

        # check if path is actually in tree
        if not tree.contains_path(path, must_be_file=True):
            return None
        
        if path == fs.tree.readme_path:
            return File(data=fs.tree.readme_str, extension='.md')

        # path is in tree, hash the path as seed for generating file contents
        hex_hash = hashlib.sha256(bytes(path, encoding="utf-8")).hexdigest()
        seed = int(hex_hash, 16) % (2 ** 32) # to be represented by 32-bit integer
        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        # populate file using information in filesystem
        file = self.file_generator.populate_file(fs=fs, path=path, rng=rng)

        return file

    
    
