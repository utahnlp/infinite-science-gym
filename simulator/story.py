import logging
from typing import Optional

import numpy as np
# from transformers import GenerationConfig

from .llm import LLM
from .prompts import StoryPrompt
from .config import StoryConfig
from .types import Story
from .utils import set_global_seed


class StoryGenerator:
    logger = logging.getLogger("StoryGenerator")

    def __init__(self, story_config: StoryConfig, llm: LLM):
        self.conf = story_config
        self.llm = llm

    def generate(
            self, 
            seed: int,
            field: Optional[str] = None, 
            domain: Optional[str] = None,
            subdomain: Optional[str] = None) -> Story:
        """
        Generate a short, realistic research experiment context for a scientific data repository using the LLM.
        Returns:
            A dict with keys: field, domain, subdomain, subdomain_description, project_title, experiment_name, description.
        """

        set_global_seed(seed)
        rng = np.random.default_rng(seed)

        if field is None:
            field = str(rng.choice(list(self.conf.taxonomy.keys())))
        if domain is None:
            domain = str(rng.choice(list(self.conf.taxonomy[field].keys())))
        if subdomain is None:
            subdomain = str(rng.choice(list(self.conf.taxonomy[field][domain].keys())))
        subdomain_description = self.conf.taxonomy[field][domain][subdomain]
        self.logger.debug(f'{field=}\n{domain=}\n{subdomain=}\n{subdomain_description=}')

        project_titles = self.llm.generate_list_of_dict(
            prompt=StoryPrompt.project_title(
                field=field, 
                domain=domain, 
                subdomain=subdomain, 
                subdomain_description=subdomain_description,
                num_project_titles=self.conf.num_project_titles), 
            required_keys=["title", "repository_name"], 
            length=self.conf.num_project_titles)
        self.logger.debug(f'{project_titles=}')
        
        project_idx = rng.integers(len(project_titles))
        title = str(project_titles[project_idx]['title'])
        root_dir_name = str(project_titles[project_idx]['repository_name'])
        self.logger.debug(f'{title=}')
        self.logger.debug(f'{root_dir_name=}')

        description = self.llm.generate_dict(
            prompt=StoryPrompt.description(
                subdomain=subdomain,
                project_title=title), 
            required_keys=["description"])["description"]
        self.logger.debug(f'{description=}')

        abstract = self.llm.generate_dict(
            prompt=StoryPrompt.abstract(
                project_title=title, 
                description=description), 
            required_keys=["abstract"])["abstract"]
        self.logger.debug(f'{abstract=}')

        return Story(
            field=field,
            domain=domain,
            subdomain=subdomain,
            subdomain_description=subdomain_description,
            root_dir_name=root_dir_name,
            title=title,
            description=description,
            abstract=abstract)
