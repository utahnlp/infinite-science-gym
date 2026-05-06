from dataclasses import dataclass


@dataclass
class StoryConfig:

    # taxonomy
    taxonomy: dict

    # decoding
    num_project_titles: int = 10 # how many candidate project names to generate
    # max_new_tokens: int = 10000
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self):
        pass
