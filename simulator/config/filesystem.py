from dataclasses import dataclass
from datetime import date
from typing import List

from .utils import default_field


@dataclass
class FileSystemConfig:

    # LLM variabels
    # max_new_tokens: int = 10000
    sampling_parameters: dict = default_field({'temperature':0.7, 'top_p':0.8, 'top_k':20})
    max_generation_retries: int = 3
    max_debug_retries: int = 2

    min_num_files: int = 15
    max_num_files: int = 10000 
    num_files_dist: dict = default_field(('beta', {'a': 1.5, 'b': 15}))
    num_file_columns: List[int] = default_field([7, 20])

    # Directory and file separators
    dir_inner_separators: List[str] = default_field(["-", "_"]) # Options for joining placeholders in a directory level
    file_separators: List[str] = default_field(["-", "_"]) # Options for joining placeholders in filenames

    # Number of placeholders in directory and file structure
    dir_levels: List[int] = default_field([2, 5]) # min (inclusive) - max (exclusive) number of directory levels
    per_level_dir: List[int] = default_field([1, 4]) # min (inclusive) - max (exclusive) placeholders per level (directory)
    file_placeholders: List[int] = default_field([1, 4]) # min (inclusive) - max (exclusive) placeholders for file template

    # Placeholder values
    max_run_number: List[int] = default_field([2, 7])
    num_experiment_names: List[int] = default_field([2, 5])
    num_researchers: List[int] = default_field([2, 5])
    num_benchmarks: List[int] = default_field([2, 5])
    date_range: List[date] = default_field([date(2023, 1, 1), date(2026, 1, 1)])
    num_dates: List[int] = default_field([3, 30])

    # File variable generation
    include_assistant_response_prefix: bool = True
    sig_figs: int = 3 # how many significant figures to truncate values to
    stoch_est_num_paths: int = 100
    stoch_est_num_rows_per_path: int = 100

    # README and metadata
    readme_prob: float = 0.5
    readme_title_prob: float = 0.5
    readme_abstract_prob: float = 0.5
    metadata_tax_prob: float = 0.5
    metadata_title_prob: float = 0.5
    metadata_abstract_prob: float = 0.5


    def __post_init__(self):
        range_vars = [
            self.num_file_columns, 
            self.dir_levels, self.per_level_dir, self.file_placeholders,
            self.max_run_number, self.num_experiment_names, self.num_researchers, 
            self.num_benchmarks, self.num_dates, 
        ]

        for range_var in range_vars:
            assert len(range_var) == 2
            assert all(type(x) == int for x in range_var)
            assert all(x > 0 for x in range_var)
            assert range_var[0] < range_var[1]
