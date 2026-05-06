from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Any

from . import (
    DirectoryTree, 
    file_variable_from_json,
    file_variable_to_json,
    FileVariable, 
    PathVariable, 
    Story,
    VariableRole
)


@dataclass
class FileSystem:
    seed: int
    readme_visibility: Dict[str, bool]
    metadata_visibility: Dict[str, bool]
    directory_template: str # Directory structure template, e.g., "/{experiment_name}/{yyyy}_{mm}_{dd}/"
    filename_template: str # Filename template, e.g., "{experiment_name}_{run_number}_{timestamp}.csv"
    path_choices: Dict[str, str]
    story: Story
    tree: DirectoryTree
    path_variables: List[PathVariable]
    file_variables: List[FileVariable]
    sort_key: str = None

    def __post_init__(self):
        pass

    @classmethod
    def from_json(cls, obj):
        return cls(
            seed=obj['seed'],
            readme_visibility=obj['readme_visibility'],
            metadata_visibility=obj['metadata_visibility'],
            directory_template=obj['directory_template'],
            filename_template=obj['filename_template'],
            path_choices=obj['path_choices'],
            story=Story.from_json(obj['story']),
            tree=DirectoryTree.from_json(obj['tree']),
            path_variables=[PathVariable.from_json(v) for v in obj['path_variables']],
            file_variables=[file_variable_from_json(v, compile_fns=True) for v in obj['file_variables']],
            sort_key=obj['sort_key'],
        )

    def to_json(self) -> Dict[str, Any]:
        return deepcopy({
            'seed': self.seed,
            'readme_visibility': self.readme_visibility,
            'metadata_visibility': self.metadata_visibility,
            'directory_template': self.directory_template,
            'filename_template': self.filename_template,
            'path_choices': self.path_choices,
            'story': self.story.to_json(),
            'tree': self.tree.to_json(),
            'path_variables': [v.to_json() for v in self.path_variables],
            'file_variables': [file_variable_to_json(v) for v in self.file_variables],
            'sort_key': self.sort_key
        })
    
    def to_str(self, exclude_tree: bool = False, exclude_code: bool = False) -> str:
        d = self.to_json()
        
        if exclude_tree:
            d.pop('tree')
        
        if exclude_code:
            fvs = d['file_variables']
            for fv in fvs:
                if fv['role'] == 'DEPENDENT':
                    fv.pop('subclass_data')

        return str(d)
