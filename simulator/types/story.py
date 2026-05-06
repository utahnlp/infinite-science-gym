from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Story:
    field: str
    domain: str
    subdomain: str
    subdomain_description: str
    root_dir_name: str
    title: str
    description: str
    abstract: str

    def to_json(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'domain': self.domain,
            'subdomain': self.subdomain,
            'subdomain_description': self.subdomain_description,
            'root_dir_name': self.root_dir_name,
            'title': self.title,
            'description': self.description,
            'abstract': self.abstract,
        }

    @classmethod
    def from_json(cls, obj):
        return cls(
            field=obj['field'],
            domain=obj['domain'],
            subdomain=obj['subdomain'],
            subdomain_description=obj['subdomain_description'],
            root_dir_name=obj['root_dir_name'],
            title=obj['title'],
            description=obj['description'],
            abstract=obj['abstract'],
        )

    def __str__(self):
        return (
            f'field: {self.field}\n'
            f'domain: {self.domain}\n'
            f'subdomain: {self.subdomain}\n'
            f'subdomain_description: {self.subdomain_description}\n'
            + '-'*80 + '\n' +
            f'root_dir_name: {self.root_dir_name}\n'
            f'title: {self.title}\n'
            f'description: {self.description}\n'
            f'abstract: {self.abstract}')

    def to_str(self, exclude_description: bool = False):
        return (
            f'field: {self.field}\n'
            f'domain: {self.domain}\n'
            f'subdomain: {self.subdomain}\n'
            f'subdomain_description: {self.subdomain_description}\n'
            + '-'*80 + '\n' +
            f'root_dir_name: {self.root_dir_name}\n'
            f'title: {self.title}\n'
            + (f'description: {self.description}\n' if not exclude_description else '') + 
            f'abstract: {self.abstract}')
