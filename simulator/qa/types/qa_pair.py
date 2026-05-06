from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from .condition import Condition, condition_from_json


class QAScope(Enum):
    METADATA = 'metadata'
    DIRECTORY = 'directory'
    NO_FILES = 'no_files'
    SINGLE_FILE = 'single_file'
    MULTIPLE_FILES = 'multiple_files'

@dataclass
class Question:
    question: str
    variables: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert all(var in self.question for var in self.variables)

    def __str__(self):
        return self.swap_variables(self.question)

    def to_json(self) -> Dict[str, str]:
        return {'question': self.question, 'variables': self.variables}
    
    def swap_variables(self, q: str) -> str:
        for var, value in self.variables.items():
            q = q.replace(var, value)
        return q


@dataclass
class Answer:
    answer: str | None
    has_answer: bool
    _type: str | None

    def __post_init__(self):
        assert self._type in [None, 'int', 'float', 'str', 'bool'] 
        if self._type == None:
            assert self.has_answer is False
            assert self.answer == None
        else: # self._type != None
            assert self.has_answer
            if self._type == 'int':
                assert type(self.answer) == int, (self.answer, type(self.answer))
            elif self._type == 'float':
                assert type(self.answer) == float, (self.answer, type(self.answer))
            elif self._type == 'str':
                assert type(self.answer) == str, (self.answer, type(self.answer))
            elif self._type == 'bool':
                assert type(self.answer) == bool, (self.answer, type(self.answer))

    def __str__(self):
        return f'{self.answer} ({self._type})' if self.has_answer else '<NO ANSWER>'

    def to_json(self) -> Dict[str, str]:
        return {'answer': self.answer, 'has_answer': self.has_answer, '_type': self._type}


@dataclass
class QuestionAnswerPair:
    question: Question
    answer: Answer
    scope: QAScope
    path_conditions: List[Condition] = field(default_factory=list)
    file_conditions: List[Condition] = field(default_factory=list)
    paraphrases: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj):
        return cls(
            question=Question(**obj['question']),
            answer=Answer(**obj['answer']),
            scope=QAScope[obj['scope']],
            path_conditions=[condition_from_json(x) for x in obj['path_conditions']],
            file_conditions=[condition_from_json(x) for x in obj['file_conditions']],
            paraphrases=obj['paraphrases'])
        
    def to_json(self) -> Dict[str, Any]:
        return {
            'question': self.question.to_json(),
            'answer': self.answer.to_json(),
            'scope': self.scope.name,
            'path_conditions': [x.to_json() for x in self.path_conditions],
            'file_conditions': [x.to_json() for x in self.file_conditions],
            'paraphrases': self.paraphrases}
