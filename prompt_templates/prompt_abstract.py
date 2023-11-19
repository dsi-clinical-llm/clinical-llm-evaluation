from abc import abstractmethod, ABC
from typing import Union, Dict
from jinja2 import Environment, Template
from utils.utils import remove_non_utf8_characters

ENVIRONMENT = Environment()


class Prompt(ABC):

    def __init__(
            self,
            ground_truth: Union[str, int],
            record_id: str = None,
            data: Dict[str, str] = {}
    ):
        self.ground_truth = ground_truth
        self.record_id = record_id
        self.prompt = self.get_prompt_template().render(**data)
        self.model_response = None

    def to_dict(self) -> Dict[str, Union[str, int]]:
        answer = self.extract_answer()
        mapped_answer = self.map_answer(str(answer))
        return {
            'record_id': self.record_id,
            'prompt': self.prompt,
            'ground_truth': self.ground_truth,
            'mapped_ground_true': self.map_answer(self.ground_truth),
            'model_response': remove_non_utf8_characters(self.model_response),
            'answer': answer,
            'mapped_answer': mapped_answer
        }

    def set_model_response(self, model_response):
        self.model_response = model_response

    def get_prompt_type(self):
        return self.__class__.__name__

    @abstractmethod
    def get_prompt_template(self) -> Template:
        pass

    @abstractmethod
    def extract_answer(self):
        pass

    @staticmethod
    def map_answer(answer):
        return answer
