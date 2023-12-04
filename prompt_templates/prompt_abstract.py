from abc import abstractmethod, ABC
from typing import Union, Dict, List
from jinja2 import Environment, Template
from utils.utils import remove_non_utf8_characters
from utils.chatgpt_utils import ChatGptUtility

ENVIRONMENT = Environment()


class AbstractPrompt(ABC):

    def __init__(
            self,
            ground_truth: Union[str, int],
            record_id: str = None,
            enable_chatgpt_utility: bool = False
    ):
        self.ground_truth = ground_truth
        self.record_id = record_id
        self.model_response = None
        self.enable_chatgpt_utility = enable_chatgpt_utility
        if enable_chatgpt_utility:
            self.chatgpt_utility = ChatGptUtility(chatgpt_model='gpt-4-1106-preview', max_new_tokens=8192)
        else:
            self.chatgpt_utility = None

    def to_dict(self) -> Dict[str, Union[str, int]]:
        answer = self.extract_answer()
        mapped_answer = self.map_answer(answer)
        return {
            'record_id': self.record_id,
            'prompt': self.get_prompt(),
            'ground_truth': self.ground_truth,
            'mapped_ground_true': self.map_answer(self.ground_truth),
            'model_response': remove_non_utf8_characters(self.model_response),
            'answer': answer,
            'mapped_answer': mapped_answer
        }

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    def set_model_response(self, model_response):
        self.model_response = model_response

    def get_prompt_type(self):
        return self.__class__.__name__

    @abstractmethod
    def extract_answer(self):
        pass

    @staticmethod
    def map_answer(answer):
        return answer


class Prompt(AbstractPrompt):

    def __init__(
            self,
            data: Dict[str, str],
            *args,
            **kwargs
    ):
        super(Prompt, self).__init__(*args, **kwargs)
        self.prompt = self.get_prompt_template().render(**data)

    @abstractmethod
    def get_prompt_template(self) -> Template:
        pass

    def get_prompt(self):
        return self.prompt


class NestedPrompt(AbstractPrompt):

    def __init__(
            self,
            prompts: List[Prompt],
            *args,
            **kwargs
    ):
        super(NestedPrompt, self).__init__(*args, **kwargs)
        self.prompts = prompts

    @abstractmethod
    def get_nested_prompt_template(self) -> Template:
        pass

    @staticmethod
    @abstractmethod
    def get_base_prompt_class() -> Prompt:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    def get_prompts(self):
        return self.prompts
