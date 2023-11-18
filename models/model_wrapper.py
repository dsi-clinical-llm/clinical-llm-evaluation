from abc import ABC, abstractmethod
import logging

from utils.llm_dataclasses import ModelParameter


class CausalLanguageModelWrapper(ABC):

    def __init__(
            self,
            max_new_tokens: int = 250,
            auto_max_new_tokens: bool = False,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 20,
            num_beams: int = 1,
            truncation_length: int = 2048
    ):
        self._max_new_tokens = max_new_tokens
        self._auto_max_new_tokens = auto_max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._num_beams = num_beams
        self._truncation_length = truncation_length

        self.get_logger().info(
            f'max_new_tokens: {max_new_tokens}\n'
            f'auto_max_new_tokens: {auto_max_new_tokens}\n'
            f'temperature: {temperature}\n'
            f'top_p: {top_p}\n'
            f'top_k: {top_k}\n'
            f'num_beams: {num_beams}\n'
            f'truncation_length: {truncation_length}\n'
        )

    @abstractmethod
    def fine_tune(self):
        pass

    @abstractmethod
    def call(self, prompt) -> str:
        pass

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)

    def get_model_parameter(self, prompt: str = '') -> ModelParameter:
        return ModelParameter(
            user_input=prompt,
            max_new_tokens=self._max_new_tokens,
            auto_max_new_tokens=self._auto_max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            num_beams=self._num_beams,
            truncation_length=self._truncation_length
        )


