from abc import ABC, abstractmethod
from typing import List
import requests
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
    def call(self, batch_of_prompts) -> List[str]:
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


class CausalLanguageModelApi(CausalLanguageModelWrapper):
    def __init__(
            self,
            server_name,
            *args,
            **kwargs
    ):
        super(CausalLanguageModelApi, self).__init__(*args, **kwargs)

        self._end_point = f'http://{server_name}:5000/api/v1/chat'
        self.get_logger().info(
            f'server_name: {server_name}\n'
        )

    def fine_tune(self):
        raise RuntimeError("CausalLanguageModelApi doesn't support fine-tuning currently.")

    def call(self, batch_of_prompts):
        results = []
        for prompt in batch_of_prompts:
            request = self.get_model_parameter(prompt=prompt).to_dict()
            response = requests.post(self._end_point, json=request)
            if response.status_code == 200:
                result = response.json()['results'][0]['history']
                answer = result['visible'][-1][1]
                results.append(answer.lower().strip())
            else:
                results.append('Unknown')
        return results
