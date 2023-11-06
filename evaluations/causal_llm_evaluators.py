from evaluations.model_wrappers import CausalLanguageModelWrapper

from abc import ABC, abstractmethod
from typing import Union
import logging
import jinja2

from datasets.dataset_dict import DatasetDict

RANDOM_SEED = 42
ENVIRONMENT = jinja2.Environment()


class CausalLanguageModelEvaluator(ABC):

    def __init__(
            self,
            evaluation_folder: str,
            model: CausalLanguageModelWrapper,
            dataset: DatasetDict,
            batch_size: int = 1,
            fine_tune_required: bool = False,
            n_of_shots: int = 0,
            test_size: Union[float, int] = 0.2
    ):
        self._evaluation_folder = evaluation_folder
        self._model = model
        self._dataset = dataset
        self._batch_size = batch_size
        self._fine_tune_required = fine_tune_required
        self._n_of_shots = n_of_shots
        self._test_size = int(test_size) if test_size > 1 else test_size

        self.get_logger().info(
            f'evaluation_folder: {evaluation_folder}\n'
            f'model: {model}\n'
            f'dataset: {dataset}\n'
            f'batch_size: {batch_size}\n'
            f'fine_tune_required: {fine_tune_required}\n'
            f'n_of_shots: {n_of_shots}\n'
        )

        if 'test' not in self._dataset:
            self._dataset = self._dataset['train'].train_test_split(seed=RANDOM_SEED, test_size=self._test_size)

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def get_prompt_template(self):
        pass

    @abstractmethod
    def compute_metrics(self, generated_answers, labels):
        pass

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)
