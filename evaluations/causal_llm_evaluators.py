from models.model_wrappers import CausalLanguageModelWrapper

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from tqdm import tqdm
import logging
import os
from pathlib import Path
import pandas as pd

from datasets.dataset_dict import DatasetDict
from prompt_templates.prompt_abstract import Prompt


class CausalLanguageModelEvaluator(ABC):

    def __init__(
            self,
            evaluation_folder: str,
            model: CausalLanguageModelWrapper,
            dataset: DatasetDict,
            batch_size: int = 1,
            fine_tune_required: bool = False,
            n_of_shots: int = 0,
            process_id: int = None
    ):
        self._evaluation_folder = evaluation_folder
        self._model = model
        self._dataset = dataset
        self._batch_size = batch_size
        self._fine_tune_required = fine_tune_required
        self._n_of_shots = n_of_shots
        self._process_id = process_id

        self.get_logger().info(
            f'evaluation_folder: {evaluation_folder}\n'
            f'model: {model}\n'
            f'dataset: {dataset}\n'
            f'batch_size: {batch_size}\n'
            f'fine_tune_required: {fine_tune_required}\n'
            f'n_of_shots: {n_of_shots}\n'
            f'process_id: {process_id}\n'
        )

        if 'test' not in self._dataset:
            raise RuntimeError("The dataset doesn't contain a test set")

    def evaluate(self):

        if self._fine_tune_required:
            # TODO
            pass

        results = dict()
        for record in tqdm(self._dataset['test']):
            for prompt_container in self.generate_prompts(record):
                response = self._model.call(prompt_container.prompt)
                prompt_container.set_model_response(response)
                # Create an empty list if this prompt type is not in
                if prompt_container.get_prompt_type() not in results:
                    results[prompt_container.get_prompt_type()] = []
                results[prompt_container.get_prompt_type()].append(prompt_container)

            if self._process_id:
                print(f'Process id: {self._process_id}')

            # Flush the records whenever the list is filled up with the batch size number of records
            self.flush_records(results)
        # Final flush to the disk
        self.flush_records(results)

        for prompt_type in results.keys():
            results = pd.read_parquet(
                self.get_results_folder(prompt_type)
            )
            labels = results.mapped_ground_true.to_numpy()
            generated_answers = results.mapped_answer.to_numpy()
            self.compute_metrics(prompt_type, generated_answers, labels)

    def flush_records(self, results):
        for prompt_type, prompt_containers in results.items():
            # Create the folder and its parent folders if not exist
            Path(self.get_results_folder(prompt_type)).mkdir(parents=True, exist_ok=True)
            records = []
            if len(prompt_containers) >= self._batch_size:
                current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
                for prompt_container in prompt_containers:
                    record = prompt_container.to_dict()
                    record['time'] = current_time
                    records.append(record)
                output_path = os.path.join(self.get_results_folder(prompt_type), f'{current_time}.parquet')
                pd.DataFrame(records).to_parquet(output_path)
                # Clear the records in the memory
                prompt_containers.clear()

    def get_results_folder(self, prompt_type) -> str:
        return os.path.join(self._evaluation_folder, prompt_type, 'results')

    def get_metrics_folder(self, prompt_type) -> str:
        return os.path.join(self._evaluation_folder, prompt_type, 'metrics')

    @abstractmethod
    def get_prompt_classes(self) -> List[Prompt]:
        pass

    @abstractmethod
    def generate_prompts(self) -> List[Prompt]:
        pass

    @abstractmethod
    def compute_metrics(self, prompt_type, generated_answers, labels):
        pass

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)
