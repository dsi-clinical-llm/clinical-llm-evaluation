from models.model_wrapper import CausalLanguageModelWrapper

from abc import ABC, abstractmethod
from typing import List
from datetime import datetime
from tqdm import tqdm
import logging
import os
import random
from pathlib import Path
import pandas as pd

from datasets.dataset_dict import DatasetDict, Dataset
from prompt_templates.prompt_abstract import Prompt
from utils.utils import find_and_delete_corrupted_parquet_files


class CausalLanguageModelEvaluator(ABC):

    def __init__(
            self,
            evaluation_folder: str,
            model: CausalLanguageModelWrapper,
            dataset: DatasetDict,
            batch_size: int = 1,
            fine_tune_required: bool = False,
            n_of_shots: int = 0,
            process_id: int = None,
            is_hallucination_test: bool = False
    ):
        self._evaluation_folder = evaluation_folder
        self._model = model
        self._dataset = dataset
        self._batch_size = batch_size
        self._fine_tune_required = fine_tune_required
        self._n_of_shots = n_of_shots
        self._process_id = process_id
        self._is_hallucination_test = is_hallucination_test

        self.get_logger().info(
            f'evaluation_folder: {evaluation_folder}\n'
            f'model: {model}\n'
            f'dataset: {dataset}\n'
            f'batch_size: {batch_size}\n'
            f'fine_tune_required: {fine_tune_required}\n'
            f'n_of_shots: {n_of_shots}\n'
            f'is_hallucination_test :{is_hallucination_test}\n'
            f'process_id: {process_id}\n'
        )

        if 'test' not in self._dataset:
            raise RuntimeError("The dataset doesn't contain a test set")

    def evaluate(self):

        if self._fine_tune_required:
            # TODO
            pass

        train_size = len(self._dataset['train'])

        results = dict()
        for record in tqdm(self._dataset['test']):

            few_shot_records = None
            if self._n_of_shots > 0:
                # Randomly select indexes
                random_indexes = random.sample(range(train_size), self._n_of_shots)
                few_shot_records = self._dataset['train'].select(random_indexes)

            for prompt_container in self.generate_prompts(record, few_shot_records):
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
            # Remove the corrupted parquet files
            find_and_delete_corrupted_parquet_files(self.get_results_folder(prompt_type))

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
    def generate_prompts(self, record, few_shot_records: Dataset = None) -> List[Prompt]:
        pass

    @staticmethod
    def map_answer(answer):
        return answer

    def compute_metrics(
            self,
            prompt_type,
            generated_answers,
            labels
    ):
        Path(self.get_metrics_folder(prompt_type)).mkdir(parents=True, exist_ok=True)
        metrics = {}
        metric_functions, metric_kwargs = self.get_metrics()
        for _, metric in metric_functions.items():
            metrics.update(
                metric.compute(
                    predictions=generated_answers,
                    references=labels,
                    **metric_kwargs
                )
            )
        metrics['total'] = len(labels)
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        metrics['time'] = current_time
        output_path = os.path.join(self.get_metrics_folder(prompt_type), f'{current_time}.parquet')
        pd.DataFrame([metrics]).to_parquet(output_path)

    @abstractmethod
    def get_metrics(self):
        raise NotImplemented(f'get_metrics needs to be implemented for {self}')

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)
