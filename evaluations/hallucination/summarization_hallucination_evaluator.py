from typing import List

import numpy as np
from datasets import Dataset

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.hallucination.summarization.summarization_hallucination_prompt import \
    SummarizationHallucinationPrompt
from utils.utils import remove_double_quotes


class HallucinationScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        hallucination_scores = 1 - predictions / references
        return {'hallucination_score': np.mean(hallucination_scores)}


class SimilarityScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        similarity_scores = np.mean(predictions / references)
        return {'similarity_score': similarity_scores}


class SummarizationHallucinationEvaluator(CausalLanguageModelEvaluator):

    def __init__(
            self,
            original_text_field,
            summary_field,
            *args, **kwargs
    ):
        super(SummarizationHallucinationEvaluator, self).__init__(*args, **kwargs)
        self._original_text_field = original_text_field
        self._summary_field = summary_field
        self.get_logger().info(
            f'original_text_field: {original_text_field}\n'
            f'summary_field: {summary_field}\n'
        )

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            SummarizationHallucinationPrompt
        ]

    def generate_prompts(self, record, few_shot_records: Dataset = None) -> List[Prompt]:
        identifier = record['record_id']
        original_text = remove_double_quotes(record[self._original_text_field])
        summary = remove_double_quotes(record[self._summary_field])

        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth='',
                record_id=identifier,
                data={'summary': summary, 'original_text': original_text}
            )
            prompts.append(prompt)
        return prompts

    def get_metrics(self):
        return {'similarity_score': SimilarityScore, 'hallucination_score': HallucinationScore}, {}
