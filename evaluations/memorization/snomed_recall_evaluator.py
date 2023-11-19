from typing import List

import numpy as np

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.memorization.snomed_recall_prompt import SnomedCodeRecallPrompt


class SnomedRecallAccuracy:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        match_score = np.sum((predictions == references).astype(int) * 1)
        accuracy = match_score / len(predictions)
        return {'accuracy': accuracy}


class SnomedRecallPointWiseScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        match_score = np.sum((predictions == references).astype(int) * 1)
        certain_index = predictions != 'uncertain'
        mismatch_score = - np.sum((predictions != references).astype(int) * 0.25 * certain_index)
        return {'point_wise_score': match_score + mismatch_score}


class SnomedRecallEvaluator(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            SnomedCodeRecallPrompt
        ]

    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[Prompt]:
        identifier = record['concept_id']
        concept_name = record['concept_name']
        label = str(record['snomed_code'])
        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth=label,
                record_id=identifier,
                data={'concept_name': concept_name}
            )
            prompts.append(prompt)
        return prompts

    def get_metrics(self):
        return {'accuracy': SnomedRecallAccuracy, 'point_wise_score': SnomedRecallPointWiseScore}, {}
