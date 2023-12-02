from typing import List

import evaluate

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.ner.pubmed_ner.pubmed_ner_prompt import PubmedNameEntityRecognition


class PubmedNameEntityRecognitionEvaluator(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            PubmedNameEntityRecognition
        ]

    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[Prompt]:
        identifier = record['abstract_number'] if 'abstract_number' in record else None
        title = record['title']
        abstract = record['abstract']
        context = title + " " + abstract
        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth='',
                record_id=identifier,
                data={'context': context}
            )
            prompts.append(prompt)

        return prompts

    def get_metrics(self) -> List[dict]:
        return {}, {}
