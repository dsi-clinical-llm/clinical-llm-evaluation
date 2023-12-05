from typing import List

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.utils.json_extract_prompt import JsonExtractPrompt


class JsonExtractTask(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            JsonExtractPrompt
        ]

    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[Prompt]:
        identifier = record['record_id']
        ground_truth = record['ground_truth']
        model_response = record['model_response']
        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth=ground_truth,
                record_id=identifier,
                data={'text': model_response}
            )
            prompts.append(prompt)
        return prompts

    def get_metrics(self):
        return {}, {}
