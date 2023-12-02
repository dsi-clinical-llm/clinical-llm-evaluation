from typing import List

import evaluate
import numpy as np

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.qa.qa_prompt import MedQuADQuestionAnswerPromptV1, MedQuADQuestionAnswerPromptBaseV1


class PointWiseScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        match_score = np.sum((predictions == references).astype(int) * 1)
        mismatch_score = - np.sum((predictions != references).astype(int) * 0.25)
        return match_score + mismatch_score


class MedQuADQaEvaluator(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            MedQuADQuestionAnswerPromptBaseV1,
            MedQuADQuestionAnswerPromptV1,
        ]


    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[Prompt]:

        identifier = record['question_id']
        question = record['question']
        label = record['answer']
        prompts = []
        for prompt_class in self.get_prompt_classes():
            few_shot_examples = self.prepare_few_shot_records(few_shot_records, prompt_class.map_answer)
            prompt = prompt_class(
                ground_truth=label,
                record_id=identifier,
                data={'question': question, 'examples': few_shot_examples}
            )
            prompts.append(prompt)
        return prompts



    def get_metrics(self) -> dict:
        # Return the regular metrics for the QA task
        rouge = evaluate.load('rouge')
        bleu = evaluate.load('bleu')
        meteor = evaluate.load('meteor')
        return {'rouge': rouge, 'bleu': bleu, 'meteor': meteor}, {}

    @staticmethod
    def prepare_few_shot_records(few_shot_records, answer_mapping_func):
        few_shot_examples = []
        if few_shot_records:
            for record in few_shot_records:
                question = record['question']
                label = record['answer']
                few_shot_examples.append({
                    'question': question,
                    'answer': label
                })
        return few_shot_examples
