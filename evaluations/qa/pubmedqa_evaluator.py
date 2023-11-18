from typing import List

import evaluate
import numpy as np

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.qa.qa_prompt import PubmedQuestionAnswerPromptJsonV1, \
    PubmedQuestionAnswerPromptJsonV2, PubmedQuestionAnswerPromptJsonV3, PubmedQuestionAnswerPromptBaseV1


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


class PubMedQaEvaluator(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            PubmedQuestionAnswerPromptBaseV1,
            PubmedQuestionAnswerPromptJsonV1,
            PubmedQuestionAnswerPromptJsonV2,
            PubmedQuestionAnswerPromptJsonV3
        ]

    def generate_prompts(
            self,
            record
    ) -> List[PubmedQuestionAnswerPromptJsonV1]:
        identifier = record['pubid']
        question = record['question']
        abstract = '\n'.join(record['context']['contexts'])
        label = record['final_decision']
        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth=label,
                record_id=identifier,
                data={'abstract': abstract, 'question': question}
            )
            prompts.append(prompt)
        return prompts

    def get_metrics(self) -> dict:
        # Return the regular metrics for the QA task
        recall_metric = evaluate.load('recall')
        precision_metric = evaluate.load('precision')
        f1_metric = evaluate.load('f1')
        return {'recall': recall_metric, 'precision': precision_metric, 'f1': f1_metric}, {'average': 'micro'}
