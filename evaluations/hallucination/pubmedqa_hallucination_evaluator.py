from typing import List

import random
import numpy as np
import evaluate

from evaluations.qa.pubmedqa_evaluator import PubMedQaEvaluator
from prompt_templates.hallucination.qa_hallucination_prompt import PubmedQuestionAnswerHallucinationPrompt
from prompt_templates.prompt_abstract import Prompt

pubmed_qa_answers = ['yes', 'no', 'maybe']


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


class PubMedQaHallucinationEvaluator(PubMedQaEvaluator):
    def get_prompt_classes(self) -> List[Prompt]:
        return [
            PubmedQuestionAnswerHallucinationPrompt
        ]

    def generate_prompts(
            self,
            record
    ) -> List[Prompt]:
        identifier = record['pubid']
        question = record['question']
        abstract = '\n'.join(record['context']['contexts'])
        label = record['final_decision']
        prompts = []

        suggested_answer = random.choice(pubmed_qa_answers)
        ground_truth = 'yes' if suggested_answer == label else 'no'

        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth=ground_truth,
                record_id=identifier,
                data={'abstract': abstract, 'question': question, 'randomly_suggested_answer': suggested_answer}
            )
            prompts.append(prompt)
        return prompts

    @staticmethod
    def get_metrics() -> dict:
        accuracy_metric = evaluate.load('accuracy')
        return {'accuracy': accuracy_metric, 'point_wise_score': PointWiseScore}
