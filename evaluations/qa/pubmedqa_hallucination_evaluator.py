from typing import List

import evaluate
import numpy as np

from prompt_templates.prompt_abstract import Prompt
from evaluations.qa.pubmedqa_evaluator import PubMedQaEvaluator
from prompt_templates.hallucination.qa_hallucination_prompt import PubmedQuestionAnswerHallucinationPrompt


class PointWiseScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        match_score = np.sum((predictions == references).astype(int) * 1)
        mismatch_score = - np.sum((predictions != references).astype(int) * 0.25)
        return {'point_wise_score': match_score + mismatch_score}


class PubMedQaHallucinationEvaluator(PubMedQaEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            PubmedQuestionAnswerHallucinationPrompt
        ]

    def get_metrics(self):
        accuracy_metric = evaluate.load('accuracy')
        return {'accuracy': accuracy_metric, 'point_wise_score': PointWiseScore}, {}
