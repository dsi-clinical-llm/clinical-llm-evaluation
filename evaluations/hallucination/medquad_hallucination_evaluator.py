from typing import List

import evaluate
import numpy as np

from prompt_templates.prompt_abstract import Prompt
from evaluations.qa.medquad_evaluator import MedQuADQaEvaluator
from prompt_templates.hallucination.qa.qa_hallucination_prompt import MedQuADQuestionAnswerHallucinationPrompt, Open_End_QA_HALLUCINATION_PROMPT_V1


class HallucinationScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        index = (references != 0)
        hallucination_scores = 1 - predictions[index] / references[index]
        return {'hallucination_score': np.mean(hallucination_scores)}


class SimilarityScore:
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        index = (references != 0)
        similarity_scores = np.mean(predictions[index] / references[index])
        return {'similarity_score': similarity_scores}




class MedQuADQaHallucinationEvaluator(MedQuADQaEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            MedQuADQuestionAnswerHallucinationPrompt
        ]

    def get_metrics(self):
        return {'similarity_score': SimilarityScore, 'hallucination_score': HallucinationScore}, {}



