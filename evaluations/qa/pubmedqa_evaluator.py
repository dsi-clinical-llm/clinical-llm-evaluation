import os
from pathlib import Path
from datetime import datetime
from typing import List

import evaluate
import pandas as pd
from datasets import load_dataset

from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.qa.qa_prompt import PubmedQuestionAnswerPromptBase, PubmedQuestionAnswerPromptV1


class PubMedQaEvaluator(CausalLanguageModelEvaluator):

    def __init__(self, *args, **kwargs):
        pubmed_qa_dataset = load_dataset('pubmed_qa', 'pqa_labeled')
        super(PubMedQaEvaluator, self).__init__(dataset=pubmed_qa_dataset, *args, **kwargs)

    def get_prompt_classes(self) -> List[PubmedQuestionAnswerPromptBase]:
        return [
            PubmedQuestionAnswerPromptBase,
            PubmedQuestionAnswerPromptV1
        ]

    def generate_prompts(
            self,
            record
    ) -> List[PubmedQuestionAnswerPromptBase]:

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

    def compute_metrics(
            self,
            prompt_type,
            generated_answers,
            labels,
            **kwargs
    ):
        Path(self.get_metrics_folder(prompt_type)).mkdir(parents=True, exist_ok=True)
        metrics = {}
        for _, metric in self.get_metrics().items():
            metrics.update(
                metric.compute(
                    predictions=generated_answers,
                    references=labels,
                    average='micro'
                )
            )
        metrics['total'] = len(labels)
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        metrics['time'] = current_time
        output_path = os.path.join(self.get_metrics_folder(prompt_type), f'{current_time}.parquet')
        pd.DataFrame([metrics]).to_parquet(output_path)

    @staticmethod
    def get_metrics() -> dict:
        recall_metric = evaluate.load('recall')
        precision_metric = evaluate.load('precision')
        f1_metric = evaluate.load('f1')
        return {'recall': recall_metric, 'precision': precision_metric, 'f1': f1_metric}
