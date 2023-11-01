import os
from datetime import datetime

import evaluate
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator, ENVIRONMENT
from prompt_templates import PUBMED_QA_PROMPT_TEMPLATE_V1


class PubMedQaEvaluator(CausalLanguageModelEvaluator):

    def __init__(self, *args, **kwargs):
        pubmed_qa_dataset = load_dataset('pubmed_qa', 'pqa_labeled')
        super(PubMedQaEvaluator, self).__init__(dataset=pubmed_qa_dataset, *args, **kwargs)

    def evaluate(self):

        batch_of_prompts = []
        generated_answers = []
        labels = []
        for record in tqdm(self._dataset['test']):
            question = record['question']
            abstract = '\n'.join(record['context']['contexts'])
            labels.append(record['final_decision'])
            prompt = self.get_prompt_template().render(question=question, abstract=abstract)
            batch_of_prompts.append(prompt)
            if len(batch_of_prompts) >= self._batch_size:
                results = self._model.call(batch_of_prompts)
                generated_answers.extend(self.extract_answer(results))
                batch_of_prompts.clear()

        # If there is left-over in the array, send the list to the model again
        if len(batch_of_prompts) > 0:
            results = self._model.call(batch_of_prompts)
            generated_answers.extend(self.extract_answer(results))
            batch_of_prompts.clear()

        self.compute_metrics(generated_answers, labels)

    def compute_metrics(
            self,
            generated_answers,
            labels
    ):
        metrics = {}
        labels_one_hot = self.one_hot_encode(labels)
        generated_answers_one_hot = self.one_hot_encode(generated_answers)
        for _, metric in self.get_metrics().items():
            metrics.update(
                metric.compute(
                    predictions=generated_answers_one_hot,
                    references=labels_one_hot,
                    average='micro'
                )
            )
        metrics['total'] = len(labels)
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        metrics['time'] = current_time
        output_path = os.path.join(self._evaluation_folder, f'{current_time}.parquet')
        pd.DataFrame([metrics]).to_parquet(output_path)

    def one_hot_encode(self, labels):
        mapper = self.get_label_mapping()
        return [mapper.get(label.lower(), 3) for label in labels]

    @staticmethod
    def get_label_mapping():
        return {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}

    @staticmethod
    def extract_answer(batch_of_answers):
        extracted_answers = []
        for answer in batch_of_answers:
            answer_parts = answer.split('##Final Decision Field:')
            if len(answer_parts) > 1:
                final_answer = answer_parts[-1].lower()
            else:
                final_answer = 'unknown'
            extracted_answers.append(final_answer)
        return extracted_answers

    @staticmethod
    def get_metrics() -> dict:
        recall_metric = evaluate.load('recall')
        precision_metric = evaluate.load('precision')
        f1_metric = evaluate.load('f1')
        return {'recall': recall_metric, 'precision': precision_metric, 'f1': f1_metric}

    def get_prompt_template(self):
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_V1)
