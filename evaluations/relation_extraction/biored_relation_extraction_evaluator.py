import json
import re
import numpy as np
from typing import List, Union
from utils.utils import extract_json_from_text

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.relation_extraction.biored.biored_re_prompt import BioRedRelationExtractionPrompt


class BioRedMetric:

    @staticmethod
    def standardize_json_string(json_string):
        cleaned_json_string = json_string.replace('array([', '[')
        cleaned_json_string = re.sub('](\s)*,(\s)*(\n)*dtype=object\)', '],', cleaned_json_string)
        cleaned_json_string = cleaned_json_string.replace('None', '""')
        cleaned_json_string = re.sub('\],\n*\s+\]', '],]', cleaned_json_string)
        cleaned_json_string = cleaned_json_string.strip()
        if cleaned_json_string[-3:] == '],]':
            cleaned_json_string = cleaned_json_string[:-3] + ']]'
        return cleaned_json_string

    @staticmethod
    def recursive_add_relations(relations_json, total_relations_json=[]):
        if isinstance(relations_json, Union[list, np.ndarray]):
            for sub_relation_json in relations_json:
                BioRedMetric.recursive_add_relations(sub_relation_json, total_relations_json)
        elif isinstance(relations_json, dict):
            total_relations_json.append(relations_json)

    @staticmethod
    def convert_to_triplets(list_of_relations):
        list_of_triplets = []
        for relations_json in list(list_of_relations):
            triplets = set()
            if isinstance(relations_json, str):
                relations_json = BioRedMetric.standardize_json_string(relations_json)
                relations_json = extract_json_from_text(relations_json)

            # Continue if there aren't any relations capatured
            if relations_json is None or len(relations_json) == 0:
                continue

            relations_json = BioRedRelationExtractionPrompt.extract_entity_relations(relations_json)

            # Recursively unflatten the nested structure
            unflattened_relations_json = []
            BioRedMetric.recursive_add_relations(relations_json, unflattened_relations_json)

            for relation_json in unflattened_relations_json:
                # This prevents any entry from being None and causing problems in constructing the triplets
                relation_json = {k: (v or '') for k, v in relation_json.items()}
                if 'entity1_identifier' not in relation_json:
                    relation_json['entity1_identifier'] = ''
                if 'relation' not in relation_json:
                    relation_json['relation'] = ''
                if 'entity2_identifier' not in relation_json:
                    relation_json['entity2_identifier'] = ''
                triplets.add(
                    f'{relation_json["entity1_identifier"]}-{relation_json["relation"]}-{relation_json["entity2_identifier"]}')
                triplets.add(
                    f'{relation_json["entity2_identifier"]}-{relation_json["relation"]}-{relation_json["entity1_identifier"]}')
            list_of_triplets.append(triplets)
        return list_of_triplets


class BioRedRecall(BioRedMetric):

    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        all_pred_triplets = BioRedMetric.convert_to_triplets(predictions)
        all_true_triplets = BioRedMetric.convert_to_triplets(references)

        recalls = []
        for pred_triplets_per_record, true_triplets_per_record in zip(all_pred_triplets, all_true_triplets):
            intersection_set = pred_triplets_per_record & true_triplets_per_record
            recall = float(len(intersection_set)) / len(true_triplets_per_record)
            recalls.append(recall)
        return {'recall': np.asarray(recalls)}


class BioRedPrecision(BioRedMetric):
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        all_pred_triplets = BioRedMetric.convert_to_triplets(predictions)
        all_true_triplets = BioRedMetric.convert_to_triplets(references)

        precisions = []
        for pred_triplets_per_record, true_triplets_per_record in zip(all_pred_triplets, all_true_triplets):
            intersection_set = pred_triplets_per_record & true_triplets_per_record
            if len(pred_triplets_per_record) > 0:
                precision = float(len(intersection_set)) / len(pred_triplets_per_record)
            else:
                precision = 0
            precisions.append(precision)
        return {'precision': np.asarray(precisions)}


class BioRedF1(BioRedMetric):

    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        recall_metric = BioRedRecall.compute(predictions, references)
        precision_metric = BioRedPrecision.compute(predictions, references)
        recall = recall_metric['recall']
        precision = precision_metric['precision']
        denominator = recall + precision
        numerator = 2 * recall * precision
        f1 = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        return {'f1': f1}


class BioRedRelationExtractionEvaluator(CausalLanguageModelEvaluator):

    def get_prompt_classes(self) -> List[Prompt]:
        return [
            BioRedRelationExtractionPrompt
        ]

    def generate_prompts(
            self,
            record,
            few_shot_records
    ) -> List[Prompt]:
        identifier = record['document_id'] if 'document_id' in record else None
        passage = record['passage']
        entities = record['entities']
        relations = record['mapped_relations']

        list_of_entities = []
        for entity in entities:
            list_of_entities.append({
                "identifier": entity['identifier'],
                "entity_name": entity['text'],
                "entity_type": entity['type'],
                "offset": entity['offset']
            })

        list_of_entities_str = json.dumps(list_of_entities)
        prompts = []
        for prompt_class in self.get_prompt_classes():
            prompt = prompt_class(
                ground_truth=relations,
                record_id=identifier,
                enable_chatgpt_utility=self._enable_chatgpt_utility,
                data={'list_of_entities': list_of_entities_str, 'passage': passage}
            )
            prompts.append(prompt)

        return prompts

    def get_metrics(self) -> List[dict]:
        return {'recall': BioRedRecall, 'precision': BioRedPrecision, 'f1': BioRedF1}, {}
