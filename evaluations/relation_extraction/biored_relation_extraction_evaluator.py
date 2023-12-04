import json
from typing import List

from prompt_templates.prompt_abstract import Prompt
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from prompt_templates.relation_extraction.biored.biored_re_prompt import BioRedRelationExtractionPrompt


class BioRedMetric:

    @staticmethod
    def convert_to_triplets(list_of_relations):
        triplets = set()
        for relation in list_of_relations:
            triplets.add(f'{relation["entity1_identifier"]}-{relation["relation"]}-{relation["entity2_identifier"]}')
            triplets.add(f'{relation["entity2_identifier"]}-{relation["relation"]}-{relation["entity1_identifier"]}')
        return triplets


class BioRedRecall(BioRedMetric):

    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        prediction_triplets = BioRedMetric.convert_to_triplets(predictions)
        reference_triplets = BioRedMetric.convert_to_triplets(references)
        intersection_set = prediction_triplets & reference_triplets

        recall = float(len(intersection_set)) / len(reference_triplets)
        return {'recall': recall}


class BioRedPrecision(BioRedMetric):
    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        prediction_triplets = BioRedMetric.convert_to_triplets(predictions)
        reference_triplets = BioRedMetric.convert_to_triplets(references)
        intersection_set = prediction_triplets & reference_triplets

        precision = float(len(intersection_set)) / len(prediction_triplets)
        return {'precision': precision}


class BioRedF1(BioRedMetric):

    @staticmethod
    def compute(
            predictions,
            references,
            **kwargs
    ):
        prediction_triplets = BioRedMetric.convert_to_triplets(predictions)
        reference_triplets = BioRedMetric.convert_to_triplets(references)
        intersection_set = prediction_triplets & reference_triplets

        recall = float(len(intersection_set)) / len(reference_triplets)
        precision = float(len(intersection_set)) / len(prediction_triplets)
        f1 = 2 * precision * recall / (precision + recall)
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
        relations = record['relations']

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
                data={'list_of_entities': list_of_entities_str, 'passage': passage}
            )
            prompts.append(prompt)

        return prompts

    def get_metrics(self) -> List[dict]:
        return {'recall': BioRedRecall, 'precision': BioRedPrecision, 'f1': BioRedF1}, {}
