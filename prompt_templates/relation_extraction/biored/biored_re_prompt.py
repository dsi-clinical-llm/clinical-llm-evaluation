import json
from jinja2 import Template, Environment

from utils.utils import extract_json_from_text
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.relation_extraction.biored.biored_re_prompt_template import BIORED_RE_PROMPT_TEMPLATE_BASE

ENVIRONMENT = Environment()


class BioRedRelationExtractionPrompt(Prompt):
    output_columns = [
        'entity1_identifier',
        'entity2_identifier',
        'entity1',
        'entity2',
        'relation'
    ]

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(BIORED_RE_PROMPT_TEMPLATE_BASE)

    def extract_answer(
            self
    ):
        extracted_relations = []

        if self.enable_chatgpt_utility:
            json_object = extract_json_from_text(
                self.chatgpt_utility.extract_json_using_chatgpt(self.model_response)
            )
        else:
            json_object = extract_json_from_text(self.model_response)

        if isinstance(json_object, dict):
            for potential_key in ['entity_relations', 'relations', 'relationships']:
                extracted_relations = json_object.get(potential_key, [])
                if len(extracted_relations) > 0:
                    break
        elif isinstance(json_object, list):
            extracted_relations = json_object

        if len(extracted_relations) == 0:
            extracted_relations.append({})

        for relation in extracted_relations:
            if 'entity1_identifier' not in relation:
                relation['entity1_identifier'] = ''
            if 'entity2_identifier' not in relation:
                relation['entity2_identifier'] = ''
            if 'entity1' not in relation:
                relation['entity1'] = ''
            if 'entity2' not in relation:
                relation['entity2'] = ''
            if 'relation' not in relation:
                relation['relation'] = ''

            for column in self.output_columns:
                relation[column] = relation[column] or ''

        cleaned_relations = []
        for relation in extracted_relations:
            cleaned_relations.append(
                [{k: v} for k, v in relation.items() if k in self.output_columns]
            )
        return json.dumps(cleaned_relations)
