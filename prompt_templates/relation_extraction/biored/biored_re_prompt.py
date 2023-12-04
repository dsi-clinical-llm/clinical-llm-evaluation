import json
from jinja2 import Template, Environment

from utils.utils import extract_json_from_text
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.relation_extraction.biored.biored_re_prompt_template import BIORED_RE_PROMPT_TEMPLATE_BASE

ENVIRONMENT = Environment()


class BioRedRelationExtractionPrompt(Prompt):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(BIORED_RE_PROMPT_TEMPLATE_BASE)

    def extract_answer(
            self
    ):
        extracted_relations = []

        json_object = extract_json_from_text(self.model_response)
        if isinstance(json_object, dict):
            extracted_relations = json_object.get('entity_relations', [])
        elif isinstance(json_object, list):
            extracted_relations = json_object

        for relation in extracted_relations:
            if 'entity1' not in relation:
                relation['entity_1'] = ''
            if 'entity2' not in relation:
                relation['entity_2'] = ''
            if 'relation' not in relation:
                relation['relation'] = ''
        return extracted_relations
