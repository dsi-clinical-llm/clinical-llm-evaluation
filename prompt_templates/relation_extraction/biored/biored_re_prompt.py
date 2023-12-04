import json
from jinja2 import Template, Environment

from utils.utils import extract_json_from_text
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.relation_extraction.biored.biored_re_prompt_template import BIORED_RE_PROMPT_TEMPLATE_BASE
from models.chatgpt_wrapper import CausalLanguageModelChatGPT

ENVIRONMENT = Environment()


class BioRedRelationExtractionPrompt(Prompt):
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
            extracted_relations = json_object.get('entity_relations', [])
        elif isinstance(json_object, list):
            extracted_relations = json_object

        if len(extracted_relations) == 0:
            extracted_relations.append({})

        for relation in extracted_relations:
            if 'entity1_identifier' not in relation:
                relation['entity1_identifier'] = ''
            if 'entity2_identifier' not in relation:
                relation['entity2_identifier'] = ''
            if 'relation' not in relation:
                relation['relation'] = ''

        return extracted_relations
