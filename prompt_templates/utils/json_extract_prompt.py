import json
from jinja2 import Template, Environment

from utils.utils import extract_json_from_text
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.utils.json_extract_prompt_template import JSON_EXTRACTION_PROMPT_TEMPLATE

ENVIRONMENT = Environment()


class JsonExtractPrompt(Prompt):

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(JSON_EXTRACTION_PROMPT_TEMPLATE)

    def extract_answer(
            self
    ):
        json_string = extract_json_from_text(self.model_response)
        return json.dumps(json_string)
