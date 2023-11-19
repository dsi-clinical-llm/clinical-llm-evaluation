from jinja2 import Template, Environment

from utils.utils import extract_from_json_response
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.memorization.snomed_recall_template import SNOMED_CODE_RECALL_PROMPT

ENVIRONMENT = Environment()


class SnomedCodeRecallPrompt(Prompt):

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(SNOMED_CODE_RECALL_PROMPT)

    def extract_answer(
            self
    ):
        answer = extract_from_json_response(
            model_response=self.model_response,
            field='snomed_code',
            default_value='unknown'
        )
        return answer

    def is_fine_tunable(self):
        pass
