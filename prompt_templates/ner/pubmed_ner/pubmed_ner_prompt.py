from jinja2 import Template, Environment

from prompt_templates.prompt_abstract import Prompt
from prompt_templates.ner.pubmed_ner.pubmed_ner_prompt_template import PUBMED_NER_PROMPT_TEMPLATE_BASE

ENVIRONMENT = Environment()


class PubmedNameEntityRecognition(Prompt):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_NER_PROMPT_TEMPLATE_BASE)

    def extract_answer(
            self
    ):
        return self.model_response.split(',')
