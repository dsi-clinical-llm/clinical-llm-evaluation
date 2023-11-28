from typing import List
from jinja2 import Template, Environment

from prompt_templates.prompt_abstract import NestedPrompt, Prompt
from prompt_templates.summarization.summarization_prompt_template import SUMMARIZATION_PROMPT_TEMPLATE, \
    NESTED_SUMMARIZATION_PROMPT_TEMPLATE

ENVIRONMENT = Environment()


class SummarizationBasePrompt(Prompt):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(SUMMARIZATION_PROMPT_TEMPLATE)

    def extract_answer(
            self
    ):
        return self.model_response


class SummarizationNestedPrompt(NestedPrompt):
    def __init__(
            self,
            ground_truth: str,
            record_id: str = None,
            prompts: List[SummarizationBasePrompt] = []
    ):
        self.ground_truth = ground_truth
        self.record_id = record_id
        self.prompts = prompts

    def get_nested_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(NESTED_SUMMARIZATION_PROMPT_TEMPLATE)

    @staticmethod
    def get_base_prompt_class() -> Prompt:
        return SummarizationBasePrompt

    def get_prompt(self):
        summaries = [prompt.model_response for prompt in self.prompts]
        return self.get_nested_prompt_template().render(summaries=summaries)

    def extract_answer(
            self
    ):
        return self.model_response
