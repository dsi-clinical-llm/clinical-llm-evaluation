from jinja2 import Template, Environment

from utils.utils import remove_illegal_chars, extract_from_json_response
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.qa.medquad_qa_prompt_template import MEDQUAD_QA_PROMPT_TEMPLATE_V1, \
    MEDQUAD_QA_PROMPT_TEMPLATE_BASE_V1

ENVIRONMENT = Environment()


class MedQuADQuestionAnswerPromptV1(Prompt):
    def __init__(
            self,
            ground_truth: str,
            record_id: str = None,
    ):
        self.ground_truth = ground_truth
        self.record_id = record_id

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(MEDQUAD_QA_PROMPT_TEMPLATE_V1)

    @staticmethod
    def get_base_prompt_class() -> Prompt:
        return MedQuADQuestionAnswerPromptV1

    def extract_answer(
            self
    ):
        return self.model_response


'''
class MedQuADQuestionAnswerPromptJsonV2(MedQuADQuestionAnswerPromptJsonV1):

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(MEDQUAD_QA_PROMPT_TEMPLATE_JSON_V2)

'''


class MedQuADQuestionAnswerPromptBaseV1(Prompt):
    def __init__(
            self,
            ground_truth: str,
            record_id: str = None,
    ):
        self.ground_truth = ground_truth
        self.record_id = record_id

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(MEDQUAD_QA_PROMPT_TEMPLATE_BASE_V1)

    @staticmethod
    def get_base_prompt_class() -> Prompt:
        return MedQuADQuestionAnswerPromptV1

    def extract_answer(
            self
    ):
        return self.model_response