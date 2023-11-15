from jinja2 import Template, Environment

from prompt_templates.prompt_abstract import Prompt
from prompt_templates.qa.pubmed_qa_prompt_template import PUBMED_QA_PROMPT_TEMPLATE_BASE_V1, \
    PUBMED_QA_PROMPT_TEMPLATE_BASE_V2, PUBMED_QA_PROMPT_TEMPLATE_COT_V1

ENVIRONMENT = Environment()


class PubmedQuestionAnswerPromptBase(Prompt):
    # dict to map answer to an integer
    answer_mapping = {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_BASE_V1)

    def extract_answer(
            self
    ):
        if self.model_response:
            answer_parts = self.model_response.lower().split('##final decision field:')
            if len(answer_parts) > 1:
                final_answer = answer_parts[-1].lower()
            else:
                final_answer = answer_parts[0]
            # Take the first word from the answer
            final_answer = final_answer.split(' ')[0]
        else:
            final_answer = 'unknown'
        return final_answer

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptBase.answer_mapping.get(answer.lower().strip(), 3)

    def is_fine_tunable(self):
        return True


class PubmedQuestionAnswerPromptV2(PubmedQuestionAnswerPromptBase):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_BASE_V2)

    def is_fine_tunable(self):
        return False


class PubmedQuestionAnswerPromptCotV1(PubmedQuestionAnswerPromptBase):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_COT_V1)

    def is_fine_tunable(self):
        return False
