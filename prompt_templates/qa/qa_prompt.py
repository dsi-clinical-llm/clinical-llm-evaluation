import re
import json
from jinja2 import Template, Environment

from prompt_templates.prompt_abstract import Prompt
from prompt_templates.qa.pubmed_qa_prompt_template import PUBMED_QA_PROMPT_TEMPLATE_COT_V1, \
    PUBMED_QA_PROMPT_TEMPLATE_BASE, PUBMED_QA_PROMPT_TEMPLATE_BASE_V1

ENVIRONMENT = Environment()

regex = re.compile('[^a-zA-Z]')


class PubmedQuestionAnswerPromptBase(Prompt):
    # dict to map answer to an integer
    answer_mapping = {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_BASE)

    def extract_answer(
            self
    ):
        final_answer = None
        if self.model_response:
            # This is done for loading the json object
            response = self.model_response.replace("&quot;", "\"")
            try:
                json_object = json.loads(response)
                final_answer = json_object.get('correct_option', 'unknown')
                final_answer = regex.sub('', final_answer)
            except Exception as e:
                print(e)

            # Try to target the json object
            if not final_answer:
                left_bracket_index = response.find('{')
                right_bracket_index = response.find('}')
                # Assuming the first match is the JSON string
                json_string = response[left_bracket_index:right_bracket_index + 1]
                try:
                    json_object = json.loads(json_string)
                    final_answer = json_object.get('correct_option', 'unknown')
                    final_answer = regex.sub('', final_answer)
                except Exception as e:
                    print(e)

        return final_answer if final_answer else 'unknown'

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptBase.answer_mapping.get(answer.lower().strip(), 3)

    def is_fine_tunable(self):
        return True


class PubmedQuestionAnswerPromptV1(PubmedQuestionAnswerPromptBase):

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_BASE_V1)


class PubmedQuestionAnswerPromptCotV1(PubmedQuestionAnswerPromptBase):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_COT_V1)
