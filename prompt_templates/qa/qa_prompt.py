import json
from jinja2 import Template, Environment

from utils.utils import remove_illegal_chars, remove_non_utf8_characters
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.qa.pubmed_qa_prompt_template import PUBMED_QA_PROMPT_TEMPLATE_JSON_V3, \
    PUBMED_QA_PROMPT_TEMPLATE_JSON_V1, PUBMED_QA_PROMPT_TEMPLATE_JSON_V2, PUBMED_QA_PROMPT_TEMPLATE_BASE_V1

ENVIRONMENT = Environment()


class PubmedQuestionAnswerPromptJsonV1(Prompt):
    # dict to map answer to an integer
    answer_mapping = {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_JSON_V1)

    def extract_answer(
            self
    ):
        final_answer = None
        if self.model_response:
            # This is done for loading the json object
            response = self.model_response.replace("&quot;", "\"")
            try:
                json_object = json.loads(response)
                final_answer = str(json_object.get('correct_option', 'unknown'))
                final_answer = remove_non_utf8_characters(remove_illegal_chars(final_answer))
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
                    final_answer = str(json_object.get('correct_option', 'unknown'))
                    final_answer = remove_non_utf8_characters(remove_illegal_chars(final_answer))
                except Exception as e:
                    print(e)

        return final_answer if final_answer else 'unknown'

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptJsonV1.answer_mapping.get(answer.lower().strip(), 3)

    def is_fine_tunable(self):
        return True


class PubmedQuestionAnswerPromptJsonV2(PubmedQuestionAnswerPromptJsonV1):

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_JSON_V2)


class PubmedQuestionAnswerPromptJsonV3(PubmedQuestionAnswerPromptJsonV1):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_JSON_V3)


class PubmedQuestionAnswerPromptBaseV1(Prompt):
    # dict to map answer to an integer
    answer_mapping = {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_BASE_V1)

    def extract_answer(
            self
    ):
        if self.model_response:
            parsed_answer = self.model_response.lower()
            parsed_answer = parsed_answer.replace('answer', '').replace('option', '')
            # Take the first word from the answer
            final_answer = parsed_answer.split(' ')[0]
            final_answer = remove_illegal_chars(final_answer)
        else:
            final_answer = 'unknown'

        return final_answer

    def is_fine_tunable(self):
        return False

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptJsonV1.answer_mapping.get(answer.lower().strip(), 3)

    def is_fine_tunable(self):
        return True
