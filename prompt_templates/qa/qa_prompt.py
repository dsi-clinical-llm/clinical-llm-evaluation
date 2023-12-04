from jinja2 import Template, Environment

from utils.utils import remove_illegal_chars, extract_from_json_response
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.qa.pubmed_qa_prompt_template import PUBMED_QA_PROMPT_TEMPLATE_JSON_V3, \
    PUBMED_QA_PROMPT_TEMPLATE_JSON_V1, PUBMED_QA_PROMPT_TEMPLATE_JSON_V2, PUBMED_QA_PROMPT_TEMPLATE_BASE_V1

ENVIRONMENT = Environment()


class PubmedQuestionAnswerPromptJsonV1(Prompt):
    # dict to map answer to an integer
    answer_idx_mapping = {'no': 0, 'yes': 1, 'maybe': 2, 'unknown': 3}
    idx_answer_mapping = {v: k for k, v in answer_idx_mapping.items()}

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(PUBMED_QA_PROMPT_TEMPLATE_JSON_V1)

    def extract_answer(
            self
    ):
        answer = extract_from_json_response(
            model_response=self.model_response,
            field='correct_option',
            default_value='unknown'
        )

        if answer == 'unknown':
            correct_option_index = extract_from_json_response(
                model_response=self.model_response,
                field='correct_option_index',
                default_value='unknown'
            )
            if correct_option_index.isnumeric():
                answer = self.idx_answer_mapping.get(int(correct_option_index), 'unknown')

        if answer == 'unknown':
            cleaned_model_response = self.model_response.strip().lower().replace('answer', '').replace('option', '')
            answer = self.recursive_parse(cleaned_model_response)

        return remove_illegal_chars(answer)

    def recursive_parse(self, cleaned_model_response):
        # first try to convert all the integer to the label
        if cleaned_model_response.isnumeric():
            if int(cleaned_model_response) in self.idx_answer_mapping:
                answer = self.idx_answer_mapping[int(cleaned_model_response)]
                return answer
            else:
                return 'unknown'
        else:
            # Handle the format like 1. Yes / Yes
            answer_parts = cleaned_model_response.split('.')
            if len(answer_parts) > 1:
                return self.recursive_parse(answer_parts[0])
            return answer_parts[0]

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptJsonV1.answer_idx_mapping.get(str(answer).lower().strip(), 3)


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
            final_answer = remove_illegal_chars(parsed_answer.split(' ')[-1])
        else:
            final_answer = 'unknown'

        return final_answer

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerPromptJsonV1.answer_idx_mapping.get(answer.lower().strip(), 3)
