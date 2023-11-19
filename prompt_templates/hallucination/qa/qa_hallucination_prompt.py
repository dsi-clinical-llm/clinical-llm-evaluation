import json
import random
from jinja2 import Template, Environment
from typing import Union, Dict

from utils.utils import remove_illegal_chars
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.hallucination.qa.qa_hallucination_template import QA_HALLUCINATION_PROMPT_V1

ENVIRONMENT = Environment()


class PubmedQuestionAnswerHallucinationPrompt(Prompt):
    # dict to map answer to an integer
    hallucination_answer_mapping = {'no': 0, 'yes': 1, 'unknown': 2}
    pubmed_qa_answers = ['yes', 'no', 'maybe']

    def __init__(
            self,
            ground_truth: Union[str, int],
            data: Dict[str, str] = {},
            *args,
            **kwargs
    ):

        suggested_answer = random.choice(self.pubmed_qa_answers)
        ground_truth = 'yes' if suggested_answer == ground_truth else 'no'
        data['suggested_answer'] = suggested_answer

        super(PubmedQuestionAnswerHallucinationPrompt, self).__init__(
            ground_truth=ground_truth,
            data=data,
            *args,
            **kwargs
        )

    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(QA_HALLUCINATION_PROMPT_V1)

    def extract_answer(
            self
    ):
        final_answer = None
        if self.model_response:
            # This is done for loading the json object
            response = self.model_response.replace("&quot;", "\"")
            try:
                json_object = json.loads(response)
                final_answer = json_object.get('is_answer_correct', 'unknown')
                final_answer = remove_illegal_chars(final_answer)
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
                    final_answer = json_object.get('is_answer_correct', 'unknown')
                    final_answer = remove_illegal_chars(final_answer)
                except Exception as e:
                    print(e)

        return final_answer if final_answer else 'unknown'

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerHallucinationPrompt.hallucination_answer_mapping.get(answer.lower().strip(), 2)

    def is_fine_tunable(self):
        pass
