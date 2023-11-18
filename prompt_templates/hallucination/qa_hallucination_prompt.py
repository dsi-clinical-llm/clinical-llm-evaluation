import json
from jinja2 import Template, Environment

from prompt_templates.prompt_abstract import Prompt
from prompt_templates.hallucination.qa_hallucination_template import QA_HALLUCINATION_PROMPT_V1

ENVIRONMENT = Environment()


class PubmedQuestionAnswerHallucinationPrompt(Prompt):
    # dict to map answer to an integer
    answer_mapping = {'no': 0, 'yes': 1, 'unknown': 2}

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
                except Exception as e:
                    print(e)

        return final_answer if final_answer else 'unknown'

    @staticmethod
    def map_answer(answer):
        return PubmedQuestionAnswerHallucinationPrompt.answer_mapping.get(answer.lower().strip(), 2)

    def is_fine_tunable(self):
        pass
