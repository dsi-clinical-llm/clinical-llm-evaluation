import json
from jinja2 import Template, Environment
from typing import Dict, Union

from utils.llm_dataclasses import SentenceMatch, SentenceMatchingData
from utils.utils import remove_non_utf8_characters, escape_double_quotes
from prompt_templates.prompt_abstract import Prompt
from prompt_templates.hallucination.summarization.summarization_hallucination_template import \
    SUMMARIZATION_HALLUCINATION_PROMPT_TEMPLATE

ENVIRONMENT = Environment()


class SummarizationHallucinationPrompt(Prompt):
    def get_prompt_template(self) -> Template:
        return ENVIRONMENT.from_string(SUMMARIZATION_HALLUCINATION_PROMPT_TEMPLATE)

    def to_dict(self) -> Dict[str, Union[str, int]]:
        matching_data = self.extract_answer()
        # We try to find the number of high quality matches (High/Moderate matches)
        n_matches = set()
        for match in matching_data.matches:
            if match.similarity_score in ['High', 'Moderate']:
                n_matches.add(match.summary_sent_no)

        return {
            'record_id': self.record_id,
            'prompt': self.prompt,
            'ground_truth': matching_data.summary_total,
            'mapped_ground_true': matching_data.summary_total,
            'model_response': remove_non_utf8_characters(self.model_response),
            'answer': len(n_matches),
            'mapped_answer': len(n_matches)
        }

    def extract_answer(
            self
    ):
        matching_data = SentenceMatchingData()
        response = escape_double_quotes(self.model_response)
        try:
            json_object = json.loads(response)
            matching_data = SentenceMatchingData(
                matches=[SentenceMatch(**match) for match in json_object["matches"]],
                no_matches=json_object["no_matches"],
                summary_total=json_object["summary_total"],
                original_text_total=json_object["original_text_total"]
            )
        except Exception as e:
            print(e)
        return matching_data
