from jinja2 import Environment

from models import CausalLanguageModelChatGPT
from prompt_templates.utils.JsonExtractPromptTemplate import JSON_EXTRACTION_PROMPT_TEMPLATE

ENVIRONMENT = Environment()
json_extract_prompt_template = ENVIRONMENT.from_string(JSON_EXTRACTION_PROMPT_TEMPLATE)


class ChatGptUtility(CausalLanguageModelChatGPT):

    def extract_json_using_chatgpt(
            self,
            text: str
    ):
        prompt = json_extract_prompt_template.render(text=text)
        return self.call(prompt)
