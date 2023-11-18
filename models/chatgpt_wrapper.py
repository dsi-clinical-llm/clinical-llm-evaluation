import os
from openai import OpenAI
from models.model_wrapper import CausalLanguageModelWrapper


class CausalLanguageModelChatGPT(CausalLanguageModelWrapper):
    model = "gpt-3.5-turbo"

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(CausalLanguageModelChatGPT, self).__init__(*args, **kwargs)
        self.openai_client = OpenAI(
            api_key=os.environ.get('OPEN_AI_KEY')
        )

    def fine_tune(self):
        raise NotImplemented('This capability has not been implemented yet')

    def call(self, prompt) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': 'You are a medical professional.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=self._max_new_tokens
        )
        return completion.choices[0].message.content
