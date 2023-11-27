import requests

from models.model_wrapper import CausalLanguageModelWrapper


class CausalLanguageModelApi(CausalLanguageModelWrapper):
    def __init__(
            self,
            server_name,
            *args,
            **kwargs
    ):
        super(CausalLanguageModelApi, self).__init__(*args, **kwargs)

        self._end_point = f'http://{server_name}:5000/api/v1/chat'
        self.get_logger().info(
            f'server_name: {server_name}\n'
        )

    def fine_tune(self):
        raise RuntimeError("CausalLanguageModelApi doesn't support fine-tuning currently.")

    def call(self, prompt):
        request = self.get_model_parameter(prompt=prompt).to_dict()
        response = requests.post(self._end_point, json=request)
        if response.status_code == 200:
            result = response.json()['results'][0]['history']
            answer = result['visible'][-1][1]
        else:
            answer = 'Unknown'
        return answer
