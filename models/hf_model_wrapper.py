from transformers import pipeline

from models.model_wrapper import CausalLanguageModelWrapper


class CausalLanguageModelHuggingFace(CausalLanguageModelWrapper):

    def __init__(
            self,
            model_name_or_path,
            do_sample=True,
            device='auto',
            *args,
            **kwargs
    ):
        super(CausalLanguageModelHuggingFace, self).__init__(*args, **kwargs)

        self._do_sample = do_sample
        self._device = device
        self._pipeline = pipeline(
            'text-generation',
            model=model_name_or_path,
            max_length=self._truncation_length,
            device=self._device,
            model_kwargs={"load_in_8bit": True}
        )

        self.get_logger().info(
            f'model_name_or_path: {model_name_or_path}\n'
            f'do_sample: {do_sample}\n'
        )

    def fine_tune(self):
        raise RuntimeError("CausalLanguageModelHuggingFace doesn't support fine-tuning currently.")

    def call(self, prompt):
        response = self._pipelin(prompt, do_sample=self._do_sample)
        return response[len(prompt):]