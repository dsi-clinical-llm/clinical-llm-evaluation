from transformers import pipeline, AutoTokenizer
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
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._pipeline = pipeline(
            'text-generation',
            model=model_name_or_path,
            tokenizer=tokenizer,
            max_length=self._truncation_length,
            device_map=self._device,
            use_fast=False,
            model_kwargs={"load_in_8bit": True}
        )

        self.get_logger().info(
            f'model_name_or_path: {model_name_or_path}\n'
            f'do_sample: {do_sample}\n'
        )

    def fine_tune(self):
        raise RuntimeError("CausalLanguageModelHuggingFace doesn't support fine-tuning currently.")

    def call(self, prompt):
        try:
            response = self._pipeline(
                prompt,
                do_sample=self._do_sample,
                top_p=self._top_p,
                max_new_tokens=self._max_new_tokens
            )
            return response[0]['generated_text'][len(prompt):]
        except Exception as e:
            self.get_logger().error(e)
            return ''
