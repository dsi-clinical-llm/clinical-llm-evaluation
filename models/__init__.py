from models.chatgpt_wrapper import CausalLanguageModelChatGPT
from models.endpoint_model_wrapper import CausalLanguageModelApi
from models.hf_model_wrapper import CausalLanguageModelHuggingFace

model_wrappers = [
    CausalLanguageModelChatGPT,
    CausalLanguageModelApi,
    CausalLanguageModelHuggingFace
]


def get_all_model_names():
    return [model_wrapper.get_name() for model_wrapper in model_wrappers]


def get_model_wrapper(
        model_name
):
    for model_wrapper in model_wrappers:
        if model_wrapper.get_name() == model_name:
            return model_wrapper

    raise RuntimeError(f'{model_name} is not a model wrapper')
