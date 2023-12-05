import sys
from models import get_all_model_names, CausalLanguageModelApi, CausalLanguageModelChatGPT, \
    CausalLanguageModelHuggingFace
from utils.llm_dataclasses import instruction_template_choices, DEFAULT_INSTRUCTION_TEMPLATE


def add_main_arguments(parser):
    parser.add_argument(
        '--evaluation_folder',
        dest='evaluation_folder',
        action='store',
        help='The path for your evaluation_folder',
        required=True
    )
    parser.add_argument(
        '--num_of_cores',
        dest='num_of_cores',
        action='store',
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        '--skip_metrics',
        dest='skip_metrics',
        action='store_true'
    )
    parser.add_argument(
        '--enable_chatgpt_utility',
        dest='enable_chatgpt_utility',
        action='store_true'
    )
    parser.add_argument(
        '--is_hallucination_test',
        dest='is_hallucination_test',
        action='store_true',
    )
    parser.add_argument(
        '--model_choice',
        dest='model_choice',
        choices=get_all_model_names(),
        required=True
    )
    parser.add_argument(
        '--max_new_tokens',
        dest='max_new_tokens',
        required=False,
        type=int,
        default=512
    )
    parser.add_argument(
        '--instruction_template',
        dest='instruction_template',
        required=False,
        choices=instruction_template_choices,
        default=DEFAULT_INSTRUCTION_TEMPLATE
    )
    parser.add_argument(
        '--truncation_length',
        dest='truncation_length',
        required=False,
        type=int,
        default=32768
    )
    parser.add_argument(
        '--top_p',
        dest='top_p',
        required=False,
        type=float,
        default=0.9
    )
    parser.add_argument(
        '--restore_checkpoint',
        dest='restore_checkpoint',
        action='store_true',
    )

    endpoint_model_parser = parser.add_argument_group(CausalLanguageModelApi.get_name())
    endpoint_model_parser.add_argument(
        '--server_name',
        dest='server_name',
        action='store',
        help='Servername for the LLM API',
        required=CausalLanguageModelApi.get_name() in sys.argv
    )
    chatgpt_model_parser = parser.add_argument_group(CausalLanguageModelChatGPT.get_name())
    chatgpt_model_parser.add_argument(
        '--chatgpt_model',
        dest='chatgpt_model',
        action='store',
        choices=CausalLanguageModelChatGPT.model_choices,
        help='The ChatGPT to use',
        required=CausalLanguageModelChatGPT.get_name() in sys.argv
    )

    hf_model_parser = parser.add_argument_group(CausalLanguageModelHuggingFace.get_name())
    hf_model_parser.add_argument(
        '--device',
        dest='device',
        action='store',
        choices=['auto', 'cpu', 'none'],
        default='auto',
        required=False
    )
    hf_model_parser.add_argument(
        '--model_name_or_path',
        dest='model_name_or_path',
        action='store',
        required=CausalLanguageModelHuggingFace.get_name() in sys.argv
    )
    return parser
