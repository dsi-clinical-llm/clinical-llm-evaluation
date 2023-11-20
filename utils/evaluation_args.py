import sys
from models import get_all_model_names, CausalLanguageModelApi


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
    return parser
