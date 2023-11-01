from evaluations import CausalLanguageModelApi, PubMedQaEvaluator


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='QA evaluation')
    parser.add_argument(
        '--server_name',
        dest='server_name',
        action='store',
        help='Servername for the LLM API',
        required=True
    )
    parser.add_argument(
        '--evaluation_folder',
        dest='evaluation_folder',
        action='store',
        help='The path for your evaluation_folder',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = create_argparser()
    model_wrapper = CausalLanguageModelApi(
        server_name=args.server_name
    )
    evaluator = PubMedQaEvaluator(
        model=model_wrapper,
        evaluation_folder=args.evaluation_folder
    )
    evaluator.evaluate()
