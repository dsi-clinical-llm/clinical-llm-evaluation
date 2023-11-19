from multiprocessing import Pool
from evaluations import PubMedQaEvaluator
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from evaluations.hallucination.pubmedqa_hallucination_evaluator import PubMedQaHallucinationEvaluator
from models import get_all_model_names, get_model_wrapper, CausalLanguageModelApi

from datasets import load_dataset
from utils.utils import create_train_test_partitions, get_all_args

TEST_SIZE = 0.2
RANDOM_SEED = 42


def create_argparser():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='QA evaluation')
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

    endpoint_model_parser = parser.add_argument_group(CausalLanguageModelApi.get_name())
    endpoint_model_parser.add_argument(
        '--server_name',
        dest='server_name',
        action='store',
        help='Servername for the LLM API',
        required=CausalLanguageModelApi.get_name() in sys.argv
    )
    return parser.parse_args()


def main(
        process_id,
        parsed_args,
        sub_dataset,
        evaluator_class: CausalLanguageModelEvaluator
):
    model_wrapper_class = get_model_wrapper(parsed_args.model_choice)
    required_args, default_args = get_all_args(model_wrapper_class)

    arg_dict = {}
    for required_arg in required_args:
        if not hasattr(parsed_args, required_arg):
            raise RuntimeError(f'{required_arg} is a required argument for {parsed_args.model_choice}')
        arg_dict[required_arg] = getattr(parsed_args, required_arg)

    for default_arg in default_args:
        if hasattr(parsed_args, default_arg):
            arg_dict[default_arg] = getattr(parsed_args, default_arg)

    arg_dict['max_new_tokens'] = parsed_args.max_new_tokens

    model_wrapper = model_wrapper_class(**arg_dict)
    evaluator = evaluator_class(
        dataset=sub_dataset,
        model=model_wrapper,
        evaluation_folder=parsed_args.evaluation_folder,
        process_id=process_id
    )
    evaluator.evaluate()


if __name__ == '__main__':

    args = create_argparser()
    pubmed_qa_dataset = load_dataset('pubmed_qa', 'pqa_labeled')

    dataset = pubmed_qa_dataset['train'].train_test_split(
        seed=RANDOM_SEED,
        test_size=TEST_SIZE
    )

    partitioned_datasets = create_train_test_partitions(
        dataset,
        n=args.num_of_cores
    )

    # Get the evaluator
    pubmed_evaluator_class = PubMedQaHallucinationEvaluator if args.is_hallucination_test else PubMedQaEvaluator

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], pubmed_evaluator_class)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(main, pool_tuples)
        p.close()
        p.join()

    print('Done')
