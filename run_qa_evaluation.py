from multiprocessing import Pool
from evaluations import PubMedQaEvaluator
from evaluations.causal_llm_evaluators import CausalLanguageModelEvaluator
from evaluations.hallucination.pubmedqa_hallucination_evaluator import PubMedQaHallucinationEvaluator
from models import get_model_wrapper

from datasets import load_dataset
from utils.utils import create_train_test_partitions, get_all_args
from utils.evaluation_args import add_main_arguments

TEST_SIZE = 0.2
RANDOM_SEED = 42


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='QA evaluation')
    add_main_arguments(parser)
    parser.add_argument(
        '--n_of_shots',
        dest='n_of_shots',
        action='store',
        type=int,
        default=0,
        required=False
    )
    return parser.parse_args()


def parallel_evaluation(
        process_id,
        parsed_args,
        sub_dataset,
        evaluator_class: CausalLanguageModelEvaluator,
        *positional_args,
        **kwargs
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
        n_of_shots=parsed_args.n_of_shots if hasattr(parsed_args, 'n_of_shots') else 0,
        restore_checkpoint=parsed_args.restore_checkpoint,
        process_id=process_id,
        *positional_args,
        **kwargs
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
        p.starmap(parallel_evaluation, pool_tuples)
        p.close()
        p.join()

    print('Done')
