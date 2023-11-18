from multiprocessing import Pool
from evaluations import CausalLanguageModelApi, PubMedQaEvaluator
from evaluations.hallucination.pubmedqa_hallucination_evaluator import PubMedQaHallucinationEvaluator

from datasets import load_dataset
from utils.utils import create_train_test_partitions

TEST_SIZE = 0.2
RANDOM_SEED = 42


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
    return parser.parse_args()


def main(
        process_id,
        server_name,
        sub_dataset,
        evaluation_folder,
        is_hallucination_test=False
):
    model_wrapper = CausalLanguageModelApi(
        server_name=server_name
    )
    evaluator_class = PubMedQaHallucinationEvaluator if is_hallucination_test else PubMedQaEvaluator
    evaluator = evaluator_class(
        dataset=sub_dataset,
        model=model_wrapper,
        evaluation_folder=evaluation_folder,
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

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args.server_name, partitioned_datasets[i], args.evaluation_folder, args.is_hallucination_test)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(main, pool_tuples)
        p.close()
        p.join()

    print('Done')
