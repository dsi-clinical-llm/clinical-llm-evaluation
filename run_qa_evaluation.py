from multiprocessing import Pool
from evaluations import CausalLanguageModelApi, PubMedQaEvaluator

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
    return parser.parse_args()


def main(
        process_id,
        server_name,
        sub_dataset,
        evaluation_folder
):
    model_wrapper = CausalLanguageModelApi(
        server_name=server_name
    )
    evaluator = PubMedQaEvaluator(
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
            (i, args.server_name, partitioned_datasets[i], args.evaluation_folder)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(main, pool_tuples)
        p.close()
        p.join()

    print('Done')
