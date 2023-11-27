import os
import random
from multiprocessing import Pool
from evaluations.summarization.summarization_evaluator import SummarizationEvaluator

from datasets import load_dataset, load_from_disk
from utils.utils import create_train_test_partitions
from utils.evaluation_args import add_main_arguments
from run_qa_evaluation import parallel_evaluation, RANDOM_SEED, TEST_SIZE


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Summarization evaluation')
    add_main_arguments(parser)
    parser.add_argument(
        '--local_dataset',
        dest='local_dataset',
        action='store',
        required=False
    )
    parser.add_argument(
        '--chunk_size',
        dest='chunk_size',
        action='store',
        type=int,
        default=1000,
        required=False
    )
    parser.add_argument(
        '--chunk_overlap',
        dest='chunk_overlap',
        action='store',
        type=int,
        default=20,
        required=False
    )
    parser.add_argument(
        '--num_of_words',
        dest='num_of_words',
        action='store',
        type=int,
        default=200,
        required=False
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = create_argparser()
    if args.local_dataset and os.path.exists(args.local_dataset):
        summarization_dataset = load_from_disk(args.local_dataset)
    else:
        summarization_dataset = load_dataset('ccdv/pubmed-summarization')

    test_size = len(summarization_dataset['test'])
    random.seed(RANDOM_SEED)
    random_indexes = random.sample(range(test_size), 100)
    summarization_dataset['test'] = summarization_dataset['test'].select(random_indexes)

    partitioned_datasets = create_train_test_partitions(
        summarization_dataset,
        n=args.num_of_cores
    )

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], SummarizationEvaluator,
             args.chunk_size, args.chunk_overlap, args.num_of_words)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(parallel_evaluation, pool_tuples)
        p.close()
        p.join()

    print('Done')
