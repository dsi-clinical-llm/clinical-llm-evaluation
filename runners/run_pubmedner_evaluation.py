from multiprocessing import Pool
from evaluations.ner.pubmed_ner_evaluator import PubmedNameEntityRecognitionEvaluator

from datasets import load_from_disk
from utils.utils import create_train_test_partitions
from utils.evaluation_args import add_main_arguments
from run_qa_evaluation import parallel_evaluation


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Pubmed NER evaluation')
    add_main_arguments(parser)
    parser.add_argument(
        '--local_dataset',
        dest='local_dataset',
        action='store',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = create_argparser()
    pubmed_ner_dataset = load_from_disk(args.local_dataset)

    partitioned_datasets = create_train_test_partitions(
        pubmed_ner_dataset,
        n=args.num_of_cores
    )

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], PubmedNameEntityRecognitionEvaluator)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(parallel_evaluation, pool_tuples)
        p.close()
        p.join()
    print('Done')
