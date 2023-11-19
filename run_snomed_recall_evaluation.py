from multiprocessing import Pool

from datasets import load_from_disk
from utils.evaluation_args import add_main_arguments
from utils.utils import create_train_test_partitions
from evaluations.memorization.snomed_recall_evaluator import SnomedRecallEvaluator
from run_qa_evaluation import main


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='SNOMED Code Recall Evaluation')
    parser = add_main_arguments(parser)
    parser.add_argument(
        '--dataset_path',
        dest='dataset_path',
        action='store',
        help='The path for your SNOMED Code dataset',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = create_argparser()
    snomed_dataset = load_from_disk(args.dataset_path)
    snomed_dataset['test'] = snomed_dataset['train']

    partitioned_datasets = create_train_test_partitions(
        snomed_dataset,
        n=args.num_of_cores
    )

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], SnomedRecallEvaluator)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(main, pool_tuples)
        p.close()
        p.join()

    print('Done')
