from multiprocessing import Pool

import pandas as pd

from evaluations.other.json_estract_task import JsonExtractTask

from utils.utils import create_train_test_partitions
from utils.evaluation_args import add_main_arguments
from run_qa_evaluation import parallel_evaluation


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='JSON extraction task')
    add_main_arguments(parser)
    parser.add_argument(
        '--input_folder',
        dest='input_folder',
        action='store',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    from datasets import DatasetDict, Dataset

    args = create_argparser()
    results_pd = pd.read_parquet(args.input_folder)
    dataset = Dataset.from_pandas(results_pd)
    dataset_dict = DatasetDict({'train': dataset, 'test': dataset})
    args.skip_metrics = True
    args.model_choice = 'CausalLanguageModelChatGPT'
    args.chatgpt_model = 'gpt-4-1106-preview'

    partitioned_datasets = create_train_test_partitions(
        dataset_dict,
        n=args.num_of_cores
    )

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], JsonExtractTask)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(parallel_evaluation, pool_tuples)
        p.close()
        p.join()
    print('Done')
