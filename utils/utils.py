from datasets import DatasetDict


def split_dataset(dataset, n):
    split_size = len(dataset) // n
    # Split the dataset
    splits = []
    for i in range(n):
        start = i * split_size
        # For the last split, take all the remaining data
        end = (i + 1) * split_size if i < n - 1 else len(dataset)
        splits.append(dataset.select(range(start, end)))
    return splits


def create_train_test_partitions(
        dataset, n
):
    if n == 1:
        return [dataset]

    train_partitions = split_dataset(dataset['train'], n)
    test_partitions = split_dataset(dataset['test'], n)
    datasets = []
    for train, test in zip(train_partitions, test_partitions):
        datasets.append(
            DatasetDict({
                'train': train,
                'test': test
            })
        )
    return datasets
