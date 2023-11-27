import inspect
import os
import re
import glob
import json
import pyarrow.parquet as pq
from datasets import DatasetDict

regex = re.compile('[^a-zA-Z]')


def extract_from_json_response(
        model_response,
        field,
        default_value='unknown'
) -> str:
    final_answer = None
    response = model_response.replace("&quot;", "\"").replace("'", "\"").replace('"', "\"")
    try:
        json_object = json.loads(response)
        final_answer = str(json_object.get(field, default_value))
        final_answer = remove_non_utf8_characters(final_answer)
    except Exception as e:
        print(e)

    # Try to target the json object
    if not final_answer:
        left_bracket_index = response.find('{')
        right_bracket_index = response.find('}')
        # Assuming the first match is the JSON string
        json_string = response[left_bracket_index:right_bracket_index + 1]
        try:
            json_object = json.loads(json_string)
            final_answer = str(json_object.get(field, default_value))
            final_answer = remove_non_utf8_characters(final_answer)
        except Exception as e:
            print(e)

    return final_answer if final_answer else default_value


def escape_double_quotes(text):
    return text.replace("&quot;", "\"").replace('"', "\"").replace("'", "\"")


def remove_double_quotes(text):
    return text.replace("&quot;", "").replace('"', "").replace("'", "")


def remove_non_utf8_characters(s):
    return s.encode('utf-8', 'ignore').decode('utf-8')


def remove_illegal_chars(text):
    return regex.sub('', text)


def find_and_delete_corrupted_parquet_files(parquet_folder):
    def is_parquet_file_corrupted(file_path):
        try:
            # Try reading the Parquet file
            pq.read_table(file_path)
            return False
        except Exception as e:
            # If an error occurs, the file might be corrupted
            print(f"Error occurred: {e}")
            return True

    # Create a search pattern for all .parquet files
    search_pattern = f"{parquet_folder}/*.parquet"

    # Use glob to find files matching the pattern
    for parquet_file in glob.glob(search_pattern):
        if is_parquet_file_corrupted(parquet_file):
            print(f"Deleting corrupted file: {parquet_file}")
            os.remove(parquet_file)


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


def get_all_args(cls):
    """
    Iteratively get all the required arguments and arguments with a default value from the current class and all its
    ancestor

    classes :param
    cls: :return:
    """
    required_args = []
    default_args = []

    # Iterate over the MRO in reverse (excluding 'object')
    for c in reversed(cls.__mro__[:-1]):
        if hasattr(c, '__init__'):
            constructor_signature = inspect.signature(c.__init__)
            for name, parameter in constructor_signature.parameters.items():
                if name in ['self', 'args', 'kwargs']:
                    continue

                if parameter.default is inspect.Parameter.empty:
                    if name not in required_args:
                        required_args.append(name)

                if parameter.default is not inspect.Parameter.empty:
                    if name not in default_args:
                        default_args.append(name)

    return required_args, default_args
