import inspect
import re
import json
from datasets import DatasetDict

regex = re.compile('[^a-zA-Z]')


def extract_from_json_response(
        model_response,
        field,
        default_value='unknown'
) -> str:
    final_answer = None
    response = model_response.replace("&quot;", "\"").replace("'", "\"")
    try:
        json_object = json.loads(response)
        final_answer = str(json_object.get(field, default_value))
        final_answer = remove_non_utf8_characters(remove_illegal_chars(final_answer))
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
            final_answer = remove_non_utf8_characters(remove_illegal_chars(final_answer))
        except Exception as e:
            print(e)

    return final_answer if final_answer else default_value


def remove_non_utf8_characters(s):
    return s.encode('utf-8', 'ignore').decode('utf-8')


def remove_illegal_chars(text):
    return regex.sub('', text)


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
