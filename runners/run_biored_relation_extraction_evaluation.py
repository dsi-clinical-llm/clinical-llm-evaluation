import json
import pandas as pd
from multiprocessing import Pool
from evaluations.relation_extraction.biored_relation_extraction_evaluator import BioRedRelationExtractionEvaluator

from datasets import DatasetDict, Dataset
from utils.utils import create_train_test_partitions
from utils.evaluation_args import add_main_arguments
from run_qa_evaluation import parallel_evaluation


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(description='BioRed evaluation')
    add_main_arguments(parser)
    parser.add_argument(
        '--local_dataset',
        dest='local_dataset',
        action='store',
        required=True
    )
    return parser.parse_args()


def create_biored_pd(file_path):
    # Function to map entity identifiers to their texts
    def map_entities_to_texts(entity_list, identifier):
        for entity in entity_list:
            if entity['identifier'] == identifier:
                return entity['text']
        return identifier

    # Load the JSON data from file
    with open(file_path, 'r') as file:
        data = json.load(file)

    document_ids = []
    combined_passages = []
    combined_entities = []
    document_relations = []

    for document in data['documents']:
        # Combining texts and annotations (entities) for all passages in the document
        document_ids.append(document['id'])
        all_texts = []
        all_entities = []
        for passage in document['passages']:
            all_texts.append(passage['text'])
            annotations = passage.get('annotations', [])
            for ann in annotations:
                for location in ann['locations']:
                    entity_info = {
                        'id': ann['id'],
                        'text': ann['text'],
                        'type': ann['infons']['type'],
                        'offset': location['offset'],
                        'length': location['length'],
                        'identifier': ann['infons']['identifier']
                    }
                all_entities.append(entity_info)

        # Combining all texts into a single passage
        combined_passage_text = ' '.join(all_texts)
        combined_passages.append(combined_passage_text)
        combined_entities.append(all_entities)

        # Extracting and formatting relations for the document
        relations = []
        for relation in document['relations']:
            entity1_id = relation['infons']['entity1']
            entity2_id = relation['infons']['entity2']
            relation_type = relation['infons']['type']

            # Finding the texts for the entities in the relations
            entity1_text = next((e['text'] for e in all_entities if e['id'] == entity1_id), entity1_id)
            entity2_text = next((e['text'] for e in all_entities if e['id'] == entity2_id), entity2_id)

            relations.append({'entity1': entity1_text, 'entity2': entity2_text, 'relation': relation_type})

        # Adding the same list of relations for the entire document
        document_relations.append(relations)

    # Creating the initial DataFrame
    df_final_combined_with_types = pd.DataFrame({
        'document_id': document_ids,
        'passage': combined_passages,
        'entities': combined_entities,
        'relations': document_relations
    })

    # Modifying the relations to replace identifiers with actual entity texts
    mapped_relations = []
    for index, row in df_final_combined_with_types.iterrows():
        modified_relations = []
        for relation in row['relations']:
            entity1_text = map_entities_to_texts(row['entities'], relation['entity1'])
            entity2_text = map_entities_to_texts(row['entities'], relation['entity2'])
            modified_relations.append({
                'entity1_identifier': relation['entity1'],
                'entity2_identifier': relation['entity2'],
                'entity1': entity1_text,
                'entity2': entity2_text,
                'relation': relation['relation']
            })
        mapped_relations.append(modified_relations)

    # Adding the 'mapped_relations' column to the DataFrame
    df_final_combined_with_types['mapped_relations'] = mapped_relations

    return df_final_combined_with_types


if __name__ == '__main__':

    args = create_argparser()
    biored_pd = create_biored_pd(args.local_dataset)
    dataset = Dataset.from_pandas(biored_pd)
    biored_dataset = DatasetDict({'train': dataset, 'test': dataset})

    partitioned_datasets = create_train_test_partitions(
        biored_dataset,
        n=args.num_of_cores
    )

    pool_tuples = []
    for i in range(args.num_of_cores):
        pool_tuples.append(
            (i, args, partitioned_datasets[i], BioRedRelationExtractionEvaluator)
        )

    with Pool(processes=args.num_of_cores) as p:
        p.starmap(parallel_evaluation, pool_tuples)
        p.close()
        p.join()
    print('Done')
