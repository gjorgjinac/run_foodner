import os
import sys
import traceback
import pandas as pd
import spacy

from foodner_food_extractor import FoodnerFoodExtractor


def run_model_and_save(ids_and_abstracts, dataset):
    save_directory=dataset
    english_model=spacy.load('en_core_web_sm')

    for extractor in [FoodnerFoodExtractor(save_extractions=True)]:
        print(extractor.name)
        df_to_save = pd.DataFrame()
        index_of_last_processed_abstract = 0
        j = 0
        while j < len(ids_and_abstracts):
            processed_files_name = os.path.join(save_directory,
                                                '{extractor_name}_{i}.csv'.format(extractor_name=extractor.name, i=j))
            if os.path.isfile(processed_files_name):
                index_of_last_processed_abstract = j
            j += 1000
        if index_of_last_processed_abstract != 0:
            df_to_save = pd.read_csv(
                os.path.join(save_directory, '{extractor_name}_{i}.csv'.format(extractor_name=extractor.name,
                                                                               i=index_of_last_processed_abstract)),
                index_col=[0])

        print(f'Starting from abstract: {index_of_last_processed_abstract}')
        i = index_of_last_processed_abstract
        save_file = os.path.join(save_directory, '{extractor_name}.csv'.format(extractor_name=extractor.name))

        if not os.path.isfile(save_file):
            for (file_name, file_content) in ids_and_abstracts[i:]:
                if not type(file_content) is str:
                    print(f'Skipping: {file_name}, following content is not a string: {file_content}')
                    continue
                doc = english_model(file_content)
                print(i)
                i += 1
                file_name = str(file_name)
                try:
                    extracted_df = extractor.extract(doc, file_name, dataset, save_entities=False)
                    extracted_df['extractor'] = extractor.name
                    extracted_df['file_name'] = file_name
                    df_to_save = df_to_save.append(extracted_df)
                except:
                    print('Error happened')
                    traceback.print_exc(file=sys.stdout)
                if i % 1000 == 0 and extractor.save_extractions:
                    df_to_save.drop_duplicates().to_csv(os.path.join(save_directory, '{extractor_name}_{i}.csv'.format(
                        extractor_name=extractor.name, i=i)))
            if df_to_save.shape[0] == 0:
                df_to_save = pd.DataFrame(
                    columns=['start_char', 'end_char', 'entity_type', 'entity_id', 'text', 'sentence',
                             'sentence_index', 'extractor', 'file_name'])
            df_to_save.drop_duplicates().to_csv(save_file)
        else:
            print('File already exists: {0}'.format(save_file))