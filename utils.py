import os
from typing import List
import pandas as pd

from spacy.lang.en import English

from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from pandas.errors import EmptyDataError

from config import global_output_directory_name


class PandasUtil:
    @staticmethod
    def write_object_list_as_dataframe_file(object_list: List, file_name: str, file_directory: str = '.',
                                            df_separator: str = ',', columns: List[str] = None):
        #dictionary_list = [object.__dict__ for object in object_list]
        FileUtil.create_directory_if_not_exists(file_directory)
        df_to_write = pd.DataFrame(object_list)
        if columns is not None:
            df_to_write = pd.DataFrame(object_list, columns=columns)
        df_to_write.to_csv(f'{file_directory}/{file_name}', sep=df_separator)

    @staticmethod
    def read_dataframe_file_as_object_list(file_name: str, file_directory: str = '.', df_separator: str = ',', nan_replacement = None):
        read_df = PandasUtil.read_dataframe_file(file_name, file_directory, df_separator, nan_replacement)
        return read_df.to_dict(orient='records')

    @staticmethod
    def read_dataframe_file(file_name: str, file_directory: str = '.', df_separator: str = ',', nan_replacement = None):
        try:
            read_df = pd.read_csv(os.path.join(file_directory, file_name), sep = df_separator, index_col=[0])
            if nan_replacement is not None:
                read_df = read_df.fillna(nan_replacement)
            return read_df
        except EmptyDataError:
            print('empty data error')
            return pd.DataFrame()
class PrintUtil:
    @staticmethod
    def print_objects_with_labels(objects_with_labels: List):
        for object_with_label in objects_with_labels:
            print(f'{object_with_label[0]}:\n{object_with_label[1]}')
        print()


class TextProcessingUtil:
    @staticmethod
    def split_into_sentences(text: str) -> List[Span]:
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        document = nlp(text)
        return list(filter(lambda s: not TextProcessingUtil.is_empty(s.text.strip()), document.sents))

    @staticmethod
    def is_empty(text) -> bool:
        return len(text.strip()) == 0

    @staticmethod
    def get_text_between_words(sentence, word1, word2) -> str:
        word1_index = sentence.find(word1)
        word2_index = sentence.find(word2)
        if word1_index < word2_index:
            return sentence[word1_index + len(word1): word2_index]
        else:
            return sentence[word2_index + len(word2): word1_index]

    @staticmethod
    def remove_text_between_brackets(text) -> str:
        while True:
            start = text.find('(')
            end = text.find(')')
            if start != -1 and end != -1:
                text = text[start + 1:end]
            else:
                break
        return text

class FileUtil:
    @staticmethod
    def read_file(file_name: str, file_directory: str = '.') -> str:
        file_path = f'{file_directory}/{file_name}'
        file = open(file_path, mode='r', encoding='utf-8' )
        file_content = file.read()
        file.close()
        return file_content

    @staticmethod
    def create_directory_if_not_exists(directory:str):
        if not os.path.isdir(directory):
            os.makedirs(os.path.join(os.getcwd(), directory))



def get_dataset_output_dir(dataset):
    return os.path.join(global_output_directory_name, dataset)

def write_to_dataset_dir(df, file_name, dataset):
    dataset_output_dir = get_dataset_output_dir(dataset)
    df.to_csv(os.path.join(dataset_output_dir, file_name))

def read_from_dataset_dir(file_name, dataset):
    dataset_output_dir = get_dataset_output_dir(dataset)
    return pd.read_csv(os.path.join(dataset_output_dir, file_name), index_col=[0])


def add_span_extensions():
    Doc.set_extension("relations", default=None)
    Doc.set_extension("entities", default=None)
    for span_extension in ['entity_type', 'entity_id', 'foodon', 'hansard', 'hansardClosest', 'hansardParent',
                           'snomedct', 'synonyms','sentence_index']:
        Span.set_extension(span_extension, default=None)

def save_as_latex_table(df, file_name):
    with open(file_name, "w") as f_out:
        f_out.write(f'{df.to_latex(float_format="{:0.0f}".format)}')