from abc import ABC

from pandas import DataFrame
from spacy.language import Language


class Extractor(ABC):
    english_model: Language
    save_extractions: bool

    def __init__(self, save_extractions=True):
        self.english_model = spacy.load('en_core_web_sm')
        self.save_extractions = save_extractions

    def extract_from_text(self, text, *args):
        doc = self.english_model(text)
        return self.extract(doc, *args)

    def extract_from_file(self, file_name, file_directory, dataset='', *args):
        text = FileUtil.read_file(file_name, file_directory)
        # print(text)
        doc = self.english_model(text)
        return self.extract(doc, file_name, dataset, *args)

    def extract(self, *args):
        raise NotImplementedError()


from typing import List, Tuple, Union

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span

from extractors.file_manipulators import FileManipulator, EntityExtractorFileManipulator
from utils import FileUtil
import pandas as pd

class EntityExtractor(Extractor):
    name: str
    file_manipulator: FileManipulator

    def __init__(self, name, save_extractions=True):
        super(EntityExtractor, self).__init__(save_extractions)
        self.name = name
        self.file_manipulator = EntityExtractorFileManipulator()

    def extract_entity(self, doc: Doc) -> Union[List[Span], DataFrame]:
        raise NotImplementedError()

    def extract(self, doc: Doc, file_name=None, dataset_source='', save_entities=True) -> Union[
        list, DataFrame, List[Span]]:
        extracted_entities = self.extract_entity(doc)
        if type(extracted_entities) is list:
            extracted_entities = list(filter(lambda s: s is not None, extracted_entities))
            doc, objects_column_names = self.file_manipulator.prepare_doc_for_saving(doc, extracted_entities, self.name)
            return pd.DataFrame(doc._.entities, columns=objects_column_names)

        return extracted_entities


class EntityLinker:
    name: str
    entity_extractor: EntityExtractor

    def __init__(self, name:str, entity_extractor: EntityExtractor):
        self.name = name
        self.entity_extractor = entity_extractor

    def link(self, dataset:str):
        raise NotImplementedError()
