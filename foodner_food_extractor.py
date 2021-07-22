import os
from collections import defaultdict
from typing import List

import numpy as np
import spacy
import torch
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from transformers import BertTokenizer, BertForTokenClassification

from config import use_gpu
from extractors.extractor_base_classes import EntityExtractor
from extractors.foodner.task_tag_lists import task_tag_values


class FoodnerFoodExtractor(EntityExtractor):

    def __init__(self, save_extractions=False):
        super().__init__('foodner', save_extractions=save_extractions)
        self.initialize_models()
        self.english_model = spacy.load('en_core_web_sm')

    def extract_entity(self, doc: Doc) -> List[Span]:
        extractions = []
        for sentence_index, sentence_to_process in enumerate(list(doc.sents)):
            food_entities = self.find_food_entities(sentence_to_process.text)
            if len(food_entities)>0:
                food_spans = self.__convert_food_words_to_doc( sentence_to_process,food_entities,sentence_index)
                extractions+=food_spans
        return extractions

    def find_food_entities(self, text) -> List:
        tokenized_sentence = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokenized_sentence])
        if use_gpu:
            input_ids = input_ids.cuda()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        food_tokens = []
        task_annotations = {}
        for task_idx, task in enumerate(self.model_names):
            model = self.models[task]

            with torch.no_grad():
                output = model(input_ids)

            label_indices = np.argmax(output[0].cpu().numpy(), axis=2)
            tag_values = task_tag_values.get(task)
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(tag_values[label_idx])
                    new_tokens.append(token)
            task_annotations[task] = new_labels
            if task == 'food-classification':
                food_tokens = new_tokens

        new_tokens_labels = list(zip(food_tokens, task_annotations['food-classification']))
        previous_label = 'O'
        entities = []
        current_entity = []
        current_entity_annotations = defaultdict(lambda x: [])

        for index, token_label_tuple in enumerate(new_tokens_labels):
            current_label = token_label_tuple[1]
            if current_label != 'O':
                if current_label == 'I-FOOD' and previous_label != 'O':
                    current_entity.append(token_label_tuple[0])
                    for task in task_tag_values.keys():
                        if task_annotations[task][index] != 'O':
                            bi_tags_removed=self.__remove_bi_tags(task_annotations[task][index])
                            current_entity_annotations[task].append(bi_tags_removed)
                else:
                    current_entity_annotations['text'] = current_entity.copy()
                    if current_entity_annotations['text'] is not None and current_entity_annotations['text'] != []:
                        for task in task_tag_values.keys():
                            current_entity_annotations[task] = list(current_entity_annotations[task]) if current_entity_annotations[task] is not None else []
                        entities.append(current_entity_annotations.copy())
                    current_entity = [token_label_tuple[0]]
                    current_entity_annotations = {'start': index - 1}
                    for task in task_tag_values.keys():
                        annotation_list = list({self.__remove_bi_tags(task_annotations[task][index])}) if \
                        task_annotations[task][
                            index] != 'O' else []
                        current_entity_annotations[task] = annotation_list

            previous_label = current_label
        entities=list(filter(lambda x: x is not None, entities))
        return entities

    def __remove_bi_tags(self, tag):
        return tag.replace('B-', '').replace('I-', '')

    def __string_value_from_dict_or_none(self, dict, dict_key):
        if dict_key in dict.keys():
            return dict[dict_key] if type(dict[dict_key]) not in [list, None] else '***'.join(dict[dict_key])
        return None

    def __add_tags_to_span(self, food_span, processed_word):
        food_span._.foodon = self.__string_value_from_dict_or_none(processed_word, 'foodon')
        # food_span._.hansard = self.__string_value_from_dict_or_none(processed_word, 'hansard')
        food_span._.hansardClosest = self.__string_value_from_dict_or_none(processed_word, 'hansard-closest')
        food_span._.hansardParent = self.__string_value_from_dict_or_none(processed_word, 'hansard-parent')
        food_span._.snomedct = self.__string_value_from_dict_or_none(processed_word, 'snomedct')
        return food_span

    def __convert_food_words_to_doc(self, sentence, food_words, sentence_index):
        food_spans = []

        for processed_word in food_words:
            food_span = sentence[processed_word['start']: processed_word['start'] + len(processed_word['text'])]
            food_span._.sentence_index=sentence_index
            food_span = self.__add_tags_to_span(food_span, processed_word)
            food_spans.append(food_span)
        return food_spans

    def initialize_models(self):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        epochs = 50
        self.models = {}
        self.model_names = list(task_tag_values.keys())

        for task_idx, task in enumerate(self.model_names):
            model = BertForTokenClassification.from_pretrained(
                os.path.join('trained_models', 'foodner', f'bert-model-{task}-e{epochs}'),
                num_labels=len(task_tag_values[task]),
                output_attentions=False,
                output_hidden_states=False)
            if use_gpu:
                model.cuda()
            self.models[task] = model
