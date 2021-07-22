import spacy

from foodner_food_extractor import FoodnerFoodExtractor
from save import run_model_and_save
from utils import add_span_extensions
import pandas as pd
if __name__ == '__main__':
    add_span_extensions()
    dataset='dataset1'
    abstracts = pd.read_csv(f'{dataset}/abstracts.csv', index_col=[0])
    run_model_and_save(abstracts[['PMID', 'abstract']].values, dataset)