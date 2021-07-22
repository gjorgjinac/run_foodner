###Code for running the FoodNER models
Pretrained models should be downloaded from https://portal.ijs.si/nextcloud/s/BqGow5foJspLmb3
and placed in a new directory trained_models/foodner

In the main.py file, an example is shown for running the model on a set of abstracts in the dataset1/abstracts.csv file.
The model outputs are saved in the dataset1/foodner.csv file

The models that are going to be run are defined in the extractors/foodner/task_tag_lists.py file
