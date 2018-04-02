import sys
#Path to Trane for imports
sys.path.append('/Users/Alexander/Documents/Trane/Trane__HDI_REPO')
path_to_datasets = '../Trane__Local_Misc/Formatted Datasets/Yelp Reviews v2/'
import pandas as pd
import trane
import json
import random
import datetime
import pickle
import featuretools as ft
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets
from sklearn.metrics import accuracy_score
from sklearn import metrics
import scikitplot as skplt

def save_obj(obj, name):
    with open('pickled_objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open('pickled_objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def generate_probs_and_nl(entity_id_column,
                            label_generating_column,
                            time_column,
                            table_meta,
                            filter_column,
                            is_pick_random_problems = True):
    generator = trane.PredictionProblemGenerator(table_meta, entity_id_column, label_generating_column, time_column, filter_column)
    probs = []

    all_probs = list(generator.generate())

    if is_pick_random_problems:
        random.shuffle(all_probs)

    for idx, prob in enumerate(all_probs):
        probs.append(prob)
        if idx + 1 == NUM_PROBLEMS_TO_GENERATE:
            break
    prediction_problems_json = trane.prediction_problems_to_json_file(
        probs, table_meta, entity_id_column, label_generating_column, time_column, "JSON Files/yelp_prediction_problems.json")

    nl_descrips = trane.generate_nl_description(
        probs, table_meta, entity_id_column, label_generating_column, time_column, trane.ConstantCutoffTime(0, 0))
    return probs, nl_descrips
def convert(str, format = None):
    return datetime.datetime.strptime(str, format)
def file_to_table_meta(filepath):
    return trane.TableMeta(json.loads(open(filepath).read()))

merged_df = load_obj('merged_df')

table_meta = file_to_table_meta(path_to_datasets + "meta.json")
entity_id_column = 'business_id'
label_generating_column = 'stars' #stars_x not stars because there are two columns with the name stars due to the merge
time_column = 'date'
filter_column = 'user_id'


NUM_PROBLEMS_TO_GENERATE = 50
probs, nl_descrips = generate_probs_and_nl(
                            entity_id_column, label_generating_column, 
                            time_column, table_meta, filter_column)

labeler = trane.Labeler()
entity_to_data_dict = trane.df_group_by_entity_id(merged_df, entity_id_column)
cutoff_time = datetime.date(2014, 1, 1)
entity_to_data_and_cutoff_dict = trane.ConstantCutoffTime(cutoff_time, cutoff_time).generate_cutoffs(entity_to_data_dict)
labels = labeler.execute(entity_to_data_and_cutoff_dict, "JSON Files/yelp_prediction_problems.json")