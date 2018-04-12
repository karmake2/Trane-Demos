import sys
#Path to Trane for imports
sys.path.append('/Users/Alexander/Documents/Trane/Trane__HDI_REPO')
path_to_datasets = '../../Trane__Local_Misc/Formatted Datasets/Yelp Reviews v2/'
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
    with open('../pickled_objects/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    try:
        with open('../pickled_objects/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None
def generate_probs(entity_id_column,
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
        probs, table_meta, entity_id_column, label_generating_column, time_column, "prediction_problems.json")

    return probs

def convert(str, format = None):
    return datetime.datetime.strptime(str, format)
def file_to_table_meta(filepath):
    return trane.TableMeta(json.loads(open(filepath).read()))

filename = 'merged_df'
merged_df = load_obj(filename)
if merged_df is None:
    
    yelp_review_df = pd.read_csv(path_to_datasets + 'yelp_review.csv')
    yelp_checkin_df = pd.read_csv(path_to_datasets + 'yelp_checkin.csv')
    yelp_business_df = pd.read_csv(path_to_datasets + 'yelp_business.csv')
    yelp_user_df = pd.read_csv(path_to_datasets + 'yelp_user.csv')

    sampled_yelp_review_df = yelp_review_df.head(1000)
    sampled_business_ids = sampled_yelp_review_df['business_id'].unique()
    sampled_user_ids = sampled_yelp_review_df['user_id'].unique()
    sampled_review_ids = sampled_yelp_review_df['review_id'].unique()
    sampled_yelp_checkin_df = yelp_checkin_df[yelp_checkin_df['business_id'].isin(sampled_business_ids)]
    sampled_yelp_business_df = yelp_business_df[yelp_business_df['business_id'].isin(sampled_business_ids)] 
    sampled_yelp_user_df = yelp_user_df[yelp_user_df['user_id'].isin(sampled_user_ids)]

    assert(len(sampled_business_ids) == len(sampled_yelp_business_df))
    assert(len(sampled_user_ids) == len(sampled_yelp_user_df))
    assert(len(sampled_review_ids) == len(sampled_yelp_review_df))

    print("Sampling Reuslts ---")
    print("Number of reviews: {}".format(len(sampled_yelp_review_df)))
    print("Number of businesses: {}".format(len(sampled_business_ids)))
    print("Number of users: {}".format(len(sampled_user_ids)))
    print("Number of checkins: {}".format(len(sampled_yelp_checkin_df)))

    merge_step_1 = pd.merge(sampled_yelp_review_df, sampled_yelp_user_df, how = 'left', on ='user_id')
    merge_step_2 = pd.merge(merge_step_1, sampled_yelp_business_df, how = 'left', on = 'business_id')
    merge_step_3 = pd.merge(merge_step_2, sampled_yelp_checkin_df, how = 'right', on = 'business_id')
    merged_df = merge_step_3
    merged_df['date'] = merged_df['date'].apply(str)
    merged_df['date'] = merged_df['date'].apply(convert, format = '%Y-%m-%d')
    merged_df = merged_df.rename(columns = {'stars_x': 'stars'})

    distinct_business_ids_in_merged_df = merged_df['business_id'].unique()
    #Note merged_df only contains 959 distinct business_ids. Checkins only contains information from 959 businesses.
    #    That's why merged_df only has 959 distinct business_ids, as opposed to the 974 unique business_ids 
    #    contained in the sample_yelp_review_df.
    #Note merged_df contains more than the 78792 unique check-ins because there are multiple reviews for some business_ids so
    #    each unique review_id is matched to a new business_id. There are 1000 review_ids, but only 974 business_ids.
    #
    save_obj(merged_df, filename)

table_meta = file_to_table_meta(path_to_datasets + "meta.json")
entity_id_column = 'business_id'
label_generating_column = 'stars'
time_column = 'date'
filter_column = 'user_id'
NUM_PROBLEMS_TO_GENERATE = 100

prediction_problems_filename = "../JSON Files/yelp_prediction_problems.json"

problem_generator_obj = trane.PredictionProblemGenerator(table_meta, entity_id_column, label_generating_column, time_column, filter_column)
problem_generator = problem_generator_obj.generator(merged_df)
problems = []
problems = list(problem_generator)

# greaterRowOp = trane.ops.GreaterRowOp(label_generating_column)
# greaterRowOp.set_thresholds(table_meta)

# for prob in probs:
#     prob.operations.append(greaterRowOp)

trane.prediction_problems_to_json_file(problems, table_meta, 
                                       entity_id_column, label_generating_column, 
                                       time_column, 
                                       prediction_problems_filename)


labeler = trane.Labeler()
entity_to_data_dict = trane.df_group_by_entity_id(merged_df, entity_id_column)
training_cutoff_time = datetime.date(2012, 1, 1)
label_cutoff_time = datetime.date(2015, 1, 1)
#First date: 2007-06-12
#Last date: 2017-12-10
entity_to_data_and_cutoff_dict = trane.ConstantCutoffTime(training_cutoff_time, label_cutoff_time).generate_cutoffs(entity_to_data_dict)
labels = labeler.execute(entity_to_data_and_cutoff_dict, prediction_problems_filename)