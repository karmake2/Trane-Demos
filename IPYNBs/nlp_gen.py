import sys
#Path to Trane for imports
sys.path.append('/Users/Alexander/Documents/Trane/Trane__HDI_REPO')
path_to_datasets = '../../Trane__Local_Misc/Formatted Datasets/Saudi Full Data - King Fahd Hospital/'
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
from dateutil import parser

def save_obj(obj, name):
    with open('../pickled_objects/Healthcare/entity=hosp_code'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    try:
        with open('../pickled_objects/Saudi_Healthcare/entity=hosp_code' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None
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
        probs, table_meta, entity_id_column, label_generating_column, time_column, "prediction_problems.json")

    nl_descrips = trane.generate_nl_description(
        probs, table_meta, entity_id_column, label_generating_column, time_column, trane.ConstantIntegerCutoffTimes(0))
    return probs, nl_descrips
def file_to_table_meta(filepath):
    return trane.TableMeta(json.loads(open(filepath).read()))
#ER Data
er_data_df = pd.read_csv(path_to_datasets + 'ER.csv')
er_data_meta = file_to_table_meta(path_to_datasets + 'ER_table_meta.json')

#INP Data
inpatient_data_df = pd.read_csv(path_to_datasets + 'Inpatient.csv')
inpatient_table_meta = file_to_table_meta(path_to_datasets + 'Inpatient_meta.json')

# #ODP Data
opd_data_df = pd.read_csv(path_to_datasets + 'OPD.csv')
# odp_data_meta = file_to_table_meta(path_to_datasets + 'ODP_table_meta.json')

#ODP No Diagnosis Data
#Note: does not contain headers
opd_data_no_diag_df = pd.read_csv(path_to_datasets + 'opd_without_diag.csv', encoding = "latin1")

unique_er_patient_ids = er_data_df['PATIENT_ID'].unique()
unique_inpatient_ids = inpatient_data_df['PATIENT_ID'].unique()
unique_opd_ids = opd_data_df['PATIENT_ID'].unique()
unqiue_opd_no_diag_ids = opd_data_no_diag_df['PATIENT_ID'].unique()

DF = inpatient_data_df
TM = inpatient_table_meta
ENTITY_ID_COLUMN = 'HOSP_CODE'
TIME_COLUMN = 'ADMIT_DATE'
LABEL_GENERATING_COLUMN = 'DURATION'
FILTER_COLUMN = 'PATIENT_ID'
DF[TIME_COLUMN] = DF[TIME_COLUMN].apply(parser.parse)
probs = load_obj('health_prediction_problems')
if probs == None:
    generator = trane.PredictionProblemGenerator(
        TM,
        ENTITY_ID_COLUMN,
        LABEL_GENERATING_COLUMN,
        TIME_COLUMN,
        FILTER_COLUMN)
    probs = generator.generate(DF)
    save_obj(probs, 'health_prediction_problems')

nl_scrips = trane.generate_nl_description(probs, TM, ENTITY_ID_COLUMN, LABEL_GENERATING_COLUMN,
                              TIME_COLUMN, None)

