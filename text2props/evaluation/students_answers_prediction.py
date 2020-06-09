from text2props.constants import (
    DATA_PATH,
    OUTPUT_DATA_PATH,
    S_ID,
    TIMESTAMP,
    CORRECT,
    Q_ID,
    DIFFICULTY,
    DISCRIMINATION,
)
from ._prediction_methods import irt_prediction_with_update
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score
from typing import Dict, List, Iterable


def evaluate_students_answers_prediction(
        df_sap: pd.DataFrame,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        dict_real_latent_traits: Dict[str, Dict[str, float]],
        dict_stored_predictions: Dict[str, List],
        output_folder: str = OUTPUT_DATA_PATH,
) -> None:
    df_sap = df_sap.sort_values([S_ID, TIMESTAMP])
    true_results = df_sap[CORRECT].values
    user_id_list = list(df_sap[S_ID].unique())

    # convert to dict of dict
    dict_estimated_latent_traits = dict()
    dict_estimated_latent_traits[DIFFICULTY] = dict()
    dict_estimated_latent_traits[DISCRIMINATION] = dict()
    for idx, q_id in enumerate(df_test[Q_ID].values):
        dict_estimated_latent_traits[DIFFICULTY][q_id] = dict_stored_predictions[DIFFICULTY][idx]
        dict_estimated_latent_traits[DISCRIMINATION][q_id] = dict_stored_predictions[DISCRIMINATION][idx]

    # add real latent traits of train questions to the dict of predicted latent traits (accuracy NOT measured on them)
    list_train_questions = list(df_train[Q_ID].unique())
    for q_id in list_train_questions:
        dict_estimated_latent_traits[DIFFICULTY][q_id] = dict_real_latent_traits[DIFFICULTY][q_id]
        dict_estimated_latent_traits[DISCRIMINATION][q_id] = dict_real_latent_traits[DISCRIMINATION][q_id]

    print("[INFO] Doing prediction with ground truth IRT latent traits...")
    dict_irt_predicted_results = irt_prediction_with_update(df_sap, dict_real_latent_traits, user_id_list)
    irt_predicted_results = []
    for user_id in user_id_list:
        irt_predicted_results.extend(dict_irt_predicted_results[user_id])
    pickle.dump(irt_predicted_results, open(os.path.join(output_folder, 'performance-prediction-irt.p'), 'wb'))
    print('IRT estimation: ' + gen_output([x >= 0.5 for x in irt_predicted_results], true_results))
    print("                AUC = %.5f" % roc_auc_score(true_results, irt_predicted_results))

    print("[INFO] Doing prediction with predicted latent traits...")
    dict_nlp_predicted_results = irt_prediction_with_update(df_sap, dict_estimated_latent_traits, user_id_list)
    nlp_predicted_results = []
    for user_id in user_id_list:
        nlp_predicted_results.extend(dict_nlp_predicted_results[user_id])
    pickle.dump(nlp_predicted_results, open(os.path.join(output_folder, 'performance-prediction-nlp.p'), 'wb'))
    print('NLP estimation: ' + gen_output([x >= 0.5 for x in nlp_predicted_results], true_results))
    print("                AUC = %.5f" % roc_auc_score(true_results, nlp_predicted_results))

    print("[INFO] Doing majority prediction...")
    majority_prediction = [np.mean(true_results)]*len(true_results)
    print('Majority: ' + gen_output([x >= 0.5 for x in majority_prediction], true_results))
    print("          AUC = %.5f" % roc_auc_score(true_results, majority_prediction))


def gen_output(predicted_results: Iterable[bool], true_results: Iterable[bool]) -> str:
    tn, fp, fn, tp = confusion_matrix(true_results, predicted_results).ravel()
    output_str = ''
    output_str += 'acc : %.3f | ' % ((tp+tn)/(tp+tn+fp+fn))
    output_str += 'prec correct : %.3f | ' % (tp/(tp+fp))
    output_str += 'rec correct : %.3f | ' % (tp/(tp+fn))
    output_str += 'prec wrong : %.3f | ' % (tn/(tn+fn))
    output_str += 'rec wrong: %.3f ' % (tn/(tn+fp))
    return output_str
