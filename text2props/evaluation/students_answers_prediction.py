from text2props.constants import (
    DATA_PATH,
    S_ID,
    TIMESTAMP,
    CORRECT,
    Q_ID,
    DIFFICULTY,
    DISCRIMINATION,
)
from ._prediction_methods import irt_prediction_with_update
import pickle
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np


def evaluate_students_answers_prediction(
        df_sap,
        df_train,
        df_test,
        dict_real_latent_traits,
        dict_stored_predictions,
        output_folder='outputs',
):
    # load sap dataframe
    df_sap = df_sap.sort_values([S_ID, TIMESTAMP])
    true_results = df_sap[CORRECT].values
    # collect the list of users
    user_id_list = list(df_sap[S_ID].unique())

    # convert to dict of dict
    dict_estimated_latent_traits = dict()
    dict_estimated_latent_traits[DIFFICULTY] = dict()
    dict_estimated_latent_traits[DISCRIMINATION] = dict()
    for idx, q_id in enumerate(df_test[Q_ID].values):
        dict_estimated_latent_traits[DIFFICULTY][q_id] = dict_stored_predictions[DIFFICULTY][idx]
        dict_estimated_latent_traits[DISCRIMINATION][q_id] = dict_stored_predictions[DISCRIMINATION][idx]

    # add the real latent traits of the test questions to the dict of predicted latent traits
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
    print("[INFO] Done")
    output_string = 'IRT estimated latent traits: '
    auc_irt = roc_auc_score(true_results, irt_predicted_results)
    print("AUC = %.5f" % auc_irt)
    bool_irt_predicted_results = [x >= 0.5 for x in irt_predicted_results]
    output_string += gen_output(bool_irt_predicted_results, true_results)
    print(output_string)

    print("[INFO] Doing prediction with predicted latent traits...")
    dict_nlp_predicted_results = irt_prediction_with_update(df_sap, dict_estimated_latent_traits, user_id_list)
    nlp_predicted_results = []
    for user_id in user_id_list:
        nlp_predicted_results.extend(dict_nlp_predicted_results[user_id])
    pickle.dump(nlp_predicted_results, open(os.path.join(output_folder, 'performance-prediction-nlp.p'), 'wb'))
    print("[INFO] Done")
    output_string = 'NLP estimated latent traits: '
    auc_nlp = roc_auc_score(true_results, nlp_predicted_results)
    print("AUC = %.5f" % auc_nlp)
    bool_nlp_predicted_results = [x >= 0.5 for x in nlp_predicted_results]
    output_string += gen_output(bool_nlp_predicted_results, true_results)
    print(output_string)

    print("[INFO] Doing majority prediction...")
    majority_prediction = [np.mean(true_results)]*len(true_results)
    print("[INFO] Done")
    output_string = 'Majority prediction: '
    print("AUC = %.5f" % roc_auc_score(true_results, majority_prediction))
    bool_majority_prediction = [x >= 0.5 for x in majority_prediction]
    output_string += gen_output(bool_majority_prediction, true_results)
    print(output_string)


def gen_output(predicted_results, true_res):
    tn, fp, fn, tp = confusion_matrix(true_res, predicted_results).ravel()
    output_str = ''
    output_str += 'acc : %.3f | ' % ((tp+tn)/(tp+tn+fp+fn))
    output_str += 'prec correct : %.3f | ' % (tp/(tp+fp))
    output_str += 'rec correct : %.3f | ' % (tp/(tp+fn))
    output_str += 'prec wrong : %.3f | ' % (tn/(tn+fn))
    output_str += 'rec wrong: %.3f ' % (tn/(tn+fp))
    return output_str
