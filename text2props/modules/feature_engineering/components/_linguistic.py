import numpy as np
import pandas as pd
import textstat
from typing import Optional
from scipy.sparse import coo_matrix
from text2props.constants import Q_TEXT, Q_ID
from ..utils import gen_correct_answers_dict, gen_wrong_answers_dict
from . import BaseFeatEngComponent

LEXICON_COUNT_QUESTION = 'lexicon_count_question'
LEXICON_COUNT_ANSWERS = 'lexicon_count_correct_choices'
LEXICON_COUNT_DISTRACTORS = 'lexicon_count_wrong_choices'
SENTENCE_COUNT_QUESTION = 'sentence_count_question'
SENTENCE_COUNT_ANSWERS = 'sentence_count_correct_choices'
SENTENCE_COUNT_DISTRACTORS = 'sentence_count_wrong_choices'
AVG_WORD_LENGTH_QUESTION = 'avg_word_len_question'
RATIO_LENGTH_QUESTION_ANSWERS = 'ratio_len_question_correct_choices'
RATIO_LEN_QUESTION_DISTRACTORS = 'ratio_len_question_wrong_choices'


class LinguisticFeaturesComponent(BaseFeatEngComponent):

    def __init__(self):
        return

    def fit_transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Transforms the input data and returns a sparse matrix. It returns a sparse matrix, although these features are
        not sparse for consistency reasons with the other components.
        :param input_df:
        :return:
        """
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        It computes and returns the linguistic features from the input DF. The DF must include the following attributes
        in its columns: Q_TEXT, Q_ID
        :param input_df:
        :return:
        """
        if Q_TEXT not in input_df.columns:
            raise ValueError("Q_TEXT should be in input_df.columns")
        if Q_ID not in input_df.columns:
            raise ValueError("Q_ID should be in input_df.columns")

        correct_ans_text_dict = gen_correct_answers_dict(input_df)
        wrong_ans_text_dict = gen_wrong_answers_dict(input_df)

        df = pd.DataFrame()

        for q_id, q_text in input_df[[Q_ID, Q_TEXT]].values:
            answers_text = correct_ans_text_dict[q_id]
            distractors_text = wrong_ans_text_dict[q_id]
            lexicon_count_question = textstat.lexicon_count(q_text)
            lexicon_count_answers = np.mean([textstat.lexicon_count(x) for x in answers_text])
            lexicon_count_distractors = np.mean([textstat.lexicon_count(x) for x in distractors_text])
            new_row = pd.DataFrame([{
                LEXICON_COUNT_QUESTION: lexicon_count_question,
                LEXICON_COUNT_ANSWERS: lexicon_count_answers,
                LEXICON_COUNT_DISTRACTORS: lexicon_count_distractors,
                SENTENCE_COUNT_QUESTION: textstat.sentence_count(q_text),
                SENTENCE_COUNT_ANSWERS: np.mean([textstat.sentence_count(x) for x in answers_text]),
                SENTENCE_COUNT_DISTRACTORS: np.mean([textstat.sentence_count(x) for x in distractors_text]),
                AVG_WORD_LENGTH_QUESTION: np.mean([len(x) for x in q_text.split(' ')]),  # todo possibly improve this
                RATIO_LENGTH_QUESTION_ANSWERS: (1 + lexicon_count_question) / (1 + lexicon_count_answers),
                RATIO_LEN_QUESTION_DISTRACTORS: (1 + lexicon_count_question) / (1 + lexicon_count_distractors),
            }])
            df = pd.concat([df, new_row], ignore_index=True)

        return coo_matrix(df.values)
