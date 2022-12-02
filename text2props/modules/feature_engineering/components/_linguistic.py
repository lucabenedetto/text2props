import numpy as np
import pandas as pd
import textstat
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
        # features about number of words
        df[LEXICON_COUNT_QUESTION] = input_df.apply(lambda r: textstat.lexicon_count(r[Q_TEXT]), axis=1)
        df[LEXICON_COUNT_ANSWERS] = input_df.apply(
            lambda r: np.mean([textstat.lexicon_count(x) for x in correct_ans_text_dict[r[Q_ID]]]), axis=1)
        df[LEXICON_COUNT_DISTRACTORS] = input_df.apply(
            lambda r: np.mean([textstat.lexicon_count(x) for x in wrong_ans_text_dict[r[Q_ID]]]), axis=1)

        # features about the number of sentences
        df[SENTENCE_COUNT_QUESTION] = input_df.apply(lambda r: textstat.sentence_count(r[Q_TEXT]), axis=1)
        df[SENTENCE_COUNT_ANSWERS] = input_df.apply(
            lambda r: np.mean([textstat.sentence_count(x) for x in correct_ans_text_dict[r[Q_ID]]]), axis=1)
        df[SENTENCE_COUNT_DISTRACTORS] = input_df.apply(
            lambda r: np.mean([textstat.sentence_count(x) for x in wrong_ans_text_dict[r[Q_ID]]]), axis=1)

        # features about the length of the words
        df[AVG_WORD_LENGTH_QUESTION] = input_df.apply(lambda r: np.mean([len(x) for x in r[Q_TEXT].split(' ')]), axis=1)

        # rations between the same features computed on the question and on the correct/wrong choices
        df[RATIO_LENGTH_QUESTION_ANSWERS] = df.apply(
            lambda r: (1 + r[LEXICON_COUNT_QUESTION]) / (1 + r[LEXICON_COUNT_ANSWERS]), axis=1)
        df[RATIO_LEN_QUESTION_DISTRACTORS] = df.apply(
            lambda r: (1 + r[LEXICON_COUNT_QUESTION]) / (1 + r[LEXICON_COUNT_DISTRACTORS]), axis=1)
        return coo_matrix(df.values)
