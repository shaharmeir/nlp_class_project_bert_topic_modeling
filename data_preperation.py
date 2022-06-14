from collections import Counter
from typing import Union
import pickle
import pandas as pd
from config import IS_ETHICAL_COLUMN, WHY_NOT_ETHICAL_TEXT_COLUMN, WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, \
    WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN, RISK_1_COLUMN, MIN_WORDS_TO_KEEP, MAX_WORDS_TO_KEEP, EMBEDDING_COLUMN, \
    ORIGINAL_EXCEL_PATH, OUTPUT_WITH_EMBEDDING_PICKLE_PATH, MIN_REASON_COUNT_TO_KEEP
from bert_embedding import get_text_embedding

pd.options.mode.chained_assignment = None


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    not_ethical_with_word_count_in_a_normal_range
    :param df:
    :return: Part of the dataframe that is not ethical, with a normal word count (not too many, and not too little)
    """
    # we want only not ethical data
    df = df[~df[IS_ETHICAL_COLUMN]]

    df[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN] = df.apply(lambda row: clean_text(row[WHY_NOT_ETHICAL_TEXT_COLUMN]), axis=1)
    # Create a new column corresponding to the length of each headline
    df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN] = df.apply(
        lambda row: len(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN].split()),
        axis=1)
    # we take only real text, and not -99/Attention based on # Counter(df[df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT] == 1][WHY_NOT_ETHICAL_CLEAN_COLUMN]).most_common()
    df = df[df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN] >= MIN_WORDS_TO_KEEP]
    # len(df[df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN] > 100]), len(df), only a small portion of the data has more than 100 words.
    df = df[df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN] <= MAX_WORDS_TO_KEEP]
    # I checked that when Risk1 does not exist, no other risk exists, which makes na rows irrelevant
    df = df[~df[RISK_1_COLUMN].isna()]
    df[RISK_1_COLUMN] = df.apply(lambda row: clean_text(row[RISK_1_COLUMN], to_lower=True), axis=1)
    # todo: עדיין יש חוסר אחידות בסוגי הסיכונים שהם תייגו אז שווה אולי לאחד ידנית על בסיס מיפוי שנעשה
    # todo: אולי שווה גם להסתכל על Risk 2 למרות שאין שם הרבה

    return df


def _leave_only_main_risks(df):
    relevant_df = df[[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, RISK_1_COLUMN, EMBEDDING_COLUMN]]
    risk_reason_to_count_mapping = dict(Counter(relevant_df[RISK_1_COLUMN]))
    relevant_risk_reasons = [risk_reason for risk_reason, count in risk_reason_to_count_mapping.items() if
                             count >= MIN_REASON_COUNT_TO_KEEP]
    relevant_df = relevant_df[relevant_df[RISK_1_COLUMN].isin(
        relevant_risk_reasons)]  # todo: maybe we can fix some of those risk reasons to be in the top reasons
    return relevant_df


def read_excel_df(df_path: str) -> pd.DataFrame:
    df = pd.read_excel(df_path)
    # fix excel format
    df = df.replace(r'\n', ' ', regex=True)
    df = df.replace(r'[^\x00-\x7F]+', '', regex=True)
    df = df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)
    return df


def clean_text(dirty_text: Union[str, int], to_lower: bool = False) -> str:
    dirty_text = str(dirty_text)
    # clean_text = re.sub(r'[^a-zA-Z0-9./ ]', r'', dirty_text)
    text_after_cleaning = " ".join(dirty_text.strip().split())
    if to_lower:
        text_after_cleaning = text_after_cleaning.lower()
    return text_after_cleaning


if __name__ == "__main__":
    topic_modeling_df = read_excel_df(ORIGINAL_EXCEL_PATH)
    topic_modeling_df = fix_df(topic_modeling_df)
    topic_modeling_df[EMBEDDING_COLUMN] = topic_modeling_df.apply(
        lambda row: get_text_embedding(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN]), axis=1)
    topic_modeling_df = topic_modeling_df.reset_index(drop=True)
    with open(OUTPUT_WITH_EMBEDDING_PICKLE_PATH, "wb") as f:
        pickle.dump(topic_modeling_df, f)
    # topic_modeling_df.to_parquet(OUTPUT_WITH_EMBEDDING_PICKLE_PATH)
    # todo: maybe remove popular words i dont want to appear in clusters
    # todo: try spelling correction since many why not ethical texts are misspelled
