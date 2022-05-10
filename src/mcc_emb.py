import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from navec import Navec
from gensim.models import KeyedVectors

nltk.download("stopwords")
from nltk.corpus import stopwords
from typing import Dict


REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,-.;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def text_prepare_ru(text):
    """
    text: a string

    return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(
        " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    return text


def text_prepare_eng(text):
    """
    text: a string

    return: modified initial string
    """
    text = BAD_SYMBOLS_RE.sub(
        "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = " ".join(
        [word for word in text.split() if word not in STOPWORDS]
    )  # delete stopwords from text
    text = text.strip()
    return text


def remove_mcc(text):
    words = text.split()
    text = " ".join(word for word in words if word not in ["mcc", "мсс"])
    return text


def remove_duplicates(text):
    words = text.split()
    text = " ".join(sorted(set(words), key=words.index))
    return text


def translate_to_eng(df, translator) -> pd.DataFrame:
    df["Description"] = df["Description"].apply(
        lambda x: translator.translate(x, src="ru", dest="en").text
    )
    return df


def process_mcc_df(mcc_codes: pd.DataFrame) -> pd.DataFrame:

    mcc_codes = mcc_codes.fillna(" ")
    mcc_codes["Description"] = mcc_codes["Название"] + " " + mcc_codes["Описание"]
    mcc_codes = mcc_codes.drop(["Название", "Описание"], axis=1)

    mcc_codes["Description"] = mcc_codes["Description"].map(text_prepare_ru)
    mcc_codes["Description"] = mcc_codes["Description"].map(remove_duplicates)
    mcc_codes["Description"] = mcc_codes["Description"].map(remove_mcc)

    return mcc_codes


def clean_mcc_df_eng(mcc_codes: pd.DataFrame) -> pd.DataFrame:
    mcc_codes["Description"] = mcc_codes["Description"].map(text_prepare_eng)
    return mcc_codes


def create_text_embed(text: str, wv_embeddings: KeyedVectors):
    embs = []

    for word in text.split():
        try:
            emb = wv_embeddings[word]
            embs.append(emb)
        except KeyError:
            continue

    embs = np.array(embs).mean(0)

    return embs


def create_mcc_embeddings_dict(
    mcc_codes: pd.DataFrame, wv_embeddings: KeyedVectors, mode="WE", model=None
) -> Dict:

    embs = {}

    for idx in tqdm(mcc_codes.index):
        mcc_code = mcc_codes.loc[idx, "MCC"]
        discr = mcc_codes.loc[idx, "Description"]


        if mode == "WE":
            embs[clck_code] = create_text_embed(discr, wv_embeddings)
        elif mode == "ST":
            embs[clck_code] = model.encode(discr)


    with open("./embeddings/mcc_emb_en.pickle", "wb") as f:
        pickle.dump(embs, f, protocol=pickle.HIGHEST_PROTOCOL)

    return embs
