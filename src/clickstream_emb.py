import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors

from typing import Dict


def load_clck_emb_dict(path: str) -> Dict:
    with open(path, "rb") as f:
        clck_embed_dict = pickle.load(f)
    return clck_embed_dict


def remove_underscore(string: str) -> str:
    text = string.split("_")
    text = " ".join([word for word in text if word not in ["-", "_", "and"]])
    return text


def remove_duplicates(text):
    words = text.split()
    text = " ".join(sorted(set(words), key=words.index))
    return text


def process_categories_df(clickstream_categories: pd.DataFrame) -> pd.DataFrame:

    # Replace age:
    clickstream_categories.loc[4, "level_1"] = "child"
    clickstream_categories.loc[5, "level_1"] = "teenager"
    clickstream_categories.loc[6, "level_1"] = "adult"

    clickstream_categories = clickstream_categories.fillna(" ")

    clickstream_categories["level_0"] = clickstream_categories["level_0"].apply(
        remove_underscore
    )
    clickstream_categories["level_1"] = clickstream_categories["level_1"].apply(
        remove_underscore
    )
    clickstream_categories["level_2"] = clickstream_categories["level_2"].apply(
        remove_underscore
    )

    clickstream_categories["Description"] = (
        clickstream_categories["level_0"]
        + " "
        + clickstream_categories["level_1"]
        + " "
        + clickstream_categories["level_2"]
    )

    clickstream_categories["Description"] = clickstream_categories["Description"].map(
        remove_duplicates
    )

    clickstream_categories.drop(["level_0", "level_1", "level_2"], axis=1, inplace=True)

    return clickstream_categories


def create_text_embed_clck(text: str, wv_embeddings: KeyedVectors) -> Dict:
    embs = []

    for word in text.split():
        try:
            emb = wv_embeddings[word]
            embs.append(emb)
        except KeyError:
            continue

    embs = np.array(embs).mean(0)

    return embs


def create_clck_embeddings_dict(
    clickstream_cat: pd.DataFrame, wv_embeddings: KeyedVectors, mode="WE", model=None
) -> Dict:

    embs = {}

    for idx in tqdm(clickstream_cat.index):
        clck_code = clickstream_cat.loc[idx, "cat_id"]
        discr = clickstream_cat.loc[idx, "Description"]

    
        if mode == "WE":
            embs[clck_code] = create_text_embed_clck(discr, wv_embeddings)
        elif mode == "ST":
            embs[clck_code] = model.encode(discr)


    with open("./embeddings/new_clck_cat_emb_en_filtered.pickle", "wb") as f:
        pickle.dump(embs, f, protocol=pickle.DEFAULT_PROTOCOL)

    return embs


def get_clck_user_embed(
    clickstream: pd.DataFrame, user_id: int, clck_embed_dict: Dict
) -> np.ndarray:
    clcks = []

    user_clicks_data = clickstream[clickstream.user_id == user_id]

    for i in user_clicks_data.index:
        clck_cat = user_clicks_data.loc[i, "cat_id"]
        clck_emb = clck_embed_dict[clck_cat]
        clcks.append(clck_emb)

    clcks = np.array(clcks).mean(0)

    return clcks


def create_clickstream_embeddings(
    clickstream: pd.DataFrame, cat_id_to_idx: Dict
) -> pd.DataFrame:

    clc_aggr = pd.DataFrame(clickstream.groupby("user_id")["cat_id"].value_counts())

    clc_emb = clc_aggr.unstack().fillna(0).reset_index()

    clc_emb = pd.concat([clc_emb["user_id"], clc_emb["cat_id"]], axis=1)

    new_clmns = [cat_id_to_idx[clmn] for clmn in clc_emb.columns[1:]]
    clc_emb.columns = ["user_id"] + new_clmns

    return clc_emb
