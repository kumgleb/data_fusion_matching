import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def get_embed_dicts(mcc_embed_path: str) -> Tuple[Dict, Dict]:
    with open(mcc_embed_path, "rb") as f:
        mcc_embed_dict = pickle.load(f)

    crk_embed_dict = {
        48: np.array([1, 0, 0]),
        50: np.array([0, 1, 0]),
        60: np.array([0, 0, 1]),
        -1: np.array([0, 0, 0]),
    }

    return mcc_embed_dict, crk_embed_dict


def get_mcc_embed(embed_dict: Dict, mcc_code: int) -> np.ndarray:
    if mcc_code == -1:
        emb = np.zeros(300)
    else:
        emb = embed_dict[mcc_code]
    return emb


def get_crk_embed(crk_embed_dict: Dict, crk_code: int) -> np.ndarray:
    return crk_embed_dict[crk_code]


def get_trns_amts_vector(trns_amts: List) -> np.ndarray:
    trns_amts = np.array(trns_amts)
    trns_amts_vec = [
        trns_amts.min(),
        trns_amts.max(),
        trns_amts.std(),
        trns_amts.mean(),
    ]
    return np.array(trns_amts_vec)


def create_transactions_embeddings(
    transactions: pd.DataFrame, mcc_code_to_idx: Dict
) -> pd.DataFrame:

    trns_aggr = pd.DataFrame(transactions.groupby("user_id")["mcc_code"].value_counts())

    trns_emb = trns_aggr.unstack().fillna(0).reset_index()

    trns_emb = pd.concat([trns_emb["user_id"], trns_emb["mcc_code"]], axis=1)

    new_clmns = [mcc_code_to_idx[clmn] for clmn in trns_emb.columns[1:]]
    trns_emb.columns = ["user_id"] + new_clmns

    return trns_emb
