import torch
import pickle
import pandas as pd
from typing import Tuple, Dict


def create_transactions_embeddings(
    transactions: pd.DataFrame, mcc_code_to_idx: Dict
) -> pd.DataFrame:

    trns_aggr = pd.DataFrame(transactions.groupby("user_id")["mcc_code"].value_counts())
    
    trns_emb = trns_aggr.unstack().fillna(0).reset_index()
    
    trns_emb = pd.concat([trns_emb["user_id"], trns_emb["mcc_code"]], axis=1)

    new_clmns = [mcc_code_to_idx[clmn] for clmn in trns_emb.columns[1:]]
    trns_emb.columns = ["user_id"] + new_clmns

    return trns_emb


def create_clickstream_embeddings(
    clickstream: pd.DataFrame, cat_id_to_idx: Dict
) -> pd.DataFrame:

    clc_aggr = pd.DataFrame(clickstream.groupby("user_id")["cat_id"].value_counts())

    clc_emb = clc_aggr.unstack().fillna(0).reset_index()

    clc_emb = pd.concat([clc_emb["user_id"], clc_emb["cat_id"]], axis=1)

    new_clmns = [cat_id_to_idx[clmn] for clmn in clc_emb.columns[1:]]
    clc_emb.columns = ["user_id"] + new_clmns
    
    return clc_emb


def load_model(weights, model, device):
    checkpoint = torch.load(weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def load_code_to_idx(mcc_code_to_idx_path, cat_code_to_idx_path) -> Tuple[Dict, Dict]:
    with open(mcc_code_to_idx_path, "rb") as f:
        mcc_code_to_idx = pickle.load(f)

    with open(cat_code_to_idx_path, "rb") as f:
        cat_code_to_idx = pickle.load(f)

    return mcc_code_to_idx, cat_code_to_idx


def load_embs(mcc_embs_path, cat_embs_path):
    with open(mcc_embs_path, "rb") as f:
        mcc_embs = pickle.load(f)

    with open(cat_embs_path, "rb") as f:
        clc_embs = pickle.load(f)

    return mcc_embs, clc_embs