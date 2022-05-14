import gc
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from scipy.spatial import distance
from typing import Tuple, Dict

from src.model import SModel
from src.data_utils import EMBDataset
from src.utils import (
    create_transactions_embeddings,
    create_clickstream_embeddings,
    load_model,
    load_code_to_idx,
)


def create_vtb_embeddings_for_model(
    transactions, model, mcc_code_to_idx, device
) -> Dict:

    bank_trans_emb = create_transactions_embeddings(transactions, mcc_code_to_idx)

    del transactions
    gc.collect()

    emb_dataset = EMBDataset(bank_trans_emb, "vtb")
    emb_dataset_loader = DataLoader(emb_dataset, 256, shuffle=False)

    embs = []
    emb_iter = iter(emb_dataset_loader)
    for i in range(len(emb_iter)):
        data = next(emb_iter)
        emb_batch = model(data.to(device), mode="anchor")
        embs.append(emb_batch)

    embs = torch.cat(embs, dim=0)
    embs = embs.detach().cpu().numpy()

    uids = bank_trans_emb["user_id"].values

    embs_dict = dict(zip(uids, embs))

    print("vtb embs shape: ", embs.shape)

    return embs_dict


def create_rtk_embeddings_for_model(
    clickstream, model, cat_code_to_idx, device
) -> Dict:

    rtk_emb = create_clickstream_embeddings(clickstream, cat_code_to_idx)

    # Add unmatched case:
    # ze = np.zeros(rtk_emb.shape[1]).tolist()
    # ze[0] = "0"
    # rtk_emb.loc[len(rtk_emb)] = ze

    del clickstream
    gc.collect()

    emb_dataset = EMBDataset(rtk_emb, "rtk")
    emb_dataset_loader = DataLoader(emb_dataset, 256, shuffle=False)

    embs = []
    emb_iter = iter(emb_dataset_loader)
    for i in range(len(emb_iter)):
        data = next(emb_iter)
        emb_batch = model(data.to(device), mode="positive")
        embs.append(emb_batch)

    embs = torch.cat(embs, dim=0)
    embs = embs.detach().cpu().numpy()

    uids = rtk_emb["user_id"].values

    # embs_dict = dict(zip(uids, embs))

    print("rtk embs shape: ", embs.shape)

    return (uids, embs)


def get_distance(xgboost_model, vtb_emb, rtk_emb):
    x = np.concatenate([vtb_emb, rtk_emb])
    dist = xgboost_model.predict_proba(x.reshape(1, -1))[0][1]
    return dist


def get_closest(vtb_emb_val, rtk_emb, xgboost_model):

    threshold = 0.05

    rtk_uids, rtk_embs = rtk_emb

    vtb_embs = np.ones_like(rtk_embs) * vtb_emb_val
    X = np.concatenate([vtb_embs, rtk_embs], axis=1)
    dist = xgboost_model.predict_proba(X)[:, 1]

    rtk_uids[dist < threshold] = 0.

    embs_dists = [(rtk_id, d) for rtk_id, d in zip(rtk_uids, dist)]

    embs_dists.sort(key=lambda x: x[1], reverse=True)

    return embs_dists[:100]


def run_inference(transactions_df, vtb_emb, rtk_emb, xgboost_model):

    submission = []

    for uid in tqdm(transactions_df["user_id"].unique()):
        vtb_emb_val = vtb_emb[uid]
        closest = get_closest(vtb_emb_val, rtk_emb, xgboost_model)
        closest_ids = np.array([obj[0] for obj in closest], dtype=object)
        submission.append([uid, closest_ids])

    submission = np.array(submission, dtype=object)

    return submission


def main():
    data, output_path = sys.argv[1:]
    transactions_df = pd.read_csv(f"{data}/transactions.csv")
    clickstream_df = pd.read_csv(f"{data}/clickstream.csv")
    print("Dataframes loaded.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    model = SModel().to(device)
    weights = "./weights/SModel_0.606.pth"
    model = load_model(weights, model, device)
    print("Model loaded.")

    mcc_code_to_idx, cat_code_to_idx = load_code_to_idx(
        "./mcc_code_to_idx.pickle",
        "./cat_code_to_idx.pickle",
    )
    print("Indexes dicts loaded.")

    vtb_emb = create_vtb_embeddings_for_model(
        transactions_df, model, mcc_code_to_idx, device
    )
    print("VTB embeddings created.")

    rtk_embs = create_rtk_embeddings_for_model(
        clickstream_df, model, cat_code_to_idx, device
    )
    print("RTK embeddings created.")

    xgboost_model = xgb.XGBClassifier()
    xgboost_model.load_model("./model_xgboost.txt")
    print("XGBoost model loaded.")

    submission = run_inference(transactions_df, vtb_emb, rtk_embs, xgboost_model)
    print("Submission done.")

    # print(submission)
    print(submission.shape)

    np.savez(output_path, submission)


if __name__ == "__main__":
    main()
