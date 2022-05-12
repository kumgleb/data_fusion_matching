import gc
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
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
        emb_batch = model(data.to(device), mode="vtb")
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

    del clickstream
    gc.collect()

    emb_dataset = EMBDataset(rtk_emb, "rtk")
    emb_dataset_loader = DataLoader(emb_dataset, 256, shuffle=False)

    embs = []
    emb_iter = iter(emb_dataset_loader)
    for i in range(len(emb_iter)):
        data = next(emb_iter)
        emb_batch = model(data.to(device), mode="rtk")
        embs.append(emb_batch)

    embs = torch.cat(embs, dim=0)
    embs = embs.detach().cpu().numpy()

    uids = rtk_emb["user_id"].values

    embs_dict = dict(zip(uids, embs))

    print("rtk embs shape: ", embs.shape)

    return embs_dict


def get_closest(vtb_emb_val, rtk_emb, metric="euclid"):

    embs_dists = []
    dist_threshold = 40

    for rtk_emb_id, rtk_emb_val in rtk_emb.items():
        if metric == "euclid":
            dist = distance.euclidean(vtb_emb_val, rtk_emb_val)
        elif metric == "cosine":
            dist = distance.cosine(vtb_emb_val, rtk_emb_val)

        # Add unmatched id:
        if dist > dist_threshold:
            rtk_emb_id = 0.

        embs_dists.append((rtk_emb_id, dist))

    embs_dists.sort(key=lambda x: x[1], reverse=False)

    return embs_dists[:100]


def run_inference(transactions_df, vtb_emb, rtk_emb):

    submission = []

    for uid in tqdm(transactions_df["user_id"].unique()):
        vtb_emb_val = vtb_emb[uid]
        closest = get_closest(vtb_emb_val, rtk_emb, metric="euclid")
        closest_ids = np.array([obj[0] for obj in closest], dtype=object)
        submission.append([uid, closest_ids])

    submission = np.array(submission, dtype=object)

    return submission


def main():
    data, output_path = sys.argv[1:]
    transactions_df = pd.read_csv(f"{data}/transactions.csv")
    clickstream_df = pd.read_csv(f"{data}/clickstream.csv")
    print("Dataframes loaded.")

    transactions_df = transactions_df[transactions_df.mcc_code != -1]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    model = SModel().to(device)
    weights = "./weights/SModel_0.774_fold_1.pth"
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

    rtk_emb = create_rtk_embeddings_for_model(
        clickstream_df, model, cat_code_to_idx, device
    )
    print("RTK embeddings created.")

    submission = run_inference(transactions_df, vtb_emb, rtk_emb)
    print("Submission done.")

    # print(submission)
    print(submission.shape)

    np.savez(output_path, submission)


if __name__ == "__main__":
    main()
