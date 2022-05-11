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
    transactions, models, mcc_code_to_idx, device
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
        emb_batch_ens = []
        for model in models:
            with torch.no_grad():
                emb_batch = model(data.to(device), mode="vtb")
                emb_batch_ens.append(emb_batch.unsqueeze(0))

        emb_batch = torch.cat(emb_batch_ens, dim=0).mean(0)

        embs.append(emb_batch)

    embs = torch.cat(embs, dim=0)
    embs = embs.detach().cpu().numpy()

    uids = bank_trans_emb["user_id"].values

    embs_dict = dict(zip(uids, embs))

    print("vtb embs shape: ", embs.shape)

    return embs_dict


def create_rtk_embeddings_for_model(
    clickstream, models, cat_code_to_idx, device
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
        emb_batch_ens = []
        for model in models:
            with torch.no_grad():
                emb_batch = model(data.to(device), mode="rtk")
                emb_batch_ens.append(emb_batch.unsqueeze(0))

        emb_batch = torch.cat(emb_batch_ens, dim=0).mean(0)
        
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
            if np.random.rand > 0.2:
                rtk_emb_id = "0"

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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    model_weights = [
        "./weights/SModel_0.774_fold_0.pth",
        "./weights/SModel_0.774_fold_1.pth",
        "./weights/SModel_0.780_fold_2.pth",
        "./weights/SModel_0.780_fold_3.pth",
        "./weights/SModel_0.780_fold_4.pth",
    ]
    raw_models = [
        SModel().to(device),
        SModel().to(device),
        SModel().to(device),
        SModel().to(device),
        SModel().to(device),
    ]
    models = [
        load_model(weights, m, device) for m, weights in zip(raw_models, model_weights)
    ]

    print(f" {len(models)} models loaded.")

    mcc_code_to_idx, cat_code_to_idx = load_code_to_idx(
        "./mcc_code_to_idx.pickle",
        "./cat_code_to_idx.pickle",
    )
    print("Indexes dicts loaded.")

    vtb_emb = create_vtb_embeddings_for_model(
        transactions_df, models, mcc_code_to_idx, device
    )
    print("VTB embeddings created.")

    rtk_emb = create_rtk_embeddings_for_model(
        clickstream_df, models, cat_code_to_idx, device
    )
    print("RTK embeddings created.")

    submission = run_inference(transactions_df, vtb_emb, rtk_emb)
    print("Submission done.")

    # print(submission)
    print(submission.shape)

    np.savez(output_path, submission)


if __name__ == "__main__":
    main()
