import torch
import pickle
import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict


def load_embeddings(
    bank_trans_emb_path: str, rtk_clck_emb_path: str
) -> Tuple[Dict, Dict]:
    with open(bank_trans_emb_path, "rb") as f:
        bank_trans_emb = pickle.load(f)

    with open(rtk_clck_emb_path, "rb") as f:
        rtk_clck_emb = pickle.load(f)

    return bank_trans_emb, rtk_clck_emb


def load_model(weights, model, device):
    checkpoint = torch.load(weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def get_vtb_encodings_csv(models, transactions_df, vtb_emb_dict, device):

    encodings = {}

    for uid in tqdm(transactions_df["user_id"].unique()):

        X = torch.tensor(vtb_emb_dict[uid], dtype=torch.float32).to(device)

        for model in models:
            encs = []
            model.eval()
            with torch.no_grad():
                enc = model(X, "anchor").cpu().numpy()
                encs.append(enc)

        enc = np.mean(encs, axis=0)
        encodings[uid] = enc

    with open("./embeddings/vtb_emb.pickle", "wb") as f:
        pickle.dump(encodings, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_rtk_encodings_csv(models, clickstream_df, rtk_clck_emb, device):

    encodings = {}

    for uid in tqdm(clickstream_df["user_id"].unique()):

        X = torch.tensor(rtk_clck_emb[uid], dtype=torch.float32).to(device)

        for model in models:
            encs = []
            model.eval()
            with torch.no_grad():
                enc = model(X, "positive").cpu().numpy()
                encs.append(enc)

        enc = np.mean(encs, axis=0)
        encodings[uid] = enc

    with open("./embeddings/rtk_emb.pickle", "wb") as f:
        pickle.dump(encodings, f, protocol=pickle.HIGHEST_PROTOCOL)
