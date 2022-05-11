import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor


def get_negative_sample(match_true: pd.DataFrame, pos_sample: str) -> str:
    found = False
    while not found:
        sample = np.random.choice(match_true.rtk)
        if sample != pos_sample:
            return sample


def get_feature_vector_vtb(df: pd.DataFrame, id: str) -> Tensor:
    fv = np.zeros(385)
    idxs = np.array(df.columns[1:]).astype(np.int32)
    data = df[df["user_id"] == id].values[0][1:]
    fv[idxs] = data
    fv = np.log(fv + 1)
    return torch.tensor(fv, dtype=torch.float32)


def get_feature_vector_rtk(df: pd.DataFrame, id: str) -> Tensor:
    fv = np.zeros(402)
    idxs = np.array(df.columns[1:]).astype(np.int32)
    data = df[df["user_id"] == id].values[0][1:]
    fv[idxs] = data
    fv = np.log(fv + 1)
    return torch.tensor(fv, dtype=torch.float32)


class APNDataset(Dataset):
    def __init__(self, df, vtb_trans_emb, rtk_clck_emb):
        self.df = df
        self.vtb_trans_emb = vtb_trans_emb
        self.rtk_clck_emb = rtk_clck_emb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        anchor_id = self.df.iloc[idx].bank
        positive_id = self.df.iloc[idx].rtk
        negative_id = get_negative_sample(self.df, positive_id)

        data = {
            "anchor": get_feature_vector_vtb(
                self.vtb_trans_emb, anchor_id
            ),
            "positive": get_feature_vector_rtk(
                self.rtk_clck_emb, positive_id
            ),
            "negative": get_feature_vector_rtk(
                self.rtk_clck_emb, negative_id
            ),
        }

        return data


class EMBDataset(Dataset):
    def __init__(self, df, mode):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        uid = self.df.loc[idx, "user_id"]

        if self.mode == "vtb":
            fv = get_feature_vector_vtb(self.df, uid)
        elif self.mode == "rtk":
            fv = get_feature_vector_rtk(self.df, uid)

        return fv
