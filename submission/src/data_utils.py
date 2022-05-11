import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor


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