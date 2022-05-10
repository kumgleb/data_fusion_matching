import torch
import numpy as np
from torch.utils.data import Dataset


def get_feature_vector_vtb(df, id, vtb_embs):
    data = df[df["user_id"] == id].values[0][1:]

    nonzero_counts_idxs = data.nonzero()[0]
    nonzero_counts = data[nonzero_counts_idxs].astype(np.float32)

    weights = torch.tensor(nonzero_counts)
    weights = torch.log(weights + 1)
    weights = F.softmax(weights, dim=0)

    embs = torch.as_tensor(vtb_embs[nonzero_counts_idxs])
    embs = embs * weights.unsqueeze(1)
    embs = embs.sum(0)

    return embs.type(torch.float32)


def get_feature_vector_rtk(df, id, rtk_embs):
    data = df[df["user_id"] == id].values[0][1:]

    nonzero_counts_idxs = data.nonzero()[0]
    nonzero_counts = data[nonzero_counts_idxs].astype(np.float32)

    weights = torch.tensor(nonzero_counts)
    weights = torch.log(weights + 1)
    weights = F.softmax(weights, dim=0)

    embs = torch.as_tensor(rtk_embs[nonzero_counts_idxs])
    embs = embs * weights.unsqueeze(1)
    embs = embs.sum(0)

    return embs.type(torch.float32)


class EMBDataset(Dataset):
    def __init__(self, df, mode, vtb_embs, rtk_embs):
        self.df = df
        self.mode = mode
        self.vtb_embs = vtb_embs
        self.rtk_embs = rtk_embs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        uid = self.df.loc[idx, "user_id"]

        if self.mode == "vtb":
            fv = get_feature_vector_vtb(self.df, uid, self.vtb_embs)
        elif self.mode == "rtk":
            fv = get_feature_vector_rtk(self.df, uid, self.rtk_embs)

        return fv