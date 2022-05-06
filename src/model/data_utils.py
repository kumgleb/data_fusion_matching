import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


def get_negative_sample(match_true, pos_sample):
    found = False
    while not found:
        sample = np.random.choice(match_true.rtk)
        if sample != pos_sample:
            return sample


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


class APNDataset(Dataset):
    def __init__(self, df, bank_trans_emb, rtk_clck_emb, vtb_embs, rtk_embs):
        self.df = df
        self.bank_trans_emb = bank_trans_emb
        self.rtk_clck_emb = rtk_clck_emb
        self.vtb_embs = vtb_embs
        self.rtk_embs = rtk_embs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        anchor_id = self.df.iloc[idx].bank
        positive_id = self.df.iloc[idx].rtk
        negative_id = get_negative_sample(self.df, positive_id)

        data = {
            "anchor": get_feature_vector_vtb(
                self.bank_trans_emb, anchor_id, self.vtb_embs
            ),
            "positive": get_feature_vector_rtk(
                self.rtk_clck_emb, positive_id, self.rtk_embs
            ),
            "negative": get_feature_vector_rtk(
                self.rtk_clck_emb, negative_id, self.rtk_embs
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
