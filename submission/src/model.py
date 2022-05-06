from torch import nn


class SModel(nn.Module):
    def __init__(
        self, anch_inp_dim=386, pn_inp_dim=402, h_dim=256, emb_dim=256
    ) -> None:
        super(SModel, self).__init__()
        self.anch_inp = nn.Sequential(
            nn.Linear(anch_inp_dim, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
            nn.Dropout(0.2),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
        )

        self.pn_inp = nn.Sequential(
            nn.Linear(pn_inp_dim, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
            nn.Dropout(0.2),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, emb_dim),
        )

    def forward(self, x, mode):
        if mode == "anchor":
            x = self.anch_inp(x)
            x = self.net(x)
        elif mode in ["positive", "negative"]:
            x = self.pn_inp(x)
            x = self.net(x)
        return x
