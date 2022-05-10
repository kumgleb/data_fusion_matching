from torch import nn


class SModel(nn.Module):
    def __init__(
        self, vtb_inp_dim=386, rtk_inp_dim=402, h_dim=256, emb_dim=256
    ) -> None:
        super(SModel, self).__init__()

        self.vtb_net = nn.Sequential(
            nn.Linear(inp_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.Droppout(0.3),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.Droppout(0.3),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, emb_dim),
        )

        self.rtk_net = nn.Sequential(
            nn.Linear(inp_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.Droppout(0.3),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.Droppout(0.3),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, emb_dim),
        )

    def forward(self, x, mode):

        if mode == "vtb":
            x = self.vtb_net(x)
        elif mode == "rtk":
            x = self.rtk_net(x)

        return x
