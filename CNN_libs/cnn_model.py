"""1D convolutional U-Net mapping multi-angle EEI to elasto-petrophysical logs.

Input  (batch, n_chi + 2, L): EEI at n_chi angles as channels, plus a mask
       channel (1 inside the window) and a normalised-TWT channel.
Output per time sample:
  mu      (batch, 5, L)  mean of vp, vs, rho, porosity, vshale (standardised)
  logvar  (batch, 5, L)  log variance of the same (aleatoric uncertainty)
  logits  (batch, 4, L)  facies scores: shale, silt, sand, calcareous sand

Dropout stays in every block so the network can be sampled with Monte Carlo
dropout at inference (epistemic uncertainty).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, dilation=1, p_drop=0.1):
        super().__init__()
        pad = dilation * (k // 2)
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p_drop)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        for c in (self.c1, self.c2):
            nn.init.kaiming_normal_(c.weight, nonlinearity="relu")

    def forward(self, x):
        h = self.drop(F.gelu(self.b1(self.c1(x))))
        h = self.b2(self.c2(h))
        return F.gelu(h + self.skip(x))


class EEINet(nn.Module):
    def __init__(self, n_in, n_reg=5, n_cls=4, width=48, levels=3, p_drop=0.1):
        super().__init__()
        w = [width * 2 ** i for i in range(levels + 1)]
        self.stem = ResBlock(n_in, w[0], p_drop=p_drop)
        self.enc = nn.ModuleList([ResBlock(w[i], w[i], p_drop=p_drop) for i in range(levels)])
        self.down = nn.ModuleList([nn.Conv1d(w[i], w[i + 1], 4, stride=2, padding=1)
                                   for i in range(levels)])
        # dilated blocks widen the receptive field at the coarsest scale
        self.mid = nn.Sequential(*[ResBlock(w[-1], w[-1], k=3, dilation=d, p_drop=p_drop)
                                   for d in (1, 2, 4)])
        self.up = nn.ModuleList([nn.ConvTranspose1d(w[i + 1], w[i], 4, stride=2, padding=1)
                                 for i in reversed(range(levels))])
        self.dec = nn.ModuleList([ResBlock(2 * w[i], w[i], p_drop=p_drop)
                                  for i in reversed(range(levels))])
        self.mu = nn.Conv1d(w[0], n_reg, 1)
        self.logvar = nn.Conv1d(w[0], n_reg, 1)
        self.logits = nn.Conv1d(w[0], n_cls, 1)
        self.levels = levels

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for enc, down in zip(self.enc, self.down):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], dim=1))
        return self.mu(h), self.logvar(h).clamp(-10.0, 5.0), self.logits(h)


class KerasNet(nn.Module):
    """Port of the Keras PetroNet of the manuscript notebooks
    (Elastic_properties_from_seismic_HTMT_04_June.ipynb): the angles x time gather is a
    one-channel image; 4 blocks Conv2D -> BatchNorm -> ReLU -> Dropout(0.3) with
    (filters, kernel) = (64, 2x7), (32, 2x15), (16, 1x30), (8, 1x60), stride 2 in time,
    'same' padding; then Flatten -> Dense(n_reg * L). The notebook ends with a ReLU on
    raw-unit outputs; here the targets are standardised, so there is no final ReLU.

    Same interface as EEINet: takes (batch, n_chi + 2, L) and ignores the mask/TWT
    channels; it has no aleatoric head (logvar = -30, so MC-dropout spread is the only
    uncertainty) and no facies head (logits = 0); train with --reg-loss mse --cls-weight 0.
    """

    BLOCKS = ((64, (2, 7)), (32, (2, 15)), (16, (1, 30)), (8, (1, 60)))

    def __init__(self, n_chi, n_reg, n_cls, length, p_drop=0.3):
        super().__init__()
        self.n_chi, self.n_reg, self.n_cls, self.length = n_chi, n_reg, n_cls, length
        layers, cin, h = [], 1, length
        self.pads = []
        for cout, k in self.BLOCKS:
            layers.append(nn.ModuleDict(dict(conv=nn.Conv2d(cin, cout, k, stride=(1, 2)),
                                              bn=nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
                                              drop=nn.Dropout(p_drop))))
            # Keras 'same': out = ceil(n / s), total pad = (out - 1) s + k - n, more at the end
            pa = k[0] - 1
            out = -(-h // 2)
            pt = max((out - 1) * 2 + k[1] - h, 0)
            self.pads.append((pt // 2, pt - pt // 2, pa // 2, pa - pa // 2))
            cin, h = cout, out
            nn.init.xavier_uniform_(layers[-1]["conv"].weight)
            nn.init.zeros_(layers[-1]["conv"].bias)
        self.blocks = nn.ModuleList(layers)
        self.fc = nn.Linear(cin * n_chi * h, n_reg * length)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = x[:, None, :self.n_chi]                         # (B, 1, n_chi, L)
        for blk, pad in zip(self.blocks, self.pads):
            h = blk["drop"](F.relu(blk["bn"](blk["conv"](F.pad(h, pad)))))
        mu = self.fc(h.flatten(1)).view(-1, self.n_reg, self.length)
        z = mu.new_zeros(mu.shape[0], self.n_cls, self.length)
        return mu, torch.full_like(mu, -30.0), z


def build_model(model_args):
    """EEINet or KerasNet from the model_args stored in model.pt."""
    a = dict(model_args)
    if a.pop("arch", "unet") == "keras":
        return KerasNet(a["n_in"] - 2, a["n_reg"], a["n_cls"], a["length"], a["p_drop"])
    a.pop("length", None)
    return EEINet(**a)


def enable_mc_dropout(model):
    """Eval mode (BatchNorm uses running statistics) with dropout still active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


def enable_keras_training_mode(model):
    """Keras model(X, training=True), as in the manuscript's MC dropout: dropout active
    and BatchNorm normalising with the statistics of the batch (the field line) instead
    of the running averages. Running averages are left unchanged."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0                 # running = (1 - 0) running + 0 batch
    return model
