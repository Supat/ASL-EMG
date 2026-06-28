"""
Off-device training: TCN encoder + subject-adversarial (DANN) head for
subject-independent ASL-alphabet decoding from Myo EMG+IMU.

Trains on a Mac/Colab with PyTorch (NOT Carnets). Evaluated under the same
Leave-One-Subject-Out protocol as decode_alphabet.ipynb so the number is
comparable to the classical baseline (LOSO ~0.295). Exports the inference path
(encoder + letter head) to Core ML via coremltools.

Why DANN here: LOSO is a domain-shift problem. A gradient-reversal domain head
that classifies which of the 8 *training* subjects a sample came from forces the
encoder to learn subject-invariant features (domain generalization — needs no
target-subject data, so it stays pure-LOSO).

Run:  python train_tcn_dann.py            # full 9-fold LOSO
      python train_tcn_dann.py --quick    # 2 folds, few epochs (smoke)
"""
import argparse, io, os, re, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

EMG = [f'EMG{i}' for i in range(8)]
IMU = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ', 'OR', 'OP', 'OY']
COLS = EMG + IMU
T = 50            # fixed window length (timesteps)
C = 17            # channels: 8 EMG + 9 IMU

# ---------------------------------------------------------------- data loading
def find_zip():
    root = os.getcwd()
    for _ in range(5):
        for dp, _d, files in os.walk(root):
            if 'alphabet_fingerspelling_dyfav.zip' in files:
                return os.path.join(dp, 'alphabet_fingerspelling_dyfav.zip')
        root = os.path.dirname(root)
    raise FileNotFoundError('alphabet_fingerspelling_dyfav.zip not found')

def load():
    ZF = zipfile.ZipFile(find_zip())
    def read(p):
        df = pd.read_csv(io.BytesIO(ZF.read(p)), header=None, names=COLS)
        return df[(df[EMG] != 0).any(axis=1)].reset_index(drop=True)
    def parse(p):
        u = next(x for x in p.split('/') if x.lower().startswith('user'))
        l = re.search(r'alphabet_([a-z]+)_', os.path.basename(p).lower()).group(1)
        return u, l
    paths = [p for p in ZF.namelist() if p.endswith('.csv')]
    rows = [(*parse(p), p) for p in paths]
    users = sorted({u for u, _, _ in rows})
    letters = sorted({l for _, l, _ in rows})
    # per-subject normalization stats (label-free): EMG 99th-pct ref, IMU mean/std
    ref, imu_mu, imu_sd = {}, {}, {}
    for u in users:
        emg = np.concatenate([read(p)[EMG].abs().values for uu, _, p in rows if uu == u])
        ref[u] = np.maximum(np.percentile(emg, 99, axis=0), 1.0)
        imu = np.concatenate([read(p)[IMU].values for uu, _, p in rows if uu == u]).astype(float)
        imu_mu[u], imu_sd[u] = imu.mean(0), imu.std(0) + 1e-8

    def to_fixed(seq):                       # (L,17) -> (17,T)
        L = len(seq)
        if L >= T:
            s = (L - T) // 2
            seq = seq[s:s + T]
        else:
            seq = np.pad(seq, ((0, T - L), (0, 0)))
        return seq.T.astype(np.float32)

    X, y, g = [], [], []
    for u, l, p in rows:
        d = read(p)
        emg = d[EMG].values.astype(float) / ref[u]        # signed, scaled
        imu = (d[IMU].values.astype(float) - imu_mu[u]) / imu_sd[u]
        X.append(to_fixed(np.concatenate([emg, imu], axis=1)))
        y.append(letters.index(l)); g.append(users.index(u))
    return np.stack(X), np.array(y), np.array(g), users, letters

# ------------------------------------------------------------------- augment
def augment(x, jitter=0.05, warp=0.15, rng=None):
    rng = rng or np.random
    x = x + rng.normal(0, jitter, x.shape).astype(np.float32)        # jitter
    knots = rng.normal(1.0, warp, (x.shape[0], 4)).astype(np.float32)  # per-channel
    grid = np.linspace(0, 3, x.shape[1])
    curve = np.stack([np.interp(grid, [0, 1, 2, 3], k) for k in knots])  # (C,T)
    return (x * curve).astype(np.float32)

class DS(Dataset):
    def __init__(self, X, y, d, train):
        self.X, self.y, self.d, self.train = X, y, d, train
        self.rng = np.random.RandomState(0)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]
        if self.train:
            x = augment(x, rng=self.rng)
        return torch.from_numpy(x), int(self.y[i]), int(self.d[i])

# --------------------------------------------------------------------- model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lamb * g, None

class Chomp(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return x[:, :, :-self.c].contiguous() if self.c else x

class TBlock(nn.Module):
    def __init__(self, ci, co, k, d, p=0.2):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, k, padding=pad, dilation=d), Chomp(pad), nn.BatchNorm1d(co), nn.ReLU(), nn.Dropout(p),
            nn.Conv1d(co, co, k, padding=pad, dilation=d), Chomp(pad), nn.BatchNorm1d(co), nn.ReLU(), nn.Dropout(p))
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
    def forward(self, x):
        r = x if self.down is None else self.down(x)
        return torch.relu(self.net(x) + r)

class TCN_DANN(nn.Module):
    def __init__(self, n_letters, n_domains, ch=64, feat=64):
        super().__init__()
        self.bn = nn.BatchNorm1d(C)
        self.tcn = nn.Sequential(TBlock(C, ch, 5, 1), TBlock(ch, ch, 5, 2), TBlock(ch, feat, 5, 4))
        self.label = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(), nn.Dropout(0.3), nn.Linear(feat, n_letters))
        self.domain = nn.Sequential(nn.Linear(feat, feat), nn.ReLU(), nn.Linear(feat, n_domains))
    def encode(self, x):
        h = self.tcn(self.bn(x))
        return h.mean(-1)                       # global average pool over time
    def forward(self, x, lamb=0.0):
        f = self.encode(x)
        return self.label(f), self.domain(GradReverse.apply(f, lamb))

class Infer(nn.Module):                          # Core ML export path (no domain head)
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.softmax(self.m.label(self.m.encode(x)), -1)

# ----------------------------------------------------------------- training
def run_fold(Xtr, ytr, dtr, Xte, yte, n_letters, epochs, device):
    doms = sorted(set(dtr.tolist())); dmap = {u: i for i, u in enumerate(doms)}
    dtr2 = np.array([dmap[u] for u in dtr])
    model = TCN_DANN(n_letters, len(doms)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    dl = DataLoader(DS(Xtr, ytr, dtr2, train=True), batch_size=64, shuffle=True)
    for ep in range(epochs):
        p = ep / max(epochs - 1, 1)
        lamb = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0      # DANN schedule
        model.train()
        for xb, yb, db in dl:
            xb, yb, db = xb.to(device), yb.to(device), db.to(device)
            ylog, dlog = model(xb, lamb)
            loss = ce(ylog, yb) + ce(dlog, db)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model.encode(torch.from_numpy(Xte).to(device))
        pred = model.label(pred).argmax(1).cpu().numpy()
    return (pred == yte).mean(), model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--export', default='ASLAlphabetTCN.mlpackage')
    args = ap.parse_args()
    epochs = 8 if args.quick else args.epochs
    torch.manual_seed(0); np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X, y, g, users, letters = load()
    print(f'Loaded {len(X)} recordings | {len(users)} users | {len(letters)} letters | X {X.shape}')
    folds = sorted(set(g.tolist()))
    if args.quick: folds = folds[:2]

    accs = []
    last_model = None
    for u in folds:
        te = g == u; tr = ~te
        acc, model = run_fold(X[tr], y[tr], g[tr], X[te], y[te], len(letters), epochs, device)
        last_model = model
        accs.append(acc)
        print(f'  hold-out {users[u]:7s}  LOSO acc = {acc:.3f}')
    print(f'\nTCN+DANN mean LOSO accuracy = {np.mean(accs):.3f} +/- {np.std(accs):.3f}'
          f'   (classical baseline = 0.295)')

    # Core ML export of the inference path (encoder + letter head)
    try:
        import coremltools as ct
        infer = Infer(last_model).eval().cpu()
        ex = torch.rand(1, C, T)
        ts = torch.jit.trace(infer, ex)
        ml = ct.convert(ts, inputs=[ct.TensorType(name='emg_imu', shape=(1, C, T))],
                        classifier_config=ct.ClassifierConfig(letters))
        ml.save(args.export)
        print('Core ML model saved to', args.export)
    except Exception as e:
        print('Core ML export skipped:', type(e).__name__, e)

if __name__ == '__main__':
    main()
