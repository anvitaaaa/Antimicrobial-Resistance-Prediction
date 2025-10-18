#!/usr/bin/env python3
"""
domain_adapt.py
- Simple DANN implementation using PyTorch.
- Trains a shared feature extractor, label predictor, and domain discriminator via GRL.
- Note: This is intentionally compact for clarity, not production-optimized.
"""
import argparse, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, x):
        return self.net(x)

def load_tensor_data(X_path, y_path, domain_path=None):
    X = pd.read_parquet(X_path).values.astype(np.float32)

    # read y as 1D
    y_df = pd.read_csv(y_path)
    if y_df.shape[1] == 1:
        y = y_df.iloc[:,0].values.astype(np.float32)
    else:
        raise ValueError(f"Expected y CSV to have a single column, got {y_df.shape[1]}")
    
    if domain_path:
        dom_df = pd.read_csv(domain_path)
        if dom_df.shape[1] == 1:
            dom = dom_df.iloc[:,0].astype('category').cat.codes.values
        else:
            raise ValueError(f"Expected domain CSV to have a single column, got {dom_df.shape[1]}")
    else:
        dom = np.zeros(len(y), dtype=int)

    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(dom)


def train_dann(X_src, y_src, dom_src, X_tgt, y_tgt, dom_tgt, epochs=20, batch_size=256, lr=1e-3, device='cpu'):
    input_dim = X_src.shape[1]
    fe = FeatureExtractor(input_dim).to(device)
    lp = LabelPredictor().to(device)
    dd = DomainDiscriminator().to(device)
    opt = optim.Adam(list(fe.parameters()) + list(lp.parameters()) + list(dd.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    src_ds = TensorDataset(X_src, y_src, dom_src)
    tgt_ds = TensorDataset(X_tgt, y_tgt, dom_tgt)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        fe.train(); lp.train(); dd.train()
        total_label_loss = 0.0; total_dom_loss = 0.0
        for (bx, by, bdom), (bxt, byt, bdmt) in zip(src_loader, tgt_loader):
            bx = bx.to(device); by = by.to(device)
            bxt = bxt.to(device)
            # forward
            feat_s = fe(bx)
            logits = lp(feat_s)
            label_loss = bce(logits, by)
            # domain loss: combine source and target
            feat_all = torch.cat([feat_s, fe(bxt)], dim=0)
            # gradient reversal
            p = float(epoch) / epochs
            lambd = 2. / (1.+np.exp(-10*p)) - 1
            feat_rev = GRL.apply(feat_all, lambd)
            dom_logits = dd(feat_rev)
            dom_labels = torch.cat([torch.zeros(feat_s.size(0), dtype=torch.long), torch.ones(feat_all.size(0)-feat_s.size(0), dtype=torch.long)], dim=0).to(device)
            dom_loss = ce(dom_logits, dom_labels)
            loss = label_loss + dom_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_label_loss += label_loss.item(); total_dom_loss += dom_loss.item()
        print(f"Epoch {epoch+1}/{epochs} label_loss={total_label_loss/len(src_loader):.4f} dom_loss={total_dom_loss/len(src_loader):.4f}")
    return fe, lp, dd

def evaluate(fe, lp, X, y, device='cpu'):
    fe.eval(); lp.eval()
    X = X.to(device); y = y.to(device)
    with torch.no_grad():
        feats = fe(X)
        logits = lp(feats)
        probs = torch.sigmoid(logits).cpu().numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    auc = roc_auc_score(y.cpu().numpy(), probs)
    ap = average_precision_score(y.cpu().numpy(), probs)
    print("Eval AUROC:", auc, "AUPRC:", ap)
    return probs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--X_src", required=True)
    p.add_argument("--y_src", required=True)
    p.add_argument("--dom_src", required=True)
    p.add_argument("--X_tgt", required=True)
    p.add_argument("--y_tgt", required=True)
    p.add_argument("--dom_tgt", required=True)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xs, ys, doms = load_tensor_data(args.X_src, args.y_src, args.dom_src)
    Xt, yt, domt = load_tensor_data(args.X_tgt, args.y_tgt, args.dom_tgt)
    fe, lp, dd = train_dann(Xs, ys, doms, Xt, yt, domt, epochs=args.epochs, device=device)
    evaluate(fe, lp, Xt, yt, device=device)
