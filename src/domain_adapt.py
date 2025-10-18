#!/usr/bin/env python3
"""
domain_adapt.py
- DANN (Domain-Adversarial Neural Network) with training graphs.
"""
import argparse, pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, classification_report

# ================== Gradient Reversal Layer ==================
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

# ================== Network Blocks ==================
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
    def forward(self, x): return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

class DomainDiscriminator(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, x): return self.net(x)

# ================== Data Loader ==================
def load_tensor_data(X_path, y_path, domain_path=None):
    X = pd.read_parquet(X_path).values.astype(np.float32)
    y_df = pd.read_csv(y_path)
    if y_df.shape[1] != 1:
        raise ValueError(f"Expected y CSV to have a single column, got {y_df.shape[1]}")
    y = y_df.iloc[:,0].values.astype(np.float32)
    if domain_path:
        dom_df = pd.read_csv(domain_path)
        if dom_df.shape[1] != 1:
            raise ValueError(f"Expected domain CSV to have a single column, got {dom_df.shape[1]}")
        dom = dom_df.iloc[:,0].astype('category').cat.codes.values
    else:
        dom = np.zeros(len(y), dtype=int)
    return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(dom)

# ================== Training ==================
def train_dann(X_src, y_src, dom_src, X_tgt, y_tgt, dom_tgt, epochs=20, batch_size=256, lr=1e-3, device='cpu'):
    input_dim = X_src.shape[1]
    fe = FeatureExtractor(input_dim).to(device)
    lp = LabelPredictor().to(device)
    dd = DomainDiscriminator().to(device)

    opt = optim.Adam(list(fe.parameters()) + list(lp.parameters()) + list(dd.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    src_loader = DataLoader(TensorDataset(X_src, y_src, dom_src), batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(TensorDataset(X_tgt, y_tgt, dom_tgt), batch_size=batch_size, shuffle=True)

    label_losses, dom_losses, auc_scores = [], [], []

    for epoch in range(epochs):
        fe.train(); lp.train(); dd.train()
        total_label_loss, total_dom_loss = 0.0, 0.0

        for (bx, by, bdom), (bxt, byt, bdmt) in zip(src_loader, tgt_loader):
            bx, by, bxt = bx.to(device), by.to(device), bxt.to(device)

            feat_s = fe(bx)
            logits = lp(feat_s)
            label_loss = bce(logits, by)

            feat_all = torch.cat([feat_s, fe(bxt)], dim=0)
            p = float(epoch) / epochs
            lambd = 2. / (1.+np.exp(-10*p)) - 1
            feat_rev = GRL.apply(feat_all, lambd)
            dom_logits = dd(feat_rev)
            dom_labels = torch.cat([
                torch.zeros(feat_s.size(0), dtype=torch.long),
                torch.ones(feat_all.size(0)-feat_s.size(0), dtype=torch.long)
            ], dim=0).to(device)

            dom_loss = ce(dom_logits, dom_labels)
            loss = label_loss + dom_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_label_loss += label_loss.item()
            total_dom_loss += dom_loss.item()

        # Evaluate on target domain each epoch
        fe.eval(); lp.eval()
        with torch.no_grad():
            feats = fe(X_tgt.to(device))
            logits = lp(feats)
            probs = torch.sigmoid(logits).cpu().numpy()
            auc = roc_auc_score(y_tgt.numpy(), probs)
        auc_scores.append(auc)

        label_losses.append(total_label_loss / len(src_loader))
        dom_losses.append(total_dom_loss / len(src_loader))
        print(f"Epoch {epoch+1}/{epochs} | Label Loss: {label_losses[-1]:.4f} | Dom Loss: {dom_losses[-1]:.4f} | Target AUC: {auc:.4f}")

    # ======= PLOTTING =======
    plt.figure(figsize=(10,5))
    plt.plot(label_losses, label='Label Loss', linewidth=2)
    plt.plot(dom_losses, label='Domain Loss', linewidth=2)
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(auc_scores, color='purple', linewidth=2)
    plt.xlabel("Epochs"); plt.ylabel("AUC on Target Domain")
    plt.title("Domain Adaptation Progress")
    plt.grid(True); plt.tight_layout()
    plt.show()

    return fe, lp, dd

# ================== Evaluation ==================
def evaluate(fe, lp, X, y, device='cpu'):
    fe.eval(); lp.eval()
    with torch.no_grad():
        feats = fe(X.to(device))
        logits = lp(feats)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y.numpy(), probs)
    ap = average_precision_score(y.numpy(), probs)
    acc = accuracy_score(y.numpy(), preds)
    print("\n=== Final Evaluation on Target Domain ===")
    print("AUROC:", auc)
    print("AUPRC:", ap)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y.numpy(), preds, digits=4))
    return probs

# ================== Main ==================
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
    print("Using device:", device)

    Xs, ys, doms = load_tensor_data(args.X_src, args.y_src, args.dom_src)
    Xt, yt, domt = load_tensor_data(args.X_tgt, args.y_tgt, args.dom_tgt)

    fe, lp, dd = train_dann(Xs, ys, doms, Xt, yt, domt, epochs=args.epochs, device=device)
    evaluate(fe, lp, Xt, yt, device=device)
