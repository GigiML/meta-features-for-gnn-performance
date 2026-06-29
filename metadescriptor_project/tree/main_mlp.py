from typing import Sequence
import argparse
from sklearn.metrics import r2_score
import numpy as np
import torch
from src.tree.dataset_dtr import Dataset
import torch.nn as nn
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import csv
from itertools import product
from sklearn.preprocessing import StandardScaler


def save_results_csv(results, output_file):
    if not results:
        print("No results to save.")
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_file}")


class MLPRegressor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.0
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(model, train_loader, optimizer, loss_fn, device, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            n += bs
        mean_loss = epoch_loss / n
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch + 1:03d} train_loss={mean_loss:.4f}")


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            mae = torch.mean(torch.abs(pred - yb))
            bs = xb.size(0)

            total_loss += loss.item() * bs
            total_mae += mae.item() * bs
            n += bs

            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(pred.cpu().numpy())

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all) if len(y_true_all) > 1 else np.nan
    return total_loss / n, total_mae / n, r2


def parse_hidden_dims(text: str):
    return tuple(int(x) for x in text.split(",")) if text else (128, 64)


def prepare_data(summary_file, perf_file, perf, family, source, ablation):
    dataset_names, feature_names, target_names, X, y = Dataset.get_dataset(
        summary_file=summary_file,
        perf_file=perf_file,
        perf=perf,
        family=family,
        source=source,
        ablation=ablation,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    dataset_names = np.asarray(dataset_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return {
        "dataset_names": dataset_names,
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "X": X,
        "y": y,
        "in_dim": X.shape[1],
        "scaler": scaler,
    }


def make_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_split(X_train, y_train, X_val, y_val, in_dim, hidden_dims):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, 64)
    model = MLPRegressor(in_dim, parse_hidden_dims(hidden_dims), 0).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, optimizer, loss_fn, device, 100)

    train_mse, train_mae, train_r2 = evaluate(model, train_loader, loss_fn, device)
    val_mse, val_mae, val_r2 = evaluate(model, val_loader, loss_fn, device)
    return train_mse, train_mae, train_r2, val_mse, val_mae, val_r2


def run_holdout(X, y, in_dim, hidden_dims, family, source, ablation):
    X_train, X_val, y_train, y_val = X, X, y, y
    result = train_one_split(X_train, y_train, X_val, y_val, in_dim, hidden_dims)
    print(
        f"holdout train_mse={result[0]:.4f} train_mae={result[1]:.4f} train_r2={result[2]:.4f} "
        f"val_mse={result[3]:.4f} val_mae={result[4]:.4f} val_r2={result[5]:.4f}"
    )
    return {
        "family": family,
        "source": source,
        "ablation": ablation,
        "train_mse": result[0],
        "train_mae": result[1],
        "train_r2": result[2],
        "val_mse": result[3],
        "val_mae": result[4],
        "val_r2": result[5],
    }


def run_kfold(X, y, in_dim, hidden_dims, model="GCN", perf="Accuracy"):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/{10} ---")
        result = train_one_split(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], in_dim, hidden_dims
        )
        metrics.append(result)
        print(
            f"fold={fold} train_mse={result[0]:.4f} train_mae={result[1]:.4f} train_r2={result[2]:.4f} "
            f"val_mse={result[3]:.4f} val_mae={result[4]:.4f} val_r2={result[5]:.4f}"
        )
    metrics = np.array(metrics)
    print(
        f"\nkfold mean_train_mse={metrics[:, 0].mean():.4f} mean_train_mae={metrics[:, 1].mean():.4f} mean_train_r2={metrics[:, 2].mean():.4f} "
        f"mean_val_mse={metrics[:, 3].mean():.4f} mean_val_mae={metrics[:, 4].mean():.4f} mean_val_r2={metrics[:, 5].mean():.4f}"
    )
    mean_results = [
        {
            "mean_train_mse": np.nanmean(metrics[:, 0]),
            "mean_train_mae": np.nanmean(metrics[:, 1]),
            "mean_train_r2": np.nanmean(metrics[:, 2]),
            "mean_val_mse": np.nanmean(metrics[:, 3]),
            "mean_val_mae": np.nanmean(metrics[:, 4]),
            "mean_val_r2": np.nanmean(metrics[:, 5]),
        }
    ]
    output_file = f"{model}_{perf}_kfold_results.csv"
    save_results_csv(mean_results, output_file)
    return mean_results


def run_loo(
    X, y, in_dim, hidden_dims, dataset_names=None, model="GCN", perf="Accuracy"
):
    loo = LeaveOneOut()
    metrics = []
    for fold, (train_idx, val_idx) in enumerate(loo.split(X), start=1):
        print(f"\n--- LOO fold {fold}/{len(X)} ---")
        result = train_one_split(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], in_dim, hidden_dims
        )
        # metrics.append(result)
        print(
            f"loo_fold={fold} train_mse={result[0]:.4f} train_mae={result[1]:.4f} train_r2={result[2]:.4f} "
            f"val_mse={result[3]:.4f} val_mae={result[4]:.4f} val_r2={result[5]:.4f}"
        )
        row = {
            "tested_dataset": dataset_names[val_idx[0]]
            if dataset_names is not None
            else val_idx[0],
            "train_mse": result[0],
            "train_mae": result[1],
            "train_r2": result[2],
            "val_mse": result[3],
            "val_mae": result[4],
            "val_r2": result[5],
        }
        metrics.append(row)
    output_file = f"{model}_{perf}_loo_results.csv"
    save_results_csv(metrics, output_file)
    # print(
    #     f"\nloo mean_train_mse={np.nanmean(metrics[:,0]):.4f} mean_train_mae={np.nanmean(metrics[:,1]):.4f} mean_train_r2={np.nanmean(metrics[:,2]):.4f} "
    #     f"mean_val_mse={np.nanmean(metrics[:,3]):.4f} mean_val_mae={np.nanmean(metrics[:,4]):.4f} mean_val_r2={np.nanmean(metrics[:,5]):.4f}"
    # )


def main():
    parser = argparse.ArgumentParser(
        description="Main script for MLP regression on dataset features"
    )
    parser.add_argument("--summary-file", type=str, default="summary.csv")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--perf", type=str, default="Accuracy")
    parser.add_argument(
        "--family", type=str, default=None, choices=["None", "F1", "F2", "F3", "F4"]
    )
    parser.add_argument(
        "--source", type=str, default=None, choices=["NORMAL", "GSPAN", "None"]
    )
    parser.add_argument("--ablation", type=bool, default=False)
    parser.add_argument(
        "--mode", type=str, choices=["holdout", "kfold", "loo"], default="holdout"
    )
    parser.add_argument("--hidden-dims", type=str, default="128,64")
    args = parser.parse_args()

    target_file = f"{args.model}.csv"

    data = prepare_data(
        summary_file=args.summary_file,
        perf_file=target_file,
        perf=args.perf,
        family=None,
        source=None,
        ablation=True,
    )

    X = data["X"]
    y = data["y"]
    in_dim = data["in_dim"]
    hidden_dims = args.hidden_dims
    dataset_names = data["dataset_names"]

    print(f"Loaded {len(y)} samples with in_dim={in_dim}")
    print(f"Target: {data['target_names']}")
    print(f"First 5 features: {data['feature_names'][:5]}")

    if args.mode == "holdout":
        families = ["F1", "F2", "F3", "F4"]
        sources = ["NORMAL", "GSPAN"]
        flags_configs = product(families, sources)

        all_results = []

        data = prepare_data(
            summary_file=args.summary_file,
            perf_file=target_file,
            perf=args.perf,
            family=None,
            source=None,
            ablation=True,
        )

        X = data["X"]
        y = data["y"]
        in_dim = data["in_dim"]
        hidden_dims = args.hidden_dims

        global_result = run_holdout(
            X, y, in_dim, hidden_dims, family=None, source=None, ablation=True
        )
        all_results.append(global_result)

        for family, source in product(families, sources):
            for ablation in [True, False]:
                data = prepare_data(
                    summary_file=args.summary_file,
                    perf_file=target_file,
                    perf=args.perf,
                    family=family,
                    source=source,
                    ablation=ablation,
                )

                X = data["X"]
                y = data["y"]
                in_dim = data["in_dim"]

                result = run_holdout(
                    X,
                    y,
                    in_dim,
                    hidden_dims,
                    family=family,
                    source=source,
                    ablation=ablation,
                )
                all_results.append(result)
        output_file = f"{args.model}_{args.perf}_holdout_results.csv"
        save_results_csv(all_results, output_file)

    elif args.mode == "kfold":
        run_kfold(X, y, in_dim, hidden_dims, args.model, args.perf)
    else:
        run_loo(X, y, in_dim, hidden_dims, dataset_names, args.model, args.perf)


if __name__ == "__main__":
    main()
