"""Self-contained algorithms used by the paper experiments."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from torch import nn, optim


def seed_everything(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norm, 1e-12)


def load_features(
    data_dir: str | Path,
    dataset: str,
    num_classes: int | None = None,
    ordering: str = "class_iid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    if dataset == "cub200":
        train_name = "cub200_resnet50_train.mat"
        test_name = "cub200_resnet50_test.mat"
    elif dataset == "fgvc_aircraft48":
        train_name = "fgvc_aircraft48_resnet50_train.mat"
        test_name = "fgvc_aircraft48_resnet50_test.mat"
    else:
        raise ValueError("unsupported dataset: %s" % dataset)
    train = loadmat(data_dir / train_name)
    test = loadmat(data_dir / test_name)
    X_train = normalize_rows(train["X"].astype(np.float32))
    y_train = train["y"].ravel().astype(np.int64)
    X_test = normalize_rows(test["X"].astype(np.float32))
    y_test = test["y"].ravel().astype(np.int64)
    if num_classes is not None:
        train_mask = y_train < num_classes
        test_mask = y_test < num_classes
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
    if ordering == "class_iid":
        order = np.argsort(y_train, kind="stable")
        X_train, y_train = X_train[order], y_train[order]
    elif ordering == "iid":
        rng = np.random.RandomState(66)
        order = rng.permutation(len(y_train))
        X_train, y_train = X_train[order], y_train[order]
    else:
        raise ValueError("ordering must be class_iid or iid")
    return X_train, y_train, X_test, y_test


def compute_means(
    X: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    X_t = X if torch.is_tensor(X) else torch.as_tensor(X, dtype=torch.float32)
    y_t = y if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.long)
    return torch.stack([X_t[y_t == c].mean(dim=0) for c in range(num_classes)])


def ncm_scores(
    X: torch.Tensor, means: torch.Tensor, W: torch.Tensor | None = None
) -> torch.Tensor:
    if W is not None:
        X = X @ W.T
        means = means @ W.T
    return X @ means.T - 0.5 * (means * means).sum(dim=1)


def evaluate_scores(scores: torch.Tensor, y: torch.Tensor) -> float:
    prediction = scores.argmax(dim=1).detach().cpu().numpy()
    return float(accuracy_score(y.detach().cpu().numpy(), prediction) * 100.0)


class NCMML(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        means: torch.Tensor,
        seed: int = 111,
        initial_W: torch.Tensor | None = None,
    ):
        super().__init__()
        self.register_buffer("means", means.detach().clone())
        if initial_W is None:
            W = self.pca_init(self.means, proj_dim, seed)
        else:
            W = initial_W.detach().clone().to(self.means)
        if tuple(W.shape) != (proj_dim, input_dim):
            raise ValueError("invalid initial W shape: %s" % (tuple(W.shape),))
        self.W = nn.Parameter(W)

    @staticmethod
    def pca_init(means: torch.Tensor, proj_dim: int, seed: int) -> torch.Tensor:
        centered = means - means.mean(dim=0, keepdim=True)
        with torch.no_grad():
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            W = Vh[: min(proj_dim, Vh.shape[0])].clone()
            if W.shape[0] < proj_dim:
                generator = torch.Generator().manual_seed(seed)
                extra = torch.randn(
                    proj_dim - W.shape[0], means.shape[1], generator=generator
                ).to(means)
                extra /= extra.norm(dim=1, keepdim=True).clamp(min=1e-8)
                W = torch.cat([W, extra], dim=0)
        return W

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return ncm_scores(X, self.means, self.W)


def train_ncmml(
    X: torch.Tensor,
    y: torch.Tensor,
    means: torch.Tensor,
    proj_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    initial_W: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    reseed: bool = True,
) -> tuple[torch.Tensor, float]:
    device = X.device
    if reseed:
        seed_everything(seed, device)
    sync(device)
    started = time.perf_counter()
    model = NCMML(
        X.shape[1], proj_dim, means, seed=seed, initial_W=initial_W
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(
        reduction="none" if sample_weights is not None else "mean"
    )
    n = len(y)
    mb = min(batch_size, n)
    for _ in range(epochs):
        permutation = np.random.permutation(n)
        for offset in range(0, n, mb):
            batch_np = permutation[offset : offset + mb]
            batch = torch.as_tensor(batch_np, device=device, dtype=torch.long)
            loss_items = criterion(model(X[batch]), y[batch])
            if sample_weights is None:
                loss = loss_items
            else:
                weights = sample_weights[batch]
                loss = (loss_items * weights).sum() / weights.sum().clamp(min=1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    sync(device)
    elapsed = time.perf_counter() - started
    return model.W.detach().clone(), float(elapsed)


def closest_pair(X: torch.Tensor) -> tuple[int, int]:
    distances = torch.cdist(X, X).square()
    distances.fill_diagonal_(float("inf"))
    flat = int(torch.argmin(distances).item())
    i, j = divmod(flat, len(X))
    return min(i, j), max(i, j)


def compress_exstream(
    X: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = X.detach().clone()
    counts = torch.ones(len(X), device=X.device)
    while len(prototypes) > capacity:
        i, j = closest_pair(prototypes)
        merged_count = counts[i] + counts[j]
        merged = (
            counts[i] * prototypes[i] + counts[j] * prototypes[j]
        ) / merged_count
        keep = [k for k in range(len(prototypes)) if k not in (i, j)]
        prototypes = torch.cat(
            [prototypes[keep], merged.unsqueeze(0)], dim=0
        )
        counts = torch.cat([counts[keep], merged_count.unsqueeze(0)])
    return prototypes, counts


@dataclass
class IncrementalResult:
    final_accuracy: float
    accuracy_curve: list[float]
    training_s: float
    w_update_s: float
    w_updates: int
    trainable_parameters: int
    final_slots: int
    prototype_mass: float
    final_W: torch.Tensor | None
    means: torch.Tensor


def run_hybrid_exstream_ncmml(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    num_classes: int,
    proj_dim: int,
    capacity: int,
    epochs_per_update: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    warm_start: bool = True,
) -> IncrementalResult:
    device = X_train.device
    seed_everything(seed, device)
    class_samples = [X_train[y_train == c] for c in range(num_classes)]
    means = torch.stack([samples.mean(dim=0) for samples in class_samples])
    stored_X: list[torch.Tensor] = []
    stored_count: list[torch.Tensor] = []
    W = None
    accuracy_curve: list[float] = []
    w_time = 0.0
    total_training_s = 0.0
    for class_id in range(num_classes):
        sync(device)
        phase_started = time.perf_counter()
        current = class_samples[class_id]
        if class_id >= 1:
            training_parts = stored_X + [current]
            label_parts = [
                torch.full(
                    (len(stored_X[c]),), c, device=device, dtype=torch.long
                )
                for c in range(class_id)
            ] + [
                torch.full(
                    (len(current),), class_id, device=device, dtype=torch.long
                )
            ]
            weight_parts = stored_count + [
                torch.ones(len(current), device=device)
            ]
            train_X = torch.cat(training_parts)
            train_y = torch.cat(label_parts)
            train_w = torch.cat(weight_parts)
            W, elapsed = train_ncmml(
                train_X,
                train_y,
                means[: class_id + 1],
                proj_dim,
                epochs_per_update,
                learning_rate,
                batch_size,
                seed,
                initial_W=W if warm_start else None,
                sample_weights=train_w,
                reseed=False,
            )
            w_time += elapsed
        prototypes, counts = compress_exstream(current, capacity)
        stored_X.append(prototypes)
        stored_count.append(counts)
        sync(device)
        total_training_s += time.perf_counter() - phase_started
        accuracy_curve.append(
            evaluate_scores(
                ncm_scores(X_test[y_test <= class_id], means[: class_id + 1], W),
                y_test[y_test <= class_id],
            )
        )
    mass = float(sum(float(count.sum().item()) for count in stored_count))
    return IncrementalResult(
        final_accuracy=accuracy_curve[-1],
        accuracy_curve=accuracy_curve,
        training_s=float(total_training_s),
        w_update_s=float(w_time),
        w_updates=max(num_classes - 1, 0),
        trainable_parameters=proj_dim * X_train.shape[1],
        final_slots=sum(len(part) for part in stored_X),
        prototype_mass=mass,
        final_W=W,
        means=means,
    )


class MLP(nn.Module):
    def __init__(
        self,
        shape: Iterable[int],
        seed: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        torch.manual_seed(seed)
        sizes = list(shape)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(sizes[:-2], sizes[1:-1]):
            linear = nn.Linear(input_dim, output_dim)
            nn.init.normal_(
                linear.weight,
                mean=0.0,
                std=math.sqrt(2.0 / (input_dim + output_dim)),
            )
            nn.init.constant_(linear.bias, 1.0)
            layers.append(linear)
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_dim))
            if activation == "relu":
                act: nn.Module = nn.ReLU()
            elif activation == "elu":
                act = nn.ELU()
            else:
                raise ValueError("unsupported activation: %s" % activation)
            layers.extend([act, nn.Dropout(dropout)])
        output = nn.Linear(sizes[-2], sizes[-1])
        nn.init.normal_(
            output.weight,
            mean=0.0,
            std=math.sqrt(2.0 / (sizes[-2] + sizes[-1])),
        )
        nn.init.constant_(output.bias, 1.0)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def train_mlp_epoch(
    model: MLP,
    optimizer: optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    order = np.random.permutation(len(y))
    mb = min(batch_size, len(y))
    for offset in range(0, len(y), mb):
        start = min(offset, len(y) - mb)
        batch = torch.as_tensor(
            order[start : start + mb], device=X.device, dtype=torch.long
        )
        loss = criterion(model(X[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def predict_mlp(model: MLP, X: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(X), batch_size):
            predictions.append(
                model(X[offset : offset + batch_size]).argmax(dim=1).cpu()
            )
    return torch.cat(predictions).numpy()


def update_exstream_lists(
    prototypes: list[list[torch.Tensor]],
    counts: list[list[float]],
    x: torch.Tensor,
    class_id: int,
    capacity: int,
) -> None:
    if len(prototypes[class_id]) < capacity:
        prototypes[class_id].append(x.detach().clone())
        counts[class_id].append(1.0)
        return
    X = torch.stack(prototypes[class_id])
    i, j = closest_pair(X)
    w_i, w_j = counts[class_id][i], counts[class_id][j]
    merged = (w_i * X[i] + w_j * X[j]) / (w_i + w_j)
    prototypes[class_id][i] = x.detach().clone()
    counts[class_id][i] = 1.0
    prototypes[class_id][j] = merged.detach().clone()
    counts[class_id][j] = w_i + w_j


def benchmark_predict(
    predict_fn, num_samples: int, device: torch.device, warmups: int, repeats: int
) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmups):
            predict_fn()
        timings = []
        for _ in range(repeats):
            sync(device)
            started = time.perf_counter()
            predict_fn()
            sync(device)
            timings.append(time.perf_counter() - started)
    values = np.asarray(timings)
    total_s = float(values.sum())
    return {
        "inference_total_s": total_s,
        "inference_mean_s": float(values.mean()),
        "throughput_samples_per_s": float(
            num_samples * repeats / total_s
        ),
    }
