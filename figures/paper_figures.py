"""Generate the four figures associated with the paper's method/results."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import NCMML  # noqa: E402


def fig1(output: Path):
    points = np.asarray([
        [-3.0, -1.6], [-1.15, 1.20], [-0.25, 1.00],
        [2.15, 2.15], [2.65, -1.25],
    ])
    counts = np.asarray([1, 2, 4, 1, 3])
    incoming = np.asarray([0.75, -2.45])
    distances = ((points[:, None] - points[None]) ** 2).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    if i > j:
        i, j = j, i
    merged_count = counts[i] + counts[j]
    merged = (counts[i] * points[i] + counts[j] * points[j]) / merged_count
    keep = [k for k in range(len(points)) if k not in (i, j)]
    after = np.concatenate([points[keep], merged[None], incoming[None]])
    after_counts = np.concatenate([counts[keep], [merged_count, 1]])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3.2, 3.1)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    axes[0].set_title("Full class buffer")
    axes[1].set_title("Merge closest pair")
    axes[2].set_title("Insert newest sample")
    axes[0].scatter(points[:, 0], points[:, 1], s=90 + 35 * counts)
    axes[0].scatter(*incoming, marker="D", s=140, facecolors="none")
    axes[1].scatter(points[:, 0], points[:, 1], s=90 + 35 * counts)
    axes[1].plot(points[[i, j], 0], points[[i, j], 1], color="tab:red")
    axes[1].scatter(*merged, s=90 + 35 * merged_count, color="tab:purple")
    axes[2].scatter(
        after[:, 0], after[:, 1], s=90 + 35 * after_counts,
        c=["tab:blue"] * len(keep) + ["tab:purple", "tab:orange"],
    )
    for ax, pts, weights in (
        (axes[0], points, counts),
        (axes[2], after, after_counts),
    ):
        for point, weight in zip(pts, weights):
            ax.annotate("w=%d" % weight, point, xytext=(0, 9),
                        textcoords="offset points", ha="center")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def fig2(output: Path, seed: int = 111):
    rng = np.random.RandomState(seed)
    centers = np.asarray([
        [-2.3, -1.2, -1.3], [-1.3, 1.3, 1.3],
        [2.3, -1.2, -1.3], [1.3, 1.3, 1.3],
    ], dtype=np.float32)
    covariance = np.diag([0.68**2, 0.82**2, 3.8**2])
    X, y = [], []
    for class_id, center in enumerate(centers):
        X.append(rng.multivariate_normal(center, covariance, 160))
        y.append(np.full(160, class_id))
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    means = np.stack([X[y == c].mean(axis=0) for c in range(4)]).astype(np.float32)
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    model = NCMML(3, 2, torch.as_tensor(means), seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for _ in range(160):
        loss = criterion(model(X_t), y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    W = model.W.detach().numpy()
    projected = X @ W.T
    projected_means = means @ W.T
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    fig = plt.figure(figsize=(11.5, 5.2))
    left = fig.add_subplot(1, 2, 1, projection="3d")
    right = fig.add_subplot(1, 2, 2)
    for c, color in enumerate(colors):
        mask = y == c
        left.scatter(*X[mask].T, s=10, alpha=0.4, color=color)
        left.scatter(*means[c], s=130, color=color, edgecolor="black")
        right.scatter(*projected[mask].T, s=11, alpha=0.5, color=color)
        right.scatter(*projected_means[c], s=140, color=color, edgecolor="black")
    left.set_xlabel("Feature 1")
    left.set_ylabel("Feature 2")
    left.set_zlabel("Noisy feature")
    right.set_xlabel("Metric dimension 1")
    right.set_ylabel("Metric dimension 2")
    right.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def fig4(output: Path):
    width, height = 1300, 570
    boxes = [
        (60, 150, 300, 150, "Old classes\\nExStream prototypes\\nwith merge counts"),
        (60, 350, 300, 120, "Current class\\nfull samples, weight=1"),
        (500, 220, 310, 150, "Count-weighted NCMML\\nWarm-start W from\\nprevious class phase"),
        (950, 150, 290, 120, "Updated metric W\\nNCM inference"),
        (950, 350, 290, 120, "Compress current class\\nfor future phases"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (
            width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="a" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z"/></marker></defs>',
    ]
    for x, y, w, h, label in boxes:
        parts.append(
            '<rect x="%d" y="%d" width="%d" height="%d" rx="12" fill="#f4f4f4" stroke="#333"/>' % (x, y, w, h))
        lines = label.split("\\n")
        for index, line in enumerate(lines):
            parts.append(
                '<text x="%d" y="%d" font-family="Arial" font-size="19" text-anchor="middle">%s</text>' % (
                    x + w // 2, y + h // 2 + (index - len(lines) // 2) * 25,
                    html.escape(line)))
    for x1, y1, x2, y2 in (
        (360, 225, 495, 270), (360, 410, 495, 320),
        (810, 270, 945, 210), (810, 320, 945, 405),
    ):
        parts.append(
            '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="2.5" marker-end="url(#a)"/>' % (x1, y1, x2, y2))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def fig5(output: Path, result_json: Path):
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    classes = payload["seen_classes"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for capacity, curve in sorted(payload["curves"].items(), key=lambda x: int(x[0])):
        sampled_x = classes[9::10]
        sampled_y = curve[9::10]
        label = "Full buffer" if int(capacity) == 30 else "Buffer size=%s" % capacity
        ax.plot(sampled_x, sampled_y, marker="o", label=label)
    ax.set_xlabel("Number of seen classes", fontsize=15)
    ax.set_ylabel("Accuracy on currently seen classes (%)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=["fig1", "fig2", "fig4", "fig5", "all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument(
        "--capacity-results",
        type=Path,
        default=ROOT / "outputs" / "fig5_capacity" / "capacity_study.json",
    )
    args = parser.parse_args()
    selected = ["fig1", "fig2", "fig4", "fig5"] if args.figure == "all" else [args.figure]
    if "fig1" in selected:
        fig1(args.output_dir / "fig1_exstream_mechanism.png")
    if "fig2" in selected:
        fig2(args.output_dir / "fig2_ncmml_projection.png")
    if "fig4" in selected:
        fig4(args.output_dir / "fig4_training_framework.svg")
    if "fig5" in selected:
        if not args.capacity_results.exists():
            raise FileNotFoundError(
                "Run experiments/fig5_capacity.py before drawing Fig. 5")
        fig5(args.output_dir / "fig5_capacity.png", args.capacity_results)


if __name__ == "__main__":
    main()
