"""CLI implementations for all quantitative results reported in the paper."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import optim

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from configs import paper_config as cfg  # noqa: E402
from src import (  # noqa: E402
    MLP,
    benchmark_predict,
    compute_means,
    evaluate_scores,
    load_features,
    ncm_scores,
    predict_mlp,
    run_hybrid_exstream_ncmml,
    seed_everything,
    sync,
    train_mlp_epoch,
    train_ncmml,
    update_exstream_lists,
)


def device_for(gpu_id: int) -> torch.device:
    return torch.device(
        "cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu"
    )


def write_payload(output_dir: Path, stem: str, payload: dict, rows: list[dict]):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / (stem + ".json")
    csv_path = output_dir / (stem + ".csv")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Saved", json_path)
    if rows:
        print("Saved", csv_path)


def tensors(data_dir: Path, dataset: str, num_classes: int, device):
    arrays = load_features(data_dir, dataset, num_classes, "class_iid")
    X_train, y_train, X_test, y_test = arrays
    return (
        torch.as_tensor(X_train, dtype=torch.float32, device=device),
        torch.as_tensor(y_train, dtype=torch.long, device=device),
        torch.as_tensor(X_test, dtype=torch.float32, device=device),
        torch.as_tensor(y_test, dtype=torch.long, device=device),
    )


def make_mlp(num_classes: int, device, seed: int) -> MLP:
    model = MLP(
        [cfg.INPUT_DIM, *cfg.MLP_HIDDEN_DIMS, num_classes],
        seed=seed,
        batch_norm=cfg.MLP_BATCH_NORM,
        dropout=cfg.MLP_DROPOUT,
        activation=cfg.MLP_ACTIVATION,
    )
    return model.to(device)


def offline_mlp(X_train, y_train, X_test, y_test, num_classes, args):
    device = X_train.device
    seed_everything(args.seed, device)
    model = make_mlp(num_classes, device, args.seed)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.MLP_LEARNING_RATE,
        weight_decay=cfg.MLP_WEIGHT_DECAY,
    )
    sync(device)
    started = time.perf_counter()
    for _ in range(args.offline_epochs):
        train_mlp_epoch(
            model, optimizer, X_train, y_train, cfg.MLP_BATCH_SIZE
        )
    sync(device)
    training_s = time.perf_counter() - started
    prediction = predict_mlp(model, X_test)
    accuracy = float(
        accuracy_score(y_test.cpu().numpy(), prediction) * 100.0
    )
    inference = benchmark_predict(
        lambda: predict_mlp(model, X_test),
        len(X_test),
        device,
        args.inference_warmups,
        args.inference_repeats,
    )
    return {
        "accuracy": accuracy,
        "training_s": training_s,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        **inference,
    }


def offline_ncmml(X_train, y_train, X_test, y_test, num_classes, dim, args):
    means = compute_means(X_train, y_train, num_classes)
    W, training_s = train_ncmml(
        X_train,
        y_train,
        means,
        dim,
        args.offline_epochs,
        cfg.LEARNING_RATE,
        cfg.BATCH_SIZE,
        args.seed,
    )
    accuracy = evaluate_scores(ncm_scores(X_test, means, W), y_test)
    inference = benchmark_predict(
        lambda: ncm_scores(X_test, means, W).argmax(dim=1).cpu().numpy(),
        len(X_test),
        X_train.device,
        args.inference_warmups,
        args.inference_repeats,
    )
    return {
        "accuracy": accuracy,
        "training_s": training_s,
        "trainable_parameters": int(W.numel()),
        **inference,
    }


def online_mlp(X_train, y_train, X_test, y_test, num_classes, args):
    device = X_train.device
    seed_everything(args.seed, device)
    model = make_mlp(num_classes, device, args.seed)
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.MLP_LEARNING_RATE,
        weight_decay=cfg.MLP_WEIGHT_DECAY,
    )
    prototypes = [[] for _ in range(num_classes)]
    counts = [[] for _ in range(num_classes)]
    sync(device)
    started = time.perf_counter()
    for x, y in zip(X_train, y_train):
        class_id = int(y.item())
        update_exstream_lists(
            prototypes, counts, x, class_id, args.capacity
        )
        replay_X, replay_y = [], []
        for c in range(num_classes):
            if prototypes[c]:
                replay_X.extend(prototypes[c])
                replay_y.extend([c] * len(prototypes[c]))
        if len(replay_y) > 1:
            train_mlp_epoch(
                model,
                optimizer,
                torch.stack(replay_X),
                torch.as_tensor(replay_y, device=device),
                cfg.MLP_BATCH_SIZE,
            )
    sync(device)
    training_s = time.perf_counter() - started
    prediction = predict_mlp(model, X_test)
    accuracy = float(
        accuracy_score(y_test.cpu().numpy(), prediction) * 100.0
    )
    inference = benchmark_predict(
        lambda: predict_mlp(model, X_test),
        len(X_test),
        device,
        args.inference_warmups,
        args.inference_repeats,
    )
    return {
        "accuracy": accuracy,
        "training_s": training_s,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        **inference,
    }


def table1(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.cub_data, "cub200", 200, device
    )
    offline_mlp_result = offline_mlp(
        X_train, y_train, X_test, y_test, 200, args
    )
    offline_ncmml_result = offline_ncmml(
        X_train, y_train, X_test, y_test, 200, cfg.PROJECTION_DIM, args
    )
    online_mlp_result = online_mlp(
        X_train, y_train, X_test, y_test, 200, args
    )
    hybrid = run_hybrid_exstream_ncmml(
        X_train,
        y_train,
        X_test,
        y_test,
        200,
        cfg.PROJECTION_DIM,
        args.capacity,
        args.epochs_per_update,
        cfg.LEARNING_RATE,
        cfg.BATCH_SIZE,
        args.seed,
        warm_start=True,
    )
    projected_means = hybrid.means @ hybrid.final_W.T
    projected_norms = (projected_means * projected_means).sum(dim=1)
    hybrid_inference = benchmark_predict(
        lambda: (
            (X_test @ hybrid.final_W.T) @ projected_means.T
            - 0.5 * projected_norms
        ).argmax(dim=1).cpu().numpy(),
        len(X_test),
        device,
        args.inference_warmups,
        args.inference_repeats,
    )
    online_ncmml_result = {
        "accuracy": hybrid.final_accuracy,
        "training_s": hybrid.training_s,
        "trainable_parameters": hybrid.trainable_parameters,
        **hybrid_inference,
    }
    rows = []
    for name, result, baseline in (
        ("Offline-MLP", offline_mlp_result, offline_mlp_result),
        ("Offline-NCMML", offline_ncmml_result, offline_mlp_result),
        ("ExStream-MLP", online_mlp_result, online_mlp_result),
        ("ExStream-NCMML", online_ncmml_result, online_mlp_result),
    ):
        efficiency = result["accuracy"] / result["training_s"]
        baseline_efficiency = baseline["accuracy"] / baseline["training_s"]
        throughput = result.get("throughput_samples_per_s")
        baseline_throughput = baseline.get("throughput_samples_per_s")
        rows.append(
            {
                "method": name,
                "trainable_parameters": result["trainable_parameters"],
                "gpu_training_time_s": result["training_s"],
                "final_accuracy": result["accuracy"],
                "training_efficiency_x": efficiency / baseline_efficiency,
                "inference_throughput_x": (
                    throughput / baseline_throughput
                    if throughput is not None and baseline_throughput else ""
                ),
            }
        )
    write_payload(
        args.output / "table1",
        "efficiency_comparison",
        {
            "protocol": {
                "projection_dim": cfg.PROJECTION_DIM,
                "capacity": args.capacity,
                "offline_epochs": args.offline_epochs,
                "epochs_per_update": args.epochs_per_update,
            },
            "rows": rows,
        },
        rows,
    )


def table2(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.cub_data, "cub200", 200, device
    )
    means = compute_means(X_train, y_train, 200)
    raw = evaluate_scores(ncm_scores(X_test, means), y_test)
    offline = offline_ncmml(
        X_train, y_train, X_test, y_test, 200, cfg.PROJECTION_DIM, args
    )
    hybrid = run_hybrid_exstream_ncmml(
        X_train,
        y_train,
        X_test,
        y_test,
        200,
        cfg.PROJECTION_DIM,
        args.capacity,
        args.epochs_per_update,
        cfg.LEARNING_RATE,
        cfg.BATCH_SIZE,
        args.seed,
    )
    rows = [
        {
            "method": "Raw NCM",
            "stored_data": "Class means",
            "metric_learning": "No",
            "incremental": "Yes",
            "final_accuracy": raw,
        },
        {
            "method": "Offline NCMML",
            "stored_data": "All samples",
            "metric_learning": "Yes",
            "incremental": "No",
            "final_accuracy": offline["accuracy"],
        },
        {
            "method": "ExStream-NCMML",
            "stored_data": "Compressed prototypes",
            "metric_learning": "Yes",
            "incremental": "Yes",
            "final_accuracy": hybrid.final_accuracy,
        },
    ]
    write_payload(
        args.output / "table2",
        "distance_method_accuracy",
        {"projection_dim": cfg.PROJECTION_DIM, "rows": rows},
        rows,
    )


def table3(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.cub_data, "cub200", 200, device
    )
    rows = []
    original_epochs = args.offline_epochs
    args.offline_epochs = args.dimension_offline_epochs
    for dim in cfg.DIMENSION_SWEEP:
        offline = offline_ncmml(
            X_train, y_train, X_test, y_test, 200, dim, args
        )
        hybrid = run_hybrid_exstream_ncmml(
            X_train,
            y_train,
            X_test,
            y_test,
            200,
            dim,
            args.capacity,
            args.epochs_per_update,
            cfg.LEARNING_RATE,
            cfg.BATCH_SIZE,
            args.seed,
        )
        rows.append(
            {
                "projection_dim": dim,
                "offline_accuracy": offline["accuracy"],
                "exstream_accuracy": hybrid.final_accuracy,
            }
        )
    args.offline_epochs = original_epochs
    write_payload(
        args.output / "table3",
        "projection_dimension",
        {"rows": rows},
        rows,
    )


def capacity(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.cub_data, "cub200", 200, device
    )
    rows = []
    curves = {}
    for cap in cfg.CAPACITY_SWEEP:
        result = run_hybrid_exstream_ncmml(
            X_train,
            y_train,
            X_test,
            y_test,
            200,
            cfg.PROJECTION_DIM,
            cap,
            args.epochs_per_update,
            cfg.LEARNING_RATE,
            cfg.BATCH_SIZE,
            args.seed,
        )
        rows.append(
            {
                "capacity": cap,
                "final_accuracy": result.final_accuracy,
                "training_s": result.training_s,
            }
        )
        curves[str(cap)] = result.accuracy_curve
    write_payload(
        args.output / "fig5_capacity",
        "capacity_study",
        {"seen_classes": list(range(1, 201)), "curves": curves, "rows": rows},
        rows,
    )


def warm_start(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.cub_data, "cub200", 200, device
    )
    rows = []
    curves = {}
    for seed in args.warm_seeds:
        for strategy, warm in (("Cold-start", False), ("Warm-start", True)):
            result = run_hybrid_exstream_ncmml(
                X_train,
                y_train,
                X_test,
                y_test,
                200,
                cfg.PROJECTION_DIM,
                args.capacity,
                args.warm_epochs,
                cfg.LEARNING_RATE,
                cfg.BATCH_SIZE,
                seed,
                warm_start=warm,
            )
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "final_accuracy": result.final_accuracy,
                    "average_accuracy": float(np.mean(result.accuracy_curve)),
                    "gpu_training_time_s": result.w_update_s,
                }
            )
            curves["%s_%d" % (strategy, seed)] = result.accuracy_curve
    write_payload(
        args.output / "warm_start",
        "warm_start_ablation",
        {"seen_classes": list(range(1, 201)), "curves": curves, "rows": rows},
        rows,
    )


def aircraft(args):
    device = device_for(args.gpu_id)
    X_train, y_train, X_test, y_test = tensors(
        args.aircraft_data, "fgvc_aircraft48", 48, device
    )
    means = compute_means(X_train, y_train, 48)
    raw = evaluate_scores(ncm_scores(X_test, means), y_test)
    old_epochs = args.offline_epochs
    args.offline_epochs = args.aircraft_epochs
    metric = offline_ncmml(
        X_train, y_train, X_test, y_test, 48, 256, args
    )
    args.offline_epochs = old_epochs
    rows = [
        {"method": "Raw NCM", "final_accuracy": raw},
        {"method": "NCMML (D=256)", "final_accuracy": metric["accuracy"]},
    ]
    write_payload(
        args.output / "aircraft",
        "fgvc_aircraft48_reference",
        {"rows": rows},
        rows,
    )


COMMANDS = {
    "table1": table1,
    "table2": table2,
    "table3": table3,
    "capacity": capacity,
    "warm_start": warm_start,
    "aircraft": aircraft,
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=sorted(COMMANDS))
    parser.add_argument("--cub-data", type=Path, default=cfg.DATA_ROOT / "cub200")
    parser.add_argument(
        "--aircraft-data", type=Path, default=cfg.DATA_ROOT / "fgvc_aircraft48"
    )
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--capacity", type=int, default=cfg.BUFFER_CAPACITY)
    parser.add_argument("--offline-epochs", type=int, default=cfg.OFFLINE_EPOCHS)
    parser.add_argument(
        "--dimension-offline-epochs",
        type=int,
        default=cfg.DIMENSION_SWEEP_OFFLINE_EPOCHS,
    )
    parser.add_argument(
        "--epochs-per-update", type=int, default=cfg.EPOCHS_PER_UPDATE
    )
    parser.add_argument(
        "--warm-epochs", type=int, default=cfg.WARM_START_ABLATION_EPOCHS
    )
    parser.add_argument("--warm-seeds", type=int, nargs="+", default=[111, 222, 333])
    parser.add_argument("--aircraft-epochs", type=int, default=200)
    parser.add_argument(
        "--inference-warmups", type=int, default=cfg.INFERENCE_WARMUPS
    )
    parser.add_argument(
        "--inference-repeats", type=int, default=cfg.INFERENCE_REPEATS
    )
    return parser


def main():
    args = build_parser().parse_args()
    COMMANDS[args.experiment](args)


if __name__ == "__main__":
    main()
