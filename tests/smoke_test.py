"""Fast, data-free validation of the standalone reproduction package."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import MLP, compress_exstream, run_hybrid_exstream_ncmml
from configs import paper_config as cfg


def main():
    assert cfg.PROJECTION_DIM == 199
    assert cfg.PROJECTION_DIM * cfg.INPUT_DIM == 407552
    torch.manual_seed(7)
    X = torch.randn(30, 16)
    prototypes, counts = compress_exstream(X, capacity=8)
    assert len(prototypes) == 8
    assert int(counts.sum().item()) == 30
    reconstructed = (
        prototypes * counts.unsqueeze(1)
    ).sum(dim=0) / counts.sum()
    assert torch.allclose(reconstructed, X.mean(dim=0), atol=1e-6)

    y = torch.arange(3).repeat_interleave(4)
    train = torch.randn(12, 8) + y.float().unsqueeze(1)
    result = run_hybrid_exstream_ncmml(
        train, y, train, y,
        num_classes=3,
        proj_dim=2,
        capacity=2,
        epochs_per_update=1,
        learning_rate=1e-3,
        batch_size=8,
        seed=111,
    )
    assert result.w_updates == 2
    assert result.trainable_parameters == 16
    assert int(result.prototype_mass) == 12
    assert len(result.accuracy_curve) == 3

    mlp = MLP(
        [2048, 350, 300, 200],
        seed=111,
        batch_norm=True,
        dropout=0.25,
        activation="elu",
    )
    assert sum(parameter.numel() for parameter in mlp.parameters()) == 883950
    print("Smoke test passed")


if __name__ == "__main__":
    main()
