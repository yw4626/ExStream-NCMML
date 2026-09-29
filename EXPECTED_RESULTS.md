# Expected results and protocol correction

The submitted manuscript reports the following reference values:

- Raw NCM on CUB-200: about 58.61%.
- Offline NCMML: about 67.28%.
- ExStream-NCMML: 61.46% at `D=199` and 61.81% at `D=256`.
- ExStream-MLP: about 54.02%.
- Warm-start ablation: 59.90% final / 66.05% average accuracy.
- Cold-start ablation: 49.63% final / 58.57% average accuracy.
- FGVC-Aircraft48: Raw NCM 29.10%, NCMML (`D=256`) 42.50%.

## Important Table I correction

The current manuscript combines values produced under different projection
dimensions:

- `65,536` NCMML parameters corresponds to `D=32` (`32 × 2048`).
- The 61.46% ExStream-NCMML accuracy corresponds to `D=199`.
- The 67.28% Offline-NCMML value was measured in the high-dimensional sweep.

This package uses the internally consistent protocol selected for release:

```text
D = 199
NCMML trainable parameters = 199 × 2048 = 407,552
```

Consequently, regenerated training times, throughput and final accuracy should
be taken from the new output files rather than copied verbatim from the current
Table I. Small timing variation is expected across GPUs, drivers and PyTorch
versions.

## Dimension sweep reference

| D | Offline NCMML | ExStream-NCMML |
|---:|---:|---:|
| 16 | 61.82 | 54.31 |
| 32 | 66.52 | 58.39 |
| 64 | 67.35 | 60.01 |
| 199 | 67.28 | 61.46 |
| 256 | 67.28 | 61.81 |
| 512 | 67.16 | 62.20 |
| 1024 | 67.40 | 62.43 |
| 2048 | 67.41 | 62.65 |

Exact floating-point results can vary slightly because CUDA kernels and timing
are platform dependent.
