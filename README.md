# ExStream-NCMML paper reproduction

This directory is the standalone source package for:

> A Lightweight Memory-Efficient Framework for Distance-Based
> Class-Incremental Learning

It contains only the algorithms, experiment entry points, figure generators
and documentation needed by the manuscript. Datasets and historical result
directories are not bundled.

## 1. Environment

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```bash
python -m pip install -r requirements.txt
```

All experiments fall back to CPU, but the 200-class incremental sweeps are
substantially slower without a GPU.

## 2. Data

Follow [data/README.md](data/README.md). The default layout is:

```text
data/
├── cub200/
│   ├── cub200_resnet50_train.mat
│   └── cub200_resnet50_test.mat
└── fgvc_aircraft48/
    ├── fgvc_aircraft48_resnet50_train.mat
    └── fgvc_aircraft48_resnet50_test.mat
```

The package uses frozen 2048-dimensional ImageNet-pretrained ResNet-50
features. It does not train the feature extractor.

## 3. Reproduction protocol

The shared CUB-200 protocol is defined in
`configs/paper_config.py`:

- class-IID stream over 200 classes;
- per-class ExStream buffer capacity 8;
- count-weighted cross entropy;
- warm-started NCMML;
- projection dimension `D=199` outside the dimension ablation;
- Adam, learning rate `1e-3`, mini-batch size 128;
- seed 111 unless a multi-seed ablation is requested.

The D=199 release correction is explained in
[EXPECTED_RESULTS.md](EXPECTED_RESULTS.md).

## 4. Paper table and ablation commands

Run commands from this directory.

### Table I: cost, accuracy and throughput

```bash
python experiments/table1_efficiency.py
```

Outputs:

```text
outputs/table1/efficiency_comparison.csv
outputs/table1/efficiency_comparison.json
```

### Table II: Raw NCM, Offline NCMML and ExStream-NCMML

```bash
python experiments/table2_accuracy.py
```

### Table III: projection dimension

```bash
python experiments/table3_dimension.py
```

The exact dimensions are `16, 32, 64, 199, 256, 512, 1024, 2048`.

### Fig. 5: ExStream capacity

```bash
python experiments/fig5_capacity.py
python figures/paper_figures.py fig5
```

Capacities 8, 12 and 16 are compressed settings. Capacity 30 is the
full-buffer reference for CUB-200.

### Warm-start ablation

```bash
python experiments/warm_start_ablation.py
```

This runs paired cold/warm variants with seeds 111, 222 and 333 and 30 epochs
per class update.

### FGVC-Aircraft48 text result

```bash
python experiments/fgvc_aircraft_reference.py
```

## 5. Method figures

Fig. 1, Fig. 2 and Fig. 4 do not require CUB data:

```bash
python figures/paper_figures.py fig1
python figures/paper_figures.py fig2
python figures/paper_figures.py fig4
```

After running the capacity experiment, all four can be generated with:

```bash
python figures/paper_figures.py all
```

## 6. Convenience launcher

Examples:

```bash
python run_all.py table1 table2
python run_all.py table3 capacity figures
python run_all.py warm_start aircraft
python run_all.py all
```

Existing JSON outputs are skipped by default. Add `--force` to rerun them.
Use `--gpu-id`, `--cub-data` and `--aircraft-data` to override defaults.

## 7. Metrics

Final accuracy:

```text
correct test predictions / number of test samples
```

Training efficiency:

```text
final accuracy / synchronized GPU training time
```

Inference throughput:

```text
(number of test samples × formal repeats) /
sum of synchronized formal inference times
```

Warm-up inference runs are excluded. Offline and online training times have
different semantics: Offline is one final full-data fit, whereas Online is
the cumulative cost of all stream updates. They should not be interpreted as
the same training schedule.

## 8. Source layout

```text
configs/paper_config.py       shared protocol
src/core.py                   NCM, NCMML, ExStream, Hybrid and MLP
experiments/                  numerical table/ablation entry points
figures/paper_figures.py      Fig. 1/2/4/5 generation
run_all.py                    grouped launcher
outputs/                      generated at runtime
```

The package has no Python import dependency on files outside this directory.

## 9. Quick validation

The data-free smoke test checks prototype counts and mean preservation,
incremental metric updates, and MLP parameter accounting:

```bash
python tests/smoke_test.py
```

This test does not reproduce the full numerical experiments; those require
the feature files described in [data/README.md](data/README.md).

## 10. Acknowledgments

This package builds on the original
[ExStream implementation](https://github.com/tyler-hayes/ExStream).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and the
preserved upstream license notice.
