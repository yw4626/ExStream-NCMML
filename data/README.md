# Data placement

The feature files are intentionally not bundled with this source package.

Place the CUB-200 ResNet-50 features here:

```text
data/cub200/
├── cub200_resnet50_train.mat
└── cub200_resnet50_test.mat
```

Place the optional FGVC-Aircraft48 features here:

```text
data/fgvc_aircraft48/
├── fgvc_aircraft48_resnet50_train.mat
└── fgvc_aircraft48_resnet50_test.mat
```

Every MAT file must contain:

- `X`: an `N × 2048` array of frozen ResNet-50 embeddings.
- `y`: integer labels starting at zero.

The loaders apply per-sample L2 normalization. CUB-200 experiments use the
official 5,994-image train split and 5,794-image test split. The Aircraft
reference uses 48 variants with 30 train and 30 test images per class.

Alternative locations can be supplied with `--cub-data` and
`--aircraft-data`.
