# Paper-to-code map

| Paper item | Command | Primary output |
|---|---|---|
| Table I | `python experiments/table1_efficiency.py` | `outputs/table1/efficiency_comparison.csv` |
| Table II | `python experiments/table2_accuracy.py` | `outputs/table2/distance_method_accuracy.csv` |
| Table III | `python experiments/table3_dimension.py` | `outputs/table3/projection_dimension.csv` |
| Fig. 1 | `python figures/paper_figures.py fig1` | `outputs/figures/fig1_exstream_mechanism.png` |
| Fig. 2 | `python figures/paper_figures.py fig2` | `outputs/figures/fig2_ncmml_projection.png` |
| Fig. 4 | `python figures/paper_figures.py fig4` | `outputs/figures/fig4_training_framework.svg` |
| Fig. 5 data | `python experiments/fig5_capacity.py` | `outputs/fig5_capacity/capacity_study.json` |
| Fig. 5 image | `python figures/paper_figures.py fig5` | `outputs/figures/fig5_capacity.png` |
| Warm-start ablation | `python experiments/warm_start_ablation.py` | `outputs/warm_start/warm_start_ablation.csv` |
| Aircraft text result | `python experiments/fgvc_aircraft_reference.py` | `outputs/aircraft/fgvc_aircraft48_reference.csv` |

## Provenance

The standalone implementation was consolidated from these development files:

- `utils.py`
- `CLU-ALgorithmtest/stream_buffers.py`
- `VoiceGuidedReplay/grouped_mlp_experiment/baseline_model.py`
- `VoiceGuidedReplay/ncm_experiment/ncmml_model.py`
- `VoiceGuidedReplay/ncm_experiment/streaming_ncmml_hybrid_exstream_model.py`
- `VoiceGuidedReplay/ncm_experiment/run_offline_ncmml_vs_mlp_cost.py`
- `VoiceGuidedReplay/ncm_experiment/run_online_exstream_ncmml_vs_single_mlp_cost.py`
- `VoiceGuidedReplay/ncm_experiment/run_ncmml_d_offline_vs_incremental_exstream_cub200.py`
- `VoiceGuidedReplay/ncm_experiment/run_ncmml_cub200_capacity_study.py`
- `VoiceGuidedReplay/ncm_experiment/run_ncmml_cub200_warm_start_ablation.py`
- `VoiceGuidedReplay/ncm_experiment/plot_exstream_mechanism_demo.py`
- `VoiceGuidedReplay/ncm_experiment/plot_ncmml_dimensionality_reduction_demo.py`
- `VoiceGuidedReplay/ncm_experiment/plot_training_structure_diagram.py`

Video code, Grouped-MLP architecture exploration, negative-sample experiments,
old result caches and machine-specific absolute paths are intentionally
excluded.
