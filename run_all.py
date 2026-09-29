"""Convenience launcher for the paper reproduction package."""
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "table1": ("table1", ROOT / "outputs/table1/efficiency_comparison.json"),
    "table2": ("table2", ROOT / "outputs/table2/distance_method_accuracy.json"),
    "table3": ("table3", ROOT / "outputs/table3/projection_dimension.json"),
    "capacity": ("capacity", ROOT / "outputs/fig5_capacity/capacity_study.json"),
    "warm_start": (
        "warm_start", ROOT / "outputs/warm_start/warm_start_ablation.json"),
    "aircraft": (
        "aircraft", ROOT / "outputs/aircraft/fgvc_aircraft48_reference.json"),
}


def run(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets", nargs="+",
        choices=[*EXPERIMENTS, "figures", "all"])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cub-data", type=Path, default=ROOT / "data/cub200")
    parser.add_argument(
        "--aircraft-data", type=Path, default=ROOT / "data/fgvc_aircraft48")
    parser.add_argument(
        "--force", action="store_true",
        help="rerun experiments even when their JSON output exists")
    args = parser.parse_args()

    targets = list(EXPERIMENTS) + ["figures"] if "all" in args.targets else args.targets
    for target in targets:
        if target == "figures":
            run([
                sys.executable, ROOT / "figures/paper_figures.py", "all"])
            continue
        experiment, result_path = EXPERIMENTS[target]
        if result_path.exists() and not args.force:
            print("Skipping existing", result_path)
            continue
        run([
            sys.executable,
            ROOT / "experiments/paper_experiments.py",
            experiment,
            "--gpu-id", args.gpu_id,
            "--cub-data", args.cub_data,
            "--aircraft-data", args.aircraft_data,
        ])


if __name__ == "__main__":
    main()
