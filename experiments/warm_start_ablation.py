"""Run the cold-start versus warm-start ablation."""
import sys
from paper_experiments import main

if __name__ == "__main__":
    sys.argv.insert(1, "warm_start")
    main()
