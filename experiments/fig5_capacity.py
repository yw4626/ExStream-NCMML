"""Run the capacity study used by Fig. 5."""
import sys
from paper_experiments import main

if __name__ == "__main__":
    sys.argv.insert(1, "capacity")
    main()
