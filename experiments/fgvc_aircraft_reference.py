"""Reproduce the FGVC-Aircraft48 Raw NCM and NCMML text result."""
import sys
from paper_experiments import main

if __name__ == "__main__":
    sys.argv.insert(1, "aircraft")
    main()
