"""Allow `python -m orchestrator.evaluation ...` invocation."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
