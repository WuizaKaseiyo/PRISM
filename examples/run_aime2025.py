#!/usr/bin/env python3
"""Convenience entry point — delegates to evaluate.aime2025.__main__."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate.aime2025.__main__ import main

if __name__ == "__main__":
    main()
