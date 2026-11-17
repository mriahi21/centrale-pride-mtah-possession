"""Possession labelling for MTAH 2027 Phase 1.

Importing this package puts the repository root on sys.path so that the vendored upstream modules
(`games`, `starter.*`, `evaluation.*`) are importable from anywhere.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
