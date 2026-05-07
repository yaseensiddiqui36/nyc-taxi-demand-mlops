"""
Root conftest.py — loaded by pytest before any test module is imported.
Inserts the project root into sys.path so that 'from src.xxx import ...'
resolves correctly in CI without needing PYTHONPATH set externally.
"""
import sys
from pathlib import Path

# Ensure project root is on the import path
sys.path.insert(0, str(Path(__file__).parent))
