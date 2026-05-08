"""
Root conftest.py — loaded by pytest before any test module is imported.

Multiple layers of path fixing are needed because feast and other packages
can create a 'src' namespace package that shadows our local src/ directory.
"""
import importlib
import sys
from pathlib import Path

ROOT = str(Path(__file__).parent)

# 1. Put project root first in sys.path
while ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

# 2. Evict any stale 'src' namespace already cached in sys.modules
#    (feast or other packages can pre-register a src namespace package)
for _key in list(sys.modules.keys()):
    if _key == "src" or _key.startswith("src."):
        del sys.modules[_key]

# 3. Force the import machinery to re-scan for packages
importlib.invalidate_caches()
