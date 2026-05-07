"""
Run `feast apply` with environment variables loaded from .env
Usage: python scripts/feast_apply.py
"""
import os
import subprocess
import sys
from pathlib import Path

# Load .env from project root
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Use the feast CLI from the same venv as the running Python
venv_scripts = Path(sys.executable).parent
feast_exe = venv_scripts / "feast.exe" if sys.platform == "win32" else venv_scripts / "feast"

feature_repo = Path(__file__).parent.parent / "feature_repo"
result = subprocess.run([str(feast_exe), "apply"], cwd=feature_repo)
exit(result.returncode)
