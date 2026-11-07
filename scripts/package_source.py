"""Package only source code and docs. Never bundle competition/customer data."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
with zipfile.ZipFile(ROOT / 'forcecrm-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for folder in ['forcecrm', 'scripts', 'web', 'tests', 'docs', 'notebooks']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc']:
                archive.write(path, path.relative_to(ROOT))
    for name in ['run.py', 'requirements.txt', 'requirements-ai.txt', 'README.md', '.gitignore']:
        archive.write(ROOT / name, name)
print(ROOT / 'forcecrm-source.zip')
