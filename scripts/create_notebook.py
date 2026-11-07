"""Rebuild the portable Colab notebook from maintained cells."""
import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'outputs': [], 'metadata': {}, 'source': text.splitlines(keepends=True)})


markdown('''# ForceCRM — temporary Colab experimentation

This notebook runs the same Python analysis pipeline as the final localhost app. **Colab is cloud computing, not compliant local-laptop execution.** Only upload competition/customer data you are authorized to transfer to Google. No external AI inference API is used.

Use Runtime → Change runtime type → T4 GPU if available. Work in notebook cells; no public tunnels are created. Download outputs before the runtime expires.
''')
code('''from pathlib import Path
import os, io, zipfile, subprocess, sys, time, json
from google.colab import files

DEMO = False  # @param {type:"boolean"}

uploaded = files.upload()  # Upload forcecrm-source.zip (or your source ZIP).
zip_name = next((name for name in uploaded if name.lower().endswith('.zip')), None)
if zip_name is None:
    raise ValueError('Upload the project ZIP first.')
project_root = Path('/content/forcecrm')
project_root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(uploaded[zip_name])) as archive:
    # Prevent ZIP path traversal and excessive decompression.
    if sum(item.file_size for item in archive.infolist()) > 500 * 1024**2:
        raise ValueError('Use a source-only ZIP smaller than 500 MB unpacked.')
    for item in archive.infolist():
        destination = (project_root / item.filename).resolve()
        if not destination.is_relative_to(project_root.resolve()):
            raise ValueError('Unsafe path inside ZIP.')
    archive.extractall(project_root)
locations = list(project_root.rglob('requirements-ai.txt'))
if len(locations) != 1:
    raise ValueError('ZIP must contain exactly one ForceCRM project.')
os.chdir(locations[0].parent)
sys.path.insert(0, str(Path.cwd()))
print('Project:', Path.cwd())
''')
code('''required = ['vwHackCompany.csv', 'vwHackCompanyActivity.csv', 'vwHackActivities.csv']
missing = [name for name in required if not Path(name).exists()]
if missing:
    print('Upload these CSVs:', missing)
    source_files = files.upload()
    for name in required:
        if name in source_files:
            Path(name).write_bytes(source_files[name])
assert all(Path(name).exists() for name in required), 'Missing one or more CSV files.'
if not DEMO and not list(Path('.').glob('*.xlsx')):
    print('Upload the customer workbook (.xlsx).')
    portfolio_uploads = files.upload()
    for name, data in portfolio_uploads.items():
        if name.lower().endswith('.xlsx'):
            Path(Path(name).name).write_bytes(data)
''')
markdown('''## Install dependencies and provision models

This stage needs internet and several GB of temporary disk. Ollama is installed from its official distribution. The embedding revision is recorded. If Colab requests a restart because of dependency versions, restart the session and rerun the import/project-directory cell without reinstalling packages, then continue.
''')
code('''requirements = 'requirements.txt' if DEMO else 'requirements-ai.txt'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements], check=True)
''')
provision = '''import urllib.request
installer = Path('/tmp/install-ollama.sh')
urllib.request.urlretrieve('https://ollama.com/install.sh', installer)
subprocess.run(['sh', str(installer)], check=True)
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'
os.environ['OLLAMA_NO_CLOUD'] = '1'
ollama_log = open('/tmp/forcecrm-ollama.log', 'a')
ollama_process = subprocess.Popen(['ollama', 'serve'], stdout=ollama_log, stderr=subprocess.STDOUT)
import requests
for attempt in range(60):
    try:
        if requests.get('http://127.0.0.1:11434/api/tags', timeout=2).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(1)
else:
    raise RuntimeError(Path('/tmp/forcecrm-ollama.log').read_text()[-4000:])
subprocess.run(['ollama', 'pull', 'ministral-3:3b'], check=True)
subprocess.run([sys.executable, '-m', 'scripts.download_models'], check=True)
'''
from textwrap import indent
code('if not DEMO:\n' + indent(provision, '    '))
markdown('''## Analyze

Set `DEMO = False` for the uploaded portfolio with semantic embeddings and local-to-VM Ministral. Demo uses synthetic customer selection/revenue and lexical features; it does not measure real performance. All existing customer venues are excluded from recommendations by VAT.
''')
code('''portfolio_files = sorted(Path('.').glob('*.xlsx'))
PORTFOLIO = portfolio_files[0] if portfolio_files else None  # Change if multiple files exist.
command = [sys.executable, '-m', 'scripts.analyze', '--output', 'outputs/colab']
if DEMO:
    command.append('--demo')
else:
    if PORTFOLIO is None:
        raise ValueError('Upload your .xlsx portfolio before setting DEMO=False.')
    command.extend(['--portfolio', str(PORTFOLIO)])
subprocess.run(command, check=True)
''')
code('''import pandas as pd
from IPython.display import display, Markdown
result_dir = Path('outputs/colab')
profiles = json.loads((result_dir / 'profiles.json').read_text('utf-8'))
report = json.loads((result_dir / 'report.json').read_text('utf-8'))
display(pd.DataFrame([{key: p[key] for key in ['profile_id', 'title', 'customer_count', 'revenue_share', 'stability', 'confidence_score']} for p in profiles]))
for profile in profiles:
    print(profile['profile_id'], profile['description_status'])
    print(profile['description'])
display(pd.DataFrame(report['validation'].get('metrics', [])))
display(pd.read_csv(result_dir / 'lookalikes.csv', dtype={'company_id': str}).head(20))
''')
code('''result_zip = Path('forcecrm-colab-results.zip')
with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in result_dir.iterdir():
        if path.suffix in ['.json', '.csv']:
            archive.write(path, path.name)
files.download(str(result_zip))
''')
markdown('''## Return to the final local application

On your laptop, install Python 3.12 and Ollama, then run `python run.py --setup-models` once and `python run.py`. Open http://127.0.0.1:8000. The final app runs without Colab or an external inference API.

Sources: GEMI Open Data; Geocoding powered by OpenStreetMap contributors (ODbL). Competition-only data. OpenAI Codex assisted development, documentation and testing.
''')
for index, cell in enumerate(cells, 1):
    cell['id'] = f'forcecrm-{index:02d}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}, 'colab': {'name': 'ForceCRM_Colab.ipynb'}},
            'nbformat': 4, 'nbformat_minor': 5}
path = Path(__file__).resolve().parents[1] / 'notebooks' / 'ForceCRM_Colab.ipynb'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), 'utf-8')
print(path)
