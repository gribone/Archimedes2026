"""One-command local bootstrap and launch: python run.py [--demo]."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--setup-models', action='store_true', help='Download the models and exit')
    mode.add_argument('--first-run', action='store_true', help='Provision models online and launch the local app')
    mode.add_argument('--demo', action='store_true', help='Install only lightweight demo dependencies')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    os.chdir(ROOT)
    env = ROOT / '.venv'
    executable = env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not executable.exists():
        subprocess.check_call([sys.executable, '-m', 'venv', str(env)])
    requirements = ROOT / ('requirements.txt' if args.demo else 'requirements-ai.txt')
    marker = env / ('demo-installed' if args.demo else 'ai-installed')
    signature = requirements.read_text() + (ROOT / 'requirements.txt').read_text()
    if not marker.exists() or marker.read_text() != signature:
        subprocess.check_call([str(executable), '-m', 'pip', 'install', '-r', str(requirements)])
        marker.write_text(signature)
    portable = ROOT / '.tools' / 'ollama' / 'ollama.exe'
    ollama = shutil.which('ollama') or (str(portable) if portable.exists() else None)
    os.environ.update(OLLAMA_HOST='127.0.0.1:11434', OLLAMA_NO_CLOUD='1')
    if portable.exists() and ollama == str(portable):
        os.environ['OLLAMA_MODELS'] = str(ROOT / 'models' / 'ollama')
    ollama_process = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def ollama_ready():
        try:
            with opener.open('http://127.0.0.1:11434/api/tags', timeout=2) as response:
                return response.status == 200
        except OSError:
            return False
    if not args.demo and ollama and not ollama_ready():
        log_dir = ROOT / '.cache'
        log_dir.mkdir(exist_ok=True)
        with (log_dir / 'ollama.log').open('a', encoding='utf-8') as log:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            ollama_process = subprocess.Popen([ollama, 'serve'], stdout=log, stderr=log, creationflags=flags)
        for _ in range(30):
            if ollama_ready():
                break
            if ollama_process.poll() is not None:
                raise SystemExit('Ollama startup failed. See .cache/ollama.log')
            time.sleep(1)
    if args.setup_models or args.first_run:
        if ollama is None:
            raise SystemExit('Install Ollama from https://ollama.com/download, then run: python run.py --first-run')
        try:
            subprocess.check_call([str(executable), '-m', 'scripts.download_models'])
            subprocess.check_call([ollama, 'pull', 'ministral-3:3b'])
        except BaseException:
            if ollama_process:
                ollama_process.terminate()
            raise
        if args.setup_models:
            if ollama_process:
                ollama_process.terminate()
            print('Models provisioned. Start with: python run.py')
            return
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      OLLAMA_NO_CLOUD='1', TOKENIZERS_PARALLELISM='false')
    print(f'ForceCRM: http://127.0.0.1:{args.port}', flush=True)
    try:
        subprocess.check_call([str(executable), '-X', 'utf8', '-m', 'uvicorn', 'forcecrm.server:app', '--host', '127.0.0.1', '--port', str(args.port)])
    finally:
        if ollama_process:
            ollama_process.terminate()


if __name__ == '__main__':
    main()
