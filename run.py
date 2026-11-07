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


def default_llm():
    """The shipped Ollama model, read from forcecrm/narrative.py without importing dependencies."""
    import re
    match = re.search(r"^DEFAULT_LLM = '([^']+)'", (ROOT / 'forcecrm' / 'narrative.py').read_text('utf-8'), re.M)
    return match.group(1)


def supported(python):
    """True if this interpreter is Python 3.12 or 3.13 (the versions the pinned wheels support)."""
    try:
        out = subprocess.run([python, '-c', 'import sys; print(sys.version_info[0], sys.version_info[1])'],
                             capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return False
    return len(out) == 2 and (3, 12) <= (int(out[0]), int(out[1])) < (3, 14)


def create_venv(env):
    """Create .venv with a supported Python, even when run.py is launched by e.g. Python 3.14."""
    if (3, 12) <= sys.version_info[:2] < (3, 14):
        subprocess.check_call([sys.executable, '-m', 'venv', str(env)])
        return
    for name in ('python3.12', 'python3.13'):
        found = shutil.which(name)
        if found and supported(found):
            print(f'Python {sys.version.split()[0]} is not supported by the pinned packages; using {found}.', flush=True)
            subprocess.check_call([found, '-m', 'venv', str(env)])
            return
    uv = shutil.which('uv')
    if uv:  # uv downloads a managed Python 3.12 when none is installed
        print('Creating .venv with a uv-managed Python 3.12.', flush=True)
        subprocess.check_call([uv, 'venv', '--python', '3.12', str(env)])
        return
    raise SystemExit(f'Python 3.12 or 3.13 is required (found {sys.version.split()[0]}). '
                     'Install Python 3.12 and run: python3.12 run.py --first-run')


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
        create_venv(env)
    # The pinned wheels need Python 3.12 or 3.13 *inside the venv*; the launcher itself can be any version.
    if not supported(str(executable)):
        raise SystemExit(f'{env} uses an unsupported Python (3.12 or 3.13 is required). '
                         'Delete the .venv folder and run python run.py again to recreate it.')
    requirements = ROOT / ('requirements.txt' if args.demo else 'requirements-ai.txt')
    marker = env / ('demo-installed' if args.demo else 'ai-installed')
    signature = requirements.read_text() + (ROOT / 'requirements.txt').read_text()
    if not marker.exists() or marker.read_text() != signature:
        has_pip = subprocess.call([str(executable), '-m', 'pip', '--version'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        uv = shutil.which('uv')
        if not has_pip and uv:
            subprocess.check_call([uv, 'pip', 'install', '--python', str(executable), '-r', str(requirements)])
        else:
            if not has_pip:
                subprocess.check_call([str(executable), '-m', 'ensurepip', '--upgrade'])
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
            subprocess.check_call([ollama, 'pull', default_llm()])
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
