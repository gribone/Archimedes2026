"""Local comparison runner. Missing models are recorded, never replaced with fake scores."""
import argparse
import json
import os
from pathlib import Path
import time
import gc
import platform
import subprocess
import numpy as np
import psutil


def hardware():
    """CPU and RAM of the machine running the benchmark, recorded with every result file."""
    cpu = platform.processor()
    try:
        if platform.system() == 'Darwin':
            cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() or cpu
        elif platform.system() == 'Linux':
            cpu = next(l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name'))
    except Exception:
        pass
    return {'cpu': cpu or platform.machine(), 'ram_gib': psutil.virtual_memory().total / 1024**3,
            'platform': platform.platform()}


def embeddings(args):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    from forcecrm.data import load_universe
    _, _, d, _ = load_universe(args.data)
    # Challenge-derived weak labels: same 4-digit family versus other foodservice families.
    # This measures hierarchy recovery, not human-annotated semantic correctness.
    sample = d[d.activity_id.str.startswith('56')].sort_values('activity_id').head(400)
    families = sample.activity_id.str[:4].to_numpy()
    results = []
    for path in args.embedding_paths:
        row = {'model_path': path, 'task': 'Greek activity hierarchy retrieval proxy', 'descriptions': len(sample)}
        try:
            start = time.perf_counter()
            model = SentenceTransformer(path, local_files_only=True, trust_remote_code=False, device='cpu')
            model.max_seq_length = 256
            row['load_seconds'] = time.perf_counter() - start
            start = time.perf_counter()
            prefix = 'query: ' if 'e5' in path.lower() else ''
            values = model.encode([prefix + t for t in sample.description], normalize_embeddings=True, batch_size=32)
            row['seconds_per_description'] = (time.perf_counter() - start) / len(sample)
            similarity = values @ values.T
            np.fill_diagonal(similarity, -np.inf)
            top = np.argsort(-similarity, axis=1)[:, :5]
            eligible = np.array([(families == family).sum() > 1 for family in families])
            row['family_hit_at_5'] = float(np.mean(np.any(families[top[eligible]] == families[eligible, None], axis=1)))
            row['rss_mb_after_encoding'] = psutil.Process().memory_info().rss / 1024**2
            row['model_bytes_on_disk'] = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            row['status'] = 'measured'
            del model
            gc.collect()
        except Exception as exc:
            row.update(status='unavailable', error=str(exc))
        results.append(row)
    return {'kind': 'embedding', 'hardware': hardware(), 'limitations': 'Weak code-family labels are not a Greek human relevance benchmark. RSS includes Python and prior allocations, not isolated model peak memory.', 'results': results}


def llms(args):
    from forcecrm.narrative import narrate
    import httpx
    with httpx.Client(trust_env=False, timeout=10) as client:
        response = client.get('http://127.0.0.1:11434/api/tags')
        response.raise_for_status()
        available = {model['name'] for model in response.json().get('models', [])}
    profiles = json.loads(args.profiles.read_text('utf-8'))[:args.limit_profiles]
    results = []
    for model in args.llm_models:
        if model not in available:
            results.append({'model': model, 'status': 'unavailable', 'valid_output_rate': None, 'profiles': []})
            continue
        rows, resident = [], None
        for profile in profiles:
            start = time.perf_counter()
            try:
                result = narrate(profile, args.data, model=model)
                llm = result['llm']
                rows.append({'profile_id': profile['profile_id'], 'valid_grounded_output': True,
                             'llm_sentences': llm['llm_sentences'], 'template_sentences': llm['template_sentences'],
                             'llm_calls': llm['calls'], 'rejected_drafts': len(llm['rejected_drafts']),
                             'generation_seconds': llm['seconds'], 'wall_seconds': time.perf_counter()-start,
                             'description': result['description'], 'digest': llm['digest'],
                             'human_greek_fluency_1_to_5': None, 'human_coherence_1_to_5': None})
            except Exception as exc:
                rows.append({'profile_id': profile['profile_id'], 'valid_grounded_output': False, 'error': str(exc)})
            with httpx.Client(trust_env=False, timeout=10) as client:
                loaded = client.get('http://127.0.0.1:11434/api/ps').json().get('models', [])
            resident = next((m.get('size') for m in loaded if m.get('name') == model), resident)
        valid = [r for r in rows if r['valid_grounded_output']]
        written = sum(r['llm_sentences'] for r in valid)
        total = written + sum(r['template_sentences'] for r in valid)
        results.append({'model': model, 'status': 'measured',
                        'valid_output_rate': len(valid)/max(1, len(rows)),
                        'llm_sentence_share': written/total if total else None,
                        'ollama_resident_bytes': resident, 'profiles': rows})
    return {'kind': 'llm', 'hardware': hardware(), 'limitations': 'Each fact sentence is rewritten separately and must pass deterministic number, percentage, name and wording checks; failing sentences keep their verified template. llm_sentence_share measures how often a model passes those checks, not fluency. Fluency still requires blinded human review. Missing models count as unavailable, not quality failures.', 'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['embeddings', 'llms'])
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--profiles', type=Path, default=Path('outputs/cli/profiles.json'))
    parser.add_argument('--limit-profiles', type=int, default=6, help='Use the same first N profiles for every LLM')
    parser.add_argument('--embedding-paths', nargs='+', default=['models/e5-small','models/e5-base','models/multilingual-minilm'])
    parser.add_argument('--llm-models', nargs='+', default=['ministral-3:3b','qwen3:1.7b','gemma3:1b'])
    parser.add_argument('--output', type=Path, default=Path('outputs/benchmark.json'))
    args = parser.parse_args()
    result = embeddings(args) if args.kind == 'embeddings' else llms(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
