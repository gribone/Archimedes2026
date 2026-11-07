"""Constrained Greek narration: the LLM selects verified sentences and variants.

Arbitrary model prose is never displayed. Each sentence retains its evidence ID.
"""
import hashlib
import json
import time
import httpx

OLLAMA = 'http://127.0.0.1:11434'
DEFAULT_LLM = 'qwen3:1.7b'


def fact_catalog(profile):
    n = profile['customer_count']
    r = profile['revenue_share'] * 100
    facts = [
        {'id': 'customers', 'source': 'customer_count', 'variants': [f'Το προφίλ περιλαμβάνει {n} διαφορετικούς πελάτες (ΑΦΜ).', f'Σε αυτό το προφίλ ανήκουν {n} πελάτες με διαφορετικό ΑΦΜ.']},
        {'id': 'revenue', 'source': 'revenue_share', 'variants': [f'Αντιστοιχεί στο {r:.1f}% του τζίρου του πελατολογίου.', f'Το μερίδιό του στον τζίρο του πελατολογίου είναι {r:.1f}%.']},
    ]
    for i, feature in enumerate(profile['significant_features'][:6]):
        detail = f"{feature['label']} ({feature['count']}/{feature['total']} ΑΦΜ, {feature['lift']:.2f}× έναντι του συνόλου, q={feature['q_value']:.3g})"
        facts.append({'id': f'feature_{i}', 'source': f'significant_features[{i}]',
                      'variants': [f'Χαρακτηριστικό με αυξημένη παρουσία: {detail}.', f'Παρατηρείται αυξημένη παρουσία για το χαρακτηριστικό: {detail}.']})
    if not profile['significant_features']:
        facts.append({'id': 'no_features', 'source': 'significant_features', 'variants': [
            'Δεν εντοπίστηκαν χαρακτηριστικά που περνούν τα όρια στατιστικής τεκμηρίωσης.',
            'Με τα παρόντα δεδομένα δεν τεκμηριώνονται στατιστικά διακριτά χαρακτηριστικά.']})
    return facts


def validate_draft(draft, facts):
    allowed = {f['id']: f for f in facts}
    choices = draft.get('sentences') if isinstance(draft, dict) else None
    if not isinstance(choices, list) or not 3 <= len(choices) <= len(facts):
        raise ValueError('Invalid number of grounded sentences.')
    ids, text, claims = [], [], []
    for choice in choices:
        if not isinstance(choice, dict) or choice.get('fact_id') not in allowed or type(choice.get('variant')) is not int or choice['variant'] not in [0, 1]:
            raise ValueError('Model returned an unsupported fact or variant.')
        fact_id = choice['fact_id']
        if fact_id in ids:
            raise ValueError('Duplicate model fact.')
        ids.append(fact_id)
        fact = allowed[fact_id]
        sentence = fact['variants'][choice['variant']]
        text.append(sentence)
        claims.append({'fact_id': fact_id, 'source': fact['source'], 'sentence': sentence})
    if not {'customers', 'revenue'}.issubset(ids):
        raise ValueError('Model omitted required portfolio facts.')
    return ' '.join(text), claims


def narrate(profile, root, model=DEFAULT_LLM, demo=False):
    facts = fact_catalog(profile)
    fallback = {'sentences': [{'fact_id': f['id'], 'variant': 0} for f in facts]}
    if demo:
        text, claims = validate_draft(fallback, facts)
        return {'description': text, 'ai_generated': False, 'description_status': 'DEMO: deterministic template, no LLM',
                'claims': claims, 'llm': None}
    # Compact evidence avoids long duplicated Greek variants overwhelming small CPU models.
    # Both variants express exactly the same fact; variant 0 is direct, variant 1 introductory.
    evidence = [{'id': f['id'], 'fact': f['variants'][0]} for f in facts]
    prompt = ('Σύνταξε σύντομη ελληνική περιγραφή επιλέγοντας 3 έως 5 γεγονότα σε φυσική σειρά. '
              'Πρώτα customers, μετά revenue, μετά τα πιο χρήσιμα χαρακτηριστικά. Κάθε id μόνο μία φορά. '
              'variant=0: άμεση διατύπωση, variant=1: εισαγωγική διατύπωση. '
              'Επέστρεψε JSON sentences με fact_id και variant, χωρίς ελεύθερο κείμενο.\n'
              + json.dumps(evidence, ensure_ascii=False, separators=(',', ':')))
    schema = {'type': 'object', 'properties': {'sentences': {'type': 'array', 'minItems': 3,
              'maxItems': min(5, len(facts)), 'items': {'type': 'object', 'properties': {
              'fact_id': {'type': 'string', 'enum': [f['id'] for f in facts]},
              'variant': {'type': 'integer', 'enum': [0, 1]}},
              'required': ['fact_id', 'variant'], 'additionalProperties': False}}},
              'required': ['sentences'], 'additionalProperties': False}
    with httpx.Client(trust_env=False, timeout=300) as client:
        tags = client.get(OLLAMA + '/api/tags')
        tags.raise_for_status()
        available = {m['name']: m for m in tags.json().get('models', [])}
        if model not in available:
            raise ValueError(f'Local Ollama model {model} not found. Run: ollama pull {model}')
        identity = available[model]['digest']
        key = hashlib.sha256((identity + prompt + 'seed42-temp0-schema-v2').encode()).hexdigest()
        cache = root / '.cache' / f'narrative-{key}.json'
        if cache.exists():
            return json.loads(cache.read_text('utf-8'))
        start = time.perf_counter()
        payload = {'model': model, 'prompt': prompt, 'stream': False, 'format': schema, 'keep_alive': 0,
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': 2048, 'num_predict': 240, 'top_k': 1, 'num_thread': 4}}
        if model.startswith('qwen3'):
            payload['think'] = False
        response = client.post(OLLAMA + '/api/generate', json=payload)
        response.raise_for_status()
        raw = response.json()
        try:
            text, claims = validate_draft(json.loads(raw['response']), facts)
        except (ValueError, KeyError, TypeError) as exc:
            cache.parent.mkdir(exist_ok=True)
            (cache.parent / f'invalid-narrative-{key}.json').write_text(
                json.dumps({'model': model, 'raw_response': raw.get('response'), 'error': str(exc)}, ensure_ascii=False, indent=2), 'utf-8')
            raise
        result = {'description': text, 'ai_generated': True, 'description_status': 'AI-generated • constrained Greek narration',
                  'claims': claims, 'llm': {'model': model, 'digest': identity, 'seed': 42, 'temperature': 0,
                  'seconds': time.perf_counter() - start, 'eval_count': raw.get('eval_count'), 'prompt_hash': key,
                  'num_ctx': 2048, 'num_predict': 240, 'num_thread': 4,
                  'load_seconds': raw.get('load_duration', 0) / 1e9,
                  'prompt_tokens': raw.get('prompt_eval_count'), 'prompt_seconds': raw.get('prompt_eval_duration', 0) / 1e9,
                  'raw_response': raw['response']}}
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        return result
