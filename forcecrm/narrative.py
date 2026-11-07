"""Local Greek narration with sentence-level evidence references."""
import hashlib
import json
import re
import time
import httpx

OLLAMA = 'http://127.0.0.1:11434'
DEFAULT_LLM = 'ministral-3:3b'


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


def validate_prose(draft, facts):
    catalog = {f['id']: f for f in facts}
    claims = []
    for row in draft['sentences']:
        ids = row['fact_ids']
        sentence = row['sentence'].strip()
        if not ids or any(i not in catalog for i in ids) or not re.search('[Α-Ωα-ω]', sentence):
            raise ValueError('Missing Greek prose or evidence.')
        evidence = ' '.join(catalog[i]['variants'][0] for i in ids)
        numbers = lambda text: set(re.findall(r'\d+(?:[.,]\d+)?%?', text.replace(',', '.')))
        if not numbers(sentence) <= numbers(evidence):
            raise ValueError('Unsupported numerical claim.')
        claims.append({'fact_ids': ids, 'source': [catalog[i]['source'] for i in ids], 'sentence': sentence})
    if not 3 <= len(claims) <= 6:
        raise ValueError('Expected four to six sentences.')
    return ' '.join(c['sentence'] for c in claims), claims


def narrate(profile, root, model=DEFAULT_LLM, demo=False):
    facts = fact_catalog(profile)
    fallback = {'sentences': [{'fact_id': f['id'], 'variant': 0} for f in facts]}
    if demo:
        text, claims = validate_draft(fallback, facts)
        return {'description': text, 'ai_generated': False, 'description_status': 'DEMO: deterministic template, no LLM',
                'claims': claims, 'llm': None}
    facts = facts[:6]
    evidence = [{'id': f['id'], 'fact': f['variants'][0]} for f in facts]
    for i, feature in enumerate(profile['significant_features'][:4]):
        evidence[i+2]['fact'] = (f"{feature['label']}. Σε {feature['count']} από τους "
                                 f"{feature['total']} πελάτες του προφίλ, συχνότερα από το σύνολο.")
    slots = [([f['id']], f['fact']) for f in evidence[2:5]]
    slots.extend([([f['id']], f['fact']) for f in evidence[:2]])
    prompt = ('Write a single coherent paragraph in contemporary Greek. Return one short Greek sentence '
              'for each numbered field, paraphrasing ONLY the corresponding fact. Use exactly the supplied numbers. '
              'Do not infer causes, advice or percentages. Write all sentences yourself.\n'
              + json.dumps({f's{i}': fact for i, (_, fact) in enumerate(slots)}, ensure_ascii=False))
    fields = [f's{i}' for i in range(len(slots))]
    schema = {'type': 'object', 'required': fields, 'properties': {
        field: {'type': 'string'} for field in fields}, 'additionalProperties': False}
    with httpx.Client(trust_env=False, timeout=300) as client:
        tags = client.get(OLLAMA + '/api/tags')
        tags.raise_for_status()
        available = {m['name']: m for m in tags.json().get('models', [])}
        if model not in available:
            raise ValueError(f'Local Ollama model {model} not found. Run: ollama pull {model}')
        identity = available[model]['digest']
        key = hashlib.sha256((identity + prompt + 'seed42-temp0-full-paragraph-v3').encode()).hexdigest()
        start = time.perf_counter()
        payload = {'model': model, 'prompt': prompt, 'stream': False, 'format': schema, 'keep_alive': '5m',
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': 2048, 'num_predict': 900, 'top_k': 1, 'num_thread': 4}}
        if model.startswith('qwen3'):
            payload['think'] = False
        for attempt in range(3):
            response = client.post(OLLAMA + '/api/generate', json=payload)
            response.raise_for_status()
            raw = response.json()
            try:
                generated = json.loads(raw['response'])
                draft = {'sentences': [{'sentence': generated[field], 'fact_ids': ids}
                                       for field, (ids, _) in zip(fields, slots)]}
                text, claims = validate_prose(draft, facts)
                cited = {i for claim in claims for i in claim['fact_ids']}
                if not {'customers', 'revenue', facts[2]['id']} <= cited:
                    raise ValueError('Include the customer count, revenue and strongest finding.')
                check = dict(payload, prompt=('Έλεγξε κάθε πρόταση έναντι μόνο των αναφερόμενων fact_ids. '
                             'Απόρριψε αν αλλάζει το νόημα, τον αριθμό, τη γεωγραφία ή προσθέτει συμπέρασμα. '
                             'supported=true μόνο αν τεκμηριώνονται όλες οι προτάσεις. '
                             + json.dumps([evidence, claims], ensure_ascii=False)),
                             format={'type': 'object', 'properties': {'supported': {'type': 'boolean'}},
                                     'required': ['supported']})
                verdict = client.post(OLLAMA + '/api/generate', json=check)
                verdict.raise_for_status()
                if json.loads(verdict.json()['response']).get('supported') is not True:
                    raise ValueError('Local evidence review rejected the draft.')
                break
            except (ValueError, KeyError, TypeError) as exc:
                if attempt == 2:
                    raise ValueError(f'Could not verify the Greek description after three drafts: {exc}') from exc
                payload['prompt'] = prompt + f'\nPrevious draft failed: {exc}. Use short literal paraphrases only. Revision {attempt+1}.'
        result = {'description': text, 'ai_generated': True, 'description_status': 'AI-generated • constrained Greek narration',
                  'claims': claims, 'llm': {'model': model, 'digest': identity, 'seed': 42, 'temperature': 0,
                  'seconds': time.perf_counter() - start, 'eval_count': raw.get('eval_count'), 'prompt_hash': key,
                  'num_ctx': 2048, 'num_predict': 900, 'num_thread': 4,
                  'load_seconds': raw.get('load_duration', 0) / 1e9,
                  'prompt_tokens': raw.get('prompt_eval_count'), 'prompt_seconds': raw.get('prompt_eval_duration', 0) / 1e9,
                  'raw_response': raw['response']}}
        return result
