"""Grounded profile titles: the LLM picks from data-backed phrases; it never writes free text.

Every candidate phrase is derived from a statistically significant trait (place, legal form)
or from the profile's measured activity mix, so any title the model assembles stays true.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import time
import httpx

from .narrative import OLLAMA

# Short Greek family names for the most common HORECA activity codes (KAD).
FAMILIES = {
    'καφετέριες': ['56300007', '56300008'],
    'αναψυκτήρια': ['56300003', '56110911'],
    'καφέ-μπαρ': ['56300004'],
    'καφενεία': ['56300005', '56300006', '56300014'],
    'μπαρ': ['56300001', '56300000'],
    'εστιατόρια/ταβέρνες': ['56110000', '56110100', '56110103', '56110104', '56110202', '56110200'],
    'ψητοπωλεία': ['56110109', '56110203', '56110902'],
    'πιτσαρίες': ['56110108', '56110914'],
    'fast food': ['56110105', '56110201', '56110901', '56110910'],
    'σνακ μπαρ/ουζερί': ['56110110', '56110111', '56110204', '56110907'],
    'ζαχαροπλαστεία': ['56110113', '56110913'],
    'κυλικεία/καντίνες': ['56110903', '56110906', '56222000', '56222004', '56120001'],
    'catering': ['56210000', '56220000', '56221901'],
}
CODE_FAMILY = {code: family for family, codes in FAMILIES.items() for code in codes}
HOTEL_BANDS = {'≥10': 'Τουριστικοί δήμοι', '1–<10': 'Δήμοι με μέτρια τουριστική κίνηση',
               '<1': 'Δήμοι με χαμηλή τουριστική κίνηση'}
MIN_FAMILY_SHARE = .15


def family_name(code, description):
    if code in CODE_FAMILY:
        return CODE_FAMILY[code]
    text = description or code
    text = text.split(' ΑΠΟ ')[-1]
    text = re.split(r'[,(\[]| ΜΕ | ΧΩΡΙΣ ', text)[0].strip()
    return text.lower()[:40] or code


def activity_mix(member_codes, member_vats, names):
    """Share of distinct VATs in the profile that operate each activity family."""
    families_by_vat = {}
    for vat, codes in zip(member_vats, member_codes):
        families_by_vat.setdefault(vat, set()).update(family_name(code, names.get(code)) for code in codes)
    counts = Counter(f for families in families_by_vat.values() for f in families)
    total = max(1, len(families_by_vat))
    return [{'family': f, 'share': n / total, 'count': n} for f, n in counts.most_common()]


def title_options(profile):
    places, legal = [], []
    for feature in profile['significant_features']:
        kind, _, value = feature['feature'].partition(':')
        # A title phrase must describe a meaningful part of the profile, not a small enclave.
        if feature['lift'] < 1.5 or feature['profile_rate'] < MIN_FAMILY_SHARE:
            continue
        if kind == 'hotel_band' and value in HOTEL_BANDS:
            places.append({'phrase': HOTEL_BANDS[value], 'feature': feature})
        elif kind == 'prefecture_name':
            places.append({'phrase': f'Π.Ε. {value}', 'feature': feature})
        elif kind == 'municipality_name':
            places.append({'phrase': f'Δήμος {value.split(" / ")[0]}', 'feature': feature})
        elif kind == 'legal_type' and feature['profile_rate'] >= .5:
            legal.append({'phrase': f'κυρίως {value}', 'feature': feature})
    # Widest coverage first: small models favour the first options they see.
    places.sort(key=lambda p: -p['feature']['profile_rate'])
    places = places[:5] or [{'phrase': 'Πανελλαδική κατανομή', 'feature': None}]
    activities = [m for m in profile['activity_mix'] if m['share'] >= MIN_FAMILY_SHARE][:4]
    activities = [{'phrase': m['family'], 'share': m['share']} for m in activities] or \
                 [{'phrase': 'μικτή εστίαση', 'share': None}]
    return places, activities, legal


def assemble(places, activities, legal):
    title = ' & '.join(places) + ' · ' + ', '.join(activities)
    return title + (f' · {legal}' if legal else '')


def default_choice(places, activities, legal):
    return {'second_place': 'none', 'activities': [a['phrase'] for a in activities[:2]],
            'legal': legal[0]['phrase'] if legal else 'none'}


def validate_choice(choice, places, activities, legal):
    # The widest-coverage place always leads; the model may only add a second one.
    lead, extra_places = places[0]['phrase'], {p['phrase'] for p in places[1:]} | {'none'}
    allowed_activities = {a['phrase'] for a in activities}
    allowed_legal = {l['phrase'] for l in legal} | {'none'}
    if not isinstance(choice, dict):
        raise ValueError('Title choice is not an object.')
    chosen_activities = choice.get('activities')
    if not isinstance(chosen_activities, list) or not 1 <= len(chosen_activities) <= 2 \
            or len(set(chosen_activities)) != len(chosen_activities) or not set(chosen_activities) <= allowed_activities:
        raise ValueError('Title choice uses an unsupported activity phrase.')
    if choice.get('second_place') not in extra_places:
        raise ValueError('Title choice uses an unsupported place phrase.')
    chosen_places = [lead] + ([] if choice['second_place'] == 'none' else [choice['second_place']])
    if choice.get('legal') not in allowed_legal:
        raise ValueError('Title choice uses an unsupported legal phrase.')
    legal_phrase = None if choice['legal'] == 'none' else choice['legal']
    return assemble(chosen_places, chosen_activities, legal_phrase)


def make_title(profile, root, model, demo=False):
    places, activities, legal = title_options(profile)
    fallback = default_choice(places, activities, legal)
    if demo:
        return {'title': validate_choice(fallback, places, activities, legal), 'title_source': 'template'}
    evidence = {
        'title_starts_with': places[0]['phrase'],
        'second_place_options': [{'phrase': p['phrase'], 'lift': round(p['feature']['lift'], 2) if p['feature'] else None,
                    'share': round(p['feature']['profile_rate'], 2) if p['feature'] else None} for p in places[1:]],
        'activities': [{'phrase': a['phrase'], 'share': round(a['share'], 2) if a['share'] else None} for a in activities],
        'legal': [{'phrase': l['phrase'], 'share': round(l['feature']['profile_rate'], 2)} for l in legal],
    }
    prompt = ('Δημιούργησε σύντομο, περιγραφικό τίτλο για ένα προφίλ πελατών εστίασης. '
              'share = ποσοστό των πελατών του προφίλ, lift = πόσο πιο συχνό είναι από την αγορά. '
              'Ο τίτλος ξεκινά πάντα με το title_starts_with. Πρόσθεσε second_place μόνο αν προσθέτει '
              'σημαντική πληροφορία (μεγάλο share), αλλιώς "none". '
              'Διάλεξε 1-2 activities με το μεγαλύτερο share. '
              'Για legal διάλεξε τη φράση αν υπάρχει, αλλιώς "none". '
              'Χρησιμοποίησε μόνο τις φράσεις που δίνονται.\n' + json.dumps(evidence, ensure_ascii=False))
    schema = {'type': 'object', 'properties': {
        'second_place': {'type': 'string', 'enum': [p['phrase'] for p in places[1:]] + ['none']},
        'activities': {'type': 'array', 'minItems': 1, 'maxItems': 2,
                       'items': {'type': 'string', 'enum': [a['phrase'] for a in activities]}},
        'legal': {'type': 'string', 'enum': [l['phrase'] for l in legal] + ['none']}},
        'required': ['second_place', 'activities', 'legal'], 'additionalProperties': False}
    with httpx.Client(trust_env=False, timeout=300) as client:
        tags = client.get(OLLAMA + '/api/tags')
        tags.raise_for_status()
        available = {m['name']: m for m in tags.json().get('models', [])}
        if model not in available:
            raise ValueError(f'Local Ollama model {model} not found. Run: ollama pull {model}')
        key = hashlib.sha256((available[model]['digest'] + prompt + 'title-v4').encode()).hexdigest()
        cache = root / '.cache' / f'title-{key}.json'
        if cache.exists():
            return json.loads(cache.read_text('utf-8'))
        start = time.perf_counter()
        payload = {'model': model, 'prompt': prompt, 'stream': False, 'format': schema, 'keep_alive': 0,
                   'options': {'temperature': 0, 'seed': 42, 'num_ctx': 2048, 'num_predict': 120, 'top_k': 1,
                               'num_thread': 4}}
        if model.startswith('qwen3'):
            payload['think'] = False
        response = client.post(OLLAMA + '/api/generate', json=payload)
        response.raise_for_status()
        raw = response.json().get('response')
        try:
            choice = json.loads(raw)
            title = validate_choice(choice, places, activities, legal)
            source = 'ai'
        except (ValueError, TypeError):
            # An invalid model choice falls back to the deterministic, equally grounded title.
            choice, title, source = fallback, validate_choice(fallback, places, activities, legal), 'template-fallback'
        result = {'title': title, 'title_source': source,
                  'title_choice': {'model': model, 'choice': choice, 'raw_response': raw,
                                   'seconds': time.perf_counter() - start}}
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), 'utf-8')
        return result
