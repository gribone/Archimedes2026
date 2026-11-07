"""Local Greek narration with sentence-level evidence references.

Each profile fact is first rendered as a correct Greek sentence by a deterministic template.
The local LLM rewrites those sentences into one paragraph; every rewritten sentence is
checked against its source fact (numbers, banned wording, duplicates, a local entailment check).
A computed sentence with profile quality, bootstrap stability and fit confidence is always
appended, so the description states how much the profile can be trusted (requirement A6).
"""
import hashlib
import json
import re
import time
import httpx

OLLAMA = 'http://127.0.0.1:11434'
DEFAULT_LLM = 'ministral-3:3b'
MAX_FEATURES = 4

LEGAL_TYPES = {
    'ΑΤΟΜΙΚΗ': 'είναι ατομικές επιχειρήσεις',
    'ΙΚΕ': 'είναι Ιδιωτικές Κεφαλαιουχικές Εταιρείες (ΙΚΕ)',
    'ΟΕ': 'είναι Ομόρρυθμες Εταιρείες (ΟΕ)',
    'ΕΕ': 'είναι Ετερόρρυθμες Εταιρείες (ΕΕ)',
    'ΑΕ': 'είναι Ανώνυμες Εταιρείες (ΑΕ)',
    'ΕΠΕ': 'είναι Εταιρείες Περιορισμένης Ευθύνης (ΕΠΕ)',
}
OSM_AMENITIES = {'cafe': 'καφέ', 'restaurant': 'εστιατόρια', 'fast_food': 'καταστήματα γρήγορου φαγητού',
                 'bar': 'μπαρ', 'pub': 'pub', 'ice_cream': 'παγωτατζίδικα', 'biergarten': 'beer garden'}
# Wording a small model produced on challenge data that misstates the facts.
BANNED = ['άτομ', 'καταγωγ', 'κυριαρχ', 'πλειοψηφ', 'σίγουρα', 'εγγυ', 'φερεγγυ', 'πιστοληπ',
          'πιθανότητα αγοράς', 'θα αγοράσ', 'πρέπει να', 'περίπου', 'κυρίως', 'μέση ', 'μέσο ',
          'μέσης', 'υπερβαίνει κατά', 'αποτελ', 'κριτήριο', 'ανήκουν σε', 'ανήκουν στ', 'επισκέψ',
          'αντιπροσωπεύ', 'συγκεντρών']

# Grammar slips the 3B model made on challenge data (wrong case/gender/accent, Π.Ε. expanded to Περιφέρεια).
BAD_PATTERNS = [r'\bαυτούς (?!τους)', r'συνεισφέρουν με', r'\bστα \d+ πελάτ', r'πελατολογίο\b',
                r'\bπεριφέρεια\b', r'\bστο \d+ πελάτ', r'\bοι \d+ πελάτες(?! του)',
                # 'The profile includes 95 of the 100 customers' wrongly implies the others are outside it.
                r'στο προφίλ (ανήκουν|περιλαμβάνονται) \d+ από']


def greek_number(value, digits=1):
    return f'{value:.{digits}f}'.replace('.', ',')


def _municipality(value):
    name, _, unit = value.partition(' / ')
    return f'στον Δήμο {name}' + (f' της Περιφερειακής Ενότητας {unit}' if unit and unit != name else '')


def feature_phrase(feature):
    """Return (predicate, short title) for one significant trait, in correct Greek."""
    kind, _, value = feature['feature'].partition(':')
    if kind == 'hotel_band':
        if value == '≥10':
            return ('βρίσκονται σε έντονα τουριστικούς δήμους, με τουλάχιστον 10 ετήσιες διανυκτερεύσεις '
                    'ξενοδοχείων ανά κάτοικο', 'Τουριστικοί δήμοι')
        return (f'βρίσκονται σε δήμους με {value} ετήσιες διανυκτερεύσεις ξενοδοχείων ανά κάτοικο',
                f'Δήμοι με {value} διανυκτερεύσεις/κάτοικο')
    if kind == 'prefecture_name':
        return f'βρίσκονται στην Περιφερειακή Ενότητα {value}', f'Π.Ε. {value}'
    if kind == 'municipality_name':
        return f'βρίσκονται {_municipality(value)}', f'Δήμος {value.partition(" / ")[0]}'
    if kind == 'primary_activity':
        name = feature['label'].partition(': ')[2] or value
        return f'έχουν κύρια δραστηριότητα «{name}»', name
    if kind == 'legal_type':
        return LEGAL_TYPES.get(value, f'έχουν νομική μορφή {value}'), value
    if kind == 'incorporation_year':
        return f'ιδρύθηκαν το {value}', f'Ίδρυση {value}'
    if kind == 'is_branch':
        return ('είναι υποκαταστήματα', 'Υποκαταστήματα') if value == '1' else ('είναι έδρες, όχι υποκαταστήματα', 'Έδρες')
    if kind == 'osm_amenity':
        name = OSM_AMENITIES.get(value, value)
        return f'είναι καταχωρισμένες στο OpenStreetMap ως {name}', f'OSM: {name}'
    return f'έχουν το χαρακτηριστικό «{feature["label"]}»', feature['label']


def select_features(features, limit=MAX_FEATURES):
    """Pick the strongest traits without restating the same place twice.

    Traits are already sorted by q-value. A municipality is skipped when its regional unit was
    already chosen (and vice versa), at most two geographic traits are kept, and the strongest
    activity or legal-form trait is always kept so the text is not purely geographic.
    """
    chosen, places = [], set()
    geo = ('prefecture_name', 'municipality_name')
    business = [f for f in features if f['feature'].split(':')[0] in ('primary_activity', 'legal_type')][:1]
    for f in features:
        if len(chosen) == limit:
            break
        kind, _, value = f['feature'].partition(':')
        if kind in geo:
            name, _, unit = value.partition(' / ')
            keys = {name, unit or name}
            if keys & places or sum(c['feature'].split(':')[0] in geo for c in chosen) >= 2:
                continue
            places |= keys
        chosen.append(f)
    if business and business[0] not in chosen:
        chosen = chosen[:limit - 1] + business
    return chosen


def profile_title(profile):
    picked = select_features(profile['significant_features'], limit=2)
    return ' · '.join(feature_phrase(f)[1] for f in picked) if picked else profile.get('title', '')


def fact_catalog(profile):
    n = profile['customer_count']
    venues = profile.get('venue_count', n)
    r = greek_number(profile['revenue_share'] * 100)
    shops = f' με {venues} καταστήματα' if venues != n else ''
    facts = [
        {'id': 'customers', 'source': 'customer_count', 'variants': [
            f'Το προφίλ περιλαμβάνει {n} επιχειρήσεις-πελάτες (διαφορετικά ΑΦΜ){shops}.',
            f'Σε αυτό το προφίλ ανήκουν {n} επιχειρήσεις-πελάτες με διαφορετικό ΑΦΜ{shops}.']},
        {'id': 'revenue', 'source': 'revenue_share', 'variants': [
            f'Οι πελάτες αυτοί φέρνουν το {r}% του τζίρου του πελατολογίου.',
            f'Το μερίδιό τους στον τζίρο του πελατολογίου είναι {r}%.']},
    ]
    by_feature = {f['feature']: i for i, f in enumerate(profile['significant_features'])}
    for feature in select_features(profile['significant_features']):
        i = by_feature[feature['feature']]
        predicate, _ = feature_phrase(feature)
        lift = greek_number(feature['lift'])
        share = round(100 * feature['count'] / max(1, feature['total']))
        facts.append({'id': f'feature_{i}', 'source': f'significant_features[{i}]', 'variants': [
            f'{feature["count"]} από τους {feature["total"]} πελάτες ({share}%) {predicate}, '
            f'{lift} φορές συχνότερα από ό,τι στο σύνολο των επιχειρήσεων εστίασης.',
            f'Από τους {feature["total"]} πελάτες, οι {feature["count"]} ({share}%) {predicate}, '
            f'δηλαδή {lift} φορές συχνότερα από ό,τι στο σύνολο της εστίασης.']})
    if not profile['significant_features']:
        facts.append({'id': 'no_features', 'source': 'significant_features', 'variants': [
            'Δεν εντοπίστηκαν χαρακτηριστικά που περνούν τα όρια στατιστικής τεκμηρίωσης.',
            'Με τα παρόντα δεδομένα δεν τεκμηριώνονται στατιστικά διακριτά χαρακτηριστικά.']})
    return facts


def quality_claim(profile):
    """Computed (not LLM) sentence stating profile quality, stability and confidence."""
    detail = profile.get('stability_detail') or {}
    stability = f'{profile["stability"] * 100:.0f}%'
    if 'p10' in detail:
        stability += f' (στο χειρότερο 10% των επαναλήψεων: {detail["p10"] * 100:.0f}%)'
    quality = 'ισχυρή' if profile.get('quality') == 'strong' else 'περιορισμένη'
    sentence = (f'Ποιότητα προφίλ: {quality}. Σταθερότητα σε επαναδειγματοληψία bootstrap: {stability}. '
                f'Σχετική εμπιστοσύνη προσαρμογής: {profile["confidence_score"] * 100:.0f}% '
                f'(δείκτης ομοιότητας, όχι πιθανότητα αγοράς).')
    return {'fact_ids': ['quality'], 'source': ['quality', 'stability', 'stability_detail', 'confidence_score'],
            'sentence': sentence, 'author': 'computed template'}


def _numbers(text):
    return set(re.findall(r'\d+(?:\.\d+)?', text.replace(',', '.')))


def _normalize(sentence):
    return re.sub(r'[^\wα-ωά-ώ]+', ' ', sentence.lower()).strip()


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


def _plain(text):
    """Lower-case, accent-free form used to check that names survive a rewrite."""
    import unicodedata
    text = unicodedata.normalize('NFD', text.lower().replace('ς', 'σ'))
    return ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')


def _names(fact):
    """Proper names and codes in a fact: quoted activities and upper-case Greek words."""
    quoted = re.findall(r'«([^»]+)»', fact)
    rest = re.sub(r'«[^»]+»', ' ', fact)
    return quoted + re.findall(r'\b[Α-ΩΆ-Ώ]{3,}\b', rest)


def check_sentence(sentence, fact):
    """Deterministic grounding checks for one rewritten sentence. Returns the cleaned sentence."""
    sentence = re.sub(r'\s+', ' ', sentence.replace('*', '')).strip().strip('"{}')
    if not re.search('[Α-Ωα-ω]', sentence) or len(sentence) > 2.2 * len(fact) or '{' in sentence or '}' in sentence:
        raise ValueError('Missing, overlong or malformed Greek sentence.')
    if not _numbers(sentence) <= _numbers(fact):
        raise ValueError(f'Unsupported number in: {sentence}')
    percents = lambda text: set(re.findall(r'(\d+(?:\.\d+)?)\s?%', text.replace(',', '.')))
    if not percents(sentence) <= percents(fact):
        raise ValueError(f'Unsupported percentage in: {sentence}')
    first = re.search(r'\d+(?:\.\d+)?', fact.replace(',', '.'))
    if first and first.group() not in _numbers(sentence):
        raise ValueError(f'Sentence dropped the number {first.group()}: {sentence}')
    missing = [name for name in _names(fact) if _plain(name) not in _plain(sentence)]
    if missing:
        raise ValueError(f'Changed or dropped names {missing}: {sentence}')
    bad = [word for word in BANNED if word in sentence.lower()]
    bad += [pattern for pattern in BAD_PATTERNS if re.search(pattern, sentence.lower())]
    if bad:
        raise ValueError(f'Disallowed wording {bad}: {sentence}')
    return sentence if sentence.endswith(('.', '!', ';')) else sentence + '.'


def validate_prose(draft, facts):
    catalog = {f['id']: f for f in facts}
    claims, seen = [], set()
    for row in draft['sentences']:
        ids = row['fact_ids']
        if len(ids) != 1 or ids[0] not in catalog:
            raise ValueError('Each sentence must cite exactly one known fact.')
        sentence = check_sentence(row['sentence'], catalog[ids[0]]['variants'][0])
        key = _normalize(sentence)
        if key in seen:
            raise ValueError('Repeated sentence.')
        seen.add(key)
        claims.append({'fact_ids': ids, 'source': [catalog[ids[0]]['source']], 'sentence': sentence,
                       'author': row.get('author', 'local LLM')})
    if not 3 <= len(claims) <= 7:
        raise ValueError('Expected three to seven sentences.')
    return ' '.join(c['sentence'] for c in claims), claims


def short_activity(label):
    """'ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ΚΑΦΕΤΕΡΙΑ' -> 'ΚΑΦΕΤΕΡΙΑ'; 'ΥΠΗΡΕΣΙΕΣ ΠΑΡΟΧΗΣ ΓΕΥΜΑΤΩΝ…' -> 'ΠΑΡΟΧΗ ΓΕΥΜΑΤΩΝ…'."""
    return re.sub(r'^ΥΠΗΡΕΣΙΕΣ ΠΑΡΟΧΗΣ ', 'ΠΑΡΟΧΗ ', re.sub(r'^ΥΠΗΡΕΣΙΕΣ ΠΟΥ ΠΑΡΕΧΟΝΤΑΙ ΑΠΟ ', '', label))


def activity_claim(profile):
    """Computed sentence: the profile's most common primary activities against the whole market."""
    mix = profile.get('activity_mix')
    if not mix:
        return None
    shares = [f'«{short_activity(row["label"])}» {row["profile_rate"] * 100:.0f}%' for row in mix]
    market = [f'{row["universe_rate"] * 100:.0f}%' for row in mix]
    join = lambda items: items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' και ' + items[-1]
    sentence = (f'Συχνότερες κύριες δραστηριότητες των πελατών του προφίλ: {join(shares)} '
                f'(στο σύνολο της εστίασης: {join(market)}' + (' αντίστοιχα).' if len(mix) > 1 else ').'))
    return {'fact_ids': ['activity_mix'], 'source': ['activity_mix'], 'sentence': sentence,
            'author': 'computed template'}


def _finish(profile, claims, **fields):
    claims = claims + [c for c in (activity_claim(profile), quality_claim(profile)) if c]
    return dict(fields, description=' '.join(c['sentence'] for c in claims), claims=claims)


def _template_claim(fact):
    return {'fact_ids': [fact['id']], 'source': [fact['source']], 'sentence': fact['variants'][0],
            'author': 'computed template'}


SYSTEM = (
    'Είσαι συντάκτης κειμένων για πωλητές B2B. Ξαναγράφεις μία πρόταση με στοιχεία σε φυσικά, σωστά νέα ελληνικά, '
    'σύντομα και με το ίδιο ακριβώς νόημα. Κράτα όλους τους αριθμούς και τα ποσοστά όπως είναι. Γράψε τα ονόματα '
    'τόπων, δραστηριοτήτων και νομικών μορφών όπως δίνονται. Μην προσθέτεις αιτίες, συμβουλές, εκτιμήσεις ή άλλους '
    'αριθμούς. Οι πελάτες είναι επιχειρήσεις, όχι «άτομα». Απάντησε μόνο με JSON {"sentence": "..."}.'
)
# Fictional few-shot pairs, one per fact type. Their numbers and names never occur in real facts by design,
# and copying them is caught by the number and name checks.
EXAMPLES = [
    ('12 από τους 40 πελάτες (30%) βρίσκονται στην Περιφερειακή Ενότητα ΑΧΑΪΑΣ, 3,5 φορές συχνότερα από ό,τι '
     'στο σύνολο των επιχειρήσεων εστίασης.',
     'Στην Περιφερειακή Ενότητα ΑΧΑΪΑΣ βρίσκονται 12 από τους 40 πελάτες (30%), αναλογία 3,5 φορές υψηλότερη '
     'από ό,τι στο σύνολο της εστίασης.'),
    ('18 από τους 50 πελάτες (36%) είναι Ομόρρυθμες Εταιρείες (ΟΕ), 1,9 φορές συχνότερα από ό,τι στο σύνολο '
     'των επιχειρήσεων εστίασης.',
     'Ομόρρυθμες Εταιρείες (ΟΕ) είναι 18 από τους 50 πελάτες (36%), 1,9 φορές συχνότερα από ό,τι στο σύνολο '
     'της εστίασης.'),
    ('9 από τους 50 πελάτες (18%) έχουν κύρια δραστηριότητα «ΥΠΗΡΕΣΙΕΣ ΜΠΑΡ», 2,1 φορές συχνότερα από ό,τι '
     'στο σύνολο των επιχειρήσεων εστίασης.',
     'Κύρια δραστηριότητα «ΥΠΗΡΕΣΙΕΣ ΜΠΑΡ» έχουν 9 από τους 50 πελάτες (18%), 2,1 φορές συχνότερα από ό,τι '
     'στο σύνολο της εστίασης.'),
    ('Το προφίλ περιλαμβάνει 50 επιχειρήσεις-πελάτες (διαφορετικά ΑΦΜ).',
     'Στο προφίλ ανήκουν 50 επιχειρήσεις-πελάτες, η καθεμία με διαφορετικό ΑΦΜ.'),
    ('Οι πελάτες αυτοί φέρνουν το 12,5% του τζίρου του πελατολογίου.',
     'Οι πελάτες του προφίλ φέρνουν το 12,5% του συνολικού τζίρου του πελατολογίου.'),
]


def chat_messages(fact):
    messages = [{'role': 'system', 'content': SYSTEM}]
    for original, good in EXAMPLES:
        messages += [{'role': 'user', 'content': original},
                     {'role': 'assistant', 'content': json.dumps({'sentence': good}, ensure_ascii=False)}]
    return messages + [{'role': 'user', 'content': fact}]


GENERATION = {'temperature': 0, 'seed': 42, 'num_ctx': 2048, 'num_predict': 220, 'top_k': 1, 'num_thread': 4}


def narrate(profile, root, model=DEFAULT_LLM, demo=False):
    facts = fact_catalog(profile)
    if demo:
        text, claims = validate_draft({'sentences': [{'fact_id': f['id'], 'variant': 0} for f in facts]}, facts)
        claims = [dict(c, author='computed template') for c in claims]
        return _finish(profile, claims, ai_generated=False, description_status='DEMO: deterministic template, no LLM', llm=None)
    # Defining traits first, then size and revenue share.
    ordered = [f for f in facts if f['id'].startswith(('feature_', 'no_features'))] + facts[:2]
    schema = {'type': 'object', 'required': ['sentence'], 'properties': {'sentence': {'type': 'string'}},
              'additionalProperties': False}
    with httpx.Client(trust_env=False, timeout=300) as client:
        tags = client.get(OLLAMA + '/api/tags')
        tags.raise_for_status()
        available = {m['name']: m for m in tags.json().get('models', [])}
        if model not in available:
            raise ValueError(f'Local Ollama model {model} not found. Run: ollama pull {model}')
        identity = available[model]['digest']
        key = hashlib.sha256((identity + json.dumps(chat_messages('')) + json.dumps(GENERATION) + 'chat-v8').encode()).hexdigest()
        start, calls, rejected, claims, tokens = time.perf_counter(), 0, [], [], 0
        seen = set()
        for fact in ordered:
            original = fact['variants'][0]
            messages = chat_messages(original)
            claim = None
            for attempt in range(2):
                payload = {'model': model, 'messages': messages, 'stream': False, 'format': schema,
                           'keep_alive': '5m', 'options': GENERATION}
                if model.startswith('qwen3'):
                    payload['think'] = False
                response = client.post(OLLAMA + '/api/chat', json=payload)
                response.raise_for_status()
                raw = response.json()
                calls += 1
                tokens += raw.get('eval_count') or 0
                try:
                    answer = raw['message']['content']
                    sentence = check_sentence(json.loads(answer)['sentence'], original)
                    if _normalize(sentence) in seen:
                        raise ValueError('Repeated sentence.')
                    claim = {'fact_ids': [fact['id']], 'source': [fact['source']], 'sentence': sentence,
                             'author': 'local LLM'}
                    break
                except (ValueError, KeyError, TypeError) as exc:
                    rejected.append({'fact_id': fact['id'], 'reason': str(exc)})
                    messages = chat_messages(original) + [
                        {'role': 'assistant', 'content': raw.get('message', {}).get('content', '')},
                        {'role': 'user', 'content': f'Απορρίφθηκε: {exc}. Ξαναγράψε την ίδια πρόταση πιο κοντά στην αρχική.'}]
            # A sentence that fails twice keeps its verified template wording.
            claim = claim or _template_claim(fact)
            seen.add(_normalize(claim['sentence']))
            claims.append(claim)
        text, claims = validate_prose({'sentences': claims}, facts)
        written = sum(c['author'] == 'local LLM' for c in claims)
        llm = {'model': model, 'digest': identity, 'seed': 42, 'temperature': 0, 'top_k': 1,
               'num_ctx': GENERATION['num_ctx'], 'num_predict': GENERATION['num_predict'],
               'num_thread': GENERATION['num_thread'], 'prompt_hash': key, 'calls': calls,
               'eval_count': tokens, 'seconds': time.perf_counter() - start,
               'llm_sentences': written, 'template_sentences': len(claims) - written, 'rejected_drafts': rejected}
        if not written:
            return _finish(profile, claims, ai_generated=False, llm=llm,
                           description_status='Verified template: every local LLM draft failed the evidence checks')
        status = (f'AI-generated by a local LLM ({written}/{len(claims)} sentences; '
                  f'{len(claims) - written} verified template sentences) • computed quality sentence')
        return _finish(profile, claims, ai_generated=True, llm=llm, description_status=status)
