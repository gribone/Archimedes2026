"""In-app help chat backed by the same local Ollama model that writes the profile descriptions.

Scope and guardrails (see docs/RESPONSIBLE_AI.md):
- The assistant only explains how to use the app and how to read its results. It answers from a fixed
  Greek help text (HELP) plus a read-only summary of the current analysis; it never changes profiles,
  scores or rankings.
- Questions about creditworthiness or solvency get a fixed refusal without calling the model.
- Hand-off to support does not depend on the 3B model's judgement: keyword rules and repeated
  questions trigger it deterministically; the model's own [ΥΠΟΣΤΗΡΙΞΗ] marker is an extra signal.
- Every answer is labelled as AI-generated in the UI. Generation is greedy (temperature 0, seed 42).
"""
from __future__ import annotations

import json
import re
import threading

import httpx

from .db import fold
from .narrative import DEFAULT_LLM, OLLAMA

MARKER = '[ΥΠΟΣΤΗΡΙΞΗ]'
MAX_HISTORY = 6          # previous messages sent back to the model
MAX_MESSAGE_CHARS = 1000
GENERATION = {'temperature': 0, 'seed': 42, 'top_k': 1, 'num_ctx': 4096, 'num_predict': 320, 'num_thread': 4}
SUPPORT = {'name': 'Υποστήριξη Bizcover (demo)', 'email': 'support@bizcover.example',
           'phone': '+30 210 000 0000', 'hours': 'Δευτέρα–Παρασκευή, 09:00–17:00'}

HELP = """ΟΔΗΓΟΣ ΤΗΣ ΕΦΑΡΜΟΓΗΣ Bizcover «Ποιος είναι ο επόμενος πελάτης;»
Σκοπός: η εφαρμογή βρίσκει ομάδες ομοίων πελατών (προφίλ) σε ένα πελατολόγιο επιχειρήσεων εστίασης (ΚΑΔ 56) και προτείνει επιχειρήσεις που δεν είναι ακόμη πελάτες αλλά μοιάζουν με αυτούς. Όλη η ανάλυση και το AI τρέχουν τοπικά στον υπολογιστή.

ΜΕΝΟΥ (αριστερά): Επισκόπηση, Προφίλ πελατών, Νέες ευκαιρίες, Τεκμηρίωση. Πάνω δεξιά: κουμπί σκοτεινού/φωτεινού θέματος, κουμπί «?» για επανάληψη της ξενάγησης, όνομα χρήστη και «Αποσύνδεση».

1. ΕΠΙΣΚΟΠΗΣΗ
- Η Επισκόπηση είναι η πρώτη σελίδα. Στην κάρτα «Ξεκινήστε από τους πελάτες σας» σύρετε ή επιλέξτε αρχείο Excel (.xlsx, έως 10 MB) με στήλες ΑΦΜ και τζίρο (έσοδα). Υπάρχει σύνδεσμος «Λήψη προτύπου» για έτοιμο κενό αρχείο. Μετά πατήστε «Ανάλυση πελατολογίου».
- «Δοκιμή με demo δεδομένα»: τρέχει την εφαρμογή με τυχαίο δείγμα, χωρίς δικό σας αρχείο. Οι αριθμοί του demo δεν μετρούν πραγματική απόδοση.
- Ο τζίρος χρησιμοποιείται μόνο ως βάρος στην ανάλυση (οι πελάτες με μεγαλύτερο τζίρο μετρούν περισσότερο).
- Η ανάλυση διαρκεί συνήθως 1–3 λεπτά· η πρώτη φορά διαρκεί περισσότερο. Όσο τρέχει, εμφανίζεται μήνυμα προόδου.
- «Περισσότερες επιλογές» (εμφανίζονται μόνο στον λογαριασμό «Developer», που επιλέγεται στη σελίδα σύνδεσης):
  α) Βάρη ομοιότητας: πόσο μετρά η δραστηριότητα (65), η γεωγραφία (20), η νομική μορφή (10) και η ηλικία της επιχείρησης (5). Οι προεπιλογές έχουν ελεγχθεί· η αλλαγή τους μπορεί να μειώσει την ακρίβεια. Υπάρχει «Επαναφορά προεπιλογών».
  β) Φίλτρο με ζωντανά δεδομένα ΓΕΜΗ: ελέγχει αν οι κορυφαίες προτάσεις είναι ακόμη ενεργές. Είναι η μόνη λειτουργία που χρησιμοποιεί διαδίκτυο, χρειάζεται προσωπικό κλειδί api_key από το opendata.businessportal.gr και στέλνει μόνο τους ΑΦΜ των προτεινόμενων επιχειρήσεων, ποτέ στοιχεία των πελατών σας.
- Μετά την ανάλυση, οι κάρτες στην κορυφή δείχνουν πόσες επιχειρήσεις υπάρχουν συνολικά, πόσους πελάτες αναλύσαμε, πόσα προφίλ βρέθηκαν και πόσες νέες ευκαιρίες.
- «Εξαγωγή αποτελεσμάτων» (πάνω δεξιά): κατεβάζει όλα τα αρχεία σε ένα .zip.

2. ΠΡΟΦΙΛ ΠΕΛΑΤΩΝ
- Η εφαρμογή βρίσκει 1 έως 6 προφίλ. Κάθε κάρτα προφίλ ανοίγει/κλείνει με κλικ.
- Περιγραφή: ελληνικό κείμενο από τοπικό μοντέλο AI (σημειώνεται ως AI), όπου κάθε πρόταση στηρίζεται σε μετρημένο στοιχείο.
- Πελάτες και μερίδιο τζίρου: πόσοι πελάτες ανήκουν στο προφίλ και τι ποσοστό του συνολικού τζίρου φέρνουν.
- Σταθερότητα: πόσο συχνά ξαναβρίσκεται το ίδιο προφίλ όταν επαναλαμβάνουμε την ανάλυση με τυχαίες παραλλαγές του πελατολογίου (bootstrap). Όσο πιο κοντά στο 100%, τόσο πιο σταθερό το προφίλ. Η εφαρμογή δεν ορίζει όριο «καλής» ή «κακής» σταθερότητας.
- Ταίριασμα πελατών (confidence): πόσο κοντά είναι κατά μέσο όρο οι πελάτες στο «κέντρο» του προφίλ. Δεν είναι πιθανότητα αγοράς. Περνώντας τον κέρσορα πάνω από το ⓘ εμφανίζεται εξήγηση.
- Τεκμηρίωση ανά προφίλ: «Ισχυρή» (σταθερότητα τουλάχιστον 75% και τουλάχιστον 2 τεκμηριωμένα χαρακτηριστικά) ή «Περιορισμένη».
- «Τι τους κάνει ξεχωριστούς»: χαρακτηριστικά που εμφανίζονται στους πελάτες του προφίλ πολύ συχνότερα από ό,τι στην αγορά. «Πόσο συχνότερα» = πόσες φορές συχνότερα (lift). «Βεβαιότητα» (Πολύ υψηλή, Υψηλή, Επαρκής) = πόσο απίθανο είναι το εύρημα να οφείλεται σε τύχη, μετά από στατιστικό έλεγχο Fisher με διόρθωση για πολλαπλές συγκρίσεις. Οι ακριβείς τιμές p-value και q-value, οι χρόνοι και οι πλήρεις αναφορές εμφανίζονται μόνο στον λογαριασμό «Developer».
- «Τι δουλειά κάνουν οι πελάτες του»: οι συχνότερες κύριες δραστηριότητες (ΚΑΔ) στους πελάτες του προφίλ, σε σύγκριση με την αγορά.
- Κουμπί «Δείτε τις ευκαιρίες αυτού του προφίλ»: ανοίγει τις Νέες ευκαιρίες φιλτραρισμένες σε αυτό το προφίλ. Μόνο το πρώτο προφίλ είναι ανοιχτό· τα άλλα ανοίγουν με κλικ.
- Πάνελ «Από όλο το πελατολόγιο»: χαρακτηριστικά κοινά σε όλους τους πελάτες, όχι σε ένα μόνο προφίλ.
- Ένας πελάτης μπορεί να μην ταιριάζει σε κανένα προφίλ· αυτό είναι αποδεκτό αποτέλεσμα.

3. ΝΕΕΣ ΕΥΚΑΙΡΙΕΣ
- Χάρτης «Γεωγραφία ευκαιριών»: πού βρίσκονται οι πρώτες 2.000 προτεινόμενες επιχειρήσεις. Ο χάρτης είναι τοπικός, χωρίς διαδίκτυο.
- Πίνακας: μία πρόταση ανά ΑΦΜ. «Κατάταξη» δείχνει τη συνολική θέση και τη θέση μέσα στο προφίλ. «Ταίριασμα» συνδυάζει την ομοιότητα με το προφίλ και το πόσο συχνά παρόμοιες επιχειρήσεις είναι ήδη πελάτες. «Γιατί προτείνεται» εξηγεί την πρόταση.
- Φίλτρα: επιλογή προφίλ, αναζήτηση με επωνυμία, πόλη ή ΑΦΜ, και (αν έγινε έλεγχος ΓΕΜΗ) απόκρυψη ανενεργών. Κουμπιά «Προηγούμενη/Επόμενη» για σελίδες.
- Οι προτάσεις είναι για ανθρώπινη αξιολόγηση: επιβεβαιώστε την πραγματική δραστηριότητα πριν επικοινωνήσετε.

4. ΤΕΚΜΗΡΙΩΣΗ
- Εσωτερικός έλεγχος: κρύβουμε το 20% των πελατών πριν την ανάλυση και μετράμε πόσους ξαναβρίσκουμε στις πρώτες 500 και 2.000 προτάσεις (Recall@500, Recall@2000· π.χ. «15 από 60 κρυμμένους πελάτες βρέθηκαν στις πρώτες 500 προτάσεις»), σε σύγκριση με απλή μέθοδο αναφοράς (baseline: ίδια κύρια δραστηριότητα).
- Ποιότητα δεδομένων (κάλυψη GPS, πλήθος καταστημάτων), μεθοδολογία, περιορισμοί και λήψη μεμονωμένων αρχείων: profiles.json, assignments.csv, lookalikes.csv, report.json.

ΠΕΡΙΟΡΙΣΜΟΙ
- Η εφαρμογή ΔΕΝ υπολογίζει πιστοληπτική ικανότητα, φερεγγυότητα ή πιθανότητα αγοράς.
- Οι ΚΑΔ είναι αυτοδηλωμένοι· κάποιοι δευτερεύοντες δεν αντιστοιχούν σε πραγματική δραστηριότητα.
- Τα προφίλ σε αυτό το πελατολόγιο είναι κυρίως γεωγραφικά (π.χ. τουριστικά νησιά, Χαλκιδική–Θεσσαλονίκη, Αττική).
- Η στατιστική τεκμηρίωση είναι διερευνητική, γιατί τα ίδια δεδομένα σχηματίζουν και ελέγχουν τα προφίλ.

ΣΥΧΝΑ ΠΡΟΒΛΗΜΑΤΑ
- «Local Ollama model not found»: τρέξτε `ollama pull ministral-3:3b` ή ξεκινήστε την εφαρμογή με `python run.py`.
- Λάθος αρχείο: πρέπει να είναι .xlsx με στήλη ΑΦΜ (κείμενο, με τα αρχικά μηδενικά) και στήλη τζίρου.
- «An analysis is already running»: περιμένετε να τελειώσει η τρέχουσα ανάλυση.
"""

SYSTEM = f"""Είσαι ο βοηθός χρήσης της εφαρμογής Bizcover. Απαντάς ΜΟΝΟ στα ελληνικά, σύντομα (έως 5 προτάσεις), φιλικά και απλά, για χρήστες χωρίς τεχνικές γνώσεις. Απευθύνεσαι πάντα στον πληθυντικό ευγενείας («εσείς»).
Κανόνες:
1. Απαντάς μόνο με βάση τον ΟΔΗΓΟ και τα ΣΤΟΙΧΕΙΑ ΤΗΣ ΑΝΑΛΥΣΗΣ παρακάτω. Μην επινοείς κουμπιά, λειτουργίες ή αριθμούς. Κάθε αριθμός που γράφεις πρέπει να υπάρχει αυτούσιος παρακάτω.
2. Δεν αλλάζεις και δεν σχολιάζεις ως «σωστά/λάθος» τα προφίλ, τα σκορ ή την κατάταξη· μόνο εξηγείς τι σημαίνουν.
3. Ποτέ δεν κρίνεις αν μια επιχείρηση είναι φερέγγυα, αξιόπιστη στις πληρωμές ή αν θα αγοράσει.
4. Αν η ερώτηση δεν απαντιέται από τον ΟΔΗΓΟ, ή ο χρήστης θέλει άνθρωπο, ή αναφέρει σφάλμα που δεν λύνεται από τον ΟΔΗΓΟ, ξεκίνα την απάντηση με {MARKER} και πρότεινε την ομάδα υποστήριξης. Μην μαντεύεις αιτίες σφαλμάτων που δεν γράφει ο ΟΔΗΓΟΣ.
5. Αν η ερώτηση δεν αφορά την εφαρμογή, απάντησε μόνο: «Μπορώ να βοηθήσω μόνο με τη χρήση του Bizcover. Τι θα θέλατε να μάθετε για την εφαρμογή;»

{HELP}"""

REFUSAL_CREDIT = ('Δεν μπορώ να εκτιμήσω φερεγγυότητα, πιστοληπτική ικανότητα ή αν μια επιχείρηση θα πληρώσει ή θα αγοράσει. '
                  'Το Bizcover δείχνει μόνο ομοιότητα με τους υπάρχοντες πελάτες σας, για να αποφασίσετε εσείς με ποιους θα επικοινωνήσετε. '
                  'Για τέτοιες αποφάσεις χρησιμοποιήστε τις επίσημες διαδικασίες της εταιρείας σας.')
BUSY_ANALYSIS = ('Τώρα τρέχει μια ανάλυση που χρησιμοποιεί το ίδιο τοπικό μοντέλο AI. '
                 'Για να μη βαρύνει ο υπολογιστής, θα σας απαντήσω μόλις τελειώσει. '
                 'Αν χρειάζεστε βοήθεια αμέσως, πατήστε «Επικοινωνία με υποστήριξη».')
HANDOFF = ('Φυσικά. Παρακάτω θα βρείτε τα στοιχεία της ομάδας υποστήριξης. '
           'Αν θέλετε, στείλτε τους και αυτή τη συνομιλία με ένα κλικ, ώστε να μη χρειαστεί να εξηγήσετε ξανά.')
BUSY_CHAT = 'Ολοκληρώνω μια άλλη απάντηση. Δοκιμάστε ξανά σε λίγα δευτερόλεπτα.'
UNAVAILABLE = ('Ο βοηθός δεν μπορεί να συνδεθεί στο τοπικό μοντέλο AI (Ollama). '
               f'Ξεκινήστε την εφαρμογή με `python run.py` ή τρέξτε `ollama pull {DEFAULT_LLM}`. '
               'Μέχρι τότε, η ομάδα υποστήριξης μπορεί να βοηθήσει.')

# Matched on fold()-ed text (lower case, no accents), so «ΛΆΘΟΣ», «λαθος» and «λάθος» all match.
CREDIT_TERMS = ['φερεγγυ', 'πιστοληπ', 'πιστωτικ', 'αξιοχρε', 'κακοπληρ', 'θα πληρωσ', 'θα με πληρωσ',
                'δανει', 'credit', 'solvency', 'creditworth']
# Asking for a person: answered with a fixed reply and the support card, without the model.
HUMAN_TERMS = ['ανθρωπ', 'υπαλληλ', 'υποστηριξ', 'εκπροσωπ', 'τηλεφων', 'support', 'human', 'agent']
# Reporting a problem or frustration: the model still tries to help, and the support card is shown too.
PROBLEM_TERMS = ['λαθος', 'δεν δουλευ', 'δεν λειτουργ', 'χαλασ', 'σφαλμα', 'error', 'bug', 'κολλησ', 'κολλαει',
                 'παραπον', 'δεν βοηθ', 'αχρηστ', 'not working', 'broken', 'wrong', 'useless']
# fold() also casefolds final ς to σ, so the terms go through the same function.
CREDIT_TERMS = [fold(t) for t in CREDIT_TERMS]
HUMAN_TERMS = [fold(t) for t in HUMAN_TERMS]
PROBLEM_TERMS = [fold(t) for t in PROBLEM_TERMS]
REASONS = {'human': 'Ο χρήστης ζήτησε να μιλήσει με άνθρωπο',
           'problem': 'Ο χρήστης ανέφερε πρόβλημα',
           'repeat': 'Ο χρήστης επανέλαβε την ίδια ερώτηση',
           'model': 'Ο βοηθός δεν βρήκε την απάντηση στον οδηγό',
           'unavailable': 'Το τοπικό μοντέλο δεν είναι διαθέσιμο'}

_chat_lock = threading.Lock()


def _words(text):
    return set(re.findall(r'\w{3,}', fold(text)))


def is_credit_question(text):
    folded = fold(text)
    return any(term in folded for term in CREDIT_TERMS)


def escalation_reason(message, history):
    """Deterministic hand-off rules; they do not depend on the language model."""
    folded = fold(message)
    if any(term in folded for term in HUMAN_TERMS):
        return 'human'
    if any(term in folded for term in PROBLEM_TERMS):
        return 'problem'
    words = _words(message)
    for previous in history:
        if previous.get('role') != 'user':
            continue
        before = _words(previous.get('content', ''))
        if words and before and len(words & before) / len(words | before) >= 0.8:
            return 'repeat'
    return None


def clean_history(history):
    cleaned = []
    for item in (history or [])[-MAX_HISTORY:]:
        if not isinstance(item, dict) or item.get('role') not in ('user', 'assistant'):
            continue
        content = str(item.get('content', ''))[:MAX_MESSAGE_CHARS].replace(MARKER, '').strip()
        if content:
            cleaned.append({'role': item['role'], 'content': content})
    return cleaned


def analysis_context(result):
    """Read-only facts about the current analysis, so the assistant can explain what the user sees."""
    if not result:
        return 'Δεν έχει ολοκληρωθεί ακόμη ανάλυση σε αυτή τη συνεδρία.'
    lines = []
    report = result.get('report') or {}
    if report.get('demo'):
        lines.append('Η τρέχουσα ανάλυση έγινε με demo δεδομένα.')
    for p in result.get('profiles') or []:
        share = p.get('revenue_share')
        stability, confidence = p.get('stability'), p.get('confidence')
        lines.append(
            f"- {p.get('profile_id')}: «{p.get('title')}» · {p.get('customer_count')} πελάτες"
            + (f" · {round(share * 100)}% του τζίρου" if isinstance(share, (int, float)) else '')
            + (f" · σταθερότητα {round(stability * 100)}%" if isinstance(stability, (int, float)) else '')
            + (f" · εμπιστοσύνη προσαρμογής {round(confidence * 100)}%" if isinstance(confidence, (int, float)) else ''))
    for m in ((report.get('validation') or {}).get('metrics') or []):
        if isinstance(m.get('recall'), (int, float)):
            lines.append(f"- Recall@{m.get('k')}: {round(m['recall'] * 100)}% "
                         f"(baseline {round((m.get('baseline_recall') or 0) * 100)}%)")
    return 'Προφίλ της τρέχουσας ανάλυσης:\n' + '\n'.join(lines) if lines else 'Η ανάλυση δεν βρήκε προφίλ.'


def messages_for(message, history, context):
    system = SYSTEM + '\nΣΤΟΙΧΕΙΑ ΤΗΣ ΑΝΑΛΥΣΗΣ\n' + context
    return [{'role': 'system', 'content': system}, *clean_history(history), {'role': 'user', 'content': message}]


def _event(kind, **fields):
    return json.dumps({'type': kind, **fields}, ensure_ascii=False) + '\n'


def _fixed(text, **meta):
    yield _event('meta', ai=False, **meta)
    yield _event('token', text=text)
    yield _event('done')


def stream_reply(message, history, context, analysis_running=False, model=DEFAULT_LLM, client_factory=None):
    """Yield NDJSON events: meta (escalate/ai/model), token (text), done. Never raises mid-stream."""
    message = str(message or '').strip()[:MAX_MESSAGE_CHARS]
    history = clean_history(history)
    reason = escalation_reason(message, history)
    escalate = {'escalate': bool(reason), 'reason': REASONS.get(reason)} if reason else {'escalate': False}
    if is_credit_question(message):
        yield from _fixed(REFUSAL_CREDIT, **escalate)
        return
    if reason == 'human':
        yield from _fixed(HANDOFF, **escalate)
        return
    if analysis_running:
        yield from _fixed(BUSY_ANALYSIS, **escalate)
        return
    if not _chat_lock.acquire(blocking=False):
        yield from _fixed(BUSY_CHAT, escalate=False)
        return
    try:
        payload = {'model': model, 'messages': messages_for(message, history, context), 'stream': True,
                   'keep_alive': '5m', 'options': GENERATION}
        if model.startswith('qwen3'):
            payload['think'] = False
        factory = client_factory or (lambda: httpx.Client(trust_env=False, timeout=httpx.Timeout(120, connect=3)))
        text = ''
        sent_meta = started = False
        try:
            with factory() as client, client.stream('POST', OLLAMA + '/api/chat', json=payload) as response:
                if response.status_code != 200:
                    raise httpx.HTTPError(f'Ollama returned {response.status_code}')
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = (chunk.get('message') or {}).get('content', '')
                    text += piece
                    # Hold the first characters back until we know whether the reply opens with the marker.
                    if not sent_meta and (len(text.lstrip()) >= len(MARKER) or chunk.get('done')):
                        flagged = text.lstrip().startswith(MARKER)
                        if flagged and not reason:
                            escalate = {'escalate': True, 'reason': REASONS['model']}
                        yield _event('meta', ai=True, model=model, **escalate)
                        sent_meta = True
                        piece = text.replace(MARKER, '').lstrip()
                    elif sent_meta:
                        piece = piece.replace(MARKER, '')
                    else:
                        continue
                    if not started:  # no leading blank space once the marker is removed
                        piece = piece.lstrip()
                        started = bool(piece)
                    if piece:
                        yield _event('token', text=piece)
                    if chunk.get('done'):
                        break
            if not sent_meta:
                yield _event('meta', ai=True, model=model, **escalate)
            if MARKER in text and not escalate['escalate']:
                yield _event('meta', ai=True, model=model, escalate=True, reason=REASONS['model'])
            yield _event('done')
        except (httpx.HTTPError, json.JSONDecodeError, OSError):
            if sent_meta:
                yield _event('token', text='\n\n(Η απάντηση διακόπηκε: το τοπικό μοντέλο δεν απάντησε.)')
                yield _event('meta', ai=True, model=model, escalate=True, reason=REASONS['unavailable'])
                yield _event('done')
            else:
                yield from _fixed(UNAVAILABLE, escalate=True, reason=REASONS['unavailable'])
    finally:
        _chat_lock.release()
