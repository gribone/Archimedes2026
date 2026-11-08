import json

import httpx
import pytest

from forcecrm import assistant, auth, db

H = {'X-ForceCRM': 'local'}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from forcecrm import server
    monkeypatch.setenv('FORCECRM_AUTH', '1')
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    with TestClient(server.app) as c:
        yield c


def test_password_hash_roundtrip():
    stored = auth.hash_password('μυστικό')
    assert stored.startswith('scrypt$') and 'μυστικό' not in stored
    assert auth.verify_password('μυστικό', stored)
    assert not auth.verify_password('άλλο', stored)
    assert not auth.verify_password('x', 'garbage')


def test_new_username_creates_account_existing_checks_password(tmp_path):
    con = db.connect(tmp_path)
    user, token, created = auth.login(con, '  Μαρία ', 'pw1')
    assert created and user['username'] == 'Μαρία' and not user['tutorial_done']
    assert auth.session_user(con, token)['user_id'] == user['user_id']
    _, _, created = auth.login(con, 'Μαρία', 'pw1')
    assert not created
    with pytest.raises(auth.AuthError):
        auth.login(con, 'Μαρία', 'wrong')
    with pytest.raises(auth.AuthError):
        auth.login(con, '', 'pw')
    auth.set_tutorial_done(con, user['user_id'])
    assert auth.session_user(con, token)['tutorial_done']
    auth.logout(con, token)
    assert auth.session_user(con, token) is None


def test_api_requires_login(client):
    assert client.get('/api/runs').status_code == 401
    assert client.get('/api/auth/me').status_code == 401
    page = client.get('/', follow_redirects=False)
    assert page.status_code == 303 and page.headers['location'] == '/login.html'
    assert client.get('/login.html').status_code == 200

    r = client.post('/api/auth/login', data={'username': 'demo', 'password': 'a'}, headers=H)
    assert r.status_code == 200 and r.json()['created']
    cookie = r.headers['set-cookie'].lower()
    assert 'httponly' in cookie and 'samesite=strict' in cookie
    assert client.get('/api/auth/me').json()['user']['tutorial_done'] is False
    assert client.get('/api/runs').status_code == 200
    assert client.get('/', follow_redirects=False).status_code == 200

    assert client.post('/api/auth/tutorial', data={'done': 'true'}, headers=H).status_code == 200
    assert client.get('/api/auth/me').json()['user']['tutorial_done'] is True

    client.post('/api/auth/logout', headers=H)
    assert client.get('/api/runs').status_code == 401
    assert client.post('/api/auth/login', data={'username': 'demo', 'password': 'b'}, headers=H).status_code == 401
    r = client.post('/api/auth/login', data={'username': 'demo', 'password': 'a'}, headers=H)
    assert r.status_code == 200 and not r.json()['created']


def test_support_ticket_is_stored(client):
    client.post('/api/auth/login', data={'username': 'u', 'password': 'p'}, headers=H)
    r = client.post('/api/support/tickets', headers=H,
                    json={'reason': 'test', 'transcript': [{'role': 'user', 'content': 'βοήθεια'}]})
    assert r.status_code == 201 and r.json()['contact']['email']
    from forcecrm import server
    con = db.connect(server.ROOT)
    row = con.execute('SELECT * FROM support_tickets').fetchone()
    assert json.loads(row['transcript'])[0]['content'] == 'βοήθεια' and row['user_id'] is not None


def events(stream):
    return [json.loads(line) for line in stream]


def text_of(evts):
    return ''.join(e['text'] for e in evts if e['type'] == 'token')


def fake_ollama(chunks, status=200):
    def handler(request):
        body = '\n'.join(json.dumps({'message': {'content': c}, 'done': i == len(chunks) - 1})
                         for i, c in enumerate(chunks))
        return httpx.Response(status, text=body)
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def never_called():
    raise AssertionError('the model must not be called')


def test_credit_questions_are_refused_without_the_model():
    evts = events(assistant.stream_reply('Είναι φερέγγυα η επιχείρηση 123;', [], '', client_factory=never_called))
    assert evts[0]['ai'] is False and 'φερεγγυότητα' in text_of(evts)


def test_busy_analysis_does_not_call_the_model():
    evts = events(assistant.stream_reply('Τι είναι το lift;', [], '', analysis_running=True, client_factory=never_called))
    assert 'ανάλυση' in text_of(evts)


def test_rules_escalate_without_the_model():
    assert assistant.escalation_reason('Θέλω να μιλήσω με ΆΝΘΡΩΠΟ', []) == 'human'
    assert assistant.escalation_reason('Το αρχείο βγάζει λάθος', []) == 'problem'
    history = [{'role': 'user', 'content': 'Πώς ανεβάζω το αρχείο Excel;'}]
    assert assistant.escalation_reason('πώς ανεβάζω το αρχείο excel', history) == 'repeat'
    assert assistant.escalation_reason('Τι σημαίνει σταθερότητα;', history) is None


def test_human_request_gets_fixed_handoff_without_the_model():
    evts = events(assistant.stream_reply('Θέλω να μιλήσω με έναν άνθρωπο', [], '', client_factory=never_called))
    assert evts[0]['escalate'] and evts[0]['ai'] is False and 'υποστήριξης' in text_of(evts)


def test_model_reply_streams_and_marker_escalates():
    evts = events(assistant.stream_reply('Τι είναι το lift;', [], '', client_factory=fake_ollama(['Το lift ', 'δείχνει…'])))
    meta = [e for e in evts if e['type'] == 'meta']
    assert meta[0]['ai'] and not meta[0]['escalate'] and text_of(evts) == 'Το lift δείχνει…'
    evts = events(assistant.stream_reply('Πότε βγαίνει η νέα έκδοση;', [], '',
                                         client_factory=fake_ollama(['[ΥΠΟ', 'ΣΤΗΡΙΞΗ] Δεν ', 'το γνωρίζω.'])))
    assert evts[0]['escalate'] and text_of(evts) == 'Δεν το γνωρίζω.'


def test_unavailable_model_hands_off_to_support():
    evts = events(assistant.stream_reply('Γεια', [], '', client_factory=fake_ollama([], status=500)))
    assert evts[0]['escalate'] and 'Ollama' in text_of(evts)


def test_analysis_context_uses_only_profile_facts():
    text = assistant.analysis_context({'profiles': [{'profile_id': 'ICP-01', 'title': 'Νησιά', 'customer_count': 5,
                                                     'revenue_share': 0.25, 'stability': 0.93, 'confidence': 0.5}],
                                       'report': {}})
    assert 'ICP-01' in text and '25%' in text and '93%' in text
    assert 'Δεν έχει' in assistant.analysis_context(None)


def test_reply_has_no_leading_space_after_marker():
    evts = events(assistant.stream_reply('Υπάρχει εφαρμογή για κινητά;', [], '',
                                         client_factory=fake_ollama(['[ΥΠΟΣΤΗΡΙΞΗ]', '\n ', ' Όχι ακόμη.'])))
    assert evts[0]['escalate'] and text_of(evts) == 'Όχι ακόμη.'


def test_demo_login_needs_no_account_and_cannot_be_used_from_the_form(tmp_path):
    import sqlite3
    from forcecrm import auth
    con = sqlite3.connect(tmp_path / 'demo.sqlite')
    con.row_factory = sqlite3.Row
    con.execute('CREATE TABLE users (user_id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, '
                'password_hash TEXT NOT NULL, created_at TEXT NOT NULL)')
    user, token = auth.demo_login(con)
    assert user['username'] == auth.DEMO_USERNAME and user['tutorial_done'] is True
    assert auth.session_user(con, token)['user_id'] == user['user_id']
    again, _ = auth.demo_login(con)
    assert again['user_id'] == user['user_id']  # reused, not a new account each time
    import pytest
    with pytest.raises(auth.AuthError):  # no password ever opens the demo user through the form
        auth.login(con, auth.DEMO_USERNAME, auth.DEMO_PASSWORD_MARKER)
