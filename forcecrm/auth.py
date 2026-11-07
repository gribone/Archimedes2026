"""Local demo login: users, scrypt password hashes and session cookies, all in the SQLite store.

Mock-account policy (demo polish, not an identity system): a username that does not exist yet is
created on its first sign-in with whatever password was typed; an existing username must match its
stored password. Nothing leaves the machine; there is no external identity provider.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone

COOKIE = 'forcecrm_session'
SESSION_DAYS = 7
USERNAME_MAX, PASSWORD_MIN, PASSWORD_MAX = 64, 1, 256
_SCRYPT = {'n': 2 ** 14, 'r': 8, 'p': 1}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TEXT NOT NULL, reason TEXT, transcript TEXT NOT NULL);
"""


class AuthError(ValueError):
    pass


def ensure_schema(con):
    con.executescript(SCHEMA)
    columns = {row[1] for row in con.execute('PRAGMA table_info(users)')}
    if 'tutorial_done_at' not in columns:  # added after T2 created the users table
        con.execute('ALTER TABLE users ADD COLUMN tutorial_done_at TEXT')
    if 'last_login_at' not in columns:
        con.execute('ALTER TABLE users ADD COLUMN last_login_at TEXT')


def _now():
    return datetime.now(timezone.utc)


def normalize_username(username):
    name = unicodedata.normalize('NFC', str(username or '')).strip()
    if not name or len(name) > USERNAME_MAX or any(unicodedata.category(c).startswith('C') for c in name):
        raise AuthError(f'Το όνομα χρήστη πρέπει να έχει 1–{USERNAME_MAX} χαρακτήρες.')
    return name


def hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    b64 = lambda raw: base64.b64encode(raw).decode()
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(digest)}"


def verify_password(password, stored):
    try:
        _, n, r, p, salt, digest = stored.split('$')
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                                dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def _user(row):
    return {'user_id': row['user_id'], 'username': row['username'], 'created_at': row['created_at'],
            'tutorial_done': bool(row['tutorial_done_at'])}


def login(con, username, password):
    """Return (user, session_token, created). Creates the account if the username is new."""
    ensure_schema(con)
    name = normalize_username(username)
    password = str(password or '')
    if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        raise AuthError('Συμπληρώστε κωδικό πρόσβασης.')
    row = con.execute('SELECT * FROM users WHERE username = ?', (name,)).fetchone()
    created = row is None
    if created:
        try:
            con.execute('INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                        (name, hash_password(password), _now().isoformat()))
        except sqlite3.IntegrityError:  # created concurrently by another request
            created = False
        row = con.execute('SELECT * FROM users WHERE username = ?', (name,)).fetchone()
    if not created and not verify_password(password, row['password_hash']):
        raise AuthError('Λάθος κωδικός για αυτό το όνομα χρήστη.')
    token = secrets.token_urlsafe(32)
    now = _now()
    con.execute('DELETE FROM sessions WHERE expires_at < ?', (now.isoformat(),))
    con.execute('INSERT INTO sessions VALUES (?, ?, ?, ?)', (_token_hash(token), row['user_id'], now.isoformat(),
                (now + timedelta(days=SESSION_DAYS)).isoformat()))
    con.execute('UPDATE users SET last_login_at = ? WHERE user_id = ?', (now.isoformat(), row['user_id']))
    con.commit()
    return _user(row), token, created


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def session_user(con, token):
    if not token:
        return None
    ensure_schema(con)
    row = con.execute('SELECT u.* FROM sessions s JOIN users u ON u.user_id = s.user_id '
                      'WHERE s.token_hash = ? AND s.expires_at > ?', (_token_hash(token), _now().isoformat())).fetchone()
    return _user(row) if row else None


def logout(con, token):
    if token:
        ensure_schema(con)
        con.execute('DELETE FROM sessions WHERE token_hash = ?', (_token_hash(token),))
        con.commit()


def set_tutorial_done(con, user_id, done=True):
    ensure_schema(con)
    con.execute('UPDATE users SET tutorial_done_at = ? WHERE user_id = ?',
                (_now().isoformat() if done else None, user_id))
    con.commit()


def create_ticket(con, user_id, reason, transcript):
    ensure_schema(con)
    cursor = con.execute('INSERT INTO support_tickets (user_id, created_at, reason, transcript) VALUES (?, ?, ?, ?)',
                         (user_id, _now().isoformat(), reason, transcript))
    con.commit()
    return cursor.lastrowid
