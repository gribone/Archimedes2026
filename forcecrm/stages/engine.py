"""A small, explicit stage runner: named stages, Merkle cache keys, per-stage timing.

Each stage is a pure function of the artifacts of the stages it `needs` and of the run
parameters it declares. Its cache key is

    sha256(stage name, hash of the stage's source files, its parameter values,
           the cache keys of the stages it needs)

so a key changes exactly when something the stage can see changes. Large arrays are never
hashed: only the root inputs (CSV, portfolio and model fingerprints) are hashed once, and the
keys chain from there. Persisted artifacts live in `.cache/stages/<stage>-<key>.pkl`; they are
written and read only by this module, on this machine.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PACKAGE = Path(__file__).resolve().parents[1]
KEEP_PER_STAGE = 4  # most recent artifacts kept per stage; older ones are pruned


def sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def source_hash(files) -> str:
    """Hash of the code a stage runs, so editing the code invalidates its artifacts."""
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b'\0' + (PACKAGE / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Stage:
    name: str
    requirement: str            # brief requirement the stage implements, e.g. 'A1'
    needs: tuple[str, ...]
    params: tuple[str, ...]     # run parameters that change this stage's output
    code: tuple[str, ...]       # files under forcecrm/ whose source is part of the key
    run: Callable[..., Any]     # run(ctx) -> artifact
    persist: bool = True        # write the artifact to disk (False: cheap, or not picklable)
    message: str = ''           # progress text shown to the user
    deterministic: bool = True  # False only for the LLM stage; documented, not cached as a whole


@dataclass
class Record:
    stage: str
    key: str
    status: str                 # 'computed' | 'cached' | 'partial' (some sub-results reused)
    seconds: float


@dataclass
class Runner:
    """Resolves stages on demand.

    `params` are JSON-serialisable values that enter cache keys: run options and the root
    fingerprints (CSV, portfolio, model hashes). `objects` are the in-memory inputs those
    fingerprints stand for (the portfolio frame, the data folder); they never enter a key."""
    stages: dict[str, Stage]
    params: dict[str, Any]
    objects: dict[str, Any] = field(default_factory=dict)
    cache_dir: Path | None = None       # None disables the disk cache
    force: frozenset[str] = frozenset()  # stages recomputed even when an artifact exists
    progress: Callable[[str], None] = lambda message: None
    records: list[Record] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)  # stage details for report.json
    _memory: dict[str, Any] = field(default_factory=dict)
    _keys: dict[tuple, str] = field(default_factory=dict)

    def variant(self, **overrides) -> 'Runner':
        """Same runner with some parameters changed (e.g. default weights for a comparison).
        Shares the in-memory artifacts: stages whose keys do not change are not recomputed."""
        return Runner(self.stages, {**self.params, **overrides}, self.objects, self.cache_dir, self.force,
                      self.progress, self.records, self.notes, self._memory, {})

    def key(self, name: str) -> str:
        stage = self.stages[name]
        values = tuple(_freeze(self.params.get(p)) for p in stage.params)
        memo = (name, values)
        if memo not in self._keys:
            self._keys[memo] = sha256({'stage': name, 'code': source_hash(stage.code + ('stages/definitions.py',)),
                                       'params': {p: self.params.get(p) for p in stage.params},
                                       'needs': {n: self.key(n) for n in stage.needs}})
        return self._keys[memo]

    def get(self, name: str):
        """The artifact of `name`, computing (and caching) it and its dependencies as needed."""
        key = self.key(name)
        if key in self._memory:
            return self._memory[key]
        stage = self.stages[name]
        path = self.cache_dir / f'{name}-{key[:24]}.pkl' if self.cache_dir and stage.persist else None
        if path is not None and name not in self.force and path.exists():
            # Cache hit: the inputs are not needed, so upstream stages are not touched.
            start = time.perf_counter()
            with open(path, 'rb') as handle:
                artifact = pickle.load(handle)
            status = 'cached'
        else:
            inputs = {n: self.get(n) for n in stage.needs}
            start = time.perf_counter()
            if stage.message:
                self.progress(stage.message)
            context = Context(self, inputs)
            artifact = stage.run(context)
            status = context.status  # a stage with its own sub-cache may report 'cached' or 'partial'
            if path is not None:
                self._write(path, artifact)
        self.records.append(Record(name, key, status, time.perf_counter() - start))
        self._memory[key] = artifact
        return artifact

    def _write(self, path: Path, artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with open(temporary, 'wb') as handle:
            pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)  # atomic: a crash never leaves a half-written artifact
        stale = sorted(path.parent.glob(path.name.split('-')[0] + '-*.pkl'),
                       key=lambda p: p.stat().st_mtime, reverse=True)[KEEP_PER_STAGE:]
        for old in stale:
            old.unlink(missing_ok=True)

    def summary(self):
        """Per-stage records for report.json: key prefix, cache status and seconds."""
        return [{'stage': r.stage, 'key': r.key[:16], 'status': r.status, 'seconds': round(r.seconds, 3)}
                for r in self.records]


@dataclass
class Context:
    runner: Runner
    inputs: dict[str, Any]
    status: str = 'computed'

    def __getitem__(self, name):
        return self.inputs[name]

    def param(self, name):
        return self.runner.params.get(name)

    def obj(self, name):
        return self.runner.objects[name]

    @property
    def forced(self):
        return self.runner.force

    def progress(self, message):
        self.runner.progress(message)


def _freeze(value):
    return json.dumps(value, sort_keys=True, default=str)
