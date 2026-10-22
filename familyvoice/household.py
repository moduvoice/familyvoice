"""One object for the app: audio in -> who spoke (or silence) + messages to say.

    hh = Household.create(model="campp")          # thresholds + embedder for that model
    r = hh.hear(pcm16k, ts=time.time(), transcript="tell sunhee to eat", cues=[...])
    r -> {"speaker_id", "state", "name", "score", "deliver": [Message, ...]}

The app (ASR + LLM) turns the transcript into cues:
    {"kind": "self_intro", "name": "sunhee"}            "It's Sunhee"
    {"kind": "call", "name": "sunhee"}                  "Sunhee!" (vocative)
    {"kind": "addressed", "name": "sunhee"}             "Thanks, Sunhee"
    {"kind": "send", "to": "sunhee", "text": "..."}     "Tell Sunhee to eat"
Cues are applied only when this utterance's speaker is CONFIRMED (except 'call', whose answer is
checked on the following utterances, and 'send', which is always stored - the sender name is
filled in later if the sender is not yet known).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .names import NameBinder
from .relay import Mailbox
from .tracker import VoiceTracker

HERE = Path(__file__).resolve().parent


class Household:
    def __init__(self, tracker: VoiceTracker, embedder=None):
        self.tracker = tracker
        self.names = NameBinder()
        self.mail = Mailbox()
        self.embedder = embedder
        tracker.on_merge = self.names.merge
        tracker.can_merge = lambda a, b: not self.names.names_conflict(a, b)

    @classmethod
    def create(cls, model: str = "r34", level: str = "strict", embedder=True, profile: str = "server", **kw):
        th = json.loads((HERE / "thresholds" / f"{model}.json").read_text())
        t = th["levels"][level]
        t.update(bins=th["bins"], mean=th.get("mean"))
        emb = None
        if embedder:
            from .embedder import load_embedder
            emb = load_embedder(model, profile=profile)
        kw.setdefault("confirm_on_form", True)   # the utterance that completes a new voice group counts as confirmed (measured: 0 misdeliveries)
        return cls(VoiceTracker(t, **kw), emb)

    def hear(self, audio=None, ts: float | None = None, emb=None, voiced_s: float | None = None,
             cues: list[dict] | None = None) -> dict:
        ts = time.time() if ts is None else ts
        if emb is None:
            emb, voiced_s = self.embedder(audio)
        r = self.tracker.observe(emb, voiced_s, ts)
        sid = r["speaker_id"]
        if r.get("formed"):     # evidence that waited on the utterances of a new group follows them into the cluster
            for u in r["formed_from"]:
                self.names.merge(r["formed"], "u:" + u)
                for m in self.mail.msgs.values():
                    if m.from_speaker == "u:" + u:
                        m.from_speaker = r["formed"]
        key = sid or ("u:" + r["utt_id"])   # unconfirmed utterances still collect cues (never bound until clustered)
        self.names.heard_speaker(key, ts)
        for c in cues or []:
            k = c["kind"]
            if k == "call":
                self.names.heard_call(c["name"], key, ts)
            elif k == "send":
                self.mail.post(c["to"], c["text"], key, self.names.name_of(sid) if sid else None, ts)
                self.names.cue("sender", key, c["to"], ts)
            else:
                self.names.cue(k, key, c["name"], ts)
        name = self.names.name_of(sid) if sid else None
        deliver = []
        if r["state"] == "confirmed" and name:
            for m in self.mail.due(name, ts):
                if m.from_name is None and m.from_speaker:
                    m.from_name = self.names.name_of(m.from_speaker)
                self.mail.mark_delivered(m.id, sid, ts)
                deliver.append(m)
        r.update(name=name, deliver=deliver)
        return r
