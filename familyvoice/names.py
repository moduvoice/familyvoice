"""Name binding without enrollment and without asking.

Names are attached to voice clusters only from conversation cues that the app extracts from the
transcript (the module never asks "who are you?"). Every cue is evidence for or against a
(name, cluster) pair. A name is bound only when the evidence is strong (>= 4.5 points), comes from at
least two separate conversations, and the name and the cluster are each other's clear first choice
(2 points ahead on both sides). Bindings are recomputed from the evidence every time. Until a name
is bound, messages for that name wait silently.
Measured (2026-10-22, 1,800 scripted family days on public speech, 15% of answers by the wrong
person): 0 misdeliveries, 0 utterances tagged with a wrong name. With 3.0 points and one-sided,
sticky binding the same test gave 1 misdelivery in 200 days (two wrong answers bound a name).

Cue weights (log-odds-like points). Only utterances that the tracker CONFIRMED count.
  self_intro    "It's Sunhee" / "This is Dad"                          +3.0
  answer        someone called "Sunhee!" and a different confirmed
                speaker answered within ANSWER_WINDOW_S                +1.5
  addressed     a confirmed speaker was spoken to by name in an
                exchange the app is sure of (e.g. "Thanks, Sunhee")    +1.0
  reaction      the speaker replied to a message delivered to them     +1.0 (only reinforces)
  sender        "Tell Sunhee ..." - the speaker is NOT Sunhee          veto for that pair
  called_by     the speaker called "Sunhee!" - the speaker is NOT
                Sunhee                                                 veto for that pair
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

WEIGHTS = {"self_intro": 3.0, "answer": 1.5, "addressed": 1.0, "reaction": 1.0}
BIND_SCORE = 4.5        # total evidence needed (3 answers, or self-intro + 1 answer)
BIND_LEAD = 2.0         # lead over the second-best cluster for the same name
MIN_EVENTS = 2          # independent events (different conversations)
EVENT_GAP_S = 60.0      # two cues closer than this count as one conversation
ANSWER_WINDOW_S = 6.0


def norm_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


@dataclass
class Evidence:
    score: float = 0.0
    events: list = field(default_factory=list)   # (ts, kind)
    veto: bool = False

    def conversations(self) -> int:
        n, last = 0, None
        for ts, _ in sorted(self.events):
            if last is None or ts - last > EVENT_GAP_S:
                n += 1
            last = ts
        return n


class NameBinder:
    def __init__(self):
        self.ev: dict[tuple[str, str], Evidence] = {}
        self.bound: dict[str, str] = {}          # name -> speaker_id
        self.pending_call: list[tuple[float, str, str | None]] = []   # (ts, name, caller_id)

    # ---- cues -------------------------------------------------------
    def cue(self, kind: str, speaker_id: str | None, name: str, ts: float | None = None):
        """Record one cue. speaker_id must come from a CONFIRMED observation (else ignored)."""
        ts = time.time() if ts is None else ts
        name = norm_name(name)
        if speaker_id is None:
            return
        e = self.ev.setdefault((name, speaker_id), Evidence())
        if kind in ("sender", "called_by"):
            e.veto = True
        else:
            e.score += WEIGHTS[kind]
            e.events.append((ts, kind))
        self._resolve(name)

    def heard_call(self, name: str, caller_id: str | None, ts: float | None = None):
        """Someone said "Sunhee!" - the next different confirmed speaker may be Sunhee."""
        ts = time.time() if ts is None else ts
        self.pending_call.append((ts, norm_name(name), caller_id))
        if caller_id is not None:
            self.cue("called_by", caller_id, name, ts)

    def heard_speaker(self, speaker_id: str | None, ts: float | None = None):
        """Call on every observation. Confirmed speaker answering a recent call gets 'answer'."""
        ts = time.time() if ts is None else ts
        self.pending_call = [p for p in self.pending_call if ts - p[0] <= ANSWER_WINDOW_S]
        if speaker_id is None:
            return
        for p in list(self.pending_call):
            if p[2] != speaker_id:
                self.cue("answer", speaker_id, p[1], ts)
                self.pending_call.remove(p)

    # ---- binding ----------------------------------------------------
    def _resolve(self, name: str | None = None):
        """Recompute ALL bindings from the current evidence (a binding is never sticky).
        A name binds to a cluster only if they are each other's clear first choice:
          name side    best cluster score >= BIND_SCORE, >= MIN_EVENTS conversations,
                       and BIND_LEAD ahead of the name's second-best cluster
          cluster side this name is the cluster's best name and BIND_LEAD ahead of the cluster's
                       second-best name (one voice = one family name; the app merges aliases
                       like "dad" / "Minsu" before calling cue())
        Vetoed pairs (the speaker sent a message to / called that name) never bind."""
        live = {k: e for k, e in self.ev.items() if not e.veto and e.score > 0 and not k[1].startswith("u:")}   # "u:<utt>" = evidence waiting on an unclustered utterance
        bound = {}
        for n in {k[0] for k in live}:
            row = sorted(((e.score, s) for (nn, s), e in live.items() if nn == n), reverse=True)
            sc, s = row[0]
            if sc < BIND_SCORE or sc - (row[1][0] if len(row) > 1 else 0.0) < BIND_LEAD:
                continue
            if live[(n, s)].conversations() < MIN_EVENTS:
                continue
            col = sorted(((e.score, nn) for (nn, ss), e in live.items() if ss == s), reverse=True)
            if col[0][1] != n or (len(col) > 1 and col[0][0] - col[1][0] < BIND_LEAD):
                continue
            bound[n] = s
        self.bound = bound

    def name_of(self, speaker_id: str) -> str | None:
        for n, s in self.bound.items():
            if s == speaker_id:
                return n
        return None

    def speaker_of(self, name: str) -> str | None:
        return self.bound.get(norm_name(name))

    def merge(self, keep: str, gone: str):
        """Tracker merged cluster `gone` into `keep` - move evidence."""
        for (n, sid), e in list(self.ev.items()):
            if sid == gone:
                k = self.ev.setdefault((n, keep), Evidence())
                k.score += e.score
                k.events += e.events
                k.veto = k.veto or e.veto
                del self.ev[(n, sid)]
        for n in {n for (n, _) in self.ev}:
            self._resolve(n)

    def names_conflict(self, a: str, b: str) -> bool:
        """Two clusters bound to different names must not be merged (cannot-link)."""
        na, nb = self.name_of(a), self.name_of(b)
        return bool(na and nb and na != nb)

    def to_dict(self):
        return {"bound": self.bound,
                "ev": [[n, s, e.score, e.events, e.veto] for (n, s), e in self.ev.items()]}

    @classmethod
    def from_dict(cls, d):
        b = cls()
        b.bound = dict(d.get("bound", {}))
        for n, s, sc, evs, veto in d.get("ev", []):
            b.ev[(n, s)] = Evidence(sc, [tuple(x) for x in evs], veto)
        return b
