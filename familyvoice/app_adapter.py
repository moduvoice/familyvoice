"""Adapter for the Family Relay app (work/app/relay/speaker): FR_SPEAKER_MODULE=familyvoice.app_adapter:FamilyVoiceSpeaker

    FamilyVoiceSpeaker(household, store)
    observe(audio_bytes, ts) -> {speaker_id, state, score, name, discarded, viz_xy, ...}
    note_transcript(result, transcript)   name cues from the STT text (module decides names; app never binds)
    bind_name(speaker_id, name, evidence) kept for the protocol; treated as one strong cue, not an override
    speakers(), points()

Environment: FV_MODEL (r34 = ResNet34-LM default | campp = CAM++-LM), FV_PROFILE (server = fp32 ONNX default | device = int8 ONNX built on first use),
FV_LEVEL (strict), FV_CONFIRM_ON_FORM (1 = the utterance that completes a new voice group is confirmed).
State: voice clusters are kept in memory and written to the app store (one item per cluster, float16, < 100 KB),
the unconfirmed pool lives in memory only. Audio: 16 kHz WAV is read directly; anything else (webm/opus) goes
through ffmpeg (must be on PATH).
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .names import NameBinder
from .tracker import VoiceTracker

HERE = Path(__file__).resolve().parent
SR = 16000
STOP = {"okay", "ok", "here", "home", "back", "fine", "good", "sorry", "done", "heading", "going", "ready", "hungry",
        "tired", "late", "not", "so", "just", "the", "a", "an", "it", "that", "this", "you", "me", "him", "her", "them",
        "thanks", "thank", "morning", "hi", "hey", "hello", "yes", "no", "please", "alexa", "what", "how", "why", "when"}


def decode(audio: bytes) -> np.ndarray:
    if audio[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio)) as w:
            if w.getframerate() == SR and w.getsampwidth() == 2:
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
                return x.reshape(-1, w.getnchannels()).mean(1)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                       input=audio, capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768


def name_cues(text: str) -> list[dict]:
    """English/Korean surface cues. Capitalised words only for English (STT keeps names capitalised)."""
    t = text.strip()
    cues = []
    for m in re.finditer(r"\b(?:I'm|I am|it's|It's|this is|This is)\s+([A-Z][a-z]+)\b", t):
        if m.group(1).lower() not in STOP:
            cues.append({"kind": "self_intro", "name": m.group(1)})
    m = re.match(r"^(?:(?:hey|hi|good morning|morning|thanks|thank you)[, ]+)?([A-Z][a-z]+)\s*[,!?]", t, re.I)
    if m and m.group(1).lower() not in STOP and m.group(1)[0].isupper():
        cues.append({"kind": "call", "name": m.group(1)})
    m = re.search(r",\s*([A-Z][a-z]+)\s*[.!?]*$", t)
    if m and m.group(1).lower() not in STOP:
        cues.append({"kind": "call", "name": m.group(1)})
    for m in re.finditer(r"\b(?:tell|ask|remind)\s+([A-Z][a-z]+)\b", t, re.I):
        if m.group(1).lower() not in STOP and m.group(1)[0].isupper():
            cues.append({"kind": "send", "name": m.group(1)})
    m = re.match(r"^([가-힣]{1,4}?)(?:야|아)[,!\s]", t)       # 순희야, ...
    if m:
        cues.append({"kind": "call", "name": m.group(1)})
    m = re.search(r"(?:저|나)\s*([가-힣]{1,4}?)(?:예요|이에요|야|이야|입니다)", t)
    if m:
        cues.append({"kind": "self_intro", "name": m.group(1)})
    return cues


class FamilyVoiceSpeaker:
    def __init__(self, household: str, store=None):
        self.household, self.store = household, store
        self.pk = f"household#{household}"
        self.model = os.environ.get("FV_MODEL", "r34")
        self.profile = os.environ.get("FV_PROFILE", "server")
        th = json.loads((HERE / "thresholds" / f"{self.model}.json").read_text())
        t = dict(th["levels"][os.environ.get("FV_LEVEL", "strict")])
        t.update(bins=th["bins"], mean=th["mean"])
        self.tracker = VoiceTracker(t, confirm_on_form=os.environ.get("FV_CONFIRM_ON_FORM", "1") == "1")
        self.names = NameBinder()
        self.tracker.on_merge = self.names.merge
        self.tracker.can_merge = lambda a, b: not self.names.names_conflict(a, b)
        from .embedder import load_embedder
        self.embed = load_embedder(self.model, profile=self.profile, threads=int(os.environ.get("FV_THREADS", "2")))
        rp = np.random.default_rng(7).standard_normal((len(t["mean"]), 2)).astype(np.float32)
        self.proj = rp / np.linalg.norm(rp, axis=0, keepdims=True)
        self.meta: dict[str, dict] = {}
        self._points: list[dict] = []
        self._load()

    # ---------------------------------------------------------- persistence (app store)
    def _load(self):
        if not self.store:
            return
        for r in self.store.query(self.pk, "fv#cl#"):
            mem = np.frombuffer(base64.b64decode(r["mem"]), np.float16).reshape(-1, r["dim"]).astype(np.float32)
            self.tracker.clusters[r["id"]] = {"mem": list(mem), "added": r["added"], "created": r["created"], "last": r["last"]}
            self.tracker._cent(r["id"])
            self.meta[r["id"]] = r.get("meta", {})
        nm = self.store.get(self.pk, "fv#names")
        if nm:
            self.names = NameBinder.from_dict(nm["d"])
            self.tracker.on_merge = self.names.merge

    def _save(self, cid):
        if not self.store:
            return
        c = self.tracker.clusters.get(cid)
        if c is None:
            return
        m = np.stack(c["mem"]).astype(np.float16)
        self.store.put(self.pk, f"fv#cl#{cid}", {"id": cid, "dim": int(m.shape[1]), "mem": base64.b64encode(m.tobytes()).decode(),
                                                  "added": c["added"], "created": c["created"], "last": c["last"],
                                                  "meta": self.meta.get(cid, {})})
        self.store.put(self.pk, "fv#names", {"d": self.names.to_dict()})

    # ---------------------------------------------------------- protocol
    def observe(self, audio_bytes: bytes, ts: float) -> dict:
        t0 = time.time()
        when = datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")
        try:
            wav = decode(audio_bytes)
        except Exception as e:  # undecodable audio: no voice
            return {"speaker_id": None, "state": "unknown", "score": 0.0, "name": None, "discarded": True, "reason": f"decode: {e}"[:120]}
        emb, voiced = self.embed(wav)
        r = self.tracker.observe(emb, voiced, ts)
        sid = r["speaker_id"]
        if r.get("formed"):   # cues that waited on the utterances of a new group follow them into the cluster
            for u in r["formed_from"]:
                self.names.merge(r["formed"], "u:" + u)
        key = sid or ("u:" + r["utt_id"])
        self.names.heard_speaker(key, ts)
        self._last_ts = ts
        x = emb - np.asarray(self.tracker.mean) if self.tracker.mean is not None else emb
        x = x / (np.linalg.norm(x) + 1e-9)
        xy = (x @ self.proj)
        xy = [round(float(1 / (1 + math.exp(-3 * v))), 4) for v in xy]
        discarded = r.get("why") == "short"      # junk (too short) never enters the pool; other unknowns wait in the pool
        if sid:
            m = self.meta.setdefault(sid, {"first_heard": when, "utterances": 0})
            m["utterances"] += 1
            m["last_heard"], m["last_score"] = when, round(float(r["score"]), 3)
            self._save(sid)
        self._points = (self._points + [{"ts": when, "x": xy[0], "y": xy[1], "speaker_id": sid, "state": r["state"], "discarded": discarded}])[-120:]
        name = self.names.name_of(sid) if sid else None
        return {"speaker_id": sid, "state": r["state"], "score": round(float(r["score"] or 0.0), 4), "name": name.title() if name else None,
                "discarded": discarded, "viz_xy": xy, "voiced_s": round(voiced, 2), "margin": r.get("margin"),
                "nearest": r.get("nearest"), "utt_id": r["utt_id"], "module": f"familyvoice/{self.model}/{self.profile}", "ms": round((time.time() - t0) * 1000, 1)}

    def note_transcript(self, result: dict, transcript: str):
        """Cues from the STT text. Unconfirmed utterances keep their cues under "u:<utt_id>" until they join a cluster."""
        sid = result.get("speaker_id") if result.get("state") == "confirmed" else None
        key = sid or ("u:" + result["utt_id"] if result.get("utt_id") else None)
        if key is None:
            return
        ts = getattr(self, "_last_ts", time.time())
        for c in name_cues(transcript or ""):
            if c["kind"] == "call":
                self.names.heard_call(c["name"], key, ts)
            elif c["kind"] == "send":
                self.names.cue("sender", key, c["name"], ts)
            else:
                self.names.cue(c["kind"], key, c["name"], ts)
        if sid:
            self._save(sid)

    def bind_name(self, speaker_id: str, name: str, evidence: str) -> dict:
        self.names.cue("self_intro", speaker_id, name)
        self._save(speaker_id)
        return {"speaker_id": speaker_id, "name": self.names.name_of(speaker_id), "evidence": evidence}

    def speakers(self) -> list[dict]:
        out = []
        for cid, c in self.tracker.clusters.items():
            m = self.meta.get(cid, {})
            cen = c["c"] @ self.proj
            out.append({"speaker_id": cid, "name": (self.names.name_of(cid) or "").title() or None, "utterances": m.get("utterances", len(c["mem"])),
                        "first_heard": m.get("first_heard"), "last_heard": m.get("last_heard"), "last_score": m.get("last_score"),
                        "center": [round(float(1 / (1 + math.exp(-3 * v))), 3) for v in cen]})
        return out

    def points(self) -> list[dict]:
        return list(self._points)
