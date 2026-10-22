"""Enrollment-free, ask-free incremental speaker tracking.

Every voiced utterance is embedded and compared with the household's voice clusters.
  confirmed  best cluster score >= T_acc[duration bin] AND best - second >= margin
             -> the utterance joins that cluster (only confirmed utterances ever update a cluster,
                so a cluster never drifts toward someone else; the newest MAX_MEMBERS keep it current
                as a voice slowly changes - a cold, a growing child)
  tentative  close to a cluster but not sure -> stay silent
  unknown    not like anyone -> stay silent
Unconfirmed utterances wait in a pool. When at least K of them agree with each other
(every pair >= T_pair) they become a NEW cluster - or join an existing cluster if their centre is
clearly that cluster (>= T_merge and ahead of the runner-up). One-off voices (a visitor, TV, a
shout) never form a cluster and age out of the pool.
Clusters whose centres meet (>= T_merge) merge; a cluster that has grown into two clearly different
halves (< T_split) is split. Thresholds come from calibration on public data per model
(thresholds/<model>.json, 'strict' = 0.1% same-gender false-accept per comparison).
Principles taken from the dmdb face/voice clustering: only clean confident samples build a
centroid, runner-up margin beats absolute thresholds, junk is never clustered, and an empty answer
is better than a wrong one.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

MIN_VOICED_S = 1.0


def _n(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class VoiceTracker:
    def __init__(self, thresholds: dict, k: int = 3, margin: float = 0.05, max_members: int = 64,
                 pool_size: int = 60, pool_ttl_s: float = 7 * 24 * 3600, split_every: int = 16, confirm_on_form: bool = False):
        self.th = thresholds            # {"bins": [[lo,hi],..], "T_acc": [..], "T_tent": [..], "T_pair", "T_merge", "T_split", "mean": [...]}
        self.k, self.margin, self.max_members = k, margin, max_members
        self.pool_size, self.pool_ttl, self.split_every = pool_size, pool_ttl_s, split_every
        self.confirm_on_form = confirm_on_form   # the utterance that completes a new >=K mutually-consistent group counts as confirmed
        self.mean = np.asarray(thresholds.get("mean"), dtype=np.float32) if thresholds.get("mean") is not None else None
        self.clusters: dict[str, dict] = {}   # id -> {"mem": [vec], "c": vec, "added": int, "created": ts, "last": ts}
        self.pool: list[tuple[float, np.ndarray]] = []
        self.on_merge = None                  # callback(keep_id, gone_id) - e.g. NameBinder.merge
        self.can_merge = None                 # callback(a_id, b_id) -> bool - e.g. not NameBinder.names_conflict

    # ------------------------------------------------------------------
    def _bin(self, dur):
        for i, (lo, hi) in enumerate(self.th["bins"]):
            if dur < hi:
                return i
        return len(self.th["bins"]) - 1

    def _prep(self, emb):
        e = np.asarray(emb, dtype=np.float32)
        if self.mean is not None:
            e = e - self.mean
        return _n(e)

    def _ids_and_scores(self, x):
        ids = list(self.clusters)
        if not ids:
            return ids, np.zeros(0)
        return ids, np.stack([self.clusters[i]["c"] for i in ids]) @ x

    def _cent(self, cid):
        c = self.clusters[cid]
        c["c"] = _n(np.mean(np.stack(c["mem"]), 0))

    # ------------------------------------------------------------------
    def observe(self, emb, voiced_s: float, ts: float | None = None) -> dict:
        """emb: raw speaker embedding of one utterance. voiced_s: voiced seconds (VAD).
        Returns {speaker_id, state, score, margin, utt_id}; speaker_id is set only when confirmed. When this utterance completes
        a group, also {formed: cluster id, formed_from: [utt_id, ...]} so evidence attached to those utterances can follow them."""
        uid = uuid.uuid4().hex[:10]
        ts = time.time() if ts is None else ts
        if voiced_s < MIN_VOICED_S:
            return {"speaker_id": None, "state": "unknown", "score": None, "margin": None, "why": "short", "utt_id": uid}
        x = self._prep(emb)
        b = self._bin(voiced_s)
        ids, S = self._ids_and_scores(x)
        s1 = s2 = -1.0
        best = None
        if len(S):
            o = np.argsort(-S)
            best, s1 = ids[o[0]], float(S[o[0]])
            s2 = float(S[o[1]]) if len(S) > 1 else -1.0
            if s1 >= self.th["T_acc"][b] and s1 - s2 >= self.margin:
                c = self.clusters[best]
                c["mem"] = (c["mem"] + [x])[-self.max_members:]
                c["added"] += 1
                c["last"] = ts
                self._cent(best)
                best = self._maintain(best)
                return {"speaker_id": best, "state": "confirmed", "score": s1, "margin": s1 - s2, "utt_id": uid}
        self.pool = [p for p in self.pool if ts - p[0] <= self.pool_ttl][-(self.pool_size - 1):]
        self.pool.append((ts, x, uid))
        formed = self._try_form(ts)
        extra = {"utt_id": uid}
        if formed:
            extra.update(formed=formed, formed_from=self.formed_from)
        if formed and self.confirm_on_form:
            c = self.clusters[formed]
            return {"speaker_id": formed, "state": "confirmed", "score": float(c["c"] @ x), "margin": None, "why": "formed", **extra}
        state = "tentative" if best is not None and s1 >= self.th["T_tent"][b] else "unknown"
        return {"speaker_id": None, "state": state, "score": s1 if best else None,
                "margin": (s1 - s2) if best else None, "nearest": best, **extra}

    def _try_form(self, ts):
        if len(self.pool) < self.k:
            return
        P = np.stack([p[1] for p in self.pool])
        G = P @ P.T
        x = len(self.pool) - 1
        grp = [x]
        for j in np.argsort(-G[x]):
            if j != x and G[x, j] >= self.th["T_pair"] and all(G[j, g] >= self.th["T_pair"] for g in grp):
                grp.append(int(j))
        if len(grp) < self.k:
            return
        cen = _n(P[grp].mean(0))
        ids, S = self._ids_and_scores(cen)
        mem = [self.pool[g][1] for g in grp]
        if len(S) and float(S.max()) >= self.th["T_merge"]:
            o = np.argsort(-S)
            s2 = float(S[o[1]]) if len(S) > 1 else -1.0
            if float(S[o[0]]) - s2 < self.margin:
                return
            cid = ids[o[0]]
            c = self.clusters[cid]
            c["mem"] = (c["mem"] + mem)[-self.max_members:]
            c["added"] += len(mem)
            self._cent(cid)
        else:
            cid = "spk_" + uuid.uuid4().hex[:8]
            self.clusters[cid] = {"mem": mem, "c": cen, "added": len(mem), "created": ts, "last": ts}
        gs = set(grp)
        self.formed_from = [self.pool[g][2] for g in grp]
        self.pool = [p for i, p in enumerate(self.pool) if i not in gs]
        return cid

    def _maintain(self, cid):
        ids = [i for i in self.clusters if i != cid]
        if ids:
            S = np.stack([self.clusters[i]["c"] for i in ids]) @ self.clusters[cid]["c"]
            j = int(np.argmax(S))
            other = ids[j]
            if S[j] >= self.th["T_merge"] and (self.can_merge is None or self.can_merge(cid, other)):
                keep, gone = (cid, other) if self.clusters[cid]["created"] <= self.clusters[other]["created"] else (other, cid)
                k, g = self.clusters[keep], self.clusters.pop(gone)
                k["mem"] = (k["mem"] + g["mem"])[-self.max_members:]
                k["added"] += g["added"]
                self._cent(keep)
                if self.on_merge:
                    self.on_merge(keep, gone)
                cid = keep
        c = self.clusters[cid]
        if c["added"] % self.split_every == 0 and len(c["mem"]) >= 2 * self.k + 2:
            m = np.stack(c["mem"])
            cs = m[[0, int(np.argmin(m @ m[0]))]]
            lab = None
            for _ in range(8):
                lab = np.argmax(m @ cs.T, 1)
                if lab.min() == lab.max():
                    return cid
                cs = _n(np.stack([m[lab == i].mean(0) for i in (0, 1)]))
            if float(cs[0] @ cs[1]) < self.th["T_split"] and min((lab == 0).sum(), (lab == 1).sum()) >= self.k:
                c["mem"] = [v for v, l in zip(c["mem"], lab) if l == 0]
                self._cent(cid)
                nid = "spk_" + uuid.uuid4().hex[:8]
                self.clusters[nid] = {"mem": [v for v, l in zip(m, lab) if l == 1], "c": cs[1],
                                      "added": int((lab == 1).sum()), "created": c["created"], "last": c["last"]}
        return cid

    # ------------------------------------------------------------------
    def forget(self, speaker_id: str):
        self.clusters.pop(speaker_id, None)

    def save(self, path):
        d = {"clusters": {k: {"mem": np.stack(v["mem"]).tolist(), "added": v["added"], "created": v["created"], "last": v["last"]}
                          for k, v in self.clusters.items()},
             "pool": [[t, v.tolist(), u] for t, v, u in self.pool]}
        Path(path).write_text(json.dumps(d))

    def load(self, path):
        d = json.loads(Path(path).read_text())
        self.clusters = {}
        for k, v in d["clusters"].items():
            self.clusters[k] = {"mem": [np.asarray(x, np.float32) for x in v["mem"]], "added": v["added"],
                                "created": v["created"], "last": v["last"]}
            self._cent(k)
        self.pool = [(t, np.asarray(v, np.float32), u) for t, v, u in d["pool"]]
