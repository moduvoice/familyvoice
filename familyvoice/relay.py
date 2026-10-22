"""24-hour message box keyed by recipient NAME (not by voice cluster).

Delivery happens only when the tracker CONFIRMS a speaker AND that speaker's cluster is bound to the
recipient name. Otherwise the device stays silent and waits for the next utterance - it never asks
"are you Sunhee?". Replies from the recipient become a message back to the sender, and the sender
can ask "Did Sunhee eat?" to get the status.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict, field

TTL_S = 24 * 3600


@dataclass
class Message:
    id: str
    to_name: str
    text: str
    from_speaker: str | None      # cluster id of the sender (None if the sender was not confirmed)
    from_name: str | None         # resolved at delivery time if still unknown
    created: float
    delivered: float | None = None
    delivered_to: str | None = None
    reply_id: str | None = None
    meta: dict = field(default_factory=dict)


class Mailbox:
    def __init__(self, ttl_s: float = TTL_S):
        self.ttl = ttl_s
        self.msgs: dict[str, Message] = {}

    def post(self, to_name: str, text: str, from_speaker: str | None, from_name: str | None = None,
             ts: float | None = None, **meta) -> Message:
        ts = time.time() if ts is None else ts
        m = Message(uuid.uuid4().hex[:12], to_name.strip().lower(), text, from_speaker, from_name, ts, meta=meta)
        self.msgs[m.id] = m
        return m

    def expire(self, now: float | None = None):
        now = time.time() if now is None else now
        for k in [k for k, m in self.msgs.items() if now - m.created > self.ttl]:
            del self.msgs[k]

    def due(self, name: str | None, now: float | None = None) -> list[Message]:
        """Undelivered messages for a confirmed, named speaker (oldest first)."""
        self.expire(now)
        if not name:
            return []
        return sorted((m for m in self.msgs.values() if m.delivered is None and m.to_name == name),
                      key=lambda m: m.created)

    def mark_delivered(self, msg_id: str, speaker_id: str, now: float | None = None):
        m = self.msgs[msg_id]
        m.delivered = time.time() if now is None else now
        m.delivered_to = speaker_id

    def status_for(self, from_speaker: str, to_name: str) -> list[Message]:
        """'Did Sunhee eat?' - messages this sender left for that name (newest first)."""
        self.expire()
        return sorted((m for m in self.msgs.values() if m.from_speaker == from_speaker and m.to_name == to_name.lower()),
                      key=lambda m: -m.created)

    def to_dict(self):
        return {"ttl": self.ttl, "msgs": [asdict(m) for m in self.msgs.values()]}

    @classmethod
    def from_dict(cls, d):
        b = cls(d.get("ttl", TTL_S))
        for x in d.get("msgs", []):
            b.msgs[x["id"]] = Message(**x)
        return b
