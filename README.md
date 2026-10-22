# familyvoice

**Household speaker recognition with no enrollment and no questions.**
A shared family device hears voices, clusters them by itself, learns names from ordinary
conversation, and passes messages to the right person the next time *that voice* speaks -
"Dad asked me to tell you to eat your dinner." If it is not sure who is talking, it says nothing
and waits for the next utterance. It never asks "who are you?".

```
voice ─▶ ResNet34 embedding ─▶ VoiceTracker ──confirmed──▶ NameBinder ──name──▶ Mailbox ──▶ "Dad says …"
                                   │  tentative / unknown → silence
                                   └─ pool: ≥3 mutually-consistent unknown utterances → new person
```

## Why this exists (what AWS / Alexa do not offer, checked 2026-10-22)
| Service | Speaker identity across sessions without enrollment? |
|---|---|
| Amazon Connect Voice ID | No - enrollment-based caller authentication; **end of support 2026-05-20** ([doc](https://docs.aws.amazon.com/connect/latest/adminguide/amazonconnect-voiceid-end-of-support.html)) |
| Amazon Transcribe diarization | No - labels `spk_0..spk_29` inside one recording/stream only ([doc](https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html)) |
| Amazon Chime SDK speaker search | No - phone-call pipelines, consent, 10 s + 10 s of speech, then the app calls `CreateVoiceProfile` ([doc](https://docs.aws.amazon.com/chime-sdk/latest/dg/va-data-flow.html)) |
| Alexa voice ID / skill personalization | No - user enrolls ("Alexa, learn my voice"); unrecognized speakers get no `person` object ([doc](https://developer.amazon.com/en-US/docs/alexa/custom-skills/add-personalization-to-your-skill.html)) |
| Amazon Nova 2 Sonic | No speaker identification feature listed ([doc](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html)) |

## Install / use
```bash
pip install "familyvoice[device] @ git+https://github.com/moduvoice/familyvoice"
# or from a checkout:
pip install -e .              # numpy + onnxruntime + huggingface_hub (no torch)
pip install -e ".[device]"    # + onnx, to build the int8 model for a device CPU
```
```python
from familyvoice import Household
hh = Household.create()                       # ResNet34-LM, fp32 ONNX
# Household.create(model="campp")             # CAM++-LM, faster, a little less accurate
# Household.create(profile="device")          # int8 ONNX built once on first use
r = hh.hear(pcm_16k_float32, ts=time.time(),
            cues=[{"kind": "send", "to": "sunhee", "text": "eat your dinner"}])
# r = {"speaker_id": "spk_1a2b3c4d" | None, "state": "confirmed|tentative|unknown",
#      "score": 0.71, "margin": 0.33, "name": "dad" | None, "utt_id": "...", "deliver": [Message, ...]}
for m in r["deliver"]:
    say(f"{m.from_name or 'Someone'} asked me to tell you: {m.text}")
```
Cues come from your ASR/LLM (the module does not do ASR):
`{"kind":"call","name":"sunhee"}` ("Sunhee!") · `{"kind":"self_intro","name":"sunhee"}` ("It's Sunhee") ·
`{"kind":"addressed","name":"sunhee"}` ("Thanks, Sunhee") · `{"kind":"send","to":"sunhee","text":"…"}` ("Tell Sunhee …").
`familyvoice.app_adapter.FamilyVoiceSpeaker(household, store)` wraps all of this for an app that passes raw
audio bytes (16 kHz WAV, or webm/opus through ffmpeg) and the STT text (`note_transcript`) - see below.

### Settings
| What | Python | `FamilyVoiceSpeaker` env | Values |
|---|---|---|---|
| model | `Household.create(model=)` | `FV_MODEL` | `r34` (default, ResNet34-LM) · `campp` (CAM++-LM) · `r293` (accuracy reference, too big for a device) |
| profile | `profile=` | `FV_PROFILE` | `server` (default, fp32 ONNX) · `device` (int8 ONNX, built once on first use, needs `onnx`) |
| strictness | `level=` | `FV_LEVEL` | `strict` (default, 0.1 % false accept per comparison) · `strictest` · `loose` (not recommended) |
| threads | `load_embedder(threads=)` | `FV_THREADS` | default 2 |

### Models (downloaded on first use - not stored in this repository)
| key | file | source | params / size | license |
|---|---|---|---|---|
| `r34` | `voxceleb_resnet34_LM.onnx` | [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) | 6.6 M / 26.5 MB fp32 | CC BY 4.0 (model card) |
| `campp` | `voxceleb_CAM++_LM.onnx` | [Wespeaker/wespeaker-voxceleb-campplus-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM) | 7.2 M / 29.3 MB | Apache-2.0 (model card), trained on VoxCeleb (CC BY 4.0) |
| `r293` | `voxceleb_resnet293_LM.onnx` | [Wespeaker/wespeaker-voxceleb-resnet293-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) | 28.6 M / 114 MB | CC BY 4.0 |

Attribution: WeSpeaker (https://github.com/wenet-e2e/wespeaker), trained on VoxCeleb (Nagrani et al., Chung et al.).
The thresholds in `familyvoice/thresholds/` were calibrated by us on LibriSpeech dev (CC BY 4.0) and speechocean762 train (CC BY 4.0) speakers.

## Plugging into a voice app
- Call `observe()` / `hear()` once per user turn, **in parallel with ASR** (it needs only the audio); pass the transcript afterwards.
- Prepend deliveries to whatever the assistant answers: *"Before that - Dad asked me to tell you to eat your dinner."*
- `state != "confirmed"` or no `name` -> say nothing about identity. Never ask.
- Deployment: run it on the device or in the process that already receives the microphone audio; state is a few
  people x 64 vectors (< 1 MB). Alexa skills / Alexa+ MCP receive text, not raw audio - the module sits in front
  of your own microphone path.

## Rules (thresholds calibrated per model, never hand-tuned per family)
1. **Confirm** an utterance only if its best cluster score ≥ `T_acc[duration bin]` *and* beats the runner-up by ≥ 0.05.
   `T_acc` = 99.9th percentile of same-gender impostor scores, computed **separately for adults and children and
   the larger value kept** (a mixed calibration set let siblings absorb each other).
2. Only confirmed utterances update a cluster (newest 64 kept, so a slowly changing voice is followed).
3. Everything else is **silent** and waits in a pool; ≥ 3 pool utterances that all agree pairwise (≥ `T_pair`) become
   a new person, and the utterance that completes the group is itself confirmed. One-off voices (guests, TV) never
   form a cluster; utterances with < 1 s of voice are dropped.
4. Clusters merge when their centres meet (≥ `T_merge`, never across two different bound names); a cluster that has
   become two clearly different halves (< `T_split`) is split.
5. **Names** bind only when a name and a cluster are each other's clear first choice: ≥ 4.5 evidence points
   (answer to a call 1.5, self-introduction 3.0, addressed 1.0) from ≥ 2 conversations, 2 points ahead of every rival
   on *both* sides. "Tell Sunhee…" / calling "Sunhee!" vetoes Sunhee for that speaker. Cues heard before a voice has
   a cluster wait on the utterance and follow it into the cluster. Bindings are recomputed every time (never sticky).
6. **Delivery** only when the recipient's named cluster is confirmed; messages expire after 24 h.

## Measured (2026-10-22, public data; full report [`report/index.html`](report/index.html) with the raw result files in `report/data/`)
Calibration on LibriSpeech dev + speechocean762 train speakers; tests on disjoint LibriSpeech test + speechocean762
test speakers; 1-4 s utterances; "15 % noisy" = 15 % of utterances with reverb (RT60 0.2-0.6 s) + noise (SNR 5-20 dB).

| | ResNet34-LM (default) | CAM++-LM |
|---|---|---|
| EER, adults, 1 s / 2 s / 3 s | 8.2 / 2.4 / 1.4 % | 9.7 / 2.7 / 1.7 % |
| Simulated families, adults, 15 % noisy (1,000 families x 2 types, names assumed known): misdelivered | **0 / 24,516** | **0 / 24,516** |
| … delivered / on the recipient's first utterance / p90 delay | 99.0 % / 64 % / 5 utts | 98.4 % / 57 % / 6 utts |
| Same, 50 % noisy: misdelivered / delivered | 0 / 24,516 · 96.9 % | 0 / 24,516 · 94.3 % |
| End-to-end package, 3 adults, 50-utterance day, names learned from cues, 300 runs, 15 % noisy: right / wrong / not yet | 157 / **0** / 143 | 134 / **0** / 166 |
| CPU latency, 1 thread x86, 1 s / 3 s utterance | 22 / 64 ms | 11 / 29 ms |
| CPU latency, ARM Graviton3 (c7gd), 3 s utterance: fp32 1 / 2 threads · int8 (`profile="device"`) 1 / 2 threads | 313 / 175 ms · 103 / 63 ms | - |
| int8 model size / cosine to fp32 embedding (18 demo clips, min / median) | 6.7 MB / 0.975 / 0.983 (thresholds are fp32-calibrated; int8 accuracy not measured) | - |

- **Not yet delivered = name not learned yet.** Every undelivered message above was a confirmed recipient whose
  voice had no name yet. With names known at once the same audio delivers 291-300 of 300 with 0 misdeliveries,
  median 3-5 of the recipient's utterances. Delivery speed is set by how often the family uses names.
- **Children / siblings are the limit.** Children's 1 s EER is 16.6 % (ResNet34). Simulated sibling families with names
  assumed known: 10 misdelivered of 3,805 messages (15 % noisy). With names learned from conversation (the real path)
  misdeliveries were 0 in 300 sibling runs, because a cluster that mixes two siblings collects two names and never
  binds - but then most sibling messages wait (61 / 300 delivered in a 30-utterance day).
- Synthetic room noise and read speech, not real far-field household audio. Not authentication.

## Licenses
Code: Apache-2.0. Models: downloaded at runtime, licenses per model card (ResNet34-LM CC BY 4.0, CAM++-LM Apache-2.0). Calibration data: LibriSpeech (CC BY 4.0),
speechocean762 (CC BY 4.0) - not redistributed.
