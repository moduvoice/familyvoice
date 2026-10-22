"""familyvoice demo: three voices, no enrollment, no questions.

    pip install -e .[dev]
    python examples/relay_demo.py dad1.wav dad2.wav ... --who dad,dad,sunhee,...

Every file is one utterance (16 kHz mono). The script plays a tiny family day:
  - everybody just talks (clusters form by themselves; nobody is asked anything)
  - someone calls "Sunhee!" and Sunhee answers -> the name attaches to her voice cluster
  - Dad says "Tell Sunhee to eat her lunch" -> stored for 24 h
  - the next time Sunhee's voice is CONFIRMED, the device says the message; otherwise it stays silent
"""
import sys
import soundfile as sf
from familyvoice import Household


def main(files, who):
    hh = Household.create()                   # ResNet34-LM fp32; Household.create(model="campp") or profile="device" (int8)
    ts = 0.0
    for f, w in zip(files, who):
        pcm, sr = sf.read(f, dtype="float32")
        assert sr == 16000
        ts += 5
        cues = []
        if w.startswith("call:"):                # "call:sunhee" = this utterance called Sunhee
            cues = [{"kind": "call", "name": w[5:]}]
        if w.startswith("send:"):                # "send:sunhee:eat your lunch"
            _, to, text = w.split(":", 2)
            cues = [{"kind": "send", "to": to, "text": text}]
        r = hh.hear(pcm, ts=ts, cues=cues)
        line = f"{f}: {r['state']:<9} speaker={r['speaker_id']} name={r['name']}"
        for m in r["deliver"]:
            line += f"\n   >> SAY: {m.from_name or 'Someone'} asked me to tell you: {m.text}"
        print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    i = args.index("--who")
    main(args[:i], args[i + 1].split(","))
