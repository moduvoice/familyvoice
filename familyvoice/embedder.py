"""Speaker embedder without torch: numpy Kaldi fbank + onnxruntime (WeSpeaker ONNX models).

load_embedder("r34", profile="server"|"device") -> callable(pcm_float32_16k) -> (embedding, voiced_seconds)
Models (downloaded from Hugging Face on first use; ResNet34/ResNet293 CC BY 4.0, CAM++ Apache-2.0 per model card - attribute WeSpeaker/VoxCeleb):
  campp : (option)  Wespeaker/wespeaker-voxceleb-campplus-LM  voxceleb_CAM++_LM.onnx   (7.2M params, 29 MB)
  r34   : (default) Wespeaker/wespeaker-voxceleb-resnet34-LM  voxceleb_resnet34_LM.onnx  (6.6M params, 26.5 MB)
  r293  : Wespeaker/wespeaker-voxceleb-resnet293-LM voxceleb_resnet293_LM.onnx (28.6M params, 114 MB) - accuracy reference, too big for a device default
"""
from __future__ import annotations

import numpy as np

SR = 16000
REPOS = {"campp": ("Wespeaker/wespeaker-voxceleb-campplus-LM", "voxceleb_CAM++_LM.onnx"),
         "r34": ("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx"),
         "r293": ("Wespeaker/wespeaker-voxceleb-resnet293-LM", "voxceleb_resnet293_LM.onnx")}


def _mel_banks(n_mels=80, n_fft=512, sr=SR, lo=20.0, hi=0.0):
    hi = sr / 2 + hi if hi <= 0 else hi
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    mlo, mhi = mel(lo), mel(hi)
    d = (mhi - mlo) / (n_mels + 1)
    fft_mel = mel(np.arange(n_fft // 2) * sr / n_fft)
    banks = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(n_mels):
        l, c, r = mlo + m * d, mlo + (m + 1) * d, mlo + (m + 2) * d
        up = (fft_mel - l) / (c - l)
        down = (r - fft_mel) / (r - c)
        banks[m, : n_fft // 2] = np.maximum(0, np.minimum(up, down))
    return banks


_BANKS = _mel_banks()


def fbank(wav: np.ndarray) -> np.ndarray:
    """Kaldi-compatible 80-dim log mel fbank (25 ms hamming, 10 ms shift, dither 0, preemph 0.97,
    snip_edges) on int16-scaled audio, then per-utterance mean normalisation (WeSpeaker recipe)."""
    x = np.asarray(wav, np.float64) * 32768.0
    fl, fs, nfft = 400, 160, 512
    n = 1 + (len(x) - fl) // fs
    if n < 1:
        return np.zeros((0, 80), np.float32)
    idx = np.arange(fl)[None] + fs * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    fr = np.concatenate([fr[:, :1] - 0.97 * fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], 1)
    fr = fr * np.hamming(fl)[None]   # numpy hamming == symmetric 0.54-0.46cos(2pi n/(N-1)) == kaldi hamming
    spec = np.abs(np.fft.rfft(fr, nfft)) ** 2
    f = np.log(np.maximum(spec @ _BANKS.T.astype(np.float64), np.finfo(np.float32).eps))
    f = f - f.mean(0, keepdims=True)
    return f.astype(np.float32)


def voiced_seconds(wav: np.ndarray, rel: float = 0.03) -> float:
    """Crude energy VAD: seconds of 20 ms frames above rel * max RMS (use a real VAD in the app)."""
    n = 320
    m = len(wav) // n
    if m == 0:
        return 0.0
    e = np.sqrt((np.asarray(wav[: m * n]).reshape(m, n) ** 2).mean(1))
    return float((e > rel * e.max()).sum() * n / SR)


class OnnxEmbedder:
    def __init__(self, path, threads: int = 2, providers=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, wav):
        wav = np.asarray(wav, np.float32)
        f = fbank(wav)[None]
        e = self.s.run(None, {self.inp: f})[-1][0]
        return e, voiced_seconds(wav)


DEFAULT_MODEL = "r34"


def model_path(model: str = DEFAULT_MODEL, profile: str = "server") -> str:
    """Download the fp32 ONNX on first use (Hugging Face cache). profile="device" builds an int8 copy next to it
    once (ONNX Runtime dynamic quantization of Conv/MatMul weights, needs the `onnx` package) and returns that."""
    import os
    from huggingface_hub import hf_hub_download
    repo, fn = REPOS[model]
    p = hf_hub_download(repo, fn)
    if profile != "device":
        return p
    q = os.path.join(os.environ.get("FV_CACHE", os.path.dirname(p)), fn.replace(".onnx", ".int8.onnx"))
    if not os.path.exists(q):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(p, q, weight_type=QuantType.QInt8)
    return q


def load_embedder(model: str = DEFAULT_MODEL, profile: str = "server", **kw):
    """model: r34 (default, ResNet34-LM) | campp (CAM++-LM) | r293 (reference). profile: server (fp32) | device (int8)."""
    return OnnxEmbedder(model_path(model, profile), **kw)
